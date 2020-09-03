#include "spn_diff.h"
#include "common.h"
#include "spn_diff_kernel.cuh"
#include <cuda_profiler_api.h>

#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include <chrono>
#include <thread>

unsigned long long last_round_trails[50]={0}; //For tracing number of trails explored, verify correctness

struct GPU_manager_t{
	//Cluster Output (Partial)
	//For last rounds
	long long cluster_size=0;
	double cluster_prob=0.0;

	//CPU
	// int cluster_size_cpu_only=0;
	// double cluster_prob_cpu_only=0.0;

	//MITM
	const int mitm_cache_array_length = 134217728; //3 Sbox * 9 (5pos + 4 val) bit = 27 bit = this value
	int* mitm_cluster_size_cache;
	float* mitm_prob_cache;

	GPU_manager_t(){
		//MITM 
		mitm_cluster_size_cache = new int[mitm_cache_array_length];
		mitm_prob_cache = new float[mitm_cache_array_length];

		for (int i=0;i< mitm_cache_array_length;i++){
			mitm_cluster_size_cache[i] = 0;
			mitm_prob_cache[i] = 0;
		}

		// gpu_kernel = new GPU_Kenerl_t();
		// Explicitly called for init
	}

	~GPU_manager_t(){
		delete[] mitm_cluster_size_cache;
		delete[] mitm_prob_cache;
	}

	void init(int gpu_id){
		 gpu_kernel = new GPU_Kenerl_t(gpu_id);
	}

	void compute_diff_cluster(double& cluster_prob, long long& cluster_size, unsigned char* dx, unsigned char* dy, int num_round){
		//Calculating Sbox Number and its position
		int sbox_index[16];
		for (int i=0;i<16;i++){
			sbox_index[i] = 16;
		}
		int sbox_index_ptr =0;
		for (int i=0;i<16;i++){
			if (dx[i] > 0){
				sbox_index[sbox_index_ptr] = i;
				sbox_index_ptr++;
			}
		}
		int nb_branch = 1;
		for (int i=0;i<sbox_index_ptr;i++){
			nb_branch *= SPN_DIFF::diff_table_size_host[dx[sbox_index[i]]];
		}
		float prob = 1.0;

		this->gpu_kernel->kernel_compute(1,dx, dy, sbox_index, &sbox_index_ptr,&nb_branch, &prob, 0, num_round);
	}

	void reduction(double& result_prob, long long& result_size){
		long long gpu_size=0;
		double gpu_prob=0;

		this->gpu_kernel->kernel_reduction(gpu_prob, gpu_size);

		//Combined CPU (if any) with gpu
		this->cluster_size += gpu_size;
		this->cluster_prob += gpu_prob;

		result_prob = this->cluster_prob;
		result_size = this->cluster_size;
	}

	void reset(){
		for (int i=0;i< mitm_cache_array_length;i++){
			mitm_cluster_size_cache[i] = 0;
			mitm_prob_cache[i] = 0;
		}

		this->cluster_size = 0;
		this->cluster_prob = 0;
	}

	GPU_Kenerl_t* gpu_kernel;
	
};
const int gpu_num =1;
GPU_manager_t trifle_gpu_manager_arr[gpu_num];

void spn_diff_init(){
	SPN_DIFF::init(); //Init Common GPU first

    for (int i=0;i<gpu_num;i++){
		trifle_gpu_manager_arr[i].init(i); //Init all other GPU
	}

};

//FULL GPU 
void spn_diffcluster_gpu(double& cluster_prob, long long& cluster_size, unsigned char* dx, unsigned char* dy){
	//UNUSED
}

void testing_gpu_kernel(double& cluster_prob, long long& cluster_size, std::chrono::steady_clock::time_point &end_b4_reduc){
	//NOTE: change dx, dy here

	unsigned char dx_16[MAX_SBOX] = {
        0x0, 0x0, 0x0, 0xf,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0xf
	};

	unsigned char dy_16[MAX_SBOX] = {
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x5, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x5, 0x0, 0x0
	};

	std::cout <<"\nDX : ";
	for (int i=0;i<16;i++){
		if (i%4==0){
			std::cout << " ";
		}
		std::cout << std::hex << (int) dx_16[i];
	}
	std::cout << std::dec;
	std::cout <<"\nDY : ";
	for (int i=0;i<16;i++){
		if (i%4==0){
			std::cout << " ";
		}
		std::cout << std::hex << (int) dy_16[i];
	}
	std::cout << std::dec;

	//Do Permutate
	{
		unsigned long long front_64 = 0;
		#pragma unroll
		for (int i = 0; i < 16; i++) {
			if ( dy_16[i] > 0) {
				front_64 |= SPN_DIFF::perm_lookup_host_reversed[i][dy_16[i]];
			}
		}
		#pragma unroll
		for (int i = 0; i < 16; i++) {
			dy_16[i] = (front_64 >> ((15 - i) * 4)) & 0xf;  
		}
	}

	int number_rounds = MAX_ROUND_FORWARD;

	std::cout<< "\nExecuting GPU";
	std::cout << log2(CLUSTER_PROB_BOUND_FORWARD);
	std::cout.flush();
	trifle_gpu_manager_arr[0].compute_diff_cluster(cluster_prob, cluster_size, dx_16, dy_16, number_rounds);
	end_b4_reduc = std::chrono::steady_clock::now();
	std::cout<< "\nExecuting Reduction";
	std::cout.flush();
	trifle_gpu_manager_arr[0].reduction(cluster_prob, cluster_size);

	//If other thread, also need to call cudasetdevice
};