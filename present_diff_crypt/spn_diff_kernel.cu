#include "spn_diff_kernel.cuh"
#include <iostream>
#include <cstring>

#include <cooperative_groups.h>
using namespace cooperative_groups; 

namespace SPN_DIFF{
    /*
	* BC specific permutation and DTT
    */

    //Contains configuration (macro / c++ global variable) intended to be used across different translation unit
    // __shared__ unsigned long long perm_lookup_shared[MAX_SBOX][16][2];
    __device__ unsigned long long perm_lookup_global_forward[MAX_SBOX][16];
    __device__ unsigned long long perm_lookup_global_reversed[MAX_SBOX][16];
    // __device__ unsigned long long perm_lookup_device[MAX_SBOX][16][2]; 
    
    unsigned char perm_host[BLOCK_SIZE_BIT];
    unsigned char perm_host_reversed[BLOCK_SIZE_BIT];

    unsigned long long perm_lookup_host[MAX_SBOX][16]; //8192 bytes, 8KB, one SM can have 49KB should be fine
    unsigned long long perm_lookup_host_reversed[MAX_SBOX][16];

    __shared__ unsigned int diff_table_shared[16][8];  //NOTE: init in kernel by 1st thread of the block.
    __device__ unsigned int diff_table_global[][8] = {
		{0x0	,0x0	,0x0	,0x0	,0x0	,0x0	,0x0	,0x0},
		{0x3	,0x7	,0x9	,0xd	,0x0	,0x0	,0x0	,0x0},
		{0x5	,0x3	,0x6	,0xa	,0xc	,0xd	,0xe	,0x0},
		{0x6	,0x1	,0x3	,0x4	,0x7	,0xa	,0xb	,0x0},
		{0x5	,0x6	,0x7	,0x9	,0xa	,0xc	,0xe	,0x0},
		{0xc	,0x1	,0x4	,0x9	,0xa	,0xb	,0xd	,0x0},
		{0xb	,0xf	,0x2	,0x6	,0x8	,0xc	,0x0	,0x0},
		{0x1	,0xf	,0x2	,0x6	,0x8	,0xc	,0x0	,0x0},
		{0xb	,0xf	,0x3	,0x7	,0x9	,0xd	,0x0	,0x0},
		{0x4	,0xe	,0x2	,0x6	,0x8	,0xc	,0x0	,0x0},
		{0x5	,0x2	,0x3	,0x8	,0xa	,0xd	,0xe	,0x0},
		{0x8	,0x1	,0x4	,0x9	,0xa	,0xb	,0xd	,0x0},
		{0x5	,0x2	,0x7	,0x8	,0x9	,0xa	,0xe	,0x0},
		{0x2	,0x1	,0x3	,0x4	,0x7	,0xa	,0xb	,0x0},
		{0x2	,0x3	,0x6	,0x7	,0x8	,0x9	,0xc	,0xd},
		{0x1	,0x4	,0xe	,0xf	,0x0	,0x0	,0x0	,0x0},
	};
    __device__ unsigned int diff_table_global_reversed[][8] = {
		{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
		{0x7, 0xf, 0x3, 0x5, 0xb, 0xd, 0x0, 0x0},
		{0xd, 0xe, 0xa, 0xc, 0x6, 0x7, 0x9, 0x0},
		{0x1, 0x2, 0xe, 0x3, 0x8, 0xa, 0xd, 0x0},
		{0x9, 0xf, 0x5, 0xb, 0x3, 0xd, 0x0, 0x0},
		{0x2, 0x4, 0xa, 0xc, 0x0, 0x0, 0x0, 0x0},
		{0x3, 0x4, 0x2, 0xe, 0x6, 0x7, 0x9, 0x0},
		{0x1, 0x4, 0xc, 0x8, 0xe, 0x3, 0xd, 0x0},
		{0xb, 0xa, 0xc, 0x6, 0x7, 0x9, 0xe, 0x0},
		{0x1, 0x4, 0x5, 0xb, 0x8, 0xc, 0xe, 0x0},
		{0x2, 0x4, 0x5, 0xa, 0xb, 0x3, 0xc, 0xd},
		{0x6, 0x8, 0x5, 0xb, 0x3, 0xd, 0x0, 0x0},
		{0x5, 0x2, 0x4, 0x6, 0x7, 0x9, 0xe, 0x0},
		{0x1, 0x2, 0x8, 0xa, 0x5, 0xb, 0xe, 0x0},
		{0x9, 0xf, 0x2, 0x4, 0xa, 0xc, 0x0, 0x0},
		{0x6, 0x7, 0x8, 0xf, 0x0, 0x0, 0x0, 0x0}
    };

    __shared__ float prob_table_shared[16][8];  //NOTE: init in kernel by 1st thread of the block.
    __device__ float prob_table_global[16][8]={
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0.25f , 0.25f , 0.25f , 0.25f , 0, 0, 0, 0},
		{0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0},
		{0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0},
		{0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0},
		{0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0},
		{0.25f , 0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0, 0},
		{0.25f , 0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0, 0},
		{0.25f , 0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0, 0},
		{0.25f , 0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0, 0},
		{0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0},
		{0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0},
		{0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0},
		{0.25f , 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0},
		{0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f},
		{0.25f , 0.25f , 0.25f , 0.25f , 0, 0, 0, 0}
	};
    __device__ float prob_table_global_reversed[16][8]={
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0.25, 0.25, 0.125, 0.125, 0.125, 0.125, 0, 0}, 
		{0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0}, 
		{0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0}, 
		{0.25, 0.25, 0.125, 0.125, 0.125, 0.125, 0, 0}, 
		{0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0}, 
		{0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0}, 
		{0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0}, 
		{0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0}, 
		{0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0}, 
		{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125}, 
		{0.25, 0.25, 0.125, 0.125, 0.125, 0.125, 0, 0}, 
		{0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0}, 
		{0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0}, 
		{0.25, 0.25, 0.125, 0.125, 0.125, 0.125, 0, 0}, 
		{0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0} 
	};
    
    __shared__ unsigned int diff_table_size_shared[16];
    __device__ unsigned int diff_table_size_global[16] = {1 , 4 , 7 , 7 , 7 , 7 , 6 , 6 , 6 , 6 , 7 , 7 , 7 , 7 , 8 , 4};  
    __device__ unsigned int diff_table_size_global_reversed[16] = {1, 6, 7, 7, 6, 4, 7, 7, 7, 7, 8, 6, 7, 7, 6, 4};  
    unsigned int diff_table_size_host[16]= {1 , 4 , 7 , 7 , 7 , 7 , 6 , 6 , 6 , 6 , 7 , 7 , 7 , 7 , 8 , 4 };  
    unsigned int diff_table_size_host_reversed[16]  = {1, 6, 7, 7, 6, 4, 7, 7, 7, 7, 8, 6, 7, 7, 6, 4};  

    __shared__ unsigned long long branch_size_block_shared[1];
    __shared__ float prob_per_as_shared[32]; //MAX_AS
    __shared__ float prob_per_round_remaining_shared[32]; //MAX_ROUND_FORWARD

    /*
    * DX and DY changes
    */
	//Constant memory because it is accessed by the same warp @ the same addresses. (broadcasting) else request will be serialized
    __constant__ unsigned char final_dy_constant[16] = {
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x6, 0x0, 0x6, 0x0
	};
    unsigned char final_dy_host[16] = {
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x6, 0x0, 0x6, 0x0
	};

    unsigned char ref_dx_host[16] = {
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0xb,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0
    };

    __constant__ float CLUSTER_PROB_BOUND_const = 0;     
};

GPU_Kenerl_t::GPU_Kenerl_t(int gpu_id, bool is_MITM_used){
    //Create its own stream..
    cudaStreamCreate( &(this->stream_obj) );

    //DEBUG: Set prinf limit 10MB
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10000000);

    //Called @ different GPU threads (each with its own cpu thread)
    auto cudaStatus = cudaSetDevice(gpu_id);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! CudaDeviceNumber :%d", gpu_id );
		goto Error;
    }

    std::cout << "\nTransfered perm_LUhost from host to device";
    cudaStatus = cudaMemcpyToSymbol(SPN_DIFF::perm_lookup_global_forward, SPN_DIFF::perm_lookup_host, sizeof(unsigned long long)*16*16);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy perm_lookup_global_forward failed!");
		goto Error;
    }
    
    std::cout << "\nTransfered perm_LUhost Reversed from host to device";
    cudaStatus = cudaMemcpyToSymbol(SPN_DIFF::perm_lookup_global_reversed, SPN_DIFF::perm_lookup_host_reversed, sizeof(unsigned long long)*16*16);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy perm_lookup_global_reversed failed!");
		goto Error;
    }
    
    //Allocate Memory HERE
    //Input Allocation
    cudaStatus = cudaMalloc((void**)& device_dx, sizeof(unsigned char)* 16 * MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_dx @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_dx, 0, sizeof(unsigned char)* 16 * MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_dx failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_sbox_index, sizeof(int)* MAX_AS * MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_sbox_index @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_sbox_index, 0, sizeof(int) * MAX_AS * MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_sbox_index failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_sbox_num, sizeof(int) * MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_sbox_num @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_sbox_num, 0, sizeof(int) * MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_sbox_num failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_prob, sizeof(float) * MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_prob @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_prob, 0, sizeof(float) * MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_prob failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_branch_size, sizeof(int) * ( (MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD) + 4) ); // + 4 to accomodate 4 loading @ same time
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_branch_size @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_branch_size, 0, sizeof(int) * ( (MAX_PATH_PER_ROUND * MAX_ROUND_FORWARD) + 4) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_branch_size failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_branch_size_thread, sizeof(unsigned long long) * ( (GRID_THREAD_SIZE * MAX_ROUND_FORWARD) + 4) );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_branch_size_thread @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_branch_size_thread, 0, sizeof(unsigned long long) * ( (GRID_THREAD_SIZE * MAX_ROUND_FORWARD) + 4) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_branch_size_thread failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_branch_size_block, sizeof(unsigned long long) * ( (BLOCK_NUM * MAX_ROUND_FORWARD) + 4 + 2) );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_branch_size_block @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_branch_size_block, 0, sizeof(unsigned long long) * ( (BLOCK_NUM * MAX_ROUND_FORWARD) + 4 + 2) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_branch_size_block failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_total_branch_size_block, sizeof(unsigned long long) * ( ((BLOCK_NUM * MAX_ROUND_FORWARD) + 1) * 2) );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_branch_size_block @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_total_branch_size_block, 0, sizeof(unsigned long long) * ( ((BLOCK_NUM * MAX_ROUND_FORWARD) + 1) * 2) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_branch_size_block failed!");
        goto Error;
    }

    //Final (Needs to be reduced) Output Allocation
    cudaStatus = cudaMalloc((void**)& device_cluster_size_final, sizeof(unsigned long long)* THREAD_PER_BLOCK * BLOCK_NUM);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc  device_cluster_size_final @init failed!");
        goto Error;
    }
    cudaStatus = cudaMemset(device_cluster_size_final, 0, sizeof(unsigned long long)* THREAD_PER_BLOCK * BLOCK_NUM);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_cluster_size_final failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_prob_final, sizeof(double)*  THREAD_PER_BLOCK * BLOCK_NUM);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc device_prob_final @init failed!");
        goto Error;
    }
    cudaStatus = cudaMemset(device_prob_final, 0, sizeof(double)*  THREAD_PER_BLOCK * BLOCK_NUM);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_prob_final failed!");
        goto Error;
    }

    //MITM Allocation
    if (is_MITM_used){
        cudaStatus = cudaMalloc((void**)& MITM_prob_interm_global, sizeof(float)*  GPU_Kenerl_t::MITM_size);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc  MITM_prob_interm_global @init failed!");
            goto Error;
        }
        cudaStatus = cudaMemset(MITM_prob_interm_global, 0, sizeof(float)*  GPU_Kenerl_t::MITM_size );
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemset MITM_prob_interm_global failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)& MITM_size_interm_global, sizeof(unsigned long long)*  GPU_Kenerl_t::MITM_size);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc  MITM_prob_interm_global @init failed!");
            goto Error;
        }
        cudaStatus = cudaMemset(MITM_size_interm_global, 0, sizeof(unsigned long long)*  GPU_Kenerl_t::MITM_size);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemset MITM_prob_interm_global failed!");
            goto Error;
        }
    }

    //Intermediate sync variable
    cudaStatus = cudaMalloc((void**)& device_last_dx_ptr, sizeof(int) * ( 2) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc device_last_dx_ptr @init failed!");
        goto Error;
    }
    cudaStatus = cudaMemset(device_last_dx_ptr, 0, sizeof(int) * (2) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_last_dx_ptr failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_branches_sum_before_dx, sizeof(unsigned long long) * ( 2 + 2 + 2 ) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc device_branches_sum_before_dx @init failed!");
        goto Error;
    }
    cudaStatus = cudaMemset(device_branches_sum_before_dx, 0, sizeof(unsigned long long) * (2 + 2 + 2 ) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_branches_sum_before_dx failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_has_operation, (sizeof(bool) * 2 ) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc  device_has_operation @init failed!");
        goto Error;
    }
    cudaStatus = cudaMemset(device_has_operation, 0, (sizeof(bool) * 2) );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset device_has_operation failed!");
        goto Error;
    }

    //Finished
    std::cout <<"\n----\n";

    return;
    
    Error:
    std::cout << "\nCritical CUDA Error. ";
    if (cudaStatus != cudaSuccess) {
        cudaError_t err = cudaGetLastError();
        std::cout << "\nCRITICAL ERROR from init...";
        fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
        std::cout << "\nExiting the program manually...";
        getchar();
        exit(-1);
    }
};

//Kernel - true_round used for bounding
// __launch_bounds__(THREAD_PER_BLOCK, 16) //FOR V100,P100
__global__ void kernel_diff(
    unsigned char* device_dx, int* device_sbox_index, float* device_prob, int* device_branch_size,
    unsigned long long* device_cluster_size_final, double* device_prob_final,
    int* device_last_dx_ptr, bool* device_has_operation, unsigned long long* device_branches_sum_before_dx,
    unsigned long long* device_branch_size_thread, unsigned long long* device_branch_size_block,
    unsigned long long* device_total_branch_size_block){
    // printf("\nInteger: %i, %i, block_size (threads per blocks) : %i",threadIdx.x, blockIdx.x,blockDim.x);
    grid_group grid = this_grid();

    if (threadIdx.x <32){
        if (threadIdx.x<16){
            SPN_DIFF::diff_table_size_shared[threadIdx.x] = SPN_DIFF::diff_table_size_global[threadIdx.x];

            for (int j = 0; j < 8; j++) {
                SPN_DIFF::diff_table_shared[threadIdx.x][j] = SPN_DIFF::diff_table_global[threadIdx.x][j];
                SPN_DIFF::prob_table_shared[threadIdx.x][j] = SPN_DIFF::prob_table_global[threadIdx.x][j]; 
            }
        }

        SPN_DIFF::prob_per_as_shared[threadIdx.x] = powf(CLUSTER_1AS_BEST_PROB, threadIdx.x+1); 
        SPN_DIFF::prob_per_round_remaining_shared[threadIdx.x] = powf(CLUSTER_PROB_INDIV, threadIdx.x);
    }
    __syncthreads();

	//Computing target array index (id and output_range)
    //I - THREAD ID / total thread (including all block) - Used to coordinate splitting of tasks
    const int thread_id_default = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int cur_r = 0;
    int flip_0_1 = 0; 
    // int flip_iter = 0;
    
    long long cur_iter = -1; //This has to be signed
    int dx_ptr_shared[MAX_ROUND_FORWARD]; 
    // int dx_ptr[MAX_ROUND_FORWARD] = {0}; //From 0- DX_num for that rounds
    unsigned long long branch_sum_before_dx_ptr_shared[MAX_ROUND_FORWARD] = {0};
    unsigned long long branch_sum_before_block_thread_ptr_shared[MAX_ROUND_FORWARD] = {0};
    unsigned long long branch_sum_before_block_ptr_shared[MAX_ROUND_FORWARD] = {0};
    long long iter_shared[MAX_ROUND_FORWARD]; //Has to be signed
    
    //Preparation
    //Prepare array to store, each entry to next rounds will require the storing and restoring of these into local memory
    int thread_id_arr[MAX_ROUND_FORWARD]; //Each round GRID_THREAD_SIZE is added to this
    int cur_thread_id; //Default value does not matter

    int dx_ptr = 0;
    unsigned long long branch_sum_before_dx_ptr = 0; 
    unsigned long long branch_sum_before_block_thread_ptr = 0; //From to 0-Block_num for that rounds
    // unsigned long long branch_sum_before_block_ptr = 0;

    //IO, need to be retarget after every rounds.
    //Output
    //NOTE: 32 here need to be changed
    #define output_dx(x) (( device_dx + ( 16 * MAX_PATH_PER_ROUND * (cur_r+1)) + (16 * thread_id_default * MAX_SPACE_PER_THREAD) + (x * 16) )) 
    #define output_sbox_index(x) ((  device_sbox_index + ( MAX_AS * MAX_PATH_PER_ROUND * (cur_r+1) ) + (MAX_AS* thread_id_default * MAX_SPACE_PER_THREAD) + (x * MAX_AS) ))
    #define output_prob(x) ((  device_prob +  ( MAX_PATH_PER_ROUND * (cur_r+1) ) + (thread_id_default * MAX_SPACE_PER_THREAD) + x))
    #define output_branch_size(x) (( device_branch_size + ( MAX_PATH_PER_ROUND * (cur_r+1) ) + (thread_id_default * MAX_SPACE_PER_THREAD) + x ))
    #define output_branch_size_thread() ((device_branch_size_thread + ( GRID_THREAD_SIZE * (cur_r+1) ) + thread_id_default))
    #define output_branch_size_block() ((device_branch_size_block + (BLOCK_NUM * (cur_r+1)) + blockIdx.x))
    #define output_branch_size_all() ((device_branch_size_block-2))
    // unsigned long long* output_branch_size_all = (device_branch_size_block-2);

    //Input
    #define cur_dx() ( ( device_dx + (16 * MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_sbox_index() ( ( device_sbox_index + (MAX_AS * MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_prob() ( ( device_prob + (MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_branch_size() ( ( device_branch_size + (MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_branch_size_thread() ( ( device_branch_size_thread + (GRID_THREAD_SIZE * cur_r) ) )
    #define cur_branch_size_block() ( (device_branch_size_block + (BLOCK_NUM * cur_r)) )
    
    //Mainloop -
    while(true){ //Base case, cur_0 and reamining == 0
        bool has_operation = false;
        int increment = 0; //Determine output save position..
        unsigned long long thread_branch_num_so_far = 0; //Allow accumulaction of block_thread branch_num (atomic_add with each reset)

        if(cur_iter == -1){
            cur_iter = output_branch_size_all()[flip_0_1]/GRID_THREAD_SIZE + (output_branch_size_all()[flip_0_1] % GRID_THREAD_SIZE != 0);
            cur_thread_id = thread_id_default * cur_iter;
        }
     
        //calculate block_thread_ptr.. initial 
        int block_thread_ptr = dx_ptr/MAX_SPACE_PER_THREAD;
        int block_ptr = block_thread_ptr / THREAD_PER_BLOCK;
        
        //Find the correct DX with three layer
        int loop_limit = cur_iter<MAX_SPACE_PER_THREAD?cur_iter:MAX_SPACE_PER_THREAD;
        for (int i=0;i<loop_limit;i++){
            if (dx_ptr < MAX_PATH_PER_ROUND){
                branch_sum_before_block_thread_ptr =  branch_sum_before_block_thread_ptr_shared[cur_r];
                
                //Shortcut 
                if (cur_thread_id <  (cur_branch_size()[dx_ptr] + branch_sum_before_dx_ptr)
                && cur_thread_id < (cur_branch_size_thread()[block_thread_ptr] + branch_sum_before_block_thread_ptr)
                ){
                    goto finfinddx; 
                }
    
                //Find the correct block
                unsigned long long cur_branch_size_reg0,cur_branch_size_reg1,cur_branch_size_reg2,cur_branch_size_reg3;
                unsigned long long cur_branch_size_reg[4];
                unsigned long long branches_temp = branch_sum_before_block_ptr_shared[cur_r];
                int initial_block_ptr = block_ptr;

                while(true){
                    cur_branch_size_reg0 = cur_branch_size_block()[block_ptr];
                    cur_branch_size_reg1 = cur_branch_size_block()[block_ptr+1];
                    cur_branch_size_reg2 = cur_branch_size_block()[block_ptr+2]; 
                    cur_branch_size_reg3 = cur_branch_size_block()[block_ptr+3];

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg0;
                    if (cur_branch_size_reg0 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg0;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg1;
                    if (cur_branch_size_reg1 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg1;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg2;
                    if (cur_branch_size_reg2 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg2;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;
                    
                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg3;
                    if (cur_branch_size_reg3 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg3;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;
                }

                if (true){
                    hasdx:
                    int initial_block_thread_ptr = block_thread_ptr;
                    if (initial_block_ptr == block_ptr){ //Found out block does not move
                        branches_temp = branch_sum_before_block_thread_ptr; //Take the old branch_size for block_thread (start off with offset block thread)
                        //block_thread_ptr remain unchanged...
                    }
                    else{ //New Block
                        block_thread_ptr = block_ptr * THREAD_PER_BLOCK; //Point to the 1st element of the block
                        //branches_temp remain unchanged because we are starting at 1st element...
                    }
    
                    //Find the correct block thread
                    bool is_found = false;
                    while(!is_found){
    
                        cur_branch_size_reg[0] = cur_branch_size_thread()[block_thread_ptr];
                        cur_branch_size_reg[1] = cur_branch_size_thread()[block_thread_ptr+1];
                        cur_branch_size_reg[2] = cur_branch_size_thread()[block_thread_ptr+2]; 
                        cur_branch_size_reg[3] = cur_branch_size_thread()[block_thread_ptr+3];
    
                        #pragma unroll
                        for (int i=0;i<4;i++){
                            branches_temp += cur_branch_size_reg[i];
                            if (cur_branch_size_reg[i] != 0 && cur_thread_id < branches_temp){
                                branches_temp -= cur_branch_size_reg[i];
                                block_thread_ptr += i;
                                is_found = true;
                                break;
                            }
                        }
                        if(!is_found){
                            block_thread_ptr += 4;
                        }
                    }
                    branch_sum_before_block_thread_ptr_shared[cur_r] = branches_temp;
    
                    //Advance the dx position if needed (different block_thread location)
                    if (block_thread_ptr == initial_block_thread_ptr){
                        //Start at the same location
                        branches_temp = branch_sum_before_dx_ptr;
                    }
                    else{
                        dx_ptr = block_thread_ptr * MAX_SPACE_PER_THREAD;
                    }
    
                    //Find the correct dx position
                    is_found = false;
                    while(!is_found){
    
                        cur_branch_size_reg[0] = cur_branch_size()[dx_ptr];
                        cur_branch_size_reg[1] = cur_branch_size()[dx_ptr+1];
                        cur_branch_size_reg[2] = cur_branch_size()[dx_ptr+2]; 
                        cur_branch_size_reg[3] = cur_branch_size()[dx_ptr+3];
                        
                        #pragma unroll
                        for (int i=0;i<4;i++){
                            branches_temp += cur_branch_size_reg[i];
                            if (cur_branch_size_reg[i] != 0 && cur_thread_id < branches_temp){
                                branches_temp -= cur_branch_size_reg[i];
                                dx_ptr += i;
                                is_found = true;
                                break;
                            }
                        }
                        if(!is_found){
                            dx_ptr += 4;
                        }
                    }
                    branch_sum_before_dx_ptr = branches_temp;
                }
                else{
                    //Nothing here
                }
            }
    
            finfinddx:
            ;
    
            if (dx_ptr < MAX_PATH_PER_ROUND){ //If dx_ptr is within dx_num, [0-N)
    
                has_operation = true;
                
                float prob_thread = 1.0;
                int divide_factor = 1;
                unsigned int diff_freq_index; //0-16 only
                unsigned int remaining_value = cur_thread_id - branch_sum_before_dx_ptr ; //7^8 is less than 32 bit...
    
                unsigned char* cur_dx_temp = cur_dx() + ( 16 * dx_ptr ) ; //NOTE: Need to modify to fit datastruct of different cipher
                int* cur_sbox_index_ptr = cur_sbox_index() + (MAX_AS  * dx_ptr);
                unsigned char cur_thread_partial_dy[17];
                cur_thread_partial_dy[16] = {0};
                memcpy(cur_thread_partial_dy,cur_dx_temp,16);
                int cur_sbox_index_temp[MAX_AS];
                memcpy(cur_sbox_index_temp, cur_sbox_index_ptr, sizeof(int) * MAX_AS);
                //Points to correct i_th branches of j_dx and so subs
                #pragma unroll
                for (int i = 0; i < MAX_AS; i++) {
                    unsigned char cur_val = cur_thread_partial_dy[cur_sbox_index_temp[i]];
                    diff_freq_index = (remaining_value / divide_factor) % SPN_DIFF::diff_table_size_shared[cur_val]; 
                    cur_thread_partial_dy[cur_sbox_index_temp[i]] = SPN_DIFF::diff_table_shared[cur_val][diff_freq_index]; //Assigning target val to partial_dy
                    prob_thread *= (SPN_DIFF::prob_table_shared[cur_val][diff_freq_index]);
                    divide_factor *= SPN_DIFF::diff_table_size_shared[cur_val];
                }
                prob_thread *= (*(cur_prob() + dx_ptr));
    
                if (cur_r+1 != MAX_ROUND_FORWARD){
                    //Do Permutate
                    unsigned long long front_64 = 0;
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        if ( cur_thread_partial_dy[i] > 0) {
                            front_64 |= SPN_DIFF::perm_lookup_global_forward[i][cur_thread_partial_dy[i]];
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        cur_thread_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;  
                    }
    
                    //Calculte sbox index and sbox number
                    int save_sbox_num = 0;
                    int save_branch_size = 1;
                    int save_sbox_index[16]; //Will point to non existance 32 array entry (see substitution below)
                    #pragma unroll
                    for (int i=0;i< 16;i++){
                        save_sbox_index[i] = 16; 
                    }
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        if ((cur_thread_partial_dy[i] & 0xf) > 0) {
                            save_branch_size *= SPN_DIFF::diff_table_size_shared[cur_thread_partial_dy[i]];
                            save_sbox_index[save_sbox_num] = i;
                            save_sbox_num++;
                        }
                    }
    
                    //Pruning
                    // if(true){
                    if (save_sbox_num <= MAX_AS){  //If only next round AS <= 8
                        //MATSUI BOUND
                        float estimated_com_prob = SPN_DIFF::prob_per_round_remaining_shared[(MAX_ROUND_FORWARD - cur_r - 2)] * SPN_DIFF::prob_per_as_shared[save_sbox_num-1]; //NOTE: this bound is less tight when round entered is not zero..
                        if ((estimated_com_prob * prob_thread) >= SPN_DIFF::CLUSTER_PROB_BOUND_const) {
                        // if (true) {
                            memcpy(output_dx(increment),cur_thread_partial_dy,16);
                            *output_prob(increment) = prob_thread;
                            memcpy(output_sbox_index(increment), save_sbox_index, sizeof(int) * MAX_AS );
                            *output_branch_size(increment) = save_branch_size;
    
                            thread_branch_num_so_far += save_branch_size;
                            increment += 1; 
                            
                        } 
                        // else{ *output_branch_size = 0;} 
                    }
                    // else{ *output_branch_size = 0;}
                }
                //LAST ROUNDS... no permutation and straight to savings.
                else{
                    bool is_same = true;
                    #pragma unroll
                    for (int i=0;i<16;i++){
                        if (SPN_DIFF::final_dy_constant[i] != cur_thread_partial_dy[i]){
                            is_same= false;
                            break;
                        }
                    }
            
                    if (is_same){
                        device_prob_final[thread_id_default] += prob_thread;
                        device_cluster_size_final[thread_id_default] += 1;
                    }
                }
            }
            cur_thread_id+=1;
        }
        cur_thread_id-=1;
        cur_iter = cur_iter - loop_limit;

        if(thread_id_default == 0){ 
            *(device_has_operation + (flip_0_1) ) = has_operation;
        }
        if (cur_r != MAX_ROUND_FORWARD-1){
            *output_branch_size_thread() += thread_branch_num_so_far; // so_far will be reset each sync thus adding like this is correct.
            atomicAdd(&SPN_DIFF::branch_size_block_shared[0], thread_branch_num_so_far);
            __syncthreads();
            if (threadIdx.x==0){
                // iter_shared[cur_r] = cur_iter;
                *output_branch_size_block() += SPN_DIFF::branch_size_block_shared[0];
                //Since the operation is once per output round, reset the stuff here
                atomicAdd( (output_branch_size_all()+!flip_0_1), SPN_DIFF::branch_size_block_shared[0]);

                SPN_DIFF::branch_size_block_shared[0] = 0;
            }
        }

        grid.sync(); //Wait for grid to synchronize before continue
        if (thread_id_default==0){
            output_branch_size_all()[flip_0_1] = 0;
        }
        grid.sync(); //

        has_operation = *(device_has_operation + (flip_0_1) );
        flip_0_1 = !flip_0_1;

        if(true){
            if (cur_r != MAX_ROUND_FORWARD-1 && has_operation){ //Is not last round and has operation
                //Goes forwards
                iter_shared[cur_r] = cur_iter;
                dx_ptr_shared[cur_r] = dx_ptr;
                branch_sum_before_dx_ptr_shared[cur_r] = branch_sum_before_dx_ptr;
                thread_id_arr[cur_r] = cur_thread_id;
                cur_r+=1;

                // cur_thread_id = thread_id_default; //NOTE: does not matter
                dx_ptr = 0;
                branch_sum_before_dx_ptr = 0;
                branch_sum_before_block_thread_ptr_shared[cur_r] = 0;
                branch_sum_before_block_ptr_shared[cur_r] = 0;
                cur_iter = -1; //Signal the requirement of intiialization

                if (cur_r!=MAX_ROUND_FORWARD-1){
                    *output_branch_size_thread() = 0; //HAs to be reset because of tunneling
                    if (threadIdx.x==0){
                        *output_branch_size_block()  = 0;
                    }
                }
            }
            else if(!has_operation || (cur_r == MAX_ROUND_FORWARD-1 && cur_iter == 0) ){ //Has no operation => cur_iter == 0, 
                //Goes backwards if last rounds or current rounds does not process anythings.
                do{
                    cur_r-=1;
                    if(cur_r == -1){
                        return; //NOTE: Completed computation, Base Case
                    }
                    cur_iter = iter_shared[cur_r];
                }while(cur_iter==0);

                cur_thread_id = thread_id_arr[cur_r] + 1;
                dx_ptr = dx_ptr_shared[cur_r];
                branch_sum_before_dx_ptr = branch_sum_before_dx_ptr_shared[cur_r];

                *output_branch_size_thread() = 0;
                if (threadIdx.x==0){
                    *output_branch_size_block()  = 0;
                }
            }
            else{ //Has operation and is last round and cur_iter != 0 
                //Repeat last rounds.
                cur_thread_id += 1;
            }
        }
    }
};

__launch_bounds__(THREAD_PER_BLOCK, 8) //FOR V100,P100
__global__ void kernel_diff_mitm(
    unsigned char* device_dx, int* device_sbox_index, float* device_prob, int* device_branch_size,
    unsigned long long* device_cluster_size_final, double* device_prob_final,
    int* device_last_dx_ptr, bool* device_has_operation, unsigned long long* device_branches_sum_before_dx,
    unsigned long long* device_branch_size_thread, unsigned long long* device_branch_size_block,
    unsigned long long* device_total_branch_size_block,
    float* MITM_prob_interm_global, unsigned long long* MITM_size_interm_global){
    // printf("\nInteger: %i, %i, block_size (threads per blocks) : %i",threadIdx.x, blockIdx.x,blockDim.x);
    grid_group grid = this_grid();

    if (threadIdx.x <32){
        if (threadIdx.x<16){
            SPN_DIFF::diff_table_size_shared[threadIdx.x] = SPN_DIFF::diff_table_size_global[threadIdx.x];

            for (int j = 0; j < 8; j++) {
                SPN_DIFF::diff_table_shared[threadIdx.x][j] = SPN_DIFF::diff_table_global[threadIdx.x][j];
                SPN_DIFF::prob_table_shared[threadIdx.x][j] = SPN_DIFF::prob_table_global[threadIdx.x][j]; 
            }
        }
        SPN_DIFF::prob_per_as_shared[threadIdx.x] = powf(CLUSTER_1AS_BEST_PROB, threadIdx.x+1); 
        SPN_DIFF::prob_per_round_remaining_shared[threadIdx.x] = powf(CLUSTER_PROB_INDIV, threadIdx.x);
    }
    __syncthreads(); //wait for init to be finished, sync up all threads within a block... shared memory lies within each block.

	//Computing target array index (id and output_range)
    //I - THREAD ID / total thread (including all block) - Used to coordinate splitting of tasks
    const int thread_id_default = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int cur_r = 0;
    int flip_0_1 = 0; 
    
    long long cur_iter = -1; //This has to be signed
    int dx_ptr_shared[MAX_ROUND_FORWARD]; 
    unsigned long long branch_sum_before_dx_ptr_shared[MAX_ROUND_FORWARD] = {0};
    unsigned long long branch_sum_before_block_thread_ptr_shared[MAX_ROUND_FORWARD] = {0};
    unsigned long long branch_sum_before_block_ptr_shared[MAX_ROUND_FORWARD] = {0};
    long long iter_shared[MAX_ROUND_FORWARD]; //Has to be signed
    
    //Preparation
    //Prepare array to store, each entry to next rounds will require the storing and restoring of these into local memory
    int thread_id_arr[MAX_ROUND_FORWARD]; //Each round GRID_THREAD_SIZE is added to this
    int cur_thread_id; //Default value does not matter

    // int dx_ptr[MAX_ROUND_FORWARD] = {0}; //From 0- DX_num for that rounds
    int dx_ptr = 0;
    unsigned long long branch_sum_before_dx_ptr = 0; 
    unsigned long long branch_sum_before_block_thread_ptr = 0; //From to 0-Block_num for that rounds
    // unsigned long long branch_sum_before_block_ptr = 0;

    //IO, need to be retarget after every rounds.
    //Output
    #define output_dx(x) (( device_dx + ( 16 * MAX_PATH_PER_ROUND * (cur_r+1)) + (16 * thread_id_default * MAX_SPACE_PER_THREAD) + (x * 16) )) 
    #define output_sbox_index(x) ((  device_sbox_index + ( MAX_AS * MAX_PATH_PER_ROUND * (cur_r+1) ) + (MAX_AS* thread_id_default * MAX_SPACE_PER_THREAD) + (x * MAX_AS) ))
    #define output_prob(x) ((  device_prob +  ( MAX_PATH_PER_ROUND * (cur_r+1) ) + (thread_id_default * MAX_SPACE_PER_THREAD) + x))
    #define output_branch_size(x) (( device_branch_size + ( MAX_PATH_PER_ROUND * (cur_r+1) ) + (thread_id_default * MAX_SPACE_PER_THREAD) + x ))
    #define output_branch_size_thread() ((device_branch_size_thread + ( GRID_THREAD_SIZE * (cur_r+1) ) + thread_id_default))
    #define output_branch_size_block() ((device_branch_size_block + (BLOCK_NUM * (cur_r+1)) + blockIdx.x))
    #define output_branch_size_all() ((device_branch_size_block-2))
    // unsigned long long* output_branch_size_all = (device_branch_size_block-2);

    //Input
    #define cur_dx() ( ( device_dx + (16 * MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_sbox_index() ( ( device_sbox_index + (MAX_AS * MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_prob() ( ( device_prob + (MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_branch_size() ( ( device_branch_size + (MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_branch_size_thread() ( ( device_branch_size_thread + (GRID_THREAD_SIZE * cur_r) ) )
    #define cur_branch_size_block() ( (device_branch_size_block + (BLOCK_NUM * cur_r)) )
    
    //Mainloop -
    while(true){ //Base case, cur_0 and reamining == 0
        bool has_operation = false;
        int increment = 0; //Determine output save position..
        unsigned long long thread_branch_num_so_far = 0; //Allow accumulaction of block_thread branch_num (atomic_add with each reset)

        if(cur_iter == -1){
            // cur_iter = ceil(1.0 * output_branch_size_all()[flip_0_1]/GRID_THREAD_SIZE);
            cur_iter = output_branch_size_all()[flip_0_1]/GRID_THREAD_SIZE + (output_branch_size_all()[flip_0_1] % GRID_THREAD_SIZE != 0);
            cur_thread_id = thread_id_default * cur_iter;
            // flip_iter = !flip_iter;
        }
     
        //calculate block_thread_ptr.. initial 
        int block_thread_ptr = dx_ptr/MAX_SPACE_PER_THREAD;
        int block_ptr = block_thread_ptr / THREAD_PER_BLOCK;
        
        //Find the correct DX with three layer
        int loop_limit = cur_iter<MAX_SPACE_PER_THREAD?cur_iter:MAX_SPACE_PER_THREAD;
        for (int i=0;i<loop_limit;i++){
            if (dx_ptr < MAX_PATH_PER_ROUND){ 
                branch_sum_before_block_thread_ptr =  branch_sum_before_block_thread_ptr_shared[cur_r];
                
                //Shortcut 
                if (cur_thread_id <  (cur_branch_size()[dx_ptr] + branch_sum_before_dx_ptr)
                && cur_thread_id < (cur_branch_size_thread()[block_thread_ptr] + branch_sum_before_block_thread_ptr)
                ){
                    goto finfinddx; 
                }
    
                //Find the correct block
                unsigned long long cur_branch_size_reg0,cur_branch_size_reg1,cur_branch_size_reg2,cur_branch_size_reg3;
                unsigned long long cur_branch_size_reg[4];
                unsigned long long branches_temp = branch_sum_before_block_ptr_shared[cur_r];
                int initial_block_ptr = block_ptr;

                while(true){
                    cur_branch_size_reg0 = cur_branch_size_block()[block_ptr];
                    cur_branch_size_reg1 = cur_branch_size_block()[block_ptr+1];
                    cur_branch_size_reg2 = cur_branch_size_block()[block_ptr+2]; 
                    cur_branch_size_reg3 = cur_branch_size_block()[block_ptr+3];

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg0;
                    if (cur_branch_size_reg0 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg0;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg1;
                    if (cur_branch_size_reg1 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg1;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg2;
                    if (cur_branch_size_reg2 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg2;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;
                    
                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg3;
                    if (cur_branch_size_reg3 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg3;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;
                }

                if (true){
                    hasdx:
                    int initial_block_thread_ptr = block_thread_ptr;
                    if (initial_block_ptr == block_ptr){ //Found out block does not move
                        branches_temp = branch_sum_before_block_thread_ptr; //Take the old branch_size for block_thread (start off with offset block thread)
                        //block_thread_ptr remain unchanged...
                    }
                    else{ //New Block
                        block_thread_ptr = block_ptr * THREAD_PER_BLOCK; //Point to the 1st element of the block
                        //branches_temp remain unchanged because we are starting at 1st element...
                    }
    
                    //Find the correct block thread
                    bool is_found = false;
                    while(!is_found){
    
                        cur_branch_size_reg[0] = cur_branch_size_thread()[block_thread_ptr];
                        cur_branch_size_reg[1] = cur_branch_size_thread()[block_thread_ptr+1];
                        cur_branch_size_reg[2] = cur_branch_size_thread()[block_thread_ptr+2]; 
                        cur_branch_size_reg[3] = cur_branch_size_thread()[block_thread_ptr+3];
    
                        #pragma unroll
                        for (int i=0;i<4;i++){
                            branches_temp += cur_branch_size_reg[i];
                            if (cur_branch_size_reg[i] != 0 && cur_thread_id < branches_temp){
                                branches_temp -= cur_branch_size_reg[i];
                                block_thread_ptr += i;
                                is_found = true;
                                break;
                            }
                        }
                        if(!is_found){
                            block_thread_ptr += 4;
                        }
                    }
                    branch_sum_before_block_thread_ptr_shared[cur_r] = branches_temp;
    
                    //Advance the dx position if needed (different block_thread location)
                    if (block_thread_ptr == initial_block_thread_ptr){
                        //Start at the same location
                        branches_temp = branch_sum_before_dx_ptr;
                    }
                    else{
                        dx_ptr = block_thread_ptr * MAX_SPACE_PER_THREAD;
                    }
    
                    //Find the correct dx position
                    is_found = false;
                    while(!is_found){
    
                        cur_branch_size_reg[0] = cur_branch_size()[dx_ptr];
                        cur_branch_size_reg[1] = cur_branch_size()[dx_ptr+1];
                        cur_branch_size_reg[2] = cur_branch_size()[dx_ptr+2]; 
                        cur_branch_size_reg[3] = cur_branch_size()[dx_ptr+3];
                        
                        #pragma unroll
                        for (int i=0;i<4;i++){
                            //NOTE: no need to check out of bounds if correctly impleneted, it will be filtered out at block level..
                            branches_temp += cur_branch_size_reg[i];
                            if (cur_branch_size_reg[i] != 0 && cur_thread_id < branches_temp){
                                branches_temp -= cur_branch_size_reg[i];
                                dx_ptr += i;
                                is_found = true;
                                break;
                            }
                        }
                        if(!is_found){
                            dx_ptr += 4;
                        }
                    }
                    branch_sum_before_dx_ptr = branches_temp;
                }
                else{
                    //Nothing here
                }
            }
    
            finfinddx:
            ;
    
            if (dx_ptr < MAX_PATH_PER_ROUND){ //If dx_ptr is within dx_num, [0-N)
    
                has_operation = true;
                
                float prob_thread = 1.0;
                int divide_factor = 1;
                unsigned int diff_freq_index; //0-16 only
                unsigned int remaining_value = cur_thread_id - branch_sum_before_dx_ptr ; //7^8 is less than 32 bit...
    
                unsigned char* cur_dx_temp = cur_dx() + ( 16 * dx_ptr ) ; //NOTE: Need to modify to fit datastruct of different cipher
                int* cur_sbox_index_ptr = cur_sbox_index() + (MAX_AS  * dx_ptr);
                unsigned char cur_thread_partial_dy[17];
                cur_thread_partial_dy[16] = {0};
                memcpy(cur_thread_partial_dy,cur_dx_temp,16);
                int cur_sbox_index_temp[MAX_AS];
                memcpy(cur_sbox_index_temp, cur_sbox_index_ptr, sizeof(int) * MAX_AS);
                //Points to correct i_th branches of j_dx and so subs
                #pragma unroll
                for (int i = 0; i < MAX_AS; i++) {
                    unsigned char cur_val = cur_thread_partial_dy[cur_sbox_index_temp[i]];
                    diff_freq_index = (remaining_value / divide_factor) % SPN_DIFF::diff_table_size_shared[cur_val]; 
                    cur_thread_partial_dy[cur_sbox_index_temp[i]] = SPN_DIFF::diff_table_shared[cur_val][diff_freq_index]; //Assigning target val to partial_dy
                    prob_thread *= (SPN_DIFF::prob_table_shared[cur_val][diff_freq_index]);
                    divide_factor *= SPN_DIFF::diff_table_size_shared[cur_val];
                }
                prob_thread *= (*(cur_prob() + dx_ptr));

                //Do Permutate
                unsigned long long front_64 = 0;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    if ( cur_thread_partial_dy[i] > 0) {
                        //Permutation LUTable
                        //TODO: require modify to feed in correct forward/backward
                        front_64 |= SPN_DIFF::perm_lookup_global_forward[i][cur_thread_partial_dy[i]];
                    }
                }
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    cur_thread_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;  
                }

                if (cur_r != MAX_ROUND_FORWARD-1){
                    //Calculte sbox index and sbox number
                    int save_sbox_num = 0;
                    int save_branch_size = 1;
                    int save_sbox_index[16]; //Will point to non existance 32 array entry (see substitution below)
                    #pragma unroll
                    for (int i=0;i< 16;i++){
                        save_sbox_index[i] = 16; 
                    }
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        if ((cur_thread_partial_dy[i] & 0xf) > 0) {
                            save_branch_size *= SPN_DIFF::diff_table_size_shared[cur_thread_partial_dy[i]];
                            save_sbox_index[save_sbox_num] = i;
                            save_sbox_num++;
                        }
                    }
    
                    //Pruning
                    // if(true){
                    if (save_sbox_num <= MAX_AS){  //If only next round AS <= 8
                        //MATSUI BOUND
                        float estimated_com_prob = SPN_DIFF::prob_per_round_remaining_shared[(MAX_ROUND_FORWARD - cur_r - 2)] * SPN_DIFF::prob_per_as_shared[save_sbox_num-1]; //NOTE: this bound is less tight when round entered is not zero..
                        if ((estimated_com_prob * prob_thread) >= SPN_DIFF::CLUSTER_PROB_BOUND_const) {
                        // if (true) {
                            memcpy(output_dx(increment),cur_thread_partial_dy,16);
                            *output_prob(increment) = prob_thread;
                            memcpy(output_sbox_index(increment), save_sbox_index, sizeof(int) * MAX_AS );
                            *output_branch_size(increment) = save_branch_size;
    
                            thread_branch_num_so_far += save_branch_size;
                            increment += 1; 
                            
                        } 
                        // else{ *output_branch_size = 0;} 
                    }
                    // else{ *output_branch_size = 0;}
                }
                //LAST ROUNDS... no permutation and straight to savings.
                else{
                    int sbox_num=0;
                    int sbox_index[16]={0};
                    #pragma unroll
                    for (int i=0;i<16;i++){
                        if (cur_thread_partial_dy[i] !=0){
                            sbox_index[sbox_num] = i;
                            sbox_num+=1;
                        }
                    }
        
                    if (sbox_num <=3){ //Possible to store three only...
                        //Computing appropriate index
                        int index=0;
                        #pragma unroll
                        for (int i=0;i<sbox_num;i++){
                            index|= ( ( (sbox_index[i]&0b11111) | ( (cur_thread_partial_dy[sbox_index[i]]&0b1111) << 5) ) << (i * 9) ); 
                        }

                        atomicAdd(MITM_prob_interm_global+index,prob_thread);
                        atomicAdd(MITM_size_interm_global+index,1);
                    }
                }
            }
            cur_thread_id+=1;
        }
        cur_thread_id-=1;
        // cur_iter = cur_iter > 0?cur_iter-loop_limit:cur_iter; //Make sure >0 => -1, if 0 left it
        cur_iter = cur_iter - loop_limit;

        if(thread_id_default == 0){ 
            *(device_has_operation + (flip_0_1) ) = has_operation;
        }
        if (cur_r != MAX_ROUND_FORWARD-1){
            *output_branch_size_thread() += thread_branch_num_so_far; // so_far will be reset each sync thus adding like this is correct.
            atomicAdd(&SPN_DIFF::branch_size_block_shared[0], thread_branch_num_so_far);
            __syncthreads();
            if (threadIdx.x==0){
                // iter_shared[cur_r] = cur_iter;
                *output_branch_size_block() += SPN_DIFF::branch_size_block_shared[0];
                //Since the operation is once per output round, reset the stuff here
                atomicAdd( (output_branch_size_all()+!flip_0_1), SPN_DIFF::branch_size_block_shared[0]);

                SPN_DIFF::branch_size_block_shared[0] = 0;
            }
        }

        grid.sync(); //Wait for grid to synchronize before continue
        if (thread_id_default==0){
            output_branch_size_all()[flip_0_1] = 0;
        }
        grid.sync();

        has_operation = *(device_has_operation + (flip_0_1) );
        flip_0_1 = !flip_0_1;

        if(true){
            if (cur_r != MAX_ROUND_FORWARD-1 && has_operation){ //Is not last round and has operation
                //Goes forwards
                iter_shared[cur_r] = cur_iter;
                dx_ptr_shared[cur_r] = dx_ptr;
                branch_sum_before_dx_ptr_shared[cur_r] = branch_sum_before_dx_ptr;
                thread_id_arr[cur_r] = cur_thread_id;
                cur_r+=1;

                dx_ptr = 0;
                branch_sum_before_dx_ptr = 0;
                branch_sum_before_block_thread_ptr_shared[cur_r] = 0;
                branch_sum_before_block_ptr_shared[cur_r] = 0;
                cur_iter = -1; //Signal the requirement of intiialization

                if (cur_r!=MAX_ROUND_FORWARD-1){
                    *output_branch_size_thread() = 0; //HAs to be reset because of tunneling
                    if (threadIdx.x==0){
                        *output_branch_size_block()  = 0;
                    }
                }
            }
            else if(!has_operation || (cur_r == MAX_ROUND_FORWARD-1 && cur_iter == 0) ){ //Has no operation => cur_iter == 0, 
                //Goes backwards if last rounds or current rounds does not process anythings.
                do{
                    cur_r-=1;
                    if(cur_r == -1){
                        return; //NOTE: Completed computation, Base Case
                    }
                    cur_iter = iter_shared[cur_r];
                }while(cur_iter==0);

                cur_thread_id = thread_id_arr[cur_r] + 1;
                dx_ptr = dx_ptr_shared[cur_r];
                branch_sum_before_dx_ptr = branch_sum_before_dx_ptr_shared[cur_r];

                *output_branch_size_thread() = 0;
                if (threadIdx.x==0){
                    *output_branch_size_block()  = 0;
                }
            }
            else{ //Has operation and is last round and cur_iter != 0 
                //Repeat last rounds.
                cur_thread_id += 1;
            }
        }
    }
};

__launch_bounds__(THREAD_PER_BLOCK, 8)
__global__ void kernel_diff_mitm_backward(
    unsigned char* device_dx, int* device_sbox_index, float* device_prob, int* device_branch_size,
    unsigned long long* device_cluster_size_final, double* device_prob_final,
    int* device_last_dx_ptr, bool* device_has_operation, unsigned long long* device_branches_sum_before_dx,
    unsigned long long* device_branch_size_thread, unsigned long long* device_branch_size_block,
    unsigned long long* device_total_branch_size_block,
    float* MITM_prob_interm_global, unsigned long long* MITM_size_interm_global){
    // printf("\nInteger: %i, %i, block_size (threads per blocks) : %i",threadIdx.x, blockIdx.x,blockDim.x);
    grid_group grid = this_grid();

    if (threadIdx.x <32){
        if (threadIdx.x<16){
            SPN_DIFF::diff_table_size_shared[threadIdx.x] = SPN_DIFF::diff_table_size_global_reversed[threadIdx.x];

            for (int j = 0; j < 8; j++) {
                SPN_DIFF::diff_table_shared[threadIdx.x][j] = SPN_DIFF::diff_table_global_reversed[threadIdx.x][j];
                SPN_DIFF::prob_table_shared[threadIdx.x][j] = SPN_DIFF::prob_table_global_reversed[threadIdx.x][j]; 
            }
        }
        SPN_DIFF::prob_per_as_shared[threadIdx.x] = powf(CLUSTER_1AS_BEST_PROB, threadIdx.x+1); 
        SPN_DIFF::prob_per_round_remaining_shared[threadIdx.x] = powf(CLUSTER_PROB_INDIV, threadIdx.x);
    }
    __syncthreads(); //wait for init to be finished, sync up all threads within a block... shared memory lies within each block.

	//Computing target array index (id and output_range)
    //I - THREAD ID / total thread (including all block) - Used to coordinate splitting of tasks
    const int thread_id_default = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int cur_r = 0;
    int flip_0_1 = 0; 
    
    long long cur_iter = -1; //This has to be signed
    int dx_ptr_shared[MAX_ROUND_BACKWARD]; 
    unsigned long long branch_sum_before_dx_ptr_shared[MAX_ROUND_BACKWARD] = {0};
    unsigned long long branch_sum_before_block_thread_ptr_shared[MAX_ROUND_BACKWARD] = {0};
    unsigned long long branch_sum_before_block_ptr_shared[MAX_ROUND_BACKWARD] = {0};
    long long iter_shared[MAX_ROUND_BACKWARD]; //Has to be signed
    
    //Preparation
    //Prepare array to store, each entry to next rounds will require the storing and restoring of these into local memory
    int thread_id_arr[MAX_ROUND_BACKWARD]; //Each round GRID_THREAD_SIZE is added to this
    int cur_thread_id; //Default value does not matter

    // int dx_ptr[MAX_ROUND_FORWARD] = {0}; //From 0- DX_num for that rounds
    int dx_ptr = 0;
    unsigned long long branch_sum_before_dx_ptr = 0; 
    unsigned long long branch_sum_before_block_thread_ptr = 0; //From to 0-Block_num for that rounds
    // unsigned long long branch_sum_before_block_ptr = 0;

    //IO, need to be retarget after every rounds.
    //Output
    #define output_dx(x) (( device_dx + ( 16 * MAX_PATH_PER_ROUND * (cur_r+1)) + (16 * thread_id_default * MAX_SPACE_PER_THREAD) + (x * 16) )) 
    #define output_sbox_index(x) ((  device_sbox_index + ( MAX_AS * MAX_PATH_PER_ROUND * (cur_r+1) ) + (MAX_AS* thread_id_default * MAX_SPACE_PER_THREAD) + (x * MAX_AS) ))
    #define output_prob(x) ((  device_prob +  ( MAX_PATH_PER_ROUND * (cur_r+1) ) + (thread_id_default * MAX_SPACE_PER_THREAD) + x))
    #define output_branch_size(x) (( device_branch_size + ( MAX_PATH_PER_ROUND * (cur_r+1) ) + (thread_id_default * MAX_SPACE_PER_THREAD) + x ))
    #define output_branch_size_thread() ((device_branch_size_thread + ( GRID_THREAD_SIZE * (cur_r+1) ) + thread_id_default))
    #define output_branch_size_block() ((device_branch_size_block + (BLOCK_NUM * (cur_r+1)) + blockIdx.x))
    #define output_branch_size_all() ((device_branch_size_block-2))
    // unsigned long long* output_branch_size_all = (device_branch_size_block-2);

    //Input
    #define cur_dx() ( ( device_dx + (16 * MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_sbox_index() ( ( device_sbox_index + (MAX_AS * MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_prob() ( ( device_prob + (MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_branch_size() ( ( device_branch_size + (MAX_PATH_PER_ROUND * cur_r) ) )
    #define cur_branch_size_thread() ( ( device_branch_size_thread + (GRID_THREAD_SIZE * cur_r) ) )
    #define cur_branch_size_block() ( (device_branch_size_block + (BLOCK_NUM * cur_r)) )
    
    //Mainloop -
    while(true){ //Base case, cur_0 and reamining == 0
        bool has_operation = false;
        int increment = 0; //Determine output save position..
        unsigned long long thread_branch_num_so_far = 0; //Allow accumulaction of block_thread branch_num (atomic_add with each reset)

        if(cur_iter == -1){
            // cur_iter = ceil(1.0 * output_branch_size_all()[flip_0_1]/GRID_THREAD_SIZE);
            cur_iter = output_branch_size_all()[flip_0_1]/GRID_THREAD_SIZE + (output_branch_size_all()[flip_0_1] % GRID_THREAD_SIZE != 0);
            cur_thread_id = thread_id_default * cur_iter;
        }
     
        //calculate block_thread_ptr.. initial 
        int block_thread_ptr = dx_ptr/MAX_SPACE_PER_THREAD;
        int block_ptr = block_thread_ptr / THREAD_PER_BLOCK;
        
        //Find the correct DX with three layer
        int loop_limit = cur_iter<MAX_SPACE_PER_THREAD?cur_iter:MAX_SPACE_PER_THREAD;
        for (int i=0;i<loop_limit;i++){
            if (dx_ptr < MAX_PATH_PER_ROUND){ 
                branch_sum_before_block_thread_ptr =  branch_sum_before_block_thread_ptr_shared[cur_r];
                
                //Shortcut 
                if (cur_thread_id <  (cur_branch_size()[dx_ptr] + branch_sum_before_dx_ptr)
                && cur_thread_id < (cur_branch_size_thread()[block_thread_ptr] + branch_sum_before_block_thread_ptr)
                ){
                    goto finfinddx; 
                }
    
                //Find the correct block
                unsigned long long cur_branch_size_reg0,cur_branch_size_reg1,cur_branch_size_reg2,cur_branch_size_reg3;
                unsigned long long cur_branch_size_reg[4];
                unsigned long long branches_temp = branch_sum_before_block_ptr_shared[cur_r];
                int initial_block_ptr = block_ptr;

                while(true){
                    cur_branch_size_reg0 = cur_branch_size_block()[block_ptr];
                    cur_branch_size_reg1 = cur_branch_size_block()[block_ptr+1];
                    cur_branch_size_reg2 = cur_branch_size_block()[block_ptr+2]; 
                    cur_branch_size_reg3 = cur_branch_size_block()[block_ptr+3];

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg0;
                    if (cur_branch_size_reg0 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg0;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg1;
                    if (cur_branch_size_reg1 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg1;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;

                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg2;
                    if (cur_branch_size_reg2 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg2;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;
                    
                    if( block_ptr >= BLOCK_NUM){
                        dx_ptr = MAX_PATH_PER_ROUND; //Indicate nothing to do, end of dx
                        goto finfinddx;
                    }
                    branches_temp += cur_branch_size_reg3;
                    if (cur_branch_size_reg3 != 0 && cur_thread_id < branches_temp){
                        branches_temp -= cur_branch_size_reg3;
                        branch_sum_before_block_ptr_shared[cur_r] = branches_temp;
                        goto hasdx;
                    }
                    block_ptr+=1;
                }

                if (true){
                    hasdx:
                    int initial_block_thread_ptr = block_thread_ptr;
                    if (initial_block_ptr == block_ptr){ //Found out block does not move
                        branches_temp = branch_sum_before_block_thread_ptr; //Take the old branch_size for block_thread (start off with offset block thread)
                        //block_thread_ptr remain unchanged...
                    }
                    else{ //New Block
                        block_thread_ptr = block_ptr * THREAD_PER_BLOCK; //Point to the 1st element of the block
                        //branches_temp remain unchanged because we are starting at 1st element...
                    }
    
                    //Find the correct block thread
                    bool is_found = false;
                    while(!is_found){
    
                        cur_branch_size_reg[0] = cur_branch_size_thread()[block_thread_ptr];
                        cur_branch_size_reg[1] = cur_branch_size_thread()[block_thread_ptr+1];
                        cur_branch_size_reg[2] = cur_branch_size_thread()[block_thread_ptr+2]; 
                        cur_branch_size_reg[3] = cur_branch_size_thread()[block_thread_ptr+3];
    
                        #pragma unroll
                        for (int i=0;i<4;i++){
                            branches_temp += cur_branch_size_reg[i];
                            if (cur_branch_size_reg[i] != 0 && cur_thread_id < branches_temp){
                                branches_temp -= cur_branch_size_reg[i];
                                block_thread_ptr += i;
                                is_found = true;
                                break;
                            }
                        }
                        if(!is_found){
                            block_thread_ptr += 4;
                        }
                    }
                    branch_sum_before_block_thread_ptr_shared[cur_r] = branches_temp;
    
                    //Advance the dx position if needed (different block_thread location)
                    if (block_thread_ptr == initial_block_thread_ptr){
                        //Start at the same location
                        branches_temp = branch_sum_before_dx_ptr;
                    }
                    else{
                        dx_ptr = block_thread_ptr * MAX_SPACE_PER_THREAD;
                    }
    
                    //Find the correct dx position
                    is_found = false;
                    while(!is_found){

                        cur_branch_size_reg[0] = cur_branch_size()[dx_ptr];
                        cur_branch_size_reg[1] = cur_branch_size()[dx_ptr+1];
                        cur_branch_size_reg[2] = cur_branch_size()[dx_ptr+2]; 
                        cur_branch_size_reg[3] = cur_branch_size()[dx_ptr+3];
                        
                        #pragma unroll
                        for (int i=0;i<4;i++){
                            //NOTE: no need to check out of bounds if correctly impleneted, it will be filtered out at block level..
                            branches_temp += cur_branch_size_reg[i];
                            if (cur_branch_size_reg[i] != 0 && cur_thread_id < branches_temp){
                                branches_temp -= cur_branch_size_reg[i];
                                dx_ptr += i;
                                is_found = true;
                                break;
                            }
                        }
                        if(!is_found){
                            dx_ptr += 4;
                        }
                    }
                    branch_sum_before_dx_ptr = branches_temp;
                }
                else{
                    //Nothing here
                }
            }
    
            finfinddx:
            ;
    
            if (dx_ptr < MAX_PATH_PER_ROUND){ //If dx_ptr is within dx_num, [0-N)
    
                has_operation = true;
                
                float prob_thread = 1.0;
                int divide_factor = 1;
                unsigned int diff_freq_index; //0-16 only
                unsigned int remaining_value = cur_thread_id - branch_sum_before_dx_ptr ; //7^8 is less than 32 bit...
    
                unsigned char* cur_dx_temp = cur_dx() + ( 16 * dx_ptr ) ; //NOTE: Need to modify to fit datastruct of different cipher
                int* cur_sbox_index_ptr = cur_sbox_index() + (MAX_AS  * dx_ptr);
                unsigned char cur_thread_partial_dy[17];
                cur_thread_partial_dy[16] = {0};
                memcpy(cur_thread_partial_dy,cur_dx_temp,16);
                int cur_sbox_index_temp[MAX_AS];
                memcpy(cur_sbox_index_temp, cur_sbox_index_ptr, sizeof(int) * MAX_AS);
                //Points to correct i_th branches of j_dx and so subs
                #pragma unroll
                for (int i = 0; i < MAX_AS; i++) {
                    unsigned char cur_val = cur_thread_partial_dy[cur_sbox_index_temp[i]];
                    diff_freq_index = (remaining_value / divide_factor) % SPN_DIFF::diff_table_size_shared[cur_val]; 
                    cur_thread_partial_dy[cur_sbox_index_temp[i]] = SPN_DIFF::diff_table_shared[cur_val][diff_freq_index]; //Assigning target val to partial_dy
                    prob_thread *= (SPN_DIFF::prob_table_shared[cur_val][diff_freq_index]);
                    divide_factor *= SPN_DIFF::diff_table_size_shared[cur_val];
                }
                prob_thread *= (*(cur_prob() + dx_ptr));
    
                if (cur_r+1 != MAX_ROUND_BACKWARD){
                    //Do Permutate
                    unsigned long long front_64 = 0;
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        if ( cur_thread_partial_dy[i] > 0) {
                            //Permutation LUTable
                            front_64 |= SPN_DIFF::perm_lookup_global_reversed[i][cur_thread_partial_dy[i]];
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        cur_thread_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;  
                    }
   
                    //Calculte sbox index and sbox number
                    int save_sbox_num = 0;
                    int save_branch_size = 1;
                    int save_sbox_index[16]; //Will point to non existance 32 array entry (see substitution below)
                    #pragma unroll
                    for (int i=0;i< 16;i++){
                        save_sbox_index[i] = 16; 
                    }
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        if ((cur_thread_partial_dy[i] & 0xf) > 0) {
                            save_branch_size *= SPN_DIFF::diff_table_size_shared[cur_thread_partial_dy[i]];
                            save_sbox_index[save_sbox_num] = i;
                            save_sbox_num++;
                        }
                    }
    
                    //Pruning
                    // if(true){
                    if (save_sbox_num <= MAX_AS){  //If only next round AS <= 8
                        //MATSUI BOUND
                        float estimated_com_prob = SPN_DIFF::prob_per_round_remaining_shared[(MAX_ROUND_BACKWARD - cur_r - 2)] * SPN_DIFF::prob_per_as_shared[save_sbox_num-1]; //NOTE: this bound is less tight when round entered is not zero..
                        if ((estimated_com_prob * prob_thread) >= SPN_DIFF::CLUSTER_PROB_BOUND_const) {
                            memcpy(output_dx(increment),cur_thread_partial_dy,16);
                            *output_prob(increment) = prob_thread;
                            memcpy(output_sbox_index(increment), save_sbox_index, sizeof(int) * MAX_AS );
                            *output_branch_size(increment) = save_branch_size;
    
                            thread_branch_num_so_far += save_branch_size;
                            increment += 1; 
                            
                        } 
                        // else{ *output_branch_size = 0;} 
                    }
                    // else{ *output_branch_size = 0;}
                }
                //LAST ROUNDS... no permutation and straight to savings.
                else{
                    int sbox_num=0;
                    int sbox_index[16]={0};
                    #pragma unroll
                    for (int i=0;i<16;i++){
                        if (cur_thread_partial_dy[i] !=0){
                            sbox_index[sbox_num] = i;
                            sbox_num+=1;
                        }
                    }
        
                    if (sbox_num <=3){ //Possible to store three only...
                        //Computing appropriate index
                        int index=0;
                        #pragma unroll
                        for (int i=0;i<sbox_num;i++){
                            index|= ( ( (sbox_index[i]&0b11111) | ( (cur_thread_partial_dy[sbox_index[i]]&0b1111) << 5) ) << (i * 9) ); 
                        }
        
                        unsigned long long target_size =  MITM_size_interm_global[index];
                        if(target_size > 0){ //Exist connection
                            double target_prob = ( (1.0 * prob_thread) * MITM_prob_interm_global[index]);

                            //DEBUG: enable back
                            //Add to collection
                            device_prob_final[thread_id_default] += target_prob;
                            device_cluster_size_final[thread_id_default] += target_size;
                        }
                    }
                }
            }
            cur_thread_id+=1;
        }
        cur_thread_id-=1;
        // cur_iter = cur_iter > 0?cur_iter-loop_limit:cur_iter; //Make sure >0 => -1, if 0 left it
        cur_iter = cur_iter - loop_limit;

        if(thread_id_default == 0){ 
            *(device_has_operation + (flip_0_1) ) = has_operation;
        }
        if (cur_r != MAX_ROUND_BACKWARD-1){
            *output_branch_size_thread() += thread_branch_num_so_far; // so_far will be reset each sync thus adding like this is correct.
            atomicAdd(&SPN_DIFF::branch_size_block_shared[0], thread_branch_num_so_far);
            __syncthreads();
            if (threadIdx.x==0){
                // iter_shared[cur_r] = cur_iter;
                *output_branch_size_block() += SPN_DIFF::branch_size_block_shared[0];
                //Since the operation is once per output round, reset the stuff here
                atomicAdd( (output_branch_size_all()+!flip_0_1), SPN_DIFF::branch_size_block_shared[0]);

                SPN_DIFF::branch_size_block_shared[0] = 0;
            }
        }

        grid.sync(); //Wait for grid to synchronize before continue
        if (thread_id_default==0){
            output_branch_size_all()[flip_0_1] = 0;
        }
        grid.sync();

        has_operation = *(device_has_operation + (flip_0_1) );
        flip_0_1 = !flip_0_1;

        if(true){
            if (cur_r != MAX_ROUND_BACKWARD-1 && has_operation){ //Is not last round and has operation
                //Goes forwards
                iter_shared[cur_r] = cur_iter;
                dx_ptr_shared[cur_r] = dx_ptr;
                branch_sum_before_dx_ptr_shared[cur_r] = branch_sum_before_dx_ptr;
                thread_id_arr[cur_r] = cur_thread_id;
                cur_r+=1;

                // cur_thread_id = thread_id_default; //NOTE: does not matter
                dx_ptr = 0;
                branch_sum_before_dx_ptr = 0;
                branch_sum_before_block_thread_ptr_shared[cur_r] = 0;
                branch_sum_before_block_ptr_shared[cur_r] = 0;
                cur_iter = -1; //Signal the requirement of intiialization

                if (cur_r!=MAX_ROUND_BACKWARD-1){
                    *output_branch_size_thread() = 0; //HAs to be reset because of tunneling
                    if (threadIdx.x==0){
                        *output_branch_size_block()  = 0;
                    }
                }
            }
            else if(!has_operation || (cur_r == MAX_ROUND_BACKWARD-1 && cur_iter == 0) ){ //Has no operation => cur_iter == 0, 
                //Goes backwards if last rounds or current rounds does not process anythings.
                do{
                    cur_r-=1;
                    if(cur_r == -1){
                        return; //NOTE: Completed computation, Base Case
                    }
                    cur_iter = iter_shared[cur_r];
                }while(cur_iter==0);

                cur_thread_id = thread_id_arr[cur_r] + 1;
                dx_ptr = dx_ptr_shared[cur_r];
                branch_sum_before_dx_ptr = branch_sum_before_dx_ptr_shared[cur_r];

                *output_branch_size_thread() = 0;
                if (threadIdx.x==0){
                    *output_branch_size_block()  = 0;
                }
            }
            else{ //Has operation and is last round and cur_iter != 0 
                //Repeat last rounds.
                cur_thread_id += 1;
            }
        }
    }
};

//Kernel Launch preparation from here
//NOTE: Branch Size is assumed to be zero...
void GPU_Kenerl_t::kernel_compute(int branch_size, unsigned char* dx, unsigned char* dy, int* sbox_index, int* sbox_num, int* nb_size, float* cur_prob, int cur_r, int target_round){
    cudaError_t cudaStatus;
    if (branch_size >1){
        printf("\nInitial DX > 1 size is not supported..");
        return;
    }

    cudaStatus = cudaMemcpyToSymbol(SPN_DIFF::final_dy_constant, dy, sizeof(unsigned char)*16);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol final_dy_constant failed!");
    }

    cudaStatus = cudaMemcpy(device_dx, dx, sizeof(unsigned char) * 16 * branch_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dx failed!");
    }
    cudaStatus = cudaMemcpy(device_sbox_index, sbox_index, sizeof(int) * MAX_AS * branch_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy sbox_index failed!");
    }
    // cudaStatus = cudaMemcpy(device_sbox_num, sbox_num, sizeof(int) * branch_size, cudaMemcpyHostToDevice);
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaMemcpy sbox_num failed!");
    // }
    cudaStatus = cudaMemcpy(device_prob, cur_prob, sizeof(float) * branch_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy cur_prob failed!");
    }
    cudaStatus = cudaMemcpy(device_branch_size, nb_size, sizeof(int) * branch_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy nb_size failed!");
    }

    unsigned long long* nb_size_longlong = new unsigned long long();
    *nb_size_longlong = *nb_size;
    cudaStatus = cudaMemcpy(device_branch_size_block, nb_size_longlong, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy nb_size failed!");
    }

    cudaStatus = cudaMemcpy(device_branch_size_thread, nb_size_longlong, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy cur_prob failed!");
    }
    unsigned long long* device_branch_size_block2 = (device_branch_size_block+2);
    cudaStatus = cudaMemcpy(device_branch_size_block2, nb_size_longlong, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy nb_size failed!");
    }

    // Starting Kernel
    // int *round_to_process = new int();
    // *round_to_process = target_round - cur_r;
    int *round_offset = new int();
    *round_offset = cur_r;
    // int *branch_size_ptr = new int();
    // *branch_size_ptr = branch_size;
    // int round_to_process = target_round - cur_r;
    // int round_offset = cur_r;

    bool is_MITM =true;
    if (is_MITM){
        void** args = new void*[14];
        args[0] = &device_dx;
        args[1] = &device_sbox_index;
        args[2] = &device_prob;
        args[3] = &device_branch_size;

        args[4] = &device_cluster_size_final;
        args[5] = &device_prob_final;

        args[6] = &device_last_dx_ptr;
        args[7] = &device_has_operation;
        args[8] = &device_branches_sum_before_dx;
        
        args[9] = &device_branch_size_thread;
        args[10] = &device_branch_size_block2;
        args[11] = &device_total_branch_size_block;

        args[12] = &MITM_prob_interm_global;
        args[13] = &MITM_size_interm_global;
    
        dim3 dimGrid(BLOCK_NUM, 1, 1);
        dim3 dimBlock(THREAD_PER_BLOCK, 1, 1);

        std::cout << "\nTransfered constant matsui bound from host to device";
        cudaStatus = cudaMemcpyToSymbol(SPN_DIFF::CLUSTER_PROB_BOUND_const, &CLUSTER_PROB_BOUND_FORWARD, sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyToSymbol CLUSTER_PROB_BOUND_FORWARD failed!");
            getchar();
            exit(-1);
        }
    
        cudaStatus = cudaLaunchCooperativeKernel((void*) kernel_diff_mitm, dimGrid, dimBlock, args);
        if (cudaStatus != cudaSuccess) {
            cudaError_t err = cudaGetLastError();
            fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
            std::cout << "\nExiting the program manually...";
            getchar();
            exit(-1);
        }

        // cudaStatus = cudaDeviceSynchronize();
        // if (cudaStatus != cudaSuccess) {
        //     cudaError_t err = cudaGetLastError();
        //     fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
        //     std::cout << "\nExiting the program manually...";
        //     getchar();
        //     exit(-1);
        // }    

        //Backwards
        cudaStatus = cudaMemset(device_branch_size_block+1, 0, sizeof(unsigned long long));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemset device_branch_size_block+1 during backwards failed!");
        }
        cudaStatus = cudaMemset(device_branch_size_block2+BLOCK_NUM, 0, sizeof(unsigned long long)*BLOCK_NUM );
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemset device_branch_size_block2 during backwards failed!");
        }
        cudaStatus = cudaMemset(device_branch_size_thread + GRID_THREAD_SIZE, 0, sizeof(unsigned long long)*GRID_THREAD_SIZE);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemset device_branch_size_thread during backwards failed!");
        }

        //Transfer DY part
        std::cout << "\nTransfered constant matsui bound from host to device";
        cudaStatus = cudaMemcpyToSymbol(SPN_DIFF::CLUSTER_PROB_BOUND_const, &CLUSTER_PROB_BOUND_BACKWARD, sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyToSymbol CLUSTER_PROB_BOUND_BACKWARD failed!");
            getchar();
            exit(-1);
        }

        cudaStatus = cudaMemcpy(device_dx, dy, sizeof(unsigned char) * 16 * branch_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy dx failed!");
        }
        *nb_size = 1;
        int temp_index_ptr = 0;
        for (int i=0;i<16;i++){
            sbox_index[i] = 16; 
        }
        for (int i=0;i<16;i++){
            // sbox_index[i] = dy[i] > 0? i : 0;
            if(dy[i]>0){
                sbox_index[temp_index_ptr] = i; 
                *(nb_size) = *(nb_size) * (SPN_DIFF::diff_table_size_host_reversed[dy[i]]);
                temp_index_ptr+=1;
            }
        }
        *cur_prob = 1.0f;

        cudaStatus = cudaMemcpy(device_sbox_index, sbox_index, sizeof(int) * MAX_AS * branch_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy sbox_index failed!");
        }
        cudaStatus = cudaMemcpy(device_prob, cur_prob, sizeof(float) * branch_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy cur_prob failed!");
        }
        cudaStatus = cudaMemcpy(device_branch_size, nb_size, sizeof(int) * branch_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy nb_size failed!");
        }
    
        *nb_size_longlong = *nb_size; //Because nb_size is int, cast to long long in this case.
        cudaStatus = cudaMemcpy(device_branch_size_block, nb_size_longlong, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy nb_size failed!");
        }
        cudaStatus = cudaMemcpy(device_branch_size_thread, nb_size_longlong, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy cur_prob failed!");
        }
        // unsigned long long* device_branch_size_block2 = (device_branch_size_block+2);
        cudaStatus = cudaMemcpy(device_branch_size_block2, nb_size_longlong, sizeof(unsigned long long), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy nb_size failed!");
        }
    
        cudaStatus = cudaLaunchCooperativeKernel((void*) kernel_diff_mitm_backward, dimGrid, dimBlock, args);
        if (cudaStatus != cudaSuccess) {
            cudaError_t err = cudaGetLastError();
            fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
            std::cout << "\nExiting the program manually...";
            getchar();
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            cudaError_t err = cudaGetLastError();
            fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
            std::cout << "\nExiting the program manually...";
            getchar();
            exit(-1);
        }    

        delete args;
    }
    else{
        void** args = new void*[12];
        // args[0] = &round_offset;
        // args[1] = &round_to_process;
        // args[2] = &branch_size;
        // int* device_sbox_index2 = device_sbox_index+1;
        // bool* device_has_operation2 = device_has_operation + 2;

        // args[0] = round_offset;
        // args[0] = round_to_process;
        // args[2] = branch_size_ptr;

        args[0] = &device_dx;
        args[1] = &device_sbox_index;
        args[2] = &device_prob;
        args[3] = &device_branch_size;

        args[4] = &device_cluster_size_final;
        args[5] = &device_prob_final;

        args[6] = &device_last_dx_ptr;
        args[7] = &device_has_operation;
        args[8] = &device_branches_sum_before_dx;
        
        args[9] = &device_branch_size_thread;
        args[10] = &device_branch_size_block2;
        args[11] = &device_total_branch_size_block;

        dim3 dimGrid(BLOCK_NUM, 1, 1);
        dim3 dimBlock(THREAD_PER_BLOCK, 1, 1);

        std::cout << "\nTransfered constant matsui bound from host to device";
        cudaStatus = cudaMemcpyToSymbol(SPN_DIFF::CLUSTER_PROB_BOUND_const, &CLUSTER_PROB_BOUND, sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyToSymbol CLUSTER_PROB_BOUND_const failed!");
            getchar();
            exit(-1);
        }
    
        cudaStatus = cudaLaunchCooperativeKernel((void*) kernel_diff, dimGrid, dimBlock, args);
        if (cudaStatus != cudaSuccess) {
            cudaError_t err = cudaGetLastError();
            fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
            std::cout << "\nExiting the program manually...";
            getchar();
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            cudaError_t err = cudaGetLastError();
            fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
            std::cout << "\nExiting the program manually...";
            getchar();
            exit(-1);
        }
    

        delete args;
    }
    //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);
    // initialize, then launch
    // cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount*numBlocksPerSm, numThreads, args);
    // dim3 dimGrid(numSms * numBlocksPerSm, 1, 1);

    // cudaLaunchCooperativeKernel(
    //     const T *func,
    //     dim3 gridDim,
    //     dim3 blockDim,
    //     void **args,
    //     size_t sharedMem = 0,
    //     cudaStream_t stream = 0
    // )       
}

void GPU_Kenerl_t::kernel_reduction(double& gpu_prob, long long& gpu_size){
    long long size_arr[GRID_THREAD_SIZE];
    double prob_arr[GRID_THREAD_SIZE];

    const int size = GRID_THREAD_SIZE;
    auto cudaStatus = cudaMemcpy(size_arr, device_cluster_size_final, sizeof(unsigned long long)* size, cudaMemcpyDeviceToHost);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (device_cluster_size_final) failed!");\
            getchar();
        }
    #endif

    cudaStatus = cudaMemcpy(prob_arr, device_prob_final, sizeof(double)* size, cudaMemcpyDeviceToHost);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (device_prob_final) failed!");\
            getchar();
        }
    #endif
    
    printf("\nBefore Reduction \t GPU_Cluster_size : %lld\t GPU_Prob : %f",gpu_size, gpu_prob);
    for (int i=0;i< size; i++ ){
        gpu_size += size_arr[i];
        gpu_prob += prob_arr[i];
    }
    printf("\nAfter Reduction \t GPU_Cluster_size : %lld\t GPU_Prob : %f",gpu_size, gpu_prob);

}

//Called Once (1) @ program entry.
void SPN_DIFF::init(){
    int pi=0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "\nGPU Info :";
    std::cout << "\nSM numbers: " << deviceProp.multiProcessorCount;
    cudaDeviceGetAttribute(&pi, cudaDevAttrCooperativeLaunch, 0);
    std::cout << "\nSupport Cooperative Groups (Grid): " << (pi==1? " True":" FALSE");
    
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel_diff, THREAD_PER_BLOCK, 0);
    std::cout << "\nMax Blocks Per SM : " << numBlocksPerSm;
    if (deviceProp.multiProcessorCount != 9){
        std::cout << "\nPress enter key to conitnue";
        std::cout << "\n----------\n";
        getchar();
    }
    if  (pi==0){
        std::cout << "\nCooperative Groups not supported on target GPU";
        exit(-1);
    }

    // std::cout <<"\nInit Trifle Reverse Differential Table:{\n";

    // std::cout <<"\nPRESENT Permutation:{\n";
    for (int i = 0; i < 64; i++) {
		// if (i%16==0){
		// 	std:: cerr <<"\n";
		// }
        SPN_DIFF::perm_host[i] = (i / 4) + ((i % 4) * 16);
        // std::cout << (int) SPN_DIFF::perm_host[i]<< ",";
    }
    // std::cout << "\n}\n" ;

    // std::cout <<"\nPresent Permutation Reversed:{\n";
    for (int i=0;i<64;i++){
        SPN_DIFF::perm_host_reversed[SPN_DIFF::perm_host[i]] = i;
    }
    // for (int i=0;i<64;i++){
    //     std::cout << (int) SPN_DIFF::perm_host_reversed[i]<< ",";
    // }
    // std::cout << "}\n" ;

    //--
    // std::cout <<"\n4bit Permutation LUTable * 32 (Size is 32*16*16 is 8192Bytes) :{\n";
    for (int sbox_pos=0;sbox_pos<16;sbox_pos++){
        for (int sbox_val=0;sbox_val<16;sbox_val++){
            unsigned char dx[16] = {0};
            dx[sbox_pos] = sbox_val;

            //Do permutation
            unsigned long long front_64 = 0, front_64_reversed=0;
			for (int i = 0; i < 16; i++) {
				if (dx[i] > 0) {
					for (int j = 0; j < 4; j++) {
                        //Actually filtered_bit
						unsigned long long filtered_word = ((dx[i] & (0x1 << j)) >> j) & 0x1;
						if (filtered_word == 0) continue; //no point continue if zero, go to next elements

                        int bit_pos = (SPN_DIFF::perm_host[((15 - i) * 4) + j]);
                        int bit_pos_reversed = (SPN_DIFF::perm_host_reversed[((15 - i) * 4) + j]);

						front_64 |= (filtered_word << bit_pos);
						front_64_reversed |= (filtered_word << bit_pos_reversed);
					}
				}
			}
            
            //Front 64, 0-15, Back64 - 16-31
            SPN_DIFF::perm_lookup_host[sbox_pos][sbox_val]=front_64;

            SPN_DIFF::perm_lookup_host_reversed[sbox_pos][sbox_val]=front_64_reversed;
        }
    }
    // std::cout << "}\n" ;
};