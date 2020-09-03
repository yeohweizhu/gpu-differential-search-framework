#ifndef SPN_DIFF_KERNEL_GUARD
#define SPN_DIFF_KERNEL_GUARD 

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Cipher Customization 
#define MAX_SBOX 16 //32 x 4 = 128bit Sbox
#define BLOCK_SIZE_BIT (MAX_SBOX * 4)

#define THREAD_PER_BLOCK 128 //256 //128 //32
#define BLOCK_NUM 41 //GTX 1060 - 81, T4 - , P100 - , V100 -
#define GRID_THREAD_SIZE (THREAD_PER_BLOCK * BLOCK_NUM)
#define MAX_SPACE_PER_THREAD 64 //NOTE: THIS Need to be multiple of 4 because of indexing
#define MAX_PATH_PER_ROUND (GRID_THREAD_SIZE * MAX_SPACE_PER_THREAD)  // Number of path per round before pausing and continue to next rounds
//NOTE: Multiple of 4 is recommended for efficiency here

//MITM Criteria
#define MAX_ROUND_FORWARD 8 //For NON MITM, use MAX_ROUND_FORWARD
#define MAX_ROUND_BACKWARD 8
// #define MAX_ROUND (MAX_ROUND_FORWARD+MAX_ROUND_BACKWARD)

//Pruning Criteria3
#define MAX_AS 4
// Cluster Bound is optimized for TRIFLE
const float CLUSTER_PROB_BOUND = (pow(pow(2, -3), MAX_ROUND_FORWARD - 2) * pow(2, -2) * pow(2, -2) * pow(2, -62)); 
const float CLUSTER_PROB_BOUND_FORWARD = (pow(2, -62)); 
const float CLUSTER_PROB_BOUND_BACKWARD = (pow(2, -62)); 

const float CLUSTER_PROB_BOUND_LOG2 = log2(CLUSTER_PROB_BOUND);

//Pruning Criteria - Assumption
//Best Prob is assumed when calculating pruning criteria
#define CLUSTER_1AS_BEST_PROB 0.25f //4/16 if cipher have 8 then the value is 0.5f
#define CLUSTER_PROB_INDIV 0.25f 

namespace SPN_DIFF{
    void init();

    //These are for 4-bit SBOX
    /*
	* BC specific permutation and DTT
	*/
    //Contains configuration (macro / c++ global variable) intended to be used across different translation unit
    extern unsigned char perm_bit_host[BLOCK_SIZE_BIT]; 
    extern unsigned char perm_bit_host_reversed[BLOCK_SIZE_BIT]; 
    //Permutation is in bit

    //[0] front  ||   [1] back // 0 1 ... 32, in documentation this is invereted
    extern unsigned long long perm_lookup_host[MAX_SBOX][16];
    extern unsigned long long perm_lookup_host_reversed[MAX_SBOX][16];

    extern unsigned int diff_table_host[][8]; //There can ever be 8 maximum diff pattern for 4 bit sboxes
    extern unsigned int diff_table_host_reversed[][8];

    extern float prob_table_host[16][8]; // Basic refernce prob table (sorted)
    extern unsigned int freq_table_host[][8]; //Only used for sorting
    extern unsigned int diff_table_size_host[16]; //NB - Number of branches, Used for work acquisition

    extern unsigned char final_dy_host[16];
    extern unsigned char ref_dx_host[16];
};

struct GPU_Kenerl_t{
    //MITM Forward Output
    //Size of 3 Sbox with 32 position information
    //134217728
    static const int MITM_size = 134217728;
    float* MITM_prob_interm_global;
    unsigned long long* MITM_size_interm_global;

    //Input / Intermediate

    unsigned char* device_dx; // unsigned char*32 * MAX_instance_per_round * Rounds [0,n-1]
    int* device_sbox_index; //int * 8 (MAX_SBOX) * Rounds
    int* device_sbox_num; 
    float* device_prob;
    int* device_branch_size; //Computed from sbox NB by working thread.
    
    //Final Output (Need to be reduced)
    unsigned long long* device_cluster_size_final; //long long * thread num * thread block 
    double* device_prob_final;         //float * thread_num * thread_block

    //Intermediate sync variable
    int* device_last_dx_ptr; 
    bool* device_has_operation;
    unsigned long long* device_branches_sum_before_dx;
    unsigned long long* device_branch_size_thread;
    unsigned long long* device_branch_size_block;
    unsigned long long* device_total_branch_size_block;

    //CUDA Specific Stream Configuration
    cudaStream_t stream_obj;

    //Constructor
    GPU_Kenerl_t(int gpu_id, bool is_MITM_used=true);

    void kernel_reduction(double& gpu_prob, long long& gpu_size);

    //Kernel Compute Recursively (Simulated using cooperative groups)
    void kernel_compute(int branch_size, unsigned char* dx, unsigned char* dy, int* sbox_index, int* sbox_num, int* nb_size, float* cur_prob, int cur_r, int target_round);
};

#endif