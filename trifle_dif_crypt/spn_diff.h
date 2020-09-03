#ifndef SPN_DIFF_GUARD
#define SPN_DIFF_GUARD 

#include <chrono>

//Expected Communication of Task receiving in another class..

//Init GPU and CPU processing...
void spn_diff_init();

//CPU only counterpart
void spn_diffcluster_cpu(double& cluster_prob, long long& cluster_size);
void MITM_diffcluster_cpu(double& cluster_prob, long long& cluster_size);

//GPU pure dynamic  
void spn_diffcluster_gpu(double& cluster_prob, long long& cluster_size, unsigned char* dx, unsigned char* dy);
void MITM_diffcluster_gpu(double& cluster_prob, long long& cluster_size);

//Testing Unit
void testing_gpu_kernel(double& cluster_prob, long long& cluster_size, std::chrono::steady_clock::time_point &end_b4_reduc);

#endif

