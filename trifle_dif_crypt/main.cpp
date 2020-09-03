#include <iostream>
#include <chrono>
#include <thread>
#include <math.h>
#include "common.h"

#include "spn_diff.h"

int main(){
    //Init Var
    spn_diff_init();
    double double_after_reduction=0;
    long long cluster_num_gpu_after_reduction=0;
    std::chrono::steady_clock::time_point end_before_reduction = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    //MAIN Functions
    testing_gpu_kernel(double_after_reduction, cluster_num_gpu_after_reduction, end_before_reduction);

    //Called main programed. Printing output
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\n\nCluster Probabilities: log2: " << std::dec << log2(double_after_reduction) << " , base_10 : " << double_after_reduction;
	std::cout << "\nNumber of Cluster Trails:" << cluster_num_gpu_after_reduction;

    std::cout << "\nTime difference (s ) = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count();
	std::cout << "\nTime difference (us) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "\nTime difference (us) discount reduction = " << std::chrono::duration_cast<std::chrono::microseconds>( (end_before_reduction) - begin).count();
	std::cout << "\nTime difference (ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();
    std::cout <<std::endl;

    // std::cout << "\n\nPress enter to continue";
    // getchar();

    return 0;
}