#building
mode=0
for i in "$*"
do
   case $i in
	debug)	mode=1
		break
		;;
	default)
		mode=2
		break
		;;
	callgrind)	mode=3
						break
						;;
	gprof) mode=3
			break
			;;
   esac
done

case "$mode" in
	# --resource-usage
	# T4 -- sm_75
	# V100 -- sm_70
	# 1060 -- sm_61
	0) echo "Building Optimized RDC"
		nvcc -O3 -lineinfo -Xptxas -O3,-v --default-stream per-thread -arch=sm_61 -rdc=true spn_diff_kernel.cu common.cpp spn_diff.cpp main.cpp -o o.out
		;;
	1) echo "Building Debug RDC"
		nvcc -G -g  -arch=sm_61 -rdc=true spn_diff_kernel.cu common.cpp spn_diff.cpp main.cpp -o o.out
		;;
    2)
		echo "Building Optimized"
		nvcc -O3 -Xptxas -O3,-v --default-stream per-thread common.cpp spn_diff_kernel.cu spn_diff.cpp main.cpp -o o.out\
    	--generate-code arch=compute_61,code=sm_61 \
    	# --generate-code arch=compute_52,code=sm_52 \
		;;
	3)
		echo "Building Debug"
		#building -G device code, -g host code
		echo "Compiling Debug"
		nvcc -G -g common.cpp kernel_trifle.cu trifle.cpp main.cpp -o exec.out\
		--generate-code arch=compute_61,code=sm_61
		# --generate-code arch=compute_52,code=sm_52 \
		;;
	*)
		echo "Building Optmized with GProf (Not implemented)"
		echo "Compiling Callgrind (Performance)"
		echo "Compiling Optmized Code with Gprof enabled"
		nvcc -O3 -Xptxas -O3,-v --default-stream per-thread -pg common.cpp kernel_trifle.cu trifle.cpp main.cpp -o exec.out\
		--generate-code arch=compute_61,code=sm_61
		# --generate-code arch=compute_52,code=sm_52 \
		;;
esac
