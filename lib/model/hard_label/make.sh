
# Choose cuda arch as you need
CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

echo "Compiling hard label kernels by nvcc..."
cd src
nvcc --expt-relaxed-constexpr -c -o hard_label_kernel.cu.o hard_label_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH -std=c++11 && 
cd ../ &&
python build.py
