TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

# Choose cuda arch as you need
CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

echo "Compiling average distance loss by nvcc..."
cd src
nvcc -I $TF_INC --expt-relaxed-constexpr -c -o average_distance_loss_kernel.cu.o average_distance_loss_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH -std=c++11 && 
cd ../ &&
python build.py
