# Compile  
$ nvcc vector_cpu_add.cu -o vector_cpu_vector  
$ nvcc vector_gpu_add.cu -o vector_gpu_vector  
# Execute
$ time vector_cpu_add    
$ time vector_gpu_add  

# Execute result
$ time ./vector_cpu_add

real    0m0.214s
user    0m0.182s
sys     0m0.032s

$ time ./vector_gpu_add
out[0] = 3.000000
PASSED

real    0m1.505s
user    0m0.668s
sys     0m0.766s




