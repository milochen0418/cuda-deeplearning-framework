# Compile  
$ nvcc vector_cpu_add.cu -o vector_cpu_vector  
$ nvcc vector_gpu_add.cu -o vector_gpu_vector  
# Execute
$ time vector_cpu_add    
$ time vector_gpu_add  
