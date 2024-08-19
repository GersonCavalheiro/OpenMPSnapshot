function m2c_barrier
% Perform barrier in a parallel mode. Within MATLAB, this is a no-op.
% For OpenMP, it uses the pragma omp barrier.
% For CUDA, it calls the CUDA intrinsic function __syncthreads. This implies 
% that all the threads will be assigned to the same block if m2c_barrier is
% used for a kernel function. Therefore, this function should be rarely used.
