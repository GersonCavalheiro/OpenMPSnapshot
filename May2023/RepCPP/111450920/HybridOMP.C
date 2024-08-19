#include <stdio.h>
#include <omp.h>

#include "HybridOMP.H"

#ifdef __cplusplus
extern "C" {
#endif

int hyb_tid_to_device(int threadid)
{
int num_cpu_threads = omp_get_num_threads() - hyb_num_gpu_available();
return threadid - num_cpu_threads;
}

int hyb_tid_to_device_(int threadid)
{
int num_cpu_threads = omp_get_num_threads() - hyb_num_gpu_available();
return threadid - num_cpu_threads;
}

int hyb_num_gpu_available(void)
{
static int num_gpus = -1;
#pragma omp threadprivate(num_gpus)

if (num_gpus > -1)
return num_gpus;

num_gpus = 0;
int num_devices_visible = omp_get_num_devices();

for (int i=0; i<num_devices_visible; i++)
{
int A[1] = {-1};
#pragma omp target device(i)
{
A[0] = omp_is_initial_device();
}
if (A[0]==0)
num_gpus++;
else
printf("A[0]=%d\n",A[0]);
}

printf("HybridOMP: Able to use offloading: %s, #devices visible: %d #devices available: %d\n", num_gpus>0?"yes":"NO", num_devices_visible, num_gpus);
return num_gpus;
}

void hyb_set_devices(void)
{
int dev = hyb_tid_to_device(omp_get_thread_num());
if (dev>=0) {
omp_set_default_device(dev);
printf("HybridOMP: Setting default offloading device for thread %d to device %d.\n", omp_get_thread_num(), dev);
}

}



#ifdef __cplusplus
} 
#endif
