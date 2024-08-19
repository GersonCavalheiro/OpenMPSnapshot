#include <stdio.h>
#include <omp.h>
#define DELAY_VAL 10000000ULL 
int main(void) {
int isHost = 0;
clock_t ck_start = clock();
#pragma omp target map(from: isHost)
{ isHost = omp_is_initial_device(); 
for(long long int i=0;i<DELAY_VAL;i++);
}
if (isHost < 0) {
printf("Runtime error, isHost=%d\n", isHost);
}
printf ("Kernel: %ld clicks.\n", clock()-ck_start);
printf("Target region executed on the %s\n", isHost ? "host" : "device");
return isHost;
}
