#include <stdio.h>
#include <omp.h>
int main(void) {
int isHost = -1;
#pragma omp target map(from:isHost)
{ isHost = omp_is_initial_device(); }
if (isHost < 0) {
printf("Runtime error, isHost=%d\n", isHost);
}
printf("Target region executed on the %s\n", isHost ? "host" : "device");
return isHost;
}
