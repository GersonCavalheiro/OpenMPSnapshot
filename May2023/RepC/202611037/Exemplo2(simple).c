#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char *argv[]) {
int i, myid, n = 100000;
float a[n][4];
#pragma omp parallel default(none) private (i, myid)  shared(a,n)
myid = omp_get_thread_num(); 
for (i = 0; i < n; i++){
a[i][myid] = 1.0;
}
}
