
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

omp_lock_t lockA;
omp_lock_t lockB;
int A=1;
int B=4;

void proc_a() {
int x,y;
omp_set_lock(&lockA);
x = A;
x = x*2;
A = x;
omp_unset_lock(&lockA);  
omp_set_lock(&lockB);
y = B;
y -= x;
B = y;
omp_unset_lock(&lockB);
}

void proc_b() {
int x,y;
omp_set_lock(&lockB);
x = B;
x = x/2;
B = x;
omp_unset_lock(&lockB); 
omp_set_lock(&lockA);
y = A;
y += x;
A = y;
omp_unset_lock(&lockA);
}

int main(int argc, char** argv) {
omp_init_lock(&lockA);
omp_init_lock(&lockB);
#pragma omp parallel num_threads(8)
{
while(1) {
int x=rand();
printf("."); fflush(stdout);
if(x%2==0) {
proc_a();
} else {
proc_b();
}
}
}
return 0;
}

