#include <stdio.h>
#define TAM 1000
int main(int argc, char *argv[]){
int i, n = TAM, a[TAM], b[TAM], c[TAM];
#pragma acc data copyin(a,b) copyout(c)
{
#pragma acc parallel loop
{
for (i=0; i < n; i++){
c[i] = a[i] * b[i];
}
}
}
}
