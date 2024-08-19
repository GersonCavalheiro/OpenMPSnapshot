#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 45 
#define M 15 
void fibonacci(int A[]){
int i;
A[0] = 0;
A[1] = 1;
for(i = 2; i < N; i++){
A[i] = A[i - 1] + A[i - 2];
}
}
void factorial(int A[]){
int i;
A[0] = A[1] = 1;
for(i = 2; i < M; i++){
A[i] = i * A[i - 1];
}
}
int main(){
unsigned int fib[N];
unsigned int fac[M];
int i;
#pragma omp parallel sections
{
#pragma omp section
{
fibonacci(fib);
}
#pragma omp section
{
factorial(fac);
}
}
printf("Fibonacci Series:\n");
for(i = 0; i < N; i++){
printf("%u ", fib[i]);
}
printf("\n");
printf("Factorial:\n");
for(i = 0; i < M; i++){
printf("%u ", fac[i]);
}
printf("\n");
return 0;
}
