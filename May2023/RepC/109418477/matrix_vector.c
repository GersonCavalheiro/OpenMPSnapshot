#include <stdio.h>
#include <math.h>
int main(){
const int size = 10000;
int i,j, chunk;
double A[size*size];
double x[size], b[size];
double temp = 0.0;
chunk =10;
#pragma omp parallel private(i,j) shared(A,x,b,chunk)
{
#pragma omp for schedule(dynamic, chunk)
for(j=0;j<size;j++){
for(i=0; i<size; i++){
A[i+size*j] = sin(0.01*(i+size*j));
}
b[j] = cos(0.01*j);
x[j] = 0.0;
}
}
#pragma omp parallel private(i,j) shared(A,x,b,chunk, temp)
{
#pragma omp for reduction(+:temp)
for(j=0;j<size;j++){
temp = 0.0; 
for(i=0;i<size;i++){
temp += A[i+size*j]*b[i];
}
x[j] = temp;
}
}
printf(" x[%d] = %g\n", 5050, x[5050]);
return 0;
}
