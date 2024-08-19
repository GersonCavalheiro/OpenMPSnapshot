

#include <iostream>  
#include <omp.h> 
#include <sys/time.h>
#include <stdlib.h>

using namespace std;

#define Max 1024

int A[Max*Max] = {0};  
int B[Max*Max] = {0};
int C[Max*Max] = {0};

void matrix_Init()
{
#pragma omp parallel for num_threads(64)
for(int i = 0 ; i < Max*Max ; i++ ) {
srand((unsigned)time(NULL));
A[i] = rand() % 1000;    
B[i] = rand() % 1000;
}
}




void matrix_Multi()
{
#pragma omp parallel for num_threads(64)

#pragma omp parallel shared(A,B,C)private(i,j,k)
{
#pragma omp for schedule(dynamic)
for(int i = 0 ; i < Max ; i++){
for(int j = 0; j < Max ; j++){
int c_index = i*Max+j;
int a_index = i*Max+0;
int b_index = 0*Max+j;
C[c_index] = 0;
for(int k =0; k < Max; k++){
C[c_index] += A[a_index]*B[b_index];
a_index += 1;
b_index += Max;
}                    
}
}
}    
}

int main()  
{ 
float time_use = 0;
struct timeval start;
struct timeval end;
matrix_Init();

gettimeofday(&start, NULL);

matrix_Multi();

gettimeofday(&end, NULL);
time_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
cout << "time_use is "<< time_use/1000000 << endl;
return 0;  
}