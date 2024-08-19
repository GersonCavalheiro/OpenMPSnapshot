#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define MAX_SIZE 1024
int arr[MAX_SIZE][MAX_SIZE];
int brr[MAX_SIZE][MAX_SIZE];
int crr[MAX_SIZE][MAX_SIZE];
int main()
{
int i, j;
int thread_num;
double start_time, end_time;
FILE *fp = fopen("matrix.txt","r");
printf("Input the number of thread\n");
scanf("%d",&thread_num);
for(i = 0; i < MAX_SIZE; i++){
for(j = 0; j < MAX_SIZE; j++){
fscanf(fp,"%d",&arr[i][j]);
}
}
for(i = 0; i < MAX_SIZE; i++){
for(j = 0; j < MAX_SIZE; j++){
fscanf(fp,"%d",&brr[i][j]);
}
}
fclose(fp);
start_time = omp_get_wtime();
omp_set_num_threads(thread_num);
#pragma omp parallel
{
int i, j, k;
int id = omp_get_thread_num();
int id_start, id_end;
int sum = 0;
id_start = (MAX_SIZE / thread_num) * id;
id_end = (MAX_SIZE / thread_num) * (id + 1);
if(id == thread_num - 1)id_end = MAX_SIZE;
for(i = id_start; i < id_end; i++){
for(j = 0; j < MAX_SIZE; j++){
sum = 0;
for(k = 0; k < MAX_SIZE; k++){
sum += (arr[i][k] * brr[k][j]);
}
crr[i][j] = sum;
}
}
}
end_time = omp_get_wtime();
fp = fopen("matrix_omp.txt","w");
for(i = 0; i < MAX_SIZE; i++){
for(j = 0; j < MAX_SIZE; j++){
fprintf(fp,"%d ",crr[i][j]);
}
fprintf(fp,"\n");
}
fclose(fp);
printf("Matrix Mulitplication OpenMp VER: Elapsed time is %f (sec)\n",end_time - start_time);
return 0;
}
