#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<time.h>
#include<string.h>
#include<stdlib.h>
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define CLK CLOCK_MONOTONIC
struct timespec diff(struct timespec start, struct timespec end);
struct timespec diff(struct timespec start, struct timespec end){
struct timespec temp;
if((end.tv_nsec-start.tv_nsec)<0){
temp.tv_sec = end.tv_sec-start.tv_sec-1;
temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
}
else{
temp.tv_sec = end.tv_sec-start.tv_sec;
temp.tv_nsec = end.tv_nsec-start.tv_nsec;
}
return temp;
}
int main(int argc, char* argv[])
{
struct timespec start_e2e, end_e2e, start_alg, end_alg, e2e, alg;
clock_gettime(CLK, &start_e2e);
if(argc < 3){
printf( "Usage: %s n p \n", argv[0] );
return -1;
}
int N=atoi(argv[1]);	
int P=atoi(argv[2]);	
char *problem_name = "matrix_multiplication";
char *approach_name = "block";
FILE* outputFile;
char outputFileName[50];		
sprintf(outputFileName,"output/%s_%s_%s_%s_output.txt",problem_name,approach_name,argv[1],argv[2]);	
/
for(k=0; k<N; k++) {
#pragma omp parallel for schedule(static,N/P) private(i)
for(i=0; i<N; i++) {
x[i] = dis[k][i];
y[i] = dis[i][k];
}
#pragma omp parallel for schedule(static,N/P) private(i,j)
for(i=0; i<N; i++) {
if(i == k) continue;
for(j=0; j<N; j++) {
if(dis[i][j] > y[i]+x[j]) {
dis[i][j] = y[i]+x[j];
}
}
}	
}
clock_gettime(CLK, &end_alg);	
clock_gettime(CLK, &end_e2e);
e2e = diff(start_e2e, end_e2e);
alg = diff(start_alg, end_alg);
printf("%s,%s,%d,%d,%d,%ld,%d,%ld\n", problem_name, approach_name, N, P, e2e.tv_sec, e2e.tv_nsec, alg.tv_sec, alg.tv_nsec);
return 0;
}