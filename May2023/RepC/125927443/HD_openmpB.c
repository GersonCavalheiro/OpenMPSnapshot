#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#define CHUNK_SIZE 10
#define MASTER_THREAD 0
void PrintSomeMessage(char *p)
{
printf("p: %s\n",p);
}
int main(int argc,char **argv)
{
int m,n,I,nthreads;
if(argc<5)
{
printf("|Not enough arguments as input..\n");
exit(EXIT_FAILURE);
}
else if(argc>5)
{
printf("|Too many arguments...\n");
exit(EXIT_FAILURE);
}
else
{
m=atoi(argv[1]),n=atoi(argv[2]),I=atoi(argv[3]),nthreads=atoi(argv[4]);
int i=0,j=0,TotalHammingDistance=0;
char **setA = malloc(m * sizeof(char *)); 
for(i = 0; i < m; i++)
setA[i] = malloc((I+1) * sizeof(char));  
char **setB = malloc(n * sizeof(char *)); 
for(i = 0; i < n; i++)
setB[i] = malloc((I+1) * sizeof(char));  
for (i=0;i<m;i++){
for(j=0;j<I;j++){
setA[i][j]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[rand() % 62];
}
setA[i][I]='\0';
}
for (i=0;i<n;i++){
for(j=0;j<I;j++){
setB[i][j]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[rand() % 62];
}
setB[i][I]='\0';
}
printf("\n setA setB ready \n" );
int **HamDist = malloc(m * sizeof(int *)); 
for(i = 0; i < m; i++)
HamDist[i] = malloc(n * sizeof(int));
printf("\n HamDist set \n" );
clock_t start=clock();
omp_set_num_threads(nthreads);
int k=0,h=0;
for (k=0;k<m;k++){
for(h=0;h<n;h++){
int count=0;
#pragma omp parallel shared(setA,setB,HamDist) reduction(+:TotalHammingDistance)
{
uint l=0;
for(l=0;l<strlen(setB[h]);l++)
{
if(setA[i][l]!=setB[j][l]) {
count++;
}
}
HamDist[i][j]=count;
TotalHammingDistance+=HamDist[i][j];
}
}
}
clock_t end =clock();
double hamm_time=(double)(end-start)/CLOCKS_PER_SEC;
printf("\n|Total Hamming execution time= %f",hamm_time);
printf("\n\n*|The Total Hamming Distance is: %d\n",TotalHammingDistance );
}
return 0;
}
