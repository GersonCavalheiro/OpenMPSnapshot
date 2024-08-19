#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define B 50
#define K 7

struct timeval start, end;
double timeclock;
int i,j,k;

void BacaFile(FILE *file, char* matrix[B][K]){
gettimeofday(&start, NULL);
printf("Reading File... \n");
char word[20];
if (!file){
printf("File tidak ditemukan"); 
}

#pragma omp parallel num_threads(7)
{
#pragma omp single nowait
{
if(!feof(file)){ 
#pragma omp task
{
for(i = 0; i < B; i++){
for(j = 0; j < K-1; j++){
fscanf(file,"%s",word); 
matrix[i][j]=strdup(word); 
}
}
}
}
}
}
gettimeofday(&end, NULL);
timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
printf("Reading File Success \n");printf("Clock Time = %g \n",timeclock);
printf("\n \n");
}

void ParseAndSave(char* dataX[B][K],float parse[B][K-2]){
gettimeofday(&start, NULL);
printf("Parsing Data from String to Number...\n");

#pragma omp parallel num_threads(7)
{
#pragma omp for schedule(static,2)
for(i=0;i<B;i++){
parse[i][0]=atof(dataX[i][0]);
parse[i][1]=atof(dataX[i][3]);
parse[i][2]=atof(dataX[i][4]);
parse[i][2]=(2016-parse[i][2])*365;
parse[i][3]=atof(dataX[i][5]);
}
}

gettimeofday(&end, NULL);
timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
printf("Parsing Complete \n");printf("Clock Time = %g \n",timeclock);
printf("\n \n");
}

void Predict(float parse[B][K-2]){
gettimeofday(&start, NULL);
printf("Calculating data to Predict Nuclear Waste Mass\n"); j=4;

#pragma omp parallel num_threads(7)
{
#pragma omp for schedule(static,2)
for(i=0;i<B;i++){
parse[i][j]=parse[i][3]*(pow(0.5,(parse[i][2]/parse[i][1])));
}
}

gettimeofday(&end, NULL);
timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
printf("Calculation Complete! \n");printf("Clock Time = %g \n",timeclock);
printf("\n \n");
}

void FindNotZero(float parse[B][K-2],char* dataX[B][K]){
gettimeofday(&start, NULL); int th_id;
printf("Print Ex-Nuclear Plant that still have Nuclear Waste \n");

#pragma omp parallel num_threads(7)
{
#pragma omp for schedule(static,2)
for(i=0;i<B;i++){
if(parse[i][4]!=0){
th_id=omp_get_thread_num();
printf(" Thread Id: %d ",th_id);
printf(" Nomor: %s\n Nama Pabrik: %s\n Bentuk Limbah: %s\tWaktu Paruh Isotop: %s Hari\t Tahun ditutup: %s\t Massa Limbah pada Tahun Tersebut: %s Pound\n Massa Limbah pada saat ini: %10.60f Pound\n \n",dataX[i][0],dataX[i][1],dataX[i][2],dataX[i][3],dataX[i][4],dataX[i][5],parse[i][4]);
}
}
}

gettimeofday(&end, NULL);
timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
printf("Clock Time = %g \n",timeclock);
}

int main(){
FILE *baca;
char* data[B][K]; 
float parsed[B][K-2];
float wp,cy,massa; 

gettimeofday(&start, NULL);

baca=fopen("database 5000.txt","r");

BacaFile(baca,data);
fclose(baca);

ParseAndSave(data,parsed);

Predict(parsed);

FindNotZero(parsed,data);

gettimeofday(&end, NULL);
timeclock = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
printf("\nClock Time of Entire Process = %g \n",timeclock);
return 0;
}
