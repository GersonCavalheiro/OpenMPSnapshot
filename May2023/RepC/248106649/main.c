#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
int main(int argc, char **argv) {
if(argc!=7) {
perror("wrong arguments");
exit(1);
}
int64_t a=atoi(argv[1]);
int64_t b=atoi(argv[2]);
int64_t x=atoi(argv[3]);
int64_t N=atoi(argv[4]);
double p=atof(argv[5]);
short P=atoi(argv[6]);
size_t count;
double timet;
double c;
double T=0;
size_t tcount=0, ccount=0;
unsigned int seed;
time_t rawtime;
FILE *file=NULL;
if(x<a && x>b && a>b) {
perror("wrong x, a, b");
exit(2);
}
time(&rawtime);
seed=(unsigned int)rawtime;
srand(seed);
omp_set_dynamic(0);
omp_set_num_threads(P);
#pragma omp parallel 
{
long ns;
time_t s;
struct timespec spec;
clock_gettime(CLOCK_REALTIME, &spec);
ns=spec.tv_nsec; 
s=spec.tv_sec;
if(ns>999999999) {
s++;
ns=0;
}
double time_spent=0;
#pragma omp for 
for(int i=0;i<N;i++){
x=atoi(argv[3]);
count=0;
while(a<x && x<b){	
seed++;
double c=(rand_r(&seed)%1000)/1000.0;
if(c<=p) {
x++;
count++;
}
else if(c>p) {
x--;
count++;
}
}
if(x==b) {	
ccount++;
}
tcount+=count; 
clock_gettime(CLOCK_REALTIME, &spec);
time_t s_end=spec.tv_sec;
long ns_end=spec.tv_nsec; 
if (ns_end-ns<0) {
s_end--;
ns_end+=1000000000;
}
time_spent=s_end-s+(ns_end-ns)/1000000000.;
T+=time_spent;
}
}
c=ccount/(double)N;
timet=tcount/(double)N;
x=atoi(argv[3]);
file=fopen("stats.txt", "w");
if (file==NULL){ 
perror("fopen");
exit(3);
}
fprintf(file,"%f\t %f\t %f\t %ld\t %ld\t %ld\t %ld\t %f\t %d\n",c,timet,T,a,b,x,N,p,P);
fclose(file);
return 0;
}