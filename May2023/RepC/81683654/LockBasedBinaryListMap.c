#define _GNU_SOURCE
#include <sched.h>
#include"LockBasedBinaryListMap.h"
#include"InputMethod.h"
#include<sys/time.h>
int main(int argc, char *argv[]){
srand48(time(NULL));
int cycle=3;
int t, c, f, d, i;
struct timeval start, end;
int num_threads=1;
int max_thread=24;
char *file_name[1]={"2000000"};
int num_umbral[4];
float reduction=1.2;
for( f = 0; f < (sizeof(file_name)/sizeof(char*)); ++f){
char path[80]="../Data/";								
strcat(path,file_name[f]);
strcat(path,".txt");
input_array_t input_array = get_input_array(path);			
for ( t = 1; t <= max_thread; ++t){
if(t==2||t==4||t==8||t==12||t==24){
num_threads=t;
num_umbral[0] = input_array.num_input*0.0001;
num_umbral[1] = input_array.num_input*0.0002;
num_umbral[2] = input_array.num_input*0.001;
num_umbral[3] = input_array.num_input*0.002;
for( d = 0; d < (sizeof(num_umbral)/sizeof(int)); ++d){
omp_set_num_threads(num_threads);						
double avg_time_spent = 0;
for( c = 0; c < cycle; ++c){
map_t *map;													
int thread_ending[num_threads];
for(i=0;i<num_threads;i++)
thread_ending[i]=0;
if(num_umbral[d]<num_threads*2){
printf("Umbral should be greater than num_threads\n");
break;
}
else
map=map_init(map,num_umbral[d]);	
int prev_map_size=0;
#pragma omp parallel firstprivate(i,input_array,reduction) shared(map, start, prev_map_size, thread_ending)
{
cpu_set_t mask;
CPU_ZERO(&mask);
CPU_SET (omp_get_thread_num(), &mask);
if( sched_setaffinity(0, sizeof(cpu_set_t), &mask) == -1)
printf("Error cannot do sched_setaffinityn\n");
#pragma omp critical
{
gettimeofday(&start, NULL);
}
#pragma omp barrier
for(i=omp_get_thread_num();i<input_array.num_input;i+=omp_get_num_threads()){
map_insert(
map,
input_array.input[i].key,
input_array.input[i].value,
input_array.input[i].ranking_1,
input_array.input[i].ranking_2,
input_array.input[i].frequency,
reduction,
input_array.num_input,
i,
thread_ending,
&prev_map_size
);
}
}
gettimeofday(&end, NULL);
double time_spent = ((end.tv_sec  - start.tv_sec) * 1000000u + 
end.tv_usec - start.tv_usec) / 1.e6;
avg_time_spent+=time_spent;
free_map(map);
}
avg_time_spent=avg_time_spent/cycle;
fflush(stdout);
printf("%8f(seg) AVG_Time of file %-7s with %3d threads and umbral = %-7d;\n",avg_time_spent,file_name[f],num_threads,num_umbral[d]);
fflush(stdout);
}
}
}
free_input_array(&input_array);
}
return 0;
}