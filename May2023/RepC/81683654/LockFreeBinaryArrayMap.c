#define _GNU_SOURCE
#include <sched.h>
#include"LockFreeBinaryArrayMap.h"
#include"InputMethod.h"
#include<sys/time.h>
int main(int argc, char *argv[]){
srand48(time(NULL));
int cycle=3;
int t, c, f, d, e, i;
struct timeval start, end;
int num_threads=1;
int max_thread=24;
char *file_name[1]={"2000000"};
int num_umbral[4];
int array_pointers_umbral[4];
float reduction=1.2;
for( f = 0; f < (sizeof(file_name)/sizeof(char*)); ++f){
char path[80]="../Data/";								
strcat(path,file_name[f]);
strcat(path,".txt");
input_array_t input_array = get_input_array(path);			
for ( t = 1; t <= max_thread; ++t){
if(t==2||t==4||t==8||t==12||t==24){
num_threads=t;
omp_set_num_threads(num_threads);						
num_umbral[0] = input_array.num_input*0.0001;
num_umbral[1] = input_array.num_input*0.0002;
num_umbral[2] = input_array.num_input*0.001;
num_umbral[3] = input_array.num_input*0.002;
char thread_using[num_threads][189819];
int thread_ending[num_threads];
int node_inserted[num_threads];
for( d = 0; d < (sizeof(num_umbral)/sizeof(int)); ++d){
array_pointers_umbral[0] = 1;
array_pointers_umbral[1] = 2;
array_pointers_umbral[2] = 5;
array_pointers_umbral[3] = 10;
for(e = 0; e < (sizeof(array_pointers_umbral)/sizeof(int)); ++e){
double avg_time_spent = 0;
for( c = 0; c < cycle; ++c){
map_t *map;													
for(i=0;i<num_threads;i++){
CopyString(thread_using[i],"*");
thread_ending[i]=0;
node_inserted[i]=0;
}
if(num_umbral[d]<num_threads)
break;
else
map=map_init(map,num_umbral[d]);
Array_Pointers_t *array_pointers;
array_pointers=(Array_Pointers_t *)malloc(sizeof(Array_Pointers_t));
array_pointers->umbral=array_pointers_umbral[e];
int size_array_pointers=input_array.num_input/array_pointers->umbral;
array_pointers->node=(node_t **)malloc(sizeof(node_t*)*size_array_pointers);
array_pointers->size=size_array_pointers;
array_pointers->size_instant=0;
int prev_map_size=0;
#pragma omp parallel firstprivate(i,input_array,reduction) shared(map, start, prev_map_size, thread_ending, thread_using, node_inserted, array_pointers)
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
thread_using,
&prev_map_size,
node_inserted,
array_pointers);
}
}
for(i=0;i<num_threads;i++)
map->size+=node_inserted[i];
gettimeofday(&end, NULL);
double time_spent = ((end.tv_sec  - start.tv_sec) * 1000000u + 
end.tv_usec - start.tv_usec) / 1.e6;
avg_time_spent+=time_spent;
free_map(map);
free_array_pointers(array_pointers);
}
avg_time_spent=avg_time_spent/cycle;
fflush(stdout);
printf("%8f(seg) AVG_Time of file %-7s with %3d threads and map->umbral = %-7d, array->umbral = %-7d;\n",avg_time_spent,file_name[f],num_threads,num_umbral[d],array_pointers_umbral[e]);
fflush(stdout);
}
}
}
}
free_input_array(&input_array);
}
return 0;
}