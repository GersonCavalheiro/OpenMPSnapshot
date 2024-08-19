#define _GNU_SOURCE
#include <sched.h>
#include"ConcurrentLinkedListMap.h"
#include"InputMethod.h"
#include<sys/time.h>
int main(int argc, char *argv[]){
srand48(time(NULL));
int cycle=3;
int t, c, f, i;
struct timeval start, end;
int num_threads=1;
int max_thread=24;
char *file_name[1]={"100000"};
for( f = 0; f < (sizeof(file_name)/sizeof(char*)); ++f){
char path[80]="../Data/";								
strcat(path,file_name[f]);
strcat(path,".txt");
input_array_t input_array = get_input_array(path);			
for ( t = 1; t <= max_thread; ++t){
if(t==2||t==4||t==8||t==12||t==24){
num_threads=t;
omp_set_num_threads(num_threads);						
double avg_time_spent = 0;
for( c = 0; c < cycle; ++c){
map_t *map;												
map=map_init(map);					
int node_inserted[num_threads];
for(i=0;i<num_threads;i++)
node_inserted[i]=0;
#pragma omp parallel firstprivate(i,input_array) shared(map,start,node_inserted)
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
node_inserted);
}
}
for(i=0;i<num_threads;i++)
map->size+=node_inserted[i];
gettimeofday(&end, NULL);
double time_spent = ((end.tv_sec  - start.tv_sec) * 1000000u + 
end.tv_usec - start.tv_usec) / 1.e6;
avg_time_spent+=time_spent;
free_map(map);
}
avg_time_spent=avg_time_spent/cycle;
printf("%8f(seg) AVG_Time of file %-7s with %3d threads.\n",avg_time_spent,file_name[f],num_threads);
}
}
free_input_array(&input_array);
}
return 0;
}
