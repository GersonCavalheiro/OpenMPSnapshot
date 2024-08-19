#define _GNU_SOURCE
#include <errno.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
struct cpuinfo {
int cpu;
int core;
int socket;
};
int main(int argc, char **argv)
{
int opt, movement = 0;
unsigned long int sleep = 100;
while((opt = getopt(argc, argv, "s:mh")) != -1) {
switch (opt)
{
case 's':
errno = 0;
sleep = strtoul(optarg, NULL, 10);
if(errno) {
perror("strtoul");
exit(EXIT_FAILURE);
}
break;
case 'm':
movement = 1;
break;
default:
printf(
"\n=====================================================\n"
"omp thread distribution and thread migration checker\n"
"=====================================================\n"
"options:\n"
"-m watch thread migration\n"
"-s <sleep> time in us\n\n"
"hints:\n"
"get initial configuration:     ./foo | sort -n -k 2\n"
"check for thread migration:    ./foo -m | grep <socket|core|cpu>    (ctrl+c to cancel)\n\n"
"=====================================================\n"
"useful omp environment variables:\n"
"=====================================================\n"
"OMP_NUM_THREADS=<n>\n"
"OMP_PLACES={sockets|cores|threads|...}\n"
"OMP_PROC_BIND={false|true|master|close|spread}\n"
"OMP_DISPLAY_AFFINITY={false|true}\n"
"OMP_DISPLAY_ENV={false|true}\n\n");
exit(EXIT_SUCCESS);
break;
}
}
char buf[BUFSIZ];
int num_cpus = 0;
struct cpuinfo *cpulist = NULL;
FILE *f = popen("lscpu -p", "r");
while( fgets(buf, BUFSIZ-1, f) ) {
if(buf[0] != '#') {
if( !(cpulist = realloc(cpulist, (num_cpus + 1) * sizeof(struct cpuinfo))) ) {
perror("realloc");
exit(EXIT_FAILURE);
}
if(sscanf(buf, "%d,%d,%d,%*s\n", 
&cpulist[num_cpus].cpu, 
&cpulist[num_cpus].core, 
&cpulist[num_cpus].socket) == 3) {
num_cpus++;
}
}
}
pclose(f);
#pragma omp parallel
{
int current_cpu;
int cpu = sched_getcpu();
printf( "thread %3d running on cpu %3d core %2d and socket %d\n", 
omp_get_thread_num(), 
cpulist[cpu].cpu, 
cpulist[cpu].core, 
cpulist[cpu].socket );
#pragma omp barrier
while (movement)
{
current_cpu = sched_getcpu();
if (cpulist[current_cpu].socket != cpulist[cpu].socket) {
printf( "thread %3d changed %6s from %3d to %3d\n", 
omp_get_thread_num(), 
"socket",
cpulist[cpu].socket, 
cpulist[current_cpu].socket );
cpu = current_cpu;
} 
else if (cpulist[current_cpu].core != cpulist[cpu].core) {
printf( "thread %3d changed %6s from %3d to %3d\n", 
omp_get_thread_num(), 
"core",
cpulist[cpu].core, 
cpulist[current_cpu].core );
cpu = current_cpu;
}
else if (cpulist[current_cpu].cpu != cpulist[cpu].cpu) {
printf( "thread %3d changed %6s from %3d to %3d\n", 
omp_get_thread_num(), 
"cpu",
cpulist[cpu].cpu, 
cpulist[current_cpu].cpu );
cpu = current_cpu;
}
usleep(sleep);
}     
}
free(cpulist);
exit(EXIT_SUCCESS);
}
