#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

float f1(float x, int intensity);
float f2(float x, int intensity);
float f3(float x, int intensity);
float f4(float x, int intensity);

#ifdef __cplusplus
}
#endif

using namespace std;

float parallel_integrate (int argc, char* argv[]);

int main (int argc, char* argv[]) {
#pragma omp parallel
{
int fd = open (argv[0], O_RDONLY);
if (fd != -1) {
close (fd);
}
else {
std::cerr<<"something is amiss"<<std::endl;
}
}

if (argc < 9) {
std::cerr<<"Usage: "<<argv[0]<<" <functionid> <a> <b> <n> <intensity> <nbthreads> <scheduling> <granularity>"<<std::endl;
return -1;
}



float integrate = parallel_integrate(argc, argv);


std::cout<<integrate<<std::endl;

return 0;
}


float parallel_integrate (int argc, char* argv[]){

int function = atoi(argv[1]);
float a = atof(argv[2]); 
float b = atof(argv[3]);
int n = atoi(argv[4]);
long intensity = atol(argv[5]); 
int granularity = atoi(argv[8]);
int thread_count = atoi(argv[6]);
string sync = argv[7];
float sum=0.0;

float temp,func_param;
temp = (b-a)/n;


omp_set_num_threads(thread_count);
std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
switch(function){
case 1:
#pragma omp parallel for schedule(dynamic,granularity) reduction(+:sum)     
for(int i=0;i<n;i++){
func_param = a+((i+0.5)*temp); 
sum += (f1(func_param,intensity)*temp); 
}  
break; 
case 2:
#pragma omp parallel for schedule(dynamic,granularity) reduction(+:sum)     
for(int i=0;i<n;i++){
func_param = a+((i+0.5)*temp); 
sum += (f2(func_param,intensity)*temp); 
}  
break; 
case 3:
#pragma omp parallel for schedule(dynamic,granularity) reduction(+:sum)     
for(int i=0;i<n;i++){
func_param = a+((i+0.5)*temp); 
sum += (f3(func_param,intensity)*temp); 
}  
break;   
default:
#pragma omp parallel for schedule(dynamic,granularity) reduction(+:sum)     
for(int i=0;i<n;i++){
func_param = a+((i+0.5)*temp); 
sum += (f4(func_param,intensity)*temp);  
} 
break;
} 

std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

std::chrono::duration<double> elapsed_seconds = end-start;

std::cerr<<elapsed_seconds.count()<<std::endl;

return sum;
}