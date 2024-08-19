#include <iostream>	
#include <cstdio>	
#include <cstdlib>	
#include <cmath>	
#include <chrono>	
#include <omp.h>	
#include <algorithm>
#include <array>
#include <sys/time.h>
using namespace std;
using namespace std::chrono;
#define PI 3.14159
int nsteps = 100;                   	
int tpoints = 100;                  	
double values[102];             		
double oldval[102];                   
double newval[102];                   
double rc = 0;								
double function(double x){
double r;
r = sin(2.0 * PI * x);
return r;
}
double wave_function(int i){
const double dtime = 0.3;
const double c = 1.0;
const double dx = 1.0;
double tau, sqtau;
double rc;
tau = (c * dtime / dx);
sqtau = tau * tau;
rc = (2.0 * values[i]) - oldval[i] + (sqtau * (values[i - 1] - (2.0 * values[i]) + values[i + 1]));
return rc;
}
int main() {
auto start = high_resolution_clock::now();
for (int i = 1; i <= tpoints+1; i++) {
int t = omp_get_thread_num();
double x = ((double)(i % 10) / (double)(tpoints - 1));
values[i] = function(x);
oldval[i] = values[i];
if(i < 10 || i > 90){
printf("thread#= %d  newval[%d]= %lf\n", t, i, values[i]);
}
}
printf("\n*****\nNOW FOR THE UPDATES\n*****\n");
for (int i = 1; i < tpoints+1; i++) {
int t = omp_get_thread_num();
rc = wave_function(i);
newval[i]= rc;
oldval[i] = values[i];
values[i] = newval[i];
if(i < 10 || i > 90){
printf("thread#= %d  newval[%d]= %lf\n", t, i, values[i]);
}
}
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout << "Time taken using serial: " << duration.count() << " microseconds" << endl;
auto begin = high_resolution_clock::now();
int i;
#pragma omp parallel for  schedule(static) shared(values, oldval, newval, tpoints) private(i)
for (i = 0; i <= tpoints+1; i++) {
int t = omp_get_thread_num();
double x = ((double)(i % 10) / (double)(tpoints - 1));
values[i] = function(x);
oldval[i] = values[i];
if(i < 10 || i > 90){
printf("thread#= %d  newval[%d]= %.8lf\n", t, i, values[i]);
}
}
printf("\n*****\nNOW FOR THE UPDATES\n*****\n");
#pragma omp parallel for schedule(guided, 1)  shared(values, oldval, tpoints) private(newval, i, rc)
for (i = 1; i <= tpoints+1; i++) {
int t = omp_get_thread_num();
#pragma omp critical
{
rc = wave_function(i);
newval[i] = rc;
oldval[i] = values[i];
values[i] = newval[i];
}
if(i < 10 || i > 90){
printf("thread#= %d  newval[%d]= %.8lf\n", t, i, values[i]);
}
}
auto end = high_resolution_clock::now();
auto time = duration_cast<microseconds>(begin - end);
cout << "Time taken using openmp: " << duration.count() << " microseconds" << endl;
return 0;
}
