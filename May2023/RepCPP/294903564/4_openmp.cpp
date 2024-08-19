#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include <omp.h>

double f (double x, double y)
{
return x*x + (y-2) * (y-2);
}

double find_min (double a_1, double a_2, double b, double K, double (*g)(double, double))
{
double min_f, now_f;
double x,y;
double a_h = (a_2-a_1);
double b_h = (b - a_1);
min_f = f(a_1,a_1);
for(int i = 0; i < K; i ++){
x = a_1 + (double)rand() / (double)RAND_MAX * a_h;
y = a_1 + (double)rand() / (double)RAND_MAX * b_h;
now_f = f(x, y);
if (now_f < min_f) {
min_f = now_f;
}
}
return min_f;
}

int main(){
double A, B;
int K;

A = 0.0;
B = 4.0;
K = 100000000;
double s = 0.; 

int NUM_THREADS = 8;
double H = (B-A)/NUM_THREADS;

double r[NUM_THREADS];

#pragma omp parallel for num_threads(1)
for(int i = 0; i < NUM_THREADS; i ++){
r[i] = find_min(A + i*H,A + (i+1)*H, B,K,f);
}

double min_f = r[0];
for(int i = 0; i < NUM_THREADS; i ++){
if (r[i] < min_f){
min_f = r[i];
}
}
printf("%lf\n", min_f);
}