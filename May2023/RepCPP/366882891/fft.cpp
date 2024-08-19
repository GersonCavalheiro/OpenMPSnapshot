#include <complex>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>

#define SIZE 1048576 * 8
#define THREADS 4

#define PI 3.1415926
#define BORDER 4
using namespace std;
typedef complex<double> C; 

C* FFT(C *Arr, int N, int level = 0)
{
if(N <= 1) return Arr;
C OmegaN;
C Omega;
OmegaN.real(cos(2.0 * (PI / N)));
OmegaN.imag(sin(2.0 * (PI / N)));    
Omega.real(1.0);
Omega.imag(0.0);
C *A0 = new C[N / 2];
C *A1 = new C[N / 2];
int k = 0;
int m = 0;    
for(int i = 0; i < N; i++) {
if(i % 2 == 0) {
A0[k] = Arr[i];
k++;
}
else {
A1[m] = Arr[i];
m++;
}
}

C *Y0;
C *Y1;

if(level < BORDER) {
#pragma omp task shared(Y0)
{	
Y0 = FFT(A0, N / 2, level + 1);
}
#pragma omp task shared(Y1) 
{
Y1 = FFT(A1, N / 2, level + 1);
}
#pragma omp taskwait
}
else {
Y0 = FFT(A0, N / 2, level + 1);
Y1 = FFT(A1, N / 2, level + 1);
}

C *Y = new C[N];

for(k = 0; k < N / 2; k++)
{
C temp;
temp = Y1[k] * Omega;
Y[k] = Y0[k] + temp;
Y[k + (N / 2)] = Y0[k] - temp;
Omega = Omega * OmegaN;
}
delete [] A0;
delete [] A1;
return Y;
}


int main() {

C *vin = new C[SIZE];
C *vout;
for(int i = 0; i < SIZE; ++i) {
vin[i].real((sin(i)));
vin[i].imag(0);
}



double time_d = 0;
double time_end = 0;
double time_start = 0;

std::cout << "Calculations begin, cores: " << THREADS << std::endl;
time_start = omp_get_wtime();

#pragma omp parallel num_threads(THREADS)
{
#pragma omp single
{
vout = FFT(vin, SIZE);
}
}

time_end = omp_get_wtime();
time_d = time_end - time_start;

std::ofstream fout;
std::string file_name = "fft.txt";
fout.open(file_name);

for(int i = 0; i < SIZE; ++i) {
double x = sqrt((vout[i].real() * vout[i].real()) + (vout[i].imag() * vout[i].imag()));
fout << x << " ";
}

delete [] vin;
delete [] vout;

std::cout << "Time: " << double(time_d) << " seconds" << std::endl;
}
