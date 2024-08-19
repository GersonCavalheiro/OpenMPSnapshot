#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif


double f(long int i, long int N2) {
long double partial = (i - 0.5);
return  partial * partial / (N2);
}


int main(int argc, char *argv[]) {
unsigned long long N = 1000000000;
unsigned long long N2 = N*N;

printf("N: %llu - N^2: %llu\n", N, N2);
long double sum = .0;


for(unsigned long long i=1; i<=N; i++) {
sum += 4 / ( 1 + f(i, N2) );
}

sum = sum / N;
printf("%.20LF\n", sum);
}
