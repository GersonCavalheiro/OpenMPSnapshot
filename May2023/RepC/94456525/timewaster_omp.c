#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
double func(double x){
return sin(x)*exp(x)/(x + 1.);
}
double integ(double upper){
int i, n;
double lower, step, sum, result;
n = 10000; 
lower = 0.;
step  = (upper - lower)/n;
sum = func(lower) + func(upper);
for (i = 1; i < n; i++){
if (i % 3 == 0){
sum += 2.*func(lower + i*step);
}
else {
sum += 3.*func(lower + i*step);
}
}
return (3.*step/8.)*sum;
}
int main(int argc, char *argv[]){
clock_t tic, toc;
int i, iter;
double x, limit;
tic = clock();
if ( argc != 2 ) {
printf( "Usage: %s n \n", argv[0] );
exit(0);
} 
sscanf(argv[1], "%d", &iter);
srand(time(NULL));
#pragma omp parallel for private(i,limit,x)   
for (i = 0; i < iter; i++){
limit = rand() % 200;
x=integ(limit);
}
toc = clock();
printf("Elapsed time: %f seconds\n", (double)(toc - tic)/CLOCKS_PER_SEC);
return(0);
}
