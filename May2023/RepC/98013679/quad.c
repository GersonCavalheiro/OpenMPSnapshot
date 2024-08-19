





# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

#define MAX_NUM_THREADS 8
#define MAX 1024
#define ACCURACY 0.01

typedef struct Results Results;

struct Results {
double val;
double time;
};


inline double f(const double x) {
register const double pi = 3.141592653589793;
double value;
value = 50.0 / (pi * (2500.0 * x * x + 1.0));
return value;
}






void seqQuad(const unsigned n, const double a, const double b, double *total, double *execTime) {
unsigned i;
double total_q = 0.0;
double wtime;
double x;

wtime = omp_get_wtime();

for (i = 0; i < n; i++) {
x = ((double)(n - i - 1)*a + (double)(i)*b) / (double)(n - 1);
total_q = total_q + f(x);
}

wtime = omp_get_wtime() - wtime;

total_q = (b - a) * total_q / (double)n;

*total = total_q;
*execTime = (double)wtime;
}

Results sequential(const unsigned n, const double a, const double b) {
Results results;
seqQuad(n, a, b, &results.val, &results.time);
return results;
}






void parQuad(const unsigned n, const double a, const double b, double *total, double *execTime) {
int i;
double total_q = 0.0;
double wtime;
double x;

wtime = omp_get_wtime();

#pragma omp parallel for default(none) private(i, x) reduction(+:total_q) schedule(static, 100)
for (i = 0; i < n; i++) {
x = ((double)(n - i - 1)*a + (double)(i)*b) / (double)(n - 1);
total_q = total_q + f(x);
}

wtime = omp_get_wtime() - wtime;

total_q = (b - a) * total_q / (double)n;

*total = total_q;
*execTime = (double)wtime;
}

Results parallel(const unsigned n, const double a, const double b) {
Results results;
parQuad(n, a, b, &results.val, &results.time);
return results;
}


void compareAndPrint(const unsigned n, const double a, const double b) {
Results seq, par;

seq = sequential(n, a, b);
par = parallel(n, a, b);

printf("  Sequential estimate quadratic rule   = %24.16f\n", seq.val);
printf("  Parallel estimate quadratic rule     = %24.16f\n", par.val);
printf("Sequential time quadratic rule   = %f s\n", seq.time);
printf("Parallel time quadratic rule     = %f s\n", par.time);
if (fabs(seq.val - par.val) < ACCURACY)
printf("\tTest PASSED!\n");
else
printf("\a\tTest FAILED!!!\n");
printf("\n");
}


int main(int argc, char *argv[]) {
unsigned n;
double a;
double b;
const double exact = 0.49936338107645674464;

if (argc != 4) {
n = 10000000;
a = 0.0;
b = 10.0;
}
else {
n = (unsigned)atoi(argv[1]);
a = atof(argv[2]);
b = atof(argv[3]);
}

printf("\n");
printf("QUAD:\n");
printf("  Estimate the integral of f(x) from A to B.\n");
printf("  f(x) = 50 / ( pi * ( 2500 * x * x + 1 ) ).\n");
printf("\n");
printf("  A        = %f\n", a);
printf("  B        = %f\n", b);
printf("  N        = %u\n", n);
printf("\n");

compareAndPrint(n, a, b);

printf("  Normal end of execution.\n");
printf("\n");

getchar();
return 0;
}