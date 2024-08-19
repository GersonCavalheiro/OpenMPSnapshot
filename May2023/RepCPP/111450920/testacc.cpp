#include <stdio.h>
#include <chrono>
#include <stdlib.h>

using namespace std;

extern "C" void zaxpy_(long *start, long *end, long *len, double *X, double *Y, double *Z);


int main(int argc, char const* argv[])
{
double *X, *Y, *Z;
long len = atoi(argv[1]);
int i;

X = (double *) malloc(len * sizeof(double));
Y = (double *) malloc(len * sizeof(double));
Z = (double *) malloc(len * sizeof(double));

#pragma acc kernels
{

}

for (i=0; i<len; i++) {
X[i] = 1.5;
Y[i] = 2.3;
Z[i] = 0.0;
}
long len2 = len/1.1;
long one = 1;
chrono::steady_clock::time_point begin, end;

#pragma acc data copy(X, Y, Z, zero, len, len2)
{
{
begin = chrono::steady_clock::now();
#pragma acc kernels
zaxpy_(&one, &len2, &len, X, Y, Z);
end = chrono::steady_clock::now();
}
}

printf("%ld %.1f %.1f %ld microseconds\n", len, Z[4], Z[len-4], chrono::duration_cast<chrono::microseconds>(end - begin).count());
return 0;
}
