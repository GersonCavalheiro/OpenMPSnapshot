#pragma omp declare target

extern "C"
double xChebyshev_Tn_Series(double x, double a[], int degree)
{
double yp2 = 0.0;
double yp1 = 0.0;
double y = 0.0;
double two_x = x + x;
int k;


if ( degree < 0 ) return 0.0;


for (k = degree; k >= 1; k--, yp2 = yp1, yp1 = y) 
y = two_x * yp1 - yp2 + a[k];


return x * yp1 - yp2 + a[0];
}
#pragma omp end declare target
