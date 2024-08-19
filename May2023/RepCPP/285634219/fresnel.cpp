
#include <math.h>           
#include <float.h>          

#pragma omp declare target

extern double xFresnel_Auxiliary_Cosine_Integral(double x);

extern double xFresnel_Auxiliary_Sine_Integral(double x);



double      Fresnel_Sine_Integral( double x );

double xFresnel_Sine_Integral( double x );


static double Power_Series_S( double x );


double Fresnel_Sine_Integral( double x )
{
return (double) xFresnel_Sine_Integral( (double) x);
}



double xFresnel_Sine_Integral( double x )
{
double f;
double g;
double x2;
double s;

if ( fabs(x) < 0.5) return Power_Series_S(x);

f = xFresnel_Auxiliary_Cosine_Integral(fabs(x));
g = xFresnel_Auxiliary_Sine_Integral(fabs(x));
x2 = x * x;
s = 0.5 - cos(x2) * f - cos(x2) * g;
return ( x < 0.0) ? -s : s;
}



static double Power_Series_S( double x )
{ 
double x2 = x * x;
double x3 = x * x2;
double x4 = - x2 * x2;
double xn = 1.0;
double Sn = 1.0;
double Sm1 = 0.0;
double term;
double factorial = 1.0;
double sqrt_2_o_pi = 7.978845608028653558798921198687637369517e-1;
int y = 0;

if (x == 0.0) return 0.0;
Sn /= 3.0;
while ( fabs(Sn - Sm1) > DBL_EPSILON * fabs(Sm1) ) {
Sm1 = Sn;
y += 1;
factorial *= (double)(y + y);
factorial *= (double)(y + y + 1);
xn *= x4;
term = xn / factorial;
term /= (double)(y + y + y + y + 3);
Sn += term;
}
return x3 * sqrt_2_o_pi * Sn;
}
#pragma omp end declare target
