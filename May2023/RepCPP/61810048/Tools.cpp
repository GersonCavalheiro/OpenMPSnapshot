
#include "Tools.h"

void Tools::distributeArray(
int nb_chunks,
int nb_elems,
int *imin_table,
int *length_table )
{

if( nb_chunks >= nb_elems ) {
#pragma omp simd
for( int chunk = 0 ; chunk < nb_elems ; chunk ++ ) {
imin_table[chunk] = chunk;
length_table[chunk] = 1;
}
#pragma omp simd
for( int chunk = nb_elems ; chunk < nb_chunks ; chunk ++ ) {
imin_table[chunk] = nb_elems;
length_table[chunk] = 0;
}
} else {

int quotient;
int remainder;

quotient = nb_elems/nb_chunks;

remainder = nb_elems%nb_chunks;

#pragma omp simd
for( int chunk = 0 ; chunk < remainder ; chunk ++ ) {
imin_table[chunk] =  chunk*quotient+chunk;
length_table[chunk] = quotient + 1;
}
#pragma omp simd
for( int chunk = remainder ; chunk < nb_chunks ; chunk ++ ) {
imin_table[chunk] = remainder + chunk*quotient;
length_table[chunk] = quotient;
}
}
}


void Tools::GaussLegendreQuadrature( double x_min, double x_max, double *roots,
double *weights, int number_of_roots, double eps )
{

if( number_of_roots <= 0 ) {
ERROR("Gauss-Legendre quadrature: Number of iteration <= 1");
}
if( x_max < x_min ) {
ERROR("Gauss-Legendre quadrature: x_max < x_min");
}
if( eps <= 0 ) {
ERROR("Gauss-Legendre quadrature: accuracy threshold epsilon <= 0");
}

double P_prev;
double P_curr;
double P_next;

double P_derivative;

double root;
double root_prev;

int half_number_of_roots=( number_of_roots+1 )/2;
double x_average=0.5*( x_min+x_max );
double x_half_length=0.5*( x_max-x_min );

for( int i_root=0; i_root<=half_number_of_roots-1; i_root++ ) { 

root=cos( M_PI*( i_root+1.0-0.25 )/( number_of_roots+0.5 ) );


do {

P_next=root;
P_curr=1.0;

for( int order=2; order<=number_of_roots; order++ ) {

P_prev=P_curr;
P_curr=P_next;

P_next=( ( 2.0*order-1.0 )*root*P_curr-( order-1.0 )*P_prev )/order;
}

P_derivative=number_of_roots*( root*P_next-P_curr )/( root*root-1.0 );

root_prev=root;

root=root_prev-P_next/P_derivative;

} while( fabs( root-root_prev ) > eps );

roots[i_root]=x_average-x_half_length*root;
roots[number_of_roots-1-i_root]=x_average+x_half_length*root;

weights[i_root]=2.0*x_half_length/( ( 1.0-root*root )*P_derivative*P_derivative );
weights[number_of_roots-1-i_root]=weights[i_root];

}
}

bool Tools::fileCreated( const std::string &filename )
{
std::ifstream file( filename.c_str() );
return !file.fail();
}

double Tools::BesselK(double nu, double z)
{
double K;
if (z > 3000.0) {
K = Tools::asymptoticBesselK(nu, z);
} else {
K = boost::math::cyl_bessel_k( nu, z);
}
return K;
}

double Tools::asymptoticBesselK(double nu, double z)
{
double mu = 4 * nu*nu;
double iz = 1/z;
double D = iz*0.125;
double C = std::sqrt(M_PI*0.5*iz)*std::exp(-z);
double K;
K = 1 + (mu-1)*0.125*iz
+ (mu-1)*(mu-9)*0.5*std::pow(D,2)
+ (mu-1)*(mu-9)*(mu-25)*std::pow(D,3)/6.0;
K *= C;

return K;
}
