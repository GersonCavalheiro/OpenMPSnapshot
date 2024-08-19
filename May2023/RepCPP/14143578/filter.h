#pragma once
#include "dg/functors.h"
#include "fast_interpolation.h"



namespace dg
{

namespace create
{


template<class UnaryOp, class real_type>
dg::Operator<real_type> modal_filter( UnaryOp op, const DLT<real_type>& dlt )
{
Operator<real_type> backward=dlt.backward();
Operator<real_type> forward=dlt.forward();
Operator<real_type> filter( dlt.n(), 0);
for( unsigned i=0; i<dlt.n(); i++)
filter(i,i) = op( i);
filter = backward*filter*forward;
return filter;
}

} 

namespace detail{

template<class real_type>
DG_DEVICE void pix_sort( real_type& a, real_type& b)
{
if( a > b) 
{
real_type tmp = a;
a = b;
b = tmp;
}
}

template<class real_type>
DG_DEVICE real_type median3( real_type* p)
{
pix_sort(p[0],p[1]) ; pix_sort(p[1],p[2]) ; pix_sort(p[0],p[1]) ;
return (p[1]) ;
}

template<class real_type>
DG_DEVICE real_type median5( real_type* p)
{
pix_sort(p[0],p[1]) ; pix_sort(p[3],p[4]) ; pix_sort(p[0],p[3]) ;
pix_sort(p[1],p[4]) ; pix_sort(p[1],p[2]) ; pix_sort(p[2],p[3]) ;
pix_sort(p[1],p[2]) ; return (p[2]) ;
}

template<class real_type>
DG_DEVICE real_type median9( real_type* p)
{
pix_sort(p[1], p[2]) ; pix_sort(p[4], p[5]) ; pix_sort(p[7], p[8]) ;
pix_sort(p[0], p[1]) ; pix_sort(p[3], p[4]) ; pix_sort(p[6], p[7]) ;
pix_sort(p[1], p[2]) ; pix_sort(p[4], p[5]) ; pix_sort(p[7], p[8]) ;
pix_sort(p[0], p[3]) ; pix_sort(p[5], p[8]) ; pix_sort(p[4], p[7]) ;
pix_sort(p[3], p[6]) ; pix_sort(p[1], p[4]) ; pix_sort(p[2], p[5]) ;
pix_sort(p[4], p[7]) ; pix_sort(p[4], p[2]) ; pix_sort(p[6], p[4]) ;
pix_sort(p[4], p[2]) ; return (p[4]) ;
}

template<class real_type, class Functor>
DG_DEVICE real_type median( unsigned i, const int* row_offsets,
const int* column_indices, Functor f, const real_type* x )
{
int n = row_offsets[i+1]-row_offsets[i];
if( n == 3)
{
real_type p[3];
int k = row_offsets[i];
for( int l = 0; l<3; l++)
p[l] =  f(x[column_indices[k+l]]);
return detail::median3( p);
}
if ( n == 5)
{
real_type p[5];
int k = row_offsets[i];
for( int l = 0; l<5; l++)
p[l] =  f(x[column_indices[k+l]]);
return detail::median5(p);
}
if( n == 9)
{
real_type p[9];
int k = row_offsets[i];
for( int l = 0; l<9; l++)
p[l] =  f(x[column_indices[k+l]]);
return detail::median9( p);

}
int less, greater, equal;
real_type  min, max, guess, maxltguess, mingtguess;

min = max = f(x[column_indices[row_offsets[i]]]) ;
for (int k=row_offsets[i]+1 ; k<row_offsets[i+1] ; k++) {
if (f(x[column_indices[k]])<min) min=f(x[column_indices[k]]);
if (f(x[column_indices[k]])>max) max=f(x[column_indices[k]]);
}

while (1) {
guess = (min+max)/2;
less = 0; greater = 0; equal = 0;
maxltguess = min ;
mingtguess = max ;
for (int k=row_offsets[i]; k<row_offsets[i+1]; k++) {
if (f(x[column_indices[k]])<guess) {
less++;
if (f(x[column_indices[k]])>maxltguess)
maxltguess = f(x[column_indices[k]]) ;
} else if (f(x[column_indices[k]])>guess) {
greater++;
if (f(x[column_indices[k]])<mingtguess)
mingtguess = f(x[column_indices[k]]) ;
} else equal++;
}
if (less <= (n+1)/2 && greater <= (n+1)/2) break ;
else if (less>greater) max = maxltguess ;
else min = mingtguess;
}
if (less >= (n+1)/2) return maxltguess;
else if (less+equal >= (n+1)/2) return guess;
else return mingtguess;
}

}


struct CSRMedianFilter
{
template<class real_type>
DG_DEVICE
void operator()( unsigned i, const int* row_offsets,
const int* column_indices, const real_type* values,
const real_type* x, real_type* y)
{
y[i] = detail::median( i, row_offsets, column_indices, []DG_DEVICE(double x){return x;}, x);
}
};



template<class real_type>
struct CSRSWMFilter
{
CSRSWMFilter( real_type alpha) : m_alpha( alpha) {}
DG_DEVICE
void operator()( unsigned i, const int* row_offsets,
const int* column_indices, const real_type* values,
const real_type* x, real_type* y)
{
real_type median = detail::median( i, row_offsets, column_indices,
[]DG_DEVICE(double x){return x;}, x);
real_type amd = detail::median( i, row_offsets, column_indices,
[median]DG_DEVICE(double x){return fabs(x-median);}, x);

if( fabs( x[i] - median) > m_alpha*amd)
{
y[i] = median;
}
else
y[i] = x[i];
}
private:
real_type m_alpha ;
};


struct CSRAverageFilter
{
template<class real_type>
DG_DEVICE
void operator()( unsigned i, const int* row_offsets,
const int* column_indices, const real_type* values,
const real_type* x, real_type* y)
{
y[i] = 0;
int n = row_offsets[i+1]-row_offsets[i];
for( int k=row_offsets[i]; k<row_offsets[i+1]; k++)
y[i] += x[column_indices[k]]/(real_type)n;
}
};

struct CSRSymvFilter
{
template<class real_type>
DG_DEVICE
void operator()( unsigned i, const int* row_offsets,
const int* column_indices, const real_type* values,
const real_type* x, real_type* y)
{
y[i] = 0;
for( int k=row_offsets[i]; k<row_offsets[i+1]; k++)
y[i] += x[column_indices[k]]*values[k];
}
};


template<class real_type>
struct CSRSlopeLimiter
{
CSRSlopeLimiter( real_type mod = (real_type)0) :
m_mod(mod) {}
DG_DEVICE
void operator()( unsigned i, const int* row_offsets,
const int* column_indices, const real_type* values,
const real_type* x, real_type* y)
{
int k = row_offsets[i];
int n = (row_offsets[i+1] - row_offsets[i])/3;
if( n == 0) 
return;
for( int u=0; u<n; u++)
y[column_indices[k+1*n+u]] = x[column_indices[k+1*n + u]]; 
real_type uM = 0, u0 = 0, uP = 0, u1 = 0;
for( int u=0; u<n; u++)
{
uM += x[column_indices[k+0*n + u]]*fabs(values[k+u]);
u0 += x[column_indices[k+1*n + u]]*fabs(values[k+u]);
u1 += x[column_indices[k+1*n + u]]*values[k+n+u];
uP += x[column_indices[k+2*n + u]]*fabs(values[k+u]);
}
if( values[k]<0) 
uM *= -1;
if( values[k+2*n]>0) 
uP *= -1;

dg::MinMod minmod;
if( fabs( u1) <= m_mod)
return;
real_type m = minmod( u1, uP - u0, u0 - uM);
if( m == u1)
return;
for( int u=0; u<n; u++)
y[column_indices[k+1*n+u]] =
values[k+2*n]>0 ? u0 - m*values[k+2*n+u] : u0 + m*values[k+2*n+u];
}
private:
real_type m_mod;
};


}
