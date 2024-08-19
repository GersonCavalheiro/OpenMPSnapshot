#pragma once
#include <boost/math/special_functions.hpp>
namespace dg {
namespace mat {

template < class T = double>
struct BESSELI0
{
BESSELI0( ) {}

T operator() ( T x) const
{
return boost::math::cyl_bessel_i(0, x);
}
};

template < class T = double>
struct GAMMA0
{
GAMMA0( ) {}

T operator() ( T x) const
{
return exp(x)*boost::math::cyl_bessel_i(0, x);
}
};



template<class T>
T phi1( T x){
if ( fabs( x) < 1e-16 )
return 1.;
if ( fabs(x) < 1)
return expm1( x)/x;
return (exp( x) - 1)/x;
}


template<class T>
T phi2( T x){
if ( fabs( x) < 1e-16 )
return 1./2;
if ( fabs(x) < 1e-2)
return 1./2.+x*(1./6.+x*(1./24. + x/120.));
return ((exp( x) - 1)/x - 1)/x;
}


template<class T>
T phi3( T x){
if ( fabs( x) < 1e-16 )
return 1./6.;
if ( fabs(x) < 1e-2)
return 1./6. + x*(1./24.+x*(1./120. +x/720.));
return (((exp( x) - 1)/x - 1)/x-1./2.)/x;
}

template<class T>
T phi4( T x){
if ( fabs( x) < 1e-16 )
return 1./24.;
if ( fabs(x) < 1e-2)
return 1./24. + x*(1./120.+x*(1./720. + x/5040));
return ((((exp( x) - 1)/x - 1)/x-1./2.)/x-1./6.)/x;
}

}
}
