

#pragma once

#include <cmath>

namespace advscicomp {

double sin_plus_cos(double x)
{
return sin(x) + cos(x);
}



struct sc_f 
{
double operator() (double x) const
{
return sin(x) + cos(x);
}
};

class psc_f 
{
public: 
psc_f(double alpha) : alpha(alpha) {}
double operator() (double x) const
{
return sin(alpha * x) + cos(x);
}
private:
double alpha; 
};



template< typename F, typename T>
T inline fin_diff( F f, const T& x, const T& h)
{
return (f(x+h) - f(x)) / h;
}


template <typename F, typename T>
class derivative 
{
public: 
derivative(const F& f, const T& h) : f(f), h(h) {}

T operator() (const T& x) const
{
return ( f(x+h) - f(x)) / h;
}
private:
const F& f;
T	h;
};  





template <typename F, typename T, unsigned N>
class nth_derivative
{
using prev_derivative = nth_derivative<F, T, N-1>;
public:
nth_derivative(const F& f, const T& h) : h(h), fp(f, h) {}

T operator() (const T& x) const
{
return (fp(x+h) - fp(x)) / h;
}
private:
T	h;
prev_derivative fp;
}; 



template <typename F, typename T>
class nth_derivative<F, T, 1>
{
public:
nth_derivative(const F& f, const T& h) : h(h), f(f) {}

T operator() (const T& x) const
{
return (f(x+h) - f(x)) / h;
}
private:
T		h;
const F& 	f;
}; 



template <unsigned N, typename F, typename T>
nth_derivative<F, T, N>  
make_nth_derivative(const F& f, const T& h)
{
return nth_derivative<F, T, N>(f, h);
}





}


