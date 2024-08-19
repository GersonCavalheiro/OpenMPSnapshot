

#pragma once

#include <stdexcept>
#include <functional>
#include <sstream>


namespace advscicomp {


template<typename T, typename FuncT>
T NewtonIteration(T const& x, FuncT func, FuncT deriv)
{
return x - func(x)/deriv(x);
}


template<typename T, typename FuncT>
T NewtonsMethod(T const& x0, unsigned iterations, double tolerance, FuncT func, FuncT deriv)
{
using std::abs; 
T x_prev = x0;  
for (unsigned ii=0; ii<iterations; ++ii)
{
T x_next = NewtonIteration(x_prev, func, deriv); 

if ( abs(double(x_next - x_prev)) < tolerance)   
return x_next;

x_prev = x_next; 
}

std::stringstream err_msg;
err_msg << "too many iterations in newton's method, last approximation was " << x_prev;
throw std::runtime_error(err_msg.str());
}


}

