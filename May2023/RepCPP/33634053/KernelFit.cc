

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "KernelFit.hh"

template<class T>
KernelFit1D<T>::KernelFit1D(const std::vector<T> &x, const std::vector<T> &y,
const T &bandwidth){


if ( x.empty() || y.empty() )
throw KernelFitError("From KernelFit1D::KernelFit1D(), "
"one or both input vectors are empty!");

if ( x.size() != y.size() )
throw KernelFitError("From KernelFit1D::KernelFit1D(), input vectors "
"must be equal in length!");

if ( bandwidth <= 0.0 )
throw KernelFitError("From KernelFit1D::KernelFit1D(), the bandwidth "
"must be greater than zero!");

_x = x;
_y = y;
_b = bandwidth * bandwidth; 

}

template<class T>
std::vector<T> KernelFit1D<T>::Solve(const std::vector<T> &x){


if ( x.empty() )
throw KernelFitError("From KernelFit1D::Solve(), the input vector "
"cannot be empty!");

std::vector<T> f( x.size(), 0.0);

#pragma omp parallel for shared(f)
for (std::size_t i = 0; i <  x.size(); i++){

T sum = 0.0;

for (std::size_t j = 0; j < _x.size(); j++){

T W   = Kernel(_x[j] - x[i]);
f[i] += W * _y[j];
sum  += W;
}

f[i] /= sum;
}

return f;
}

template<class T>
std::vector<T> KernelFit1D<T>::Solve(const std::vector<T> &x, T (*W)(T)){


if ( x.empty() )
throw KernelFitError("From KernelFit1D::Solve(), the input vector "
"cannot be empty!");

std::vector<T> f( x.size(), 0.0);

#pragma omp parallel for shared(f)
for (std::size_t i = 0; i <  x.size(); i++){

T sum = 0.0;

for (std::size_t j = 0; j < _x.size(); j++){

T WW  = W(_x[j] - x[i]);
f[i] += WW * _y[j];
sum  += WW;
}

f[i] /= sum;
}

return f;
}

template<class T>
std::vector<T> KernelFit1D<T>::StdDev(const std::vector<T> &x){


if ( x.empty() )
throw KernelFitError("From KernelFit1D::StdDiv(), the input vector "
"cannot be empty!");

std::vector<T> f = Solve( _x );

std::vector<T> var(_x.size(), 0.0);
for (std::size_t i = 0; i < _x.size(); i++)
var[i] = pow(_y[i] - f[i], 2.0);

KernelFit1D<T> profile(_x, var, _b);
std::vector<T> stdev = profile.Solve(x);

for (std::size_t i = 0; i < x.size(); i++)
stdev[i] = sqrt(stdev[i]);

return stdev;
}

template<class T>
KernelFit2D<T>::KernelFit2D(const std::vector<T> &x, const std::vector<T> &y,
const std::vector<T> &z, const T &bandwidth){


if ( x.empty() || y.empty() || z.empty() )
throw KernelFitError("From KernelFit2D::KernelFit2D(), one or more "
"input vectors were empty!");

if ( x.size() != y.size() || x.size() != z.size() )
throw KernelFitError("From KernelFit2D::KernelFit2D(), input vectors "
"must be equal in length!");

if ( bandwidth <= 0.0 )
throw KernelFitError("From KernelFit2D::KernelFit2D(), the bandwidth "
"must be greater than zero!");

_x = x;
_y = y;
_z = z;
_b = bandwidth * bandwidth; 

}

template<class T>
std::vector< std::vector<T> > KernelFit2D<T>::Solve(const std::vector<T> &x,
const std::vector<T> &y){


if ( x.empty() || y.empty() )
throw KernelFitError("From KernelFit2D::Solve(), one or both of "
"`x` and `y` were empty!");

std::vector< std::vector<T> > f(x.size(), std::vector<T>(y.size(), 0.0));

#pragma omp parallel for shared(f)
for (std::size_t i = 0; i < x.size(); i++)
for (std::size_t j = 0; j < y.size(); j++){

T sum = 0.0;

for (std::size_t k = 0; k < _x.size(); k++){

T W      = Kernel(x[i] - _x[k], y[j] - _y[k]);
f[i][j] += W * _z[k];
sum     += W;
}

f[i][j] /= sum;
}

return f;
}

template<class T>
std::vector< std::vector<T> > KernelFit2D<T>::Solve(const std::vector<T> &x,
const std::vector<T> &y, T (*W)(T, T)){


if ( x.empty() || y.empty() )
throw KernelFitError("From KernelFit2D::Solve(), one or both of "
"`x` and `y` were empty!");

std::vector< std::vector<T> > f( x.size(), std::vector<T>(y.size(), 0.0));

#pragma omp parallel for shared(f)
for (std::size_t i = 0; i < x.size(); i++)
for (std::size_t j = 0; j < y.size(); j++){

T sum = 0.0;

for (std::size_t k = 0; k < _x.size(); k++){

T WW     = W(x[i] - _x[k], y[j] - _y[k]);
f[i][j] += WW * _z[k];
sum     += WW;
}

f[i][j] /= sum;
}

return f;
}

template<class T>
std::vector< std::vector<T> > KernelFit2D<T>::StdDev(const std::vector<T> &x,
const std::vector<T> &y){


if ( x.empty() || y.empty() )
throw KernelFitError("From KernelFit2D::StdDev(), one or both of the "
"input vectors were empty!");

std::vector<T> f(_x.size(), 0.0);

#pragma omp parallel for shared(f)
for (std::size_t i = 0; i < _x.size(); i++){

T sum = 0.0;

for (std::size_t j = 0; j < _x.size(); j++){

T W   = Kernel(_x[i] - _x[j], _y[i] - _y[j]);
f[i] += W * _z[j];
sum  += W;
}

f[i] /= sum;
}

std::vector<T> var(_x.size(), 0.0);
for (std::size_t i = 0; i < _x.size(); i++)
var[i] = pow(_z[i] - f[i], 2.0);

KernelFit2D<T> profile(_x, _y, var, _b);
std::vector< std::vector<T> > stdev = profile.Solve(x, y);

#pragma omp parallel for shared(stdev)
for (std::size_t i = 0; i < x.size(); i++)
for (std::size_t j = 0; j < y.size(); j++)
stdev[i][j] = sqrt(stdev[i][j]);

return stdev;
}

template class KernelFit1D<float>;
template class KernelFit2D<float>;
template class KernelFit1D<double>;
template class KernelFit2D<double>;
template class KernelFit1D<long double>;
template class KernelFit2D<long double>;
