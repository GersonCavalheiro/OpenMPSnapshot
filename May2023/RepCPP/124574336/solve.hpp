
#ifndef BOOST_MATH_TOOLS_SOLVE_HPP
#define BOOST_MATH_TOOLS_SOLVE_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/assert.hpp>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4996 4267 4244)
#endif

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

namespace boost{ namespace math{ namespace tools{

template <class T>
boost::numeric::ublas::vector<T> solve(
const boost::numeric::ublas::matrix<T>& A_,
const boost::numeric::ublas::vector<T>& b_)
{

boost::numeric::ublas::matrix<T> A(A_);
boost::numeric::ublas::vector<T> b(b_);
boost::numeric::ublas::permutation_matrix<> piv(b.size());
lu_factorize(A, piv);
lu_substitute(A, piv, b);
boost::numeric::ublas::vector<T> delta(b.size());
for(unsigned k = 0; k < 1; ++k)
{
noalias(delta) = prod(A_, b);
delta -= b_;
lu_substitute(A, piv, delta);
b -= delta;

T max_error = 0;

for(unsigned i = 0; i < delta.size(); ++i)
{
T err = fabs(delta[i] / b[i]);
if(err > max_error)
max_error = err;
}
}

return b;
}

}}} 

#endif 


