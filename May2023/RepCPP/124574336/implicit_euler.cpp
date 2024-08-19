


#include <boost/config.hpp>
#ifdef BOOST_MSVC
#pragma warning(disable:4996)
#endif

#define BOOST_TEST_MODULE odeint_implicit_euler

#include <boost/test/unit_test.hpp>

#include <utility>
#include <iostream>

#include <boost/numeric/odeint/stepper/implicit_euler.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::unit_test;
using namespace boost::numeric::odeint;

typedef double value_type;
typedef boost::numeric::ublas::vector< value_type > state_type;
typedef boost::numeric::ublas::matrix< value_type > matrix_type;


struct sys
{
void operator()( const state_type &x , state_type &dxdt , const value_type t ) const
{
dxdt( 0 ) = x( 0 ) + 2 * x( 1 );
dxdt( 1 ) = x( 1 );
}
};

struct jacobi 
{
void operator()( const state_type &x , matrix_type &jacobi , const value_type t ) const
{
jacobi( 0 , 0 ) = 1;
jacobi( 0 , 1 ) = 2;
jacobi( 1 , 0 ) = 0;
jacobi( 1 , 1 ) = 1;
}
};

BOOST_AUTO_TEST_SUITE( implicit_euler_test )

BOOST_AUTO_TEST_CASE( test_euler )
{
implicit_euler< value_type > stepper;
state_type x( 2 );
x(0) = 0.0; x(1) = 1.0;

value_type eps = 1E-12;


stepper.do_step( std::make_pair( sys() , jacobi() ) , x , 0.0 , 0.1 );

using std::abs;

BOOST_CHECK_MESSAGE( abs( x(0) - 20.0/81.0 ) < eps , x(0) - 20.0/81.0 );
BOOST_CHECK_MESSAGE( abs( x(1) - 10.0/9.0 ) < eps , x(0) - 10.0/9.0 );

}

BOOST_AUTO_TEST_SUITE_END()
