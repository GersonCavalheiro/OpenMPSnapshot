

#include <boost/config.hpp>
#ifdef BOOST_MSVC
#pragma warning(disable:4996)
#endif

#define BOOST_TEST_MODULE numeric_adams_bashforth_moulton

#include <iostream>
#include <cmath>

#include <boost/test/unit_test.hpp>

#include <boost/mpl/vector.hpp>

#include <boost/numeric/odeint.hpp>

using namespace boost::unit_test;
using namespace boost::numeric::odeint;
namespace mpl = boost::mpl;

typedef double value_type;

typedef value_type state_type;


struct simple_rhs
{
void operator()( const state_type& x , state_type &dxdt , const double t ) const
{
dxdt = t;
}
};

BOOST_AUTO_TEST_SUITE( numeric_abm_time_dependent_test )



template< class Stepper >
struct perform_abm_time_dependent_test
{
void operator()( void )
{
Stepper stepper;
const int o = stepper.order()+1; 

const state_type x0 = 0.0;
state_type x1 = x0;
double t = 0.0;
double dt = 0.1;
const int steps = 10;

integrate_n_steps( boost::ref(stepper) , simple_rhs(), x1 , t , dt , steps );
BOOST_CHECK_LT( std::abs( 0.5 - x1 ) , std::pow( dt , o ) );
}
};

typedef mpl::vector<
adams_bashforth_moulton< 2 , state_type > ,
adams_bashforth_moulton< 3 , state_type > ,
adams_bashforth_moulton< 4 , state_type > ,
adams_bashforth_moulton< 5 , state_type > ,
adams_bashforth_moulton< 6 , state_type > ,
adams_bashforth_moulton< 7 , state_type > ,
adams_bashforth_moulton< 8 , state_type >
> adams_bashforth_moulton_steppers;

BOOST_AUTO_TEST_CASE_TEMPLATE( abm_time_dependent_test , Stepper, adams_bashforth_moulton_steppers )
{
perform_abm_time_dependent_test< Stepper > tester;
tester();
}

BOOST_AUTO_TEST_SUITE_END()
