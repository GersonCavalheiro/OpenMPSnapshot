

#include <boost/config.hpp>
#ifdef BOOST_MSVC
#pragma warning(disable:4996)
#endif

#define BOOST_TEST_MODULE numeric_runge_kutta

#include <iostream>
#include <cmath>

#include <boost/array.hpp>

#include <boost/test/unit_test.hpp>

#include <boost/mpl/vector.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/stepper/extrapolation_stepper.hpp>

using namespace boost::unit_test;
using namespace boost::numeric::odeint;
namespace mpl = boost::mpl;

typedef double value_type;

typedef boost::array< double , 2 > state_type;

struct osc
{
void operator()( const state_type &x , state_type &dxdt , const double t ) const
{
dxdt[0] = x[1];
dxdt[1] = -x[0];
}
};


template< class StepperCategory >
struct resetter
{
template< class Stepper >
static void reset( Stepper &stepper ) { }
};

template< >
struct resetter< explicit_error_stepper_fsal_tag >
{
template< class Stepper >
static void reset( Stepper &stepper ) 
{ stepper.reset(); }
};


BOOST_AUTO_TEST_SUITE( numeric_runge_kutta_test )



template< class Stepper >
struct perform_runge_kutta_test
{
void operator()( void )
{

Stepper stepper;
const int o = stepper.order()+1; 

const state_type x0 = {{ 0.0 , 1.0 }};
state_type x1;
const double t = 0.0;

double dt = 0.5;
stepper.do_step( osc() , x0 , t , x1 , dt );
const double f = 2.0 * std::abs( sin(dt) - x1[0] ) / std::pow( dt , o ); 

std::cout << o << " , " << f << std::endl;


while( f*std::pow( dt , o ) > 1E-16 )
{
resetter< typename Stepper::stepper_category >::reset( stepper );

stepper.do_step( osc() , x0 , t , x1 , dt );
std::cout << "Testing dt=" << dt << std::endl;
BOOST_CHECK_LT( std::abs( sin(dt) - x1[0] ) , f*std::pow( dt , o ) );
dt *= 0.5;
}
}
};



template< class Stepper >
struct perform_runge_kutta_error_test
{
void operator()( void )
{
Stepper stepper;
const int o = stepper.error_order()+1; 

const state_type x0 = {{ 0.0 , 1.0 }};
state_type x1 , x_err;
const double t = 0.0;

double dt = 0.5;
stepper.do_step( osc() , x0 , t , x1 , dt , x_err );
const double f = 2.0 * std::abs( x_err[0] ) / std::pow( dt , o );

std::cout << o << " , " << f << " , " << x0[0] << std::endl;


while( f*std::pow( dt , o ) > 1E-16 )
{
resetter< typename Stepper::stepper_category >::reset( stepper );

stepper.do_step( osc() , x0 , t , x1 , dt , x_err );
std::cout << "Testing dt=" << dt << ": " << x_err[1] << std::endl;
BOOST_CHECK_SMALL( std::abs( x_err[0] ) , f*std::pow( dt , o ) );
dt *= 0.5;
}
}
};


typedef mpl::vector<
euler< state_type > ,
modified_midpoint< state_type > ,
runge_kutta4< state_type > ,
runge_kutta4_classic< state_type > ,
runge_kutta_cash_karp54_classic< state_type > ,
runge_kutta_cash_karp54< state_type > ,
runge_kutta_dopri5< state_type > ,
runge_kutta_fehlberg78< state_type > ,
extrapolation_stepper< 4, state_type > ,
extrapolation_stepper< 6, state_type > ,
extrapolation_stepper< 8, state_type > ,
extrapolation_stepper< 10, state_type >
> runge_kutta_steppers;

BOOST_AUTO_TEST_CASE_TEMPLATE( runge_kutta_test , Stepper, runge_kutta_steppers )
{
perform_runge_kutta_test< Stepper > tester;
tester();
}


typedef mpl::vector<
runge_kutta_cash_karp54_classic< state_type > ,
runge_kutta_cash_karp54< state_type > ,
runge_kutta_dopri5< state_type > ,
runge_kutta_fehlberg78< state_type > ,
extrapolation_stepper< 4, state_type > ,
extrapolation_stepper< 6, state_type > ,
extrapolation_stepper< 8, state_type > ,
extrapolation_stepper< 10, state_type >
> runge_kutta_error_steppers;

BOOST_AUTO_TEST_CASE_TEMPLATE( runge_kutta_error_test , Stepper, runge_kutta_error_steppers )
{
perform_runge_kutta_error_test< Stepper > tester;
tester();
}

BOOST_AUTO_TEST_SUITE_END()
