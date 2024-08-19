

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

using namespace boost::unit_test;
using namespace boost::numeric::odeint;
namespace mpl = boost::mpl;

typedef double value_type;

typedef boost::array< double , 1 > state_type;

struct osc
{
void operator()( const state_type &x, const state_type &v, state_type &a,
const double t ) const
{
a[0] = -x[0];
}
};

BOOST_AUTO_TEST_SUITE( velocity_verlet_test )

BOOST_AUTO_TEST_CASE( numeric_velocity_verlet_test )
{

velocity_verlet<state_type> stepper;
const int steps = 10;
const int o = stepper.order() + 1;

const state_type x0 = {{ 0.0 }};
const state_type v0 = {{ 1.0 }};
state_type x = x0;
state_type v = v0;
const double t = 0.0;

double dt = 0.5;
for ( int step = 0; step < steps; ++step )
{
stepper.do_step(
osc(), std::make_pair( boost::ref( x ), boost::ref( v ) ), t, dt );
}
const double f = steps * std::abs( sin( steps * dt ) - x[0] ) /
std::pow( dt, o ); 

std::cout << o << " , " << f << std::endl;


while( f*std::pow( dt , o ) > 1E-16 )
{
x = x0;
v = v0;
stepper.reset();
for ( int step = 0; step < steps; ++step )
{
stepper.do_step( osc() , std::make_pair(boost::ref(x), boost::ref(v)) , t , dt );
}
std::cout << "Testing dt=" << dt << std::endl;
BOOST_CHECK_LT( std::abs( sin( steps * dt ) - x[0] ),
f * std::pow( dt, o ) );
dt *= 0.5;
}
};


BOOST_AUTO_TEST_SUITE_END()
