

#include <boost/config.hpp>
#ifdef BOOST_MSVC
#pragma warning(disable:4996)
#endif

#define BOOST_TEST_MODULE numeric_symplectic

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

typedef boost::array< double ,1 > state_type;

struct osc
{
void operator()( const state_type &q , state_type &dpdt ) const
{
dpdt[0] = -q[0];
}
};

BOOST_AUTO_TEST_SUITE( numeric_symplectic_test )



template< class Stepper >
struct perform_symplectic_test
{
void operator()( void )
{

Stepper stepper;
const int o = stepper.order()+1; 

const state_type q0 = {{ 0.0 }};
const state_type p0 = {{ 1.0 }};
state_type q1,p1;
std::pair< state_type , state_type >x1( q1 , p1 );
const double t = 0.0;

double dt = 0.5;
stepper.do_step( osc() , std::make_pair( q0 , p0 ) , t , x1 , dt );
const double f = 2.0 * std::abs( sin(dt) - x1.first[0] ) / std::pow( dt , o );

std::cout << o << " , " << f << std::endl;


while( f*std::pow( dt , o ) > 1E-16 )
{
stepper.do_step( osc() , std::make_pair( q0 , p0 ) , t , x1 , dt );
std::cout << "Testing dt=" << dt << std::endl;
BOOST_CHECK_SMALL( std::abs( sin(dt) - x1.first[0] ) , f*std::pow( dt , o ) );
dt *= 0.5;
}
}
};


typedef mpl::vector<
symplectic_euler< state_type > ,
symplectic_rkn_sb3a_mclachlan< state_type >
> symplectic_steppers;

BOOST_AUTO_TEST_CASE_TEMPLATE( symplectic_test , Stepper, symplectic_steppers )
{
perform_symplectic_test< Stepper > tester;
tester();
}

BOOST_AUTO_TEST_SUITE_END()
