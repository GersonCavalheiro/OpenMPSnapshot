

#include <limits>

#include <boost/config.hpp>
#ifdef BOOST_MSVC
#pragma warning(disable:4996)
#endif

#define BOOST_TEST_MODULE odeint_trivial_state

#include <boost/test/unit_test.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>

#include <boost/numeric/odeint/stepper/euler.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/generation.hpp>
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

using namespace boost::unit_test;
using namespace boost::numeric::odeint;

namespace mpl = boost::mpl;

struct constant_system
{
template< typename T >
void operator()( const T &x , T &dxdt , const T t ) const
{ dxdt = 1.0; }
};


BOOST_AUTO_TEST_SUITE( trivial_state_test )



typedef mpl::vector<
euler< double > ,
runge_kutta4< double > ,
euler< float , float , float , float > ,
runge_kutta4< float , float , float , float >
>::type stepper_types;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_do_step , T, stepper_types )
{
typedef T stepper_type;
stepper_type stepper;
typename stepper_type::state_type x = 0.0;
typename stepper_type::time_type t = 0.0;
typename stepper_type::time_type dt = 0.1;
stepper.do_step( constant_system() , x , t , dt );
BOOST_CHECK_CLOSE( x , 0.1 , 100*std::numeric_limits< typename stepper_type::state_type >::epsilon() );


typename stepper_type::state_type x_out;
stepper.do_step( constant_system() , x , t , x_out , dt );
BOOST_CHECK_CLOSE( x , 0.1 , 100*std::numeric_limits< typename stepper_type::state_type >::epsilon() );
BOOST_CHECK_CLOSE( x_out , 0.2 , 100*std::numeric_limits< typename stepper_type::state_type >::epsilon() );
}




typedef mpl::vector<
runge_kutta_cash_karp54< double > ,
runge_kutta_dopri5< double > ,
runge_kutta_cash_karp54< float , float , float , float > ,
runge_kutta_dopri5< float , float , float , float >
> error_stepper_types;

BOOST_AUTO_TEST_CASE_TEMPLATE( test_integrate , T , error_stepper_types )
{
typedef T stepper_type;
typename stepper_type::state_type x = 0.0;
typename stepper_type::time_type t0 = 0.0;
typename stepper_type::time_type t1 = 1.0;
typename stepper_type::time_type dt = 0.1;
integrate_adaptive( make_controlled< stepper_type >( 1e-6 , 1e-6 ) , constant_system() , x , t0 , t1 , dt );
BOOST_CHECK_CLOSE( x , 1.0 , 100*std::numeric_limits< typename stepper_type::state_type >::epsilon() );
}

BOOST_AUTO_TEST_SUITE_END()
