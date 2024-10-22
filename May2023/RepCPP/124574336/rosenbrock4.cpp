

#include <boost/config.hpp>
#ifdef BOOST_MSVC
#pragma warning(disable:4996)
#endif

#define BOOST_TEST_MODULE odeint_rosenbrock4

#include <utility>
#include <iostream>

#include <boost/test/unit_test.hpp>

#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_dense_output.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::unit_test;
using namespace boost::numeric::odeint;

typedef double value_type;
typedef boost::numeric::ublas::vector< value_type > state_type;
typedef boost::numeric::ublas::matrix< value_type > matrix_type;


struct sys
{
void operator()( const state_type &x , state_type &dxdt , const value_type &t ) const
{
dxdt( 0 ) = x( 0 ) + 2 * x( 1 );
dxdt( 1 ) = x( 1 );
}
};

struct jacobi
{
void operator()( const state_type &x , matrix_type &jacobi , const value_type &t , state_type &dfdt ) const
{
jacobi( 0 , 0 ) = 1;
jacobi( 0 , 1 ) = 2;
jacobi( 1 , 0 ) = 0;
jacobi( 1 , 1 ) = 1;
dfdt( 0 ) = 0.0;
dfdt( 1 ) = 0.0;
}
};

BOOST_AUTO_TEST_SUITE( rosenbrock4_test )

BOOST_AUTO_TEST_CASE( test_rosenbrock4_stepper )
{
typedef rosenbrock4< value_type > stepper_type;
stepper_type stepper;

typedef stepper_type::state_type state_type;
typedef stepper_type::value_type stepper_value_type;
typedef stepper_type::deriv_type deriv_type;
typedef stepper_type::time_type time_type;

state_type x( 2 ) , xerr( 2 );
x(0) = 0.0; x(1) = 1.0;

stepper.do_step( std::make_pair( sys() , jacobi() ) , x , 0.0 , 0.1 , xerr );

stepper.do_step( std::make_pair( sys() , jacobi() ) , x , 0.0 , 0.1 );


}

BOOST_AUTO_TEST_CASE( test_rosenbrock4_controller )
{
typedef rosenbrock4_controller< rosenbrock4< value_type > > stepper_type;
stepper_type stepper;

typedef stepper_type::state_type state_type;
typedef stepper_type::value_type stepper_value_type;
typedef stepper_type::deriv_type deriv_type;
typedef stepper_type::time_type time_type;

state_type x( 2 );
x( 0 ) = 0.0 ; x(1) = 1.0;

value_type t = 0.0 , dt = 0.01;
stepper.try_step( std::make_pair( sys() , jacobi() ) , x , t , dt );
}

BOOST_AUTO_TEST_CASE( test_rosenbrock4_dense_output )
{
typedef rosenbrock4_dense_output< rosenbrock4_controller< rosenbrock4< value_type > > > stepper_type;
typedef rosenbrock4_controller< rosenbrock4< value_type > > controlled_stepper_type;
controlled_stepper_type  c_stepper;
stepper_type stepper( c_stepper );

typedef stepper_type::state_type state_type;
typedef stepper_type::value_type stepper_value_type;
typedef stepper_type::deriv_type deriv_type;
typedef stepper_type::time_type time_type;
state_type x( 2 );
x( 0 ) = 0.0 ; x(1) = 1.0;
stepper.initialize( x , 0.0 , 0.1 );
std::pair< value_type , value_type > tr = stepper.do_step( std::make_pair( sys() , jacobi() ) );
stepper.calc_state( 0.5 * ( tr.first + tr.second ) , x );
}

class rosenbrock4_controller_max_dt_adaptable : public rosenbrock4_controller< rosenbrock4< value_type > >
{
public:
void set_max_dt(value_type max_dt)
{
m_max_dt = max_dt;
}
};

BOOST_AUTO_TEST_CASE( test_rosenbrock4_dense_output_ref )
{
typedef rosenbrock4_dense_output< boost::reference_wrapper< rosenbrock4_controller_max_dt_adaptable > > stepper_type;
rosenbrock4_controller_max_dt_adaptable  c_stepper;
stepper_type stepper( boost::ref( c_stepper ) );

typedef stepper_type::state_type state_type;
typedef stepper_type::value_type stepper_value_type;
typedef stepper_type::deriv_type deriv_type;
typedef stepper_type::time_type time_type;
state_type x( 2 );
x( 0 ) = 0.0 ; x(1) = 1.0;
stepper.initialize( x , 0.0 , 0.1 );
std::pair< value_type , value_type > tr = stepper.do_step( std::make_pair( sys() , jacobi() ) );
stepper.calc_state( 0.5 * ( tr.first + tr.second ) , x );

const double max_dt = 1e-8;
c_stepper.set_max_dt(max_dt);
stepper.do_step( std::make_pair( sys() , jacobi() ) );
BOOST_CHECK_CLOSE(max_dt, stepper.current_time_step(), 1e-14);
}

BOOST_AUTO_TEST_CASE( test_rosenbrock4_copy_dense_output )
{
typedef rosenbrock4_controller< rosenbrock4< value_type > > controlled_stepper_type;
typedef rosenbrock4_dense_output< controlled_stepper_type > stepper_type;

controlled_stepper_type  c_stepper;
stepper_type stepper( c_stepper );
stepper_type stepper2( stepper );
}

BOOST_AUTO_TEST_SUITE_END()
