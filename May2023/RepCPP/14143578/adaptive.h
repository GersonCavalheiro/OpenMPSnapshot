#pragma once

#include "backend/memory.h"
#include "ode.h"
#include "runge_kutta.h"

namespace dg
{


static auto l2norm = [] ( const auto& x){ return sqrt( dg::blas1::dot(x,x));};


static auto fast_l2norm  = []( const auto& x){ return sqrt( dg::blas1::reduce(
x, (double)0, dg::Sum(), dg::Square()));};

static auto i_control = []( auto dt, auto eps, unsigned embedded_order, unsigned order)
{
using value_type = std::decay_t<decltype(dt[0])>;
return dt[0]*pow( eps[0], -1./(value_type)embedded_order);
};
static auto pi_control = []( auto dt, auto eps, unsigned embedded_order, unsigned order)
{
using value_type = std::decay_t<decltype(dt[0])>;
if( dt[1] == 0)
return i_control( dt, eps, embedded_order, order);
value_type m_k1 = -0.8, m_k2 = 0.31;
value_type factor = pow( eps[0], m_k1/(value_type)embedded_order)
* pow( eps[1], m_k2/(value_type)embedded_order);
return dt[0]*factor;
};

static auto pid_control = []( auto dt, auto eps, unsigned embedded_order, unsigned order)
{
using value_type = std::decay_t<decltype(dt[0])>;
if( dt[1] == 0)
return i_control( dt, eps, embedded_order, order);
if( dt[2] == 0)
return pi_control( dt, eps, embedded_order, order);
value_type m_k1 = -0.58, m_k2 = 0.21, m_k3 = -0.1;
value_type factor = pow( eps[0], m_k1/(value_type)embedded_order)
* pow( eps[1], m_k2/(value_type)embedded_order)
* pow( eps[2], m_k3/(value_type)embedded_order);
return dt[0]*factor;
};

static auto ex_control = [](auto dt, auto eps, unsigned embedded_order, unsigned order)
{
using value_type = std::decay_t<decltype(dt[0])>;
if( dt[1] == 0)
return i_control( dt, eps, embedded_order, order);
value_type m_k1 = -0.367, m_k2 = 0.268;
value_type factor = pow( eps[0], m_k1/(value_type)embedded_order)
* pow( eps[0]/eps[1], m_k2/(value_type)embedded_order);
return dt[0]*factor;
};
static auto im_control = []( auto dt, auto eps, unsigned embedded_order, unsigned order)
{
using value_type = std::decay_t<decltype(dt[0])>;
if( dt[1] == 0)
return i_control( dt, eps, embedded_order, order);
value_type m_k1 = -0.98, m_k2 = -0.95;
value_type factor = pow( eps[0], m_k1/(value_type)embedded_order)
*  pow( eps[0]/eps[1], m_k2/(value_type)embedded_order);
return dt[0]*dt[0]/dt[1]*factor;
};
static auto imex_control = []( auto dt, auto eps, unsigned embedded_order, unsigned order)
{
using value_type = std::decay_t<decltype(dt[0])>;
value_type h1 = ex_control( dt, eps, embedded_order, order);
value_type h2 = im_control( dt, eps, embedded_order, order);
return fabs( h1) < fabs( h2) ? h1 : h2;
};


namespace detail{
template<class value_type>
struct Tolerance
{
Tolerance( value_type rtol, value_type atol, value_type size) :
m_rtol(rtol*sqrt(size)), m_atol( atol*sqrt(size)){}
DG_DEVICE
void operator()( value_type u0, value_type& delta) const{
delta = delta/ ( m_rtol*fabs(u0) + m_atol);
}
private:
value_type m_rtol, m_atol;
};
} 




template<class Stepper>
struct Adaptive
{
using stepper_type = Stepper;
using container_type = typename Stepper::container_type; 
using value_type = typename Stepper::value_type; 
Adaptive() = default;

template<class ...StepperParams>
Adaptive(StepperParams&& ...ps): m_stepper(std::forward<StepperParams>(ps)...),
m_next(m_stepper.copyable()), m_delta(m_stepper.copyable())
{
dg::blas1::copy( 1., m_next);
m_size = dg::blas1::dot( m_next, 1.);
}
template<class ...Params>
void construct(Params&& ...ps)
{
*this = Adaptive(  std::forward<Params>(ps)...);
}

stepper_type& stepper() { return m_stepper;}
const stepper_type& stepper() const { return m_stepper;}


template< class ODE,
class ControlFunction = value_type (std::array<value_type,3>,
std::array< value_type,3>, unsigned , unsigned),
class ErrorNorm = value_type( const container_type&)>
void step( ODE&& ode,
value_type t0,
const container_type& u0,
value_type& t1,
container_type& u1,
value_type& dt,
ControlFunction control,
ErrorNorm norm,
value_type rtol,
value_type atol,
value_type reject_limit = 2
)
{
m_stepper.step( std::forward<ODE>(ode), t0, u0, m_t_next, m_next, dt,
m_delta);
m_nsteps++;
dg::blas1::subroutine( detail::Tolerance<value_type>( rtol, atol,
m_size), u0, m_delta);
m_eps0 = norm( m_delta);
m_dt0 = dt;
if( m_eps0 > reject_limit || std::isnan( m_eps0) )
{
dt = control( std::array<value_type,3>{m_dt0, 0, m_dt2},
std::array<value_type,3>{m_eps0, m_eps1, m_eps2},
m_stepper.embedded_order(),
m_stepper.order());
if( fabs( dt) > 0.9*fabs(m_dt0))
dt = 0.9*m_dt0;
m_failed = true; m_nfailed++;
dg::blas1::copy( u0, u1);
t1 = t0;
return;
}
if( m_eps0 < 1e-30) 
{
dt = 1e14*m_dt0; 
m_eps0 = 1e-30; 
}
else
{
dt = control( std::array<value_type,3>{m_dt0, m_dt1, m_dt2},
std::array<value_type,3>{m_eps0, m_eps1, m_eps2},
m_stepper.embedded_order(),
m_stepper.order());
if( fabs(dt) > 100*fabs(m_dt0))
dt = 100*m_dt0;
}
m_eps2 = m_eps1;
m_eps1 = m_eps0;
m_dt2 = m_dt1;
m_dt1 = m_dt0;
dg::blas1::copy( m_next, u1);
t1 = m_t_next;
m_failed = false;
}

bool failed() const {
return m_failed;
}
const unsigned& nfailed() const {
return m_nfailed;
}
unsigned& nfailed() {
return m_nfailed;
}
const unsigned& nsteps() const {
return m_nsteps;
}
unsigned& nsteps() {
return m_nsteps;
}


const value_type& get_error( ) const{
return m_eps0;
}
private:
void reset_history(){
m_eps1 = m_eps2 = 1.;
m_dt1 = m_dt2 = 0.;
}
bool m_failed = false;
unsigned m_nfailed = 0;
unsigned m_nsteps = 0;
Stepper m_stepper;
container_type m_next, m_delta;
value_type m_size, m_eps0 = 1, m_eps1=1, m_eps2=1;
value_type m_t_next = 0;
value_type m_dt0 = 0., m_dt1 = 0., m_dt2 = 0.;
};




struct EntireDomain
{
template<class T>
bool contains( T& t) const { return true;}
};


template<class ContainerType>
struct AdaptiveTimeloop : public aTimeloop<ContainerType>
{
using value_type = dg::get_value_type<ContainerType>;
using container_type = ContainerType;
AdaptiveTimeloop( ) = default;



AdaptiveTimeloop( std::function<void (value_type, const ContainerType&,
value_type&, ContainerType&, value_type&)> step)  :
m_step(step){
m_dt_current = dg::Buffer<value_type>( 0.);
}

template<class Adaptive, class ODE, class ErrorNorm =
value_type( const container_type&), class
ControlFunction = value_type(std::array<value_type,3>,
std::array<value_type,3>, unsigned, unsigned)>
AdaptiveTimeloop(
Adaptive&& adapt,
ODE&& ode,
ControlFunction control,
ErrorNorm norm,
value_type rtol,
value_type atol,
value_type reject_limit = 2)
{

m_step = [=, cap = std::tuple<Adaptive, ODE>(std::forward<Adaptive>(adapt),
std::forward<ODE>(ode))  ]( auto t0, auto y0, auto& t,
auto& y, auto& dt) mutable
{
std::get<0>(cap).step( std::get<1>(cap), t0, y0, t, y, dt, control, norm,
rtol, atol, reject_limit);
};
m_dt_current = dg::Buffer<value_type>( 0.);
}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = AdaptiveTimeloop( std::forward<Params>( ps)...);
}


void set_dt( value_type dt){
m_dt_current = dg::Buffer<value_type>(dt);
}


template< class Domain>
void integrate_in_domain(
value_type t0,
const ContainerType& u0,
value_type& t1,
ContainerType& u1,
value_type dt,
Domain&& domain,
value_type eps_root
);

virtual AdaptiveTimeloop* clone() const{return new
AdaptiveTimeloop(*this);}
private:
virtual void do_integrate(value_type& t0, const container_type& u0,
value_type t1, container_type& u1, enum to mode) const;
std::function<void( value_type, const ContainerType&, value_type&,
ContainerType&, value_type&)> m_step;
virtual value_type do_dt( ) const { return m_dt_current.data();}
dg::Buffer<value_type> m_dt_current ;
};

template< class ContainerType>
void AdaptiveTimeloop<ContainerType>::do_integrate(
value_type& t_current,
const ContainerType& u0,
value_type t1,
ContainerType& u1,
enum to mode
)const
{
value_type deltaT = t1-t_current;
bool forward = (deltaT > 0);

value_type& dt_current = m_dt_current.data();
if( dt_current == 0)
dt_current = forward ? 1e-6 : -1e-6; 
if( (dt_current < 0 && forward) || ( dt_current > 0 && !forward) )
throw dg::Error( dg::Message(_ping_)<<"Error in AdaptiveTimeloop: Timestep has wrong sign! You cannot change direction mid-step: dt "<<dt_current);

blas1::copy( u0, u1 );
while( (forward && t_current < t1) || (!forward && t_current > t1))
{
if( dg::to::exact == mode
&&( (forward && t_current + dt_current > t1)
|| (!forward && t_current + dt_current < t1) ) )
dt_current = t1-t_current;
if( dg::to::at_least == mode
&&( (forward && dt_current > deltaT)
|| (!forward && dt_current < deltaT) ) )
dt_current = deltaT;
try{
m_step( t_current, u1, t_current, u1, dt_current);
}catch ( dg::Error& e)
{
e.append( dg::Message(_ping_) << "Error in AdaptiveTimeloop::integrate");
throw;
}
if( !std::isfinite(dt_current) || fabs(dt_current) < 1e-9*fabs(deltaT))
{
value_type dt_current0 = dt_current;
dt_current = 0.;
throw dg::Error(dg::Message(_ping_)<<"Adaptive integrate failed to converge! dt = "<<std::scientific<<dt_current0);
}
}
}

template< class ContainerType>
template< class Domain>
void AdaptiveTimeloop<ContainerType>::integrate_in_domain(
value_type t0,
const ContainerType& u0,
value_type& t1,
ContainerType& u1,
value_type dt,
Domain&& domain,
value_type eps_root
)
{
value_type t_current = t0, dt_current = dt;
blas1::copy( u0, u1 );
ContainerType& current(u1);
if( t1 == t0)
return;
bool forward = (t1 - t0 > 0);
if( dt == 0)
dt_current = forward ? 1e-6 : -1e-6; 

ContainerType last( u0);
while( (forward && t_current < t1) || (!forward && t_current > t1))
{
t0 = t_current;
dg::blas1::copy( current, last);
if( (forward && t_current+dt_current > t1) || (!forward && t_current +
dt_current < t1) )
dt_current = t1-t_current;
try{
m_step( t_current, current, t_current, current, dt_current);
}catch ( dg::Error& e)
{
e.append( dg::Message(_ping_) << "Error in AdaptiveTimeloop::integrate");
throw;
}
if( !std::isfinite(dt_current) || fabs(dt_current) < 1e-9*fabs(t1-t0))
throw dg::Error(dg::Message(_ping_)<<"integrate_in_domain failed to converge! dt = "<<std::scientific<<dt_current);
if( !domain.contains( current) )
{
t1 = t_current;

dg::blas1::copy( last, current);

int j_max = 50;
for(int j=0; j<j_max; j++)
{
if( fabs(t1-t0) < eps_root*fabs(t1) + eps_root)
{
return;
}
dt_current = (t1-t0)/2.;
t_current = t0; 
value_type failed = t_current;
m_step( t_current, current, t_current, current, dt_current);
if( failed == t_current)
{
dt_current = (t1-t0)/4.;
break; 
}

if( domain.contains( current) )
{
t0 = t_current;
dg::blas1::copy( current, last);
}
else
{
t1 = t_current;
dg::blas1::copy( last, current);
}
if( (j_max - 1) == j)
throw dg::Error( dg::Message(_ping_)<<"integrate_in_domain: too many steps in root finding!");
}
}
}
}

}
