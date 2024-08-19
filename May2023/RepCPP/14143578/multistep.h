#pragma once

#include <map>
#include <tuple>
#include "ode.h"
#include "runge_kutta.h"
#include "multistep_tableau.h"


namespace dg{



template<class ContainerType>
struct FilteredExplicitMultistep;


template<class ContainerType>
struct ExplicitMultistep
{
using value_type = get_value_type<ContainerType>;
using container_type = ContainerType; 
ExplicitMultistep() = default;
ExplicitMultistep( ConvertsToMultistepTableau<value_type> tableau, const ContainerType& copyable): m_fem( tableau, copyable){ }
template<class ...Params>
void construct(Params&& ...ps)
{
*this = ExplicitMultistep(  std::forward<Params>(ps)...);
}
const ContainerType& copyable()const{ return m_fem.copyable();}


template< class ExplicitRHS>
void init( ExplicitRHS& rhs, value_type t0, const ContainerType& u0, value_type dt){
dg::IdentityFilter id;
m_fem.init( std::tie(rhs, id), t0, u0, dt);
}


template< class ExplicitRHS>
void step( ExplicitRHS& rhs, value_type& t, ContainerType& u){
dg::IdentityFilter id;
m_fem.step( std::tie(rhs, id), t, u);
}

private:
FilteredExplicitMultistep<ContainerType> m_fem;
};



template<class ContainerType>
struct ImExMultistep
{
using value_type = get_value_type<ContainerType>;
using container_type = ContainerType; 
ImExMultistep() = default;


ImExMultistep( ConvertsToMultistepTableau<value_type> tableau,
const ContainerType& copyable):
m_t(tableau)
{
unsigned size_f = 0;
for( unsigned i=0; i<m_t.steps(); i++ )
{
if( m_t.im( i+1) != 0 )
size_f = i+1;
}
m_im.assign( size_f, copyable);
m_u.assign( m_t.steps(), copyable);
m_ex.assign( m_t.steps(), copyable);
m_tmp = copyable;
m_counter = 0;
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = ImExMultistep( std::forward<Params>( ps)...);
}
const ContainerType& copyable()const{ return m_tmp;}


template< class ExplicitRHS, class ImplicitRHS, class Solver>
void init( const std::tuple<ExplicitRHS, ImplicitRHS, Solver>& ode,
value_type t0, const ContainerType& u0, value_type dt);


template< class ExplicitRHS, class ImplicitRHS, class Solver>
void step( const std::tuple<ExplicitRHS, ImplicitRHS, Solver>& ode,
value_type& t, ContainerType& u);

private:
dg::MultistepTableau<value_type> m_t;
std::vector<ContainerType> m_u, m_ex, m_im;
ContainerType m_tmp;
value_type m_tu, m_dt;
unsigned m_counter; 
};

template< class ContainerType>
template< class RHS, class Diffusion, class Solver>
void ImExMultistep<ContainerType>::init( const std::tuple<RHS, Diffusion, Solver>& ode, value_type t0, const ContainerType& u0, value_type dt)
{
m_tu = t0, m_dt = dt;
unsigned s = m_t.steps();
blas1::copy(  u0, m_u[s-1]);
m_counter = 0;
if( s-1-m_counter < m_im.size())
std::get<1>(ode)( m_tu, m_u[s-1-m_counter], m_im[s-1-m_counter]);
std::get<0>(ode)( t0, u0, m_ex[s-1]); 
}

template<class ContainerType>
template< class RHS, class Diffusion, class Solver>
void ImExMultistep<ContainerType>::step( const std::tuple<RHS, Diffusion, Solver>& ode, value_type& t, ContainerType& u)
{
unsigned s = m_t.steps();
if( m_counter < s - 1)
{
std::map<unsigned, std::string> order2method{
{1, "ARK-4-2-3"},
{2, "ARK-4-2-3"},
{3, "ARK-4-2-3"},
{4, "ARK-6-3-4"},
{5, "ARK-8-4-5"},
{6, "ARK-8-4-5"},
{7, "ARK-8-4-5"}
};
ARKStep<ContainerType> ark( order2method.at( m_t.order()), u);
ContainerType tmp ( u);
ark.step( ode, t, u, t, u, m_dt, tmp);
m_counter++;
m_tu = t;
dg::blas1::copy( u, m_u[s-1-m_counter]);
if( s-1-m_counter < m_im.size())
std::get<1>(ode)( m_tu, m_u[s-1-m_counter], m_im[s-1-m_counter]);
std::get<0>(ode)( m_tu, m_u[s-1-m_counter], m_ex[s-1-m_counter]);
return;
}
dg::blas1::axpbypgz( m_t.a(0), m_u[0], m_dt*m_t.ex(0), m_ex[0], 0., m_tmp);
for (unsigned i = 1; i < s; i++)
dg::blas1::axpbypgz( m_t.a(i), m_u[i], m_dt*m_t.ex(i), m_ex[i], 1., m_tmp);
for (unsigned i = 0; i < m_im.size(); i++)
dg::blas1::axpby( m_dt*m_t.im(i+1), m_im[i], 1., m_tmp);
t = m_tu = m_tu + m_dt;

std::rotate( m_u.rbegin(), m_u.rbegin() + 1, m_u.rend());
std::rotate( m_ex.rbegin(), m_ex.rbegin() + 1, m_ex.rend());
if( !m_im.empty())
std::rotate( m_im.rbegin(), m_im.rbegin() + 1, m_im.rend());
value_type alpha = m_dt*m_t.im(0);
std::get<2>(ode)( alpha, t, u, m_tmp);

blas1::copy( u, m_u[0]); 
if( 0 < m_im.size())
dg::blas1::axpby( 1./alpha, u, -1./alpha, m_tmp, m_im[0]);
std::get<0>(ode)(m_tu, m_u[0], m_ex[0]); 
}


template<class ContainerType>
struct ImplicitMultistep
{

using value_type = get_value_type<ContainerType>;
using container_type = ContainerType; 
ImplicitMultistep() = default;


ImplicitMultistep( ConvertsToMultistepTableau<value_type> tableau, const
ContainerType& copyable): m_t( tableau)
{
unsigned size_f = 0;
for( unsigned i=0; i<m_t.steps(); i++ )
{
if( m_t.im( i+1) != 0 )
size_f = i+1;
}
m_f.assign( size_f, copyable);
m_u.assign( m_t.steps(), copyable);
m_tmp = copyable;
m_counter = 0;
}

template<class ...Params>
void construct(Params&& ...ps)
{
*this = ImplicitMultistep(  std::forward<Params>(ps)...);
}
const ContainerType& copyable()const{ return m_tmp;}


template<class ImplicitRHS, class Solver>
void init(const std::tuple<ImplicitRHS, Solver>& ode, value_type t0, const ContainerType& u0, value_type dt);


template<class ImplicitRHS, class Solver>
void step(const std::tuple<ImplicitRHS, Solver>& ode, value_type& t, container_type& u);
private:
dg::MultistepTableau<value_type> m_t;
value_type m_tu, m_dt;
std::vector<ContainerType> m_u;
std::vector<ContainerType> m_f;
ContainerType m_tmp;
unsigned m_counter = 0; 
};
template< class ContainerType>
template<class ImplicitRHS, class Solver>
void ImplicitMultistep<ContainerType>::init(const std::tuple<ImplicitRHS, Solver>& ode,
value_type t0, const ContainerType& u0, value_type dt)
{
m_tu = t0, m_dt = dt;
dg::blas1::copy( u0, m_u[m_u.size()-1]);
m_counter = 0;
unsigned s = m_t.steps();
if( s-1-m_counter < m_f.size())
std::get<0>(ode)( m_tu, m_u[s-1-m_counter], m_f[s-1-m_counter]);
}

template< class ContainerType>
template<class ImplicitRHS, class Solver>
void ImplicitMultistep<ContainerType>::step(const std::tuple<ImplicitRHS, Solver>& ode,
value_type& t, container_type& u)
{
unsigned s = m_t.steps();
if( m_counter < s - 1)
{
std::map<unsigned, enum tableau_identifier> order2method{
{1, IMPLICIT_EULER_1_1},
{2, SDIRK_2_1_2},
{3, SDIRK_4_2_3},
{4, SDIRK_5_3_4},
{5, SANCHEZ_6_5},
{6, SANCHEZ_7_6},
{7, SANCHEZ_7_6}
};
ImplicitRungeKutta<ContainerType> dirk(
order2method.at(m_t.order()), u);
dirk.step( ode, t, u, t, u, m_dt);
m_counter++;
m_tu = t;
dg::blas1::copy( u, m_u[s-1-m_counter]);
if( s-1-m_counter < m_f.size())
std::get<0>(ode)( m_tu, m_u[s-1-m_counter], m_f[s-1-m_counter]);
return;
}
dg::blas1::axpby( m_t.a(0), m_u[0], 0., m_tmp);
for (unsigned i = 1; i < s; i++)
dg::blas1::axpby( m_t.a(i), m_u[i], 1., m_tmp);
for (unsigned i = 0; i < m_f.size(); i++)
dg::blas1::axpby( m_dt*m_t.im(i+1), m_f[i], 1., m_tmp);
t = m_tu = m_tu + m_dt;

std::rotate(m_u.rbegin(), m_u.rbegin() + 1, m_u.rend());
if( !m_f.empty())
std::rotate(m_f.rbegin(), m_f.rbegin() + 1, m_f.rend());
value_type alpha = m_dt*m_t.im(0);
std::get<1>(ode)( alpha, t, u, m_tmp);

dg::blas1::copy( u, m_u[0]);
if( 0 < m_f.size())
dg::blas1::axpby( 1./alpha, u, -1./alpha, m_tmp, m_f[0]);
}




template<class ContainerType>
struct FilteredExplicitMultistep
{
using value_type = get_value_type<ContainerType>;
using container_type = ContainerType; 
FilteredExplicitMultistep(){ m_u.resize(1); 
}


FilteredExplicitMultistep( ConvertsToMultistepTableau<value_type> tableau,
const ContainerType& copyable): m_t(tableau)
{
m_f.assign( m_t.steps(), copyable);
m_u.assign( m_t.steps(), copyable);
m_counter = 0;
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = FilteredExplicitMultistep( std::forward<Params>( ps)...);
}
const ContainerType& copyable()const{ return m_u[0];}


template< class ExplicitRHS, class Limiter>
void init( const std::tuple<ExplicitRHS, Limiter>& ode, value_type t0, const ContainerType& u0, value_type dt);


template< class ExplicitRHS, class Limiter>
void step( const std::tuple<ExplicitRHS, Limiter>& ode, value_type& t, ContainerType& u);

private:
dg::MultistepTableau<value_type> m_t;
std::vector<ContainerType> m_u, m_f;
value_type m_tu, m_dt;
unsigned m_counter; 
};
template< class ContainerType>
template< class ExplicitRHS, class Limiter>
void FilteredExplicitMultistep<ContainerType>::init( const std::tuple<ExplicitRHS, Limiter>& ode, value_type t0, const ContainerType& u0, value_type dt)
{
m_tu = t0, m_dt = dt;
unsigned s = m_t.steps();
dg::blas1::copy( u0, m_u[s-1]);
std::get<1>(ode)( m_u[s-1]);
std::get<0>(ode)(m_tu, m_u[s-1], m_f[s-1]); 
m_counter = 0;
}

template<class ContainerType>
template<class ExplicitRHS, class Limiter>
void FilteredExplicitMultistep<ContainerType>::step(const std::tuple<ExplicitRHS, Limiter>& ode, value_type& t, ContainerType& u)
{
unsigned s = m_t.steps();
if( m_counter < s-1)
{
std::map<unsigned, enum tableau_identifier> order2method{
{1, SSPRK_2_2},
{2, SSPRK_2_2},
{3, SSPRK_3_3},
{4, SSPRK_5_4},
{5, SSPRK_5_4},
{6, SSPRK_5_4},
{7, SSPRK_5_4}
};
ShuOsher<ContainerType> rk( order2method.at(m_t.order()), u);
rk.step( ode, t, u, t, u, m_dt);
m_counter++;
m_tu = t;
blas1::copy(  u, m_u[s-1-m_counter]);
std::get<0>(ode)( m_tu, m_u[s-1-m_counter], m_f[s-1-m_counter]);
return;
}
t = m_tu = m_tu + m_dt;
dg::blas1::axpby( m_t.a(0), m_u[0], m_dt*m_t.ex(0), m_f[0], u);
for (unsigned i = 1; i < s; i++){
dg::blas1::axpbypgz( m_t.a(i), m_u[i], m_dt*m_t.ex(i), m_f[i], 1., u);
}
std::get<1>(ode)( u);
std::rotate( m_f.rbegin(), m_f.rbegin()+1, m_f.rend());
std::rotate( m_u.rbegin(), m_u.rbegin()+1, m_u.rend());
blas1::copy( u, m_u[0]); 
std::get<0>(ode)(m_tu, m_u[0], m_f[0]); 
}



template<class ContainerType>
struct MultistepTimeloop : public aTimeloop<ContainerType>
{
using container_type = ContainerType;
using value_type = dg::get_value_type<ContainerType>;
MultistepTimeloop( ) = default;

MultistepTimeloop( std::function<void ( value_type&, ContainerType&)>
step, value_type dt ) : m_step(step), m_dt(dt){}

template<class Stepper, class ODE>
MultistepTimeloop(
Stepper&& stepper, ODE&& ode, value_type t0, const
ContainerType& u0, value_type dt )
{
stepper.init( ode, t0, u0, dt);
m_step = [=, cap = std::tuple<Stepper, ODE>(std::forward<Stepper>(stepper),
std::forward<ODE>(ode))  ]( auto& t, auto& y) mutable
{
std::get<0>(cap).step( std::get<1>(cap), t, y);
};
m_dt = dt;
}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = MultistepTimeloop( std::forward<Params>( ps)...);
}

virtual MultistepTimeloop* clone() const{return new
MultistepTimeloop(*this);}
private:
virtual void do_integrate(value_type& t0, const container_type& u0,
value_type t1, container_type& u1, enum to mode) const;
std::function<void ( value_type&, ContainerType&)> m_step;
virtual value_type do_dt( ) const { return m_dt;}
value_type m_dt;
};

template< class ContainerType>
void MultistepTimeloop<ContainerType>::do_integrate(
value_type&  t_begin, const ContainerType&
begin, value_type t_end, ContainerType& end,
enum to mode ) const
{
bool forward = (t_end - t_begin > 0);
if( (m_dt < 0 && forward) || ( m_dt > 0 && !forward) )
throw dg::Error( dg::Message(_ping_)<<"Timestep has wrong sign! dt "<<m_dt);
if( m_dt == 0)
throw dg::Error( dg::Message(_ping_)<<"Timestep may not be zero in MultistepTimeloop!");
dg::blas1::copy( begin, end);
if( is_divisable( t_end-t_begin, m_dt))
{
unsigned N = (unsigned)round((t_end - t_begin)/m_dt);
for( unsigned i=0; i<N; i++)
m_step( t_begin, end);
return;
}
else
{
if( dg::to::exact == mode)
throw dg::Error( dg::Message(_ping_) << "In a multistep integrator dt "
<<m_dt<<" has to integer divide the interval "<<t_end-t_begin);
unsigned N = (unsigned)floor( (t_end-t_begin)/m_dt);
for( unsigned i=0; i<N+1; i++)
m_step( t_begin, end);
}
}


} 
