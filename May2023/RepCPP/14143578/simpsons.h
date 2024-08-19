#pragma once

#include <list>
#include "backend/exceptions.h"
#include "blas1.h"


namespace dg{


template<class ContainerType>
struct Simpsons
{
using value_type = get_value_type<ContainerType>;
using container_type = ContainerType; 

Simpsons( unsigned order = 3): m_counter(0), m_order(order), m_t0(0)
{
set_order(order);
}
void set_order( unsigned order){
m_order=order;
m_t.resize( order-1);
m_u.resize(order-1);
if( !(order == 2 || order == 3))
throw dg::Error(dg::Message()<<"Integration order must be either 2 or 3!");
}
unsigned get_order() const{return m_order;}

void init(  value_type t0, const ContainerType& u0) {
m_integral = u0;
for( auto& u: m_u)
u = u0;
for( auto& t: m_t)
t = 0;
m_t.front() = t0;
flush();
m_t0 = t0;
}

void flush() {
m_counter = 0; 
dg::blas1::scal( m_integral, 0.);
m_t0 = m_t.front();
}

void add( value_type t_new, const ContainerType& u_new){
if( t_new < m_t.front())
throw dg::Error(dg::Message()<<"New time must be strictly larger than old time!");
auto pt0 = m_t.begin();
auto pt1 = std::next( pt0);
auto pu0 = m_u.begin();
auto pu1 = std::next( pu0);
value_type t0 = *pt1, t1 = *pt0, t2 = t_new;
if( m_counter % 2 == 0 || m_order == 2)
{
dg::blas1::axpbypgz( 0.5*(t2 - t1), u_new,
0.5*(t2 - t1), *pu0 , 1., m_integral);
}
else
{
value_type pre0 = (2.*t0-3.*t1+t2)*(t2-t0)/(6.*(t0-t1));
value_type pre1 = (t2-t0)*(t2-t0)*(t2-t0)/(6.*(t0-t1)*(t1-t2));
value_type pre2 = (t0-3.*t1+2.*t2)*(t0-t2)/(6.*(t1-t2));

dg::blas1::axpby( pre2, u_new, 1., m_integral);
dg::blas1::axpbypgz(
pre1-0.5*(t1-t0), *pu0, 
pre0-0.5*(t1-t0), *pu1,
1., m_integral);
}
m_t.splice( pt0, m_t, pt1, m_t.end());
m_u.splice( pu0, m_u, pu1, m_u.end());
m_t.front() = t_new; 
dg::blas1::copy( u_new, m_u.front());
m_counter++;
}


const ContainerType& get_integral() const{
return m_integral;
}

std::array<value_type,2> get_boundaries() const{
std::array<value_type,2> times{ m_t0, m_t.front()};
return times;
}
private:
unsigned m_counter, m_order;
ContainerType m_integral;
std::list<value_type> m_t;
std::list<ContainerType> m_u;
value_type m_t0;
};

}
