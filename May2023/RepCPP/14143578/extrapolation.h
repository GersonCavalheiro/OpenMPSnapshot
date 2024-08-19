#pragma once

#include "blas.h"
#include "topology/operator.h"

namespace dg
{


template<class ContainerType0, class ContainerType1>
std::vector<double> least_squares( const std::vector<ContainerType0>& bs, const ContainerType1 & b)
{
unsigned size = bs.size();
dg::Operator<double> op( size, 0.); 
thrust::host_vector<double> rhs( size, 0.), opIi(rhs); 
std::vector<double> a(size,0.);
for( unsigned i=0; i<size; i++)
{
for( unsigned j=i; j<size; j++)
op(i,j) = dg::blas1::dot( bs[i], bs[j]);
for( unsigned j=0; j<i; j++)
op(i,j) = op(j,i);
rhs[i] = dg::blas1::dot( bs[i], b);
}
auto op_inv = dg::create::inverse( op);
for( unsigned i=0; i<size; i++)
{
for( unsigned j=0; j<size; j++)
opIi[j] = op_inv(i,j);
a[i] = dg::blas1::dot( rhs, opIi) ;
}
return a;
}


template<class ContainerType0, class ContainerType1>
struct LeastSquaresExtrapolation
{
using value_type = get_value_type<ContainerType0>;
using container_type = ContainerType0;
LeastSquaresExtrapolation( ){ m_counter = 0; }

LeastSquaresExtrapolation( unsigned max, const ContainerType0& copyable0, const ContainerType1& copyable1) {
set_max(max, copyable0, copyable1);
}
void set_max( unsigned max, const ContainerType0& copyable0,
const ContainerType1& copyable1)
{
m_counter = 0;
m_x.assign( max, copyable0);
m_y.assign( max, copyable1);
m_max = max;
}
unsigned get_max( ) const{
return m_counter;
}


void extrapolate( double alpha, const ContainerType0& x, double beta,
ContainerType1& y) const{
unsigned size = m_counter;
thrust::host_vector<double> rhs( size, 0.), a(rhs), opIi(rhs); 
for( unsigned i=0; i<size; i++)
rhs[i] = dg::blas1::dot( m_x[i], x);
dg::blas1::scal( y, beta);
for( unsigned i=0; i<size; i++)
{
for( unsigned j=0; j<size; j++)
opIi[j] = m_op_inv(i,j);
a[i] = dg::blas1::dot( rhs, opIi) ;
dg::blas1::axpby( alpha*a[i], m_y[i], 1., y);
}
}

void extrapolate( const ContainerType0& x, ContainerType1& y) const{
extrapolate( 1., x, 0., y);
}


void update( const ContainerType0& x_new, const ContainerType1& y_new){
if( m_max == 0) return;
unsigned size = m_counter < m_max ? m_counter + 1 : m_max;
dg::Operator<double> op( size, 0.), op_inv( size, 0.); 
op(0,0) = dg::blas1::dot( x_new, x_new);
for( unsigned j=1; j<size; j++)
op(0,j) = op( j, 0) = dg::blas1::dot( x_new, m_x[j-1]);
for( unsigned i=1; i<size; i++)
for( unsigned j=1; j<size; j++)
op(i,j) = m_op(i-1,j-1);
try{
op_inv = dg::create::inverse( op);
}
catch ( std::runtime_error & e){
return;
}
m_op_inv = op_inv, m_op = op;
if( m_counter < m_max)
m_counter++;
std::rotate( m_x.rbegin(), m_x.rbegin()+1, m_x.rend());
std::rotate( m_y.rbegin(), m_y.rbegin()+1, m_y.rend());
blas1::copy( x_new, m_x[0]);
blas1::copy( y_new, m_y[0]);
}

private:
unsigned m_max, m_counter;
std::vector<ContainerType0> m_x;
std::vector<ContainerType1> m_y;
dg::Operator<double> m_op, m_op_inv;
};


template<class ContainerType>
struct Extrapolation
{
using value_type = get_value_type<ContainerType>;
using container_type = ContainerType;

Extrapolation( ){ m_counter = 0; }

Extrapolation( unsigned max, const ContainerType& copyable) {
set_max(max, copyable);
}
void set_max( unsigned max, const ContainerType& copyable)
{
m_counter = 0;
m_x.assign( max, copyable);
m_t.assign( max, 0);
m_max = max;
}

unsigned get_max( ) const{
return m_counter;
}



bool exists( value_type t)const{
if( m_max == 0) return false;
for( unsigned i=0; i<m_counter; i++)
if( fabs(t - m_t[i]) <1e-14)
return true;
return false;
}


template<class ContainerType0>
void extrapolate( value_type t, ContainerType0& new_x) const{
switch(m_counter)
{
case(0): dg::blas1::copy( 0, new_x);
break;
case(1): dg::blas1::copy( m_x[0], new_x);
break;
case(3): {
value_type f0 = (t-m_t[1])*(t-m_t[2])/(m_t[0]-m_t[1])/(m_t[0]-m_t[2]);
value_type f1 =-(t-m_t[0])*(t-m_t[2])/(m_t[0]-m_t[1])/(m_t[1]-m_t[2]);
value_type f2 = (t-m_t[0])*(t-m_t[1])/(m_t[2]-m_t[0])/(m_t[2]-m_t[1]);
dg::blas1::evaluate( new_x, dg::equals(), dg::PairSum(),
f0, m_x[0], f1, m_x[1], f2, m_x[2]);
break;
}
default: {
value_type f0 = (t-m_t[1])/(m_t[0]-m_t[1]);
value_type f1 = (t-m_t[0])/(m_t[1]-m_t[0]);
dg::blas1::axpby( f0, m_x[0], f1, m_x[1], new_x);
}
}
}


template<class ContainerType0>
void derive( value_type t, ContainerType0& dot_x) const{
switch(m_counter)
{
case(0): dg::blas1::copy( 0, dot_x);
break;
case(1): dg::blas1::copy( 0, dot_x);
break;
case(3): {
value_type f0 =-(-2.*t+m_t[1]+m_t[2])/(m_t[0]-m_t[1])/(m_t[0]-m_t[2]);
value_type f1 = (-2.*t+m_t[0]+m_t[2])/(m_t[0]-m_t[1])/(m_t[1]-m_t[2]);
value_type f2 =-(-2.*t+m_t[0]+m_t[1])/(m_t[2]-m_t[0])/(m_t[2]-m_t[1]);
dg::blas1::evaluate( dot_x, dg::equals(), dg::PairSum(),
f0, m_x[0], f1, m_x[1], f2, m_x[2]);
break;
}
default: {
value_type f0 = 1./(m_t[0]-m_t[1]);
value_type f1 = 1./(m_t[1]-m_t[0]);
dg::blas1::axpby( f0, m_x[0], f1, m_x[1], dot_x);
}
}
}


template<class ContainerType0>
void extrapolate( ContainerType0& new_x) const{
value_type t = m_t[0] +1.;
extrapolate( t, new_x);
}

template<class ContainerType0>
void derive( ContainerType0& dot_x) const{
derive( m_t[0], dot_x);
}


template<class ContainerType0>
void update( value_type t_new, const ContainerType0& new_entry){
if( m_max == 0) return;
for( unsigned i=0; i<m_counter; i++)
if( fabs(t_new - m_t[i]) <1e-14)
{
blas1::copy( new_entry, m_x[i]);
return;
}
if( m_counter < m_max) 
m_counter++;
std::rotate( m_x.rbegin(), m_x.rbegin()+1, m_x.rend());
std::rotate( m_t.rbegin(), m_t.rbegin()+1, m_t.rend());
m_t[0] = t_new;
blas1::copy( new_entry, m_x[0]);
}

template<class ContainerType0>
void update( const ContainerType0& new_entry){
value_type t_new = m_t[0] + 1;
update( t_new, new_entry);
}


const ContainerType& head()const{
return m_x[0];
}
ContainerType& tail(){
return m_x[m_max-1];
}
const ContainerType& tail()const{
return m_x[m_max-1];
}

private:
unsigned m_max, m_counter;
std::vector<value_type> m_t;
std::vector<ContainerType> m_x;
};


}
