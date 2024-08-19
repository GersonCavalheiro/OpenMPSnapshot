#pragma once

#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include "dg/algorithm.h"
#include "functors.h"

namespace dg{
namespace mat{



template<class real_type>
struct FunctionalButcherTableau{
using value_type = real_type;
using function_type = std::function<value_type(value_type)>;
FunctionalButcherTableau() = default;

FunctionalButcherTableau(unsigned s, unsigned order,
const function_type* a , const function_type* b , const real_type* c):
m_a(a, a+s*s), m_b(b, b+s), m_c(c, c+s), m_bt(b,b+s), m_q(order), m_p(order), m_s(s){}

FunctionalButcherTableau(unsigned s, unsigned embedded_order, unsigned order,
const function_type* a, const function_type* b, const function_type* bt, const real_type* c):
m_a(a, a+s*s), m_b(b,b+s), m_c(c,c+s), m_bt(bt, bt+s), m_q(order), m_p(embedded_order), m_s(s), m_embedded(true){}


function_type a( unsigned i, unsigned j) const {
return m_a(i,j);
}

real_type c( unsigned i) const {
return m_c[i];
}

function_type b( unsigned j) const {
return m_b[j];
}

function_type bt( unsigned j) const {
return m_bt[j];
}
unsigned num_stages() const  {
return m_s;
}
unsigned order() const {
return m_q;
}
unsigned embedded_order() const{
return m_p;
}
bool isEmbedded()const{
return m_embedded;
}
bool isImplicit()const{
for( unsigned i=0; i<m_s; i++)
for( unsigned j=i; j<m_s; j++)
if( a(i,j) != 0)
return true;
return false;
}
private:
dg::Operator<function_type> m_a;
std::vector<function_type> m_b;
std::vector<real_type> m_c;
std::vector<function_type> m_bt;
unsigned m_q, m_p, m_s;
bool m_embedded = false;
};

namespace func_tableau{

template<class real_type>
FunctionalButcherTableau<real_type> explicit_euler_1_1( )
{
auto zero = [&](real_type){return 0;};
using function_type = std::function<real_type(real_type)>;
function_type a[1] = {zero};
function_type b[1] = {[&](real_type x){return dg::mat::phi1(x);}};
real_type c[1] = {0.};
return FunctionalButcherTableau<real_type>( 1,1, a,b,c);
}
template<class real_type>
FunctionalButcherTableau<real_type> midpoint_2_2()
{
auto zero = [&](real_type){return 0;};
using function_type = std::function<real_type(real_type)>;
function_type a[4] = {  zero, zero,
[&](real_type x){return 0.5*dg::mat::phi1(x/2.);},
zero};
function_type b[2] = { zero,
[&](real_type x){return dg::mat::phi1(x);}};
real_type c[2] = {0, 0.5};
return FunctionalButcherTableau<real_type>( 2,2, a,b,c);
}
template<class real_type>
FunctionalButcherTableau<real_type> classic_4_4()
{
auto zero = [&](real_type){return 0;};
using function_type = std::function<real_type(real_type)>;
function_type a[16] = {
zero,zero,zero,zero,
[&](real_type x){return 0.5*dg::mat::phi1(x/2.);}, zero,zero,zero,
zero, [&](real_type x){return 0.5*dg::mat::phi1(x/2.);}, zero,zero,
[&](real_type x){return 0.5*dg::mat::phi1(x/2.)*(exp(x/2.)-1);},zero,[&](real_type x){return dg::mat::phi1(x/2.);},zero
};
function_type b[4] = {
[&](real_type x){return dg::mat::phi1(x)-3.*dg::mat::phi2(x)+4.*dg::mat::phi3(x);},
[&](real_type x){return 2.*dg::mat::phi2(x)-4.*dg::mat::phi3(x);},
[&](real_type x){return 2.*dg::mat::phi2(x)-4.*dg::mat::phi3(x);},
[&](real_type x){return -dg::mat::phi2(x)+4.*dg::mat::phi3(x);}
};
real_type c[4] = {0, 0.5, 0.5, 1.};
return FunctionalButcherTableau<real_type>( 4,4, a,b,c);
}
template<class real_type>
FunctionalButcherTableau<real_type> hochbruck_3_3_4()
{
auto zero = [&](real_type){return 0;};
using function_type = std::function<real_type(real_type)>;
function_type a[9] = {
zero,zero,zero,
[&](real_type x){return 0.5*dg::mat::phi1(x/2.);}, zero,zero,
zero, [&](real_type x){return dg::mat::phi1(x);}, zero
};
function_type b[3] = {
[&](real_type x){return dg::mat::phi1(x)-14.*dg::mat::phi3(x)+36.*dg::mat::phi4(x);},
[&](real_type x){return 16.*dg::mat::phi3(x)-48.*dg::mat::phi4(x);},
[&](real_type x){return -2.*dg::mat::phi3(x)+12.*dg::mat::phi4(x);},
};
function_type bt[3] = {
[&](real_type x){return dg::mat::phi1(x)-14.*dg::mat::phi3(x);},
[&](real_type x){return 16.*dg::mat::phi3(x);},
[&](real_type x){return -2.*dg::mat::phi3(x);}
};
real_type c[3] = {0, 0.5, 1.};
return FunctionalButcherTableau<real_type>( 3,3,4, a,b,bt,c);
}


}


enum func_tableau_identifier{
EXPLICIT_EULER_1_1, 
MIDPOINT_2_2, 
CLASSIC_4_4,
HOCHBRUCK_3_3_4
};

namespace create{

static std::unordered_map<std::string, enum func_tableau_identifier> str2id{
{"Euler", EXPLICIT_EULER_1_1},
{"Midpoint-2-2", MIDPOINT_2_2},
{"Runge-Kutta-4-4", CLASSIC_4_4},
{"Hochbruck-3-3-4", HOCHBRUCK_3_3_4},
};
static inline enum func_tableau_identifier str2func_tableau( std::string name)
{
if( str2id.find(name) == str2id.end())
throw dg::Error(dg::Message(_ping_)<<"Tableau "<<name<<" not found!");
else
return str2id[name];
}
static inline std::string func_tableau2str( enum func_tableau_identifier id)
{
for( auto name: str2id)
{
if( name.second == id)
return name.first;
}
throw dg::Error(dg::Message(_ping_)<<"Tableau conversion failed!");
}

template<class real_type>
FunctionalButcherTableau<real_type> func_tableau( enum func_tableau_identifier id)
{
switch(id){
case EXPLICIT_EULER_1_1:
return func_tableau::explicit_euler_1_1<real_type>();
case MIDPOINT_2_2:
return func_tableau::midpoint_2_2<real_type>();
case CLASSIC_4_4:
return func_tableau::classic_4_4<real_type>();
case HOCHBRUCK_3_3_4:
return func_tableau::hochbruck_3_3_4<real_type>();
}
return FunctionalButcherTableau<real_type>(); 
}


template<class real_type>
FunctionalButcherTableau<real_type> func_tableau( std::string name)
{
return func_tableau<real_type>( str2func_tableau(name));
}

}





template<class real_type>
struct ConvertsToFunctionalButcherTableau
{
using value_type = real_type;
ConvertsToFunctionalButcherTableau( FunctionalButcherTableau<real_type> tableau): m_t(tableau){}


ConvertsToFunctionalButcherTableau( enum tableau_identifier id):m_t( create::func_tableau<real_type>(id)){}

ConvertsToFunctionalButcherTableau( std::string name):m_t( create::func_tableau<real_type>(name)){}
ConvertsToFunctionalButcherTableau( const char* name):m_t( create::func_tableau<real_type>(std::string(name))){}
operator FunctionalButcherTableau<real_type>( )const{
return m_t;
}
private:
FunctionalButcherTableau<real_type> m_t;
};

}
}
