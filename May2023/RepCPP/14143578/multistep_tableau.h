#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace dg{

template<class real_type>
struct MultistepTableau
{
using value_type = real_type;
MultistepTableau(){}

MultistepTableau( unsigned steps, unsigned order, const
std::vector<real_type>& a_v, const std::vector<real_type>& b_v,
const std::vector<real_type>& c_v): m_steps(steps), m_order(order),
m_a(a_v), m_b(b_v), m_c(c_v){
if( m_c.empty())
m_c.assign( steps+1, 0);
if( m_b.empty())
m_b.assign( steps, 0);
}


real_type a( unsigned i){ return m_a[i];}

real_type ex( unsigned i){ return m_b[i];}

real_type im( unsigned i){ return m_c[i];}
unsigned steps() const  {
return m_steps;
}
unsigned order() const {
return m_order;
}
bool isExplicit() const{
for( unsigned i=0; i<m_steps; i++)
if( m_b[i]!=0)
return true;
return false;
}
bool isImplicit() const{
for( unsigned i=0; i<m_steps+1; i++)
if( m_c[i]!=0)
return true;
return false;
}
private:
unsigned m_steps, m_order;
std::vector<real_type> m_a, m_b, m_c;
};

namespace tableau
{
template<class real_type>
MultistepTableau<real_type> imex_euler_1_1()
{
unsigned steps = 1, order = 1;
std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
a[0] = b[0] = c[0] = 1;
return MultistepTableau<real_type>( steps, order, a, b, c);
}
template<class real_type>
MultistepTableau<real_type> imex_adams_2_2()
{
unsigned steps = 2, order = 2;
std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
a[0] = 1.;
b[0] =  3./2.;
b[1] = -1./2.;
c[0] = 9./16.;
c[1] = 3./8.;
c[2] = 1./16.;
return MultistepTableau<real_type>( steps, order, a, b, c);
}
template<class real_type>
MultistepTableau<real_type> imex_adams_3_3()
{
unsigned steps = 3, order = 3;
std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
a[0] = 1.;
b[0] =  23./12.;
b[1] = -4./3.;
b[2] = 5./12.;
c[0] =   4661./10000.;
c[1] =  15551./30000.;
c[2] =     1949/30000;
c[3] = -1483./30000.;
return MultistepTableau<real_type>( steps, order, a, b, c);
}
template<class real_type>
MultistepTableau<real_type> imex_koto_2_2()
{
unsigned steps = 2, order = 2;
std::vector<real_type> am(steps,0), bm(steps, 0), cm(steps+1,0);
std::vector<real_type> ap(steps+1,0), bp(steps+1, 0), cp(steps+1,0);
real_type a = 20., b = 20.;
ap[0] = a;
ap[1] = 1-2.*a;
ap[2] = a-1;
cp[0] =  b;
cp[1] = 0.5+a-2*b;
cp[2] = 0.5-a+b;
bp[1] = 0.5+a;
bp[2] = 0.5-a;
am[0] = -ap[1]/a, am[1] = -ap[2]/a;
bm[0] = bp[1]/a, bm[1] = bp[2]/a;
cm[0] = cp[0]/a, cm[1] = cp[1]/a, cm[2] = cp[2]/a;
return MultistepTableau<real_type>( steps, order, am, bm, cm);
}

template<class real_type>
MultistepTableau<real_type> imex_bdf(unsigned steps)
{
unsigned order = steps;
std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
switch( steps)
{
case( 2):
a[0] =  4./3.;  b[0] = 4./3.;
a[1] = -1./3.;  b[1] = -2./3.;
c[0] = 2./3.;
break;
case(3):
a[0] =  18./11.;    b[0] =  18./11.;
a[1] = -9./11.;     b[1] = -18./11.;
a[2] = 2./11.;      b[2] = 6./11.;
c[0] = 6./11.;
break;
case(4):
a[0] =  48./25.;    b[0] =  48./25.;
a[1] = -36./25.;    b[1] = -72./25.;
a[2] =  16./25.;    b[2] =  48./25.;
a[3] = - 3./25.;    b[3] = -12./25.;
c[0] = 12./25.;
break;
case(5):
a[0] = 300./137.;    b[0] = 300./137.;
a[1] = -300./137.;   b[1] = -600./137.;
a[2] = 200./137.;    b[2] = 600./137.;
a[3] = -75./137.;    b[3] = -300./137.;
a[4] = 12./137.;     b[4] = 60./137.;
c[0] = 60./137.;
break;
case (6):
a = {360./147.,-450./147.,400./147.,-225./147.,72./147.,-10./147.};
b = {360./147.,-900./147.,1200./147.,-900./147.,360./147.,-60./147.};
c[0] = 60./147.;
break;
}
return MultistepTableau<real_type>( steps, order, a, b, c);
}

template<class real_type>
MultistepTableau<real_type> imex_tvb(unsigned steps)
{
unsigned order = steps;
std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
switch( steps)
{
case(3):
a[0] =  3909./2048.;     b[0] =  18463./12288.;
a[1] = -1367./1024.;     b[1] = -1271./768.;
a[2] =  873./2048.;      b[2] = 8233./12288.;
c[0] =  1089./2048.;
c[1] = -1139./12288.;
c[2] = -367./6144.;
c[3] =  1699./12288.;
break;
case(4):
a[0] =  21531./8192.;     b[0] =  13261./8192.;
a[1] = -22753./8192.;     b[1] = -75029./24576.;
a[2] =  12245./8192.;     b[2] =  54799./24576.;
a[3] = -2831./8192. ;     b[3] = -15245./24576.;
c[0] =  4207./8192.;
c[1] = -3567./8192.;
c[2] =  697./24576.;
c[3] = 4315./24576.;
c[4] = -41./384.;
break;
case(5):
a[0] =  13553./4096.;     b[0] = 10306951./5898240.;
a[1] = -38121./8192.;     b[1] = -13656497./2949120.;
a[2] =  7315./2048.;      b[2] = 1249949./245760.;
a[3] = -6161/4096. ;      b[3] = -7937687./2949120.;
a[4] = 2269./8192.;       b[4] = 3387361./5898240.;
c[0] =  4007./8192.;
c[1] =  -4118249./5898240.;
c[2] =  768703./2949120.;
c[3] = 47849./245760.;
c[4] = -725087./2949120.;
c[5] = 502321./5898240.;
break;
}
return MultistepTableau<real_type>( steps, order, a, b, c);
}

template<class real_type>
MultistepTableau<real_type> ab(unsigned order)
{
unsigned steps = order;
std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
a[0]= 1.;
switch (order){
case 1: b = {1}; break;
case 2: b = {1.5, -0.5}; break;
case 3: b = { 23./12., -4./3., 5./12.}; break;
case 4: b = {55./24., -59./24., 37./24., -3./8.}; break;
case 5: b = { 1901./720., -1387./360., 109./30., -637./360., 251./720.}; break;
default: throw dg::Error(dg::Message()<<"Order "<<order<<" not implemented in AdamsBashforth!");
}
return MultistepTableau<real_type>( steps, order, a, b, c);
}

template<class real_type>
MultistepTableau<real_type> tvb(unsigned steps)
{
unsigned order = steps;
std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
switch (steps){
case 1:
a = {1.};
b = {1.}; break;
case 2:
a = {4./3., -1./3.};
b = {4./3., -2./3.}; break; 
case 3: 
a[0] =  1.908535476882378;  b[0] =  1.502575553858997;
a[1] = -1.334951446162515;  b[1] = -1.654746338401493;
a[2] = 0.426415969280137;   b[2] = 0.670051276940255;
break;
case 4: 
a[0] = 2.628241000683208;   b[0] = 1.618795874276609;
a[1] = -2.777506277494861;  b[1] = -3.052866947601049;
a[2] = 1.494730011212510;   b[2] = 2.229909318681302;
a[3] = -0.345464734400857;  b[3] = -0.620278703629274;
break;
case 5: 
a[0] = 3.308891758551210;   b[0] = 1.747442076919292;
a[1] = -4.653490937946655;  b[1] = -4.630745565661800;
a[2] = 3.571762873789854;   b[2] = 5.086056171401077;
a[3] = -1.504199914126327;  b[3] = -2.691494591660196;
a[4] = 0.277036219731918;   b[4] = 0.574321855183372;
break;
case 6: 
a[0] = 4.113382628475685;   b[0] = 1.825457674048542;
a[1] = -7.345730559324184;  b[1] = -6.414174588309508;
a[2] = 7.393648314992094;   b[2] = 9.591671249204753;
a[3] = -4.455158576186636;  b[3] = -7.583521888026967;
a[4] = 1.523638279938299;   b[4] = 3.147082225022105;
a[5] = -0.229780087895259;  b[5] = -0.544771649561925;
break;
default: throw dg::Error(dg::Message()<<"Order "<<steps<<" not implemented in TVB scheme!");
}
return MultistepTableau<real_type>( steps, order, a, b, c);
}
template<class real_type>
MultistepTableau<real_type> ssp(unsigned steps)
{
std::vector<real_type> a(steps,0), b(steps, 0), c(steps+1,0);
unsigned order = 0;
switch (steps){
case 1: order = 1;
a = {1.};
b = {1.}; break;
case 2: order = 2;
a = {4./5., 1./5.};
b = {8./5., -2./5.}; break; 
case 3: order = 2;
a = { 3./4., 0., 1./4.};
b = { 3./2., 0., 0. }; break; 
case 4: order = 2;
a = {8./9., 0., 0., 1./9.};
b = {4./3., 0., 0., 0.}; break; 
case 5: order = 3;
a = {25./32., 0., 0., 0., 7./32.};
b = {25./16.,0.,0.,0.,5./16.}; break; 
case 6: order = 3;
a = {108./125.,0.,0.,0.,0.,17./125.};
b = {36./25.,0.,0.,0.,0.,6./25.}; break; 
default: throw dg::Error(dg::Message()<<"Stage "<<steps<<" not implemented in SSP scheme!");
}
return MultistepTableau<real_type>( steps, order, a, b, c);
}

}

enum multistep_identifier{
IMEX_EULER_1_1,
IMEX_ADAMS_2_2,
IMEX_ADAMS_3_3,
IMEX_KOTO_2_2,
IMEX_BDF_2_2,
IMEX_BDF_3_3,
IMEX_BDF_4_4,
IMEX_BDF_5_5,
IMEX_BDF_6_6,
IMEX_TVB_3_3,
IMEX_TVB_4_4,
IMEX_TVB_5_5,
AB_1_1,
AB_2_2,
AB_3_3,
AB_4_4,
AB_5_5,
eBDF_1_1,
eBDF_2_2,
eBDF_3_3,
eBDF_4_4,
eBDF_5_5,
eBDF_6_6,
TVB_1_1,
TVB_2_2,
TVB_3_3,
TVB_4_4,
TVB_5_5,
TVB_6_6,
SSP_1_1,
SSP_2_2,
SSP_3_2,
SSP_4_2,
SSP_5_3,
SSP_6_3,
BDF_1_1,
BDF_2_2,
BDF_3_3,
BDF_4_4,
BDF_5_5,
BDF_6_6,
};

namespace create{

static std::unordered_map<std::string, enum multistep_identifier> str2lmsid{
{"Euler", IMEX_EULER_1_1},
{"Euler-1-1", IMEX_EULER_1_1},
{"ImEx-Adams-2-2", IMEX_ADAMS_2_2},
{"ImEx-Adams-3-3", IMEX_ADAMS_3_3},
{"ImEx-Koto-2-2", IMEX_KOTO_2_2},
{"ImEx-BDF-2-2", IMEX_BDF_2_2},
{"ImEx-BDF-3-3", IMEX_BDF_3_3},
{"Karniadakis",  IMEX_BDF_3_3},
{"ImEx-BDF-4-4", IMEX_BDF_4_4},
{"ImEx-BDF-5-5", IMEX_BDF_5_5},
{"ImEx-BDF-6-6", IMEX_BDF_6_6},
{"ImEx-TVB-3-3", IMEX_TVB_3_3},
{"ImEx-TVB-4-4", IMEX_TVB_4_4},
{"ImEx-TVB-5-5", IMEX_TVB_5_5},
{"AB-1-1", AB_1_1},
{"AB-2-2", AB_2_2},
{"AB-3-3", AB_3_3},
{"AB-4-4", AB_4_4},
{"AB-5-5", AB_5_5},
{"eBDF-1-1", eBDF_1_1},
{"eBDF-2-2", eBDF_2_2},
{"eBDF-3-3", eBDF_3_3},
{"eBDF-4-4", eBDF_4_4},
{"eBDF-5-5", eBDF_5_5},
{"eBDF-6-6", eBDF_6_6},
{"TVB-1-1", TVB_1_1},
{"TVB-2-2", TVB_2_2},
{"TVB-3-3", TVB_3_3},
{"TVB-4-4", TVB_4_4},
{"TVB-5-5", TVB_5_5},
{"TVB-6-6", TVB_6_6},
{"SSP-1-1", SSP_1_1},
{"SSP-2-2", SSP_2_2},
{"SSP-3-2", SSP_3_2},
{"SSP-4-2", SSP_4_2},
{"SSP-5-3", SSP_5_3},
{"SSP-6-3", SSP_6_3},
{"BDF-1-1", BDF_1_1},
{"BDF-2-2", BDF_2_2},
{"BDF-3-3", BDF_3_3},
{"BDF-4-4", BDF_4_4},
{"BDF-5-5", BDF_5_5},
{"BDF-6-6", BDF_6_6},
};
static inline enum multistep_identifier str2lmstableau( std::string name)
{
if( str2lmsid.find(name) == str2lmsid.end())
throw dg::Error(dg::Message(_ping_)<<"Multistep coefficients for "<<name<<" not found!");
else
return str2lmsid[name];
}
static inline std::string lmstableau2str( enum multistep_identifier id)
{
for( auto name: str2lmsid)
{
if( name.second == id)
return name.first;
}
throw dg::Error(dg::Message(_ping_)<<"Tableau conversion failed!");
}

template<class real_type>
MultistepTableau<real_type> lmstableau( enum multistep_identifier id)
{
switch(id){
case IMEX_EULER_1_1:
return dg::tableau::imex_euler_1_1<real_type>();
case IMEX_ADAMS_2_2:
return dg::tableau::imex_adams_2_2<real_type>();
case IMEX_ADAMS_3_3:
return dg::tableau::imex_adams_3_3<real_type>();
case IMEX_KOTO_2_2:
return dg::tableau::imex_koto_2_2<real_type>();
case IMEX_BDF_2_2:
return dg::tableau::imex_bdf<real_type>(2);
case IMEX_BDF_3_3:
return dg::tableau::imex_bdf<real_type>(3);
case IMEX_BDF_4_4:
return dg::tableau::imex_bdf<real_type>(4);
case IMEX_BDF_5_5:
return dg::tableau::imex_bdf<real_type>(5);
case IMEX_BDF_6_6:
return dg::tableau::imex_bdf<real_type>(6);
case IMEX_TVB_3_3:
return dg::tableau::imex_tvb<real_type>(3);
case IMEX_TVB_4_4:
return dg::tableau::imex_tvb<real_type>(4);
case IMEX_TVB_5_5:
return dg::tableau::imex_tvb<real_type>(5);
case AB_1_1:
return dg::tableau::ab<real_type>(1);
case AB_2_2:
return dg::tableau::ab<real_type>(2);
case AB_3_3:
return dg::tableau::ab<real_type>(3);
case AB_4_4:
return dg::tableau::ab<real_type>(4);
case AB_5_5:
return dg::tableau::ab<real_type>(5);
case eBDF_1_1:
return dg::tableau::imex_euler_1_1<real_type>();
case eBDF_2_2:
return dg::tableau::imex_bdf<real_type>(2);
case eBDF_3_3:
return dg::tableau::imex_bdf<real_type>(3);
case eBDF_4_4:
return dg::tableau::imex_bdf<real_type>(4);
case eBDF_5_5:
return dg::tableau::imex_bdf<real_type>(5);
case eBDF_6_6:
return dg::tableau::imex_bdf<real_type>(6);
case TVB_1_1:
return dg::tableau::imex_euler_1_1<real_type>();
case TVB_2_2:
return dg::tableau::tvb<real_type>(2);
case TVB_3_3:
return dg::tableau::tvb<real_type>(3);
case TVB_4_4:
return dg::tableau::tvb<real_type>(4);
case TVB_5_5:
return dg::tableau::tvb<real_type>(5);
case TVB_6_6:
return dg::tableau::tvb<real_type>(6);
case SSP_1_1:
return dg::tableau::ssp<real_type>(1);
case SSP_2_2:
return dg::tableau::ssp<real_type>(2);
case SSP_3_2:
return dg::tableau::ssp<real_type>(3);
case SSP_4_2:
return dg::tableau::ssp<real_type>(4);
case SSP_5_3:
return dg::tableau::ssp<real_type>(5);
case SSP_6_3:
return dg::tableau::ssp<real_type>(6);
case BDF_1_1:
return dg::tableau::imex_euler_1_1<real_type>();
case BDF_2_2:
return dg::tableau::imex_bdf<real_type>(2);
case BDF_3_3:
return dg::tableau::imex_bdf<real_type>(3);
case BDF_4_4:
return dg::tableau::imex_bdf<real_type>(4);
case BDF_5_5:
return dg::tableau::imex_bdf<real_type>(5);
case BDF_6_6:
return dg::tableau::imex_bdf<real_type>(6);
}
return MultistepTableau<real_type>(); 
}


template<class real_type>
MultistepTableau<real_type> lmstableau( std::string name)
{
return lmstableau<real_type>( str2lmstableau(name));
}

}







template<class real_type>
struct ConvertsToMultistepTableau
{
using value_type = real_type;
ConvertsToMultistepTableau( MultistepTableau<real_type> tableau): m_t(tableau){}


ConvertsToMultistepTableau( enum tableau_identifier id):m_t( dg::create::lmstableau<real_type>(id)){}

ConvertsToMultistepTableau( std::string name):m_t(
dg::create::lmstableau<real_type>(name)){}

ConvertsToMultistepTableau( const char* name):m_t(
dg::create::lmstableau<real_type>(std::string(name))){}
operator MultistepTableau<real_type>( )const{
return m_t;
}
private:
MultistepTableau<real_type> m_t;
};

}
