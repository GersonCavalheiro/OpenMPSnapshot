#pragma once
#include <string>
#include <vector>
#ifdef JSONCPP_VERSION_STRING
#include <dg/file/json_utilities.h>
#endif

namespace dg
{
namespace geo
{
namespace solovev
{


struct Parameters
{
double A, 
R_0, 
pp, 
pi, 
a,  
elongation, 
triangularity; 
std::vector<double> c;  
std::string description;
#ifdef JSONCPP_VERSION_STRING

Parameters( const dg::file::WrappedJsonValue& js) {
A   = js.get("A", 0).asDouble();
pp  = js.get("PP", 1).asDouble();
pi  = js.get("PI", 1).asDouble();
c.resize(12);
for (unsigned i=0;i<12;i++)
c[i] = js["c"].get(i,0.).asDouble();

R_0  =          js.get( "R_0", 0.).asDouble();
a        = R_0* js.get( "inverseaspectratio", 0.).asDouble();
elongation=     js.get( "elongation", 1.).asDouble();
triangularity=  js.get( "triangularity", 0.).asDouble();
try{
description = js.get( "description", "standardX").asString();
} catch ( std::exception& err)
{
if( isToroidal())
description = "none";
else if( !hasXpoint())
description = "standardO";
else
description = "standardX";
}
}

Json::Value dump( ) const
{
Json::Value js;
js["A"] = A;
js["PP"] = pp;
js["PI"] = pi;
for (unsigned i=0;i<12;i++) js["c"][i] = c[i];
js["R_0"] = R_0;
js["inverseaspectratio"] = a/R_0;
js["elongation"] = elongation;
js["triangularity"] = triangularity;
js[ "equilibrium"] = "solovev";
js[ "description"] = description;
return js;
}
#endif 

bool hasXpoint( ) const{
bool Xpoint = false;
for( int i=7; i<12; i++)
if( fabs(c[i]) >= 1e-10)
Xpoint = true;
return Xpoint;
}

bool isToroidal() const{
if( pp == 0)
return true;
return false;
}
void display( std::ostream& os = std::cout ) const
{
os << "Solovev Geometrical parameters are: \n"
<<" A               = "<<A<<"\n"
<<" Prefactor Psi   = "<<pp<<"\n"
<<" Prefactor I     = "<<pi<<"\n";
for( unsigned i=0; i<12; i++)
os<<" c"<<i+1<<"\t\t = "<<c[i]<<"\n";

os  <<" R0            = "<<R_0<<"\n"
<<" a             = "<<a<<"\n"
<<" epsilon_a     = "<<a/R_0<<"\n"
<<" description   = "<<description<<"\n"
<<" elongation    = "<<elongation<<"\n"
<<" triangularity = "<<triangularity<<"\n";
os << std::flush;

}
};
} 
} 
} 
