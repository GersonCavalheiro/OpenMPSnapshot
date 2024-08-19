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
namespace polynomial
{


struct Parameters
{
double R_0, 
pp, 
pi, 
a,  
elongation, 
triangularity; 
unsigned M, 
N; 
std::vector<double> c;  
std::string description;
#ifdef JSONCPP_VERSION_STRING

Parameters( const dg::file::WrappedJsonValue& js) {
pp  = js.get( "PP", 1).asDouble();
pi  = js.get( "PI", 1).asDouble();
M = js.get( "M", 1).asUInt();
N = js.get( "N", 1).asUInt();
c.resize(M*N);
for (unsigned i=0;i<M*N;i++)
c[i] = js["c"].get(i,0.).asDouble();

R_0  = js.get( "R_0", 0.).asDouble();
a  = R_0*js.get( "inverseaspectratio", 0.).asDouble();
elongation=js.get( "elongation", 1.).asDouble();
triangularity=js.get( "triangularity", 0.).asDouble();
description = js.get( "description", "standardX").asString();
}

Json::Value dump( ) const
{
Json::Value js;
js["M"] = M;
js["N"] = N;
js["PP"] = pp;
js["PI"] = pi;
for (unsigned i=0;i<N*N;i++) js["c"][i] = c[i];
js["R_0"] = R_0;
js["inverseaspectratio"] = a/R_0;
js["elongation"] = elongation;
js["triangularity"] = triangularity;
js[ "equilibrium"] = "polynomial";
js[ "description"] = description;
return js;
}
#endif 

bool isToroidal() const{
if( pp == 0)
return true;
return false;
}
void display( std::ostream& os = std::cout ) const
{
os << "Polynomial Geometrical parameters are: \n"
<<" Prefactor Psi   = "<<pp<<"\n"
<<" Prefactor I     = "<<pi<<"\n"
<<" number in R     = "<<M<<"\n"
<<" number in Z     = "<<N<<"\n";
for( unsigned i=0; i<M*N; i++)
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
