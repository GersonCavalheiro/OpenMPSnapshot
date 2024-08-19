#pragma once
#include <string>
#include "backend/exceptions.h"



namespace dg
{

enum bc{
PER = 0, 
DIR = 1, 
DIR_NEU = 2, 
NEU_DIR = 3, 
NEU = 4 
};



static inline std::string bc2str( bc bcx)
{
std::string s;
switch(bcx)
{
case(dg::PER): s = "PERIODIC"; break;
case(dg::DIR): s = "DIRICHLET"; break;
case(dg::NEU): s = "NEUMANN"; break;
case(dg::DIR_NEU): s = "DIR_NEU"; break;
case(dg::NEU_DIR): s = "NEU_DIR"; break;
default: s = "Not specified!!";
}
return s;
}


static inline bc str2bc( std::string s)
{
if( s=="PER"||s=="per"||s=="periodic"||s=="Periodic" || s == "PERIODIC")
return PER;
if( s=="DIR"||s=="dir"||s=="dirichlet"||s=="Dirichlet" || s == "DIRICHLET")
return DIR;
if( s=="NEU"||s=="neu"||s=="neumann"||s=="Neumann" || s=="NEUMANN")
return NEU;
if( s=="NEU_DIR"||s=="neu_dir" )
return NEU_DIR;
if( s=="DIR_NEU"||s=="dir_neu" )
return DIR_NEU;
throw std::runtime_error( "Boundary condition '"+s+"' not recognized!");
}


static inline bc inverse( bc bound)
{
if( bound == DIR) return NEU;
if( bound == NEU) return DIR;
if( bound == DIR_NEU) return NEU_DIR;
if( bound == NEU_DIR) return DIR_NEU;
return PER;
}

enum direction{
forward, 
backward, 
centered 
};



static inline direction str2direction( std::string s)
{
if( "forward" == s)
return forward;
if( "backward" == s)
return backward;
if( "centered" == s)
return centered;
throw std::runtime_error( "Direction '"+s+"' not recognized!");
}

static inline std::string direction2str( enum direction dir)
{
std::string s;
switch(dir)
{
case(dg::forward): s = "forward"; break;
case(dg::backward): s = "backward"; break;
case(dg::centered): s = "centered"; break;
default: s = "Not specified!!";
}
return s;
}


static inline direction inverse( direction dir)
{
if( dir == forward) return backward;
if( dir == backward) return forward;
return centered;
}

enum space{
lspace, 
xspace 
};

enum class coo2d : char
{
x = 'x', 
y = 'y', 
};
enum class coo3d : char
{
x = 'x', 
y = 'y', 
z = 'z', 
xy = 'a', 
yz = 'b', 
xz = 'c', 
};

}
