#pragma once
#include <functional>
#include "blas1.h"
#include "backend/tensor_traits.h"


namespace dg
{


enum class to
{
exact, 
at_least 
};


inline static std::string to2str( enum to mode)
{
std::string s;
switch(mode)
{
case(dg::to::exact): s = "exact"; break;
case(dg::to::at_least): s = "at_least"; break;
default: s = "Not specified!!";
}
return s;
}


template<class ContainerType>
struct aTimeloop
{
using value_type = dg::get_value_type<ContainerType>;
using container_type = ContainerType;


void integrate( value_type t0, const ContainerType& u0,
value_type t1, ContainerType& u1)
{
if( t0 == t1)
{
dg::blas1::copy( u0, u1);
return;
}
value_type time = t0;
try{
do_integrate( time, u0, t1, u1, dg::to::exact);
}
catch ( dg::Error& err)
{
err.append_line( dg::Message(_ping_) << "Error in aTimeloop::integrate at time "<<time<<" with t0 "<<t0<<" and t1 "<<t1);
throw;
}
}


void integrate( value_type& t0, const ContainerType& u0,
value_type t1, ContainerType& u1, enum to mode )
{
if( t0 == t1)
{
dg::blas1::copy( u0, u1);
return;
}

value_type t_begin = t0;
try{
do_integrate( t0, u0, t1, u1, mode);
}
catch ( dg::Error& err)
{
err.append_line( dg::Message(_ping_) << "Error in aTimeloop::integrate at time "<<t0<<" with t0 "<<t_begin<<" and t1 "<<t1 << " and mode "<<to2str(mode));
throw;
}
}


value_type get_dt() const { return do_dt(); }



virtual aTimeloop* clone() const=0;

virtual ~aTimeloop(){}
protected:
aTimeloop(){}
aTimeloop(const aTimeloop& ){}
aTimeloop& operator=(const aTimeloop& ){ return *this; }
private:
virtual void do_integrate(value_type& t0, const container_type& u0,
value_type t1, container_type& u1, enum to mode) const = 0;
virtual value_type do_dt() const = 0;


};


}
