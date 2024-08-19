#pragma once

namespace dg
{
namespace geo
{


template<class real_type>
struct aRealGenerator2d
{
real_type width()  const{return do_width();}
real_type height() const{return do_height();}
bool isOrthogonal() const { return do_isOrthogonal(); }


void generate(
const thrust::host_vector<real_type>& zeta1d,
const thrust::host_vector<real_type>& eta1d,
thrust::host_vector<real_type>& x,
thrust::host_vector<real_type>& y,
thrust::host_vector<real_type>& zetaX,
thrust::host_vector<real_type>& zetaY,
thrust::host_vector<real_type>& etaX,
thrust::host_vector<real_type>& etaY) const
{
unsigned size = zeta1d.size()*eta1d.size();
x.resize(size), y.resize(size);
zetaX = zetaY = etaX = etaY =x ;
do_generate( zeta1d, eta1d, x,y,zetaX,zetaY,etaX,etaY);
}


virtual aRealGenerator2d* clone() const=0;
virtual ~aRealGenerator2d(){}

protected:
aRealGenerator2d(){}
aRealGenerator2d(const aRealGenerator2d& ){}
aRealGenerator2d& operator=(const aRealGenerator2d& ){ return *this; }
private:
virtual void do_generate(
const thrust::host_vector<real_type>& zeta1d,
const thrust::host_vector<real_type>& eta1d,
thrust::host_vector<real_type>& x,
thrust::host_vector<real_type>& y,
thrust::host_vector<real_type>& zetaX,
thrust::host_vector<real_type>& zetaY,
thrust::host_vector<real_type>& etaX,
thrust::host_vector<real_type>& etaY) const = 0;
virtual real_type do_width() const =0;
virtual real_type do_height() const =0;
virtual bool do_isOrthogonal()const{return false;}


};
using aGenerator2d = dg::geo::aRealGenerator2d<double>;

}
}
