#pragma once

namespace dg
{
namespace geo
{


template<class real_type>
struct aRealGeneratorX2d
{
real_type zeta0(real_type fx) const{return do_zeta0(fx);}
real_type zeta1(real_type fx) const{return do_zeta1(fx);}
real_type eta0(real_type fy) const{return do_eta0(fy);}
real_type eta1(real_type fy) const{return do_eta1(fy);}
bool isOrthogonal() const { return do_isOrthogonal(); }


void generate(
const thrust::host_vector<real_type>& zeta1d,
const thrust::host_vector<real_type>& eta1d,
unsigned nodeX0, unsigned nodeX1,
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
do_generate( zeta1d, eta1d,nodeX0,nodeX1,x,y,zetaX,zetaY,etaX,etaY);
}


virtual aRealGeneratorX2d* clone() const=0;
virtual ~aRealGeneratorX2d(){}

protected:
aRealGeneratorX2d(){}
aRealGeneratorX2d(const aRealGeneratorX2d& src){}
aRealGeneratorX2d& operator=(const aRealGeneratorX2d& src){
return *this;
}
private:
virtual void do_generate(
const thrust::host_vector<real_type>& zeta1d,
const thrust::host_vector<real_type>& eta1d,
unsigned nodeX0, unsigned nodeX1,
thrust::host_vector<real_type>& x,
thrust::host_vector<real_type>& y,
thrust::host_vector<real_type>& zetaX,
thrust::host_vector<real_type>& zetaY,
thrust::host_vector<real_type>& etaX,
thrust::host_vector<real_type>& etaY) const = 0;
virtual bool do_isOrthogonal()const{return false;}
virtual real_type do_zeta0(real_type fx) const=0;
virtual real_type do_zeta1(real_type fx) const=0;
virtual real_type do_eta0(real_type fy) const=0;
virtual real_type do_eta1(real_type fy) const=0;


};
using aGeneratorX2d = dg::geo::aRealGeneratorX2d<double>;

}
}
