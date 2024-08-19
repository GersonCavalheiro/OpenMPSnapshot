#pragma once

#include "dg/algorithm.h"
#include "generator.h"
#include "utilities.h"
#include "adaption.h"


namespace dg
{
namespace geo
{
namespace orthogonal
{

namespace detail
{

struct Fpsi
{

Fpsi( const CylindricalFunctorsLvl1& psi, const CylindricalSymmTensorLvl1& chi, double x0, double y0, int firstline):
psip_(psi), fieldRZYTconf_(psi, x0, y0, chi),fieldRZYTequl_(psi, x0, y0, chi), fieldRZtau_(psi, chi)
{
X_init = x0, Y_init = y0;
while( fabs( psi.dfx()(X_init, Y_init)) <= 1e-10 && fabs( psi.dfy()( X_init, Y_init)) <= 1e-10)
X_init +=  1.;
m_firstline = firstline;
}
void find_initial( double psi, double& R_0, double& Z_0)
{
unsigned N = 50;
std::array<double, 2> begin2d{ {0,0} }, end2d(begin2d), end2d_old(begin2d);
begin2d[0] = end2d[0] = end2d_old[0] = X_init;
begin2d[1] = end2d[1] = end2d_old[1] = Y_init;
double eps = 1e10, eps_old = 2e10;
while( (eps < eps_old || eps > 1e-7) && eps > 1e-14)
{
eps_old = eps; end2d_old = end2d;
N*=2;
double psi0 = psip_.f()(X_init, Y_init);
using Vec = std::array<double,2>;
dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>(
"Feagin-17-8-10", {0,0}), fieldRZtau_);
odeint.integrate_steps( psi0, begin2d, psi, end2d, N);
eps = sqrt( (end2d[0]-end2d_old[0])*(end2d[0]-end2d_old[0]) +
(end2d[1]-end2d_old[1])*(end2d[1]-end2d_old[1]));
}
X_init = R_0 = end2d_old[0], Y_init = Z_0 = end2d_old[1];
}

double construct_f( double psi, double& R_0, double& Z_0)
{
find_initial( psi, R_0, Z_0);
std::array<double, 3> begin{ {0,0,0} }, end(begin);
begin[0] = R_0, begin[1] = Z_0;
double eps = 1e10, eps_old = 2e10;
unsigned N = 50;
while( (eps < eps_old || eps > 1e-7)&& eps > 1e-14)
{
eps_old = eps; N*=2;
using Vec = std::array<double,3>;
dg::RungeKutta<Vec> rk( "Feagin-17-8-10", begin);
dg::SinglestepTimeloop<Vec> odeint;
if( m_firstline == 0)
odeint = dg::SinglestepTimeloop<Vec>( rk,
fieldRZYTconf_);
if( m_firstline == 1)
odeint = dg::SinglestepTimeloop<Vec>( rk,
fieldRZYTequl_);
odeint.integrate_steps( 0., begin, 2.*M_PI, end, N);
eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
}
double f_psi = 2.*M_PI/end[2];
return f_psi;
}
double operator()( double psi)
{
double R_0, Z_0;
return construct_f( psi, R_0, Z_0);
}

private:
int m_firstline;
double X_init, Y_init;
CylindricalFunctorsLvl1 psip_;
CylindricalSymmTensorLvl1 chi_;
dg::geo::ribeiro::FieldRZYT fieldRZYTconf_;
dg::geo::equalarc::FieldRZYT fieldRZYTequl_;
dg::geo::FieldRZtau fieldRZtau_;

};

template<class real_type>
void compute_rzy( const CylindricalFunctorsLvl1& psi, const CylindricalSymmTensorLvl1& chi,
const thrust::host_vector<real_type>& y_vec,
thrust::host_vector<real_type>& r,
thrust::host_vector<real_type>& z,
real_type R_0, real_type Z_0, real_type f_psi, int mode )
{

thrust::host_vector<real_type> r_old(y_vec.size(), 0), r_diff( r_old);
thrust::host_vector<real_type> z_old(y_vec.size(), 0), z_diff( z_old);
r.resize( y_vec.size()), z.resize(y_vec.size());
std::array<real_type,2> begin{ {0,0} }, end(begin), temp(begin);
begin[0] = R_0, begin[1] = Z_0;
dg::geo::ribeiro::FieldRZY fieldRZYconf(psi, chi);
dg::geo::equalarc::FieldRZY fieldRZYequi(psi, chi);
fieldRZYconf.set_f(f_psi);
fieldRZYequi.set_f(f_psi);
unsigned steps = 1;
real_type eps = 1e10, eps_old=2e10;
while( (eps < eps_old||eps > 1e-7) && eps > 1e-14)
{
eps_old = eps, r_old = r, z_old = z;
using Vec = std::array<real_type,2>;
dg::RungeKutta<Vec> rk ( "Feagin-17-8-10", {0,0});
dg::SinglestepTimeloop<Vec> odeint;
if( mode == 0)
odeint = dg::SinglestepTimeloop<Vec>( rk, fieldRZYconf);
if( mode == 1)
odeint = dg::SinglestepTimeloop<Vec>( rk, fieldRZYequi);
odeint.integrate_steps( 0., begin, y_vec[0], end, steps);
r[0] = end[0], z[0] = end[1];
for( unsigned i=1; i<y_vec.size(); i++)
{
temp = end;
odeint.integrate_steps( y_vec[i-1], temp, y_vec[i], end, steps);
r[i] = end[0], z[i] = end[1];
}
dg::blas1::axpby( 1., r, -1., r_old, r_diff);
dg::blas1::axpby( 1., z, -1., z_old, z_diff);
real_type er = dg::blas1::dot( r_diff, r_diff);
real_type ez = dg::blas1::dot( z_diff, z_diff);
real_type ar = dg::blas1::dot( r, r);
real_type az = dg::blas1::dot( z, z);
eps =  sqrt( er + ez)/sqrt(ar+az);
steps*=2;
}
r = r_old, z = z_old;

}

struct Nemov
{
Nemov( const CylindricalFunctorsLvl2 psi, const CylindricalSymmTensorLvl1& chi, double f0, int mode):
f0_(f0), mode_(mode),
psip_(psi), chi_(chi), lapPsi_(psi,chi) { }
void initialize(
const thrust::host_vector<double>& r_init, 
const thrust::host_vector<double>& z_init, 
thrust::host_vector<double>& h_init) 
{
unsigned size = r_init.size();
h_init.resize( size);
for( unsigned i=0; i<size; i++)
{
if(mode_ == 0)
h_init[i] = f0_;
if(mode_ == 1)
{
double x = r_init[i], y = z_init[i];
double psipR = psip_.dfx()(x, y), psipZ = psip_.dfy()(x,y);
double chiRR = chi_.xx()(x, y),
chiRZ = chi_.xy()(x, y),
chiZZ = chi_.yy()(x, y);
double psip2 = chiRR*psipR*psipR + 2.*chiRZ*psipR*psipZ + chiZZ*psipZ*psipZ;
h_init[i]  = f0_/sqrt(psip2); 
}
}
}

void operator()(double t, const std::array<thrust::host_vector<double>,3 >& y, std::array<thrust::host_vector<double>,3>& yp)
{
unsigned size = y[0].size();
for( unsigned i=0; i<size; i++)
{
double xx = y[0][i], yy = y[1][i];
double psipR = psip_.dfx()(xx, yy), psipZ = psip_.dfy()(xx,yy);
double chiRR = chi_.xx()(xx, yy),
chiRZ = chi_.xy()(xx, yy),
chiZZ = chi_.yy()(xx, yy);
double psip2 =   chiRR*psipR*psipR + 2.*chiRZ*psipR*psipZ + chiZZ*psipZ*psipZ;
yp[0][i] =  (chiRR*psipR + chiRZ*psipZ)/psip2/f0_;
yp[1][i] =  (chiRZ*psipR + chiZZ*psipZ)/psip2/f0_;
yp[2][i] = y[2][i]*( - lapPsi_(xx,yy) )/psip2/f0_;
}
}
private:
double f0_;
int mode_;
CylindricalFunctorsLvl2 psip_;
CylindricalSymmTensorLvl1 chi_;
dg::geo::detail::LaplaceChiPsi lapPsi_;
};

template<class Nemov>
void construct_rz( Nemov nemov,
double x_0, 
const thrust::host_vector<double>& x_vec,  
const thrust::host_vector<double>& r_init, 
const thrust::host_vector<double>& z_init, 
thrust::host_vector<double>& r,
thrust::host_vector<double>& z,
thrust::host_vector<double>& h
)
{
unsigned N = 1;
double eps = 1e10, eps_old=2e10;
std::array<thrust::host_vector<double>,3> begin;
thrust::host_vector<double> h_init( r_init.size(), 0.);
nemov.initialize( r_init, z_init, h_init);
begin[0] = r_init, begin[1] = z_init, begin[2] = h_init;
std::array<thrust::host_vector<double>,3> end(begin), temp(begin);
unsigned sizeX = x_vec.size(), sizeY = r_init.size();
unsigned size2d = x_vec.size()*r_init.size();
r.resize(size2d), z.resize(size2d), h.resize(size2d);
double x0=x_0, x1 = x_vec[0];
thrust::host_vector<double> r_new(r_init), r_old(r_init), r_diff(r_init);
thrust::host_vector<double> z_new(z_init), z_old(z_init), z_diff(z_init);
thrust::host_vector<double> h_new(h_init); 
for( unsigned i=0; i<sizeX; i++)
{
N = 1;
eps = 1e10, eps_old=2e10;
begin = temp;
while( (eps < eps_old || eps > 1e-6) && eps > 1e-13)
{
r_old = r_new, z_old = z_new; eps_old = eps;
temp = begin;
x0 = i==0?x_0:x_vec[i-1], x1 = x_vec[i];
using Vec = std::array<thrust::host_vector<double>,3>;
dg::RungeKutta<Vec> rk( "Feagin-17-8-10", temp);
dg::SinglestepTimeloop<Vec>( rk, nemov).integrate_steps( x0, temp,
x1, end, N);
for( unsigned j=0; j<sizeY; j++)
{
r_new[j] = end[0][j],  z_new[j] = end[1][j];
h_new[j] = end[2][j];
}
temp = end;
dg::blas1::axpby( 1., r_new, -1., r_old, r_diff);
dg::blas1::axpby( 1., z_new, -1., z_old, z_diff);
dg::blas1::pointwiseDot( r_diff, r_diff, r_diff);
dg::blas1::pointwiseDot( 1., z_diff, z_diff, 1., r_diff);
try{
eps = sqrt( dg::blas1::dot( r_diff, 1.)/sizeY); 
} catch ( dg::Error& )
{
eps = eps_old;
r_new = r_old , z_new = z_old;
}
N*=2;
if( (eps > eps_old && N > 1024 && eps > 1e-6) || N > 64000)
throw dg::Error(dg::Message(_ping_) <<
"Grid generator encountered loss of convergence integrating from x = "
<<x0<<" to x = "<<x1);
}
for( unsigned j=0; j<sizeY; j++)
{
unsigned idx = sizeX*j+i;
r[idx] = r_new[j],  z[idx] = z_new[j], h[idx] = h_new[j];
}
}

}

} 

}



struct SimpleOrthogonal : public aGenerator2d
{

SimpleOrthogonal(const CylindricalFunctorsLvl2& psi, double psi_0, double
psi_1, double x0, double y0, double psi_firstline, int mode =0
):
SimpleOrthogonal( psi, CylindricalSymmTensorLvl1(), psi_0, psi_1, x0, y0,
psi_firstline, mode)
{
m_orthogonal = true;
}

SimpleOrthogonal(const CylindricalFunctorsLvl2& psi, const
CylindricalSymmTensorLvl1& chi, double psi_0, double psi_1,
double x0, double y0, double psi_firstline, int mode = 0
):
psi_(psi), chi_(chi)
{
assert( psi_1 != psi_0);
m_firstline = mode;
orthogonal::detail::Fpsi fpsi(psi, chi, x0, y0, mode);
f0_ = fabs( fpsi.construct_f( psi_firstline, R0_, Z0_));
if( psi_1 < psi_0) f0_*=-1;
lz_ =  f0_*(psi_1-psi_0);
m_orthogonal = false;
m_zeta_first = f0_*(psi_firstline-psi_0);
}


double f0() const{return f0_;}
virtual SimpleOrthogonal* clone() const override final{return new SimpleOrthogonal(*this);}

private:
virtual double do_width() const override final{return lz_;}
virtual double do_height() const override final{return 2.*M_PI;}
virtual bool do_isOrthogonal() const override final{return m_orthogonal;}
virtual void do_generate(
const thrust::host_vector<double>& zeta1d,
const thrust::host_vector<double>& eta1d,
thrust::host_vector<double>& x,
thrust::host_vector<double>& y,
thrust::host_vector<double>& zetaX,
thrust::host_vector<double>& zetaY,
thrust::host_vector<double>& etaX,
thrust::host_vector<double>& etaY) const override final
{
thrust::host_vector<double> r_init, z_init;
orthogonal::detail::compute_rzy( psi_, chi_, eta1d, r_init, z_init,
R0_, Z0_, f0_, m_firstline);
orthogonal::detail::Nemov nemov(psi_, chi_, f0_, m_firstline);
thrust::host_vector<double> h;
orthogonal::detail::construct_rz(nemov, m_zeta_first, zeta1d, r_init,
z_init, x, y, h);
unsigned size = x.size();
for( unsigned idx=0; idx<size; idx++)
{
double psipR = psi_.dfx()(x[idx], y[idx]);
double psipZ = psi_.dfy()(x[idx], y[idx]);
double chiRR = chi_.xx()( x[idx], y[idx]),
chiRZ = chi_.xy()( x[idx], y[idx]),
chiZZ = chi_.yy()( x[idx], y[idx]);
zetaX[idx] = f0_*psipR;
zetaY[idx] = f0_*psipZ;
etaX[idx] = -h[idx]*(chiRZ*psipR + chiZZ*psipZ);
etaY[idx] = +h[idx]*(chiRR*psipR + chiRZ*psipZ);
}
}
CylindricalFunctorsLvl2 psi_;
CylindricalSymmTensorLvl1 chi_;
double f0_, lz_, R0_, Z0_;
int m_firstline;
double m_zeta_first;
bool m_orthogonal;
};

}
}
