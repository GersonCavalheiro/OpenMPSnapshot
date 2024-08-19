#pragma once

#include "dg/algorithm.h"
#include "fluxfunctions.h"
#include "ribeiro.h"


namespace dg
{
namespace geo
{

namespace flux
{
namespace detail
{

struct Fpsi
{

Fpsi( const CylindricalFunctorsLvl2& psip, const CylindricalFunctorsLvl1& ipol, double x0, double y0, bool verbose = false):
psip_(psip), fieldRZYT_(psip, ipol, x0, y0), fieldRZtau_(psip),m_verbose(verbose)
{
double R_O = x0, Z_O = y0;
dg::geo::findOpoint( psip, R_O, Z_O);
fieldRZYT_ = dg::geo::flux::FieldRZYT(psip, ipol, R_O, Z_O);
X_init = x0, Y_init = y0;
while( fabs( psip.dfx()(X_init, Y_init)) <= 1e-10 && fabs( psip.dfy()( X_init, Y_init)) <= 1e-10)
X_init +=  1.;
}
void find_initial( double psi, double& R_0, double& Z_0)
{
unsigned N = 50;
std::array<double, 2> begin2d{ {0,0} }, end2d(begin2d), end2d_old(begin2d);
if(m_verbose)std::cout << "In init function\n";
begin2d[0] = end2d[0] = end2d_old[0] = X_init;
begin2d[1] = end2d[1] = end2d_old[1] = Y_init;
double eps = 1e10, eps_old = 2e10;
using Vec = std::array<double,2>;
dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>( "Feagin-17-8-10",
begin2d), fieldRZtau_);
while( (eps < eps_old || eps > 1e-7) && eps > 1e-14)
{
eps_old = eps; end2d_old = end2d;
N*=2; odeint.integrate_steps( psip_.f()(X_init, Y_init), begin2d,
psi, end2d, N);
eps = sqrt( (end2d[0]-end2d_old[0])*(end2d[0]-end2d_old[0]) + (end2d[1]-end2d_old[1])*(end2d[1]-end2d_old[1]));
}
X_init = R_0 = end2d_old[0], Y_init = Z_0 = end2d_old[1];
if(m_verbose)std::cout << "In init function error: psi(R,Z)-psi0: "<<psip_.f()(X_init, Y_init)-psi<<"\n";
}

double construct_f( double psi, double& R_0, double& Z_0)
{
find_initial( psi, R_0, Z_0);
std::array<double,3> begin{ {0,0,0} }, end(begin), end_old(begin);
begin[0] = R_0, begin[1] = Z_0;
double eps = 1e10, eps_old = 2e10;
unsigned N = 50;
unsigned nan_counter = 0;
using Vec = std::array<double,3>;
dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>( "Feagin-17-8-10",
begin), fieldRZYT_);
while( (eps < eps_old || eps > 1e-7)&& eps > 1e-14)
{
eps_old = eps, end_old = end; N*=2;
odeint.integrate_steps( 0., begin, 2*M_PI, end, N);
eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) +
(end[1]-begin[1])*(end[1]-begin[1]));
if(m_verbose)std::cout << "\t error "<<eps<<" with "<<N<<" steps\t";
if( std::isnan( eps) && nan_counter < 4) eps = 1e10, end = end_old, nan_counter++;
}
if(m_verbose)std::cout << "\t error "<<eps<<" with "<<N<<" steps\t";
if(m_verbose)std::cout <<end_old[2] << " "<<end[2] <<"\n";
double f_psi = 2.*M_PI/end[2]; 
return f_psi;
}

double operator()( double psi)
{
double R_0, Z_0;
return construct_f( psi, R_0, Z_0);
}
double f_prime( double psi)
{
double deltaPsi = fabs(psi)/100.;
double fofpsi[4];
fofpsi[1] = operator()(psi-deltaPsi);
fofpsi[2] = operator()(psi+deltaPsi);
double fprime = (-0.5*fofpsi[1]+0.5*fofpsi[2])/deltaPsi, fprime_old;
double eps = 1e10, eps_old=2e10;
while( eps < eps_old)
{
deltaPsi /=2.;
fprime_old = fprime;
eps_old = eps;
fofpsi[0] = fofpsi[1], fofpsi[3] = fofpsi[2];
fofpsi[1] = operator()(psi-deltaPsi);
fofpsi[2] = operator()(psi+deltaPsi);
fprime  = (+ 1./12.*fofpsi[0]
- 2./3. *fofpsi[1]
+ 2./3. *fofpsi[2]
- 1./12.*fofpsi[3]
)/deltaPsi;
eps = fabs((fprime - fprime_old)/fprime);
}
return fprime_old;
}

private:
double X_init, Y_init;
CylindricalFunctorsLvl1 psip_;
dg::geo::flux::FieldRZYT fieldRZYT_;
dg::geo::FieldRZtau fieldRZtau_;
bool m_verbose;

};

} 
}


struct FluxGenerator : public aGenerator2d
{

FluxGenerator( const CylindricalFunctorsLvl2& psi, const CylindricalFunctorsLvl1& ipol, double psi_0, double psi_1, double x0, double y0, int mode=0, bool verbose = false):
psi_(psi), ipol_(ipol), mode_(mode), m_verbose( verbose)
{
psi0_ = psi_0, psi1_ = psi_1;
assert( psi_1 != psi_0);
if( mode==0)
{
flux::detail::Fpsi fpsi(psi, ipol, x0, y0, m_verbose);
f0_ = fabs( fpsi.construct_f( psi_0, x0_, y0_));
}
else
{
ribeiro::detail::Fpsi fpsi(psi, x0, y0, mode, m_verbose);
f0_ = fabs( fpsi.construct_f( psi_0, x0_, y0_));
}
if( psi_1 < psi_0) f0_*=-1;
lx_ =  f0_*(psi_1-psi_0);
x0_=x0, y0_=y0, psi0_=psi_0, psi1_=psi_1;
if(m_verbose)std::cout << "lx = "<<lx_<<"\n";
}

virtual FluxGenerator* clone() const override final{return new FluxGenerator(*this);}

private:
virtual double do_width() const override final{return lx_;}
virtual double do_height() const override final{return 2.*M_PI;}
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
thrust::host_vector<double> psi_x(zeta1d);
for( unsigned i=0; i<psi_x.size(); i++)
psi_x[i] = zeta1d[i]/f0_ +psi0_;

if(m_verbose)std::cout << "In grid function:"<<std::endl;
flux::detail::Fpsi fpsi(psi_, ipol_, x0_, y0_, m_verbose);
dg::geo::flux::FieldRZYRYZY fieldRZYRYZY(psi_, ipol_);
ribeiro::detail::Fpsi fpsiRibeiro(psi_, x0_, y0_, mode_, m_verbose);
dg::geo::equalarc::FieldRZYRYZY fieldRZYRYZYequalarc(psi_);
thrust::host_vector<double> fx_;
fx_.resize( zeta1d.size());
thrust::host_vector<double> f_p(fx_);
unsigned Nx = zeta1d.size(), Ny = eta1d.size();
for( unsigned i=0; i<zeta1d.size(); i++)
{
thrust::host_vector<double> ry, zy;
thrust::host_vector<double> yr, yz, xr, xz;
double R0, Z0;
if(mode_==0)dg::geo::detail::compute_rzy( fpsi, fieldRZYRYZY, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i], m_verbose);
if(mode_==1)dg::geo::detail::compute_rzy( fpsiRibeiro, fieldRZYRYZYequalarc, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i], m_verbose);
for( unsigned j=0; j<Ny; j++)
{
x[j*Nx+i]  = ry[j], y[j*Nx+i]  = zy[j];
etaX[j*Nx+i] = yr[j], etaY[j*Nx+i] = yz[j];
zetaX[j*Nx+i] = xr[j]/fx_[i]*f0_, zetaY[j*Nx+i] = xz[j]/fx_[i]*f0_;
}
}
}
CylindricalFunctorsLvl2 psi_;
CylindricalFunctorsLvl1 ipol_;
double f0_, lx_, x0_, y0_, psi0_, psi1_;
int mode_;
bool m_verbose;
};


struct RibeiroFluxGenerator : public aGenerator2d
{

RibeiroFluxGenerator( const CylindricalFunctorsLvl2& psi, double psi_0, double psi_1, double x0, double y0, int mode=0, bool verbose = false):
psip_(psi), mode_(mode), m_verbose(verbose)
{
psi0_ = psi_0, psi1_ = psi_1;
assert( psi_1 != psi_0);
ribeiro::detail::Fpsi fpsi(psi, x0, y0, mode, m_verbose);
f0_ = fabs( fpsi.construct_f( psi_0, x0_, y0_));
if( psi_1 < psi_0) f0_*=-1;
lx_ =  f0_*(psi_1-psi_0);
x0_=x0, y0_=y0, psi0_=psi_0, psi1_=psi_1;
if(m_verbose)std::cout << "lx = "<<lx_<<"\n";
}
virtual RibeiroFluxGenerator* clone() const{return new RibeiroFluxGenerator(*this);}

private:
virtual double do_width() const{return lx_;}
virtual double do_height() const{return 2.*M_PI;}
virtual void do_generate(
const thrust::host_vector<double>& zeta1d,
const thrust::host_vector<double>& eta1d,
thrust::host_vector<double>& x,
thrust::host_vector<double>& y,
thrust::host_vector<double>& zetaX,
thrust::host_vector<double>& zetaY,
thrust::host_vector<double>& etaX,
thrust::host_vector<double>& etaY) const
{
thrust::host_vector<double> psi_x(zeta1d);
for( unsigned i=0; i<psi_x.size(); i++)
psi_x[i] = zeta1d[i]/f0_ +psi0_;

ribeiro::detail::Fpsi fpsi(psip_, x0_, y0_, mode_, m_verbose);
dg::geo::ribeiro::FieldRZYRYZY fieldRZYRYZYribeiro(psip_);
dg::geo::equalarc::FieldRZYRYZY fieldRZYRYZYequalarc(psip_);
thrust::host_vector<double> fx_;
fx_.resize( zeta1d.size());
thrust::host_vector<double> f_p(fx_);
unsigned Nx = zeta1d.size(), Ny = eta1d.size();
for( unsigned i=0; i<zeta1d.size(); i++)
{
thrust::host_vector<double> ry, zy;
thrust::host_vector<double> yr, yz, xr, xz;
double R0, Z0;
if(mode_==0)dg::geo::detail::compute_rzy( fpsi, fieldRZYRYZYribeiro, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i], m_verbose);
if(mode_==1)dg::geo::detail::compute_rzy( fpsi, fieldRZYRYZYequalarc, psi_x[i], eta1d, ry, zy, yr, yz, xr, xz, R0, Z0, fx_[i], f_p[i], m_verbose);
for( unsigned j=0; j<Ny; j++)
{
x[j*Nx+i]  = ry[j], y[j*Nx+i]  = zy[j];
etaX[j*Nx+i] = yr[j], etaY[j*Nx+i] = yz[j];
zetaX[j*Nx+i] = xr[j]/fx_[i]*f0_, zetaY[j*Nx+i] = xz[j]/fx_[i]*f0_;
}
}
}
CylindricalFunctorsLvl2 psip_;
double f0_, lx_, x0_, y0_, psi0_, psi1_;
int mode_;
bool m_verbose;
};
}
}