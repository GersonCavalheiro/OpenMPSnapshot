#pragma once
#include <exception>
#include "dg/algorithm.h"
#include "dg/matrix/matrix.h"
#include "parameters.h"
namespace esol
{

template< class Geometry, class Matrix, class container >
struct Esol
{

Esol( const Geometry& g, const Parameters& p );
const container& potential( int i) const { return m_psi[i];}
const container& density(   int i) const { return m_N[i];}
const container& psi2() const {return m_psi2;}
const container& gradn(int i) const { return m_gradn[i]; }
const container& gradphi(int i) const { return m_gradphi[i]; }
const Geometry& grid() const {return m_multigrid.grid(0);}
const container& volume() const {return m_volume;}
void compute_vorticity ( double alpha, const container& in, double beta, container& result)
{
m_lapMperp.set_chi(m_binv);
dg::blas2::symv( -1.0*alpha, m_lapMperp, in, beta, result);
m_lapMperp.set_chi(1.);
}
void compute_diff( double alpha, const container& nme, double beta, container& result)
{
if( m_p.nu != 0)
{
dg::blas2::gemv( m_lapMperpN, nme, m_iota);
dg::blas2::gemv( -alpha*m_p.nu, m_lapMperpN, m_iota, beta, result);
}
else
dg::blas1::scal( result, beta);
}

void gamma1_y( const container& y, container& yp)
{
if (m_p.tau[1]==0.0) 
dg::blas1::copy(y,yp);
else
{
std::vector<unsigned> number = m_multigrid.solve( m_multi_g1dag, yp, y, m_p.eps_gamma1);
if(  number[0] == m_multigrid.max_iter())
throw dg::Fail( m_p.eps_gamma1);
}

}

void gamma1inv_y( const container& y, container& yp)
{
if (m_p.tau[1]==0.0)
dg::blas1::copy(y, yp);
else
{
dg::blas2::symv( m_multi_g1dag[0], y, yp); 
}
}

void invLap_y( const container& y, container& yp)
{
m_multigrid.project( m_binv, m_multi_chi);
for( unsigned u=0; u<3; u++) 
m_multi_elliptic[u].set_chi( m_multi_chi[u]);

std::vector<unsigned> number = m_multigrid.solve( m_multi_elliptic, yp, y, m_p.eps_pol);
if(  number[0] == m_multigrid.max_iter())
throw dg::Fail( m_p.eps_pol[0]);
}

void solveSne(double t, const container& SNi, const container& potential, const container& y0, container& Sne)
{
if (m_p.source_rel == "zero-pol" || m_p.source_rel == "finite-pol")
{
if (m_p.tau[1]==0.0)
dg::blas1::copy(SNi,Sne);
else
{
m_gamma_SNi_ex.extrapolate(t, m_omega);
std::vector<unsigned> number = m_multigrid.solve( m_multi_g1dag, m_omega, SNi, m_p.eps_gamma1);
m_gamma_SNi_ex.update(t, m_omega);
if(  number[0] == m_multigrid.max_iter())
throw dg::Fail( m_p.eps_gamma1);
dg::blas1::axpby(1.0, m_omega, 0.0, Sne);
}
}
if (m_p.source_rel == "finite-pol")
{
if (m_p.equations == "ff-lwl")
{
dg::blas1::pointwiseDot(1.0, m_binv, m_binv, SNi, 0.0, m_chi); 
m_lapMperp.set_chi( m_chi);
dg::blas2::symv(-1.0, m_lapMperp, potential, 1.0, Sne);
m_lapMperp.set_chi( 1.);
}
else if (m_p.equations == "ff-O2")
{
}
}
}

void operator()( double t, const std::array<container,2>& y, std::array<container,2>& yp);

private:
const container& compute_psi( double t, const container& potential);
const container& polarisation( double t, const std::array<container,2>& y);

container m_chi, m_omega, m_iota, m_gamma_n, m_psi1, m_psi2, m_rho_m1, m_phi_m1, m_gamma0sqrtinv_rho_m1, m_gamma0sqrt_phi_m1,  m_logn, m_hp, m_hm, m_source, m_prof;
const container m_binv; 
std::array<container,2> m_psi, m_N, m_dN, m_gradn, m_gradphi;

dg::Elliptic<Geometry, Matrix, container>  m_lapMperp, m_lapMperpN; 
std::vector<dg::Elliptic<Geometry, Matrix, container> > m_multi_elliptic;
std::vector<dg::Helmholtz<Geometry,  Matrix, container> > m_multi_g1, m_multi_g1dag, m_multi_g0;

dg::mat::MatrixSqrt<container> m_sqrt;

dg::Advection<Geometry, Matrix, container> m_adv;

dg::Average<container > m_polavg; 

dg::MultigridCG2d<Geometry, Matrix, container> m_multigrid;
dg::Extrapolation<container> m_phi_ex, m_psi1_ex, m_gamma_n_ex, m_gamma0sqrt_phi_ex, m_rho_ex, m_gamma0sqrtinv_rho_ex, m_gamma_SNi_ex;
std::vector<container> m_multi_chi, m_multi_iota;

Matrix m_centered[2], m_centeredN;

const container m_volume;

const esol::Parameters m_p;
};

template< class Geometry, class M, class container>
Esol< Geometry, M,  container>::Esol( const Geometry& grid, const Parameters& p ):
m_chi( evaluate( dg::zero, grid)), m_omega(m_chi), m_iota(m_chi), m_gamma_n(m_chi), m_psi1(m_chi), m_psi2(m_chi), m_rho_m1(m_chi), m_phi_m1(m_chi), m_gamma0sqrtinv_rho_m1(m_chi), m_gamma0sqrt_phi_m1(m_chi), m_logn(m_chi),
m_hp( dg::evaluate(dg::PolynomialHeaviside(p.lx*p.xfac_sep, p.sigma_sep, 1), grid)),
m_hm( dg::evaluate(dg::PolynomialHeaviside(p.lx*p.xfac_sep, p.sigma_sep, -1), grid)),
m_binv( evaluate( dg::LinearX( p.kappa, 1.-p.kappa*p.posX*p.lx), grid)), 
m_lapMperp( grid, dg::centered),
m_lapMperpN( grid, p.bc_N_x, p.bc_y, dg::centered),
m_multigrid( grid, 3),
m_phi_ex( 2, m_chi),  m_psi1_ex(2, m_chi),  m_gamma_n_ex( 2, m_chi), m_gamma0sqrt_phi_ex( 2, m_chi), m_rho_ex(2, m_chi), m_gamma0sqrtinv_rho_ex(2, m_chi), m_gamma_SNi_ex(2, m_chi),
m_volume( dg::create::volume(grid)),
m_polavg(grid, dg::coo2d::y,"simple"),
m_p(p)
{
if(p.source_type == "flux") {
if (p.source_shape == "cauchy"){
m_source = dg::evaluate( dg::CauchyX( p.xfac_s*p.lx, p.sigma_s, p.omega_s)  , grid);
}
else if (p.source_shape == "gaussian"){
m_source = dg::evaluate( dg::GaussianX( p.xfac_s*p.lx, p.sigma_s, p.omega_s)  , grid);
}
}
else if (p.source_type == "forced") {
m_source = dg::evaluate(dg::PolynomialHeaviside(p.lx*p.xfac_sep, p.sigma_sep, -1), grid);
}


m_psi[0] = m_psi[1] = m_dN[0] = m_dN[1] = m_N[0] = m_N[1]  = m_gradn[0] = m_gradn[1] = m_gradphi[0] = m_gradphi[1]= m_chi; 
m_multi_chi= m_multigrid.project( m_chi);
m_multi_iota= m_multigrid.project( m_chi);
m_multi_elliptic.resize(3);
m_adv.construct(grid,p.bc_N_x, p.bc_y);
m_centered[0] = dg::create::dx( grid, grid.bcx(), dg::centered);
m_centeredN = dg::create::dx( grid, p.bc_N_x, dg::centered);
m_centered[1] = dg::create::dy( grid, grid.bcy(), dg::centered);
for( unsigned u=0; u<3; u++)
{
m_multi_elliptic[u].construct( m_multigrid.grid(u), dg::centered, p.jfactor);
m_multi_g0.push_back( {-p.tau[1], {m_multigrid.grid(u), dg::centered, p.jfactor}});
m_multi_g1.push_back( {-0.5*p.tau[1], {m_multigrid.grid(u), dg::centered, p.jfactor}});
m_multi_g1dag.push_back( {-0.5*p.tau[1], {m_multigrid.grid(u), p.bc_N_x, p.bc_y, dg::centered, p.jfactor}});
}
m_sqrt.construct( m_multi_g0[0], +1, m_volume, p.eps_gamma0, 1., p.maxiter_sqrt, p.maxiter_cauchy);

if(p.bgproftype == "tanh"){
m_prof = dg::evaluate( dg::TanhProfX(p.lx*p.xfac_p, p.ln,-1.0, p.bgprofamp,p.profamp), grid);
}
else if(p.bgproftype == "exp"){
m_prof = dg::evaluate( dg::ExpProfX(p.profamp, p.bgprofamp, p.ln), grid);
}
}

template< class G,  class M, class container>
const container& Esol<G,  M,  container>::compute_psi( double t, const container& potential)
{

if (m_p.tau[1] == 0.0) {
dg::blas1::copy( potential, m_psi1); 
}
else {
m_psi1_ex.extrapolate( t, m_psi1);
std::vector<unsigned> number = m_multigrid.solve( m_multi_g1, m_psi1, potential, m_p.eps_gamma1);
m_psi1_ex.update( t, m_psi1);
if(  number[0] == m_multigrid.max_iter())
throw dg::Fail( m_p.eps_gamma1);
}

if ( m_p.equations == "ff-O2") {
m_multi_elliptic[0].variation(-0.5, m_binv, m_psi1, 0.0, m_psi2);
dg::blas1::axpby( 1.0, m_psi1, 1.0, m_psi2, m_psi[1]);
}
else if  (m_p.equations == "ff-lwl") {
m_multi_elliptic[0].variation(-0.5, m_binv, potential, 0.0, m_psi2);
dg::blas1::axpby( 1.0, m_psi1, 1.0, m_psi2, m_psi[1]);
}
else { 
m_multi_elliptic[0].variation(potential, m_psi2);
dg::blas1::scal(m_psi2, -0.5); 
if (m_p.equations == "ff-O2-OB") {
dg::blas2::symv(m_multi_g0[0], m_psi2, m_chi);
dg::blas1::copy(m_chi, m_psi2);
}
dg::blas1::axpby( 1.0, m_psi1, 0.0, m_psi[1]); 
}
return m_psi[1];
}

template<class G,  class M,  class container>
const container& Esol<G,  M, container>::polarisation( double t, const std::array<container,2>& y)
{
if( m_p.equations == "ff-lwl" || m_p.equations == "ff-O2") {
dg::blas1::transform( y[1], m_chi, dg::PLUS<>( m_p.mu[1]*(m_p.bgprofamp + m_p.profamp))); 
dg::blas1::pointwiseDot(1.0, m_binv, m_binv, m_chi, 0.0, m_chi); 
m_multigrid.project( m_chi, m_multi_chi);
for( unsigned u=0; u<3; u++) 
m_multi_elliptic[u].set_chi( m_multi_chi[u]);
}

if (m_p.tau[1]==0.0)
{
dg::blas1::copy(y[1],m_gamma_n);
}
else
{
m_gamma_n_ex.extrapolate(t, m_gamma_n);
std::vector<unsigned> number = m_multigrid.solve( m_multi_g1dag, m_gamma_n, y[1], m_p.eps_gamma1);
m_gamma_n_ex.update(t, m_gamma_n);
if(  number[0] == m_multigrid.max_iter())
throw dg::Fail( m_p.eps_gamma1);
}
dg::blas1::axpby( 1., m_gamma_n, -1., y[0], m_omega); 
if (m_p.equations == "ff-O2-OB") {            
dg::blas2::symv(m_multi_g0[0], m_omega, m_chi);
dg::blas1::copy(m_chi, m_omega);
}
else if (m_p.equations == "ff-O2") {
dg::blas1::axpby(-1.0, m_gamma0sqrtinv_rho_m1, 1.0, m_omega, m_chi);
dg::blas1::copy(m_omega, m_gamma0sqrtinv_rho_m1);
dg::apply( m_sqrt, m_chi, m_omega);
dg::blas1::axpby( 1.0, m_rho_m1, 1.0, m_omega); 
dg::blas1::copy(m_omega, m_rho_m1);
}

if( m_p.equations == "ff-O2" ) {
m_gamma0sqrt_phi_ex.extrapolate(t, m_iota);
std::vector<unsigned> number = m_multigrid.solve( m_multi_elliptic, m_iota, m_omega, m_p.eps_pol);
m_gamma0sqrt_phi_ex.update( t, m_iota); 
if(  number[0] == m_multigrid.max_iter())
throw dg::Fail( m_p.eps_pol[0]);

dg::blas1::axpby(1.0, m_iota, -1.0, m_gamma0sqrt_phi_m1, m_chi); 
dg::blas1::copy(m_iota, m_gamma0sqrt_phi_m1);
dg::apply( m_sqrt, m_chi, m_psi[0]); 
dg::blas1::axpby( 1.0, m_phi_m1, 1.0, m_psi[0]); 
dg::blas1::copy(m_psi[0], m_phi_m1);
}
else { 
m_phi_ex.extrapolate(t, m_psi[0]);
std::vector<unsigned> number = m_multigrid.solve( m_multi_elliptic, m_psi[0], m_omega, m_p.eps_pol);
m_phi_ex.update( t, m_psi[0]);
if(  number[0] == m_multigrid.max_iter())
throw dg::Fail( m_p.eps_pol[0]);
}

return m_psi[0];
}


template< class G,  class M,  class container>
void Esol<G,  M,  container>::operator()( double t, const std::array<container,2>& y, std::array<container,2>& yp)
{
if (m_p.formulation == "conservative")
{
assert( y.size() == 2);
assert( y.size() == yp.size());

m_psi[0] = polarisation( t, y);
m_psi[1] = compute_psi( t, m_psi[0]);

for( unsigned i=0; i<y.size(); i++) 
{
dg::blas1::transform( y[i], m_N[i], dg::PLUS<double>(m_p.bgprofamp + m_p.profamp) );

dg::blas2::symv( -1., m_centered[1], m_psi[i], 0., m_chi); 
dg::blas2::symv(  1., m_centered[0], m_psi[i], 0., m_iota); 
if (i==0)
{
dg::blas1::copy(m_iota, m_gradphi[0]);
dg::blas1::copy(m_chi, m_gradphi[1]);
dg::blas1::scal(m_gradphi[1], -1.0);
}
m_adv.upwind( -1., m_chi, m_iota, y[i], 0., yp[i]);  
dg::blas1::pointwiseDot( m_binv, yp[i], yp[i]);

dg::blas2::symv( m_centered[1], y[i], m_iota);
if (i==0)
{
dg::blas2::symv( m_centeredN, y[i], m_gradn[0]); 
dg::blas1::copy(m_iota, m_gradn[1]);
}
dg::blas2::symv( m_centered[1], m_psi[i], m_omega);
dg::blas1::pointwiseDot( m_omega, m_N[i], m_omega);        
dg::blas1::axpbypgz( m_p.kappa, m_omega, m_p.tau[i]*m_p.kappa, m_iota, 1., yp[i]);

compute_diff( 1., y[i], 1., yp[i]);            
}

if (m_p.alpha != 0.0)
{
if (m_p.hwmode == "modified")
{
dg::blas1::transform( m_N[0], m_logn, dg::LN<double>());
m_polavg(m_logn, m_iota);       
m_polavg(m_psi[0], m_chi);        
dg::blas1::axpby(1., m_psi[0],  -1., m_chi, m_chi);       
dg::blas1::axpbypgz(1., m_chi, -1., m_logn, 1.0, m_iota); 
}
else if (m_p.hwmode == "ordinary")
{
m_polavg(m_N[0], m_iota);       
dg::blas1::pointwiseDivide(m_N[0], m_iota, m_iota);
dg::blas1::transform( m_iota, m_chi, dg::LN<double>());
dg::blas1::axpby(1., m_psi[0], -1., m_chi, m_iota); 
}
else if (m_p.hwmode == "ordinary_nonper")
{
dg::blas1::transform( m_N[0], m_chi, dg::LN<double>());
dg::blas1::axpby(1., m_psi[0], -1., m_chi, m_iota); 
}
dg::blas1::pointwiseDot(m_p.alpha, m_iota, m_hm, 1.0,yp[0]);

}

if (m_p.lambda !=0.0)
{
dg::blas1::axpby(-1.,m_psi[0], 0., m_omega, m_omega);      
dg::blas1::transform(m_omega, m_omega, dg::EXP<double>()); 
if (m_p.renormalize == false) dg::blas1::pointwiseDot(-m_p.lambda/sqrt(2.*M_PI*fabs(m_p.mu[0])),m_hp, m_omega, m_N[0], 1.0,yp[0]); 
else dg::blas1::pointwiseDot(-m_p.lambda*sqrt(1.+m_p.tau[1]),m_hp, m_omega, m_N[0], 1.0,yp[0]); 

dg::blas1::pointwiseDot(m_N[0], m_hp, m_iota); 
dg::blas1::axpby(-sqrt(1.+m_p.tau[1])*m_p.lambda, m_iota, 1.0, yp[1]);        
dg::blas1::pointwiseDot( y[0], m_hp, m_iota); 
dg::blas2::symv( m_lapMperpN, m_iota, m_omega); 
dg::blas1::axpby(-sqrt(1.+m_p.tau[1])*m_p.lambda*0.5*m_p.tau[1]*m_p.mu[1], m_omega, 1.0, yp[1]);
}

if (m_p.omega_s != 0.0)
{
if (m_p.source_type == "flux")
{
dg::blas1::axpby( m_p.omega_s, m_source, 1.0, yp[1]);
solveSne(t, m_source, m_psi[0], y[0], m_omega);
}
else if (m_p.source_type == "forced")
{
m_polavg(m_N[1], m_iota);       
dg::blas1::axpby(1.0, m_prof, -1.0, m_iota, m_iota); 
dg::blas1::pointwiseDot(m_iota, m_source, m_omega); 
dg::blas1::transform(m_omega, m_chi, dg::POSVALUE<double>()); 
dg::blas1::axpby(m_p.omega_s, m_chi, 1.0, yp[1]); 
solveSne(t, m_chi, m_psi[0], y[0], m_omega);
}
dg::blas1::axpby( m_p.omega_s, m_omega, 1.0, yp[0]);
}

if (m_p.omega_n !=0.0)
{
dg::blas1::axpby(1.0, m_p.n_min, -1.0, m_N[1], m_omega); 
dg::blas1::transform(m_omega, m_chi, dg::POSVALUE<double>()); 
dg::blas1::axpby(m_p.omega_n, m_chi, 1.0, yp[1]); 
solveSne(t, m_chi, m_psi[0], y[0], m_omega);        
dg::blas1::axpby( m_p.omega_n, m_omega, 1.0, yp[0]);
}
}

else if (m_p.formulation == "ln")
{
assert( y.size() == 2);
assert( y.size() == yp.size());
for( unsigned i=0; i<y.size(); i++) 
{
dg::blas1::transform( y[i], m_N[i], dg::EXP<double>() );
dg::blas1::scal(m_N[i], m_p.bgprofamp + m_p.profamp);
dg::blas1::axpby(1.0, m_N[i], -1.0, m_p.bgprofamp + m_p.profamp, m_dN[i]);
}    
m_psi[0] = polarisation( t, m_dN);
m_psi[1] = compute_psi( t, m_psi[0]);

for( unsigned i=0; i<y.size(); i++) 
{
dg::blas2::symv( -1., m_centered[1], m_psi[i], 0., m_chi); 
dg::blas2::symv(  1., m_centered[0], m_psi[i], 0., m_iota); 
if (i==0)
{
dg::blas1::copy(m_iota, m_gradphi[0]); 
dg::blas1::copy(m_chi, m_gradphi[1]);  
dg::blas1::scal(m_gradphi[1], -1.0);   
}
m_adv.upwind( -1., m_chi, m_iota, y[i], 0., yp[i]);   
dg::blas1::pointwiseDot( m_binv, yp[i], yp[i]);       

dg::blas2::symv( m_centered[1], y[i], m_iota);   
if (i==0)
{
dg::blas2::symv( m_centeredN, y[i], m_gradn[0]); 
dg::blas1::copy(m_iota, m_gradn[1]);             
}
dg::blas2::symv( m_centered[1], m_psi[i], m_omega);  
dg::blas1::axpbypgz( m_p.kappa, m_omega, m_p.tau[i]*m_p.kappa, m_iota, 1., yp[i]); 

compute_diff( 1., m_dN[i], 0.0, m_omega);
dg::blas1::pointwiseDivide(  m_omega, m_N[i], m_omega);
dg::blas1::axpby(1.0, m_omega, 1.0, yp[i]); 
}

if (m_p.alpha != 0.0)
{
if (m_p.hwmode == "modified")
{
m_polavg(y[0], m_iota);       
m_polavg(m_psi[0], m_chi);        
dg::blas1::axpby(1., m_psi[0],  -1., m_chi, m_chi);       
dg::blas1::axpbypgz(1., m_chi, -1., y[0], 1.0, m_iota); 
}
else if (m_p.hwmode == "ordinary")
{
m_polavg(m_N[0], m_iota);       
dg::blas1::pointwiseDivide(m_N[0], m_iota, m_iota);
dg::blas1::transform( m_iota, m_chi, dg::LN<double>());
dg::blas1::axpby(1., m_psi[0], -1., m_chi, m_iota); 
}
else if (m_p.hwmode == "ordinary_nonper")
{
dg::blas1::transform( m_N[0], m_chi, dg::LN<double>());
dg::blas1::axpby(1., m_psi[0], -1., m_chi, m_iota); 
}
dg::blas1::pointwiseDivide(m_iota,  m_N[0], m_iota);
dg::blas1::pointwiseDot(m_p.alpha, m_iota, m_hm, 1.0,yp[0]);
}

if (m_p.lambda !=0.0)
{
dg::blas1::axpby(-1., m_psi[0], 0., m_omega, m_omega);      
dg::blas1::transform(m_omega, m_omega, dg::EXP<double>()); 
if (m_p.renormalize == false) dg::blas1::pointwiseDot(-m_p.lambda/sqrt(2.*M_PI*fabs(m_p.mu[0])), m_hp, m_omega, 0.0, m_omega); 
else dg::blas1::pointwiseDot(-m_p.lambda*sqrt(1.+m_p.tau[1]), m_hp, m_omega,  1.0, yp[0]);  

dg::blas1::pointwiseDot(m_N[0], m_hp, m_iota); 
dg::blas1::pointwiseDivide(-sqrt(1.+m_p.tau[1])*m_p.lambda, m_iota, m_N[1], 1.0, yp[1]); 
dg::blas1::pointwiseDot(m_dN[0], m_hp, m_iota); 
dg::blas2::symv( m_lapMperpN, m_iota, m_omega); 
dg::blas1::pointwiseDivide(-sqrt(1.+m_p.tau[1])*m_p.lambda*0.5*m_p.tau[1]*m_p.mu[1], m_omega, m_N[1], 1.0, yp[1]);
}

if (m_p.omega_s != 0.0)
{
if (m_p.source_type == "flux")
{
dg::blas1::pointwiseDivide( m_p.omega_s, m_source, m_N[1], 1.0, yp[1]);
solveSne(t, m_source, m_psi[0], m_dN[0], m_omega);
}
else if (m_p.source_type == "forced")
{
m_polavg(m_N[1], m_iota);       
dg::blas1::axpby(1.0, m_prof, -1.0, m_iota, m_iota); 
dg::blas1::pointwiseDot(m_iota, m_source, m_omega); 
dg::blas1::transform(m_omega, m_chi, dg::POSVALUE<double>()); 
dg::blas1::pointwiseDivide(m_p.omega_s, m_chi, m_N[1], 1.0, yp[1]); 
solveSne(t, m_chi, m_psi[0], m_dN[0], m_omega);
}
dg::blas1::pointwiseDivide( m_p.omega_s, m_omega, m_N[0], 1.0, yp[0]);
}
}

return;
}

}
