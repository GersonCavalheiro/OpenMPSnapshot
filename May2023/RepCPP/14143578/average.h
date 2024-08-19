#pragma once

#include <thrust/host_vector.h>
#include "dg/topology/weights.h"
#include "magnetic_field.h"
#include "flux.h"


namespace dg
{
namespace geo
{


template <class container >
struct FluxSurfaceIntegral
{

FluxSurfaceIntegral(const dg::Grid2d& g2d, const TokamakMagneticField& mag, double width_factor = 1.):
m_f(dg::evaluate(dg::one, g2d)), m_g(m_f), m_delta(m_f),
m_psi( dg::evaluate( mag.psip(), g2d)),
m_w2d ( dg::create::weights( g2d))
{
thrust::host_vector<double> psipR  = dg::evaluate( mag.psipR(), g2d);
thrust::host_vector<double> psipZ  = dg::evaluate( mag.psipZ(), g2d);
double psipRmax = dg::blas1::reduce( psipR, 0., dg::AbsMax<double>()  );
double psipZmax = dg::blas1::reduce( psipZ, 0., dg::AbsMax<double>()  );
double deltapsi = 0.5*(psipZmax*g2d.hy() +psipRmax*g2d.hx())/g2d.nx();
m_eps = deltapsi*width_factor;
}
double get_deltapsi() const{return m_eps;}


void set_left( const container& f){
dg::blas1::copy( f, m_f);
}

void set_right( const container& g){
dg::blas1::copy( g, m_g);
}

double operator()(double psip0)
{
dg::GaussianX delta( psip0, m_eps, 1./(sqrt(2.*M_PI)*m_eps));
dg::blas1::evaluate( m_delta, dg::equals(), delta, m_psi);
dg::blas1::pointwiseDot( 1., m_delta, m_f, m_g, 0., m_delta);
return dg::blas1::dot( m_delta, m_w2d);
}
private:
double m_eps;
container m_f, m_g, m_delta, m_psi;
const container m_w2d;
};


template<class container>
struct FluxVolumeIntegral
{

template<class Geometry2d>
FluxVolumeIntegral(const Geometry2d& g2d, const TokamakMagneticField& mag):
m_f(dg::evaluate(dg::one, g2d)), m_g(m_f), m_heavi(m_f),
m_psi( dg::pullback( mag.psip(), g2d)),
m_w2d ( dg::create::volume( g2d))
{
}


void set_left( const container& f){
dg::blas1::copy( f, m_f);
}

void set_right( const container& g){
dg::blas1::copy( g, m_g);
}

double operator()(double psip0)
{
dg::Heaviside heavi( psip0, -1);
dg::blas1::evaluate( m_heavi, dg::equals(), heavi, m_psi);
dg::blas1::pointwiseDot( 1., m_heavi, m_f, m_g, 0., m_heavi);
return dg::blas1::dot( m_heavi, m_w2d);
}
private:
double m_eps;
container m_f, m_g, m_heavi, m_psi;
const container m_w2d;
};



template <class container >
struct FluxSurfaceAverage
{

FluxSurfaceAverage( const dg::Grid2d& g2d, const TokamakMagneticField& mag, const container& f, container weights, double width_factor = 1.) :
m_avg( g2d,mag, width_factor), m_area( g2d, mag, width_factor)
{
m_avg.set_left( f);
m_avg.set_right( weights);
m_area.set_right( weights);
}

double get_deltapsi() const{return m_avg.get_deltapsi;}


void set_container( const container& f){
m_avg.set_left( f);
}

double operator()(double psip0)
{
return m_avg(psip0)/m_area(psip0);
}
private:
FluxSurfaceIntegral<container> m_avg, m_area;
};






struct SafetyFactorAverage
{

SafetyFactorAverage(const dg::Grid2d& g2d, const TokamakMagneticField& mag, double width_factor = 1.) :
m_fsi( g2d, mag, width_factor)
{
thrust::host_vector<double> alpha = dg::evaluate( mag.ipol(), g2d);
thrust::host_vector<double> R = dg::evaluate( dg::cooX2d, g2d);
dg::blas1::pointwiseDivide( alpha, R, alpha);
m_fsi.set_left( alpha);
}
void set_weights( const thrust::host_vector<double>& weights){
m_fsi.set_right( weights);
}

double operator()(double psip0)
{
return m_fsi( psip0)/(2.*M_PI);
}
private:
FluxSurfaceIntegral<thrust::host_vector<double> > m_fsi;
};




struct SafetyFactor
{
SafetyFactor( const TokamakMagneticField& mag):
m_fpsi( mag.get_psip(), mag.get_ipol(), mag.R0(), 0.,false){}


double operator()( double psip0)
{
return 1./m_fpsi( psip0);
}
private:
dg::geo::flux::detail::Fpsi m_fpsi;

};

}

}
