#pragma once

#include "dg/algorithm.h"

namespace feltor
{
namespace routines{


template<class Container>
void dot( const std::array<Container, 3>& v,
const std::array<Container, 3>& w,
Container& result)
{
dg::blas1::evaluate( result, dg::equals(), dg::PairSum(),
v[0], w[0], v[1], w[1], v[2], w[2]);
}

template<class Container>
void dot( double alpha, const std::array<Container, 3>& v,
const std::array<Container, 3>& w, double beta,
Container& result)
{
dg::blas1::evaluate( result, dg::Axpby<double>(alpha,beta), dg::PairSum(),
v[0], w[0], v[1], w[1], v[2], w[2]);
}


struct Dot{
DG_DEVICE void operator()(
double lambda,
double d0P, double d1P, double d2P,
double& c_0, double& c_1, double& c_2)
{
c_0 = lambda*(d0P);
c_1 = lambda*(d1P);
c_2 = lambda*(d2P);
}
};
template<class Container>
void scal( const Container& lambda,
const std::array<Container, 3>& a,
std::array<Container, 3>& c)
{
dg::blas1::subroutine( Dot(), lambda,
a[0], a[1], a[2], c[0], c[1], c[2]);
}

struct Times{
DG_DEVICE void operator()(
double d0P, double d1P, double d2P, 
double d0S, double d1S, double d2S,
double& c_0, double& c_1, double& c_2)
{
c_0 = (d1P*d2S-d2P*d1S);
c_1 = (d2P*d0S-d0P*d2S);
c_2 = (d0P*d1S-d1P*d0S);
}
};

template<class Container>
void times(
const std::array<Container, 3>& a,
const std::array<Container, 3>& b,
std::array<Container, 3>& c)
{
dg::blas1::subroutine( Times(),
a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}

struct Jacobian{
DG_DEVICE double operator()(
double d0P, double d1P, double d2P, 
double d0S, double d1S, double d2S,
double b_0, double b_1, double b_2)
{
return      b_0*( d1P*d2S-d2P*d1S)+
b_1*( d2P*d0S-d0P*d2S)+
b_2*( d0P*d1S-d1P*d0S);
}
};

template<class Container>
void jacobian(
const std::array<Container, 3>& a,
const std::array<Container, 3>& b,
const std::array<Container, 3>& c,
Container& result)
{
dg::blas1::evaluate( result, dg::equals(), Jacobian(),
a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
template<class Container>
void jacobian(
double alpha,
const std::array<Container, 3>& a,
const std::array<Container, 3>& b,
const std::array<Container, 3>& c,
double beta,
Container& result)
{
dg::blas1::evaluate( result, dg::Axpby<double>(alpha,beta), Jacobian(),
a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
}

template<class Container>
void slice_vector3d( const Container& transfer, Container& transfer2d, size_t local_size2d)
{
#ifdef WITH_MPI
thrust::copy(
transfer.data().begin(),
transfer.data().begin() + local_size2d,
transfer2d.data().begin()
);
#else
thrust::copy(
transfer.begin(),
transfer.begin() + local_size2d,
transfer2d.begin()
);
#endif
}

dg::geo::CurvilinearGrid2d generate_XGrid( dg::file::WrappedJsonValue config,
const dg::geo::TokamakMagneticField& mag, double& psipO, double& psipmax, double& f0)
{
unsigned npsi = config.get("n",3).asUInt();
unsigned Npsi = config.get("Npsi", 64).asUInt();
unsigned Neta = config.get("Neta", 640).asUInt();
std::cout << "Using X-point grid resolution (n("<<npsi<<"), Npsi("<<Npsi<<"), Neta("<<Neta<<"))\n";
double RO = mag.R0(), ZO = 0;
int point = dg::geo::findOpoint( mag.get_psip(), RO, ZO);
psipO = mag.psip()(RO, ZO);
std::cout << "O-point found at "<<RO<<" "<<ZO
<<" with Psip "<<psipO<<std::endl;
if( point == 1 )
std::cout << " (minimum)"<<std::endl;
if( point == 2 )
std::cout << " (maximum)"<<std::endl;
double fx_0 = config.get( "fx_0", 1./8.).asDouble(); 
psipmax = -fx_0/(1.-fx_0)*psipO;
std::cout << "psi outer in g1d_out is "<<psipmax<<"\n";
std::cout << "Generate orthogonal flux-aligned grid ... \n";
std::unique_ptr<dg::geo::aGenerator2d> generator;
if( !(mag.params().getDescription() == dg::geo::description::standardX))
generator = std::make_unique<dg::geo::SimpleOrthogonal>(
mag.get_psip(),
psipO<psipmax ? psipO : psipmax,
psipO<psipmax ? psipmax : psipO,
mag.R0() + 0.1*mag.params().a(), 0., 0.1*psipO, 1);
else
{

double RX = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
double ZX = -1.1*mag.params().elongation()*mag.params().a();
dg::geo::findXpoint( mag.get_psip(), RX, ZX);
const double psipX = mag.psip()( RX, ZX);
std::cout << "X-point set at "<<RX<<" "<<ZX<<" with Psi_p = "<<psipX<<"\n";
dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), RX, ZX) ;
generator = std::make_unique<dg::geo::SeparatrixOrthogonalAdaptor>(
mag.get_psip(), monitor_chi,
psipO<psipmax ? psipO : psipmax,
RX, ZX, mag.R0(), 0, 1, false, fx_0);
}
std::cout << "DONE!\n";
dg::geo::CurvilinearGrid2d gridX2d(*generator,
npsi, Npsi, Neta, dg::DIR, dg::PER);
f0 = ( gridX2d.x1() - gridX2d.x0() ) / ( psipmax - psipO );
return gridX2d;
}


std::vector<std::tuple<std::string, dg::HVec, std::string> >
compute_twoflux_labels( const dg::geo::CurvilinearGrid2d& gridX2d)
{
std::vector<std::tuple<std::string, dg::HVec, std::string> > map2d;
std::vector<dg::HVec > coordsX = gridX2d.map();
map2d.emplace_back( "xc", coordsX[0],
"x-coordinate in Cartesian coordinate system of FSA-grid");
map2d.emplace_back( "yc", coordsX[1],
"y-coordinate in Cartesian coordinate system of FSA-grid");
map2d.emplace_back( "vol", dg::create::volume( gridX2d),
"Volume form of FSA-grid");
return map2d;
}

std::vector<std::tuple<std::string, dg::HVec, std::string> >
compute_oneflux_labels(
dg::Average<dg::HVec>& poloidal_average,
const dg::geo::CurvilinearGrid2d& gridX2d,
const dg::geo::TokamakMagneticField& mod_mag,
double psipO, double psipmax, double f0,
dg::HVec& dvdpsip, dg::HVec& volX2d,
dg::Grid1d& g1d_out, 
dg::Grid1d& g1d_out_eta 
)
{
unsigned npsi = gridX2d.n();
unsigned Npsi = gridX2d.Nx();
unsigned Neta = gridX2d.Ny();
std::vector<dg::HVec > coordsX = gridX2d.map();
std::vector<std::tuple<std::string, dg::HVec, std::string> > map1d;
dg::SparseTensor<dg::HVec> metricX = gridX2d.metric();
volX2d = dg::tensor::volume2d( metricX);
poloidal_average( volX2d, dvdpsip, false);
g1d_out = dg::Grid1d(psipO<psipmax ? psipO : psipmax,
psipO<psipmax ? psipmax : psipO,
npsi, Npsi, psipO < psipmax ? dg::DIR_NEU : dg::NEU_DIR);
g1d_out_eta = dg::Grid1d(gridX2d.y0(), gridX2d.y1(), npsi, Neta, dg::DIR_NEU);
dg::blas1::scal( dvdpsip, 2.*M_PI*f0);
map1d.emplace_back( "dv2ddpsi", dvdpsip,
"Derivative of 2d flux volume (=area) with respect to flux label psi");
dg::direction integration_dir = psipO < psipmax ? dg::forward : dg::backward;
dg::HVec X_psi_vol = dg::integrate( dvdpsip, g1d_out, integration_dir);
map1d.emplace_back( "psi_vol2d", X_psi_vol,
"2d Flux volume (area) evaluated with X-point grid");
dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d); 
poloidal_average( volX2d, dvdpsip, false);
dg::blas1::scal( dvdpsip, 4.*M_PI*M_PI*f0);
map1d.emplace_back( "dvdpsi", dvdpsip,
"Derivative of flux volume with respect to flux label psi");
X_psi_vol = dg::integrate( dvdpsip, g1d_out, integration_dir);
map1d.emplace_back( "psi_vol", X_psi_vol,
"Flux volume evaluated with X-point grid");

dg::HVec gradZetaX = metricX.value(0,0), X_psi_area;
dg::blas1::transform( gradZetaX, gradZetaX, dg::SQRT<double>());
dg::blas1::pointwiseDot( volX2d, gradZetaX, gradZetaX); 
poloidal_average( gradZetaX, X_psi_area, false);
dg::blas1::scal( X_psi_area, 4.*M_PI*M_PI);
map1d.emplace_back( "psi_area", X_psi_area,
"Flux area evaluated with X-point grid");
dg::blas1::pointwiseDivide( gradZetaX, coordsX[0], gradZetaX); 
poloidal_average( gradZetaX, X_psi_area, false);
dg::blas1::scal( X_psi_area, 2.*M_PI);
map1d.emplace_back( "psi_arc", X_psi_area,
"Psip arc length evaluated with X-point grid");

dg::HVec rho = dg::evaluate( dg::cooX1d, g1d_out);
dg::blas1::axpby( -1./psipO, rho, +1., 1., rho); 
map1d.emplace_back("rho", rho,
"Alternative flux label rho = 1-psi/psimin");
dg::blas1::transform( rho, rho, dg::SQRT<double>());
map1d.emplace_back("rho_p", rho,
"Alternative flux label rho_p = sqrt(1-psi/psimin)");
dg::geo::SafetyFactor qprof( mod_mag);
dg::HVec psi_vals = dg::evaluate( dg::cooX1d, g1d_out);
dg::blas1::subroutine( [psipO]( double& psi){
if( (psipO < 0 && psi > 0) || (psipO>0 && psi <0))
psi = psipO/2.; 
}, psi_vals);
dg::HVec qprofile( psi_vals);
dg::blas1::evaluate( qprofile, dg::equals(), qprof, psi_vals);
map1d.emplace_back("q-profile", qprofile,
"q-profile (Safety factor) using direct integration");
map1d.emplace_back("psi_psi",    dg::evaluate( dg::cooX1d, g1d_out),
"Poloidal flux label psi (same as coordinate)");
dg::HVec psit = dg::integrate( qprofile, g1d_out, integration_dir);
map1d.emplace_back("psit1d", psit,
"Toroidal flux label psi_t integrated using q-profile");
dg::Grid1d g1d_fine(psipO<0. ? psipO : 0., psipO<0. ? 0. : psipO, npsi
,Npsi,dg::DIR_NEU);
qprofile = dg::evaluate( qprof, g1d_fine);
dg::HVec w1d = dg::create::weights( g1d_fine);
double psit_tot = dg::blas1::dot( w1d, qprofile);
if( integration_dir == dg::backward)
psit_tot *= -1;
dg::blas1::scal ( psit, 1./psit_tot);
dg::blas1::transform( psit, psit, dg::SQRT<double>());
map1d.emplace_back("rho_t", psit,
"Toroidal flux label rho_t = sqrt( psit/psit_tot)");
return map1d;
}

}
