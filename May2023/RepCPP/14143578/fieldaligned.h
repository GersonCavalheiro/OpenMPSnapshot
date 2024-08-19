#pragma once
#include <cmath>
#include <array>
#include <cusp/csr_matrix.h>

#include "dg/algorithm.h"
#include "magnetic_field.h"
#include "fluxfunctions.h"
#include "curvilinear.h"

namespace dg{
namespace geo{

enum whichMatrix
{
einsPlus = 0,   
einsPlusT, 
einsMinus, 
einsMinusT,
zeroPlus,  
zeroMinus, 
zeroPlusT, 
zeroMinusT, 
zeroForw  
};

typedef ONE FullLimiter;

typedef ZERO NoLimiter;
namespace detail{

static void parse_method( std::string method, std::string& i, std::string& p, std::string& f)
{
f = "dg";
if( method == "dg")                 i = "dg",       p = "dg";
else if( method == "linear")        i = "linear",   p = "dg";
else if( method == "cubic")         i = "cubic",    p = "dg";
else if( method == "nearest")       i = "nearest",  p = "dg";
else if( method == "dg-nearest")      i = "dg",      p = "nearest";
else if( method == "linear-nearest")  i = "linear",  p = "nearest";
else if( method == "cubic-nearest")   i = "cubic",   p = "nearest";
else if( method == "nearest-nearest") i = "nearest",  p = "nearest";
else if( method == "dg-linear")      i = "dg",       p = "linear";
else if( method == "linear-linear")  i = "linear",   p = "linear";
else if( method == "cubic-linear")   i = "cubic",    p = "linear";
else if( method == "nearest-linear") i = "nearest",  p = "linear";
else if( method == "dg-equi")       i = "dg",       p = "dg", f = "equi";
else if( method == "linear-equi")   i = "linear",   p = "dg", f = "equi";
else if( method == "cubic-equi")    i = "cubic",    p = "dg", f = "equi";
else if( method == "nearest-equi")  i = "nearest",  p = "dg", f = "equi";
else if( method == "dg-equi-nearest")      i = "dg",      p = "nearest", f = "equi";
else if( method == "linear-equi-nearest")  i = "linear",  p = "nearest", f = "equi";
else if( method == "cubic-equi-nearest")   i = "cubic",   p = "nearest", f = "equi";
else if( method == "nearest-equi-nearest") i = "nearest", p = "nearest", f = "equi";
else if( method == "dg-equi-linear")      i = "dg",       p = "linear", f = "equi";
else if( method == "linear-equi-linear")  i = "linear",   p = "linear", f = "equi";
else if( method == "cubic-equi-linear")   i = "cubic",    p = "linear", f = "equi";
else if( method == "nearest-equi-linear") i = "nearest",  p = "linear", f = "equi";
else
throw Error( Message(_ping_) << "The method "<< method << " is not recognized\n");
}

struct DSFieldCylindrical3
{
DSFieldCylindrical3( const dg::geo::CylindricalVectorLvl0& v): m_v(v){}
void operator()( double t, const std::array<double,3>& y,
std::array<double,3>& yp) const {
double R = y[0], Z = y[1];
double vz = m_v.z()(R, Z);
yp[0] = m_v.x()(R, Z)/vz;
yp[1] = m_v.y()(R, Z)/vz;
yp[2] = 1./vz;
}
private:
dg::geo::CylindricalVectorLvl0 m_v;
};

struct DSFieldCylindrical4
{
DSFieldCylindrical4( const dg::geo::CylindricalVectorLvl1& v): m_v(v){}
void operator()( double t, const std::array<double,3>& y,
std::array<double,3>& yp) const {
double R = y[0], Z = y[1];
double vx = m_v.x()(R,Z);
double vy = m_v.y()(R,Z);
double vz = m_v.z()(R,Z);
double divvvz = m_v.divvvz()(R,Z);
yp[0] = vx/vz;
yp[1] = vy/vz;
yp[2] = divvvz*y[2];
}

private:
dg::geo::CylindricalVectorLvl1 m_v;
};

struct DSField
{
DSField() = default;
DSField( const dg::geo::CylindricalVectorLvl1& v,
const dg::aGeometry2d& g ):
m_g(g)
{
dg::HVec v_zeta, v_eta;
dg::pushForwardPerp( v.x(), v.y(), v_zeta, v_eta, g);
dg::HVec vx = dg::pullback( v.x(), g);
dg::HVec vy = dg::pullback( v.y(), g);
dg::HVec vz = dg::pullback( v.z(), g);
dg::HVec divvvz = dg::pullback( v.divvvz(), g);
dg::blas1::pointwiseDivide(v_zeta,  vz, v_zeta);
dg::blas1::pointwiseDivide(v_eta,   vz, v_eta);
dzetadphi_  = dg::forward_transform( v_zeta, g );
detadphi_   = dg::forward_transform( v_eta, g );
dvdphi_     = dg::forward_transform( divvvz, g );
}
void operator()(double t, const std::array<double,3>& y, std::array<double,3>& yp) const
{
yp[0] = interpolate(dg::lspace, dzetadphi_, y[0], y[1], *m_g);
yp[1] = interpolate(dg::lspace, detadphi_,  y[0], y[1], *m_g);
yp[2] = interpolate(dg::lspace, dvdphi_,    y[0], y[1], *m_g)*y[2];
}
private:
thrust::host_vector<double> dzetadphi_, detadphi_, dvdphi_;
dg::ClonePtr<dg::aGeometry2d> m_g;
};

template<class real_type>
void integrate_all_fieldlines2d( const dg::geo::CylindricalVectorLvl1& vec,
const dg::aRealGeometry2d<real_type>& grid_field,
const dg::aRealTopology2d<real_type>& grid_evaluate,
std::array<thrust::host_vector<real_type>,3>& yp,
const thrust::host_vector<double>& vol0,
thrust::host_vector<real_type>& yp2b,
thrust::host_vector<bool>& in_boxp,
real_type deltaPhi, real_type eps)
{
std::array<thrust::host_vector<real_type>,3> y{
dg::evaluate( dg::cooX2d, grid_evaluate),
dg::evaluate( dg::cooY2d, grid_evaluate),
vol0
};
yp.fill(dg::evaluate( dg::zero, grid_evaluate));
dg::geo::detail::DSField field;
if( !dynamic_cast<const dg::CartesianGrid2d*>( &grid_field))
field = dg::geo::detail::DSField( vec, grid_field);

dg::geo::detail::DSFieldCylindrical4 cyl_field(vec);
const unsigned size = grid_evaluate.size();
dg::Adaptive<dg::ERKStep<std::array<real_type,3>>> adapt(
"Dormand-Prince-7-4-5", std::array<real_type,3>{0,0,0});
dg::AdaptiveTimeloop< std::array<real_type,3>> odeint;
if( dynamic_cast<const dg::CartesianGrid2d*>( &grid_field))
odeint = dg::AdaptiveTimeloop<std::array<real_type,3>>( adapt,
cyl_field, dg::pid_control, dg::fast_l2norm, eps, 1e-10);
else
odeint = dg::AdaptiveTimeloop<std::array<real_type,3>>( adapt,
field, dg::pid_control, dg::fast_l2norm, eps, 1e-10);

for( unsigned i=0; i<size; i++)
{
std::array<real_type,3> coords{y[0][i],y[1][i],y[2][i]}, coordsP;
real_type phi1 = deltaPhi;
odeint.set_dt( deltaPhi/2.);
odeint.integrate( 0, coords, phi1, coordsP);
yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
}
yp2b.assign( grid_evaluate.size(), deltaPhi); 
in_boxp.resize( yp2b.size());
for( unsigned i=0; i<size; i++)
{
std::array<real_type,3> coords{y[0][i],y[1][i],y[2][i]}, coordsP;
in_boxp[i] = grid_field.contains( yp[0][i], yp[1][i]) ? true : false;
if( false == in_boxp[i])
{
real_type phi1 = deltaPhi;
odeint.integrate_in_domain( 0., coords, phi1, coordsP, 0., (const
dg::aRealTopology2d<real_type>&)grid_field, eps);
yp2b[i] = phi1;
}
}
}


}


struct WallFieldlineDistance : public aCylindricalFunctor<WallFieldlineDistance>
{

WallFieldlineDistance(
const dg::geo::CylindricalVectorLvl0& vec,
const dg::aRealTopology2d<double>& domain,
double maxPhi, double eps, std::string type) :
m_domain( domain), m_cyl_field(vec),
m_deltaPhi( maxPhi), m_eps( eps), m_type(type)
{
if( m_type != "phi" && m_type != "s")
throw std::runtime_error( "Distance type "+m_type+" not recognized!\n");
}

double do_compute( double R, double Z) const
{
std::array<double,3> coords{ R, Z, 0}, coordsP(coords);
m_cyl_field( 0., coords, coordsP);
double sign = coordsP[2] > 0 ? +1. : -1.;
double phi1 = sign*m_deltaPhi; 
try{
dg::Adaptive<dg::ERKStep<std::array<double,3>>> adapt(
"Dormand-Prince-7-4-5", coords);
dg::AdaptiveTimeloop<std::array<double,3>> odeint( adapt,
m_cyl_field, dg::pid_control, dg::fast_l2norm, m_eps,
1e-10);
odeint.integrate_in_domain( 0., coords, phi1, coordsP, 0.,
m_domain, m_eps);
}catch (std::exception& e)
{
phi1 = sign*m_deltaPhi;
coordsP[2] = 1e6*phi1;
}
if( m_type == "phi")
return sign*phi1;
return coordsP[2];
}

private:
const dg::Grid2d m_domain;
dg::geo::detail::DSFieldCylindrical3 m_cyl_field;
double m_deltaPhi, m_eps;
std::string m_type;
};


struct WallFieldlineCoordinate : public aCylindricalFunctor<WallFieldlineCoordinate>
{
WallFieldlineCoordinate(
const dg::geo::CylindricalVectorLvl0& vec,
const dg::aRealTopology2d<double>& domain,
double maxPhi, double eps, std::string type) :
m_domain( domain), m_cyl_field(vec),
m_deltaPhi( maxPhi), m_eps( eps), m_type(type)
{
if( m_type != "phi" && m_type != "s")
throw std::runtime_error( "Distance type "+m_type+" not recognized!\n");
}
double do_compute( double R, double Z) const
{
double phiP = m_deltaPhi, phiM = -m_deltaPhi;
std::array<double,3> coords{ R, Z, 0}, coordsP(coords), coordsM(coords);
m_cyl_field( 0., coords, coordsP);
double sign = coordsP[2] > 0 ? +1. : -1.;
try{
dg::AdaptiveTimeloop<std::array<double,3>> odeint(
dg::Adaptive<dg::ERKStep<std::array<double,3>>>(
"Dormand-Prince-7-4-5", coords), m_cyl_field,
dg::pid_control, dg::fast_l2norm, m_eps, 1e-10);
odeint.integrate_in_domain( 0., coords, phiP, coordsP, 0.,
m_domain, m_eps);
odeint.integrate_in_domain( 0., coords, phiM, coordsM, 0.,
m_domain, m_eps);
}catch (std::exception& e)
{
phiP = m_deltaPhi;
coordsP[2] = 1e6*phiP;
phiM = -m_deltaPhi;
coordsM[2] = 1e6*phiM;
}
if( m_type == "phi")
return sign*(-phiP-phiM)/(phiP-phiM);
double sP = coordsP[2], sM = coordsM[2];
double value = sign*(-sP-sM)/(sP-sM);
if( (phiM <= -m_deltaPhi)  && (phiP >= m_deltaPhi))
return 0.; 
if( (phiM <= -m_deltaPhi))
return value*sign > 0 ? value : 0.; 
if( (phiP >= m_deltaPhi))
return value*sign < 0 ? value : 0.; 
return value;
}

private:
const dg::Grid2d m_domain;
dg::geo::detail::DSFieldCylindrical3 m_cyl_field;
double m_deltaPhi, m_eps;
std::string m_type;
};





template<class ProductGeometry, class IMatrix, class container >
struct Fieldaligned
{

Fieldaligned(){}
template <class Limiter>
Fieldaligned(const dg::geo::TokamakMagneticField& vec,
const ProductGeometry& grid,
dg::bc bcx = dg::NEU,
dg::bc bcy = dg::NEU,
Limiter limit = FullLimiter(),
double eps = 1e-5,
unsigned mx=12, unsigned my=12,
double deltaPhi = -1,
std::string interpolation_method = "linear-nearest",
bool benchmark=true
):
Fieldaligned( dg::geo::createBHat(vec),
grid, bcx, bcy, limit, eps, mx, my, deltaPhi, interpolation_method)
{
}

template <class Limiter>
Fieldaligned(const dg::geo::CylindricalVectorLvl1& vec,
const ProductGeometry& grid,
dg::bc bcx = dg::NEU,
dg::bc bcy = dg::NEU,
Limiter limit = FullLimiter(),
double eps = 1e-5,
unsigned mx=12, unsigned my=12,
double deltaPhi = -1,
std::string interpolation_method = "linear-nearest",
bool benchmark=true
);

template<class ...Params>
void construct( Params&& ...ps)
{
*this = Fieldaligned( std::forward<Params>( ps)...);
}

dg::bc bcx()const{
return m_bcx;
}
dg::bc bcy()const{
return m_bcy;
}



void set_boundaries( dg::bc bcz, double left, double right)
{
m_bcz = bcz;
const dg::Grid1d g2d( 0, 1, 1, m_perp_size);
m_left  = dg::evaluate( dg::CONSTANT(left), g2d);
m_right = dg::evaluate( dg::CONSTANT(right),g2d);
}


void set_boundaries( dg::bc bcz, const container& left, const container& right)
{
m_bcz = bcz;
m_left = left;
m_right = right;
}


void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
{
dg::split( global, m_f, *m_g);
dg::blas1::axpby( scal_left,  m_f[0],      0, m_left);
dg::blas1::axpby( scal_right, m_f[m_Nz-1], 0, m_right);
m_bcz = bcz;
}


void operator()(enum whichMatrix which, const container& in, container& out);

double deltaPhi() const{return m_deltaPhi;}
const container& hbm()const {
return m_hbm;
}
const container& hbp()const {
return m_hbp;
}
const container& sqrtG()const {
return m_G;
}
const container& sqrtGm()const {
return m_Gm;
}
const container& sqrtGp()const {
return m_Gp;
}
const container& bphi()const {
return m_bphi;
}
const container& bphiM()const {
return m_bphiM;
}
const container& bphiP()const {
return m_bphiP;
}
const container& bbm()const {
return m_bbm;
}
const container& bbo()const {
return m_bbo;
}
const container& bbp()const {
return m_bbp;
}
const ProductGeometry& grid()const{return *m_g;}


container interpolate_from_coarse_grid( const ProductGeometry& grid_coarse, const container& coarse);

void integrate_between_coarse_grid( const ProductGeometry& grid_coarse, const container& coarse, container& out );



template< class BinaryOp, class UnaryOp>
container evaluate( BinaryOp binary, UnaryOp unary,
unsigned p0, unsigned rounds) const;

std::string method() const{return m_interpolation_method;}

private:
void ePlus( enum whichMatrix which, const container& in, container& out);
void eMinus(enum whichMatrix which, const container& in, container& out);
void zero( enum whichMatrix which, const container& in, container& out);
IMatrix m_plus, m_zero, m_minus, m_plusT, m_minusT; 
container m_hbm, m_hbp;         
container m_G, m_Gm, m_Gp; 
container m_bphi, m_bphiM, m_bphiP; 
container m_bbm, m_bbp, m_bbo;  

container m_left, m_right;      
container m_limiter;            
container m_ghostM, m_ghostP;   
unsigned m_Nz, m_perp_size;
dg::bc m_bcx, m_bcy, m_bcz;
std::vector<dg::View<const container>> m_f;
std::vector<dg::View< container>> m_temp;
dg::ClonePtr<ProductGeometry> m_g;
dg::InverseKroneckerTriDiagonal2d<container> m_inv_linear;
double m_deltaPhi;
std::string m_interpolation_method;

bool m_have_adjoint = false;
void updateAdjoint( )
{
m_plusT = dg::transpose( m_plus);
m_minusT = dg::transpose( m_minus);
m_have_adjoint = true;
}
};


template<class Geometry, class IMatrix, class container>
template <class Limiter>
Fieldaligned<Geometry, IMatrix, container>::Fieldaligned(
const dg::geo::CylindricalVectorLvl1& vec,
const Geometry& grid,
dg::bc bcx, dg::bc bcy, Limiter limit, double eps,
unsigned mx, unsigned my, double deltaPhi, std::string interpolation_method, bool benchmark) :
m_g(grid),
m_interpolation_method(interpolation_method)
{

std::string inter_m, project_m, fine_m;
detail::parse_method( interpolation_method, inter_m, project_m, fine_m);
if( benchmark) std::cout << "# Interpolation method: \""<<inter_m << "\" projection method: \""<<project_m<<"\" fine grid \""<<fine_m<<"\"\n";
if( (grid.bcx() == PER && bcx != PER) || (grid.bcx() != PER && bcx == PER) )
throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting periodicity in x. The grid says "<<bc2str(grid.bcx())<<" while the parameter says "<<bc2str(bcx)));
if( (grid.bcy() == PER && bcy != PER) || (grid.bcy() != PER && bcy == PER) )
throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting boundary conditions in y. The grid says "<<bc2str(grid.bcy())<<" while the parameter says "<<bc2str(bcy)));
m_Nz=grid.Nz(), m_bcx = bcx, m_bcy = bcy, m_bcz=grid.bcz();
if( deltaPhi <=0) deltaPhi = grid.hz();
dg::Timer t;
if( benchmark) t.tic();
dg::ClonePtr<dg::aGeometry2d> grid_transform( grid.perp_grid()) ;
dg::RealGrid2d<double> grid_equidist( *grid_transform) ;
dg::RealGrid2d<double> grid_fine( *grid_transform);
grid_equidist.set( 1, grid.gx().size(), grid.gy().size());
dg::ClonePtr<dg::aGeometry2d> grid_magnetic = grid_transform;
grid_magnetic->set( grid_transform->n() < 3 ? 4 : 7, grid_magnetic->Nx(), grid_magnetic->Ny());
if( project_m != "dg" && fine_m == "dg")
{
unsigned rx = mx % grid.nx(), ry = my % grid.ny();
if( 0 != rx || 0 != ry)
{
std::cerr << "#Warning: for projection method \"const\" mx and my must be multiples of nx and ny! Rounding up for you ...\n";
mx = mx + grid.nx() - rx;
my = my + grid.ny() - ry;
}
}
if( fine_m == "equi")
grid_fine = grid_equidist;
grid_fine.multiplyCellNumbers((double)mx, (double)my);
if( benchmark)
{
t.toc();
std::cout << "# DS: High order grid gen  took: "<<t.diff()<<"\n";
t.tic();
}
std::array<thrust::host_vector<double>,3> yp_trafo, ym_trafo, yp, ym;
thrust::host_vector<bool> in_boxp, in_boxm;
thrust::host_vector<double> hbp, hbm;
thrust::host_vector<double> vol = dg::tensor::volume(grid.metric()), vol2d0;
auto vol2d = dg::split( vol, grid);
dg::assign( vol2d[0], vol2d0);
detail::integrate_all_fieldlines2d( vec, *grid_magnetic, *grid_transform,
yp_trafo, vol2d0, hbp, in_boxp, deltaPhi, eps);
detail::integrate_all_fieldlines2d( vec, *grid_magnetic, *grid_transform,
ym_trafo, vol2d0, hbm, in_boxm, -deltaPhi, eps);
dg::HVec Xf = dg::evaluate(  dg::cooX2d, grid_fine);
dg::HVec Yf = dg::evaluate(  dg::cooY2d, grid_fine);
{
dg::IHMatrix interpolate = dg::create::interpolation( Xf, Yf,
*grid_transform, dg::NEU, dg::NEU, grid_transform->n() < 3 ? "cubic" : "dg");
yp.fill(dg::evaluate( dg::zero, grid_fine));
ym = yp;
for( int i=0; i<2; i++) 
{
dg::blas2::symv( interpolate, yp_trafo[i], yp[i]);
dg::blas2::symv( interpolate, ym_trafo[i], ym[i]);
}
} 
if( benchmark)
{
t.toc();
std::cout << "# DS: Computing all points took: "<<t.diff()<<"\n";
t.tic();
}
if( inter_m == "dg")
{
dg::IHMatrix fine, projection, multi;
if( project_m == "dg")
projection = dg::create::projection( *grid_transform, grid_fine);
else
{
multi = dg::create::projection( grid_equidist, grid_fine, project_m);
fine = dg::create::inv_backproject( *grid_transform);
cusp::multiply( fine, multi, projection);
}
fine = dg::create::interpolation( yp[0], yp[1],
*grid_transform, bcx, bcy, "dg");
cusp::multiply( projection, fine, multi);
dg::blas2::transfer( multi, m_plus);
fine = dg::create::interpolation( Xf, Yf,
*grid_transform, bcx, bcy, "dg");
cusp::multiply( projection, fine, multi);
dg::blas2::transfer( multi, m_zero);
fine = dg::create::interpolation( ym[0], ym[1],
*grid_transform, bcx, bcy, "dg");
cusp::multiply( projection, fine, multi);
dg::blas2::transfer( multi, m_minus);
}
else
{
dg::IHMatrix fine, projection, multi, temp;
if( project_m == "dg")
projection = dg::create::projection( *grid_transform, grid_fine);
else
{
multi = dg::create::projection( grid_equidist, grid_fine, project_m);
fine = dg::create::inv_backproject( *grid_transform);
cusp::multiply( fine, multi, projection);
}

fine = dg::create::backproject( *grid_transform); 

multi = dg::create::interpolation( yp[0], yp[1],
grid_equidist, bcx, bcy, inter_m);
cusp::multiply( multi, fine, temp);
cusp::multiply( projection, temp, multi);
dg::blas2::transfer( multi, m_plus);

multi = dg::create::interpolation( Xf, Yf,
grid_equidist, bcx, bcy, inter_m);
cusp::multiply( multi, fine, temp);
cusp::multiply( projection, temp, multi);
dg::blas2::transfer( multi, m_zero);

multi = dg::create::interpolation( ym[0], ym[1],
grid_equidist, bcx, bcy, inter_m);
cusp::multiply( multi, fine, temp);
cusp::multiply( projection, temp, multi);
dg::blas2::transfer( multi, m_minus);
}

if( benchmark)
{
t.toc();
std::cout << "# DS: Multiplication PI    took: "<<t.diff()<<"\n";
}
dg::HVec hbphi( yp_trafo[2]), hbphiP(hbphi), hbphiM(hbphi);
hbphi = dg::pullback( vec.z(), *grid_transform);
if( dynamic_cast<const dg::CartesianGrid2d*>( grid_transform.get()))
{
for( unsigned i=0; i<hbphiP.size(); i++)
{
hbphiP[i] = vec.z()(yp_trafo[0][i], yp_trafo[1][i]);
hbphiM[i] = vec.z()(ym_trafo[0][i], ym_trafo[1][i]);
}
}
else
{
dg::HVec Ihbphi = dg::pullback( vec.z(), *grid_magnetic);
dg::HVec Lhbphi = dg::forward_transform( Ihbphi, *grid_magnetic);
for( unsigned i=0; i<yp_trafo[0].size(); i++)
{
hbphiP[i] = dg::interpolate( dg::lspace, Lhbphi, yp_trafo[0][i],
yp_trafo[1][i], *grid_magnetic);
hbphiM[i] = dg::interpolate( dg::lspace, Lhbphi, ym_trafo[0][i],
ym_trafo[1][i], *grid_magnetic);
}
}
dg::assign3dfrom2d( hbphi,  m_bphi,  grid);
dg::assign3dfrom2d( hbphiM, m_bphiM, grid);
dg::assign3dfrom2d( hbphiP, m_bphiP, grid);

dg::assign3dfrom2d( yp_trafo[2], m_Gp, grid);
dg::assign3dfrom2d( ym_trafo[2], m_Gm, grid);
m_G = vol;
container weights = dg::create::weights( grid);
dg::blas1::pointwiseDot( m_G, weights, m_G);
dg::blas1::pointwiseDot( m_Gp, weights, m_Gp);
dg::blas1::pointwiseDot( m_Gm, weights, m_Gm);

dg::assign( dg::evaluate( dg::zero, grid), m_hbm);
m_f     = dg::split( (const container&)m_hbm, grid);
m_temp  = dg::split( m_hbm, grid);
dg::assign3dfrom2d( hbp, m_hbp, grid);
dg::assign3dfrom2d( hbm, m_hbm, grid);
dg::blas1::scal( m_hbm, -1.);

thrust::host_vector<double> bbm( in_boxp.size(),0.), bbo(bbm), bbp(bbm);
for( unsigned i=0; i<in_boxp.size(); i++)
{
if( !in_boxp[i] && !in_boxm[i])
bbo[i] = 1.;
else if( !in_boxp[i] && in_boxm[i])
bbp[i] = 1.;
else if( in_boxp[i] && !in_boxm[i])
bbm[i] = 1.;
}
dg::assign3dfrom2d( bbm, m_bbm, grid);
dg::assign3dfrom2d( bbo, m_bbo, grid);
dg::assign3dfrom2d( bbp, m_bbp, grid);

m_deltaPhi = deltaPhi; 

m_perp_size = grid_transform->size();
dg::assign( dg::pullback(limit, *grid_transform), m_limiter);
dg::assign( dg::evaluate(dg::zero, *grid_transform), m_left);
m_ghostM = m_ghostP = m_right = m_left;
}


template<class G, class I, class container>
container Fieldaligned<G, I,container>::interpolate_from_coarse_grid(
const G& grid, const container& in)
{
assert( m_g->Nz() % grid.Nz() == 0);
unsigned Nz_coarse = grid.Nz(), Nz = m_g->Nz();
unsigned cphi = Nz / Nz_coarse;

container out = dg::evaluate( dg::zero, *m_g);
container helper = dg::evaluate( dg::zero, *m_g);
dg::split( helper, m_temp, *m_g);
std::vector<dg::View< container>> out_split = dg::split( out, *m_g);
std::vector<dg::View< const container>> in_split = dg::split( in, grid);
for ( int i=0; i<(int)Nz_coarse; i++)
{
dg::blas1::copy( in_split[i], out_split[i*cphi]);
dg::blas1::copy( in_split[i], m_temp[i*cphi]);
}
for ( int i=0; i<(int)Nz_coarse; i++)
{
for( int j=1; j<(int)cphi; j++)
{
dg::blas2::symv( m_minus, out_split[i*cphi+j-1], out_split[i*cphi+j]);
dg::blas2::symv( m_plus, m_temp[(i*cphi+cphi+1-j)%Nz], m_temp[i*cphi+cphi-j]);
}
}
for( int i=0; i<(int)Nz_coarse; i++)
for( int j=1; j<(int)cphi; j++)
{
double alpha = (double)(cphi-j)/(double)cphi;
double beta = (double)j/(double)cphi;
dg::blas1::axpby( alpha, out_split[i*cphi+j], beta, m_temp[i*cphi+j], out_split[i*cphi+j]);
}
return out;
}
template<class G, class I, class container>
void Fieldaligned<G, I,container>::integrate_between_coarse_grid( const G& grid, const container& in, container& out)
{
assert( m_g->Nz() % grid.Nz() == 0);
unsigned Nz_coarse = grid.Nz(), Nz = m_g->Nz();
unsigned cphi = Nz / Nz_coarse;

out = in;
container helperP( in), helperM(in), tempP(in), tempM(in);

for( int j=1; j<(int)cphi; j++)
{
dg::blas2::symv( m_minus, helperP, tempP);
dg::blas1::axpby( (double)(cphi-j)/(double)cphi, tempP, 1., out  );
helperP.swap(tempP);
dg::blas2::symv( m_plus, helperM, tempM);
dg::blas1::axpby( (double)(cphi-j)/(double)cphi, tempM, 1., out  );
helperM.swap(tempM);
}
dg::blas1::scal( out, 1./(double)cphi);
}

template<class G, class I, class container>
void Fieldaligned<G, I, container >::operator()(enum whichMatrix which, const container& f, container& fe)
{
if(     which == einsPlus  || which == einsMinusT ) ePlus(  which, f, fe);
else if(which == einsMinus || which == einsPlusT  ) eMinus( which, f, fe);
else if(which == zeroMinus || which == zeroPlus ||
which == zeroMinusT|| which == zeroPlusT ||
which == zeroForw  ) zero(   which, f, fe);
}

template< class G, class I, class container>
void Fieldaligned<G, I, container>::zero( enum whichMatrix which,
const container& f, container& f0)
{
dg::split( f, m_f, *m_g);
dg::split( f0, m_temp, *m_g);
for( unsigned i0=0; i0<m_Nz; i0++)
{
if(which == zeroPlus)
dg::blas2::symv( m_plus,   m_f[i0], m_temp[i0]);
else if(which == zeroMinus)
dg::blas2::symv( m_minus,  m_f[i0], m_temp[i0]);
else if(which == zeroPlusT)
{
if( ! m_have_adjoint) updateAdjoint( );
dg::blas2::symv( m_plusT,  m_f[i0], m_temp[i0]);
}
else if(which == zeroMinusT)
{
if( ! m_have_adjoint) updateAdjoint( );
dg::blas2::symv( m_minusT, m_f[i0], m_temp[i0]);
}
else if( which == zeroForw)
{
if ( m_interpolation_method != "dg" )
{
dg::blas2::symv( m_zero, m_f[i0], m_temp[i0]);
}
else
dg::blas1::copy( m_f[i0], m_temp[i0]);
}
}
}
template< class G, class I, class container>
void Fieldaligned<G, I, container>::ePlus( enum whichMatrix which,
const container& f, container& fpe)
{
dg::split( f, m_f, *m_g);
dg::split( fpe, m_temp, *m_g);
for( unsigned i0=0; i0<m_Nz; i0++)
{
unsigned ip = (i0==m_Nz-1) ? 0:i0+1;
if(which == einsPlus)
dg::blas2::symv( m_plus,   m_f[ip], m_temp[i0]);
else if(which == einsMinusT)
{
if( ! m_have_adjoint) updateAdjoint( );
dg::blas2::symv( m_minusT, m_f[ip], m_temp[i0]);
}
}
unsigned i0=m_Nz-1;
if( m_bcz != dg::PER)
{
if( m_bcz == dg::DIR || m_bcz == dg::NEU_DIR)
dg::blas1::axpby( 2, m_right, -1., m_f[i0], m_ghostP);
if( m_bcz == dg::NEU || m_bcz == dg::DIR_NEU)
dg::blas1::axpby( m_deltaPhi, m_right, 1., m_f[i0], m_ghostP);
dg::blas1::axpby( 1., m_ghostP, -1., m_temp[i0], m_ghostP);
dg::blas1::pointwiseDot( 1., m_limiter, m_ghostP, 1., m_temp[i0]);
}
}

template< class G, class I, class container>
void Fieldaligned<G, I, container>::eMinus( enum whichMatrix which,
const container& f, container& fme)
{
dg::split( f, m_f, *m_g);
dg::split( fme, m_temp, *m_g);
for( unsigned i0=0; i0<m_Nz; i0++)
{
unsigned im = (i0==0) ? m_Nz-1:i0-1;
if(which == einsPlusT)
{
if( ! m_have_adjoint) updateAdjoint( );
dg::blas2::symv( m_plusT, m_f[im], m_temp[i0]);
}
else if (which == einsMinus)
dg::blas2::symv( m_minus, m_f[im], m_temp[i0]);
}
unsigned i0=0;
if( m_bcz != dg::PER)
{
if( m_bcz == dg::DIR || m_bcz == dg::DIR_NEU)
dg::blas1::axpby( 2., m_left,  -1., m_f[i0], m_ghostM);
if( m_bcz == dg::NEU || m_bcz == dg::NEU_DIR)
dg::blas1::axpby( -m_deltaPhi, m_left, 1., m_f[i0], m_ghostM);
dg::blas1::axpby( 1., m_ghostM, -1., m_temp[i0], m_ghostM);
dg::blas1::pointwiseDot( 1., m_limiter, m_ghostM, 1., m_temp[i0]);
}
}

template<class G, class I, class container>
template< class BinaryOp, class UnaryOp>
container Fieldaligned<G, I,container>::evaluate( BinaryOp binary,
UnaryOp unary, unsigned p0, unsigned rounds) const
{
assert( p0 < m_g->Nz());
const dg::ClonePtr<aGeometry2d> g2d = m_g->perp_grid();
container init2d = dg::pullback( binary, *g2d);
container zero2d = dg::evaluate( dg::zero, *g2d);

container temp(init2d), tempP(init2d), tempM(init2d);
container vec3d = dg::evaluate( dg::zero, *m_g);
std::vector<container>  plus2d(m_Nz, zero2d), minus2d(plus2d), result(plus2d);
unsigned turns = rounds;
if( turns ==0) turns++;
for( unsigned r=0; r<turns; r++)
for( unsigned i0=0; i0<m_Nz; i0++)
{
dg::blas1::copy( init2d, tempP);
dg::blas1::copy( init2d, tempM);
unsigned rep = r*m_Nz + i0;
for(unsigned k=0; k<rep; k++)
{
dg::blas2::symv( m_minus, tempP, temp);
temp.swap( tempP);
dg::blas2::symv( m_plus, tempM, temp);
temp.swap( tempM);
}
dg::blas1::scal( tempP, unary(  (double)rep*m_deltaPhi ) );
dg::blas1::scal( tempM, unary( -(double)rep*m_deltaPhi ) );
dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
}
if( rounds == 0) 
{
for( unsigned i0=0; i0<m_Nz; i0++)
{
int idx = (int)i0 - (int)p0;
if(idx>=0)
result[i0] = plus2d[idx];
else
result[i0] = minus2d[abs(idx)];
thrust::copy( result[i0].begin(), result[i0].end(), vec3d.begin() + i0*m_perp_size);
}
}
else 
{
for( unsigned i0=0; i0<m_Nz; i0++)
{
unsigned revi0 = (m_Nz - i0)%m_Nz; 
dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
}
dg::blas1::axpby( -1., init2d, 1., result[0]);
for(unsigned i0=0; i0<m_Nz; i0++)
{
int idx = ((int)i0 -(int)p0 + m_Nz)%m_Nz; 
thrust::copy( result[idx].begin(), result[idx].end(), vec3d.begin() + i0*m_perp_size);
}
}
return vec3d;
}




template<class BinaryOp, class UnaryOp>
thrust::host_vector<double> fieldaligned_evaluate(
const aProductGeometry3d& grid,
const CylindricalVectorLvl0& vec,
const BinaryOp& binary,
const UnaryOp& unary,
unsigned p0,
unsigned rounds,
double eps = 1e-5)
{
unsigned Nz = grid.Nz();
const dg::ClonePtr<aGeometry2d> g2d = grid.perp_grid();
dg::HVec tempP = dg::evaluate( dg::zero, *g2d), tempM( tempP);
std::vector<dg::HVec>  plus2d(Nz, tempP), minus2d(plus2d), result(plus2d);
dg::HVec vec3d = dg::evaluate( dg::zero, grid);
dg::HVec init2d = dg::pullback( binary, *g2d);
std::array<dg::HVec,3> yy0{
dg::pullback( dg::cooX2d, *g2d),
dg::pullback( dg::cooY2d, *g2d),
dg::evaluate( dg::zero, *g2d)}, yy1(yy0), xx0( yy0), xx1(yy0); 
dg::geo::detail::DSFieldCylindrical3 cyl_field(vec);
double deltaPhi = grid.hz();
double phiM0 = 0., phiP0 = 0.;
unsigned turns = rounds;
if( turns == 0) turns++;
for( unsigned r=0; r<turns; r++)
for( unsigned  i0=0; i0<Nz; i0++)
{
unsigned rep = r*Nz + i0;
if( rep == 0)
tempM = tempP = init2d;
else
{
dg::Adaptive<dg::ERKStep<std::array<double,3>>> adapt(
"Dormand-Prince-7-4-5", std::array<double,3>{0,0,0});
dg::AdaptiveTimeloop<std::array<double,3>> odeint( adapt,
cyl_field, dg::pid_control, dg::fast_l2norm, eps,
1e-10);
for( unsigned i=0; i<g2d->size(); i++)
{
double phiM1 = phiM0 + deltaPhi;
std::array<double,3>
coords0{yy0[0][i],yy0[1][i],yy0[2][i]}, coords1;
odeint.integrate_in_domain( phiM0, coords0, phiM1, coords1,
deltaPhi, *g2d, eps);
yy1[0][i] = coords1[0], yy1[1][i] = coords1[1], yy1[2][i] =
coords1[2];
tempM[i] = binary( yy1[0][i], yy1[1][i]);

double phiP1 = phiP0 - deltaPhi;
coords0 = std::array<double,3>{xx0[0][i],xx0[1][i],xx0[2][i]};
odeint.integrate_in_domain( phiP0, coords0, phiP1, coords1,
-deltaPhi, *g2d, eps);
xx1[0][i] = coords1[0], xx1[1][i] = coords1[1], xx1[2][i] =
coords1[2];
tempP[i] = binary( xx1[0][i], xx1[1][i]);
}
std::swap( yy0, yy1);
std::swap( xx0, xx1);
phiM0 += deltaPhi;
phiP0 -= deltaPhi;
}
dg::blas1::scal( tempM, unary( -(double)rep*deltaPhi ) );
dg::blas1::scal( tempP, unary(  (double)rep*deltaPhi ) );
dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
}
if( rounds == 0) 
{
for( unsigned i0=0; i0<Nz; i0++)
{
int idx = (int)i0 - (int)p0;
if(idx>=0)
result[i0] = plus2d[idx];
else
result[i0] = minus2d[abs(idx)];
thrust::copy( result[i0].begin(), result[i0].end(), vec3d.begin() +
i0*g2d->size());
}
}
else 
{
for( unsigned i0=0; i0<Nz; i0++)
{
unsigned revi0 = (Nz - i0)%Nz; 
dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
}
dg::blas1::axpby( -1., init2d, 1., result[0]);
for(unsigned i0=0; i0<Nz; i0++)
{
int idx = ((int)i0 -(int)p0 + Nz)%Nz; 
thrust::copy( result[idx].begin(), result[idx].end(), vec3d.begin()
+ i0*g2d->size());
}
}
return vec3d;
}

}
}
