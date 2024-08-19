#pragma once

#include "dg/algorithm.h"
#include "fieldaligned.h"
#ifdef MPI_VERSION
#include "mpi_fieldaligned.h"
#endif 
#include "magnetic_field.h"



namespace dg{
namespace geo{










namespace detail
{
struct DSCentered
{
DSCentered( double alpha, double beta) : m_alpha(alpha), m_beta(beta){}
DG_DEVICE
void operator()( double& dsf, double fm, double fo, double fp, double hm,
double hp)
{
dsf = m_alpha*(
fm*( 1./(hp+hm) - 1./hm) +
fo*( 1./hm - 1./hp) +
fp*( 1./hp - 1./(hp+hm))
) + m_beta*dsf;
};

private:
double m_alpha, m_beta;
};
struct DSSCentered
{
DSSCentered( double alpha, double beta, double delta) : m_alpha(alpha), m_beta(beta), m_delta(delta){}
DG_DEVICE
void operator()( double& dssf, double fm, double fo, double fp, double hm,
double hp)
{
dssf = m_alpha*(
2.*fm/(hp+hm)/hm - 2.*fo/hp/hm + 2.*fp/(hp+hm)/hp
) + m_beta*dssf;
};

DG_DEVICE
void operator()( double& dssf, double fm, double fo, double fp,
double bPm, double bP0, double bPp)
{
double bP2 = (bPp+bP0)/2.;
double bM2 = (bPm+bP0)/2.;
double fm2 = (fo-fm)/m_delta;
double fp2 = (fp-fo)/m_delta;

dssf = m_alpha*bP0*( bP2*fp2 - bM2*fm2)/m_delta + m_beta*dssf;
}

private:
double m_alpha, m_beta, m_delta;
};
struct DSSDCentered
{
DSSDCentered( double alpha, double beta, double delta) :
m_alpha( alpha), m_beta(beta), m_delta(delta){}
DG_DEVICE
void operator()( double& dssdf, double fm, double fo, double fp, double Gm,
double Go, double Gp, double bPm, double bP0, double bPp)
{
double bP2 = (bPp+bP0)/2.;
double bM2 = (bPm+bP0)/2.;
double fm2 = (fo-fm)/m_delta;
double fp2 = (fp-fo)/m_delta;
double gp2 = (Gp + Go)/Go/2.;
double gm2 = (Gm + Go)/Go/2.;

dssdf = m_alpha*( gp2*fp2*bP2*bP2 - bM2*bM2*gm2*fm2)/m_delta + m_beta*dssdf;

};

private:
double m_alpha, m_beta, m_delta;
};

}


template<class FieldAligned, class container>
void assign_bc_along_field_2nd( const FieldAligned& fa, const container& fm,
const container& f, const container& fp, container& fmg, container& fpg,
dg::bc bound, std::array<double,2> boundary_value = {0,0})
{
double delta = fa.deltaPhi();
if( bound == dg::NEU)
{
double dbm = boundary_value[0], dbp = boundary_value[1];
dg::blas1::subroutine( [dbm, dbp, delta]DG_DEVICE( double fm, double fo,
double fp, double& fmg, double& fpg,
double hbm, double hbp, double bbm, double bbo, double bbp
){
double hm = delta, hp = delta;
double plus=0, minus=0, bothP=0, bothM = 0;
plus = dbp*hp*(hm+hp)/(2.*hbp+hm) +
fo*(2.*hbp+hm-hp)*(hm+hp)/hm/(2.*hbp+hm) + fm*hp*(-2.*hbp +
hp)/hm/(2.*hbp + hm);
minus = fp*hm*(-2.*hbm+hm)/hp/(2.*hbm+hp) -
dbm*hm*(hm+hp)/(2.*hbm+hp) +
fo*(2.*hbm-hm+hp)*(hm+hp)/hp/(2.*hbm+hp);
bothM = fo + dbp*hm*(-2.*hbm + hm)/2./(hbm+hbp) -
dbm*hm*(2.*hbp+hm)/2./(hbm+hbp);
bothP = fo + dbp*hp*(2.*hbm + hp)/2./(hbm+hbp) +
dbm*hp*(2.*hbp-hp)/2./(hbm+hbp);
fmg = (1.-bbo-bbm)*fm + bbm*minus + bbo*bothM;
fpg = (1.-bbo-bbp)*fp + bbp*plus  + bbo*bothP;
}, fm, f, fp, fmg, fpg, fa.hbm(), fa.hbp(), fa.bbm(),
fa.bbo(), fa.bbp() );
}
else
{
double fbm = boundary_value[0], fbp = boundary_value[1];
dg::blas1::subroutine( [fbm, fbp, delta]DG_DEVICE( double fm, double fo,
double fp, double& fmg, double& fpg,
double hbm, double hbp, double bbm, double bbo, double bbp
){
double hm = delta, hp = delta;
double plus=0, minus=0, bothP=0, bothM = 0;
plus  = fm*hp*(-hbp + hp)/hm/(hbp+hm) + fo*(hbp-hp)*(hm+hp)/hbp/hm
+fbp*hp*(hm+hp)/hbp/(hbp+hm);
minus = +fo*(hbm-hm)*(hm+hp)/hbm/hp + fbm*hm*(hm+hp)/hbm/(hbm+hp)
+ fp*hm*(-hbm+hm)/hp/(hbm+hp);
bothM = fbp*hm*(-hbm+hm)/hbp/(hbm+hbp) +
fo*(hbm-hm)*(hbp+hm)/hbm/hbp + fbm*hm*(hbp+hm)/hbm/(hbm+hbp);
bothP = fo*(hbp-hp)*(hbm+hp)/hbm/hbp +
fbp*hp*(hbm+hp)/hbp/(hbm+hbp) + fbm*hp*(-hbp+hp)/hbm/(hbm+hbp);
fmg = (1.-bbo-bbm)*fm + bbm*minus + bbo*bothM;
fpg = (1.-bbo-bbp)*fp + bbp*plus  + bbo*bothP;
}, fm, f, fp, fmg, fpg, fa.hbm(), fa.hbp(), fa.bbm(),
fa.bbo(), fa.bbp());
}

}

template<class FieldAligned, class container>
void assign_bc_along_field_1st( const FieldAligned& fa, const container& fm,
const container& fp, container& fmg, container& fpg,
dg::bc bound, std::array<double,2> boundary_value = {0,0})
{
double delta = fa.deltaPhi();
if( bound == dg::NEU)
{
double dbm = boundary_value[0], dbp = boundary_value[1];
dg::blas1::subroutine( [dbm, dbp, delta]DG_DEVICE( double fm, double fp,
double& fmg, double& fpg, double bbm, double bbp
){
double hm = delta, hp = delta;
double plus=0, minus=0;
plus = fm + dbp*(hp+hm);
minus = fp - dbm*(hp+hm);
fmg = (1.-bbm)*fm + bbm*minus;
fpg = (1.-bbp)*fp + bbp*plus;
}, fm, fp, fmg, fpg, fa.bbm(), fa.bbp() );
}
else
{
double fbm = boundary_value[0], fbp = boundary_value[1];
dg::blas1::subroutine( [fbm, fbp, delta]DG_DEVICE( double fm, double fp,
double& fmg, double& fpg, double hbm,
double hbp, double bbm, double bbo, double bbp
){
double hm = delta, hp = delta;
double plus=0, minus=0, bothP=0, bothM = 0;
plus  = fm + (fbp-fm)/(hbp+hm)*(hp+hm) ;
minus = fp - (hp+hm)*(fp-fbm)/(hp+hbm);
bothM = fbp + (fbp-fbm)/(hbp+hbm)*(hp+hbm);
bothP = fbp - (fbp-fbm)/(hbp+hbm)*(hbp+hm);
fmg = (1.-bbo-bbm)*fm + bbm*minus + bbo*bothM;
fpg = (1.-bbo-bbp)*fp + bbp*plus  + bbo*bothP;
}, fm, fp, fmg, fpg, fa.hbm(), fa.hbp(), fa.bbm(),
fa.bbo(), fa.bbp());
}
}


template<class FieldAligned, class container>
void swap_bc_perp( const FieldAligned& fa, const container& fm,
const container& fp, container& fmg, container& fpg)
{
dg::blas1::subroutine( []DG_DEVICE( double fm, double fp,
double& fmg, double& fpg,
double bbm, double bbo, double bbp
){
fmg = (1.-bbo-bbm)*fm + (bbm+bbo)*(-fm);
fpg = (1.-bbo-bbp)*fp + (bbp+bbo)*(-fp);
}, fm, fp, fmg, fpg, fa.bbm(), fa.bbo(), fa.bbp() );

}




template< class ProductGeometry, class IMatrix, class Matrix, class container >
struct DS
{
typedef dg::geo::Fieldaligned<ProductGeometry, IMatrix, container> FA; 
DS(){}


template <class Limiter>
DS(const dg::geo::TokamakMagneticField& vec, const ProductGeometry& grid,
dg::bc bcx = dg::NEU,
dg::bc bcy = dg::NEU,
Limiter limit = FullLimiter(),
double eps = 1e-5,
unsigned mx=10, unsigned my=10,
double deltaPhi=-1, std::string interpolation_method = "dg",
bool benchmark=true):
DS( FA( vec, grid, bcx, bcy, limit, eps, mx, my, deltaPhi,
interpolation_method, benchmark) )
{
}

template<class Limiter>
DS(const dg::geo::CylindricalVectorLvl1& vec, const ProductGeometry& grid,
dg::bc bcx = dg::NEU,
dg::bc bcy = dg::NEU,
Limiter limit = FullLimiter(),
double eps = 1e-5,
unsigned mx=10, unsigned my=10,
double deltaPhi=-1, std::string interpolation_method = "dg",
bool benchmark=true):
DS( FA( vec, grid, bcx, bcy, limit, eps, mx, my, deltaPhi,
interpolation_method, benchmark))
{
}

DS( FA fieldaligned);

template<class ...Params>
void construct( Params&& ...ps)
{
*this = DS( std::forward<Params>( ps)...);
}

void set_boundaries( dg::bc bcz, double left, double right){
m_fa.set_boundaries( bcz, left, right);
}
void set_boundaries( dg::bc bcz, const container& left, const container& right){
m_fa.set_boundaries( bcz, left, right);
}
void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right){
m_fa.set_boundaries( bcz, global, scal_left, scal_right);
}


void forward( double alpha, const container& f, double beta, container& g){
m_fa(zeroForw, f, m_tempO);
m_fa(einsPlus, f, m_tempP);
ds_forward( m_fa, alpha, m_tempO, m_tempP, beta, g);
}

void forward2( double alpha, const container& f, double beta, container& g){
m_fa(zeroForw, f, m_tempO);
m_fa(einsPlus, f, m_tempP);
m_fa(einsPlus, m_tempP, m_tempM);
ds_forward2( m_fa, alpha, m_tempO, m_tempP, m_tempM, beta, g);
}

void backward( double alpha, const container& f, double beta, container& g){
m_fa(einsMinus, f, m_tempM);
m_fa(zeroForw, f, m_tempO);
ds_backward( m_fa, alpha, m_tempM, m_tempO, beta, g);
}

void backward2( double alpha, const container& f, double beta, container& g){
m_fa(einsMinus, f, m_tempM);
m_fa(einsMinus, m_tempM, m_tempP);
m_fa(zeroForw, f, m_tempO);
ds_backward2( m_fa, alpha, m_tempP, m_tempM, m_tempO, beta, g);
}

void centered( double alpha, const container& f, double beta, container& g){
m_fa(einsPlus,  f, m_tempP);
m_fa(einsMinus, f, m_tempM);
ds_centered( m_fa, alpha, m_tempM, m_tempP, beta, g);
}
void centered_bc_along_field(
double alpha, const container& f, double beta, container& g, dg::bc bound,
std::array<double,2> boundary_value = {0,0}){
m_fa(einsPlus,  f, m_tempP);
m_fa(einsMinus, f, m_tempM);
assign_bc_along_field_2nd( m_fa, m_tempM, f, m_tempP, m_tempM, m_tempP,
bound, boundary_value);
ds_centered( m_fa, alpha, m_tempM, m_tempP, beta, g);

}

void backward( const container& f, container& g){
backward(1., f,0.,g);
}

void forward( const container& f, container& g){
forward(1.,f, 0.,g);
}

void centered( const container& f, container& g){
centered(1.,f,0.,g);
}

void divForward( double alpha, const container& f, double beta, container& g){
m_fa(einsPlus,  f, m_tempP);
m_fa(zeroForw,  f, m_tempO);
ds_divForward( m_fa, alpha, m_tempO, m_tempP, beta, g);
}
void divBackward( double alpha, const container& f, double beta, container& g){
m_fa(einsMinus,  f, m_tempM);
m_fa(zeroForw,  f, m_tempO);
ds_divBackward( m_fa, alpha, m_tempM, m_tempO, beta, g);
}
void divCentered(double alpha, const container& f, double beta, container& g){
m_fa(einsPlus,  f, m_tempP);
m_fa(einsMinus, f, m_tempM);
ds_divCentered( m_fa, alpha, m_tempM, m_tempP, beta, g);
}
void divForward(const container& f, container& g){
divForward( 1.,f,0.,g);
}
void divBackward(const container& f, container& g){
divBackward( 1.,f,0.,g);
}
void divCentered(const container& f, container& g){
divCentered( 1.,f,0.,g);
}


void ds(dg::direction dir,  const container& f, container& g){
ds(dir, 1., f, 0., g);
}

void ds(dg::direction dir, double alpha, const container& f, double beta, container& g);

void div(dg::direction dir,  const container& f, container& g){
div(dir, 1., f, 0., g);
}

void div(dg::direction dir, double alpha, const container& f, double beta, container& g);


void symv( const container& f, container& g){ symv( 1., f, 0., g);}

void symv( double alpha, const container& f, double beta, container& g);

void dss( const container& f, container& g){
dss( 1., f, 0., g);}

void dss( double alpha, const container& f, double beta, container& g){
m_fa(einsPlus, f, m_tempP);
m_fa(einsMinus, f, m_tempM);
m_fa(zeroForw,  f, m_tempO);
dss_centered( m_fa, alpha, m_tempM, m_tempO, m_tempP, beta, g);
}
void dss_bc_along_field(
double alpha, const container& f, double beta, container& g, dg::bc bound,
std::array<double,2> boundary_value = {0,0}){
m_fa(einsPlus, f, m_tempP);
m_fa(einsMinus, f, m_tempM);
m_fa(zeroForw,  f, m_tempO);
assign_bc_along_field_2nd( m_fa, m_tempM, m_tempO, m_tempP, m_tempM, m_tempP,
bound, boundary_value);
dss_centered( m_fa, alpha, m_tempM, m_tempO, m_tempP, beta, g);
}
void dssd( double alpha, const container& f, double
beta, container& g){
m_fa(einsPlus, f, m_tempP);
m_fa(einsMinus, f, m_tempM);
m_fa(zeroForw,  f, m_tempO);
dssd_centered( m_fa, alpha, m_tempM, m_tempO, m_tempP, beta, g);
}
void dssd_bc_along_field( double alpha, const
container& f, double beta, container& g, dg::bc bound,
std::array<double,2> boundary_value = {0,0}){
m_fa(einsPlus, f, m_tempP);
m_fa(einsMinus, f, m_tempM);
m_fa(zeroForw,  f, m_tempO);
assign_bc_along_field_2nd( m_fa, m_tempM, m_tempO, m_tempP, m_tempM, m_tempP,
bound, boundary_value);
dssd_centered( m_fa, alpha, m_tempM, f, m_tempP, beta, g);
}

void set_jfactor( double new_jfactor) {m_jfactor = new_jfactor;}

double get_jfactor() const {return m_jfactor;}

const container& weights()const {
return m_fa.sqrtG();
}


FA& fieldaligned(){return m_fa;}
const FA& fieldaligned()const{return m_fa;}
private:
Fieldaligned<ProductGeometry, IMatrix, container> m_fa;
container m_tempP, m_tempO, m_tempM;
Matrix m_jumpX, m_jumpY;
double m_jfactor = 1.;
};


template<class Geometry, class I, class M, class container>
DS<Geometry, I, M,container>::DS( Fieldaligned<Geometry, I, container> fa): m_fa(fa)
{
dg::blas2::transfer( dg::create::jumpX( fa.grid(), fa.bcx()), m_jumpX);
dg::blas2::transfer( dg::create::jumpY( fa.grid(), fa.bcy()), m_jumpY);
m_tempP = fa.sqrtG(), m_tempM = m_tempO = m_tempP;
}

template<class G, class I, class M, class container>
inline void DS<G,I,M,container>::ds( dg::direction dir, double alpha,
const container& f, double beta, container& dsf) {
switch( dir){
case dg::centered:
return centered( alpha, f, beta, dsf);
case dg::forward:
return forward( alpha, f, beta, dsf);
case dg::backward:
return backward( alpha, f, beta, dsf);
}
}
template<class G, class I, class M, class container>
inline void DS<G,I,M,container>::div( dg::direction dir, double alpha,
const container& f, double beta, container& dsf) {
switch( dir){
case dg::centered:
return divCentered( alpha, f, beta, dsf);
case dg::forward:
return divForward( alpha, f, beta, dsf);
case dg::backward:
return divBackward( alpha, f, beta, dsf);
}
}


template<class G,class I, class M, class container>
void DS<G,I,M,container>::symv( double alpha, const container& f, double beta, container& dsTdsf)
{
dssd( alpha, f, beta, dsTdsf);
if( m_jfactor !=0 && m_fa.method() == "dg")
{
dg::blas2::symv( -m_jfactor*alpha, m_jumpX, f, 1., dsTdsf);
dg::blas2::symv( -m_jfactor*alpha, m_jumpY, f, 1., dsTdsf);
}
};


template<class FieldAligned, class container>
void ds_forward(const FieldAligned& fa, double alpha, const container& f,
const container& fp, double beta, container& g)
{
double delta = fa.deltaPhi();
dg::blas1::subroutine( [ alpha, beta, delta]DG_DEVICE(
double& dsf, double fo, double fp, double bphi){
dsf = alpha*bphi*( fp - fo)/delta + beta*dsf;
},
g, f, fp, fa.bphi());
}

template<class FieldAligned, class container>
void ds_forward2(const FieldAligned& fa, double alpha, const container& f,
const container& fp, const container& fpp, double beta, container& g)
{
double delta = fa.deltaPhi();
dg::blas1::subroutine( [ alpha, beta, delta]DG_DEVICE(
double& dsf, double fo, double fp, double fpp, double bphi){
dsf = alpha*bphi*( -3.*fo + 4.*fp - fpp)/2./delta
+ beta*dsf;
},
g, f, fp, fpp, fa.bphi());
}


template<class FieldAligned, class container>
void ds_backward( const FieldAligned& fa, double alpha, const container& fm,
const container& f, double beta, container& g)
{
double delta = fa.deltaPhi();
dg::blas1::subroutine( [ alpha, beta, delta] DG_DEVICE(
double& dsf, double fo, double fm, double bphi){
dsf = alpha*bphi*( fo - fm)/delta + beta*dsf;
},
g, f, fm, fa.bphi());

}

template<class FieldAligned, class container>
void ds_backward2( const FieldAligned& fa, double alpha, const container& fmm, const container& fm, const container& f, double beta, container& g)
{
double delta = fa.deltaPhi();
dg::blas1::subroutine( [ alpha, beta, delta] DG_DEVICE(
double& dsf, double fo, double fm,  double fmm, double bphi){
dsf = alpha*bphi*( 3.*fo - 4.*fm + fmm)/2./delta
+ beta*dsf;
},
g, f, fm, fmm, fa.bphi());

}



template<class FieldAligned, class container>
void ds_centered( const FieldAligned& fa, double alpha, const container& fm,
const container& fp, double beta, container& g)
{
double delta=fa.deltaPhi();
dg::blas1::subroutine( [alpha,beta,delta]DG_DEVICE( double& g, double fm,
double fp, double bphi){
g = alpha*bphi*(fp-fm)/2./delta + beta*g;
}, g, fm, fp, fa.bphi());
}

template<class FieldAligned, class container>
void dss_centered( const FieldAligned& fa, double alpha, const container& fm,
const container& f, const container& fp, double beta, container& g)
{
dg::blas1::subroutine( detail::DSSCentered( alpha, beta, fa.deltaPhi()),
g, fm, f, fp, fa.bphiM(), fa.bphi(), fa.bphiP());
}

template<class FieldAligned, class container>
void dssd_centered( const FieldAligned& fa, double alpha, const container& fm,
const container& f, const container& fp, double beta, container& g)
{
dg::blas1::subroutine( detail::DSSDCentered( alpha, beta, fa.deltaPhi()),
g, fm, f, fp, fa.sqrtGm(), fa.sqrtG(), fa.sqrtGp(),
fa.bphiM(), fa.bphi(), fa.bphiP());
}


template<class FieldAligned, class container>
void ds_divBackward( const FieldAligned& fa, double alpha, const container& fm,
const container& f, double beta, container& g)
{
double delta = fa.deltaPhi();
dg::blas1::subroutine( [alpha,beta,delta] DG_DEVICE( double& dsf, double f0,
double f1, double Gm, double G0, double bPm, double bP0){
dsf = alpha*(bP0*G0*f0 - bPm*Gm*f1)/G0/delta + beta*dsf; },
g, f, fm, fa.sqrtGm(), fa.sqrtG(), fa.bphiM(), fa.bphi());
}


template<class FieldAligned, class container>
void ds_divForward( const FieldAligned& fa, double alpha, const container& f,
const container& fp, double beta, container& g)
{
double delta = fa.deltaPhi();
dg::blas1::subroutine( [alpha,beta,delta] DG_DEVICE( double& dsf, double f0,
double f1, double Gp, double G0, double bPp, double bP0){
dsf = alpha*(bPp*Gp*f1 - bP0*G0*f0)/G0/delta + beta*dsf; },
g, f, fp, fa.sqrtGp(), fa.sqrtG(), fa.bphiP(), fa.bphi());
}

template<class FieldAligned, class container>
void ds_divCentered( const FieldAligned& fa, double alpha, const container& fm, const container& fp,
double beta, container& g)
{
double delta = fa.deltaPhi();
dg::blas1::subroutine( [alpha,beta,delta]DG_DEVICE( double& dsf, double fm,
double fp, double Gm, double Gp, double G0,
double bPm, double bP0, double bPp)
{
dsf = alpha*( fp*Gp*bPp - fm*Gm*bPm )/G0/2./delta + beta*dsf;
}, g, fm, fp, fa.sqrtGm(),
fa.sqrtGp(), fa.sqrtG(), fa.bphiM(), fa.bphi(), fa.bphiP());

}


template<class FieldAligned, class container>
void ds_average( const FieldAligned& fa, double alpha,
const container& fm, const container& fp, double beta, container& g)
{
dg::blas1::subroutine( [alpha,beta]DG_DEVICE( double& g, double fm, double fp
){
g = alpha*(fp+fm)/2. + beta*g;
}, g, fm, fp);
}

template<class FieldAligned, class container>
void ds_slope( const FieldAligned& fa, double alpha,
const container& fm, const container& fp, double beta, container& g)
{
ds_centered( fa, alpha, fm, fp, beta, g);
}


}

template< class G, class I, class M, class V>
struct TensorTraits< geo::DS<G,I,M, V> >
{
using value_type = double;
using tensor_category = SelfMadeMatrixTag;
};
}
