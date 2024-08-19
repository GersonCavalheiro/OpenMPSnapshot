#pragma once

#include "dg/functors.h"
#include "magnetic_field.h"


namespace dg {
namespace geo{
namespace detail{
struct DSMetricCylindrical
{
DSMetricCylindrical( const dg::geo::TokamakMagneticField& mag):
m_bR(mag),
m_bZ(mag),
m_bP(mag),
m_bRR(mag),
m_bZR(mag),
m_bPR(mag),
m_bRZ(mag),
m_bZZ(mag),
m_bPZ(mag)
{}
void operator()( double t, const std::array<double,6>& y,
std::array<double,6>& yp) const {
double R = y[0], Z = y[1];
double vx = m_bR(R,Z);
double vy = m_bZ(R,Z);
double vz = m_bP(R,Z);
double vxx = m_bRR(R,Z);
double vyx = m_bZR(R,Z);
double vzx = m_bPR(R,Z);
double vxy = m_bRZ(R,Z);
double vyy = m_bZZ(R,Z);
double vzy = m_bPZ(R,Z);
yp[0] = vx/vz;
yp[1] = vy/vz;
double vxzx = (vxx/vz - vx*vzx/vz/vz);
double vyzx = (vyx/vz - vy*vzx/vz/vz);
double vxzy = (vxy/vz - vx*vzy/vz/vz);
double vyzy = (vyy/vz - vy*vzy/vz/vz);
yp[2] = - vxzx*y[2] - vyzx*y[3];
yp[3] = - vxzy*y[2] - vyzy*y[3];
yp[4] = - vxzx*y[4] - vyzx*y[5];
yp[5] = - vxzy*y[4] - vyzy*y[5];
}
private:
BHatR m_bR;
BHatZ m_bZ;
BHatP m_bP;
BHatRR m_bRR;
BHatZR m_bZR;
BHatPR m_bPR;
BHatRZ m_bRZ;
BHatZZ m_bZZ;
BHatPZ m_bPZ;
};
template<class real_type>
inline real_type ds_metric_norm( const std::array<real_type,6>& x0){
return sqrt( x0[0]*x0[0] +x0[1]*x0[1] );
}
}


struct DSPGenerator : public aGenerator2d
{

DSPGenerator( const dg::geo::TokamakMagneticField& mag, double R0, double R1, double Z0, double Z1, double deltaPhi):
m_R0(R0), m_R1(R1), m_Z0(Z0), m_Z1(Z1), m_deltaPhi(deltaPhi), m_dsmetric( mag)
{

}

virtual DSPGenerator* clone() const override final{return new DSPGenerator(*this);}

private:
virtual double do_width() const override final{return m_R1-m_R0;}
virtual double do_height() const override final{return m_Z1-m_Z0;}
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
unsigned Nx = zeta1d.size(), Ny = eta1d.size();
unsigned size = Nx*Ny;
for( unsigned k=0; k<Ny; k++)
for( unsigned i=0; i<Nx; i++)
{
x[k*Nx+i] = m_R0 + zeta1d[i];
y[k*Nx+i] = m_Z0 + eta1d[k];
}

for( unsigned i=0; i<size; i++)
{
std::array<double,6> coords{x[i],y[i],1, 0,0,1}, coordsP;
double phi1 = m_deltaPhi;
using Vec = std::array<double,6>;
dg::Adaptive<dg::ERKStep<Vec>> adapt( "Dormand-Prince-7-4-5", coords);
dg::AdaptiveTimeloop<Vec> loop( adapt, m_dsmetric, dg::pid_control,
detail::ds_metric_norm, 1e-8, 1e-10, 2);
loop.set_dt( m_deltaPhi/2.);
loop.integrate( 0, coords, phi1, coordsP);

x[i] = coordsP[0];
y[i] = coordsP[1];
zetaX[i] = coordsP[2];
zetaY[i] = coordsP[3];
etaX[i] = coordsP[4];
etaY[i] = coordsP[5];
}
}
double m_R0, m_R1, m_Z0, m_Z1, m_deltaPhi;
dg::geo::detail::DSMetricCylindrical m_dsmetric;
};
}
}
