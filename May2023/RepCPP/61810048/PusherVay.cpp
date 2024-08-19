
#include "PusherVay.h"

#include <iostream>
#include <cmath>

#include "Species.h"

#include "Particles.h"

PusherVay::PusherVay( Params &params, Species *species )
: Pusher( params, species )
{
}

PusherVay::~PusherVay()
{
}



void PusherVay::operator()( Particles &particles, SmileiMPI *smpi, int istart, int iend, int ithread, int ipart_buffer_offset )
{
std::vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );
std::vector<double> *Bpart = &( smpi->dynamics_Bpart[ithread] );
double *const invgf = &( smpi->dynamics_invgf[ithread][0] );

double *const __restrict__ position_x = particles.getPtrPosition( 0 );
double *const __restrict__ position_y = nDim_ > 1 ? particles.getPtrPosition( 1 ) : nullptr;
double *const __restrict__ position_z = nDim_ > 2 ? particles.getPtrPosition( 2 ) : nullptr;

double *const __restrict__ momentum_x = particles.getPtrMomentum(0);
double *const __restrict__ momentum_y = particles.getPtrMomentum(1);
double *const __restrict__ momentum_z = particles.getPtrMomentum(2);

const short *const charge = particles.getPtrCharge();

const int nparts = vecto ? Epart->size() / 3 :
particles.size(); 

const double *const __restrict__ Ex = &( ( *Epart )[0*nparts] );
const double *const __restrict__ Ey = &( ( *Epart )[1*nparts] );
const double *const __restrict__ Ez = &( ( *Epart )[2*nparts] );
const double *const __restrict__ Bx = &( ( *Bpart )[0*nparts] );
const double *const __restrict__ By = &( ( *Bpart )[1*nparts] );
const double *const __restrict__ Bz = &( ( *Bpart )[2*nparts] );

#pragma omp simd
for( int ipart=istart ; ipart<iend; ipart++ ) {

const double charge_over_mass_dts2 = ( double )( charge[ipart] )*one_over_mass_*dts2;


invgf[ipart-ipart_buffer_offset] = 1./std::sqrt( 1.0 + momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );

double upx = momentum_x[ipart] + 2.*charge_over_mass_dts2*( Ex[ipart-ipart_buffer_offset] );
double upy = momentum_y[ipart] + 2.*charge_over_mass_dts2*( Ey[ipart-ipart_buffer_offset] );
double upz = momentum_z[ipart] + 2.*charge_over_mass_dts2*( Ez[ipart-ipart_buffer_offset] );

double Tx  = charge_over_mass_dts2* ( Bx[ipart-ipart_buffer_offset] );
double Ty  = charge_over_mass_dts2* ( By[ipart-ipart_buffer_offset] );
double Tz  = charge_over_mass_dts2* ( Bz[ipart-ipart_buffer_offset] );

upx += invgf[ipart-ipart_buffer_offset]*( momentum_y[ipart]*Tz - momentum_z[ipart]*Ty );
upy += invgf[ipart-ipart_buffer_offset]*( momentum_z[ipart]*Tx - momentum_x[ipart]*Tz );
upz += invgf[ipart-ipart_buffer_offset]*( momentum_x[ipart]*Ty - momentum_y[ipart]*Tx );

double alpha = 1.0 + upx*upx + upy*upy + upz*upz;
const double T2    = Tx*Tx + Ty*Ty + Tz*Tz;


double s     = alpha - T2;
double us2   = upx*Tx + upy*Ty + upz*Tz;
us2   = us2*us2;

alpha = 1.0/std::sqrt( 0.5*( s + std::sqrt( s*s + 4.0*( T2 + us2 ) ) ) );

Tx *= alpha;
Ty *= alpha;
Tz *= alpha;

s = 1.0/( 1.0+Tx*Tx+Ty*Ty+Tz*Tz );
alpha   = upx*Tx + upy*Ty + upz*Tz;

const double pxsm = s*( upx + alpha*Tx + Tz*upy - Ty*upz );
const double pysm = s*( upy + alpha*Ty + Tx*upz - Tz*upx );
const double pzsm = s*( upz + alpha*Tz + Ty*upx - Tx*upy );




invgf[ipart-ipart_buffer_offset] = 1.0 / std::sqrt( 1.0 + pxsm*pxsm + pysm*pysm + pzsm*pzsm );

momentum_x[ipart] = pxsm;
momentum_y[ipart] = pysm;
momentum_z[ipart] = pzsm;

position_x[ipart] += dt*momentum_x[ipart]*invgf[ipart-ipart_buffer_offset];
if (nDim_>1) {
position_y[ipart] += dt*momentum_y[ipart]*invgf[ipart-ipart_buffer_offset];
if (nDim_>2) {
position_z[ipart] += dt*momentum_z[ipart]*invgf[ipart-ipart_buffer_offset];
}
}
} 



}
