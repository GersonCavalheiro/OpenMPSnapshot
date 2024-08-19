
#include "PusherPhoton.h"

#include <iostream>
#include <cmath>

#include "Species.h"
#include "Particles.h"

PusherPhoton::PusherPhoton( Params &params, Species *species )
: Pusher( params, species )
{
}

PusherPhoton::~PusherPhoton()
{
}



void PusherPhoton::operator()( Particles &particles, SmileiMPI *smpi,
int istart, int iend, int ithread, int ipart_ref )
{
double * __restrict__ invgf = &( smpi->dynamics_invgf[ithread][0] );

double *const __restrict__ position_x = particles.getPtrPosition( 0 );
double *const __restrict__ position_y = nDim_ > 1 ? particles.getPtrPosition( 1 ) : nullptr;
double *const __restrict__ position_z = nDim_ > 2 ? particles.getPtrPosition( 2 ) : nullptr;

const double *const __restrict__ momentum_x = particles.getPtrMomentum(0);
const double *const __restrict__ momentum_y = particles.getPtrMomentum(1);
const double *const __restrict__ momentum_z = particles.getPtrMomentum(2);

#pragma omp simd
for( int ipart=istart ; ipart<iend; ipart++ ) {

invgf[ipart - ipart_ref] = 1. / std::sqrt( momentum_x[ipart]*momentum_x[ipart] +
momentum_y[ipart]*momentum_y[ipart] +
momentum_z[ipart]*momentum_z[ipart] );

position_x[ipart] += dt*momentum_x[ipart]*invgf[ipart];
if (nDim_>1) {
position_y[ipart] += dt*momentum_y[ipart]*invgf[ipart];
if (nDim_>2) {
position_z[ipart] += dt*momentum_z[ipart]*invgf[ipart];
}
}

}



}
