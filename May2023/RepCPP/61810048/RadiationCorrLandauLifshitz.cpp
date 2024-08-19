
#include "RadiationCorrLandauLifshitz.h"

#include <cstring>
#include <fstream>

#include <cmath>

RadiationCorrLandauLifshitz::RadiationCorrLandauLifshitz( Params &params, Species *species, Random * rand )
: Radiation( params, species, rand )
{
}

RadiationCorrLandauLifshitz::~RadiationCorrLandauLifshitz()
{
}

void RadiationCorrLandauLifshitz::operator()(
Particles       &particles,
Particles       *photons,
SmileiMPI       *smpi,
RadiationTables &RadiationTables,
double          &radiated_energy,
int             istart,
int             iend,
int             ithread,
int             ipart_ref)
{

std::vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );
std::vector<double> *Bpart = &( smpi->dynamics_Bpart[ithread] );

int nparts = Epart->size()/3;
const double *const __restrict__ Ex = &( ( *Epart )[0*nparts] );
const double *const __restrict__ Ey = &( ( *Epart )[1*nparts] );
const double *const __restrict__ Ez = &( ( *Epart )[2*nparts] );
const double *const __restrict__ Bx = &( ( *Bpart )[0*nparts] );
const double *const __restrict__ By = &( ( *Bpart )[1*nparts] );
const double *const __restrict__ Bz = &( ( *Bpart )[2*nparts] );

const double one_over_mass_square = one_over_mass_*one_over_mass_;

const double minimum_chi_continuous = RadiationTables.getMinimumChiContinuous();

double *const __restrict__ momentum_x = particles.getPtrMomentum(0);
double *const __restrict__ momentum_y = particles.getPtrMomentum(1);
double *const __restrict__ momentum_z = particles.getPtrMomentum(2);

const short *const __restrict__ charge = particles.getPtrCharge();

const double *const __restrict__ weight = particles.getPtrWeight();

double *const __restrict__ chi = particles.getPtrChi();

double * rad_norm_energy = new double [iend-istart];
#pragma omp simd
for( int ipart=0 ; ipart<iend-istart; ipart++ ) {
rad_norm_energy[ipart] = 0;
}

double radiated_energy_loc = 0;


#pragma omp simd
for( int ipart=istart ; ipart<iend; ipart++ ) {

const double charge_over_mass_square = ( double )( charge[ipart] )*one_over_mass_square;

const double gamma = sqrt( 1.0 + momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );

const double particle_chi = Radiation::computeParticleChi( charge_over_mass_square,
momentum_x[ipart], momentum_y[ipart], momentum_z[ipart],
gamma,
Ex[ipart-ipart_ref], Ey[ipart-ipart_ref], Ez[ipart-ipart_ref] ,
Bx[ipart-ipart_ref], By[ipart-ipart_ref], Bz[ipart-ipart_ref] );

if( gamma > 1.1 && particle_chi >= minimum_chi_continuous ) {

const double temp =
RadiationTables.getRidgersCorrectedRadiatedEnergy( particle_chi, dt_ ) * gamma/( gamma*gamma - 1 );

momentum_x[ipart] -= temp*momentum_x[ipart];
momentum_y[ipart] -= temp*momentum_y[ipart];
momentum_z[ipart] -= temp*momentum_z[ipart];

rad_norm_energy[ipart - istart] = gamma - sqrt( 1.0
+ momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );

}
}


#pragma omp simd reduction(+:radiated_energy_loc)
for( int ipart=0 ; ipart<iend-istart; ipart++ ) {
radiated_energy_loc += weight[ipart]*rad_norm_energy[ipart] ;
}


#pragma omp simd
for( int ipart=istart ; ipart<iend; ipart++ ) {
const double charge_over_mass_square = ( double )( charge[ipart] )*one_over_mass_square;

const double gamma = sqrt( 1.0 + momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );

chi[ipart] = Radiation::computeParticleChi( charge_over_mass_square,
momentum_x[ipart], momentum_y[ipart], momentum_z[ipart],
gamma,
Ex[ipart-ipart_ref], Ey[ipart-ipart_ref], Ez[ipart-ipart_ref],
Bx[ipart-ipart_ref], By[ipart-ipart_ref], Bz[ipart-ipart_ref] );

}

radiated_energy += radiated_energy_loc;


delete [] rad_norm_energy;

}
