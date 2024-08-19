
#include "RadiationMonteCarlo.h"

#include <cstring>
#include <fstream>

#include <cmath>

RadiationMonteCarlo::RadiationMonteCarlo( Params &params, Species *species, Random * rand  )
: Radiation( params, species, rand )
{
radiation_photon_sampling_        = species->radiation_photon_sampling_;
max_photon_emissions_             = species->radiation_max_emissions_;
radiation_photon_gamma_threshold_ = species->radiation_photon_gamma_threshold_;
inv_radiation_photon_sampling_    = 1. / radiation_photon_sampling_;
}

RadiationMonteCarlo::~RadiationMonteCarlo()
{
}

void RadiationMonteCarlo::operator()(
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

const int nparts = Epart->size()/3;

const double *const __restrict__ Ex = &( ( *Epart )[0*nparts] );
const double *const __restrict__ Ey = &( ( *Epart )[1*nparts] );
const double *const __restrict__ Ez = &( ( *Epart )[2*nparts] );
const double *const __restrict__ Bx = &( ( *Bpart )[0*nparts] );
const double *const __restrict__ By = &( ( *Bpart )[1*nparts] );
const double *const __restrict__ Bz = &( ( *Bpart )[2*nparts] );

const double one_over_mass_square = one_over_mass_*one_over_mass_;

double cont_rad_energy;

double temp;


double *const __restrict__ position_x = particles.getPtrPosition( 0 );
double *const __restrict__ position_y = nDim_ > 1 ? particles.getPtrPosition( 1 ) : nullptr;
double *const __restrict__ position_z = nDim_ > 2 ? particles.getPtrPosition( 2 ) : nullptr;

double *const __restrict__ momentum_x = particles.getPtrMomentum(0);
double *const __restrict__ momentum_y = particles.getPtrMomentum(1);
double *const __restrict__ momentum_z = particles.getPtrMomentum(2);

const short *const __restrict__ charge = particles.getPtrCharge();

const double *const __restrict__ weight = &( particles.weight( 0 ) );

double *const __restrict__ tau = &( particles.tau( 0 ) );

double *const __restrict__ chi = &( particles.chi(0));


int nphotons;

if (photons) {
nphotons = photons->size();
photons->reserve( nphotons + radiation_photon_sampling_ * (iend - istart) * max_photon_emissions_ );
} else {
nphotons = 0;
}

double *const __restrict__ photon_position_x = photons ? photons->getPtrPosition( 0 ) : nullptr;
double *const __restrict__ photon_position_y = photons ? (nDim_ > 1 ? photons->getPtrPosition( 1 ) : nullptr) : nullptr;
double *const __restrict__ photon_position_z = photons ? (nDim_ > 2 ? photons->getPtrPosition( 2 ) : nullptr) : nullptr;

double *const __restrict__ photon_momentum_x = photons ? photons->getPtrMomentum(0) : nullptr;
double *const __restrict__ photon_momentum_y = photons ? photons->getPtrMomentum(1) : nullptr;
double *const __restrict__ photon_momentum_z = photons ? photons->getPtrMomentum(2) : nullptr;

short *const __restrict__ photon_charge = photons ? photons->getPtrCharge() : nullptr;

double *const __restrict__ photon_weight = photons ? photons->getPtrWeight() : nullptr;

double *const __restrict__ photon_chi_array = photons ? (photons->isQuantumParameter ? photons->getPtrChi() : nullptr) : nullptr;

double *const __restrict__ photon_tau = photons ? (photons->isMonteCarlo ? photons->getPtrTau() : nullptr) : nullptr;




for( int ipart=istart ; ipart<iend; ipart++ ) {

const double charge_over_mass_square = ( double )( charge[ipart] )*one_over_mass_square;

double emission_time = 0;

double local_it_time = 0;

int mc_it_nb = 0;

int i_photon_emission = 0;

while( ( local_it_time < dt_ )
&&( mc_it_nb < max_monte_carlo_iterations_ ) ) {

const double particle_gamma = std::sqrt( 1.0 + momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );
if( particle_gamma < 1.1 ){
break;
}

const double particle_chi = Radiation::computeParticleChi( charge_over_mass_square,
momentum_x[ipart], momentum_y[ipart], momentum_z[ipart],
particle_gamma,
Ex[ipart-ipart_ref], Ey[ipart-ipart_ref], Ez[ipart-ipart_ref],
Bx[ipart-ipart_ref], By[ipart-ipart_ref], Bz[ipart-ipart_ref] );


if( ( particle_chi > RadiationTables.getMinimumChiDiscontinuous() )
&& ( tau[ipart] <= epsilon_tau_ ) ) {
while( tau[ipart] <= epsilon_tau_ ) {
tau[ipart] = -std::log( 1.-rand_->uniform() );
}

}

if( tau[ipart] > epsilon_tau_ ) {

temp = RadiationTables.computePhotonProductionYield( 
particle_chi, 
particle_gamma);

emission_time = std::min( tau[ipart]/temp, dt_ - local_it_time );

tau[ipart] -= temp*emission_time;

if( tau[ipart] <= epsilon_tau_ ) {


double xi = rand_->uniform();

double photon_chi = RadiationTables.computeRandomPhotonChiWithInterpolation( particle_chi, xi);

double photon_gamma = photon_chi/particle_chi*( particle_gamma-1.0 );


double inv_old_norm_p = photon_gamma/std::sqrt( particle_gamma*particle_gamma - 1.0 );
momentum_x[ipart] -= momentum_x[ipart]*inv_old_norm_p;
momentum_y[ipart] -= momentum_y[ipart]*inv_old_norm_p;
momentum_z[ipart] -= momentum_z[ipart]*inv_old_norm_p;



if(          photons
&& ( photon_gamma >= radiation_photon_gamma_threshold_ ) 
&& ( i_photon_emission < max_photon_emissions_)) {

photons->createParticles( radiation_photon_sampling_ );

nphotons += radiation_photon_sampling_;

inv_old_norm_p = 1./std::sqrt( momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );

for( auto iphoton=nphotons-radiation_photon_sampling_; iphoton<nphotons; iphoton++ ) {


photon_position_x[iphoton]=position_x[ipart];
if (nDim_>1) {
photon_position_y[iphoton]=position_y[ipart];
if (nDim_>2) {
photon_position_z[iphoton]=position_z[ipart];
}
}

photon_momentum_x[iphoton] =
photon_gamma*momentum_x[ipart]*inv_old_norm_p;
photon_momentum_y[iphoton] =
photon_gamma*momentum_y[ipart]*inv_old_norm_p;
photon_momentum_z[iphoton] =
photon_gamma*momentum_z[ipart]*inv_old_norm_p;


photon_weight[iphoton] = weight[ipart]*inv_radiation_photon_sampling_;
photon_charge[iphoton] = 0;

if( photons->isQuantumParameter ) {
photon_chi_array[iphoton] = photon_chi;
}

if( photons->isMonteCarlo ) {
photon_tau[iphoton] = -1.;
}

} 

i_photon_emission += 1;

}
else {
photon_gamma = particle_gamma - std::sqrt( 1.0 + momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );
radiated_energy += weight[ipart]*photon_gamma;
}

tau[ipart] = -1.;
}

mc_it_nb ++;
local_it_time += emission_time;

}

else if( particle_chi <=  RadiationTables.getMinimumChiDiscontinuous()
&& tau[ipart] <= epsilon_tau_
&& particle_chi >  RadiationTables.getMinimumChiContinuous() ) {

emission_time = dt_ - local_it_time;

cont_rad_energy =
RadiationTables.getRidgersCorrectedRadiatedEnergy( particle_chi,
emission_time );

temp = cont_rad_energy*particle_gamma/( particle_gamma*particle_gamma-1. );
momentum_x[ipart] -= temp*momentum_x[ipart];
momentum_y[ipart] -= temp*momentum_y[ipart];
momentum_z[ipart] -= temp*momentum_z[ipart];

radiated_energy += weight[ipart]*( particle_gamma - std::sqrt( 1.0
+ momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] ) );

local_it_time = dt_;
}
else { 
local_it_time = dt_;
}

}

}


#pragma omp simd
for( int ipart=istart ; ipart<iend; ipart++ ) {
const double charge_over_mass_square = ( double )( charge[ipart] )*one_over_mass_square;

const double particle_gamma = std::sqrt( 1.0 + momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );

chi[ipart] = Radiation::computeParticleChi( charge_over_mass_square,
momentum_x[ipart], momentum_y[ipart], momentum_z[ipart],
particle_gamma,
Ex[ipart-ipart_ref], Ey[ipart-ipart_ref], Ez[ipart-ipart_ref],
Bx[ipart-ipart_ref], By[ipart-ipart_ref], Bz[ipart-ipart_ref] );

}
}
