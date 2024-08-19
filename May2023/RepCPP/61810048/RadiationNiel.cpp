
#include "RadiationNiel.h"

RadiationNiel::RadiationNiel( Params &params, Species *species, Random * rand  )
: Radiation( params, species, rand )
{
}

RadiationNiel::~RadiationNiel()
{
}

void RadiationNiel::operator()(
Particles       &particles,
Particles       *photons,
SmileiMPI       *smpi,
RadiationTables &RadiationTables,
double          &radiated_energy,
int istart,
int iend,
int ithread,
int ipart_ref )
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

double *const __restrict__ gamma = &( smpi->dynamics_invgf[ithread][0] );

const double one_over_mass_square = one_over_mass_*one_over_mass_;

const double sqrtdt = std::sqrt( dt_ );

const int nbparticles = iend-istart;

double temp;

int ipart;

double rad_energy;

double * diffusion = new double [nbparticles];

double * random_numbers = new double [nbparticles];

double*const __restrict__ momentum_x = particles.getPtrMomentum(0);
double*const __restrict__ momentum_y = particles.getPtrMomentum(1);
double*const __restrict__ momentum_z = particles.getPtrMomentum(2);

const short*const __restrict__ charge = particles.getPtrCharge();

const double*const __restrict__ weight = particles.getPtrWeight();

double*const __restrict__ particle_chi = particles.getPtrChi();

const double minimum_chi_continuous           = RadiationTables.getMinimumChiContinuous();
const double factor_classical_radiated_power  = RadiationTables.getFactorClassicalRadiatedPower();
const int niel_computation_method             = RadiationTables.getNielHComputationMethodIndex();



#pragma omp simd
for( ipart=istart ; ipart< iend; ipart++ ) {

const double charge_over_mass_square = ( double )( charge[ipart] )*one_over_mass_square;

gamma[ipart-ipart_ref] = std::sqrt( 1.0 + momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );

particle_chi[ipart] = Radiation::computeParticleChi( charge_over_mass_square,
momentum_x[ipart], momentum_y[ipart], momentum_z[ipart],
gamma[ipart],
Ex[ipart-ipart_ref], Ey[ipart-ipart_ref], Ez[ipart-ipart_ref],
Bx[ipart-ipart_ref], By[ipart-ipart_ref], Bz[ipart-ipart_ref] );
}




for( ipart=0 ; ipart < nbparticles; ipart++ ) {

if( particle_chi[ipart+istart] > minimum_chi_continuous ) {

random_numbers[ipart] = rand_->uniform2();
}
}

double p;
#pragma omp simd private(p,temp)
for( ipart=0 ; ipart < nbparticles; ipart++ ) {
if( particle_chi[ipart+istart] > minimum_chi_continuous ) {
temp = -std::log( ( 1.0-random_numbers[ipart] )*( 1.0+random_numbers[ipart] ) );

if( temp < 5.000000 ) {
temp = temp - 2.500000;
p = +2.81022636000e-08      ;
p = +3.43273939000e-07 + p*temp;
p = -3.52338770000e-06 + p*temp;
p = -4.39150654000e-06 + p*temp;
p = +0.00021858087e+00 + p*temp;
p = -0.00125372503e+00 + p*temp;
p = -0.00417768164e+00 + p*temp;
p = +0.24664072700e+00 + p*temp;
p = +1.50140941000e+00 + p*temp;
} else {
temp = std::sqrt( temp ) - 3.000000;
p = -0.000200214257      ;
p = +0.000100950558 + p*temp;
p = +0.001349343220 + p*temp;
p = -0.003673428440 + p*temp;
p = +0.005739507730 + p*temp;
p = -0.007622461300 + p*temp;
p = +0.009438870470 + p*temp;
p = +1.001674060000 + p*temp;
p = +2.832976820000 + p*temp;
}

random_numbers[ipart] *= p*sqrtdt*std::sqrt( 2. );

}
}


if( niel_computation_method == 0 ) {
for( ipart=0 ; ipart < nbparticles; ipart++ ) {

if( particle_chi[ipart+istart] > minimum_chi_continuous ) {

temp = RadiationTables.niel_.get( particle_chi[ipart+istart] );

diffusion[ipart] = std::sqrt( factor_classical_radiated_power*gamma[ipart+istart-ipart_ref]*temp )*random_numbers[ipart];
}
}
}
else if( niel_computation_method == 1 ) {
#pragma omp simd private(temp)
for( ipart=0 ; ipart < nbparticles; ipart++ ) {

int ipartp = ipart + istart;

if( particle_chi[ipartp] > minimum_chi_continuous ) {

temp = RadiationTools::getHNielFitOrder5( particle_chi[ipartp] );

diffusion[ipart] = std::sqrt( factor_classical_radiated_power*gamma[ipartp-ipart_ref]*temp )*random_numbers[ipart];
}
}
}
else if( niel_computation_method == 2 ) {
#pragma omp simd private(temp)
for( ipart=0 ; ipart < nbparticles; ipart++ ) {

int ipartp = ipart + istart;

if( particle_chi[ipartp] > minimum_chi_continuous ) {

temp = RadiationTools::getHNielFitOrder10( particle_chi[ipartp] );

diffusion[ipart] = std::sqrt( factor_classical_radiated_power*gamma[ipartp-ipart_ref]*temp )*random_numbers[ipart];
}
}
}
else if( niel_computation_method == 3) {

#pragma omp simd private(temp)
for( ipart=0 ; ipart < nbparticles; ipart++ ) {

int ipartp = ipart + istart;

if( particle_chi[ipartp] > minimum_chi_continuous ) {

temp = RadiationTools::getHNielFitRidgers( particle_chi[ipartp] );

diffusion[ipart] = std::sqrt( factor_classical_radiated_power*gamma[ipartp-ipart_ref]*temp )*random_numbers[ipart];
}
}
}

#pragma omp simd private(temp,rad_energy)
for( ipart=istart ; ipart<iend; ipart++ ) {
if( gamma[ipart-ipart_ref] > 1.1 && particle_chi[ipart] > minimum_chi_continuous ) {

rad_energy =
RadiationTables.getRidgersCorrectedRadiatedEnergy( particle_chi[ipart], dt_ );

temp = ( rad_energy - diffusion[ipart-istart] )
* gamma[ipart-ipart_ref]/( gamma[ipart-ipart_ref]*gamma[ipart-ipart_ref]-1. );

momentum_x[ipart] -= temp*momentum_x[ipart];
momentum_y[ipart] -= temp*momentum_y[ipart];
momentum_z[ipart] -= temp*momentum_z[ipart];

}
}



double radiated_energy_loc = 0;
double new_gamma = 0;

#pragma omp simd private(new_gamma) reduction(+:radiated_energy_loc)
for( int ipart=istart ; ipart<iend; ipart++ ) {

const double charge_over_mass_square = ( double )( charge[ipart] )*one_over_mass_square;

new_gamma = std::sqrt( 1.0
+ momentum_x[ipart]*momentum_x[ipart]
+ momentum_y[ipart]*momentum_y[ipart]
+ momentum_z[ipart]*momentum_z[ipart] );

radiated_energy_loc += weight[ipart]*( gamma[ipart] - new_gamma );

particle_chi[ipart] = Radiation::computeParticleChi( charge_over_mass_square,
momentum_x[ipart], momentum_y[ipart], momentum_z[ipart],
new_gamma,
Ex[ipart-ipart_ref], Ey[ipart-ipart_ref], Ez[ipart-ipart_ref],
Bx[ipart-ipart_ref], By[ipart-ipart_ref], Bz[ipart-ipart_ref] );

}
radiated_energy += radiated_energy_loc;

delete [] diffusion;
delete [] random_numbers;


}
