
#include "Radiation.h"

Radiation::Radiation( Params &params, Species *species, Random * rand )
{
n_dimensions_ = params.nDim_particle;

dt_   = params.timestep;

one_over_mass_ = 1./species->mass_;

norm_E_Schwinger_ = params.electron_mass*params.c_vacuum_*params.c_vacuum_
/ ( params.red_planck_cst* params.reference_angular_frequency_SI );

inv_norm_E_Schwinger_ = 1./norm_E_Schwinger_;

rand_ = rand;

nDim_          = params.nDim_particle;

}

Radiation::~Radiation()
{
}

void Radiation::computeParticlesChi( Particles &particles,
SmileiMPI *smpi,
int istart,
int iend,
int ithread, int ipart_ref )
{
std::vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );
std::vector<double> *Bpart = &( smpi->dynamics_Bpart[ithread] );

int nparts = Epart->size()/3;
double *Ex = &( ( *Epart )[0*nparts] );
double *Ey = &( ( *Epart )[1*nparts] );
double *Ez = &( ( *Epart )[2*nparts] );
double *Bx = &( ( *Bpart )[0*nparts] );
double *By = &( ( *Bpart )[1*nparts] );
double *Bz = &( ( *Bpart )[2*nparts] );

double charge_over_mass2;

double one_over_mass_square = pow( one_over_mass_, 2. );

double gamma;

double *momentum[3];
for( int i = 0 ; i<3 ; i++ ) {
momentum[i] =  &( particles.momentum( i, 0 ) );
}

short *charge = &( particles.charge( 0 ) );

double *chi = &( particles.chi( 0 ) );


#pragma omp simd
for( int ipart=istart ; ipart<iend; ipart++ ) {
charge_over_mass2 = ( double )( charge[ipart] )*one_over_mass_square;

gamma = sqrt( 1.0 + momentum[0][ipart]*momentum[0][ipart]
+ momentum[1][ipart]*momentum[1][ipart]
+ momentum[2][ipart]*momentum[2][ipart] );

chi[ipart] = Radiation::computeParticleChi( charge_over_mass2,
momentum[0][ipart], momentum[1][ipart], momentum[2][ipart],
gamma,
( *( Ex+ipart-ipart_ref ) ), ( *( Ey+ipart-ipart_ref ) ), ( *( Ez+ipart-ipart_ref ) ),
( *( Bx+ipart-ipart_ref ) ), ( *( By+ipart-ipart_ref ) ), ( *( Bz+ipart-ipart_ref ) ) );

}
}
