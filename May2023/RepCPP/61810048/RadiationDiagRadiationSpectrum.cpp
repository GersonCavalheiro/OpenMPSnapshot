
#include "RadiationDiagRadiationSpectrum.h"

RadiationDiagRadiationSpectrum::RadiationDiagRadiationSpectrum(Params& params,
Species * species, Random * rand )
: Radiation(params, species, rand)
{
}

RadiationDiagRadiationSpectrum::~RadiationDiagRadiationSpectrum()
{
}

void RadiationDiagRadiationSpectrum::operator() (
Particles &particles,
Particles *photons,
SmileiMPI *smpi,
RadiationTables &RadiationTables,
double          &radiated_energy,
int istart,
int iend,
int ithread, int ipart_ref)

{

std::vector<double> *Epart = &(smpi->dynamics_Epart[ithread]);
std::vector<double> *Bpart = &(smpi->dynamics_Bpart[ithread]);

int nparts = particles.size();
double* Ex = &( (*Epart)[0*nparts] );
double* Ey = &( (*Epart)[1*nparts] );
double* Ez = &( (*Epart)[2*nparts] );
double* Bx = &( (*Bpart)[0*nparts] );
double* By = &( (*Bpart)[1*nparts] );
double* Bz = &( (*Bpart)[2*nparts] );

double charge_over_mass2;

const double one_over_mass_2 = std::pow(one_over_mass_,2.);

double gamma;

double* momentum[3];
for ( int i = 0 ; i<3 ; i++ )
momentum[i] =  &( particles.momentum(i,0) );

short* charge = &( particles.charge(0) );

double* chi = &( particles.chi(0));


#pragma omp simd
for (int ipart=istart ; ipart<iend; ipart++ ) {
charge_over_mass2 = (double)(charge[ipart])*one_over_mass_2;

gamma = sqrt(1.0 + momentum[0][ipart]*momentum[0][ipart]
+ momentum[1][ipart]*momentum[1][ipart]
+ momentum[2][ipart]*momentum[2][ipart]);

chi[ipart] = Radiation::computeParticleChi(charge_over_mass2,
momentum[0][ipart],momentum[1][ipart],momentum[2][ipart],
gamma,
(*(Ex+ipart)),(*(Ey+ipart)),(*(Ez+ipart)),
(*(Bx+ipart)),(*(By+ipart)),(*(Bz+ipart)) );

}
}
