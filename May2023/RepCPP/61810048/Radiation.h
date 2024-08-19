
#ifndef RADIATION_H
#define RADIATION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "Params.h"
#include "Particles.h"
#include "Species.h"
#include "RadiationTables.h"
#include "Random.h"

class Radiation
{

public:
Radiation( Params &params, Species *species, Random * rand );
virtual ~Radiation();

virtual void operator()(
Particles       &particles,
Particles       *photons,
SmileiMPI       *smpi,
RadiationTables &RadiationTables,
double          &radiated_energy,
int             istart,
int             iend,
int             ithread,
int             ipart_ref = 0) = 0;

inline double __attribute__((always_inline)) computeParticleChi( double charge_over_mass2,
double px, double py, double pz,
double gamma,
double Ex, double Ey, double Ez,
double Bx, double By, double Bz )
{

return std::fabs( charge_over_mass2 )*inv_norm_E_Schwinger_
* std::sqrt( std::fabs( std::pow( Ex*px + Ey*py + Ez*pz, 2 )
- std::pow( gamma*Ex - By*pz + Bz*py, 2 )
- std::pow( gamma*Ey - Bz*px + Bx*pz, 2 )
- std::pow( gamma*Ez - Bx*py + By*px, 2 ) ) );
};

void computeParticlesChi( Particles &particles,
SmileiMPI *smpi,
int istart,
int iend,
int ithread,
int ipart_ref = 0 );


protected:


int n_dimensions_;

double one_over_mass_;

double dt_;

Random * rand_;


double norm_E_Schwinger_;

double inv_norm_E_Schwinger_;

int nDim_;

private:

};

#endif
