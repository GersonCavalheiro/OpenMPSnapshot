
#ifndef MULTIPHOTONBREITWHEELER_H
#define MULTIPHOTONBREITWHEELER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "MultiphotonBreitWheelerTables.h"
#include "Params.h"
#include "Random.h"

class MultiphotonBreitWheeler
{
public:

MultiphotonBreitWheeler( Params &params, Species *species, Random * rand );
~MultiphotonBreitWheeler();

void operator()( Particles &particles,
SmileiMPI* smpi,
Particles** new_pair,
Species ** new_pair_species,
MultiphotonBreitWheelerTables &mBW_tables,
double & pair_energy,
int istart,
int iend,
int ithread, int ipart_ref = 0 );

inline double __attribute__((always_inline)) computePhotonChi(
double kx, double ky, double kz,
double gamma,
double Ex, double Ey, double Ez,
double Bx, double By, double Bz )
{

return inv_norm_E_Schwinger_
* std::sqrt( std::fabs( std::pow( Ex*kx + Ey*ky + Ez*kz, 2 )
- std::pow( gamma*Ex - By*kz + Bz*ky, 2 )
- std::pow( gamma*Ey - Bz*kx + Bx*kz, 2 )
- std::pow( gamma*Ez - Bx*ky + By*kx, 2 ) ) );
};

void computeThreadPhotonChi( Particles &particles,
SmileiMPI *smpi,
int istart,
int iend,
int ithread, int ipart_ref = 0 );

void removeDecayedPhotons(
Particles &particles,
SmileiMPI *smpi,
int ibin, int nbin,
int *bmin, int *bmax, int ithread );



private:


int n_dimensions_;

double dt_;

int mBW_pair_creation_sampling_[2];

double mBW_pair_creation_inv_sampling_[2];

double chiph_threshold_;

Random * rand_;


double norm_E_Schwinger_;

double inv_norm_E_Schwinger_;

static constexpr double epsilon_tau_ = 1e-100;

};

#endif
