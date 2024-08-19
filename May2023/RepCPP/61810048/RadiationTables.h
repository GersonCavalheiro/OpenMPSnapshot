
#ifndef RADIATIONTABLES_H
#define RADIATIONTABLES_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include <cmath>
#include "userFunctions.h"
#include "Params.h"
#include "H5.h"
#include "Random.h"
#include "Table.h"
#include "Table2D.h"
#include "RadiationTools.h"

class RadiationTables
{


public:

RadiationTables();

~RadiationTables();

void initialization( Params &params , SmileiMPI *smpi );


double computePhotonProductionYield( const double particle_chi, 
const double particle_gamma);


double computeRandomPhotonChiWithInterpolation( double particle_chi, 
double xi);

double computeHNiel( double particle_chi, int nb_iterations, double eps );

double getHNielFromTable( double particle_chi );


inline double __attribute__((always_inline)) getRidgersCorrectedRadiatedEnergy( const double particle_chi,
const double dt )
{
return computeRidgersFit( particle_chi )*dt*particle_chi*particle_chi*factor_classical_radiated_power_;
};

static inline double __attribute__((always_inline)) computeRidgersFit( double particle_chi )
{
return std::pow( 1.0 + 4.8*( 1.0+particle_chi )*std::log( 1.0 + 1.7*particle_chi )
+ 2.44*particle_chi*particle_chi, -2.0/3.0 );
};

inline double __attribute__((always_inline)) getClassicalRadiatedEnergy( double particle_chi, double dt )
{
return dt*particle_chi*particle_chi*factor_classical_radiated_power_;
};

inline double __attribute__((always_inline)) getMinimumChiDiscontinuous()
{
return minimum_chi_discontinuous_;
}

inline double __attribute__((always_inline)) getMinimumChiContinuous()
{
return minimum_chi_continuous_;
}

inline std::string __attribute__((always_inline)) getNielHComputationMethod()
{
return this->niel_computation_method_;
}

inline int __attribute__((always_inline)) getNielHComputationMethodIndex()
{
return this->niel_computation_method_index_;
}

inline double __attribute__((always_inline)) getFactorClassicalRadiatedPower()
{
return factor_classical_radiated_power_;
}


void readHTable( SmileiMPI *smpi );

void readIntegfochiTable( SmileiMPI *smpi );

void readXiTable( SmileiMPI *smpi );

void readTables( Params &params, SmileiMPI *smpi );


void bcastTableXi( SmileiMPI *smpi );


Table niel_;


Table integfochi_;


Table2D xi_;

private:


std::string output_format_;

std::string table_path_;

bool compute_table_;

double minimum_chi_discontinuous_;

double minimum_chi_continuous_;

std::string niel_computation_method_;

int niel_computation_method_index_;


double factor_dNph_dt_;

double factor_classical_radiated_power_;

double normalized_Compton_wavelength_;

};

#endif
