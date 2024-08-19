#include "SpeciesVAdaptive.h"

#include <cmath>
#include <ctime>
#include <cstdlib>

#include <iostream>

#include <omp.h>

#include <cstring>
#include "PusherFactory.h"
#include "IonizationFactory.h"
#include "PartBoundCond.h"
#include "PartWall.h"
#include "BoundaryConditionType.h"

#include "ElectroMagn.h"
#include "Interpolator.h"
#include "InterpolatorFactory.h"
#include "Profile.h"

#include "Projector.h"
#include "ProjectorFactory.h"

#include "SimWindow.h"
#include "Patch.h"

#include "Field1D.h"
#include "Field2D.h"
#include "Field3D.h"
#include "Tools.h"

#include "DiagnosticTrack.h"

using namespace std;


SpeciesVAdaptive::SpeciesVAdaptive( Params &params, Patch *patch ) :
SpeciesV( params, patch )
{
initCluster( params );
npack_ = 0 ;
packsize_ = 0;
}

SpeciesVAdaptive::~SpeciesVAdaptive()
{
}

void SpeciesVAdaptive::scalarDynamics( double time_dual, unsigned int ispec,
ElectroMagn *EMfields, Params &params, bool diag_flag,
PartWalls *partWalls,
Patch *patch, SmileiMPI *smpi,
RadiationTables &RadiationTables,
MultiphotonBreitWheelerTables &MultiphotonBreitWheelerTables,
vector<Diagnostic *> &localDiags )
{

int ithread;
#ifdef _OPENMP
ithread = omp_get_thread_num();
#else
ithread = 0;
#endif

#ifdef  __DETAILED_TIMERS
double timer;
#endif

unsigned int iPart;

int tid( 0 );
std::vector<double> nrj_lost_per_thd( 1, 0. );

if( time_dual>time_frozen_ || Ionize ) {

smpi->dynamics_resize( ithread, nDim_field, particles->last_index.back(), params.geometry=="AMcylindrical" );

vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );

for( unsigned int i=0; i<count.size(); i++ ) {
count[i] = 0;
}

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif


Interp->fieldsWrapper( EMfields, *particles, smpi, &( particles->first_index[0] ), &( particles->last_index[particles->last_index.size()-1] ), ithread );


#ifdef  __DETAILED_TIMERS
patch->patch_timers[0] += MPI_Wtime() - timer;
#endif

for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {

if( Ionize ) {
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Ionize )( particles, particles->first_index[scell], particles->last_index[scell], Epart, patch, Proj );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[4] += MPI_Wtime() - timer;
#endif
}

if( time_dual<=time_frozen_ ) continue; 

if( Radiate ) {
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Radiate )( *particles,
radiated_photons_,
smpi,
RadiationTables, nrj_radiated_,
particles->first_index[scell], particles->last_index[scell], ithread );

#ifdef  __DETAILED_TIMERS
patch->patch_timers[5] += MPI_Wtime() - timer;
#endif
}

if( Multiphoton_Breit_Wheeler_process ) {
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Multiphoton_Breit_Wheeler_process )( *particles,
smpi,
mBW_pair_particles_,
mBW_pair_species_,
MultiphotonBreitWheelerTables,
nrj_radiated_,
particles->first_index[scell], particles->last_index[scell], ithread );

Multiphoton_Breit_Wheeler_process->computeThreadPhotonChi( *particles,
smpi,
particles->first_index[scell],
particles->last_index[scell],
ithread );

Multiphoton_Breit_Wheeler_process->removeDecayedPhotons(
*particles, smpi, scell, particles->first_index.size(), &particles->first_index[0], &particles->last_index[0], ithread );

#ifdef  __DETAILED_TIMERS
patch->patch_timers[6] += MPI_Wtime() - timer;
#endif
}
}

if( time_dual>time_frozen_ ) { 

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Push )( *particles, smpi, 0, particles->last_index.back(), ithread, 0. );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[1] += MPI_Wtime() - timer;
timer = MPI_Wtime();
#endif

for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {
double energy_lost( 0. );
if( mass_>0 ) {
for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
(*partWalls )[iwall]->apply( this, particles->first_index[scell], particles->last_index[scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += mass_ * energy_lost;
}

partBoundCond->apply( this, particles->first_index[scell], particles->last_index[scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += mass_ * energy_lost;

} else if( mass_==0 ) {
for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
(*partWalls )[iwall]->apply( this, particles->first_index[scell], particles->last_index[scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += energy_lost;
}

partBoundCond->apply( this, particles->first_index[scell], particles->last_index[scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += energy_lost;

} 
} 

computeParticleCellKeys( params );

#ifdef  __DETAILED_TIMERS
patch->patch_timers[3] += MPI_Wtime() - timer;
#endif

if( ( !particles->is_test ) && ( mass_ > 0 ) ) {

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
Proj->currentsAndDensityWrapper(
EMfields, *particles, smpi, particles->first_index[0],
particles->last_index.back(),
ithread,
diag_flag,
params.is_spectral,
ispec
);

#ifdef  __DETAILED_TIMERS
patch->patch_timers[2] += MPI_Wtime() - timer;
#endif

}
} 

for( unsigned int ithd=0 ; ithd<nrj_lost_per_thd.size() ; ithd++ ) {
nrj_bc_lost += nrj_lost_per_thd[tid];
}

}  

if (time_dual <= time_frozen_ && diag_flag &&( !particles->is_test ) ) { 

if( params.geometry != "AMcylindrical" ) {
double *b_rho = EMfields->rho_s[ispec] ? &( *EMfields->rho_s[ispec] )( 0 ) : &( *EMfields->rho_ )( 0 ) ;
for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell ++ ) { 
for( iPart=particles->first_index[scell] ; ( int )iPart<particles->last_index[scell]; iPart++ ) {
Proj->basic( b_rho, ( *particles ), iPart, 0 );
} 
}

} else { 
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( EMfields );
int n_species = patch->vecSpecies.size();
for( unsigned int imode = 0; imode<params.nmodes; imode++ ) {
int ifield = imode*n_species+ispec;
complex<double> *b_rho = emAM->rho_AM_s[ifield] ? &( *emAM->rho_AM_s[ifield] )( 0 ) : &( *emAM->rho_AM_[imode] )( 0 ) ;
for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell ++ ) { 
for( int iPart=particles->first_index[scell] ; iPart<particles->last_index[scell]; iPart++ ) {
Proj->basicForComplex( b_rho, ( *particles ), iPart, 0, imode );
}
}
}
}
} 

}





void SpeciesVAdaptive::reconfiguration( Params &params, Patch *patch )
{

bool reasign_operators = false;
float vecto_time = 0.;
float scalar_time = 0.;




(*part_comp_time_)( count,
vecto_time,
scalar_time );

if( ( vecto_time <= scalar_time && this->vectorized_operators == false )
|| ( vecto_time > scalar_time && this->vectorized_operators == true ) ) {
reasign_operators = true;
}

if( reasign_operators ) {

this->vectorized_operators = !this->vectorized_operators;



this->reconfigure_operators( params, patch );

}



}

void SpeciesVAdaptive::defaultConfigure( Params &params, Patch *patch )
{

this->vectorized_operators = ( params.adaptive_default_mode == "on" );

this->reconfigure_operators( params, patch );

}

void SpeciesVAdaptive::configuration( Params &params, Patch *patch )
{
float vecto_time = 0.;
float scalar_time = 0.;

if( particles->size() > 0 ) {

(*part_comp_time_)( count,
vecto_time,
scalar_time );

if( vecto_time <= scalar_time ) {
this->vectorized_operators = true;
} else if( vecto_time > scalar_time ) {
this->vectorized_operators = false;
}
}

else {
this->vectorized_operators = ( params.adaptive_default_mode == "on" );
}


this->reconfigure_operators( params, patch );
}

void SpeciesVAdaptive::reconfigure_operators( Params &params, Patch *patch )
{
delete Interp;
delete Proj;

Interp = InterpolatorFactory::create( params, patch, this->vectorized_operators );
Proj = ProjectorFactory::create( params, patch, this->vectorized_operators );
}


void SpeciesVAdaptive::scalarPonderomotiveUpdateSusceptibilityAndMomentum( double time_dual, unsigned int ispec,
ElectroMagn *EMfields,
Params &params, bool diag_flag,
Patch *patch, SmileiMPI *smpi,
vector<Diagnostic *> &localDiags )
{

int ithread;
#ifdef _OPENMP
ithread = omp_get_thread_num();
#else
ithread = 0;
#endif

#ifdef  __DETAILED_TIMERS
double timer;
#endif

if( time_dual>time_frozen_ ) { 

smpi->dynamics_resize( ithread, nDim_field, particles->last_index.back(), params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
Interp->fieldsAndEnvelope( EMfields, *particles, smpi, &( particles->first_index[0] ), &( particles->last_index[particles->last_index.size()-1] ), ithread );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[7] += MPI_Wtime() - timer;
#endif

if (Ionize){
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );
vector<double> *EnvEabs_part  = &( smpi->dynamics_EnvEabs_part[ithread] );
vector<double> *EnvExabs_part = &( smpi->dynamics_EnvExabs_part[ithread] );
vector<double> *Phipart = &( smpi->dynamics_PHIpart[ithread] );
Interp->envelopeFieldForIonization( EMfields, *particles, smpi, &( particles->first_index[0] ), &( particles->last_index[particles->last_index.size()-1] ), ithread );
Ionize->envelopeIonization( particles, ( particles->first_index[0] ), ( particles->last_index[particles->last_index.size()-1] ), Epart, EnvEabs_part, EnvExabs_part, Phipart, patch, Proj );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[4] += MPI_Wtime() - timer;
#endif
}
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
Proj->susceptibility( EMfields, *particles, mass_, smpi, particles->first_index[0], particles->last_index.back(), ithread );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[8] += MPI_Wtime() - timer;
#endif


#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Push )( *particles, smpi, 0, particles->last_index.back(), ithread );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[9] += MPI_Wtime() - timer;
#endif

} else { 
} 
} 


void SpeciesVAdaptive::scalarPonderomotiveUpdatePositionAndCurrents( double time_dual, unsigned int ispec,
ElectroMagn *EMfields,
Params &params, bool diag_flag, PartWalls *partWalls,
Patch *patch, SmileiMPI *smpi,
vector<Diagnostic *> &localDiags )
{

int ithread;
#ifdef _OPENMP
ithread = omp_get_thread_num();
#else
ithread = 0;
#endif

#ifdef  __DETAILED_TIMERS
double timer;
#endif

unsigned int iPart;

int tid( 0 );
std::vector<double> nrj_lost_per_thd( 1, 0. );

if( time_dual>time_frozen_ ) { 

smpi->dynamics_resize( ithread, nDim_field, particles->last_index.back(), params.geometry=="AMcylindrical" );

for( unsigned int i=0; i<count.size(); i++ ) {
count[i] = 0;
}



#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
Interp->timeCenteredEnvelope( EMfields, *particles, smpi, &( particles->first_index[0] ), &( particles->last_index[particles->last_index.size()-1] ), ithread );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[10] += MPI_Wtime() - timer;
#endif

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Push_ponderomotive_position )( *particles, smpi, particles->first_index[0], particles->last_index.back(), ithread );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[11] += MPI_Wtime() - timer;
#endif

for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {
double energy_lost( 0. );
if( mass_>0 ) {
for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
(*partWalls)[iwall]->apply( this, particles->first_index[scell], particles->last_index[scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += mass_ * energy_lost;
}

partBoundCond->apply( this, particles->first_index[scell], particles->last_index[scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += mass_ * energy_lost;

for( iPart=particles->first_index[scell] ; ( int )iPart<particles->last_index[scell]; iPart++ ) {
if ( particles->cell_keys[iPart] != -1 ) {
for( unsigned int i = 0 ; i<nDim_particle; i++ ) {
particles->cell_keys[iPart] *= this->length_[i];
particles->cell_keys[iPart] += round( ( particles->position( i, iPart )-min_loc_vec[i] ) * dx_inv_[i] );
}
count[particles->cell_keys[iPart]] ++;
}

}

} else if( mass_==0 ) {
ERROR_NAMELIST( "Particles with zero mass cannot interact with envelope",
LINK_NAMELIST + std::string("#laser-envelope-model"));
} 
}

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
if( ( !particles->is_test ) && ( mass_ > 0 ) ) {
Proj->currentsAndDensityWrapper( EMfields, *particles, smpi, particles->first_index[0], particles->last_index.back(), ithread, diag_flag, params.is_spectral, ispec );
}
#ifdef  __DETAILED_TIMERS
patch->patch_timers[12] += MPI_Wtime() - timer;
#endif

for( unsigned int ithd=0 ; ithd<nrj_lost_per_thd.size() ; ithd++ ) {
nrj_bc_lost += nrj_lost_per_thd[tid];
}

} 
else { 

if( Ionize ) {
smpi->dynamics_resize( ithread, nDim_particle, particles->last_index.back() );

vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif

Interp->fieldsWrapper( EMfields, *particles, smpi, &( particles->first_index[0] ), &( particles->last_index[particles->last_index.size()-1] ), ithread, particles->first_index[0] );

#ifdef  __DETAILED_TIMERS
patch->patch_timers[0] += MPI_Wtime() - timer;
#endif

for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Ionize )( particles, particles->first_index[scell], particles->last_index[scell], Epart, patch, Proj );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[4] += MPI_Wtime() - timer;
#endif
}
}

if( diag_flag &&( !particles->is_test ) ) {
double *b_rho=nullptr;
for( unsigned int ibin = 0 ; ibin < particles->first_index.size() ; ibin ++ ) { 
if( nDim_field==2 ) {
b_rho = EMfields->rho_s[ispec] ? &( *EMfields->rho_s[ispec] )( 0 ) : &( *EMfields->rho_ )( 0 ) ;
}
if( nDim_field==3 ) {
b_rho = EMfields->rho_s[ispec] ? &( *EMfields->rho_s[ispec] )( 0 ) : &( *EMfields->rho_ )( 0 ) ;
} else if( nDim_field==1 ) {
b_rho = EMfields->rho_s[ispec] ? &( *EMfields->rho_s[ispec] )( 0 ) : &( *EMfields->rho_ )( 0 ) ;
}
for( iPart=particles->first_index[ibin] ; ( int )iPart<particles->last_index[ibin]; iPart++ ) {
Proj->basic( b_rho, ( *particles ), iPart, 0 );
} 
}
} 

}
} 
