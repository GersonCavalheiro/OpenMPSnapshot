#include "VectorPatch.h"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <math.h>

#include "BinaryProcesses.h"
#include "DomainDecompositionFactory.h"
#include "PatchesFactory.h"
#include "Species.h"
#include "Particles.h"
#include "PeekAtSpecies.h"
#include "SimWindow.h"
#include "SolverFactory.h"
#include "DiagnosticFactory.h"
#include "LaserEnvelope.h"
#include "ElectroMagnBC.h"
#include "Laser.h"

#include "ElectroMagnBC2D_PML.h"
#include "ElectroMagnBC3D_PML.h"
#include "ElectroMagnBCAM_PML.h"

#include "SyncVectorPatch.h"
#include "interface.h"
#include "Timers.h"

using namespace std;


VectorPatch::VectorPatch()
{
domain_decomposition_ = NULL ;
}


VectorPatch::VectorPatch( Params &params )
{
domain_decomposition_ = DomainDecompositionFactory::create( params );
}


VectorPatch::~VectorPatch()
{
if( domain_decomposition_ != NULL ) {
delete domain_decomposition_;
}
}


void VectorPatch::close( SmileiMPI *smpiData )
{
for( unsigned int icoll = 0; icoll < patches_[0]->vecBPs.size(); icoll++ ) {
if( patches_[0]->vecBPs[icoll]->debug_file_ ) {
delete patches_[0]->vecBPs[icoll]->debug_file_;
}
}

closeAllDiags( smpiData );

if( diag_timers_.size() ) {
MESSAGE( "\n\tDiagnostics profile :" );
}
for( unsigned int idiag = 0 ;  idiag < diag_timers_.size() ; idiag++ ) {
double sum( 0 );
MPI_Reduce( &diag_timers_[idiag]->time_acc_, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
MESSAGE( "\t\t" << setw( 20 ) << diag_timers_[idiag]->name_ << "\t" << sum/( double )smpiData->getSize() );
}

for( unsigned int idiag = 0 ;  idiag < diag_timers_.size() ; idiag++ ) {
delete diag_timers_[idiag];
}
diag_timers_.clear();

for( unsigned int idiag=0 ; idiag<localDiags.size(); idiag++ ) {
delete localDiags[idiag];
}
localDiags.clear();

for( unsigned int idiag=0 ; idiag<globalDiags.size(); idiag++ ) {
delete globalDiags[idiag];
}
globalDiags.clear();

for( unsigned int ipatch=0 ; ipatch<size(); ipatch++ ) {
delete patches_[ipatch];
}

patches_.clear();
}

void VectorPatch::createDiags( Params &params, SmileiMPI *smpi, OpenPMDparams &openPMD, RadiationTables * radiation_tables_ )
{
globalDiags = DiagnosticFactory::createGlobalDiagnostics( params, smpi, *this, radiation_tables_ );
localDiags  = DiagnosticFactory::createLocalDiagnostics( params, smpi, *this, openPMD );

vector<string> names( 0 );
for( unsigned int i=0; i<globalDiags.size(); i++ ) {
if( globalDiags[i]->name().empty() ) continue;
if( std::find(names.begin(), names.end(), globalDiags[i]->name()) != names.end() ) {
ERROR( "Two diagnostics have the same label " << globalDiags[i]->name() );
}
names.push_back( globalDiags[i]->name() );
}
for( unsigned int i=0; i<localDiags.size(); i++ ) {
if( localDiags[i]->name().empty() ) continue;
if( std::find(names.begin(), names.end(), localDiags[i]->name()) != names.end() ) {
ERROR( "Two diagnostics have the same label " << localDiags[i]->name() );
}
names.push_back( localDiags[i]->name() );
}

for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
if( params.geometry!="AMcylindrical" ) {
for( unsigned int ifield=0 ; ifield<( *this )( ipatch )->EMfields->Jx_s.size(); ifield++ ) {
if( ( *this )( ipatch )->EMfields->Jx_s[ifield]->data_ == NULL ) {
delete( *this )( ipatch )->EMfields->Jx_s[ifield];
( *this )( ipatch )->EMfields->Jx_s[ifield]=NULL;
}

}
for( unsigned int ifield=0 ; ifield<( *this )( ipatch )->EMfields->Jy_s.size(); ifield++ ) {
if( ( *this )( ipatch )->EMfields->Jy_s[ifield]->data_ == NULL ) {
delete( *this )( ipatch )->EMfields->Jy_s[ifield];
( *this )( ipatch )->EMfields->Jy_s[ifield]=NULL;
}
}
for( unsigned int ifield=0 ; ifield<( *this )( ipatch )->EMfields->Jz_s.size(); ifield++ ) {
if( ( *this )( ipatch )->EMfields->Jz_s[ifield]->data_ == NULL ) {
delete( *this )( ipatch )->EMfields->Jz_s[ifield];
( *this )( ipatch )->EMfields->Jz_s[ifield]=NULL;
}
}
for( unsigned int ifield=0 ; ifield<( *this )( ipatch )->EMfields->rho_s.size(); ifield++ ) {
if( ( *this )( ipatch )->EMfields->rho_s[ifield]->data_ == NULL ) {
delete( *this )( ipatch )->EMfields->rho_s[ifield];
( *this )( ipatch )->EMfields->rho_s[ifield]=NULL;
}
}

} else {

ElectroMagnAM *EMfields = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
for( unsigned int ifield=0 ; ifield<EMfields->Jl_s.size(); ifield++ ) {
if( EMfields->Jl_s[ifield]->cdata_ == NULL ) {
delete EMfields->Jl_s[ifield];
EMfields->Jl_s[ifield]=NULL;
}
}
for( unsigned int ifield=0 ; ifield<EMfields->Jr_s.size(); ifield++ ) {
if( EMfields->Jr_s[ifield]->cdata_ == NULL ) {
delete EMfields->Jr_s[ifield];
EMfields->Jr_s[ifield]=NULL;
}
}
for( unsigned int ifield=0 ; ifield<EMfields->Jt_s.size(); ifield++ ) {
if( EMfields->Jt_s[ifield]->cdata_ == NULL ) {
delete EMfields->Jt_s[ifield];
EMfields->Jt_s[ifield]=NULL;
}
}

for( unsigned int ifield=0 ; ifield<EMfields->rho_AM_s.size(); ifield++ ) {
if( EMfields->rho_AM_s[ifield]->cdata_ == NULL ) {
delete EMfields->rho_AM_s[ifield];
EMfields->rho_AM_s[ifield]=NULL;
}
}

if (params.Laser_Envelope_model){
for( unsigned int ifield=0 ; ifield<EMfields->Env_Chi_s.size(); ifield++ ) {
if( EMfields->Env_Chi_s[ifield]->data_ == NULL ) {
delete EMfields->Env_Chi_s[ifield];
EMfields->Env_Chi_s[ifield]=NULL;
}
}
}
}


if( (params.Laser_Envelope_model) && (params.geometry != "AMcylindrical")) {
for( unsigned int ifield=0 ; ifield<( *this )( ipatch )->EMfields->Env_Chi_s.size(); ifield++ ) {
if( ( *this )( ipatch )->EMfields->Env_Chi_s[ifield]->data_ == NULL ) {
delete( *this )( ipatch )->EMfields->Env_Chi_s[ifield];
( *this )( ipatch )->EMfields->Env_Chi_s[ifield]=NULL;
}
}
}



}

for( unsigned int idiag = 0 ;  idiag < globalDiags.size() ; idiag++ ) {
diag_timers_.push_back( new Timer( globalDiags[idiag]->filename ) );
}
for( unsigned int idiag = 0 ;  idiag < localDiags.size() ; idiag++ ) {
diag_timers_.push_back( new Timer( localDiags[idiag]->filename ) );
}

for( unsigned int idiag = 0 ;  idiag < diag_timers_.size() ; idiag++ ) {
diag_timers_[idiag]->init( smpi );
}

}



void VectorPatch::configuration( Params &params, Timers &timers, int itime )
{

timers.reconfiguration.restart();

unsigned int npatches = this->size();

#pragma omp master
{
for( unsigned int ipatch=0 ; ipatch< npatches; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
( *this )( ipatch )->cleanMPIBuffers( ispec, params );
}
}
}
#pragma omp barrier


#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<npatches ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
species( ipatch, ispec )->defaultConfigure( params, ( *this )( ipatch ) );
}
}

timers.reconfiguration.update( params.printNow( itime ) );

}

void VectorPatch::reconfiguration( Params &params, Timers &timers, int itime )
{

timers.reconfiguration.restart();

unsigned int npatches = this->size();

#pragma omp master
{
for( unsigned int ipatch=0 ; ipatch < npatches ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
( *this )( ipatch )->cleanMPIBuffers( ispec, params );
}
}
}
#pragma omp barrier

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch < npatches ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
species( ipatch, ispec )->reconfiguration( params, ( *this )( ipatch ) );
}
}

timers.reconfiguration.update( params.printNow( itime ) );
}


void VectorPatch::sortAllParticles( Params &params )
{
if( params.cell_sorting_ ) {
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<patches_[ipatch]->vecSpecies.size(); ispec++ ) {
patches_[ipatch]->vecSpecies[ispec]->computeParticleCellKeys( params );
patches_[ipatch]->vecSpecies[ispec]->sortParticles( params, patches_[ipatch] );
}
}
}
}

void VectorPatch::dynamics( Params &params,
SmileiMPI *smpi,
SimWindow *simWindow,
RadiationTables &RadiationTables,
MultiphotonBreitWheelerTables &MultiphotonBreitWheelerTables,
double time_dual, Timers &timers, int itime )
{

#pragma omp single
{
diag_flag = ( needsRhoJsNow( itime ) || params.is_spectral );
}

timers.particles.restart();
ostringstream t;
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->restartRhoJ();
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
Species *spec = species( ipatch, ispec );

if( params.keep_position_old ) {
spec->particles->savePositions();
}

if( params.Laser_Envelope_model ) {
continue;
}

if( spec->isProj( time_dual, simWindow ) || diag_flag ) {
if( spec->vectorized_operators ) {
spec->dynamics( time_dual, ispec,
emfields( ipatch ),
params, diag_flag, partwalls( ipatch ),
( *this )( ipatch ), smpi,
RadiationTables,
MultiphotonBreitWheelerTables,
localDiags );
}
else {
if( params.vectorization_mode == "adaptive" ) {
spec->scalarDynamics( time_dual, ispec,
emfields( ipatch ),
params, diag_flag, partwalls( ipatch ),
( *this )( ipatch ), smpi,
RadiationTables,
MultiphotonBreitWheelerTables,
localDiags );
} else {
spec->Species::dynamics( time_dual, ispec,
emfields( ipatch ),
params, diag_flag, partwalls( ipatch ),
( *this )( ipatch ), smpi,
RadiationTables,
MultiphotonBreitWheelerTables,
localDiags );
}
} 
} 
} 
} 


timers.particles.update( params.printNow( itime ) );
#ifdef __DETAILED_TIMERS
timers.interpolator.update( *this, params.printNow( itime ) );
timers.pusher.update( *this, params.printNow( itime ) );
timers.projector.update( *this, params.printNow( itime ) );
timers.cell_keys.update( *this, params.printNow( itime ) );
timers.ionization.update( *this, params.printNow( itime ) );
timers.radiation.update( *this, params.printNow( itime ) );
timers.multiphoton_Breit_Wheeler_timer.update( *this, params.printNow( itime ) );
#endif

timers.syncPart.restart();
for( unsigned int ispec=0 ; ispec<( *this )( 0 )->vecSpecies.size(); ispec++ ) {
Species *spec = species( 0, ispec );
if ( (!params.Laser_Envelope_model) && (spec->isProj( time_dual, simWindow )) ){
SyncVectorPatch::exchangeParticles( ( *this ), ispec, params, smpi, timers, itime ); 
} 
} 
timers.syncPart.update( params.printNow( itime ) );

#ifdef __DETAILED_TIMERS
timers.sorting.update( *this, params.printNow( itime ) );
#endif
} 

void VectorPatch::projectionForDiags( Params &params,
SmileiMPI *smpi,
SimWindow *simWindow,
double time_dual, Timers &timers, int itime )
{

#pragma omp single
diag_flag = needsRhoJsNow( itime );

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->restartRhoJ();
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->isProj( time_dual, simWindow ) || diag_flag ) {
species( ipatch, ispec )->projectionForDiags( time_dual, ispec,
emfields( ipatch ),
params, diag_flag,
( *this )( ipatch ), smpi );
}
}

}

if( params.Laser_Envelope_model ) {
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->restartEnvChi();
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->isProj( time_dual, simWindow ) || diag_flag ) {
species( ipatch, ispec )->ponderomotiveProjectSusceptibility( time_dual, ispec,
emfields( ipatch ),
params, diag_flag,
( *this )( ipatch ), smpi,
localDiags );
} 
} 
} 
}

} 

void VectorPatch::finalizeAndSortParticles( Params &params, SmileiMPI *smpi, SimWindow *simWindow,
double time_dual, Timers &timers, int itime )
{
timers.syncPart.restart();



for( unsigned int ispec=0 ; ispec<( *this )( 0 )->vecSpecies.size(); ispec++ ) {
if( ( *this )( 0 )->vecSpecies[ispec]->isProj( time_dual, simWindow ) ) {
SyncVectorPatch::finalizeAndSortParticles( ( *this ), ispec, params, smpi, timers, itime ); 
}

}


#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->isProj( time_dual, simWindow ) || diag_flag ) {
species( ipatch, ispec )->dynamicsImportParticles( time_dual, ispec,
params,
( *this )( ipatch ), smpi,
localDiags );
}
}
}

timers.syncPart.update( params.printNow( itime ) );

} 


void VectorPatch::mergeParticles(Params &params, SmileiMPI *smpi, double time_dual,Timers &timers, int itime )
{
timers.particleMerging.restart();

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
if (species( ipatch, ispec )->has_merging_) {

if( species( ipatch, ispec )->merging_time_selection_->theTimeIsNow( itime ) ) {
species( ipatch, ispec )->mergeParticles( time_dual, ispec,
params,
( *this )( ipatch ), smpi,
localDiags );
}
}
}
}

timers.particleMerging.update( params.printNow( itime ) );

}

void VectorPatch::cleanParticlesOverhead(Params &params, Timers &timers, int itime )
{
timers.syncPart.restart();

if( itime%params.every_clean_particles_overhead==0 ) {
#pragma omp master
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->cleanParticlesOverhead( params );
}
#pragma omp barrier
}

timers.syncPart.update( params.printNow( itime ) );
}

void VectorPatch::injectParticlesFromBoundaries(Params &params, Timers &timers, unsigned int itime )
{

timers.particleInjection.restart();

#pragma omp single
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {

Patch * patch = ( *this )( ipatch );

if( ! patch->isAnyBoundary() ) continue;

vector<Particles> local_particles_vector( patch->particle_injector_vector_.size() );

for( unsigned int i_injector=0 ; i_injector<patch->particle_injector_vector_.size() ; i_injector++ ) {

ParticleInjector * particle_injector = patch->particle_injector_vector_[i_injector];

unsigned int axis = particle_injector->axis();
unsigned int min_max = particle_injector->min_max();

if( !patch->isBoundary( axis, min_max ) ) continue;

struct SubSpace init_space;
init_space.cell_index_[0] = 0;
init_space.cell_index_[1] = 0;
init_space.cell_index_[2] = 0;
init_space.box_size_[0]   = params.n_space[0];
init_space.box_size_[1]   = params.n_space[1];
init_space.box_size_[2]   = params.n_space[2];

if( min_max == 1 ) {
init_space.cell_index_[axis] = params.n_space[axis]-1;
}
init_space.box_size_[axis] = 1;

unsigned int i_species = particle_injector->getSpeciesNumber();
Species * injector_species = patch->vecSpecies[i_species];

local_particles_vector[i_injector].initialize( 0, *injector_species->particles );

ParticleCreator particle_creator;
particle_creator.associate( particle_injector, &local_particles_vector[i_injector], injector_species );

particle_creator.create( init_space, params, patch, itime );



if( !particle_injector->position_initialization_on_injector_ ) {

unsigned int number_of_particles = local_particles_vector[i_injector].size();

double position_shift[3] = {0., 0., 0.};
if( min_max == 0 ) {
position_shift[axis] = -params.cell_length[axis];
} else {
position_shift[axis] = params.cell_length[axis];
}

double * __restrict__ position_x = local_particles_vector[i_injector].getPtrPosition( 0 );
double * __restrict__ position_y = local_particles_vector[i_injector].getPtrPosition( 1 );
double * __restrict__ position_z = local_particles_vector[i_injector].getPtrPosition( 2 );

double * __restrict__ momentum_x = local_particles_vector[i_injector].getPtrMomentum( 0 );
double * __restrict__ momentum_y = local_particles_vector[i_injector].getPtrMomentum( 1 );
double * __restrict__ momentum_z = local_particles_vector[i_injector].getPtrMomentum( 2 );

if (params.nDim_field == 1) {

#pragma omp simd
for ( unsigned int ip = 0; ip < number_of_particles ; ip++ ) {
double inverse_gamma = params.timestep/std::sqrt(1. + momentum_x[ip]*momentum_x[ip] + momentum_y[ip]*momentum_y[ip]
+ momentum_z[ip]*momentum_z[ip]);

position_x[ip] += ( momentum_x[ip]
* inverse_gamma + position_shift[0]);
}

} else if (params.nDim_field == 2) {

#pragma omp simd
for ( unsigned int ip = 0; ip < number_of_particles ; ip++ ) {
double inverse_gamma = params.timestep/sqrt(1. + momentum_x[ip]*momentum_x[ip] + momentum_y[ip]*momentum_y[ip]
+ momentum_z[ip]*momentum_z[ip]);

position_x[ip] += ( momentum_x[ip]
* inverse_gamma + position_shift[0]);
position_y[ip] += ( momentum_y[ip]
* inverse_gamma + position_shift[1]);
}


} else if (params.nDim_field == 3) {

#pragma omp simd
for ( unsigned int ip = 0; ip < number_of_particles ; ip++ ) {
double inverse_gamma = params.timestep/std::sqrt(1. + momentum_x[ip]*momentum_x[ip]
+ momentum_y[ip]*momentum_y[ip] + momentum_z[ip]*momentum_z[ip]);

position_x[ip] += ( momentum_x[ip]
* inverse_gamma + position_shift[0]);
position_y[ip] += ( momentum_y[ip]
* inverse_gamma + position_shift[1]);
position_z[ip] += ( momentum_z[ip]
* inverse_gamma + position_shift[2]);
}

} 
} 
} 

for (unsigned int i_injector=0 ; i_injector<patch->particle_injector_vector_.size() ; i_injector++) {

ParticleInjector * particle_injector = patch->particle_injector_vector_[i_injector];

if (particle_injector->position_initialization_on_injector_) {

unsigned int i_injector_2 = particle_injector->position_initialization_on_injector_index_;

const unsigned int particle_number    = local_particles_vector[i_injector].size();
double *const __restrict__ px         = local_particles_vector[i_injector].getPtrPosition(0);
double *const __restrict__ py         = local_particles_vector[i_injector].getPtrPosition(1);
double *const __restrict__ pz         = local_particles_vector[i_injector].getPtrPosition(2);
const double *const __restrict__ lpvx = local_particles_vector[i_injector_2].getPtrPosition(0);
const double *const __restrict__ lpvy = local_particles_vector[i_injector_2].getPtrPosition(1);
const double *const __restrict__ lpvz = local_particles_vector[i_injector_2].getPtrPosition(2);

if (params.nDim_field == 3) {
#pragma omp simd
for ( unsigned int ip = 0; ip < particle_number ; ip++ ) {
px[ip] = lpvx[ip];
py[ip] = lpvy[ip];
pz[ip] = lpvz[ip];
}
}
else if (params.nDim_field == 2) {
#pragma omp simd
for ( unsigned int ip = 0; ip < particle_number ; ip++ ) {
px[ip] = lpvx[ip];
py[ip] = lpvy[ip];
}
}
else if (params.nDim_field == 1) {
#pragma omp simd
for ( unsigned int ip = 0; ip < particle_number ; ip++ ) {
px[ip] = lpvx[ip];
}
} 
} 

if( local_particles_vector[i_injector].size() > 0 ) {

unsigned int i_species = particle_injector->getSpeciesNumber();
Species * injector_species = species( ipatch, i_species );
Particles* particles = &local_particles_vector[i_injector];

int new_particle_number = particles->size() - 1;

for( int ip = new_particle_number ; ip >= 0 ; ip-- ) {
for( unsigned int axis = 0; axis<params.nDim_field; axis++ ) {
if( particles->Position[axis][ip] < 0. || particles->Position[axis][ip] > params.grid_length[axis] ) {
if( new_particle_number > ip ) {
particles->overwriteParticle( new_particle_number, ip );
}
new_particle_number--;
}
}
}

new_particle_number += 1;

double energy = 0.;
if( injector_species->mass_ > 0 ) {
for( int ip = 0; ip<new_particle_number; ip++ ) {
energy += particles->weight( ip )*( particles->LorentzFactor( ip )-1.0 );
}
injector_species->nrj_new_part_ += injector_species->mass_ * energy;
}
else if( injector_species->mass_ == 0 ) {
for( int ip=0; ip<new_particle_number; ip++ ) {
energy += particles->weight( ip )*( particles->momentumNorm( ip ) );
}
injector_species->nrj_new_part_ += energy;
}

if( new_particle_number > 0 ) {

particles->eraseParticleTrail( new_particle_number );
injector_species->importParticles( params, patches_[ipatch], *particles, localDiags );

}
} 
} 
} 

timers.particleInjection.update( params.printNow( itime ) );
}

void VectorPatch::computeCharge(bool old )
{
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
if (old) {
static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields)->restartRhoold();
} else {
( *this )( ipatch )->EMfields->restartRhoJ();
}
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->vectorized_operators ) {
species( ipatch, ispec )->computeCharge( ispec, emfields( ipatch ), old );
} else {
species( ipatch, ispec )->Species::computeCharge( ispec, emfields( ipatch ), old );
}
}
}

} 

void VectorPatch::computeChargeRelativisticSpecies( double time_primal, Params &params )
{
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->restartRhoJ();
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
if( ( species( ipatch, ispec )->relativistic_field_initialization_ ) &&
( (int)(time_primal/params.timestep) == species( ipatch, ispec )->iter_relativistic_initialization_ ) ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->vectorized_operators ) {
species( ipatch, ispec )->computeCharge( ispec, emfields( ipatch ) );
} else {
species( ipatch, ispec )->Species::computeCharge( ispec, emfields( ipatch ) );
}
}
}
}
} 

void VectorPatch::resetRhoJ(bool old)
{
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->restartRhoJ();
if (old)
static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields)->restartRhoold();
}
}

void VectorPatch::sumDensities( Params &params, double time_dual, Timers &timers, int itime, SimWindow *simWindow, SmileiMPI *smpi )
{
bool some_particles_are_moving = false;
unsigned int n_species( ( *this )( 0 )->vecSpecies.size() );
for( unsigned int ispec=0 ; ispec < n_species ; ispec++ ) {
if( ( *this )( 0 )->vecSpecies[ispec]->isProj( time_dual, simWindow ) ) {
some_particles_are_moving = true;
}
}
if( !some_particles_are_moving  && !diag_flag ) {
return;
}

timers.densities.restart();
if( diag_flag ) {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->computeTotalRhoJ();
}
}
timers.densities.update();

timers.syncDens.restart();
if( params.geometry != "AMcylindrical" ) {
if ( (!params.multiple_decomposition)||(itime==0) )
SyncVectorPatch::sumRhoJ( params, ( *this ), smpi, timers, itime ); 
} else {

if ( (!params.multiple_decomposition)||(itime==0) )
for( unsigned int imode = 0 ; imode < static_cast<ElectroMagnAM *>( patches_[0]->EMfields )->Jl_.size() ; imode++ ) {
SyncVectorPatch::sumRhoJ( params, ( *this ), imode, smpi, timers, itime );
}
}

if( diag_flag ) {
for( unsigned int ispec=0 ; ispec<( *this )( 0 )->vecSpecies.size(); ispec++ ) {
if( !( *this )( 0 )->vecSpecies[ispec]->particles->is_test ) {
updateFieldList( ispec, smpi );
if( params.geometry != "AMcylindrical" ) {
SyncVectorPatch::sumRhoJs( params, ( *this ), ispec, smpi, timers, itime ); 
} else {
for( unsigned int imode = 0 ; imode < static_cast<ElectroMagnAM *>( patches_[0]->EMfields )->Jl_.size() ; imode++ ) {
SyncVectorPatch::sumRhoJs( params, ( *this ), imode, ispec, smpi, timers, itime );
}
}
}
}
}
if ( ( params.geometry == "AMcylindrical" ) && (( *this )( 0 )->vecSpecies.size() > 0) ) {
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
if (emAM->isYmin){
( *this )( ipatch )->vecSpecies[0]->Proj->axisBC( emAM, diag_flag );
}
}
}
timers.syncDens.update( params.printNow( itime ) );
} 


void VectorPatch::sumSusceptibility( Params &params, double time_dual, Timers &timers, int itime, SimWindow *simWindow, SmileiMPI *smpi )
{
bool some_particles_are_moving = false;
unsigned int n_species( ( *this )( 0 )->vecSpecies.size() );
for( unsigned int ispec=0 ; ispec < n_species ; ispec++ ) {
if( ( *this )( 0 )->vecSpecies[ispec]->isProj( time_dual, simWindow ) ) {
some_particles_are_moving = true;
}
}
if( !some_particles_are_moving  && !diag_flag ) {
return;
}

timers.susceptibility.restart();
if( diag_flag ) {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->computeTotalEnvChi();
}
}

if( diag_flag ) {
for( unsigned int ispec=0 ; ispec<( *this )( 0 )->vecSpecies.size(); ispec++ ) {
if( !( *this )( 0 )->vecSpecies[ispec]->particles->is_test ) {
updateFieldList( ispec, smpi );
SyncVectorPatch::sumEnvChis( params, ( *this ), ispec, smpi, timers, itime );
}
}
}

timers.susceptibility.update();

timers.susceptibility.restart();

SyncVectorPatch::sumEnvChi( params, ( *this ), smpi, timers, itime ); 


if ( ( params.geometry == "AMcylindrical" ) && (( *this )( 0 )->vecSpecies.size() > 0) ) {
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
if (emAM->isYmin){
( *this )( ipatch )->vecSpecies[0]->Proj->axisBCEnvChi( &( *emAM->Env_Chi_ )( 0 ) );
if (diag_flag) {
unsigned int n_species = ( *this )( 0 )->vecSpecies.size();
int imode =0;
for( unsigned int ispec = 0 ; ispec < n_species ; ispec++ ) {
unsigned int ifield = imode*n_species+ispec;
double *EnvChi = emAM->Env_Chi_s    [ifield] ? &( * ( emAM->Env_Chi_s[ifield] ) )( 0 ) : NULL ;
( *this )( ipatch )->vecSpecies[ispec]->Proj->axisBCEnvChi( EnvChi );
}
}
}
}
}

timers.susceptibility.update();

} 



void VectorPatch::solveMaxwell( Params &params, SimWindow *simWindow, int itime, double time_dual, Timers &timers, SmileiMPI *smpi )
{
timers.maxwell.restart();

if (params.currentFilter_passes.size() > 0){
for( unsigned int ipassfilter=0 ; ipassfilter<*std::max_element(std::begin(params.currentFilter_passes), std::end(params.currentFilter_passes)) ; ipassfilter++ ) {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
if (params.currentFilter_model=="binomial"){
( *this )( ipatch )->EMfields->binomialCurrentFilter(ipassfilter, params.currentFilter_passes);
}
if (params.currentFilter_model=="customFIR"){
( *this )( ipatch )->EMfields->customFIRCurrentFilter(ipassfilter, params.currentFilter_passes, params.currentFilter_kernelFIR);
}
}
if (params.geometry != "AMcylindrical"){
if (params.currentFilter_model=="customFIR"){
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( listJx_, *this, smpi );
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( listJy_, *this, smpi );
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( listJz_, *this, smpi );
} else {
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( listJx_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( listJx_, *this );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( listJy_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( listJy_, *this );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( listJz_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( listJz_, *this );
}
} else {
for (unsigned int imode=0 ; imode < params.nmodes; imode++) {
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( listJl_[imode], *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( listJl_[imode], *this );
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( listJr_[imode], *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( listJr_[imode], *this );
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( listJt_[imode], *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( listJt_[imode], *this );
}
}
}
}

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
if( !params.is_spectral ) {
( *this )( ipatch )->EMfields->saveMagneticFields( params.is_spectral );
}
( *( *this )( ipatch )->EMfields->MaxwellAmpereSolver_ )( ( *this )( ipatch )->EMfields );
}

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *( *this )( ipatch )->EMfields->MaxwellFaradaySolver_ )( ( *this )( ipatch )->EMfields );
}
timers.maxwell.update( params.printNow( itime ) );


timers.syncField.restart();
if( params.geometry != "AMcylindrical" ) {
if( params.is_spectral ) {
SyncVectorPatch::exchangeE( params, ( *this ), smpi );
}
SyncVectorPatch::exchangeB( params, ( *this ), smpi );
} else {
for( unsigned int imode = 0 ; imode < static_cast<ElectroMagnAM *>( patches_[0]->EMfields )->El_.size() ; imode++ ) {
SyncVectorPatch::exchangeE( params, ( *this ), imode, smpi );
SyncVectorPatch::exchangeB( params, ( *this ), imode, smpi );
}
}
timers.syncField.update( params.printNow( itime ) );


if ( (params.multiple_decomposition) && ( itime!=0 ) && ( time_dual > params.time_fields_frozen ) ) { 
timers.syncField.restart();
if( params.is_spectral && params.geometry != "AMcylindrical" ) {
SyncVectorPatch::finalizeexchangeE( params, ( *this ) );
}

if( params.geometry != "AMcylindrical" )
SyncVectorPatch::finalizeexchangeB( params, ( *this ) );
timers.syncField.update( params.printNow( itime ) );

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->boundaryConditions( itime, time_dual, ( *this )( ipatch ), params, simWindow );
}
if ( params.EM_BCs[0][0] == "PML" ) { 
SyncVectorPatch::exchangeForPML( params, (*this), smpi );
}
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
if( !params.is_spectral ) {
( *this )( ipatch )->EMfields->centerMagneticFields();
}
}
if( params.is_spectral && params.geometry != "AMcylindrical" ) {
saveOldRho( params );
}
}

} 

void VectorPatch::solveEnvelope( Params &params, SimWindow *simWindow, int itime, double time_dual, Timers &timers, SmileiMPI *smpi )
{

if( ( *this )( 0 )->EMfields->envelope!=NULL ) {

timers.envelope.restart();
SyncVectorPatch::exchangeEnvChi( params, ( *this ), smpi );

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {

( *this )( ipatch )->EMfields->envelope->savePhiAndGradPhi();

if ( ( *this )( ipatch )->EMfields->envelope->envelope_solver == "explicit" ){
( *this )( ipatch )->EMfields->envelope->updateEnvelope( ( *this )( ipatch )->EMfields );
} else if ( ( *this )( ipatch )->EMfields->envelope->envelope_solver == "explicit_reduced_dispersion" ) {
( *this )( ipatch )->EMfields->envelope->updateEnvelopeReducedDispersion( ( *this )( ipatch )->EMfields );
}

( *this )( ipatch )->EMfields->envelope->boundaryConditions( itime, time_dual, ( *this )( ipatch ), params, simWindow, ( *this )( ipatch )->EMfields );

}

SyncVectorPatch::exchangeA( params, ( *this ), smpi );
SyncVectorPatch::finalizeexchangeA( params, ( *this ) );

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->envelope->computePhiEnvAEnvE( ( *this )( ipatch )->EMfields );
( *this )( ipatch )->EMfields->envelope->computeGradientPhi( ( *this )( ipatch )->EMfields );
( *this )( ipatch )->EMfields->envelope->centerPhiAndGradPhi();
}

SyncVectorPatch::exchangeEnvEx( params, ( *this ), smpi );
SyncVectorPatch::finalizeexchangeEnvEx( params, ( *this ) );
SyncVectorPatch::exchangeGradPhi( params, ( *this ), smpi );
SyncVectorPatch::finalizeexchangeGradPhi( params, ( *this ) );

timers.envelope.update();
}

} 

void VectorPatch::finalizeSyncAndBCFields( Params &params, SmileiMPI *smpi, SimWindow *simWindow,
double time_dual, Timers &timers, int itime )
{
if ( (!params.multiple_decomposition) && ( itime!=0 ) && ( time_dual > params.time_fields_frozen ) ) { 
if( params.geometry != "AMcylindrical" ) {
timers.syncField.restart();
SyncVectorPatch::finalizeexchangeB( params, ( *this ) );
timers.syncField.update( params.printNow( itime ) );
}

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
if ( (!params.is_spectral) || (params.geometry!= "AMcylindrical") )
( *this )( ipatch )->EMfields->boundaryConditions( itime, time_dual, ( *this )( ipatch ), params, simWindow );

}
if ( params.EM_BCs[0][0] == "PML" ) { 
SyncVectorPatch::exchangeForPML( params, (*this), smpi );
}

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
if( !params.is_spectral ) {
( *this )( ipatch )->EMfields->centerMagneticFields();
}
}
}

} 


void VectorPatch::initExternals( Params &params )
{
for( unsigned int ipatch=0; ipatch<size(); ipatch++ ) {
Patch * patch = ( *this )( ipatch );

for( unsigned ib=0; ib<2*params.nDim_field; ib++ ) {

if( patch->isBoundary(ib) && patch->EMfields->emBoundCond[ib] ) {
unsigned int nlaser = patch->EMfields->emBoundCond[ib]->vecLaser.size();
for( unsigned int ilaser = 0; ilaser < nlaser; ilaser++ ) {
patch->EMfields->emBoundCond[ib]->vecLaser[ilaser]->initFields( params, patch );
}
}

}
}

for( unsigned int ipatch=0; ipatch<size(); ipatch++ ) {
( *this )( ipatch )->EMfields->initAntennas( ( *this )( ipatch ), params );
}
}


void VectorPatch::initAllDiags( Params &params, SmileiMPI *smpi )
{

for( unsigned int idiag = 0 ; idiag < globalDiags.size() ; idiag++ ) {
globalDiags[idiag]->init( params, smpi, *this );
if( smpi->isMaster() ) {
globalDiags[idiag]->openFile( params, smpi );
}
}

for( unsigned int idiag = 0 ; idiag < localDiags.size() ; idiag++ ) {
localDiags[idiag]->init( params, smpi, *this );
}

} 


void VectorPatch::closeAllDiags( SmileiMPI *smpi )
{
if( smpi->isMaster() )
for( unsigned int idiag = 0 ; idiag < globalDiags.size() ; idiag++ ) {
globalDiags[idiag]->closeFile();
}

for( unsigned int idiag = 0 ; idiag < localDiags.size() ; idiag++ ) {
localDiags[idiag]->closeFile();
}
}


void VectorPatch::runAllDiags( Params &params, SmileiMPI *smpi, unsigned int itime, Timers &timers, SimWindow *simWindow )
{
timers.diags.restart();

vector<double> MPI_mins, MPI_maxs;
for( unsigned int idiag = 0 ; idiag < globalDiags.size() ; idiag++ ) {
diag_timers_[idiag]->restart();

DiagnosticParticleBinningBase* binning = dynamic_cast<DiagnosticParticleBinningBase*>( globalDiags[idiag] );
if( binning && binning->has_auto_limits_ ) {
#pragma omp single
binning->theTimeIsNow_ = binning->theTimeIsNow( itime );

if( binning->theTimeIsNow_ ) {
#pragma omp master
{
binning->patches_mins.resize( size() );
binning->patches_maxs.resize( size() );
}
#pragma omp barrier
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0; ipatch<size(); ipatch++ ) {
binning->calculate_auto_limits( patches_[ipatch], simWindow, ipatch );
}
#pragma omp master
{
vector<double> mins = binning->patches_mins[0];
vector<double> maxs = binning->patches_maxs[0];
for( unsigned int i=0; i<mins.size(); i++ ) {
for( unsigned int ipatch=1; ipatch<size(); ipatch++ ) {
mins[i] = min( mins[i], binning->patches_mins[ipatch][i] );
maxs[i] = max( maxs[i], binning->patches_maxs[ipatch][i] );
}
}
MPI_mins.insert( MPI_mins.end(), mins.begin(), mins.end() );
MPI_maxs.insert( MPI_maxs.end(), maxs.begin(), maxs.end() );
}
}
}
diag_timers_[idiag]->update();
}
#pragma omp master
{
vector<double> global_mins( MPI_mins.size() ), global_maxs( MPI_maxs.size() );
if( MPI_mins.size() > 0 ) {
MPI_Allreduce( &MPI_mins[0], &global_mins[0], (int) MPI_mins.size(), MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
}
if( MPI_maxs.size() > 0 ) {
MPI_Allreduce( &MPI_maxs[0], &global_maxs[0], (int) MPI_maxs.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
}
unsigned int imin = 0, imax = 0;
for( unsigned int idiag = 0 ; idiag < globalDiags.size() ; idiag++ ) {
DiagnosticParticleBinningBase* binning = dynamic_cast<DiagnosticParticleBinningBase*>( globalDiags[idiag] );
if( binning && binning->has_auto_limits_ && binning->theTimeIsNow_ ) {
for( unsigned int iaxis=0; iaxis<binning->histogram->axes.size(); iaxis++ ) {
HistogramAxis * axis = binning->histogram->axes[iaxis];
if( std::isnan( axis->min ) ) {
axis->global_min = global_mins[imin++];
}
if( std::isnan( axis->max ) ) {
axis->global_max = global_maxs[imax++];
}
}
}
}
}
#pragma omp barrier

for( unsigned int idiag = 0 ; idiag < globalDiags.size() ; idiag++ ) {
diag_timers_[idiag]->restart();

#pragma omp single
globalDiags[idiag]->theTimeIsNow_ = globalDiags[idiag]->prepare( itime );

if( globalDiags[idiag]->theTimeIsNow_ ) {
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
globalDiags[idiag]->run( ( *this )( ipatch ), itime, simWindow );
}
#pragma omp single
smpi->computeGlobalDiags( globalDiags[idiag], itime );
#pragma omp single
globalDiags[idiag]->write( itime, smpi );
}

diag_timers_[idiag]->update();
}

for( unsigned int idiag = 0 ; idiag < localDiags.size() ; idiag++ ) {
diag_timers_[globalDiags.size()+idiag]->restart();

#pragma omp single
localDiags[idiag]->theTimeIsNow_ = localDiags[idiag]->prepare( itime );
if( localDiags[idiag]->theTimeIsNow_ ) {
localDiags[idiag]->run( smpi, *this, itime, simWindow, timers );
}

diag_timers_[globalDiags.size()+idiag]->update();
}

if( diag_flag ) {
#pragma omp barrier
#pragma omp single
diag_flag = false;
#pragma omp for
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->restartRhoJs();
}
}
timers.diags.update();

if( itime==0 ) {
for( unsigned int idiag = 0 ; idiag < diag_timers_.size() ; idiag++ ) {
diag_timers_[idiag]->reboot();
}
}

} 


bool VectorPatch::isRhoNull( SmileiMPI *smpi )
{
double norm2( 0. );
double locnorm2( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
locnorm2 += ( *this )( ipatch )->EMfields->computeRhoNorm2();
}

MPI_Allreduce( &locnorm2, &norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

return ( norm2<=0. );
} 


void VectorPatch::solvePoisson( Params &params, SmileiMPI *smpi )
{
Timer ptimer( "global" );
ptimer.init( smpi );
ptimer.restart();


unsigned int iteration_max = params.poisson_max_iteration;
double           error_max = params.poisson_max_error;
unsigned int iteration=0;

double rnew_dot_rnew_local( 0. );
double rnew_dot_rnew( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->initPoisson( ( *this )( ipatch ) );
rnew_dot_rnew_local += ( *this )( ipatch )->EMfields->compute_r();
}
MPI_Allreduce( &rnew_dot_rnew_local, &rnew_dot_rnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

std::vector<Field *> Ex_;
std::vector<Field *> Ap_;

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
Ex_.push_back( ( *this )( ipatch )->EMfields->Ex_ );
Ap_.push_back( ( *this )( ipatch )->EMfields->Ap_ );
}

unsigned int nx_p2_global = ( params.n_space_global[0]+1 );
if( Ex_[0]->dims_.size()>1 ) {
nx_p2_global *= ( params.n_space_global[1]+1 );
if( Ex_[0]->dims_.size()>2 ) {
nx_p2_global *= ( params.n_space_global[2]+1 );
}
}

double ctrl = rnew_dot_rnew / ( double )( nx_p2_global );

if( smpi->isMaster() ) {
DEBUG( "Starting iterative loop for CG method" );
}
while( ( ctrl > error_max ) && ( iteration<iteration_max ) ) {
iteration++;
if( smpi->isMaster() ) {
DEBUG( "iteration " << iteration << " started with control parameter ctrl = " << ctrl*1.e14 << " x 1e-14" );
}

double r_dot_r = rnew_dot_rnew;

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->compute_Ap( ( *this )( ipatch ) );
}

SyncVectorPatch::exchangeAlongAllDirections<double,Field>( Ap_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( Ap_, *this );

double p_dot_Ap       = 0.0;
double p_dot_Ap_local = 0.0;
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
p_dot_Ap_local += ( *this )( ipatch )->EMfields->compute_pAp();
}
MPI_Allreduce( &p_dot_Ap_local, &p_dot_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->update_pand_r( r_dot_r, p_dot_Ap );
}

rnew_dot_rnew       = 0.0;
rnew_dot_rnew_local = 0.0;
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
rnew_dot_rnew_local += ( *this )( ipatch )->EMfields->compute_r();
}
MPI_Allreduce( &rnew_dot_rnew_local, &rnew_dot_rnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
if( smpi->isMaster() ) {
DEBUG( "new residual norm: rnew_dot_rnew = " << rnew_dot_rnew );
}

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->update_p( rnew_dot_rnew, r_dot_r );
}

ctrl = rnew_dot_rnew / ( double )( nx_p2_global );
if( smpi->isMaster() ) {
DEBUG( "iteration " << iteration << " done, exiting with control parameter ctrl = " << ctrl );
}

}


if( iteration_max>0 && iteration == iteration_max ) {
if( smpi->isMaster() )
WARNING( "Poisson solver did not converge: reached maximum iteration number: " << iteration
<< ", relative err is ctrl = " << 1.0e14*ctrl << " x 1e-14" );
} else {
if( smpi->isMaster() )
MESSAGE( 1, "Poisson solver converged at iteration: " << iteration
<< ", relative err is ctrl = " << 1.0e14*ctrl << " x 1e-14" );
}

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->initE( ( *this )( ipatch ) );
}

SyncVectorPatch::exchangeE( params, *this, smpi );
SyncVectorPatch::finalizeexchangeE( params, *this );

vector<double> E_Add( Ex_[0]->dims_.size(), 0. );
if( Ex_[0]->dims_.size()==3 ) {
double Ex_avg_local( 0. ), Ex_avg( 0. ), Ey_avg_local( 0. ), Ey_avg( 0. ), Ez_avg_local( 0. ), Ez_avg( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
Ex_avg_local += ( *this )( ipatch )->EMfields->computeExSum();
Ey_avg_local += ( *this )( ipatch )->EMfields->computeEySum();
Ez_avg_local += ( *this )( ipatch )->EMfields->computeEzSum();
}

MPI_Allreduce( &Ex_avg_local, &Ex_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
MPI_Allreduce( &Ey_avg_local, &Ey_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
MPI_Allreduce( &Ez_avg_local, &Ez_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

E_Add[0] = -Ex_avg/( ( params.n_space[0]+2 )*( params.n_space[1]+1 )*( params.n_space[2]+1 ) );
E_Add[1] = -Ey_avg/( ( params.n_space[0]+1 )*( params.n_space[1]+2 )*( params.n_space[2]+1 ) );;
E_Add[2] = -Ez_avg/( ( params.n_space[0]+1 )*( params.n_space[1]+1 )*( params.n_space[2]+2 ) );;
} else if( Ex_[0]->dims_.size()==2 ) {
double Ex_XminYmax = 0.0;
double Ey_XminYmax = 0.0;
double Ex_XmaxYmin = 0.0;
double Ey_XmaxYmin = 0.0;

std::vector<int> xcall( 2, 0 );
xcall[0] = 0;
xcall[1] = params.number_of_patches[1]-1;
int patch_YmaxXmin = domain_decomposition_->getDomainId( xcall );
int rank_XminYmax = smpi->hrank( patch_YmaxXmin );
xcall[0] = params.number_of_patches[0]-1;
xcall[1] = 0;
int patch_YminXmax = domain_decomposition_->getDomainId( xcall );
int rank_XmaxYmin = smpi->hrank( patch_YminXmax );

if( smpi->getRank() == rank_XminYmax ) {
Ex_XminYmax = ( *this )( patch_YmaxXmin-( this->refHindex_ ) )->EMfields->getEx_XminYmax();
Ey_XminYmax = ( *this )( patch_YmaxXmin-( this->refHindex_ ) )->EMfields->getEy_XminYmax();
}

if( smpi->getRank() == rank_XmaxYmin ) {
Ex_XmaxYmin = ( *this )( patch_YminXmax-( this->refHindex_ ) )->EMfields->getEx_XmaxYmin();
Ey_XmaxYmin = ( *this )( patch_YminXmax-( this->refHindex_ ) )->EMfields->getEy_XmaxYmin();
}

MPI_Bcast( &Ex_XminYmax, 1, MPI_DOUBLE, rank_XminYmax, MPI_COMM_WORLD );
MPI_Bcast( &Ey_XminYmax, 1, MPI_DOUBLE, rank_XminYmax, MPI_COMM_WORLD );

MPI_Bcast( &Ex_XmaxYmin, 1, MPI_DOUBLE, rank_XmaxYmin, MPI_COMM_WORLD );
MPI_Bcast( &Ey_XmaxYmin, 1, MPI_DOUBLE, rank_XmaxYmin, MPI_COMM_WORLD );

E_Add[0] = -0.5*( Ex_XminYmax+Ex_XmaxYmin );
E_Add[1] = -0.5*( Ey_XminYmax+Ey_XmaxYmin );

#ifdef _3D_LIKE_CENTERING
double Ex_avg_local( 0. ), Ex_avg( 0. ), Ey_avg_local( 0. ), Ey_avg( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
Ex_avg_local += ( *this )( ipatch )->EMfields->computeExSum();
Ey_avg_local += ( *this )( ipatch )->EMfields->computeEySum();
}

MPI_Allreduce( &Ex_avg_local, &Ex_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
MPI_Allreduce( &Ey_avg_local, &Ey_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

E_Add[0] = -Ex_avg/( ( params.n_space[0]+2 )*( params.n_space[1]+1 ) );
E_Add[1] = -Ey_avg/( ( params.n_space[0]+1 )*( params.n_space[1]+2 ) );
#endif

} else if( Ex_[0]->dims_.size()==1 ) {
double Ex_Xmin = 0.0;
double Ex_Xmax = 0.0;

unsigned int rankXmin = 0;
if( smpi->getRank() == 0 ) {
Ex_Xmin = ( *this )( ( 0 )-( this->refHindex_ ) )->EMfields->getEx_Xmin();
}
MPI_Bcast( &Ex_Xmin, 1, MPI_DOUBLE, rankXmin, MPI_COMM_WORLD );

unsigned int rankXmax = smpi->getSize()-1;
if( smpi->getRank() == smpi->getSize()-1 ) {
Ex_Xmax = ( *this )( ( params.number_of_patches[0]-1 )-( this->refHindex_ ) )->EMfields->getEx_Xmax();
}
MPI_Bcast( &Ex_Xmax, 1, MPI_DOUBLE, rankXmax, MPI_COMM_WORLD );
E_Add[0] = -0.5*( Ex_Xmin+Ex_Xmax );

#ifdef _3D_LIKE_CENTERING
double Ex_avg_local( 0. ), Ex_avg( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
Ex_avg_local += ( *this )( ipatch )->EMfields->computeExSum();
}

MPI_Allreduce( &Ex_avg_local, &Ex_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

E_Add[0] = -Ex_avg/( ( params.n_space[0]+2 ) );
#endif

}

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->centeringE( E_Add );
}

double deltaPoisson_max = 0.0;
int i_deltaPoisson_max  = -1;

#ifdef _A_FINALISER
for( unsigned int i=0; i<nx_p; i++ ) {
double deltaPoisson = abs( ( ( *Ex1D )( i+1 )-( *Ex1D )( i ) )/dx - ( *rho1D )( i ) );
if( deltaPoisson > deltaPoisson_max ) {
deltaPoisson_max   = deltaPoisson;
i_deltaPoisson_max = i;
}
}
#endif

if( smpi->isMaster() ) {
MESSAGE( 1, "Poisson equation solved. Maximum err = " << deltaPoisson_max << " at i= " << i_deltaPoisson_max );
}

ptimer.update();
MESSAGE( "Time in Poisson : " << ptimer.getTime() );

} 

void VectorPatch::solvePoissonAM( Params &params, SmileiMPI *smpi )
{

unsigned int iteration_max = params.poisson_max_iteration;
double           error_max = params.poisson_max_error;
unsigned int iteration=0;

double rnew_dot_rnew_localAM_( 0. );
double rnew_dot_rnewAM_( 0. );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->initPoisson( ( *this )( ipatch ) );
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->initPoissonFields( ( *this )( ipatch ) );
}

std::vector<Field *> El_;
std::vector<Field *> Er_;
std::vector<Field *> Et_;

std::vector<Field *> El_Poisson_;
std::vector<Field *> Er_Poisson_;
std::vector<Field *> Et_Poisson_;

std::vector<Field *> Ap_AM_;

for( unsigned int imode=0 ; imode<params.nmodes_classical_Poisson_field_init ; imode++ ) {

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->initPoisson_init_phi_r_p_Ap( ( *this )( ipatch ), imode );
rnew_dot_rnew_localAM_ += emAM->compute_r();
}

MPI_Allreduce( &rnew_dot_rnew_localAM_, &rnew_dot_rnewAM_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
El_.push_back( emAM->El_[imode] );
Er_.push_back( emAM->Er_[imode] );
Et_.push_back( emAM->Et_[imode] );
El_Poisson_.push_back( emAM->El_Poisson_ );
Er_Poisson_.push_back( emAM->Er_Poisson_ );
Et_Poisson_.push_back( emAM->Et_Poisson_ );

Ap_AM_.push_back( emAM->Ap_AM_ );
}

unsigned int nx_p2_global = ( params.n_space_global[0]+1 );
if( El_Poisson_[0]->dims_.size()>1 ) {
nx_p2_global *= ( params.n_space_global[1]+1 );
if( El_Poisson_[0]->dims_.size()>2 ) {
nx_p2_global *= ( params.n_space_global[2]+1 );
}
}

double norm2_source_term = sqrt( std::abs(rnew_dot_rnewAM_) );
double ctrl = sqrt( std::abs(rnew_dot_rnewAM_) ) / norm2_source_term; 

if( smpi->isMaster() ) {
DEBUG( "Starting iterative loop for CG method for the mode "<<imode );
}

iteration = 0;
while( ( ctrl > error_max ) && ( iteration<iteration_max ) ) {
iteration++;

if( ( smpi->isMaster() ) && ( iteration%1000==0 ) ) {
MESSAGE( "iteration " << iteration << " started with control parameter ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}

double r_dot_rAM_ = rnew_dot_rnewAM_;

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->compute_Ap_Poisson_AM( ( *this )( ipatch ), imode );
}

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Ap_AM_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ap_AM_, *this );


std::complex<double> p_dot_ApAM_       = 0.0;
std::complex<double> p_dot_Ap_localAM_ = 0.0;
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
p_dot_Ap_localAM_ += emAM->compute_pAp_AM();
}
MPI_Allreduce( &p_dot_Ap_localAM_, &p_dot_ApAM_, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->update_pand_r_AM( r_dot_rAM_, p_dot_ApAM_ );
}

rnew_dot_rnewAM_       = 0.0;
rnew_dot_rnew_localAM_ = 0.0;
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
rnew_dot_rnew_localAM_ += emAM->compute_r();
}
MPI_Allreduce( &rnew_dot_rnew_localAM_, &rnew_dot_rnewAM_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
if( smpi->isMaster() ) {
DEBUG( "new residual norm: rnew_dot_rnew = " << rnew_dot_rnewAM_ );
}

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->update_p( rnew_dot_rnewAM_, r_dot_rAM_ );
}

ctrl = sqrt( std::abs(rnew_dot_rnewAM_) )/norm2_source_term;
if( smpi->isMaster() ) {
DEBUG( "iteration " << iteration << " done, exiting with control parameter ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}

}


if( iteration_max>0 && iteration == iteration_max ) {
if( smpi->isMaster() )
WARNING( "Poisson equation solver did not converge: reached maximum iteration number: " << iteration
<< ", relative err is ctrl = " << 1.0e22*ctrl << "x 1.e-22" );
} else {
if( smpi->isMaster() )
MESSAGE( 1, "Poisson equation solver converged at iteration: " << iteration
<< ", relative err is ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}



for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->initE_Poisson_AM( ( *this )( ipatch ), imode );
} 

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( El_Poisson_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( El_Poisson_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Er_Poisson_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Er_Poisson_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Et_Poisson_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Et_Poisson_, *this );


MESSAGE( 0, "Summing fields computed with the Poisson solver to the grid fields" );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->sum_Poisson_fields_to_em_fields_AM( ( *this )( ipatch ), params, imode );
} 


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {

El_.pop_back();
Er_.pop_back();
Et_.pop_back();

Ap_AM_.pop_back();

}

}  

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->delete_phi_r_p_Ap( ( *this )( ipatch ) );
emAM->delete_Poisson_fields( ( *this )( ipatch ) );
}

for( unsigned int imode = 0 ; imode < params.nmodes_classical_Poisson_field_init ; imode++ ) {
SyncVectorPatch::exchangeE( params, ( *this ), imode, smpi );
SyncVectorPatch::finalizeexchangeE( params, ( *this ), imode ); 
}

MESSAGE( "Poisson equation solved" );

}  


void VectorPatch::runNonRelativisticPoissonModule( Params &params, SmileiMPI* smpi,  Timers &timers )
{

#pragma omp master
{
if( !isRhoNull( smpi ) ) {
TITLE( "Initializing E field through Poisson solver" );
if (params.geometry != "AMcylindrical"){
solvePoisson( params, smpi );
} else {
solvePoissonAM( params, smpi );
}
}
}
#pragma omp barrier

}

void VectorPatch::runRelativisticModule( double time_prim, Params &params, SmileiMPI* smpi,  Timers &timers )
{
computeChargeRelativisticSpecies( time_prim, params );

if (params.geometry != "AMcylindrical"){
SyncVectorPatch::sum<double,Field>( listrho_, (*this), smpi, timers, 0 );
} else {
for( unsigned int imode=0 ; imode<params.nmodes ; imode++ ) {
SyncVectorPatch::sumRhoJ( params, (*this), imode, smpi, timers, 0 );
}
}

#pragma omp master
{
if( !isRhoNull( smpi ) ) {
TITLE( "Initializing relativistic species fields" );
if (params.geometry != "AMcylindrical"){
solveRelativisticPoisson( params, smpi, time_prim );
} else {
solveRelativisticPoissonAM( params, smpi, time_prim );
}
}
}
#pragma omp barrier

resetRhoJ();

}


void VectorPatch::solveRelativisticPoisson( Params &params, SmileiMPI *smpi, double time_primal )
{





double s_gamma( 0. );
uint64_t nparticles( 0 );
for( unsigned int ispec=0 ; ispec<( *this )( 0 )->vecSpecies.size() ; ispec++ ) {
if( species( 0, ispec )->relativistic_field_initialization_ ) {
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
if( (int)(time_primal/params.timestep)==species( ipatch, ispec )->iter_relativistic_initialization_ ) {
s_gamma += species( ipatch, ispec )->sumGamma();
nparticles += species( ipatch, ispec )->getNbrOfParticles();
}
}
}
}
double gamma_global( 0. );
MPI_Allreduce( &s_gamma, &gamma_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
uint64_t nparticles_global( 0 );
MPI_Allreduce( &nparticles, &nparticles_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
MESSAGE( "GAMMA = " << gamma_global/( double )nparticles_global );


double gamma_mean = gamma_global/( double )nparticles_global;

unsigned int iteration_max = params.relativistic_poisson_max_iteration;
double           error_max = params.relativistic_poisson_max_error;
unsigned int iteration=0;

double rnew_dot_rnew_local( 0. );
double rnew_dot_rnew( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->initPoisson( ( *this )( ipatch ) );
rnew_dot_rnew_local += ( *this )( ipatch )->EMfields->compute_r();
( *this )( ipatch )->EMfields->initRelativisticPoissonFields( ( *this )( ipatch ) );
}
MPI_Allreduce( &rnew_dot_rnew_local, &rnew_dot_rnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

std::vector<Field *> Ex_;
std::vector<Field *> Ey_;
std::vector<Field *> Ez_;
std::vector<Field *> Bx_;
std::vector<Field *> By_;
std::vector<Field *> Bz_;
std::vector<Field *> Bx_m;
std::vector<Field *> By_m;
std::vector<Field *> Bz_m;

std::vector<Field *> Ex_rel_;
std::vector<Field *> Ey_rel_;
std::vector<Field *> Ez_rel_;
std::vector<Field *> Bx_rel_;
std::vector<Field *> By_rel_;
std::vector<Field *> Bz_rel_;

std::vector<Field *> Bx_rel_t_plus_halfdt_;
std::vector<Field *> By_rel_t_plus_halfdt_;
std::vector<Field *> Bz_rel_t_plus_halfdt_;
std::vector<Field *> Bx_rel_t_minus_halfdt_;
std::vector<Field *> By_rel_t_minus_halfdt_;
std::vector<Field *> Bz_rel_t_minus_halfdt_;


std::vector<Field *> Ap_;

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
Ex_.push_back( ( *this )( ipatch )->EMfields->Ex_ );
Ey_.push_back( ( *this )( ipatch )->EMfields->Ey_ );
Ez_.push_back( ( *this )( ipatch )->EMfields->Ez_ );
Bx_.push_back( ( *this )( ipatch )->EMfields->Bx_ );
By_.push_back( ( *this )( ipatch )->EMfields->By_ );
Bz_.push_back( ( *this )( ipatch )->EMfields->Bz_ );
Bx_m.push_back( ( *this )( ipatch )->EMfields->Bx_m );
By_m.push_back( ( *this )( ipatch )->EMfields->By_m );
Bz_m.push_back( ( *this )( ipatch )->EMfields->Bz_m );
Ex_rel_.push_back( ( *this )( ipatch )->EMfields->Ex_rel_ );
Ey_rel_.push_back( ( *this )( ipatch )->EMfields->Ey_rel_ );
Ez_rel_.push_back( ( *this )( ipatch )->EMfields->Ez_rel_ );
Bx_rel_.push_back( ( *this )( ipatch )->EMfields->Bx_rel_ );
By_rel_.push_back( ( *this )( ipatch )->EMfields->By_rel_ );
Bz_rel_.push_back( ( *this )( ipatch )->EMfields->Bz_rel_ );
Bx_rel_t_plus_halfdt_.push_back( ( *this )( ipatch )->EMfields->Bx_rel_t_plus_halfdt_ );
By_rel_t_plus_halfdt_.push_back( ( *this )( ipatch )->EMfields->By_rel_t_plus_halfdt_ );
Bz_rel_t_plus_halfdt_.push_back( ( *this )( ipatch )->EMfields->Bz_rel_t_plus_halfdt_ );
Bx_rel_t_minus_halfdt_.push_back( ( *this )( ipatch )->EMfields->Bx_rel_t_minus_halfdt_ );
By_rel_t_minus_halfdt_.push_back( ( *this )( ipatch )->EMfields->By_rel_t_minus_halfdt_ );
Bz_rel_t_minus_halfdt_.push_back( ( *this )( ipatch )->EMfields->Bz_rel_t_minus_halfdt_ );

Ap_.push_back( ( *this )( ipatch )->EMfields->Ap_ );
}

unsigned int nx_p2_global = ( params.n_space_global[0]+1 );
if( Ex_rel_[0]->dims_.size()>1 ) {
nx_p2_global *= ( params.n_space_global[1]+1 );
if( Ex_rel_[0]->dims_.size()>2 ) {
nx_p2_global *= ( params.n_space_global[2]+1 );
}
}


double norm2_source_term = sqrt( rnew_dot_rnew );
double ctrl = sqrt( rnew_dot_rnew ) / norm2_source_term; 

if( smpi->isMaster() ) {
DEBUG( "Starting iterative loop for CG method" );
}
while( ( ctrl > error_max ) && ( iteration<iteration_max ) ) {
iteration++;

if( ( smpi->isMaster() ) && ( iteration%1000==0 ) ) {
MESSAGE( "iteration " << iteration << " started with control parameter ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}

double r_dot_r = rnew_dot_rnew;

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->compute_Ap_relativistic_Poisson( ( *this )( ipatch ), gamma_mean );
}

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Ap_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ap_, *this );


double p_dot_Ap       = 0.0;
double p_dot_Ap_local = 0.0;
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
p_dot_Ap_local += ( *this )( ipatch )->EMfields->compute_pAp();
}
MPI_Allreduce( &p_dot_Ap_local, &p_dot_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->update_pand_r( r_dot_r, p_dot_Ap );
}

rnew_dot_rnew       = 0.0;
rnew_dot_rnew_local = 0.0;
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
rnew_dot_rnew_local += ( *this )( ipatch )->EMfields->compute_r();
}
MPI_Allreduce( &rnew_dot_rnew_local, &rnew_dot_rnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
if( smpi->isMaster() ) {
DEBUG( "new residual norm: rnew_dot_rnew = " << rnew_dot_rnew );
}

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->update_p( rnew_dot_rnew, r_dot_r );
}


ctrl = sqrt( rnew_dot_rnew )/norm2_source_term;
if( smpi->isMaster() ) {
DEBUG( "iteration " << iteration << " done, exiting with control parameter ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}

}


if( iteration_max>0 && iteration == iteration_max ) {
if( smpi->isMaster() )
WARNING( "Relativistic Poisson solver did not converge: reached maximum iteration number: " << iteration
<< ", relative err is ctrl = " << 1.0e22*ctrl << "x 1.e-22" );
} else {
if( smpi->isMaster() )
MESSAGE( 1, "Relativistic Poisson solver converged at iteration: " << iteration
<< ", relative err is ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}



for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->initE_relativistic_Poisson( ( *this )( ipatch ), gamma_mean );
} 

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Ex_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ex_rel_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Ey_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ey_rel_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Ez_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ez_rel_, *this );

vector<double> E_Add( Ex_rel_[0]->dims_.size(), 0. );
if( Ex_rel_[0]->dims_.size()==3 ) {
double Ex_avg_local( 0. ), Ex_avg( 0. ), Ey_avg_local( 0. ), Ey_avg( 0. ), Ez_avg_local( 0. ), Ez_avg( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
Ex_avg_local += ( *this )( ipatch )->EMfields->computeExrelSum();
Ey_avg_local += ( *this )( ipatch )->EMfields->computeEyrelSum();
Ez_avg_local += ( *this )( ipatch )->EMfields->computeEzrelSum();
}

MPI_Allreduce( &Ex_avg_local, &Ex_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
MPI_Allreduce( &Ey_avg_local, &Ey_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
MPI_Allreduce( &Ez_avg_local, &Ez_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

E_Add[0] = -Ex_avg/( ( params.n_space[0]+2 )*( params.n_space[1]+1 )*( params.n_space[2]+1 ) );
E_Add[1] = -Ey_avg/( ( params.n_space[0]+1 )*( params.n_space[1]+2 )*( params.n_space[2]+1 ) );;
E_Add[2] = -Ez_avg/( ( params.n_space[0]+1 )*( params.n_space[1]+1 )*( params.n_space[2]+2 ) );;
} else if( Ex_rel_[0]->dims_.size()==2 ) {
double Ex_XminYmax = 0.0;
double Ey_XminYmax = 0.0;
double Ex_XmaxYmin = 0.0;
double Ey_XmaxYmin = 0.0;

std::vector<int> xcall( 2, 0 );
xcall[0] = 0;
xcall[1] = params.number_of_patches[1]-1;
int patch_YmaxXmin = domain_decomposition_->getDomainId( xcall );
int rank_XminYmax = smpi->hrank( patch_YmaxXmin );
xcall[0] = params.number_of_patches[0]-1;
xcall[1] = 0;
int patch_YminXmax = domain_decomposition_->getDomainId( xcall );
int rank_XmaxYmin = smpi->hrank( patch_YminXmax );

if( smpi->getRank() == rank_XminYmax ) {
Ex_XminYmax = ( *this )( patch_YmaxXmin-( this->refHindex_ ) )->EMfields->getExrel_XminYmax();
Ey_XminYmax = ( *this )( patch_YmaxXmin-( this->refHindex_ ) )->EMfields->getEyrel_XminYmax();
}

if( smpi->getRank() == rank_XmaxYmin ) {
Ex_XmaxYmin = ( *this )( patch_YminXmax-( this->refHindex_ ) )->EMfields->getExrel_XmaxYmin();
Ey_XmaxYmin = ( *this )( patch_YminXmax-( this->refHindex_ ) )->EMfields->getEyrel_XmaxYmin();
}

MPI_Bcast( &Ex_XminYmax, 1, MPI_DOUBLE, rank_XminYmax, MPI_COMM_WORLD );
MPI_Bcast( &Ey_XminYmax, 1, MPI_DOUBLE, rank_XminYmax, MPI_COMM_WORLD );

MPI_Bcast( &Ex_XmaxYmin, 1, MPI_DOUBLE, rank_XmaxYmin, MPI_COMM_WORLD );
MPI_Bcast( &Ey_XmaxYmin, 1, MPI_DOUBLE, rank_XmaxYmin, MPI_COMM_WORLD );

E_Add[0] = -0.5*( Ex_XminYmax+Ex_XmaxYmin );
E_Add[1] = -0.5*( Ey_XminYmax+Ey_XmaxYmin );

#ifdef _3D_LIKE_CENTERING
double Ex_avg_local( 0. ), Ex_avg( 0. ), Ey_avg_local( 0. ), Ey_avg( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
Ex_avg_local += ( *this )( ipatch )->EMfields->computeExrelSum();
Ey_avg_local += ( *this )( ipatch )->EMfields->computeEyrelSum();
}

MPI_Allreduce( &Ex_avg_local, &Ex_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
MPI_Allreduce( &Ey_avg_local, &Ey_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

E_Add[0] = -Ex_avg/( ( params.n_space[0]+2 )*( params.n_space[1]+1 ) );
E_Add[1] = -Ey_avg/( ( params.n_space[0]+1 )*( params.n_space[1]+2 ) );;
#endif

}

else if( Ex_rel_[0]->dims_.size()==1 ) {
double Ex_Xmin = 0.0;
double Ex_Xmax = 0.0;

unsigned int rankXmin = 0;
if( smpi->getRank() == 0 ) {
Ex_Xmin = ( *this )( ( 0 )-( this->refHindex_ ) )->EMfields->getExrel_Xmin();
}
MPI_Bcast( &Ex_Xmin, 1, MPI_DOUBLE, rankXmin, MPI_COMM_WORLD );

unsigned int rankXmax = smpi->getSize()-1;
if( smpi->getRank() == smpi->getSize()-1 ) {
Ex_Xmax = ( *this )( ( params.number_of_patches[0]-1 )-( this->refHindex_ ) )->EMfields->getExrel_Xmax();
}
MPI_Bcast( &Ex_Xmax, 1, MPI_DOUBLE, rankXmax, MPI_COMM_WORLD );
E_Add[0] = -0.5*( Ex_Xmin+Ex_Xmax );

#ifdef _3D_LIKE_CENTERING
double Ex_avg_local( 0. ), Ex_avg( 0. );
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
Ex_avg_local += ( *this )( ipatch )->EMfields->computeExrelSum();
}

MPI_Allreduce( &Ex_avg_local, &Ex_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

E_Add[0] = -Ex_avg/( ( params.n_space[0]+2 ) );
#endif

}

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->centeringErel( E_Add );
}

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->initB_relativistic_Poisson( ( *this )( ipatch ), gamma_mean );
} 

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bx_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bx_rel_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( By_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( By_rel_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bz_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bz_rel_, *this );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->center_fields_from_relativistic_Poisson( ( *this )( ipatch ) );
} 

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bx_rel_t_plus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bx_rel_t_plus_halfdt_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( By_rel_t_plus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( By_rel_t_plus_halfdt_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bz_rel_t_plus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bz_rel_t_plus_halfdt_, *this );

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bx_rel_t_minus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bx_rel_t_minus_halfdt_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( By_rel_t_minus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( By_rel_t_minus_halfdt_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bz_rel_t_minus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bz_rel_t_minus_halfdt_, *this );



MESSAGE( 0, "Summing fields of relativistic species to the grid fields" );

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->sum_rel_fields_to_em_fields( ( *this )( ipatch ) );
} 

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Ex_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ex_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Ey_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ey_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Ez_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ez_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bx_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bx_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( By_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( By_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bz_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bz_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bx_m, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bx_m, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( By_m, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( By_m, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( Bz_m, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bz_m, *this );

MESSAGE( 0, "Fields of relativistic species initialized" );



MESSAGE( "Relativistic Poisson finished" );

} 



void VectorPatch::solveRelativisticPoissonAM( Params &params, SmileiMPI *smpi, double time_primal )
{




double s_gamma( 0. );
uint64_t nparticles( 0 );
for( unsigned int ispec=0 ; ispec<( *this )( 0 )->vecSpecies.size() ; ispec++ ) {
if( species( 0, ispec )->relativistic_field_initialization_ ) {
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
if( (int)(time_primal/params.timestep)==species( ipatch, ispec )->iter_relativistic_initialization_ ) {
s_gamma += species( ipatch, ispec )->sumGamma();
nparticles += species( ipatch, ispec )->getNbrOfParticles();
}
}
}
}
double gamma_global( 0. );
MPI_Allreduce( &s_gamma, &gamma_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
uint64_t nparticles_global( 0 );
MPI_Allreduce( &nparticles, &nparticles_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
MESSAGE( "GAMMA = " << gamma_global/( double )nparticles_global );


double gamma_mean = gamma_global/( double )nparticles_global;

unsigned int iteration_max = params.relativistic_poisson_max_iteration;
double           error_max = params.relativistic_poisson_max_error;
unsigned int iteration=0;

double rnew_dot_rnew_localAM_( 0. );
double rnew_dot_rnewAM_( 0. );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->initPoisson( ( *this )( ipatch ) );
( *this )( ipatch )->EMfields->initRelativisticPoissonFields( ( *this )( ipatch ) );
}

std::vector<Field *> El_;
std::vector<Field *> Er_;
std::vector<Field *> Et_;
std::vector<Field *> Bl_;
std::vector<Field *> Br_;
std::vector<Field *> Bt_;
std::vector<Field *> Bl_m;
std::vector<Field *> Br_m;
std::vector<Field *> Bt_m;

std::vector<Field *> El_rel_;
std::vector<Field *> Er_rel_;
std::vector<Field *> Et_rel_;
std::vector<Field *> Bl_rel_;
std::vector<Field *> Br_rel_;
std::vector<Field *> Bt_rel_;

std::vector<Field *> Bl_rel_t_plus_halfdt_;
std::vector<Field *> Br_rel_t_plus_halfdt_;
std::vector<Field *> Bt_rel_t_plus_halfdt_;
std::vector<Field *> Bl_rel_t_minus_halfdt_;
std::vector<Field *> Br_rel_t_minus_halfdt_;
std::vector<Field *> Bt_rel_t_minus_halfdt_;

std::vector<Field *> Ap_AM_;

for( unsigned int imode=0 ; imode<params.nmodes_rel_field_init ; imode++ ) {

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->initPoisson_init_phi_r_p_Ap( ( *this )( ipatch ), imode );
rnew_dot_rnew_localAM_ += emAM->compute_r();
}

MPI_Allreduce( &rnew_dot_rnew_localAM_, &rnew_dot_rnewAM_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
El_.push_back( emAM->El_[imode] );
Er_.push_back( emAM->Er_[imode] );
Et_.push_back( emAM->Et_[imode] );
Bl_.push_back( emAM->Bl_[imode] );
Br_.push_back( emAM->Br_[imode] );
Bt_.push_back( emAM->Bt_[imode] );
Bl_m.push_back( emAM->Bl_m[imode] );
Br_m.push_back( emAM->Br_m[imode] );
Bt_m.push_back( emAM->Bt_m[imode] );
El_rel_.push_back( emAM->El_rel_ );
Er_rel_.push_back( emAM->Er_rel_ );
Et_rel_.push_back( emAM->Et_rel_ );
Bl_rel_.push_back( emAM->Bl_rel_ );
Br_rel_.push_back( emAM->Br_rel_ );
Bt_rel_.push_back( emAM->Bt_rel_ );
Bl_rel_t_plus_halfdt_.push_back( emAM->Bl_rel_t_plus_halfdt_ );
Br_rel_t_plus_halfdt_.push_back( emAM->Br_rel_t_plus_halfdt_ );
Bt_rel_t_plus_halfdt_.push_back( emAM->Bt_rel_t_plus_halfdt_);
Bl_rel_t_minus_halfdt_.push_back( emAM->Bl_rel_t_minus_halfdt_ );
Br_rel_t_minus_halfdt_.push_back( emAM->Br_rel_t_minus_halfdt_ );
Bt_rel_t_minus_halfdt_.push_back( emAM->Bt_rel_t_minus_halfdt_ );

Ap_AM_.push_back( emAM->Ap_AM_ );
}

unsigned int nx_p2_global = ( params.n_space_global[0]+1 );
if( El_rel_[0]->dims_.size()>1 ) {
nx_p2_global *= ( params.n_space_global[1]+1 );
if( El_rel_[0]->dims_.size()>2 ) {
nx_p2_global *= ( params.n_space_global[2]+1 );
}
}

double norm2_source_term = sqrt( std::abs(rnew_dot_rnewAM_) );
double ctrl = sqrt( std::abs(rnew_dot_rnewAM_) ) / norm2_source_term; 

if( smpi->isMaster() ) {
DEBUG( "Starting iterative loop for CG method for the mode "<<imode );
}

iteration = 0;
while( ( ctrl > error_max ) && ( iteration<iteration_max ) ) {
iteration++;

if( ( smpi->isMaster() ) && ( iteration%1000==0 ) ) {
MESSAGE( "iteration " << iteration << " started with control parameter ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}

double r_dot_rAM_ = rnew_dot_rnewAM_;

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->compute_Ap_relativistic_Poisson_AM( ( *this )( ipatch ), gamma_mean, imode );
}

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Ap_AM_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Ap_AM_, *this );


std::complex<double> p_dot_ApAM_       = 0.0;
std::complex<double> p_dot_Ap_localAM_ = 0.0;
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
p_dot_Ap_localAM_ += emAM->compute_pAp_AM();
}
MPI_Allreduce( &p_dot_Ap_localAM_, &p_dot_ApAM_, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->update_pand_r_AM( r_dot_rAM_, p_dot_ApAM_ );
}

rnew_dot_rnewAM_       = 0.0;
rnew_dot_rnew_localAM_ = 0.0;
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
rnew_dot_rnew_localAM_ += emAM->compute_r();
}
MPI_Allreduce( &rnew_dot_rnew_localAM_, &rnew_dot_rnewAM_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
if( smpi->isMaster() ) {
DEBUG( "new residual norm: rnew_dot_rnew = " << rnew_dot_rnewAM_ );
}

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->update_p( rnew_dot_rnewAM_, r_dot_rAM_ );
}


ctrl = sqrt( std::abs(rnew_dot_rnewAM_) )/norm2_source_term;
if( smpi->isMaster() ) {
DEBUG( "iteration " << iteration << " done, exiting with control parameter ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}

}


if( iteration_max>0 && iteration == iteration_max ) {
if( smpi->isMaster() )
WARNING( "Relativistic Poisson solver did not converge: reached maximum iteration number: " << iteration
<< ", relative err is ctrl = " << 1.0e22*ctrl << "x 1.e-22" );
} else {
if( smpi->isMaster() )
MESSAGE( 1, "Relativistic Poisson solver converged at iteration: " << iteration
<< ", relative err is ctrl = " << 1.0e22*ctrl << " x 1.e-22" );
}



for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->initE_relativistic_Poisson_AM( ( *this )( ipatch ), gamma_mean, imode );
} 

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( El_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( El_rel_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Er_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Er_rel_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Et_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Et_rel_, *this );

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->initB_relativistic_Poisson_AM( ( *this )( ipatch ), gamma_mean );
} 

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Bl_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bl_rel_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Br_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Br_rel_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Bt_rel_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bt_rel_, *this );


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->center_fields_from_relativistic_Poisson_AM( ( *this )( ipatch ) );
} 

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Bl_rel_t_plus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bl_rel_t_plus_halfdt_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Br_rel_t_plus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Br_rel_t_plus_halfdt_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Bt_rel_t_plus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bt_rel_t_plus_halfdt_, *this );

SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Bl_rel_t_minus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bl_rel_t_minus_halfdt_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Br_rel_t_minus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Br_rel_t_minus_halfdt_, *this );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( Bt_rel_t_minus_halfdt_, *this, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( Bt_rel_t_minus_halfdt_, *this );


MESSAGE( 0, "Summing fields of relativistic species to the grid fields" );

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->sum_rel_fields_to_em_fields_AM( ( *this )( ipatch ), params, imode );
} 


for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {

El_.pop_back();
Er_.pop_back();
Et_.pop_back();
Bl_.pop_back();
Br_.pop_back();
Bt_.pop_back();
Bl_m.pop_back();
Br_m.pop_back();
Bt_m.pop_back();
El_rel_.pop_back();
Er_rel_.pop_back();
Et_rel_.pop_back();
Bl_rel_.pop_back();
Br_rel_.pop_back();
Bt_rel_.pop_back();
Bl_rel_t_plus_halfdt_.pop_back();
Br_rel_t_plus_halfdt_.pop_back();
Bt_rel_t_plus_halfdt_.pop_back();
Bl_rel_t_minus_halfdt_.pop_back();
Br_rel_t_minus_halfdt_.pop_back();
Bt_rel_t_minus_halfdt_.pop_back();

Ap_AM_.pop_back();

}

}  

for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields );
emAM->delete_phi_r_p_Ap( ( *this )( ipatch ) );
emAM->delete_relativistic_fields( ( *this )( ipatch ) );
}

for( unsigned int imode = 0 ; imode < params.nmodes_rel_field_init ; imode++ ) {
SyncVectorPatch::exchangeE( params, ( *this ), imode, smpi );
SyncVectorPatch::finalizeexchangeE( params, ( *this ), imode ); 
}
for( unsigned int imode = 0 ; imode < params.nmodes_rel_field_init ; imode++ ) {
SyncVectorPatch::exchangeB( params, ( *this ), imode, smpi );
SyncVectorPatch::finalizeexchangeB( params, ( *this ), imode ); 
}

MESSAGE( 0, "Fields of relativistic species initialized" );



MESSAGE( "Relativistic Poisson finished" );



}



void VectorPatch::loadBalance( Params &params, double time_dual, SmileiMPI *smpi, SimWindow *simWindow, unsigned int itime )
{

smpi->recompute_patch_count( params, *this, time_dual );

this->createPatches( params, smpi, simWindow );

this->exchangePatches( smpi, params );

lastIterationPatchesMoved = itime;

}


void VectorPatch::createPatches( Params &params, SmileiMPI *smpi, SimWindow *simWindow )
{
unsigned int n_moved( 0 );
recv_patches_.resize( 0 );

refHindex_ = ( *this )( 0 )->Hindex();

int nPatches_now = this->size() ;

recv_patch_id_.clear();
send_patch_id_.clear();

int istart( 0 );
for( int irk=0 ; irk<smpi->getRank() ; irk++ ) {
istart += smpi->patch_count[irk];
}

for( int ipatch=0 ; ipatch<smpi->patch_count[smpi->getRank()] ; ipatch++ ) {
recv_patch_id_.push_back( istart+ipatch );
}


for( int ipatch=0 ; ipatch < nPatches_now ; ipatch++ ) {
if( ( refHindex_+ipatch < recv_patch_id_[0] ) || ( refHindex_+ipatch > recv_patch_id_.back() ) ) {
send_patch_id_.push_back( ipatch );
}
}


int existing_patch_id = -1;
for( int ipatch=recv_patch_id_.size()-1 ; ipatch>=0 ; ipatch-- ) {
if( ( recv_patch_id_[ipatch]>=refHindex_ ) && ( recv_patch_id_[ipatch] <= refHindex_ + nPatches_now - 1 ) ) {
existing_patch_id = recv_patch_id_[ipatch];
recv_patch_id_.erase( recv_patch_id_.begin()+ipatch );
}
}


if( existing_patch_id<0 ) {
ERROR( "No patch to clone. This should never happen!" );
}
Patch *existing_patch = ( *this )( existing_patch_id-refHindex_ );


n_moved = simWindow->getNmoved();
for( unsigned int ipatch=0 ; ipatch < recv_patch_id_.size() ; ipatch++ ) {
Patch *newPatch = PatchesFactory::clone( existing_patch, params, smpi, domain_decomposition_, recv_patch_id_[ipatch], n_moved, false );
newPatch->finalizeMPIenvironment( params );
recv_patches_.push_back( newPatch );
}


} 


void VectorPatch::exchangePatches( SmileiMPI *smpi, Params &params )
{

int newMPIrank = smpi->getRank() -1;
int oldMPIrank = smpi->getRank() -1;
int istart = 0;

for( int irk=0 ; irk<smpi->getRank() ; irk++ ) {
istart += smpi->patch_count[irk];
}
int tagsend_right = 0;
int tagsend_left = 0;
int tagrecv_left = 0;
int tagrecv_right = 0;
int tag=0;


for( unsigned int ipatch=0 ; ipatch < send_patch_id_.size() ; ipatch++ ) {
if( send_patch_id_[ipatch]+refHindex_ > istart ) {
newMPIrank = smpi->getRank() + 1;
tag = tagsend_right*nrequests;
tagsend_right ++;
} else {
tag = tagsend_left*nrequests;
tagsend_left ++;
}
int irequest = 0;
smpi->isend_species( ( *this )( send_patch_id_[ipatch] ), newMPIrank, irequest, tag, params );
}

for( unsigned int ipatch=0 ; ipatch < recv_patch_id_.size() ; ipatch++ ) {
if( recv_patch_id_[ipatch] > refHindex_ ) {
oldMPIrank = smpi->getRank() + 1;
tag = tagrecv_right*nrequests;
tagrecv_right ++;
} else {
tag = tagrecv_left*nrequests;
tagrecv_left ++;
}
smpi->recv_species( recv_patches_[ipatch], oldMPIrank, tag, params );
}


for( unsigned int ipatch=0 ; ipatch < send_patch_id_.size() ; ipatch++ ) {
smpi->waitall( ( *this )( send_patch_id_[ipatch] ) );
}

smpi->barrier();


newMPIrank = smpi->getRank() -1;
oldMPIrank = smpi->getRank() -1;


for( unsigned int ipatch=0 ; ipatch < send_patch_id_.size() ; ipatch++ ) {
if( send_patch_id_[ipatch]+refHindex_ > istart ) {
newMPIrank = smpi->getRank() + 1;
tag = tagsend_right;
tagsend_right ++;
} else {
tag = tagsend_left;
tagsend_left ++;
}
int irequest = 0;
smpi->isend_fields( ( *this )( send_patch_id_[ipatch] ), newMPIrank, irequest, tag*nrequests, params );
}

for( unsigned int ipatch=0 ; ipatch < recv_patch_id_.size() ; ipatch++ ) {
if( recv_patch_id_[ipatch] > refHindex_ ) {
oldMPIrank = smpi->getRank() + 1;
tag = tagrecv_right;
tagrecv_right ++;
} else {
tag = tagrecv_left;
tagrecv_left ++;
}
int patch_tag = tag * nrequests;
smpi->recv_fields( recv_patches_[ipatch], oldMPIrank, patch_tag, params );
}


for( unsigned int ipatch=0 ; ipatch < send_patch_id_.size() ; ipatch++ ) {
smpi->waitall( ( *this )( send_patch_id_[ipatch] ) );
}

smpi->barrier();


int nPatchSend( send_patch_id_.size() );
for( int ipatch=nPatchSend-1 ; ipatch>=0 ; ipatch-- ) {
delete( *this )( send_patch_id_[ipatch] );
patches_[ send_patch_id_[ipatch] ] = NULL;
patches_.erase( patches_.begin() + send_patch_id_[ipatch] );

}

if( params.vectorization_mode == "on" ) {
for( unsigned int ipatch=0 ; ipatch<recv_patch_id_.size() ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec< recv_patches_[ipatch]->vecSpecies.size() ; ispec++ ) {
dynamic_cast<SpeciesV *>( recv_patches_[ipatch]->vecSpecies[ispec] )->computeParticleCellKeys( params );
dynamic_cast<SpeciesV *>( recv_patches_[ipatch]->vecSpecies[ispec] )->sortParticles( params, recv_patches_[ipatch] );
}
}
} else if( params.vectorization_mode == "adaptive_mixed_sort" ) {
for( unsigned int ipatch=0 ; ipatch<recv_patch_id_.size() ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec< recv_patches_[ipatch]->vecSpecies.size() ; ispec++ ) {
if( dynamic_cast<SpeciesVAdaptiveMixedSort *>( recv_patches_[ipatch]->vecSpecies[ispec] ) ) {
dynamic_cast<SpeciesVAdaptiveMixedSort *>( recv_patches_[ipatch]->vecSpecies[ispec] )->computeParticleCellKeys( params );
dynamic_cast<SpeciesVAdaptiveMixedSort *>( recv_patches_[ipatch]->vecSpecies[ispec] )->reconfigure_operators( params, recv_patches_[ipatch] );
}
}
}
} else if( params.vectorization_mode == "adaptive" ) {
for( unsigned int ipatch=0 ; ipatch<recv_patch_id_.size() ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec< recv_patches_[ipatch]->vecSpecies.size() ; ispec++ ) {
if( dynamic_cast<SpeciesVAdaptive *>( recv_patches_[ipatch]->vecSpecies[ispec] ) ) {
dynamic_cast<SpeciesVAdaptive *>( recv_patches_[ipatch]->vecSpecies[ispec] )->computeParticleCellKeys( params );
dynamic_cast<SpeciesVAdaptive *>( recv_patches_[ipatch]->vecSpecies[ispec] )->reconfigure_operators( params, recv_patches_[ipatch] );
}
}
}
}

for( unsigned int ipatch=0 ; ipatch<recv_patch_id_.size() ; ipatch++ ) {
if( recv_patch_id_[ipatch] > refHindex_ ) {
patches_.push_back( recv_patches_[ipatch] );
} else {
patches_.insert( patches_.begin()+ipatch, recv_patches_[ipatch] );
}
}
recv_patches_.clear();

for( unsigned int ipatch=0 ; ipatch<patches_.size() ; ipatch++ ) {
( *this )( ipatch )->updateMPIenv( smpi );
}
this->setRefHindex() ;
updateFieldList( smpi ) ;

} 

void VectorPatch::outputExchanges( SmileiMPI *smpi )
{
ofstream output_file;
ostringstream name( "" );
name << "debug_output"<<smpi->getRank()<<".txt" ;
output_file.open( name.str().c_str(), std::ofstream::out | std::ofstream::app );
int newMPIrank, oldMPIrank;
newMPIrank = smpi->getRank() -1;
oldMPIrank = smpi->getRank() -1;
int istart( 0 );
for( int irk=0 ; irk<smpi->getRank() ; irk++ ) {
istart += smpi->patch_count[irk];
}
for( unsigned int ipatch=0 ; ipatch < send_patch_id_.size() ; ipatch++ ) {
if( send_patch_id_[ipatch]+refHindex_ > istart ) {
newMPIrank = smpi->getRank() + 1;
}
output_file << "Rank " << smpi->getRank() << " sending patch " << send_patch_id_[ipatch]+refHindex_ << " to " << newMPIrank << endl;
}
for( unsigned int ipatch=0 ; ipatch < recv_patch_id_.size() ; ipatch++ ) {
if( recv_patch_id_[ipatch] > refHindex_ ) {
oldMPIrank = smpi->getRank() + 1;
}
output_file << "Rank " << smpi->getRank() << " receiving patch " << recv_patch_id_[ipatch] << " from " << oldMPIrank << endl;
}
output_file << "NEXT" << endl;
output_file.close();
} 

void VectorPatch::updateFieldList( SmileiMPI *smpi )
{
int nDim( 0 );
if( !dynamic_cast<ElectroMagnAM *>( patches_[0]->EMfields ) ) {
nDim = patches_[0]->EMfields->Ex_->dims_.size();
} else {
nDim = static_cast<ElectroMagnAM *>( patches_[0]->EMfields )->El_[0]->dims_.size();
}
densities.resize( 3*size() ) ; 

Bs0.resize( 2*size() ) ; 
Bs1.resize( 2*size() ) ; 
Bs2.resize( 2*size() ) ; 

densitiesLocalx.clear();
densitiesLocaly.clear();
densitiesLocalz.clear();
densitiesMPIx.clear();
densitiesMPIy.clear();
densitiesMPIz.clear();
LocalxIdx.clear();
LocalyIdx.clear();
LocalzIdx.clear();
MPIxIdx.clear();
MPIyIdx.clear();
MPIzIdx.clear();

if( !dynamic_cast<ElectroMagnAM *>( patches_[0]->EMfields ) ) {

listJx_.resize( size() ) ;
listJy_.resize( size() ) ;
listJz_.resize( size() ) ;
listrho_.resize( size() ) ;
listEx_.resize( size() ) ;
listEy_.resize( size() ) ;
listEz_.resize( size() ) ;
listBx_.resize( size() ) ;
listBy_.resize( size() ) ;
listBz_.resize( size() ) ;

if( patches_[0]->EMfields->envelope != NULL ) {
listA_.resize( size() ) ;
listA0_.resize( size() ) ;
listEnvEx_.resize( size() ) ;
listGradPhix_.resize( size() ) ;
listGradPhiy_.resize( size() ) ;
listGradPhiz_.resize( size() ) ;
listGradPhix0_.resize( size() ) ;
listGradPhiy0_.resize( size() ) ;
listGradPhiz0_.resize( size() ) ;
listEnv_Chi_.resize( size() ) ;
}

for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listJx_[ipatch] = patches_[ipatch]->EMfields->Jx_ ;
listJy_[ipatch] = patches_[ipatch]->EMfields->Jy_ ;
listJz_[ipatch] = patches_[ipatch]->EMfields->Jz_ ;
listrho_[ipatch] =patches_[ipatch]->EMfields->rho_;
listEx_[ipatch] = patches_[ipatch]->EMfields->Ex_ ;
listEy_[ipatch] = patches_[ipatch]->EMfields->Ey_ ;
listEz_[ipatch] = patches_[ipatch]->EMfields->Ez_ ;
listBx_[ipatch] = patches_[ipatch]->EMfields->Bx_ ;
listBy_[ipatch] = patches_[ipatch]->EMfields->By_ ;
listBz_[ipatch] = patches_[ipatch]->EMfields->Bz_ ;
}
if( patches_[0]->EMfields->envelope != NULL ) {
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listA_[ipatch]         = patches_[ipatch]->EMfields->envelope->A_ ;
listA0_[ipatch]        = patches_[ipatch]->EMfields->envelope->A0_ ;
listEnvEx_[ipatch]      = patches_[ipatch]->EMfields->Env_Ex_abs_ ;
listGradPhix_[ipatch]  = patches_[ipatch]->EMfields->envelope->GradPhix_ ;
listGradPhiy_[ipatch]  = patches_[ipatch]->EMfields->envelope->GradPhiy_ ;
listGradPhiz_[ipatch]  = patches_[ipatch]->EMfields->envelope->GradPhiz_ ;
listGradPhix0_[ipatch] = patches_[ipatch]->EMfields->envelope->GradPhix_m ;
listGradPhiy0_[ipatch] = patches_[ipatch]->EMfields->envelope->GradPhiy_m ;
listGradPhiz0_[ipatch] = patches_[ipatch]->EMfields->envelope->GradPhiz_m ;
listEnv_Chi_[ipatch]   = patches_[ipatch]->EMfields->Env_Chi_ ;
}
}

} else {
unsigned int nmodes = static_cast<ElectroMagnAM *>( patches_[0]->EMfields )->El_.size();
listJl_.resize( nmodes ) ;
listJr_.resize( nmodes ) ;
listJt_.resize( nmodes ) ;
listrho_AM_.resize( nmodes ) ;
if (static_cast<ElectroMagnAM *>( patches_[0]->EMfields )->rho_old_AM_[0])
listrho_old_AM_.resize( nmodes ) ;
listrho_AM_.resize( nmodes ) ;
listJls_.resize( nmodes ) ;
listJrs_.resize( nmodes ) ;
listJts_.resize( nmodes ) ;
listrhos_AM_.resize( nmodes ) ;
listEl_.resize( nmodes ) ;
listEr_.resize( nmodes ) ;
listEt_.resize( nmodes ) ;
listBl_.resize( nmodes ) ;
listBr_.resize( nmodes ) ;
listBt_.resize( nmodes ) ;

for( unsigned int imode=0 ; imode < nmodes ; imode++ ) {
listJl_[imode].resize( size() );
listJr_[imode].resize( size() );
listJt_[imode].resize( size() );
listrho_AM_[imode].resize( size() );
if (static_cast<ElectroMagnAM *>( patches_[0]->EMfields )->rho_old_AM_[imode])
listrho_old_AM_[imode].resize( size() );
listEl_[imode].resize( size() );
listEr_[imode].resize( size() );
listEt_[imode].resize( size() );
listBl_[imode].resize( size() );
listBr_[imode].resize( size() );
listBt_[imode].resize( size() );
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listJl_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->Jl_[imode] ;
listJr_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->Jr_[imode] ;
listJt_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->Jt_[imode] ;
listrho_AM_[imode][ipatch] =static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->rho_AM_[imode];
if (static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->rho_old_AM_[imode])
listrho_old_AM_[imode][ipatch] =static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->rho_old_AM_[imode];
listEl_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->El_[imode] ;
listEr_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->Er_[imode] ;
listEt_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->Et_[imode] ;
listBl_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->Bl_[imode] ;
listBr_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->Br_[imode] ;
listBt_[imode][ipatch]     = static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->Bt_[imode] ;
}
}

if( patches_[0]->EMfields->envelope != NULL ) {
listA_.resize( size() ) ;
listA0_.resize( size() ) ;
listEnvEx_.resize( size() ) ;
listGradPhil_.resize( size() ) ;
listGradPhir_.resize( size() ) ;
listGradPhil0_.resize( size() ) ;
listGradPhir0_.resize( size() ) ;
listEnv_Chi_.resize( size() ) ;
}


if( patches_[0]->EMfields->envelope != NULL ) {
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listA_[ipatch]         = patches_[ipatch]->EMfields->envelope->A_ ;
listA0_[ipatch]        = patches_[ipatch]->EMfields->envelope->A0_ ;
listEnvEx_[ipatch]      = patches_[ipatch]->EMfields->Env_Ex_abs_ ;
listGradPhil_[ipatch]  = patches_[ipatch]->EMfields->envelope->GradPhil_ ;
listGradPhir_[ipatch]  = patches_[ipatch]->EMfields->envelope->GradPhir_ ;
listGradPhil0_[ipatch] = patches_[ipatch]->EMfields->envelope->GradPhil_m ;
listGradPhir0_[ipatch] = patches_[ipatch]->EMfields->envelope->GradPhir_m ;
listEnv_Chi_[ipatch]   = patches_[ipatch]->EMfields->Env_Chi_ ;
}
}

}

B_localx.clear();
B_MPIx.clear();

B1_localy.clear();
B1_MPIy.clear();

B2_localz.clear();
B2_MPIz.clear();

for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
densities[ipatch         ] = patches_[ipatch]->EMfields->Jx_ ;
densities[ipatch+  size()] = patches_[ipatch]->EMfields->Jy_ ;
densities[ipatch+2*size()] = patches_[ipatch]->EMfields->Jz_ ;

Bs0[ipatch       ] = patches_[ipatch]->EMfields->By_ ;
Bs0[ipatch+size()] = patches_[ipatch]->EMfields->Bz_ ;

Bs1[ipatch       ] = patches_[ipatch]->EMfields->Bx_ ;
Bs1[ipatch+size()] = patches_[ipatch]->EMfields->Bz_ ;

Bs2[ipatch       ] = patches_[ipatch]->EMfields->Bx_ ;
Bs2[ipatch+size()] = patches_[ipatch]->EMfields->By_ ;
}

for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
if( ( *this )( ipatch )->has_an_MPI_neighbor( 0 ) ) {
MPIxIdx.push_back( ipatch );
}
if( ( *this )( ipatch )->has_an_local_neighbor( 0 ) ) {
LocalxIdx.push_back( ipatch );
}
}
if( nDim>1 ) {
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
if( ( *this )( ipatch )->has_an_MPI_neighbor( 1 ) ) {
MPIyIdx.push_back( ipatch );
}
if( ( *this )( ipatch )->has_an_local_neighbor( 1 ) ) {
LocalyIdx.push_back( ipatch );
}
}
if( nDim>2 ) {
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {

if( ( *this )( ipatch )->has_an_MPI_neighbor( 2 ) ) {
MPIzIdx.push_back( ipatch );
}
if( ( *this )( ipatch )->has_an_local_neighbor( 2 ) ) {
LocalzIdx.push_back( ipatch );
}
}
}
}

B_MPIx.resize( 2*MPIxIdx.size() );
B_localx.resize( 2*LocalxIdx.size() );
B1_MPIy.resize( 2*MPIyIdx.size() );
B1_localy.resize( 2*LocalyIdx.size() );
B2_MPIz.resize( 2*MPIzIdx.size() );
B2_localz.resize( 2*LocalzIdx.size() );

densitiesMPIx.resize( 3*MPIxIdx.size() );
densitiesLocalx.resize( 3*LocalxIdx.size() );
densitiesMPIy.resize( 3*MPIyIdx.size() );
densitiesLocaly.resize( 3*LocalyIdx.size() );
densitiesMPIz.resize( 3*MPIzIdx.size() );
densitiesLocalz.resize( 3*LocalzIdx.size() );

int mpix( 0 ), locx( 0 ), mpiy( 0 ), locy( 0 ), mpiz( 0 ), locz( 0 );

for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {

if( ( *this )( ipatch )->has_an_MPI_neighbor( 0 ) ) {
B_MPIx[mpix               ] = patches_[ipatch]->EMfields->By_;
B_MPIx[mpix+MPIxIdx.size()] = patches_[ipatch]->EMfields->Bz_;

densitiesMPIx[mpix                 ] = patches_[ipatch]->EMfields->Jx_;
densitiesMPIx[mpix+  MPIxIdx.size()] = patches_[ipatch]->EMfields->Jy_;
densitiesMPIx[mpix+2*MPIxIdx.size()] = patches_[ipatch]->EMfields->Jz_;
mpix++;
}
if( ( *this )( ipatch )->has_an_local_neighbor( 0 ) ) {
B_localx[locx                 ] = patches_[ipatch]->EMfields->By_;
B_localx[locx+LocalxIdx.size()] = patches_[ipatch]->EMfields->Bz_;

densitiesLocalx[locx                   ] = patches_[ipatch]->EMfields->Jx_;
densitiesLocalx[locx+  LocalxIdx.size()] = patches_[ipatch]->EMfields->Jy_;
densitiesLocalx[locx+2*LocalxIdx.size()] = patches_[ipatch]->EMfields->Jz_;
locx++;
}
}
if( nDim>1 ) {
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
if( ( *this )( ipatch )->has_an_MPI_neighbor( 1 ) ) {
B1_MPIy[mpiy               ] = patches_[ipatch]->EMfields->Bx_;
B1_MPIy[mpiy+MPIyIdx.size()] = patches_[ipatch]->EMfields->Bz_;

densitiesMPIy[mpiy                 ] = patches_[ipatch]->EMfields->Jx_;
densitiesMPIy[mpiy+  MPIyIdx.size()] = patches_[ipatch]->EMfields->Jy_;
densitiesMPIy[mpiy+2*MPIyIdx.size()] = patches_[ipatch]->EMfields->Jz_;
mpiy++;
}
if( ( *this )( ipatch )->has_an_local_neighbor( 1 ) ) {
B1_localy[locy                 ] = patches_[ipatch]->EMfields->Bx_;
B1_localy[locy+LocalyIdx.size()] = patches_[ipatch]->EMfields->Bz_;

densitiesLocaly[locy                   ] = patches_[ipatch]->EMfields->Jx_;
densitiesLocaly[locy+  LocalyIdx.size()] = patches_[ipatch]->EMfields->Jy_;
densitiesLocaly[locy+2*LocalyIdx.size()] = patches_[ipatch]->EMfields->Jz_;
locy++;
}
}
if( nDim>2 ) {
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
if( ( *this )( ipatch )->has_an_MPI_neighbor( 2 ) ) {
B2_MPIz[mpiz               ] = patches_[ipatch]->EMfields->Bx_;
B2_MPIz[mpiz+MPIzIdx.size()] = patches_[ipatch]->EMfields->By_;

densitiesMPIz[mpiz                 ] = patches_[ipatch]->EMfields->Jx_;
densitiesMPIz[mpiz+  MPIzIdx.size()] = patches_[ipatch]->EMfields->Jy_;
densitiesMPIz[mpiz+2*MPIzIdx.size()] = patches_[ipatch]->EMfields->Jz_;
mpiz++;
}
if( ( *this )( ipatch )->has_an_local_neighbor( 2 ) ) {
B2_localz[locz                 ] = patches_[ipatch]->EMfields->Bx_;
B2_localz[locz+LocalzIdx.size()] = patches_[ipatch]->EMfields->By_;

densitiesLocalz[locz                   ] = patches_[ipatch]->EMfields->Jx_;
densitiesLocalz[locz+  LocalzIdx.size()] = patches_[ipatch]->EMfields->Jy_;
densitiesLocalz[locz+2*LocalzIdx.size()] = patches_[ipatch]->EMfields->Jz_;
locz++;
}
}
}

}

if( !dynamic_cast<ElectroMagnAM *>( patches_[0]->EMfields ) ) {
for( unsigned int ipatch = 0 ; ipatch < size() ; ipatch++ ) {
listJx_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 1 );
listJy_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 2 );
listJz_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 3 );
listBx_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 6 );
listBy_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 7 );
listBz_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 8 );
listrho_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 4 );
}
if( patches_[0]->EMfields->envelope != NULL ) {
for( unsigned int ipatch = 0 ; ipatch < size() ; ipatch++ ) {
listA_ [ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listA0_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listEnvEx_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhix_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhiy_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhiz_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhix0_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhiy0_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhiz0_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listEnv_Chi_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
}

}
} else {
unsigned int nmodes = static_cast<ElectroMagnAM *>( patches_[0]->EMfields )->El_.size();
for( unsigned int imode=0 ; imode < nmodes ; imode++ ) {
for( unsigned int ipatch = 0 ; ipatch < size() ; ipatch++ ) {
listJl_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listJr_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listJt_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listBl_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listBr_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listBt_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listEl_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listEr_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listEt_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
listrho_AM_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
if (static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields )->rho_old_AM_[imode])
listrho_old_AM_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
}
if( patches_[0]->EMfields->envelope != NULL ) {
for( unsigned int ipatch = 0 ; ipatch < size() ; ipatch++ ) {
listA_ [ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listA0_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listEnvEx_ [ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhil_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhir_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhil0_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listGradPhir0_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
listEnv_Chi_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 ) ;
}

}
}
}



void VectorPatch::updateFieldList( int ispec, SmileiMPI *smpi )
{
#pragma omp barrier
if( !dynamic_cast<ElectroMagnAM *>( patches_[0]->EMfields ) ) {
#pragma omp single
{
if( patches_[0]->EMfields->Jx_s [ispec] )
{
listJxs_.resize( size() ) ;
} else
{
listJxs_.clear();
}
if( patches_[0]->EMfields->Jy_s [ispec] )
{
listJys_.resize( size() ) ;
} else
{
listJys_.clear();
}
if( patches_[0]->EMfields->Jz_s [ispec] )
{
listJzs_.resize( size() ) ;
} else
{
listJzs_.clear();
}
if( patches_[0]->EMfields->rho_s[ispec] )
{
listrhos_.resize( size() ) ;
} else
{
listrhos_.clear();
}

if( patches_[0]->EMfields->envelope != NULL )
{
if( patches_[0]->EMfields->Env_Chi_s[ispec] ) {
listEnv_Chis_.resize( size() ) ;
} else {
listEnv_Chis_.clear();
}
}
}

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
if( patches_[ipatch]->EMfields->Jx_s [ispec] ) {
listJxs_ [ipatch] = patches_[ipatch]->EMfields->Jx_s [ispec];
listJxs_ [ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
if( patches_[ipatch]->EMfields->Jy_s [ispec] ) {
listJys_ [ipatch] = patches_[ipatch]->EMfields->Jy_s [ispec];
listJys_ [ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
if( patches_[ipatch]->EMfields->Jz_s [ispec] ) {
listJzs_ [ipatch] = patches_[ipatch]->EMfields->Jz_s [ispec];
listJzs_ [ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
if( patches_[ipatch]->EMfields->rho_s[ispec] ) {
listrhos_[ipatch] = patches_[ipatch]->EMfields->rho_s[ispec];
listrhos_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}

if( patches_[0]->EMfields->envelope != NULL ) {
if( patches_[ipatch]->EMfields->Env_Chi_s[ispec] ) {
listEnv_Chis_[ipatch] = patches_[ipatch]->EMfields->Env_Chi_s[ispec];
listEnv_Chis_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
}
}
} else { 
ElectroMagnAM *emAM =  static_cast<ElectroMagnAM *>( patches_[0]->EMfields );
unsigned int nmodes = emAM->El_.size();
unsigned int n_species = emAM->n_species;
#pragma omp single
{
for( unsigned int imode=0 ; imode < nmodes ; imode++ ) {
unsigned int ifield = imode*n_species + ispec ;
if( emAM->Jl_s [ifield] ) {
listJls_[imode].resize( size() ) ;
} else {
listJls_[imode].clear();
}
if( emAM->Jr_s [ifield] ) {
listJrs_[imode].resize( size() ) ;
} else {
listJrs_[imode].clear();
}
if( emAM->Jt_s [ifield] ) {
listJts_[imode].resize( size() ) ;
} else {
listJts_[imode].clear();
}
if( emAM->rho_AM_s [ifield] ) {
listrhos_AM_[imode].resize( size() ) ;
} else {
listrhos_AM_[imode].clear();
}
}
if( patches_[0]->EMfields->envelope != NULL )
{
if( patches_[0]->EMfields->Env_Chi_s[ispec] ) {
listEnv_Chis_.resize( size() ) ;
} else {
listEnv_Chis_.clear();
}
}

}
for( unsigned int imode=0 ; imode < nmodes ; imode++ ) {
unsigned int ifield = imode*n_species + ispec ;
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
emAM =  static_cast<ElectroMagnAM *>( patches_[ipatch]->EMfields );
if( emAM->Jl_s [ifield] ) {
listJls_[imode][ipatch] = emAM->Jl_s [ifield];
listJls_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
if( emAM->Jr_s [ifield] ) {
listJrs_[imode][ipatch] = emAM->Jr_s [ifield];
listJrs_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
if( emAM->Jt_s [ifield] ) {
listJts_[imode][ipatch] = emAM->Jt_s [ifield];
listJts_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
if( emAM->rho_AM_s [ifield] ) {
listrhos_AM_[imode][ipatch] = emAM->rho_AM_s [ifield];
listrhos_AM_[imode][ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
}
}
for( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
if( patches_[0]->EMfields->envelope != NULL ) {
if( patches_[ipatch]->EMfields->Env_Chi_s[ispec] ) {
listEnv_Chis_[ipatch] = patches_[ipatch]->EMfields->Env_Chi_s[ispec];
listEnv_Chis_[ipatch]->MPIbuff.defineTags( patches_[ipatch], smpi, 0 );
}
}
}
}



}


void VectorPatch::buildPMLList( string fieldname, int idim, int min_or_max, SmileiMPI *smpi )
{
int id_bc = 2*idim + min_or_max;

listForPML_.clear();
if ( fieldname == "Bx" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( ( emfields(ipatch)->emBoundCond[id_bc] )->getBxPML() );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "By" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( ( emfields(ipatch)->emBoundCond[id_bc] )->getByPML() );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Bz" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( ( emfields(ipatch)->emBoundCond[id_bc] )->getBzPML() );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Hx" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( ( emfields(ipatch)->emBoundCond[id_bc] )->getHxPML() );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Hy" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( ( emfields(ipatch)->emBoundCond[id_bc] )->getHyPML() );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Hz" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( ( emfields(ipatch)->emBoundCond[id_bc] )->getHzPML() );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
}


void VectorPatch::buildPMLList( string fieldname, int idim, int min_or_max, SmileiMPI *smpi, int imode )
{
int id_bc = 2*idim + min_or_max;

listForPML_.clear();
if ( fieldname == "Bl" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( static_cast<ElectroMagnBCAM_PML*>( emfields(ipatch)->emBoundCond[id_bc] )->Bl_[imode] );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Br" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( static_cast<ElectroMagnBCAM_PML*>( emfields(ipatch)->emBoundCond[id_bc] )->Br_[imode] );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Bt" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( static_cast<ElectroMagnBCAM_PML*>( emfields(ipatch)->emBoundCond[id_bc] )->Bt_[imode] );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Hl" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( static_cast<ElectroMagnBCAM_PML*>( emfields(ipatch)->emBoundCond[id_bc] )->Hl_[imode] );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Hr" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( static_cast<ElectroMagnBCAM_PML*>( emfields(ipatch)->emBoundCond[id_bc] )->Hr_[imode] );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
else if ( fieldname == "Ht" ) {
for ( unsigned int ipatch=0 ; ipatch < size() ; ipatch++ ) {
listForPML_.push_back( static_cast<ElectroMagnBCAM_PML*>( emfields(ipatch)->emBoundCond[id_bc] )->Ht_[imode] );
if(listForPML_.back()){
listForPML_.back()->MPIbuff.defineTags(patches_[ipatch], smpi, 0);
}
}
}
}


void VectorPatch::applyAntennas( double time )
{
#ifdef  __DEBUG
if( nAntennas>0 ) {
#pragma omp single
TITLE( "Applying antennas at time t = " << time );
}
#endif

for( unsigned int iAntenna=0; iAntenna<nAntennas; iAntenna++ ) {

if( patches_[0]->EMfields->antennas[iAntenna].spacetime ) {

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
Antenna * A = &( patches_[ipatch]->EMfields->antennas[iAntenna] );
Field *field = patches_[ipatch]->EMfields->allFields[A->index];
patches_[ipatch]->EMfields->applyPrescribedField( field, A->space_time_profile, patches_[ipatch], time );
}

} else {

#pragma omp single
antenna_intensity_ = patches_[0]->EMfields->antennas[iAntenna].time_profile->valueAt( time );

#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
patches_[ipatch]->EMfields->applyAntenna( iAntenna, antenna_intensity_ );
}

}
}
}

void VectorPatch::applyBinaryProcesses( Params &params, int itime, Timers &timers )
{
timers.collisions.restart();

if( BinaryProcesses::debye_length_required_ ) {
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
BinaryProcesses::calculate_debye_length( params, patches_[ipatch] );
}
}

unsigned int nBPs = patches_[0]->vecBPs.size();

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
for( unsigned int iBPs=0 ; iBPs<nBPs; iBPs++ ) {
patches_[ipatch]->vecBPs[iBPs]->apply( params, patches_[ipatch], itime, localDiags );
}
}

#pragma omp single
for( unsigned int iBPs=0 ; iBPs<nBPs; iBPs++ ) {
BinaryProcesses::debug( params, itime, iBPs, *this );
}
#pragma omp barrier

timers.collisions.update();
}

void VectorPatch::allocateField( unsigned int ifield, Params &params )
{
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
if( params.geometry != "AMcylindrical" ) {
Field *field = emfields( ipatch )->allFields[ifield];
if( field->data_ != NULL ) {
continue;
}
if( ( field->name.substr( 0, 2 )=="Jx" ) && (!params.is_pxr) ) {
field->allocateDims( 0, false );
} else if( ( field->name.substr( 0, 2 )=="Jy" ) && (!params.is_pxr) ) {
field->allocateDims( 1, false );
} else if( ( field->name.substr( 0, 2 )=="Jz" ) && (!params.is_pxr) ) {
field->allocateDims( 2, false );
} else if( ( field->name.substr( 0, 2 )=="Rh" ) || (params.is_pxr) ) {
field->allocateDims();
}
} else {
cField2D *field = static_cast<cField2D *>( emfields( ipatch )->allFields[ifield] );
if( field->cdata_ != NULL ) {
continue;
}
if( ( field->name.substr( 0, 2 )=="Jl" ) && (!params.is_pxr) ) {
field->allocateDims( 0, false );
} else if( ( field->name.substr( 0, 2 )=="Jr" ) && (!params.is_pxr) ) {
field->allocateDims( 1, false );
} else if( ( field->name.substr( 0, 2 )=="Jt" ) && (!params.is_pxr) ) {
field->allocateDims( 2, false );
} else if( ( field->name.substr( 0, 2 )=="Rh" ) || (params.is_pxr) ) {
field->allocateDims();
}
}
}
}


void VectorPatch::applyExternalFields()
{
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
patches_[ipatch]->EMfields->applyExternalFields( ( *this )( ipatch ) );    
}
}


void VectorPatch::applyPrescribedFields(double time)
{
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
patches_[ipatch]->EMfields->applyPrescribedFields( ( *this )( ipatch ), time );
}
}

void VectorPatch::resetPrescribedFields()
{
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
patches_[ipatch]->EMfields->resetPrescribedFields();
}
}

void VectorPatch::saveExternalFields( Params &params )
{
if( params.save_magnectic_fields_for_SM ) {
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
patches_[ipatch]->EMfields->saveExternalFields( ( *this )( ipatch ) );    
}
}
}

string combineMemoryConsumption( SmileiMPI *smpi, long int data, string name )
{
long int maxData( 0 );
MPI_Reduce( &data, &maxData, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD );

double globalData = ( double )data / 1024./1024./1024.;
MPI_Reduce( smpi->isMaster()?MPI_IN_PLACE:&globalData, &globalData, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );

ostringstream t("");
t << setw(22) << name << ": "
<< "Master " << ( int )( ( double )data / 1024./1024. ) << " MB;   "
<< "Max " << ( int )( ( double )maxData / 1024./1024. ) << " MB;   "
<< "Global " << setprecision( 3 ) << globalData << " GB";
return t.str();
}

void VectorPatch::checkMemoryConsumption( SmileiMPI *smpi, VectorPatch *region_vecpatches )
{
long int particlesMem( 0 );
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<patches_[ipatch]->vecSpecies.size(); ispec++ ) {
particlesMem += patches_[ipatch]->vecSpecies[ispec]->getMemFootPrint();
}
}
string m = combineMemoryConsumption( smpi, particlesMem, "Particles" );
MESSAGE( m );

long int fieldsMem( 0 );
for( unsigned int ipatch=0 ; ipatch<size() ; ipatch++ ) {
fieldsMem += patches_[ipatch]->EMfields->getMemFootPrint();
}
m = combineMemoryConsumption( smpi, fieldsMem, "Fields" );
MESSAGE( m );

if( ! region_vecpatches->patches_.empty() ) {
long int RegionMem = region_vecpatches->patches_[0]->EMfields->getMemFootPrint();
m = combineMemoryConsumption( smpi, RegionMem, "SDMD grid" );
MESSAGE( m );
}

vector<Diagnostic*> allDiags( 0 );
allDiags.insert( allDiags.end(), globalDiags.begin(), globalDiags.end() );
allDiags.insert( allDiags.end(), localDiags.begin(), localDiags.end() );
for( unsigned int idiags=0 ; idiags<allDiags.size() ; idiags++ ) {
long int diagsMem = allDiags[idiags]->getMemFootPrint();
m = combineMemoryConsumption( smpi, diagsMem, allDiags[idiags]->filename );
MESSAGE( m );
}

}


void VectorPatch::saveOldRho( Params &params )
{

int n=0;
if( params.geometry!="AMcylindrical" ) {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
n = ( *this )( ipatch )->EMfields->rhoold_->dims_[0]*( *this )( ipatch )->EMfields->rhoold_->dims_[1]; 
if( params.nDim_field ==3 ) {
n*=( *this )( ipatch )->EMfields->rhoold_->dims_[2];
}
std::memcpy( ( *this )( ipatch )->EMfields->rhoold_->data_, ( *this )( ipatch )->EMfields->rho_->data_, sizeof( double )*n );
}
} else {
cField2D *rho, *rhoold;
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM* amfield = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields);
n = amfield->rho_old_AM_[0]->dims_[0] * amfield->rho_old_AM_[0]->dims_[1];
for( unsigned int imode=0 ; imode < params.nmodes ; imode++ ) {
rho = amfield->rho_AM_[imode];
rhoold = amfield->rho_old_AM_[imode];
std::memcpy( &((*rhoold)(0,0)), &((*rho)(0,0)) , sizeof( complex<double> )*n );
}
}

}
}


void VectorPatch::setMagneticFieldsForDiagnostic( Params &params )
{
if ( !params.is_spectral ) {
ERROR( "Should not come here for non spectral solver" );
}

if ( params.geometry!="AMcylindrical" ) {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagn* emfield = static_cast<ElectroMagn *>( ( *this )( ipatch )->EMfields);
if ( emfield->Bx_->data_ != emfield->Bx_m->data_ ) {
emfield->Bx_->deallocateDataAndSetTo( emfield->Bx_m );
emfield->By_->deallocateDataAndSetTo( emfield->By_m );
emfield->Bz_->deallocateDataAndSetTo( emfield->Bz_m );
}
}
}
else {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
ElectroMagnAM* amfield = static_cast<ElectroMagnAM *>( ( *this )( ipatch )->EMfields);
if ( amfield->Bl_[0]->cdata_ != amfield->Bl_m[0]->cdata_ ) {
for( unsigned int imode=0 ; imode < params.nmodes ; imode++ ) {
amfield->Bl_[imode]->deallocateDataAndSetTo( amfield->Bl_m[imode] );
amfield->Br_[imode]->deallocateDataAndSetTo( amfield->Br_m[imode] );
amfield->Bt_[imode]->deallocateDataAndSetTo( amfield->Bt_m[imode] );
}
}
}
}

}


void VectorPatch::checkExpectedDiskUsage( SmileiMPI *smpi, Params &params, Checkpoint &checkpoint )
{
if( smpi->isMaster() ) {

MESSAGE( 1, "WARNING: disk usage by non-uniform particles maybe strongly underestimated," );
MESSAGE( 1, "   especially when particles are created at runtime (ionization, pair generation, etc.)" );
MESSAGE( 1, "" );

int istart = 0, istop = params.n_time;
if( params.restart ) {
istart = checkpoint.this_run_start_step+1;
}
if( checkpoint.dump_step > 0 && checkpoint.exit_after_dump ) {
int ncheckpoint = ( istart/( int )checkpoint.dump_step ) + 1;
int nextdumptime = ncheckpoint * ( int )checkpoint.dump_step;
if( nextdumptime < istop ) {
istop = nextdumptime;
}
}

MESSAGE( 1, "Expected disk usage for diagnostics:" );
uint64_t diagnostics_footprint = 0;
for( unsigned int idiags=0 ; idiags<localDiags.size() ; idiags++ ) {
uint64_t footprint = localDiags[idiags]->getDiskFootPrint( istart, istop, patches_[0] );
diagnostics_footprint += footprint;
MESSAGE( 2, "File " << localDiags[idiags]->filename << ": " << Tools::printBytes( footprint ) );
}
for( unsigned int idiags=0 ; idiags<globalDiags.size() ; idiags++ ) {
uint64_t footprint = globalDiags[idiags]->getDiskFootPrint( istart, istop, patches_[0] );
diagnostics_footprint += footprint;
MESSAGE( 2, "File " << globalDiags[idiags]->filename << ": " << Tools::printBytes( footprint ) );
}
MESSAGE( 1, "Total disk usage for diagnostics: " << Tools::printBytes( diagnostics_footprint ) );
MESSAGE( 1, "" );

if( checkpoint.dump_step > 0 || checkpoint.dump_minutes > 0 ) {
MESSAGE( 1, "Expected disk usage for each checkpoint:" );

ElectroMagn *EM = patches_[0]->EMfields;
uint64_t n_grid_points = 1;
for( unsigned int i=0; i<params.nDim_field; i++ ) {
n_grid_points *= ( params.n_space[i] + 2*params.oversize[i]+1 );
}
n_grid_points *= params.tot_number_of_patches;
unsigned int n_fields = 9
+ EM->Exfilter.size() + EM->Eyfilter.size() + EM->Ezfilter.size()
+ EM->Bxfilter.size() + EM->Byfilter.size() + EM->Bzfilter.size();
for( unsigned int idiag=0; idiag<EM->allFields_avg.size(); idiag++ ) {
n_fields += EM->allFields_avg[idiag].size();
}
uint64_t checkpoint_fields_footprint = n_grid_points * ( uint64_t )( n_fields * sizeof( double ) );
MESSAGE( 2, "For fields: " << Tools::printBytes( checkpoint_fields_footprint ) );

uint64_t checkpoint_particles_footprint = 0;
for( unsigned int ispec=0 ; ispec<patches_[0]->vecSpecies.size() ; ispec++ ) {
Species *s = patches_[0]->vecSpecies[ispec];
Particles *p = s->particles;
uint64_t one_particle_size = 0;
one_particle_size += ( p->Position.size() + p->Momentum.size() + 1 ) * sizeof( double );
one_particle_size += 1 * sizeof( short );
if( p->tracked ) {
one_particle_size += 1 * sizeof( uint64_t );
}
PeekAtSpecies peek( params, ispec );
uint64_t number_of_particles = peek.totalNumberofParticles();
uint64_t b_size = ( s->particles->first_index.size() + s->particles->last_index.size() ) * params.tot_number_of_patches * sizeof( int );
checkpoint_particles_footprint += one_particle_size*number_of_particles + b_size;
}
MESSAGE( 2, "For particles: " << Tools::printBytes( checkpoint_particles_footprint ) );

uint64_t checkpoint_diags_footprint = 0;
n_fields = 0;
for( unsigned int idiag=0; idiag<EM->allFields_avg.size(); idiag++ ) {
n_fields += EM->allFields_avg[idiag].size();
}
checkpoint_diags_footprint += n_grid_points * ( uint64_t )( n_fields * sizeof( double ) );
for( unsigned int idiag=0; idiag<globalDiags.size(); idiag++ )
if( DiagnosticScreen *screen = dynamic_cast<DiagnosticScreen *>( globalDiags[idiag] ) ) {
checkpoint_diags_footprint += screen->getData()->size() * sizeof( double );
}
MESSAGE( 2, "For diagnostics: " << Tools::printBytes( checkpoint_diags_footprint ) );

uint64_t checkpoint_footprint = checkpoint_fields_footprint + checkpoint_particles_footprint + checkpoint_diags_footprint;
MESSAGE( 1, "Total disk usage for one checkpoint: " << Tools::printBytes( checkpoint_footprint ) );
}

}
}

void VectorPatch::runEnvelopeModule( Params &params,
SmileiMPI *smpi,
SimWindow *simWindow,
double time_dual, Timers &timers, int itime )
{
ponderomotiveUpdateSusceptibilityAndMomentum( params, smpi, simWindow, time_dual, timers, itime );

sumSusceptibility( params, time_dual, timers, itime, simWindow, smpi );

solveEnvelope( params, simWindow, itime, time_dual, timers, smpi );

ponderomotiveUpdatePositionAndCurrents( params, smpi, simWindow, time_dual, timers, itime );

}

void VectorPatch::ponderomotiveUpdateSusceptibilityAndMomentum( Params &params,
SmileiMPI *smpi,
SimWindow *simWindow,
double time_dual, Timers &timers, int itime )
{

#pragma omp single
diag_flag = needsRhoJsNow( itime );

timers.particles.restart();

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->restartEnvChi();
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->isProj( time_dual, simWindow ) || diag_flag ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->vectorized_operators )
species( ipatch, ispec )->ponderomotiveUpdateSusceptibilityAndMomentum( time_dual, ispec,
emfields( ipatch ),
params, diag_flag,
( *this )( ipatch ), smpi,
localDiags );
else {
if( params.vectorization_mode == "adaptive" ) {
species( ipatch, ispec )->scalarPonderomotiveUpdateSusceptibilityAndMomentum( time_dual, ispec,
emfields( ipatch ),
params, diag_flag,
( *this )( ipatch ), smpi,
localDiags );
} else {
species( ipatch, ispec )->Species::ponderomotiveUpdateSusceptibilityAndMomentum( time_dual, ispec,
emfields( ipatch ),
params, diag_flag,
( *this )( ipatch ), smpi,
localDiags );
}
}
} 
} 
} 

timers.particles.update( );
#ifdef __DETAILED_TIMERS
timers.interp_fields_env.update( *this, params.printNow( itime ) );
timers.proj_susceptibility.update( *this, params.printNow( itime ) );
timers.push_mom.update( *this, params.printNow( itime ) );
#endif

} 

void VectorPatch::ponderomotiveUpdatePositionAndCurrents( Params &params,
SmileiMPI *smpi,
SimWindow *simWindow,
double time_dual, Timers &timers, int itime )
{

#pragma omp single
diag_flag = needsRhoJsNow( itime );

timers.particles.restart();

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
for( unsigned int ispec=0 ; ispec<( *this )( ipatch )->vecSpecies.size() ; ispec++ ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->isProj( time_dual, simWindow ) || diag_flag ) {
if( ( *this )( ipatch )->vecSpecies[ispec]->vectorized_operators ){
species( ipatch, ispec )->ponderomotiveUpdatePositionAndCurrents( time_dual, ispec,
emfields( ipatch ),
params, diag_flag, partwalls( ipatch ),
( *this )( ipatch ), smpi,
localDiags );
} else {

if( params.vectorization_mode == "adaptive" ) {
species( ipatch, ispec )->scalarPonderomotiveUpdatePositionAndCurrents( time_dual, ispec,
emfields( ipatch ),
params, diag_flag, partwalls( ipatch ),
( *this )( ipatch ), smpi,
localDiags );
} else {
species( ipatch, ispec )->Species::ponderomotiveUpdatePositionAndCurrents( time_dual, ispec,
emfields( ipatch ),
params, diag_flag, partwalls( ipatch ),
( *this )( ipatch ), smpi,
localDiags );
}
}
} 
} 
} 

timers.particles.update( params.printNow( itime ) );
#ifdef __DETAILED_TIMERS
timers.interp_env_old.update( *this, params.printNow( itime ) );
timers.proj_currents.update( *this, params.printNow( itime ) );
timers.push_pos.update( *this, params.printNow( itime ) );
timers.cell_keys.update( *this, params.printNow( itime ) );
#endif

timers.syncPart.restart();
for( unsigned int ispec=0 ; ispec<( *this )( 0 )->vecSpecies.size(); ispec++ ) {
if( ( *this )( 0 )->vecSpecies[ispec]->isProj( time_dual, simWindow ) ) {
SyncVectorPatch::exchangeParticles( ( *this ), ispec, params, smpi, timers, itime ); 
} 
} 
timers.syncPart.update( params.printNow( itime ) );



} 


void VectorPatch::initNewEnvelope( Params &params )
{
if( ( *this )( 0 )->EMfields->envelope!=NULL ) {
for( unsigned int ipatch=0 ; ipatch<this->size() ; ipatch++ ) {
( *this )( ipatch )->EMfields->envelope->initEnvelope( ( *this )( ipatch ), ( *this )( ipatch )->EMfields );
} 
}
} 
