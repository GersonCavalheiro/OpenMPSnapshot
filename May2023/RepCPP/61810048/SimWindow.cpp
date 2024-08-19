
#include "SimWindow.h"
#include "Params.h"
#include "Species.h"
#include "SpeciesVAdaptiveMixedSort.h"
#include "SpeciesVAdaptive.h"
#include "SpeciesV.h"
#include "ElectroMagn.h"
#include "Interpolator.h"
#include "Projector.h"
#include "SmileiMPI.h"
#include "VectorPatch.h"
#include "Region.h"
#include "DiagnosticProbes.h"
#include "DiagnosticTrack.h"
#include "Hilbert_functions.h"
#include "PatchesFactory.h"
#include <iostream>
#include <omp.h>
#include <fstream>
#include <limits>
#include "ElectroMagnBC_Factory.h"
#include "DoubleGrids.h"
#include "SyncVectorPatch.h"

using namespace std;

SimWindow::SimWindow( Params &params )
{

active = false;
time_start = numeric_limits<double>::max();
velocity_x = 1.;
number_of_additional_shifts = 0;
additional_shifts_time = 0.;

#ifdef _OPENMP
max_threads = omp_get_max_threads();
#else
max_threads = 1;
#endif
patch_to_be_created.resize( max_threads );
patch_particle_created.resize( max_threads );

if( PyTools::nComponents( "MovingWindow" ) ) {
active = true;

TITLE( "Initializing moving window" );

PyTools::extract( "time_start", time_start, "MovingWindow"  );
PyTools::extract( "velocity_x", velocity_x, "MovingWindow"  );
PyTools::extract( "number_of_additional_shifts", number_of_additional_shifts, "MovingWindow"  );
PyTools::extract( "additional_shifts_time", additional_shifts_time, "MovingWindow"  );
}

cell_length_x_   = params.cell_length[0];
n_space_x_       = params.n_space[0];
additional_shifts_iteration = floor(additional_shifts_time / params.timestep + 0.5);
x_moved = 0.;      
n_moved = 0 ;      


if( active ) {

MESSAGE( 1, "Moving window is active:" );
MESSAGE( 2, "velocity_x : " << velocity_x );
MESSAGE( 2, "time_start : " << time_start );
if (number_of_additional_shifts > 0){
MESSAGE( 2, "number_of_additional_shifts : " << number_of_additional_shifts );
MESSAGE( 2, "additional_shifts_time : " << additional_shifts_time );
}
params.hasWindow = true;
} else {
params.hasWindow = false;
}

}

SimWindow::~SimWindow()
{
}

bool SimWindow::isMoving( double time_dual )
{
return active && ( ( time_dual - time_start )*velocity_x > x_moved - number_of_additional_shifts*cell_length_x_*n_space_x_*(time_dual>additional_shifts_time) );
}

void SimWindow::shift( VectorPatch &vecPatches, SmileiMPI *smpi, Params &params, unsigned int itime, double time_dual, Region& region )
{
if( ! isMoving( time_dual ) && itime != additional_shifts_iteration ) {
return;
}

unsigned int h0;
Patch *mypatch;

h0 = vecPatches( 0 )->hindex;
unsigned int nPatches = vecPatches.size();
unsigned int nSpecies( vecPatches( 0 )->vecSpecies.size() );
int nmessage( vecPatches.nrequests );

std::vector<Patch *> delete_patches_, update_patches_, send_patches_;

#ifdef _OPENMP
int my_thread = omp_get_thread_num();
#else
int my_thread = 0;
#endif

#ifdef _NO_MPI_TM
#pragma omp master
{
#endif

( patch_to_be_created[my_thread] ).clear();
( patch_particle_created[my_thread] ).clear();

#ifndef _NO_MPI_TM
#pragma omp single
#endif
{
if( n_moved == 0 ) {
MESSAGE( ">>> Window starts moving" );
}

vecPatches_old.resize( nPatches );
n_moved += params.n_space[0];
}
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#endif
for( unsigned int ipatch = 0 ; ipatch < nPatches ; ipatch++ ) {
vecPatches_old[ipatch] = vecPatches( ipatch );
vecPatches( ipatch )->EMfields->laserDisabled();
}


#ifndef _NO_MPI_TM
#pragma omp for schedule(static) private(mypatch)
#endif
for( unsigned int ipatch = 0 ; ipatch < nPatches ; ipatch++ ) {
mypatch = vecPatches_old[ipatch];
if( mypatch->MPI_neighbor_[0][1] != mypatch->MPI_me_ ) {
( patch_to_be_created[my_thread] ).push_back( ipatch );
( patch_particle_created[my_thread] ).push_back( true );
}

if( mypatch->isXmax() && mypatch->EMfields->emBoundCond[1] ) {
mypatch->EMfields->emBoundCond[1]->disableExternalFields();
}

if( mypatch->MPI_neighbor_[0][0] != mypatch->MPI_me_ ) {
delete_patches_.push_back( mypatch ); 
if( mypatch->MPI_neighbor_[0][0] != MPI_PROC_NULL ) {
if ( vecPatches_old[ipatch]->Pcoordinates[0]!=0 ) {
send_patches_.push_back( mypatch ); 
int Href_receiver = 0;
for (int irk = 0; irk < mypatch->MPI_neighbor_[0][0]; irk++) Href_receiver += smpi->patch_count[irk];
smpi->isend( vecPatches_old[ipatch], vecPatches_old[ipatch]->MPI_neighbor_[0][0], ( vecPatches_old[ipatch]->neighbor_[0][0] - Href_receiver ) * nmessage, params, false );
}
}
} else { 

if( mypatch->isXmax() )
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
mypatch->vecSpecies[ispec]->disableXmax();
}
mypatch->Pcoordinates[0] -= 1;
mypatch->neighbor_[0][1] =  mypatch->hindex;
mypatch->hindex = mypatch->neighbor_[0][0];
mypatch->MPI_neighbor_[0][1] = mypatch->MPI_me_ ;
mypatch->tmp_neighbor_[0][0] = vecPatches_old[mypatch->hindex - h0 ]->neighbor_[0][0];
mypatch->tmp_MPI_neighbor_[0][0] = vecPatches_old[mypatch->hindex - h0 ]->MPI_neighbor_[0][0];
for( unsigned int idim = 1; idim < params.nDim_field ; idim++ ) {
mypatch->tmp_neighbor_[idim][0] = vecPatches_old[mypatch->hindex - h0 ]->neighbor_[idim][0];
mypatch->tmp_neighbor_[idim][1] = vecPatches_old[mypatch->hindex - h0 ]->neighbor_[idim][1];
mypatch->tmp_MPI_neighbor_[idim][0] = vecPatches_old[mypatch->hindex - h0 ]->MPI_neighbor_[idim][0];
mypatch->tmp_MPI_neighbor_[idim][1] = vecPatches_old[mypatch->hindex - h0 ]->MPI_neighbor_[idim][1];
}
update_patches_.push_back( mypatch ); 

vecPatches.patches_[mypatch->hindex - h0 ] = mypatch ;

}
}

for( unsigned int j = 0; j < patch_to_be_created[my_thread].size();  j++ ) {
#ifndef _NO_MPI_TM
#pragma omp critical
#endif
mypatch = PatchesFactory::clone( vecPatches( 0 ), params, smpi, vecPatches.domain_decomposition_, h0 + patch_to_be_created[my_thread][j], n_moved, false );

if( mypatch->isXmin() && mypatch->EMfields->emBoundCond[0] ) {
mypatch->EMfields->emBoundCond[0]->disableExternalFields();
}

mypatch->finalizeMPIenvironment( params );
vecPatches.patches_[patch_to_be_created[my_thread][j]] = mypatch ;
if( mypatch->MPI_neighbor_[0][1] != MPI_PROC_NULL ) {
if ( mypatch->Pcoordinates[0]!=params.number_of_patches[0]-1 ) {
smpi->recv( mypatch, mypatch->MPI_neighbor_[0][1], ( mypatch->hindex - vecPatches.refHindex_ )*nmessage, params, false );
patch_particle_created[my_thread][j] = false ; 
}
}

if( mypatch->isXmin() ) {
for( auto &embc:mypatch->EMfields->emBoundCond ) {
if( embc ) {
delete embc;
}
}
mypatch->EMfields->emBoundCond = ElectroMagnBC_Factory::create( params, mypatch );
mypatch->EMfields->laserDisabled();
if (!params.multiple_decomposition)
mypatch->EMfields->emBoundCond[0]->apply(mypatch->EMfields, time_dual, mypatch);
}

mypatch->EMfields->laserDisabled();
mypatch->EMfields->updateGridSize( params, mypatch );
}


#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#endif
for( unsigned int ipatch = 0 ; ipatch < nPatches ; ipatch++ ) {
if( vecPatches_old[ipatch]->MPI_neighbor_[0][0] !=  vecPatches_old[ipatch]->MPI_me_ && vecPatches_old[ipatch]->MPI_neighbor_[0][0] != MPI_PROC_NULL ) {
smpi->waitall( vecPatches_old[ipatch] );
}
}

for( unsigned int j=0; j < update_patches_.size(); j++ ) {
mypatch = update_patches_[j];
mypatch->MPI_neighbor_[0][0] = mypatch->tmp_MPI_neighbor_[0][0];
mypatch->neighbor_[0][0] = mypatch->tmp_neighbor_[0][0];
for( unsigned int idim = 1; idim < params.nDim_field ; idim++ ) {
mypatch->MPI_neighbor_[idim][0] = mypatch->tmp_MPI_neighbor_[idim][0];
mypatch->MPI_neighbor_[idim][1] = mypatch->tmp_MPI_neighbor_[idim][1];
mypatch->neighbor_[idim][0] = mypatch->tmp_neighbor_[idim][0];
mypatch->neighbor_[idim][1] = mypatch->tmp_neighbor_[idim][1];
}

mypatch->updateTagenv( smpi );
if( mypatch->isXmin() ) {
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
mypatch->vecSpecies[ispec]->setXminBoundaryCondition();
}
}

if( mypatch->isXmin() ) {
for( auto &embc:mypatch->EMfields->emBoundCond ) {
if( embc ) {
delete embc;
}
}
mypatch->EMfields->emBoundCond = ElectroMagnBC_Factory::create( params, mypatch );
mypatch->EMfields->laserDisabled();
if (!params.multiple_decomposition)
mypatch->EMfields->emBoundCond[0]->apply(mypatch->EMfields, time_dual, mypatch);
}
if( mypatch->wasXmax( params ) ) {
for( auto &embc:mypatch->EMfields->emBoundCond ) {
if( embc ) {
delete embc;
}
}
mypatch->EMfields->emBoundCond = ElectroMagnBC_Factory::create( params, mypatch );
mypatch->EMfields->laserDisabled();
mypatch->EMfields->updateGridSize( params, mypatch );

}
}

for( unsigned int j=0; j < send_patches_.size(); j++ ) {
smpi->waitall( send_patches_[j] );
}

#ifndef _NO_MPI_TM
#pragma omp barrier
#endif

#ifndef _NO_MPI_TM
#pragma omp master
#endif
{
for( int ithread=0; ithread < max_threads ; ithread++ ) {
for( unsigned int j=0; j< ( patch_to_be_created[ithread] ).size(); j++ ) {

mypatch = vecPatches.patches_[patch_to_be_created[ithread][j]];

if( patch_particle_created[ithread][j] ) {
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
ParticleCreator particle_creator;
particle_creator.associate(mypatch->vecSpecies[ispec]);

struct SubSpace init_space;
init_space.cell_index_[0] = 0;
init_space.cell_index_[1] = 0;
init_space.cell_index_[2] = 0;
init_space.box_size_[0]   = params.n_space[0];
init_space.box_size_[1]   = params.n_space[1];
init_space.box_size_[2]   = params.n_space[2];

particle_creator.create( init_space, params, mypatch, 0 );

}

mypatch->EMfields->applyExternalFields( mypatch );
if( params.save_magnectic_fields_for_SM ) {
mypatch->EMfields->saveExternalFields( mypatch );
}

} 
} 
} 
} 
#ifndef _NO_MPI_TM
#pragma omp barrier
#endif

if( ( params.vectorization_mode == "on" ) ) {

#ifndef _NO_MPI_TM
#pragma omp for schedule(static) private(mypatch)
#endif
for( int ithread=0; ithread < max_threads ; ithread++ ) {
for( unsigned int j=0; j< ( patch_to_be_created[ithread] ).size(); j++ ) {

mypatch = vecPatches.patches_[patch_to_be_created[ithread][j]];

for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
mypatch->vecSpecies[ispec]->computeParticleCellKeys( params );
mypatch->vecSpecies[ispec]->sortParticles( params , mypatch);
}
} 
} 
}

else if( params.vectorization_mode == "adaptive_mixed_sort" ) {
#ifndef _NO_MPI_TM
#pragma omp for schedule(static) private(mypatch)
#endif
for( int ithread=0; ithread < max_threads ; ithread++ ) {
for( unsigned int j=0; j< ( patch_to_be_created[ithread] ).size(); j++ ) {

mypatch = vecPatches.patches_[patch_to_be_created[ithread][j]];

if( patch_particle_created[ithread][j] ) {
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
mypatch->vecSpecies[ispec]->configuration( params, mypatch );
}
}
else { 
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
mypatch->vecSpecies[ispec]->computeParticleCellKeys( params );
dynamic_cast<SpeciesVAdaptiveMixedSort *>( mypatch->vecSpecies[ispec] )->reconfigure_operators( params, mypatch );
}
} 
} 
} 
}
else if( params.vectorization_mode == "adaptive" ) {
#ifndef _NO_MPI_TM
#pragma omp for schedule(static) private(mypatch)
#endif
for( int ithread=0; ithread < max_threads ; ithread++ ) {
for( unsigned int j=0; j< ( patch_to_be_created[ithread] ).size(); j++ ) {

mypatch = vecPatches.patches_[patch_to_be_created[ithread][j]];

if( patch_particle_created[ithread][j] ) {
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
mypatch->vecSpecies[ispec]->computeParticleCellKeys( params );
mypatch->vecSpecies[ispec]->configuration( params, mypatch );
mypatch->vecSpecies[ispec]->sortParticles( params, mypatch );

}
}
else { 
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
mypatch->vecSpecies[ispec]->computeParticleCellKeys( params );
dynamic_cast<SpeciesVAdaptive *>( mypatch->vecSpecies[ispec] )->reconfigure_operators( params, mypatch );
}
} 
} 
} 
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(static) private(mypatch)
#endif
for( int ithread=0; ithread < max_threads ; ithread++ ) {
for( unsigned int j=0; j< ( patch_to_be_created[ithread] ).size(); j++ ) {

mypatch = vecPatches.patches_[patch_to_be_created[ithread][j]];

if( patch_particle_created[ithread][j] ) {
for( unsigned int idiag=0; idiag<vecPatches.localDiags.size(); idiag++ )
if( DiagnosticTrack *track = dynamic_cast<DiagnosticTrack *>( vecPatches.localDiags[idiag] ) ) {
track->setIDs( mypatch );
}
} 
} 
} 
#ifndef _NO_MPI_TM
#pragma omp single nowait
#endif
{
x_moved += cell_length_x_*params.n_space[0];
vecPatches.updateFieldList( smpi ) ;

vecPatches.lastIterationPatchesMoved = itime;
}

std::vector<double> poynting[2];
poynting[0].resize( params.nDim_field, 0.0 );
poynting[1].resize( params.nDim_field, 0.0 );

double energy_field_out( 0. );
double energy_field_inj( 0. );
std::vector<double> energy_part_out( nSpecies, 0. );
std::vector<double> energy_part_inj( nSpecies, 0. );
std::vector<double> ukin_new( nSpecies, 0. );
std::vector<double> ukin_bc ( nSpecies, 0. );
std::vector<double> urad    ( nSpecies, 0. );

for( unsigned int j=0; j < delete_patches_.size(); j++ ) {
mypatch = delete_patches_[j];

if( mypatch->isXmin() ) {
for( unsigned int jp=0; jp<2; jp++ ) { 
for( unsigned int i=0 ; i<params.nDim_field ; i++ ) { 
poynting[jp][i] += mypatch->EMfields->poynting[jp][i];
}
}

energy_field_out += mypatch->EMfields->nrj_mw_out + mypatch->EMfields->computeEnergy();
energy_field_inj += mypatch->EMfields->nrj_mw_inj;
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
energy_part_out[ispec] += mypatch->vecSpecies[ispec]->nrj_mw_out + mypatch->vecSpecies[ispec]->computeEnergy();
energy_part_inj[ispec] += mypatch->vecSpecies[ispec]->nrj_mw_inj;
ukin_new[ispec] += mypatch->vecSpecies[ispec]->nrj_new_part_;
ukin_bc [ispec] += mypatch->vecSpecies[ispec]->nrj_bc_lost;
urad    [ispec] += mypatch->vecSpecies[ispec]->nrj_radiated_;
}
}

delete  mypatch;
}

for( unsigned int j=0; j< patch_to_be_created[my_thread].size(); j++ ) {
mypatch = vecPatches.patches_[patch_to_be_created[my_thread][j]];

if( mypatch->isXmax() ) {
energy_field_inj += mypatch->EMfields->computeEnergy();
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
energy_part_inj[ispec] += mypatch->vecSpecies[ispec]->computeEnergy();
}
}
}

#ifndef _NO_MPI_TM
#pragma omp critical
#endif
{
for( unsigned int j=0; j<2; j++ ) { 
for( unsigned int i=0 ; i< params.nDim_field ; i++ ) { 
vecPatches( 0 )->EMfields->poynting[j][i] += poynting[j][i];
}
}

vecPatches( 0 )->EMfields->nrj_mw_out += energy_field_out;
vecPatches( 0 )->EMfields->nrj_mw_inj += energy_field_inj;
for( unsigned int ispec=0 ; ispec<nSpecies ; ispec++ ) {
vecPatches( 0 )->vecSpecies[ispec]->nrj_mw_out    += energy_part_out[ispec];
vecPatches( 0 )->vecSpecies[ispec]->nrj_mw_inj    += energy_part_inj[ispec];
vecPatches( 0 )->vecSpecies[ispec]->nrj_new_part_ += ukin_new[ispec];
vecPatches( 0 )->vecSpecies[ispec]->nrj_bc_lost   += ukin_bc [ispec];
vecPatches( 0 )->vecSpecies[ispec]->nrj_radiated_ += urad    [ispec];
}
}

#ifdef _NO_MPI_TM
} 
#endif

#pragma omp barrier
#pragma omp master
{
if (params.multiple_decomposition) {
if ( params.geometry != "AMcylindrical" )
operate(region, vecPatches, smpi, params, time_dual);
else {
operate(region, vecPatches, smpi, params, time_dual, params.nmodes);
}
}
}
#pragma omp barrier


if (params.multiple_decomposition) {
if ( params.geometry != "AMcylindrical" ) {
SyncVectorPatch::exchangeE( params, region.vecPatch_, smpi );
SyncVectorPatch::finalizeexchangeE( params, region.vecPatch_ );
SyncVectorPatch::exchangeB( params, region.vecPatch_, smpi );
SyncVectorPatch::finalizeexchangeB( params, region.vecPatch_ );
}
else {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  ) {
SyncVectorPatch::exchangeE( params, region.vecPatch_, imode, smpi );
SyncVectorPatch::exchangeB( params, region.vecPatch_, imode, smpi );
}
}
}

}

void SimWindow::operate(Region& region,  VectorPatch& vecPatches, SmileiMPI* smpi, Params& params, double time_dual)
{
region.patch_->exchangeField_movewin( region.patch_->EMfields->Ex_, params.n_space[0] );
region.patch_->exchangeField_movewin( region.patch_->EMfields->Ey_, params.n_space[0] );
region.patch_->exchangeField_movewin( region.patch_->EMfields->Ez_, params.n_space[0] );

if (region.patch_->EMfields->Bx_->data_!= region.patch_->EMfields->Bx_m->data_) {
region.patch_->exchangeField_movewin( region.patch_->EMfields->Bx_, params.n_space[0] );
region.patch_->exchangeField_movewin( region.patch_->EMfields->By_, params.n_space[0] );
region.patch_->exchangeField_movewin( region.patch_->EMfields->Bz_, params.n_space[0] );
}

region.patch_->exchangeField_movewin( region.patch_->EMfields->Bx_m, params.n_space[0] );
region.patch_->exchangeField_movewin( region.patch_->EMfields->By_m, params.n_space[0] );
region.patch_->exchangeField_movewin( region.patch_->EMfields->Bz_m, params.n_space[0] );

if (params.is_spectral) {
region.patch_->exchangeField_movewin( region.patch_->EMfields->rho_, params.n_space[0] );
region.patch_->exchangeField_movewin( region.patch_->EMfields->rhoold_, params.n_space[0] );
}


region.patch_->EMfields->laserDisabled();
region.patch_->EMfields->emBoundCond[0]->apply(region.patch_->EMfields, time_dual, region.patch_);
region.patch_->EMfields->emBoundCond[1]->apply(region.patch_->EMfields, time_dual, region.patch_);




}


void SimWindow::operate(Region& region,  VectorPatch& vecPatches, SmileiMPI* smpi, Params& params, double time_dual, unsigned int nmodes)
{
ElectroMagnAM * region_fields = static_cast<ElectroMagnAM *>( region.patch_->EMfields );

for (unsigned int imode = 0; imode < nmodes; imode++){
region.patch_->exchangeField_movewin( region_fields->El_[imode], params.n_space[0] );
region.patch_->exchangeField_movewin( region_fields->Er_[imode], params.n_space[0] );
region.patch_->exchangeField_movewin( region_fields->Et_[imode], params.n_space[0] );

if (region_fields->Bl_[imode]->cdata_!= region_fields->Bl_m[imode]->cdata_) {
region.patch_->exchangeField_movewin( region_fields->Bl_[imode], params.n_space[0] );
region.patch_->exchangeField_movewin( region_fields->Br_[imode], params.n_space[0] );
region.patch_->exchangeField_movewin( region_fields->Bt_[imode], params.n_space[0] );
}

region.patch_->exchangeField_movewin( region_fields->Bl_m[imode], params.n_space[0] );
region.patch_->exchangeField_movewin( region_fields->Br_m[imode], params.n_space[0] );
region.patch_->exchangeField_movewin( region_fields->Bt_m[imode], params.n_space[0] );

if (params.is_spectral) {
region.patch_->exchangeField_movewin( region_fields->rho_AM_[imode], params.n_space[0] );
region.patch_->exchangeField_movewin( region_fields->rho_old_AM_[imode], params.n_space[0] );
}
}


region_fields->laserDisabled();
region.patch_->EMfields->emBoundCond[0]->apply(region.patch_->EMfields, time_dual, region.patch_);
region.patch_->EMfields->emBoundCond[1]->apply(region.patch_->EMfields, time_dual, region.patch_);


}
