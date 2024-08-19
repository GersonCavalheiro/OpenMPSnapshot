#include "SpeciesVAdaptiveMixedSort.h"

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


SpeciesVAdaptiveMixedSort::SpeciesVAdaptiveMixedSort( Params &params, Patch *patch ) :
SpeciesV( params, patch )
{
initCluster( params );
npack_ = 0 ;
packsize_ = 0;

}

SpeciesVAdaptiveMixedSort::~SpeciesVAdaptiveMixedSort()
{
}


void SpeciesVAdaptiveMixedSort::resizeCluster( Params &params )
{

if( vectorized_operators ) {

unsigned int ncells = ( params.n_space[0]+1 );
for( unsigned int i=1; i < params.nDim_field; i++ ) {
ncells *= ( params.n_space[i]+1 );
}


particles->last_index.resize( ncells, 0 );
particles->first_index.resize( ncells, 0 );

particles->first_index[0] = 0;
for( unsigned int ic=1; ic < ncells; ic++ ) {
particles->first_index[ic] = particles->first_index[ic-1] + count[ic-1];
particles->last_index[ic-1]= particles->first_index[ic];
}
particles->last_index[ncells-1] = particles->last_index[ncells-2] + count.back() ;

} else {

Species::resizeCluster( params );

}
}




void SpeciesVAdaptiveMixedSort::importParticles( Params &params, Patch *patch, Particles &source_particles, vector<Diagnostic *> &localDiags )
{

if( vectorized_operators ) {
importParticles( params, patch, source_particles, localDiags );
} else {
Species::importParticles( params, patch, source_particles, localDiags );
}
}

void SpeciesVAdaptiveMixedSort::sortParticles( Params &params, Patch * patch )
{
if( vectorized_operators ) {
SpeciesV::sortParticles( params , patch);
} else {
Species::sortParticles( params, patch );
}
}

void SpeciesVAdaptiveMixedSort::defaultConfigure( Params &params, Patch *patch )
{
this->vectorized_operators = ( params.adaptive_default_mode == "on" );


this->reconfigure_operators( params, patch );


resizeCluster( params );

this->sortParticles( params , patch);

this->reconfigure_particle_importation();

}

void SpeciesVAdaptiveMixedSort::configuration( Params &params, Patch *patch )
{
float vecto_time = 0.;
float scalar_time = 0.;

this->computeParticleCellKeys( params );

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


#ifdef  __DEBUG
std::cerr << "  > Species " << this->name_ << " configuration (" << this->vectorized_operators
<< ") default: " << params.adaptive_default_mode
<< " in patch (" << patch->Pcoordinates[0] << "," <<  patch->Pcoordinates[1] << "," <<  patch->Pcoordinates[2] << ")"
<< " of MPI process " << patch->MPI_me_
<< " (vecto time: " << vecto_time
<< ", scalar time: " << scalar_time
<< ", particle number: " << particles->size()
<< ")" << '\n';
#endif

this->reconfigure_operators( params, patch );


resizeCluster( params );

this->sortParticles( params , patch);

this->reconfigure_particle_importation();

}

void SpeciesVAdaptiveMixedSort::reconfiguration( Params &params, Patch *patch )
{

bool reasign_operators = false;
float vecto_time = 0;
float scalar_time = 0;


if( !this->vectorized_operators ) {
this->computeParticleCellKeys( params );
}



(*part_comp_time_)( count,
vecto_time,
scalar_time );

if( ( vecto_time < scalar_time && this->vectorized_operators == false )
|| ( vecto_time > scalar_time && this->vectorized_operators == true ) ) {
reasign_operators = true;
}

if( reasign_operators ) {

this->vectorized_operators = !this->vectorized_operators;

#ifdef  __DEBUG
std::cerr << "  > Species " << this->name_ << " reconfiguration (" << this->vectorized_operators
<< ") in patch (" << patch->Pcoordinates[0] << "," <<  patch->Pcoordinates[1] << "," <<  patch->Pcoordinates[2] << ")"
<< " of MPI process " << patch->MPI_me_
<< " (vecto time: " << vecto_time
<< ", scalar time: " << scalar_time
<< ", particle number: " << particles->size()
<< ")" << '\n';
#endif

this->reconfigure_operators( params, patch );


resizeCluster( params );

this->sortParticles( params, patch );

this->reconfigure_particle_importation();
}
}

void SpeciesVAdaptiveMixedSort::reconfigure_operators( Params &params, Patch *patch )
{
delete Interp;
delete Push;
if( Push_ponderomotive_position ) {
delete Push_ponderomotive_position;
}
delete Proj;

Interp = InterpolatorFactory::create( params, patch, this->vectorized_operators );
Push = PusherFactory::create( params, this );
if( Push_ponderomotive_position ) {
Push_ponderomotive_position = PusherFactory::create_ponderomotive_position_updater( params, this );
}
Proj = ProjectorFactory::create( params, patch, this->vectorized_operators );
}

void SpeciesVAdaptiveMixedSort::reconfigure_particle_importation()
{
if( this->Ionize ) {
this->electron_species->vectorized_operators = this->vectorized_operators;
}
if( this->Radiate ) {
this->photon_species_->vectorized_operators = this->vectorized_operators;
}
if( this->Multiphoton_Breit_Wheeler_process ) {
for( int k=0; k<2; k++ ) {
this->mBW_pair_species_[k]->vectorized_operators = this->vectorized_operators;
}
}
}
