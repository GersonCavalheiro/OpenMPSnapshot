
#ifndef SYNCVECTORPATCH_H
#define SYNCVECTORPATCH_H

#include <vector>

#include "VectorPatch.h"

class Params;
class SmileiMPI;
class Field;
class cField;
class Timers;

class SyncVectorPatch
{
public :

static void exchangeParticles( VectorPatch &vecPatches, int ispec, Params &params, SmileiMPI *smpi, Timers &timers, int itime );
static void finalizeAndSortParticles( VectorPatch &vecPatches, int ispec, Params &params, SmileiMPI *smpi, Timers &timers, int itime );
static void finalizeExchangeParticles( VectorPatch &vecPatches, int ispec, int iDim, Params &params, SmileiMPI *smpi, Timers &timers, int itime );

static void sumRhoJ( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi, Timers &timers, int itime );
static void sumRhoJ( Params &params, VectorPatch &vecPatches, int imode, SmileiMPI *smpi, Timers &timers, int itime );
static void sumRhoJs( Params &params, VectorPatch &vecPatches, int ispec, SmileiMPI *smpi, Timers &timers, int itime );
static void sumRhoJs( Params &params, VectorPatch &vecPatches, int imode, int ispec, SmileiMPI *smpi, Timers &timers, int itime );
static void sumEnvChi( Params &params, VectorPatch &vecPatches, SmileiMPI *smp, Timers &timers, int itime );
static void sumEnvChis( Params &params, VectorPatch &vecPatches, int ispec, SmileiMPI *smp, Timers &timers, int itime );

template<typename T, typename F> static
void sum( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi, Timers &timers, int itime )
{
unsigned int nx_, ny_, nz_, h0, oversize[3], n_space[3], gsp[3];
T *pt1, *pt2;
F* field1;
F* field2;
h0 = vecPatches( 0 )->hindex;

int nPatches( vecPatches.size() );

oversize[0] = vecPatches( 0 )->EMfields->oversize[0];
oversize[1] = vecPatches( 0 )->EMfields->oversize[1];
oversize[2] = vecPatches( 0 )->EMfields->oversize[2];

n_space[0] = vecPatches( 0 )->EMfields->n_space[0];
n_space[1] = vecPatches( 0 )->EMfields->n_space[1];
n_space[2] = vecPatches( 0 )->EMfields->n_space[2];

unsigned int nComp = fields.size()/nPatches;


#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<fields.size() ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, iNeighbor ) ) {
fields[ifield]->create_sub_fields ( 0, iNeighbor, 2*oversize[0]+1+fields[ifield]->isDual_[0] );
fields[ifield]->extract_fields_sum( 0, iNeighbor, oversize[0] );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initSumField( fields[ifield], 0, smpi );
else
vecPatches( ipatch )->initSumFieldComplex( fields[ifield], 0, smpi );
}

for( unsigned int icomp=0 ; icomp<nComp ; icomp++ ) {
nx_ = fields[icomp*nPatches]->dims_[0];
ny_ = 1;
nz_ = 1;
if( fields[icomp*nPatches]->dims_.size()>1 ) {
ny_ = fields[icomp*nPatches]->dims_[1];
if( fields[icomp*nPatches]->dims_.size()>2 ) {
nz_ = fields[icomp*nPatches]->dims_[2];
}
}
gsp[0] = 1+2*oversize[0]+fields[icomp*nPatches]->isDual_[0]; 
#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=icomp*nPatches ; ifield<( icomp+1 )*nPatches ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[0][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[0][0]-h0+icomp*nPatches] );
field2 = static_cast<F *>( fields[ifield] );
pt1 = &( *field1 )( n_space[0]*ny_*nz_ );
pt2 = &( *field2 )( 0 );
for( unsigned int i = 0; i < gsp[0]* ny_*nz_ ; i++ ) {
pt1[i] += pt2[i];
}
memcpy( pt2, pt1, gsp[0]*ny_*nz_*sizeof( T ) );
}
}
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<fields.size() ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
vecPatches( ipatch )->finalizeSumField( fields[ifield], 0 );
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, ( iNeighbor+1 )%2 ) ) {
fields[ifield]->inject_fields_sum( 0, iNeighbor, oversize[0] );
}
}
}

if( fields[0]->dims_.size()>1 ) {

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<fields.size() ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, iNeighbor ) ) {
fields[ifield]->create_sub_fields ( 1, iNeighbor, 2*oversize[1]+1+fields[ifield]->isDual_[1] );
fields[ifield]->extract_fields_sum( 1, iNeighbor, oversize[1] );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initSumField( fields[ifield], 1, smpi );
else
vecPatches( ipatch )->initSumFieldComplex( fields[ifield], 1, smpi );
}

for( unsigned int icomp=0 ; icomp<nComp ; icomp++ ) {
nx_ = fields[icomp*nPatches]->dims_[0];
ny_ = 1;
nz_ = 1;
if( fields[icomp*nPatches]->dims_.size()>1 ) {
ny_ = fields[icomp*nPatches]->dims_[1];
if( fields[icomp*nPatches]->dims_.size()>2 ) {
nz_ = fields[icomp*nPatches]->dims_[2];
}
}
gsp[0] = 1+2*oversize[0]+fields[icomp*nPatches]->isDual_[0]; 
gsp[1] = 1+2*oversize[1]+fields[icomp*nPatches]->isDual_[1]; 

#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=icomp*nPatches ; ifield<( icomp+1 )*nPatches ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[1][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[1][0]-h0+icomp*nPatches] );
field2 = static_cast<F *>( fields[ifield] );
pt1 = &( *field1 )( n_space[1]*nz_ );
pt2 = &( *field2 )( 0 );
for( unsigned int j = 0; j < nx_ ; j++ ) {
for( unsigned int i = 0; i < gsp[1]*nz_ ; i++ ) {
pt1[i] += pt2[i];
}
memcpy( pt2, pt1, gsp[1]*nz_*sizeof( T ) );
pt1 += ny_*nz_;
pt2 += ny_*nz_;
}
}
}
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<fields.size() ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
vecPatches( ipatch )->finalizeSumField( fields[ifield], 1 );
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, ( iNeighbor+1 )%2 ) ) {
fields[ifield]->inject_fields_sum( 1, iNeighbor, oversize[1] );
}
}
}

if( fields[0]->dims_.size()>2 ) {

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<fields.size() ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, iNeighbor ) ) {
fields[ifield]->create_sub_fields ( 2, iNeighbor, 2*oversize[2]+1+fields[ifield]->isDual_[2] );
fields[ifield]->extract_fields_sum( 2, iNeighbor, oversize[2] );
}
}
vecPatches( ipatch )->initSumField( fields[ifield], 2, smpi );
}

for( unsigned int icomp=0 ; icomp<nComp ; icomp++ ) {
nx_ = fields[icomp*nPatches]->dims_[0];
ny_ = 1;
nz_ = 1;
if( fields[icomp*nPatches]->dims_.size()>1 ) {
ny_ = fields[icomp*nPatches]->dims_[1];
if( fields[icomp*nPatches]->dims_.size()>2 ) {
nz_ = fields[icomp*nPatches]->dims_[2];
}
}
gsp[0] = 1+2*oversize[0]+fields[icomp*nPatches]->isDual_[0]; 
gsp[1] = 1+2*oversize[1]+fields[icomp*nPatches]->isDual_[1]; 
gsp[2] = 1+2*oversize[2]+fields[icomp*nPatches]->isDual_[2]; 
#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=icomp*nPatches ; ifield<( icomp+1 )*nPatches ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[2][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[2][0]-h0+icomp*nPatches] );
field2 = static_cast<F *>( fields[ifield] );
pt1 = &( *field1 )( n_space[2] );
pt2 = &( *field2 )( 0 );
for( unsigned int j = 0; j < nx_*ny_ ; j++ ) {
for( unsigned int i = 0; i < gsp[2] ; i++ ) {
pt1[i] += pt2[i];
pt2[i] =  pt1[i];
}
pt1 += nz_;
pt2 += nz_;
}
}
}
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<fields.size() ; ifield++ ) {
unsigned int ipatch = ifield%nPatches;
vecPatches( ipatch )->finalizeSumField( fields[ifield], 2 );
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, ( iNeighbor+1 )%2 ) ) {
fields[ifield]->inject_fields_sum( 2, iNeighbor, oversize[2] );
}
}
}

} 

} 

}

static void sumAllComponents( std::vector<Field *> &fields, VectorPatch &vecPatches, SmileiMPI *smpi, Timers &timers, int itime );

void templateGenerator();

static void exchangeE( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeexchangeE( Params &params, VectorPatch &vecPatches );
static void exchangeB( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeexchangeB( Params &params, VectorPatch &vecPatches );

static void exchangeE( Params &params, VectorPatch &vecPatches, int imode, SmileiMPI *smpi );
static void finalizeexchangeE( Params &params, VectorPatch &vecPatches, int imode );   
static void exchangeB( Params &params, VectorPatch &vecPatches, int imode, SmileiMPI *smpi );
static void finalizeexchangeB( Params &params, VectorPatch &vecPatches, int imode );

static void exchangeJ( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeexchangeJ( Params &params, VectorPatch &vecPatches );

static void exchangeA( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeexchangeA( Params &params, VectorPatch &vecPatches );
static void exchangeEnvEx( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeexchangeEnvEx( Params &params, VectorPatch &vecPatches );
static void exchangeGradPhi( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeexchangeGradPhi( Params &params, VectorPatch &vecPatches );
static void exchangeEnvChi( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi );

template<typename T, typename MT> static void exchangeAlongAllDirections( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeExchangeAlongAllDirections( std::vector<Field *> fields, VectorPatch &vecPatches );

template<typename T, typename MT> static void exchangeAlongAllDirectionsNoOMP( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeExchangeAlongAllDirectionsNoOMP( std::vector<Field *> fields, VectorPatch &vecPatches );

template<typename T, typename MT> static void exchangeSynchronizedPerDirection( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void exchangeSynchronizedPerDirection( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );

static void exchangeAllComponentsAlongX( std::vector<Field *> &fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeExchangeAllComponentsAlongX( std::vector<Field *> &fields, VectorPatch &vecPatches );
static void exchangeAllComponentsAlongY( std::vector<Field *> &fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeExchangeAllComponentsAlongY( std::vector<Field *> &fields, VectorPatch &vecPatches );
static void exchangeAllComponentsAlongZ( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeExchangeAllComponentsAlongZ( std::vector<Field *> fields, VectorPatch &vecPatches );

template<typename T, typename F> static void exchangeAlongX( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeExchangeAlongX( std::vector<Field *> fields, VectorPatch &vecPatches );
template<typename T, typename F> static void exchangeAlongY( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeExchangeAlongY( std::vector<Field *> fields, VectorPatch &vecPatches );
template<typename T, typename F> static void exchangeAlongZ( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
static void finalizeExchangeAlongZ( std::vector<Field *> fields, VectorPatch &vecPatches );

static void exchangeForPML( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi );

};

#endif
