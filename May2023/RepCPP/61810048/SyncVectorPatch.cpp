
#include "SyncVectorPatch.h"

#include <vector>

#include "VectorPatch.h"
#include "Params.h"
#include "SmileiMPI.h"

using namespace std;



template void SyncVectorPatch::exchangeAlongAllDirections<double,Field>( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
template void SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
template void SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double,Field>( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
template void SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );

void SyncVectorPatch::exchangeParticles( VectorPatch &vecPatches, int ispec, Params &params, SmileiMPI *smpi, Timers &timers, int itime )
{
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
Species *spec = vecPatches.species( ipatch, ispec );
spec->extractParticles();
vecPatches( ipatch )->initExchParticles( smpi, ispec, params );
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(runtime)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->exchNbrOfParticles( smpi, ispec, params, 0, &vecPatches );
}
}

void SyncVectorPatch::finalizeAndSortParticles( VectorPatch &vecPatches, int ispec, Params &params, SmileiMPI *smpi, Timers &timers, int itime )
{
SyncVectorPatch::finalizeExchangeParticles( vecPatches, ispec, 0, params, smpi, timers, itime );

for( unsigned int iDim=1 ; iDim<params.nDim_field ; iDim++ ) {
#ifndef _NO_MPI_TM
#pragma omp for schedule(runtime)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->exchNbrOfParticles( smpi, ispec, params, iDim, &vecPatches );
}

SyncVectorPatch::finalizeExchangeParticles( vecPatches, ispec, iDim, params, smpi, timers, itime );
}

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->importAndSortParticles( smpi, ispec, params, &vecPatches );
}




}


void SyncVectorPatch::finalizeExchangeParticles( VectorPatch &vecPatches, int ispec, int iDim, Params &params, SmileiMPI *smpi, Timers &timers, int itime )
{
#ifndef _NO_MPI_TM
#pragma omp for schedule(runtime)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->endNbrOfParticles( smpi, ispec, params, iDim, &vecPatches );
}

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->prepareParticles( smpi, ispec, params, iDim, &vecPatches );
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(runtime)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->exchParticles( smpi, ispec, params, iDim, &vecPatches );
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(runtime)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->finalizeExchParticles( smpi, ispec, params, iDim, &vecPatches );
}

#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->cornersParticles( smpi, ispec, params, iDim, &vecPatches );
}
}



void SyncVectorPatch::sumRhoJ( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi, Timers &timers, int itime )
{
SyncVectorPatch::sumAllComponents( vecPatches.densities, vecPatches, smpi, timers, itime );
if( ( vecPatches.diag_flag ) || ( params.is_spectral ) ) {
SyncVectorPatch::sum<double,Field>( vecPatches.listrho_, vecPatches, smpi, timers, itime );
}
}

void SyncVectorPatch::sumEnvChi( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi, Timers &timers, int itime )
{
SyncVectorPatch::sum<double,Field>( vecPatches.listEnv_Chi_, vecPatches, smpi, timers, itime );
}

void SyncVectorPatch::sumRhoJ( Params &params, VectorPatch &vecPatches, int imode, SmileiMPI *smpi, Timers &timers, int itime )
{
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listJl_[imode], vecPatches, smpi, timers, itime );
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listJr_[imode], vecPatches, smpi, timers, itime );
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listJt_[imode], vecPatches, smpi, timers, itime );
if( ( vecPatches.diag_flag ) || ( params.is_spectral ) ) {
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listrho_AM_[imode], vecPatches, smpi, timers, itime );
if (params.is_spectral)
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listrho_old_AM_[imode], vecPatches, smpi, timers, itime );
}
}

void SyncVectorPatch::sumRhoJs( Params &params, VectorPatch &vecPatches, int ispec, SmileiMPI *smpi, Timers &timers, int itime )
{
if( vecPatches.listJxs_ .size()>0 ) {
SyncVectorPatch::sum<double,Field>( vecPatches.listJxs_, vecPatches, smpi, timers, itime );
}
if( vecPatches.listJys_ .size()>0 ) {
SyncVectorPatch::sum<double,Field>( vecPatches.listJys_, vecPatches, smpi, timers, itime );
}
if( vecPatches.listJzs_ .size()>0 ) {
SyncVectorPatch::sum<double,Field>( vecPatches.listJzs_, vecPatches, smpi, timers, itime );
}
if( vecPatches.listrhos_.size()>0 ) {
SyncVectorPatch::sum<double,Field>( vecPatches.listrhos_, vecPatches, smpi, timers, itime );
}
}

void SyncVectorPatch::sumEnvChis( Params &params, VectorPatch &vecPatches, int ispec, SmileiMPI *smpi, Timers &timers, int itime )
{
if( vecPatches.listEnv_Chis_ .size()>0 ) {
SyncVectorPatch::sum<double,Field>( vecPatches.listEnv_Chis_, vecPatches, smpi, timers, itime );
}

}
void SyncVectorPatch::sumRhoJs( Params &params, VectorPatch &vecPatches, int imode, int ispec, SmileiMPI *smpi, Timers &timers, int itime )
{
if( vecPatches.listJls_[imode].size()>0 ) {
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listJls_[imode], vecPatches, smpi, timers, itime );
}
if( vecPatches.listJrs_[imode].size()>0 ) {
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listJrs_[imode], vecPatches, smpi, timers, itime );
}
if( vecPatches.listJts_[imode] .size()>0 ) {
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listJts_[imode], vecPatches, smpi, timers, itime );
}
if( vecPatches.listrhos_AM_[imode].size()>0 ) {
SyncVectorPatch::sum<complex<double>,cField>( vecPatches.listrhos_AM_[imode], vecPatches, smpi, timers, itime );
}
}

void SyncVectorPatch::sumAllComponents( std::vector<Field *> &fields, VectorPatch &vecPatches, SmileiMPI *smpi, Timers &timers, int itime )
{
unsigned int h0, oversize[3], n_space[3];
double *pt1, *pt2;
h0 = vecPatches( 0 )->hindex;

int nPatches( vecPatches.size() );

oversize[0] = vecPatches( 0 )->EMfields->oversize[0];
oversize[1] = vecPatches( 0 )->EMfields->oversize[1];
oversize[2] = vecPatches( 0 )->EMfields->oversize[2];

n_space[0] = vecPatches( 0 )->EMfields->n_space[0];
n_space[1] = vecPatches( 0 )->EMfields->n_space[1];
n_space[2] = vecPatches( 0 )->EMfields->n_space[2];

int nDim = vecPatches( 0 )->EMfields->Jx_->dims_.size();


unsigned int nPatchMPIx = vecPatches.MPIxIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nPatchMPIx ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIxIdx[ifield];
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, iNeighbor ) ) {
vecPatches.densitiesMPIx[ifield             ]->create_sub_fields ( 0, iNeighbor, 2*oversize[0]+1+1 ); 
vecPatches.densitiesMPIx[ifield+nPatchMPIx  ]->create_sub_fields ( 0, iNeighbor, 2*oversize[0]+1+0 ); 
vecPatches.densitiesMPIx[ifield+2*nPatchMPIx]->create_sub_fields ( 0, iNeighbor, 2*oversize[0]+1+0 ); 
vecPatches.densitiesMPIx[ifield             ]->extract_fields_sum( 0, iNeighbor, oversize[0] );
vecPatches.densitiesMPIx[ifield+nPatchMPIx  ]->extract_fields_sum( 0, iNeighbor, oversize[0] );
vecPatches.densitiesMPIx[ifield+2*nPatchMPIx]->extract_fields_sum( 0, iNeighbor, oversize[0] );
}
}
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIx[ifield             ], 0, smpi ); 
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIx[ifield+  nPatchMPIx], 0, smpi ); 
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIx[ifield+2*nPatchMPIx], 0, smpi ); 
}
int nFieldLocalx = vecPatches.densitiesLocalx.size()/3;
for( int icomp=0 ; icomp<3 ; icomp++ ) {
if( nFieldLocalx==0 ) {
continue;
}

unsigned int gsp[3];
unsigned int ny_ = 1;
unsigned int nz_ = 1;
if( nDim>1 ) {
ny_ = vecPatches.densitiesLocalx[icomp*nFieldLocalx]->dims_[1];
if( nDim>2 ) {
nz_ = vecPatches.densitiesLocalx[icomp*nFieldLocalx]->dims_[2];
}
}
gsp[0] = 1+2*oversize[0]+vecPatches.densitiesLocalx[icomp*nFieldLocalx]->isDual_[0]; 

unsigned int istart =  icomp   *nFieldLocalx;
unsigned int iend    = ( icomp+1 )*nFieldLocalx;
#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=istart ; ifield<iend ; ifield++ ) {
int ipatch = vecPatches.LocalxIdx[ ifield-icomp*nFieldLocalx ];
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[0][0] ) {
pt1 = &( fields[ vecPatches( ipatch )->neighbor_[0][0]-h0+icomp*nPatches ]->data_[n_space[0]*ny_*nz_] );
pt2 = &( vecPatches.densitiesLocalx[ifield]->data_[0] );
for( unsigned int i = 0; i < gsp[0]* ny_*nz_ ; i++ ) {
pt1[i] += pt2[i];
}
memcpy( pt2, pt1, gsp[0]*ny_*nz_*sizeof( double ) );
}
}
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nPatchMPIx ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIxIdx[ifield];
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIx[ifield             ], 0 ); 
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIx[ifield+nPatchMPIx  ], 0 ); 
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIx[ifield+2*nPatchMPIx], 0 ); 
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, ( iNeighbor+1 )%2 ) ) {
vecPatches.densitiesMPIx[ifield             ]->inject_fields_sum( 0, iNeighbor, oversize[0] );
vecPatches.densitiesMPIx[ifield+nPatchMPIx  ]->inject_fields_sum( 0, iNeighbor, oversize[0] );
vecPatches.densitiesMPIx[ifield+2*nPatchMPIx]->inject_fields_sum( 0, iNeighbor, oversize[0] );
}
}

}

if( nDim>1 ) {

unsigned int nPatchMPIy = vecPatches.MPIyIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nPatchMPIy ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIyIdx[ifield];
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, iNeighbor ) ) {
vecPatches.densitiesMPIy[ifield             ]->create_sub_fields ( 1, iNeighbor, 2*oversize[1]+1+0 ); 
vecPatches.densitiesMPIy[ifield+nPatchMPIy  ]->create_sub_fields ( 1, iNeighbor, 2*oversize[1]+1+1 ); 
vecPatches.densitiesMPIy[ifield+2*nPatchMPIy]->create_sub_fields ( 1, iNeighbor, 2*oversize[1]+1+0 ); 
vecPatches.densitiesMPIy[ifield             ]->extract_fields_sum( 1, iNeighbor, oversize[1] );
vecPatches.densitiesMPIy[ifield+nPatchMPIy  ]->extract_fields_sum( 1, iNeighbor, oversize[1] );
vecPatches.densitiesMPIy[ifield+2*nPatchMPIy]->extract_fields_sum( 1, iNeighbor, oversize[1] );
}
}
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIy[ifield             ], 1, smpi ); 
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIy[ifield+nPatchMPIy  ], 1, smpi ); 
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIy[ifield+2*nPatchMPIy], 1, smpi ); 
}

int nFieldLocaly = vecPatches.densitiesLocaly.size()/3;
for( int icomp=0 ; icomp<3 ; icomp++ ) {
if( nFieldLocaly==0 ) {
continue;
}

unsigned int gsp[3];
unsigned int nx_ =  vecPatches.densitiesLocaly[icomp*nFieldLocaly]->dims_[0];
unsigned int ny_ = 1;
unsigned int nz_ = 1;
if( nDim>1 ) {
ny_ = vecPatches.densitiesLocaly[icomp*nFieldLocaly]->dims_[1];
if( nDim>2 ) {
nz_ = vecPatches.densitiesLocaly[icomp*nFieldLocaly]->dims_[2];
}
}
gsp[0] = 1+2*oversize[0]+vecPatches.densitiesLocaly[icomp*nFieldLocaly]->isDual_[0]; 
gsp[1] = 1+2*oversize[1]+vecPatches.densitiesLocaly[icomp*nFieldLocaly]->isDual_[1]; 

unsigned int istart =  icomp   *nFieldLocaly;
unsigned int iend    = ( icomp+1 )*nFieldLocaly;
#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=istart ; ifield<iend ; ifield++ ) {
int ipatch = vecPatches.LocalyIdx[ ifield-icomp*nFieldLocaly ];
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[1][0] ) {
pt1 = &( fields[vecPatches( ipatch )->neighbor_[1][0]-h0+icomp*nPatches]->data_[n_space[1]*nz_] );
pt2 = &( vecPatches.densitiesLocaly[ifield]->data_[0] );
for( unsigned int j = 0; j < nx_ ; j++ ) {
for( unsigned int i = 0; i < gsp[1]*nz_ ; i++ ) {
pt1[i] += pt2[i];
}
memcpy( pt2, pt1, gsp[1]*nz_*sizeof( double ) );
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
for( unsigned int ifield=0 ; ifield<nPatchMPIy ; ifield=ifield+1 ) {
unsigned int ipatch = vecPatches.MPIyIdx[ifield];
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIy[ifield             ], 1 ); 
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIy[ifield+nPatchMPIy  ], 1 ); 
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIy[ifield+2*nPatchMPIy], 1 ); 
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, ( iNeighbor+1 )%2 ) ) {
vecPatches.densitiesMPIy[ifield             ]->inject_fields_sum( 1, iNeighbor, oversize[1] );
vecPatches.densitiesMPIy[ifield+nPatchMPIy  ]->inject_fields_sum( 1, iNeighbor, oversize[1] );
vecPatches.densitiesMPIy[ifield+2*nPatchMPIy]->inject_fields_sum( 1, iNeighbor, oversize[1] );
}
}
}

if( nDim>2 ) {

unsigned int nPatchMPIz = vecPatches.MPIzIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nPatchMPIz ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIzIdx[ifield];
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, iNeighbor ) ) {
vecPatches.densitiesMPIz[ifield             ]->create_sub_fields ( 2, iNeighbor, 2*oversize[2]+1+0 ); 
vecPatches.densitiesMPIz[ifield+nPatchMPIz  ]->create_sub_fields ( 2, iNeighbor, 2*oversize[2]+1+0 ); 
vecPatches.densitiesMPIz[ifield+2*nPatchMPIz]->create_sub_fields ( 2, iNeighbor, 2*oversize[2]+1+1 ); 
vecPatches.densitiesMPIz[ifield             ]->extract_fields_sum( 2, iNeighbor, oversize[2] );
vecPatches.densitiesMPIz[ifield+nPatchMPIz  ]->extract_fields_sum( 2, iNeighbor, oversize[2] );
vecPatches.densitiesMPIz[ifield+2*nPatchMPIz]->extract_fields_sum( 2, iNeighbor, oversize[2] );
}
}
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIz[ifield             ], 2, smpi ); 
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIz[ifield+nPatchMPIz  ], 2, smpi ); 
vecPatches( ipatch )->initSumField( vecPatches.densitiesMPIz[ifield+2*nPatchMPIz], 2, smpi ); 
}

int nFieldLocalz = vecPatches.densitiesLocalz.size()/3;
for( int icomp=0 ; icomp<3 ; icomp++ ) {
if( nFieldLocalz==0 ) {
continue;
}

unsigned int gsp[3];
unsigned int nx_ =  vecPatches.densitiesLocalz[icomp*nFieldLocalz]->dims_[0];
unsigned int ny_ = 1;
unsigned int nz_ = 1;
if( nDim>1 ) {
ny_ = vecPatches.densitiesLocalz[icomp*nFieldLocalz]->dims_[1];
if( nDim>2 ) {
nz_ = vecPatches.densitiesLocalz[icomp*nFieldLocalz]->dims_[2];
}
}
gsp[0] = 1+2*oversize[0]+vecPatches.densitiesLocalz[icomp*nFieldLocalz]->isDual_[0]; 
gsp[1] = 1+2*oversize[1]+vecPatches.densitiesLocalz[icomp*nFieldLocalz]->isDual_[1]; 
gsp[2] = 1+2*oversize[2]+vecPatches.densitiesLocalz[icomp*nFieldLocalz]->isDual_[2]; 

unsigned int istart  =  icomp   *nFieldLocalz;
unsigned int iend    = ( icomp+1 )*nFieldLocalz;
#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=istart ; ifield<iend ; ifield++ ) {
int ipatch = vecPatches.LocalzIdx[ ifield-icomp*nFieldLocalz ];
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[2][0] ) {
pt1 = &( fields[vecPatches( ipatch )->neighbor_[2][0]-h0+icomp*nPatches]->data_[n_space[2]] );
pt2 = &( vecPatches.densitiesLocalz[ifield]->data_[0] );
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
for( unsigned int ifield=0 ; ifield<nPatchMPIz ; ifield=ifield+1 ) {
unsigned int ipatch = vecPatches.MPIzIdx[ifield];
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIz[ifield             ], 2 ); 
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIz[ifield+nPatchMPIz  ], 2 ); 
vecPatches( ipatch )->finalizeSumField( vecPatches.densitiesMPIz[ifield+2*nPatchMPIz], 2 ); 
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, ( iNeighbor+1 )%2 ) ) {
vecPatches.densitiesMPIz[ifield             ]->inject_fields_sum( 2, iNeighbor, oversize[2] );
vecPatches.densitiesMPIz[ifield+nPatchMPIz  ]->inject_fields_sum( 2, iNeighbor, oversize[2] );
vecPatches.densitiesMPIz[ifield+2*nPatchMPIz]->inject_fields_sum( 2, iNeighbor, oversize[2] );
}
}
}

} 

} 

}



void SyncVectorPatch::exchangeE( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi )
{

if( !params.full_B_exchange ) {
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listEx_, vecPatches, smpi );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listEy_, vecPatches, smpi );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listEz_, vecPatches, smpi );
} else {
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( vecPatches.listEx_, vecPatches, smpi );
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( vecPatches.listEy_, vecPatches, smpi );
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( vecPatches.listEz_, vecPatches, smpi );
}

}

void SyncVectorPatch::finalizeexchangeE( Params &params, VectorPatch &vecPatches )
{

if( !params.full_B_exchange ) {
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listEx_, vecPatches );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listEy_, vecPatches );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listEz_, vecPatches );
}
}

void SyncVectorPatch::exchangeB( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi )
{

if( vecPatches.listBx_[0]->dims_.size()==1 ) {
SyncVectorPatch::exchangeAllComponentsAlongX( vecPatches.Bs0, vecPatches, smpi );
} else {
if( params.full_B_exchange ) {
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( vecPatches.listBx_, vecPatches, smpi );
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( vecPatches.listBy_, vecPatches, smpi );
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( vecPatches.listBz_, vecPatches, smpi );

} else {
if( vecPatches.listBx_[0]->dims_.size()==2 ) {
SyncVectorPatch::exchangeAllComponentsAlongX( vecPatches.Bs0, vecPatches, smpi );
SyncVectorPatch::exchangeAllComponentsAlongY( vecPatches.Bs1, vecPatches, smpi );
} else if( vecPatches.listBx_[0]->dims_.size()==3 ) {
SyncVectorPatch::exchangeAllComponentsAlongX( vecPatches.Bs0, vecPatches, smpi );
SyncVectorPatch::exchangeAllComponentsAlongY( vecPatches.Bs1, vecPatches, smpi );
SyncVectorPatch::exchangeAllComponentsAlongZ( vecPatches.Bs2, vecPatches, smpi );
}
}
}
}

void SyncVectorPatch::finalizeexchangeB( Params &params, VectorPatch &vecPatches )
{

if( vecPatches.listBx_[0]->dims_.size()==1 ) {
SyncVectorPatch::finalizeExchangeAllComponentsAlongX( vecPatches.Bs0, vecPatches );
} else if( vecPatches.listBx_[0]->dims_.size()==2 ) {
if( !params.full_B_exchange ) {
SyncVectorPatch::finalizeExchangeAllComponentsAlongX( vecPatches.Bs0, vecPatches );
SyncVectorPatch::finalizeExchangeAllComponentsAlongY( vecPatches.Bs1, vecPatches );
}
} else if( vecPatches.listBx_[0]->dims_.size()==3 ) {
if( !params.full_B_exchange ) {
SyncVectorPatch::finalizeExchangeAllComponentsAlongX( vecPatches.Bs0, vecPatches );
SyncVectorPatch::finalizeExchangeAllComponentsAlongY( vecPatches.Bs1, vecPatches );
SyncVectorPatch::finalizeExchangeAllComponentsAlongZ( vecPatches.Bs2, vecPatches );
}
}

}

void SyncVectorPatch::exchangeJ( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi )
{

SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listJx_, vecPatches, smpi );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listJy_, vecPatches, smpi );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listJz_, vecPatches, smpi );
}

void SyncVectorPatch::finalizeexchangeJ( Params &params, VectorPatch &vecPatches )
{

SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listJx_, vecPatches );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listJy_, vecPatches );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listJz_, vecPatches );
}


void SyncVectorPatch::exchangeB( Params &params, VectorPatch &vecPatches, int imode, SmileiMPI *smpi )
{
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( vecPatches.listBl_[imode], vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listBl_[imode], vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( vecPatches.listBr_[imode], vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listBr_[imode], vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( vecPatches.listBt_[imode], vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listBt_[imode], vecPatches );
}

void SyncVectorPatch::exchangeE( Params &params, VectorPatch &vecPatches, int imode, SmileiMPI *smpi )
{
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( vecPatches.listEl_[imode], vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listEl_[imode], vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( vecPatches.listEr_[imode], vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listEr_[imode], vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( vecPatches.listEt_[imode], vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listEt_[imode], vecPatches );
}

void SyncVectorPatch::finalizeexchangeB( Params &params, VectorPatch &vecPatches, int imode )
{
}

void SyncVectorPatch::finalizeexchangeE( Params &params, VectorPatch &vecPatches, int imode )
{
}




void SyncVectorPatch::exchangeA( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi )
{
if( !params.full_Envelope_exchange ) {
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( vecPatches.listA_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listA_, vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<complex<double>,cField>( vecPatches.listA0_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listA0_, vecPatches );
} else {
SyncVectorPatch::exchangeSynchronizedPerDirection<complex<double>,cField>( vecPatches.listA_, vecPatches, smpi );
SyncVectorPatch::exchangeSynchronizedPerDirection<complex<double>,cField>( vecPatches.listA0_, vecPatches, smpi );  
}
}

void SyncVectorPatch::finalizeexchangeA( Params &params, VectorPatch &vecPatches )
{
}


void SyncVectorPatch::exchangeEnvEx( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi )
{
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listEnvEx_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listEnvEx_, vecPatches );
}

void SyncVectorPatch::finalizeexchangeEnvEx( Params &params, VectorPatch &vecPatches )
{

}





void SyncVectorPatch::exchangeGradPhi( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi )
{
if (  params.geometry != "AMcylindrical" ) {
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhix_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhix_, vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhiy_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhiy_, vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhiz_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhiz_, vecPatches );

SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhix0_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhix0_, vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhiy0_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhiy0_, vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhiz0_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhiz0_, vecPatches );
} else {
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhil_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhil_, vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhir_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhir_, vecPatches );

SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhil0_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhil0_, vecPatches );
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listGradPhir0_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listGradPhir0_, vecPatches );
}
}

void SyncVectorPatch::finalizeexchangeGradPhi( Params &params, VectorPatch &vecPatches )
{
}

void SyncVectorPatch::exchangeEnvChi( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi )
{
if( !params.full_Envelope_exchange ) {
SyncVectorPatch::exchangeAlongAllDirections<double,Field>( vecPatches.listEnv_Chi_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongAllDirections( vecPatches.listEnv_Chi_, vecPatches );

} else {
SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( vecPatches.listEnv_Chi_, vecPatches, smpi );
}
}


void SyncVectorPatch::templateGenerator()
{
SmileiMPI* smpi = NULL;
VectorPatch patches;
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double         ,Field >( patches.listEx_, patches, smpi );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<complex<double>,cField>( patches.listEx_, patches, smpi );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double         ,Field >( patches.listEx_, patches, smpi );
SyncVectorPatch::exchangeAlongAllDirectionsNoOMP<double         ,Field >( patches.listEx_, patches, smpi );
}

template<typename T, typename F>
void SyncVectorPatch::exchangeAlongAllDirections( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{
unsigned int  oversize[3];
oversize[0] = vecPatches( 0 )->EMfields->oversize[0];
oversize[1] = vecPatches( 0 )->EMfields->oversize[1];
oversize[2] = vecPatches( 0 )->EMfields->oversize[2];

for( unsigned int iDim=0 ; iDim<fields[0]->dims_.size() ; iDim++ ) {
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( iDim, iNeighbor ) ) {
fields[ipatch]->create_sub_fields  ( iDim, iNeighbor, oversize[iDim] );
fields[ipatch]->extract_fields_exch( iDim, iNeighbor, oversize[iDim] );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initExchange       ( fields[ipatch], iDim, smpi );
else
vecPatches( ipatch )->initExchangeComplex( fields[ipatch], iDim, smpi );
}
} 

unsigned int nx_, ny_( 1 ), nz_( 1 ), h0, n_space[3], gsp[3];
T *pt1, *pt2;
F *field1, *field2;
h0 = vecPatches( 0 )->hindex;

n_space[0] = vecPatches( 0 )->EMfields->n_space[0];
n_space[1] = vecPatches( 0 )->EMfields->n_space[1];
n_space[2] = vecPatches( 0 )->EMfields->n_space[2];

nx_ = fields[0]->dims_[0];
if( fields[0]->dims_.size()>1 ) {
ny_ = fields[0]->dims_[1];
if( fields[0]->dims_.size()>2 ) {
nz_ = fields[0]->dims_[2];
}
}


gsp[0] = ( oversize[0] + 1 + fields[0]->isDual_[0] ); 

#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {

if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[0][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[0][0]-h0] );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( ( n_space[0] )*ny_*nz_ );
pt2 = &( *field2 )( 0 );
memcpy( pt2, pt1, oversize[0]*ny_*nz_*sizeof( T ) );
memcpy( pt1+gsp[0]*ny_*nz_, pt2+gsp[0]*ny_*nz_, oversize[0]*ny_*nz_*sizeof( T ) );
} 

if( fields[0]->dims_.size()>1 ) {
gsp[1] = ( oversize[1] + 1 + fields[0]->isDual_[1] ); 
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[1][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[1][0]-h0] );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( n_space[1]*nz_ );
pt2 = &( *field2 )( 0 );
for( unsigned int i = 0 ; i < nx_*ny_*nz_ ; i += ny_*nz_ ) {
for( unsigned int j = 0 ; j < oversize[1]*nz_ ; j++ ) {
pt2[i+j] = pt1[i+j] ;
pt1[i+j+gsp[1]*nz_] = pt2[i+j+gsp[1]*nz_] ;
}
}
} 

if( fields[0]->dims_.size()>2 ) {
gsp[2] = ( oversize[2] + 1 + fields[0]->isDual_[2] ); 
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[2][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[2][0]-h0] );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( n_space[2] );
pt2 = &( *field2 )( 0 );
for( unsigned int i = 0 ; i < nx_*ny_*nz_ ; i += ny_*nz_ ) {
for( unsigned int j = 0 ; j < ny_*nz_ ; j += nz_ ) {
for( unsigned int k = 0 ; k < oversize[2] ; k++ ) {
pt2[i+j+k] = pt1[i+j+k] ;
pt1[i+j+k+gsp[2]] = pt2[i+j+k+gsp[2]] ;
}
}
}
}
}
} 
} 

}

void SyncVectorPatch::finalizeExchangeAlongAllDirections( std::vector<Field *> fields, VectorPatch &vecPatches )
{
unsigned oversize[3];
oversize[0] = vecPatches( 0 )->EMfields->oversize[0];
oversize[1] = vecPatches( 0 )->EMfields->oversize[1];
oversize[2] = vecPatches( 0 )->EMfields->oversize[2];

for( unsigned int iDim=0 ; iDim<fields[0]->dims_.size() ; iDim++ ) {
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
vecPatches( ipatch )->finalizeExchange( fields[ipatch], iDim );

for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( iDim, ( iNeighbor+1 )%2 ) ) {
fields[ipatch]->inject_fields_exch( iDim, iNeighbor, oversize[iDim] );
}
}
}
} 

}


template<typename T, typename F>
void SyncVectorPatch::exchangeAlongAllDirectionsNoOMP( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{
unsigned oversize[3];
oversize[0] = vecPatches( 0 )->EMfields->oversize[0];
oversize[1] = vecPatches( 0 )->EMfields->oversize[1];
oversize[2] = vecPatches( 0 )->EMfields->oversize[2];

for( unsigned int iDim=0 ; iDim<fields[0]->dims_.size() ; iDim++ ) {
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( iDim, iNeighbor ) ) {
fields[ipatch]->create_sub_fields  ( iDim, iNeighbor, oversize[iDim] );
fields[ipatch]->extract_fields_exch( iDim, iNeighbor, oversize[iDim] );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initExchange       ( fields[ipatch], iDim, smpi );
else
vecPatches( ipatch )->initExchangeComplex( fields[ipatch], iDim, smpi );
}
} 


unsigned int nx_, ny_( 1 ), nz_( 1 ), h0, n_space[3], gsp[3];
T *pt1, *pt2;
F* field1;
F* field2;
h0 = vecPatches( 0 )->hindex;

n_space[0] = vecPatches( 0 )->EMfields->n_space[0];
n_space[1] = vecPatches( 0 )->EMfields->n_space[1];
n_space[2] = vecPatches( 0 )->EMfields->n_space[2];

nx_ = fields[0]->dims_[0];
if( fields[0]->dims_.size()>1 ) {
ny_ = fields[0]->dims_[1];
if( fields[0]->dims_.size()>2 ) {
nz_ = fields[0]->dims_[2];
}
}

gsp[0] = ( oversize[0] + 1 + fields[0]->isDual_[0] ); 

for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {

if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[0][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[0][0]-h0] );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( n_space[0]*ny_*nz_ );
pt2 = &( *field2 )( 0 );
memcpy( pt2, pt1, oversize[0]*ny_*nz_*sizeof( T ) );
memcpy( pt1+gsp[0]*ny_*nz_, pt2+gsp[0]*ny_*nz_, oversize[0]*ny_*nz_*sizeof( T ) );
} 

if( fields[0]->dims_.size()>1 ) {
gsp[1] = ( oversize[1] + 1 + fields[0]->isDual_[1] ); 
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[1][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[1][0]-h0] );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( n_space[1]*nz_ );
pt2 = &( *field2 )( 0 );
for( unsigned int i = 0 ; i < nx_*ny_*nz_ ; i += ny_*nz_ ) {
for( unsigned int j = 0 ; j < oversize[1]*nz_ ; j++ ) {
pt2[i+j] = pt1[i+j] ;
pt1[i+j+gsp[1]*nz_] = pt2[i+j+gsp[1]*nz_] ;
}
}
} 

if( fields[0]->dims_.size()>2 ) {
gsp[2] = ( oversize[2] + 1 + fields[0]->isDual_[2] ); 
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[2][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[2][0]-h0] );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( n_space[2] );
pt2 = &( *field2 )( 0 );
for( unsigned int i = 0 ; i < nx_*ny_*nz_ ; i += ny_*nz_ ) {
for( unsigned int j = 0 ; j < ny_*nz_ ; j += nz_ ) {
for( unsigned int k = 0 ; k < oversize[2] ; k++ ) {
pt2[i+j+k] = pt1[i+j+k] ;
pt1[i+j+k+gsp[2]] = pt2[i+j+k+gsp[2]] ;
}
}
}
}
}
} 
} 

}


void SyncVectorPatch::finalizeExchangeAlongAllDirectionsNoOMP( std::vector<Field *> fields, VectorPatch &vecPatches )
{
unsigned oversize[3];
oversize[0] = vecPatches( 0 )->EMfields->oversize[0];
oversize[1] = vecPatches( 0 )->EMfields->oversize[1];
oversize[2] = vecPatches( 0 )->EMfields->oversize[2];

for( unsigned int iDim=0 ; iDim<fields[0]->dims_.size() ; iDim++ ) {
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
vecPatches( ipatch )->finalizeExchange( fields[ipatch], iDim );

for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( iDim, ( iNeighbor+1 )%2 ) ) {
fields[ipatch]->inject_fields_exch( iDim, iNeighbor, oversize[iDim] );
}
}

}
} 

}


template void SyncVectorPatch::exchangeSynchronizedPerDirection<double,Field>( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );
template void SyncVectorPatch::exchangeSynchronizedPerDirection<complex<double>,cField>( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi );

template<typename T, typename F>
void SyncVectorPatch::exchangeSynchronizedPerDirection( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{

unsigned int nx_, ny_( 1 ), nz_( 1 ), h0, oversize[3], n_space[3], gsp[3];
T *pt1, *pt2;
F* field1;
F* field2;
h0 = vecPatches( 0 )->hindex;

oversize[0] = vecPatches( 0 )->EMfields->oversize[0];
oversize[1] = vecPatches( 0 )->EMfields->oversize[1];
oversize[2] = vecPatches( 0 )->EMfields->oversize[2];

n_space[0] = vecPatches( 0 )->EMfields->n_space[0];
n_space[1] = vecPatches( 0 )->EMfields->n_space[1];
n_space[2] = vecPatches( 0 )->EMfields->n_space[2];

nx_ = fields[0]->dims_[0];
if( fields[0]->dims_.size()>1 ) {
ny_ = fields[0]->dims_[1];
if( fields[0]->dims_.size()>2 ) {
nz_ = fields[0]->dims_[2];
}
}

if( fields[0]->dims_.size()>2 ) {

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, iNeighbor ) ) {
fields[ipatch]->create_sub_fields  ( 2, iNeighbor, oversize[2] );
fields[ipatch]->extract_fields_exch( 2, iNeighbor, oversize[2] );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initExchange( fields[ipatch], 2, smpi );
else
vecPatches( ipatch )->initExchangeComplex( fields[ipatch], 2, smpi );
}


#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
vecPatches( ipatch )->finalizeExchange( fields[ipatch], 2 );

for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, ( iNeighbor+1 )%2 ) ) {
fields[ipatch]->inject_fields_exch( 2, iNeighbor, oversize[2] );
}
}
}

#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {

gsp[2] = ( oversize[2] + 1 + fields[0]->isDual_[2] ); 
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[2][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[2][0]-h0]  );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( n_space[2] );
pt2 = &( *field2 )( 0 );
for( unsigned int in = 0 ; in < nx_ ; in ++ ) {
unsigned int i = in * ny_*nz_;
for( unsigned int jn = 0 ; jn < ny_ ; jn ++ ) {
unsigned int j = jn *nz_;
for( unsigned int k = 0 ; k < oversize[2] ; k++ ) {
pt2[i+j+k] = pt1[i+j+k] ;
pt1[i+j+k+gsp[2]] = pt2[i+j+k+gsp[2]] ;
}
}
}
}

} 
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, iNeighbor ) ) {
fields[ipatch]->create_sub_fields  ( 1, iNeighbor, oversize[1] );
fields[ipatch]->extract_fields_exch( 1, iNeighbor, oversize[1] );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initExchange( fields[ipatch], 1, smpi );
else
vecPatches( ipatch )->initExchangeComplex( fields[ipatch], 1, smpi );
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
vecPatches( ipatch )->finalizeExchange( fields[ipatch], 1 );

for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, ( iNeighbor+1 )%2 ) ) {
fields[ipatch]->inject_fields_exch( 1, iNeighbor, oversize[1] );
}
}
}

#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {

gsp[1] = ( oversize[1] + 1 + fields[0]->isDual_[1] ); 
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[1][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[1][0]-h0]  );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( n_space[1]*nz_ );
pt2 = &( *field2 )( 0 );
for( unsigned int in = 0 ; in < nx_ ; in ++ ) {
unsigned int i = in * ny_*nz_;
for( unsigned int j = 0 ; j < oversize[1]*nz_ ; j++ ) {
pt2[i+j] = pt1[i+j] ;
pt1[i+j+gsp[1]*nz_] = pt2[i+j+gsp[1]*nz_] ;
}
}
} 

} 

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, iNeighbor ) ) {
fields[ipatch]->create_sub_fields  ( 0, iNeighbor, oversize[0] );
fields[ipatch]->extract_fields_exch( 0, iNeighbor, oversize[0] );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initExchange( fields[ipatch], 0, smpi );
else
vecPatches( ipatch )->initExchangeComplex( fields[ipatch], 0, smpi );
}

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
vecPatches( ipatch )->finalizeExchange( fields[ipatch], 0 );

for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, ( iNeighbor+1 )%2 ) ) {
fields[ipatch]->inject_fields_exch( 0, iNeighbor, oversize[0] );
}
}
}



gsp[0] = ( oversize[0] + 1 + fields[0]->isDual_[0] ); 

#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {

if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[0][0] ) {
field1 = static_cast<F *>( fields[vecPatches( ipatch )->neighbor_[0][0]-h0]  );
field2 = static_cast<F *>( fields[ipatch] );
pt1 = &( *field1 )( ( n_space[0] )*ny_*nz_ );
pt2 = &( *field2 )( 0 );
memcpy( pt2, pt1, oversize[0]*ny_*nz_*sizeof( T ) );
memcpy( pt1+gsp[0]*ny_*nz_, pt2+gsp[0]*ny_*nz_, oversize[0]*ny_*nz_*sizeof( T ) );
} 

} 

}


void SyncVectorPatch::exchangeAllComponentsAlongX( std::vector<Field *> &fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[0];

unsigned int nMPIx = vecPatches.MPIxIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nMPIx ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIxIdx[ifield];
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, iNeighbor ) ) {
vecPatches.B_MPIx[ifield      ]->create_sub_fields  ( 0, iNeighbor, oversize );
vecPatches.B_MPIx[ifield      ]->extract_fields_exch( 0, iNeighbor, oversize );
vecPatches.B_MPIx[ifield+nMPIx]->create_sub_fields  ( 0, iNeighbor, oversize );
vecPatches.B_MPIx[ifield+nMPIx]->extract_fields_exch( 0, iNeighbor, oversize );
}
}
vecPatches( ipatch )->initExchange( vecPatches.B_MPIx[ifield      ], 0, smpi ); 
vecPatches( ipatch )->initExchange( vecPatches.B_MPIx[ifield+nMPIx], 0, smpi ); 
}

unsigned int h0, n_space;
double *pt1, *pt2;
h0 = vecPatches( 0 )->hindex;

n_space = vecPatches( 0 )->EMfields->n_space[0];

int nPatches( vecPatches.size() );
int nDim = vecPatches( 0 )->EMfields->Bx_->dims_.size();

int nFieldLocalx = vecPatches.B_localx.size()/2;
for( int icomp=0 ; icomp<2 ; icomp++ ) {
if( nFieldLocalx==0 ) {
continue;
}

unsigned int ny_( 1 ), nz_( 1 ), gsp;
if( nDim>1 ) {
ny_ = vecPatches.B_localx[icomp*nFieldLocalx]->dims_[1];
if( nDim>2 ) {
nz_ = vecPatches.B_localx[icomp*nFieldLocalx]->dims_[2];
}
}
gsp = ( oversize + 1 + vecPatches.B_localx[icomp*nFieldLocalx]->isDual_[0] ); 

unsigned int istart =  icomp   *nFieldLocalx;
unsigned int iend    = ( icomp+1 )*nFieldLocalx;
#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=istart ; ifield<iend ; ifield++ ) {
int ipatch = vecPatches.LocalxIdx[ ifield-icomp*nFieldLocalx ];

if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[0][0] ) {
pt1 = &( fields[vecPatches( ipatch )->neighbor_[0][0]-h0+icomp*nPatches]->data_[n_space*ny_*nz_] );
pt2 = &( vecPatches.B_localx[ifield]->data_[0] );
memcpy( pt2, pt1, oversize*ny_*nz_*sizeof( double ) );
memcpy( pt1+gsp*ny_*nz_, pt2+gsp*ny_*nz_, oversize*ny_*nz_*sizeof( double ) );
} 

} 
}

}

void SyncVectorPatch::finalizeExchangeAllComponentsAlongX( std::vector<Field *> &fields, VectorPatch &vecPatches )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[0];

unsigned int nMPIx = vecPatches.MPIxIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nMPIx ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIxIdx[ifield];
vecPatches( ipatch )->finalizeExchange( vecPatches.B_MPIx[ifield      ], 0 ); 
vecPatches( ipatch )->finalizeExchange( vecPatches.B_MPIx[ifield+nMPIx], 0 ); 
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, ( iNeighbor+1 )%2 ) ) {
vecPatches.B_MPIx[ifield      ]->inject_fields_exch( 0, iNeighbor, oversize );
vecPatches.B_MPIx[ifield+nMPIx]->inject_fields_exch( 0, iNeighbor, oversize );
}
}
}

}


void SyncVectorPatch::exchangeAllComponentsAlongY( std::vector<Field *> &fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[1];

unsigned int nMPIy = vecPatches.MPIyIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nMPIy ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIyIdx[ifield];
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, iNeighbor ) ) {
vecPatches.B1_MPIy[ifield      ]->create_sub_fields  ( 1, iNeighbor, oversize );
vecPatches.B1_MPIy[ifield      ]->extract_fields_exch( 1, iNeighbor, oversize );
vecPatches.B1_MPIy[ifield+nMPIy]->create_sub_fields  ( 1, iNeighbor, oversize );
vecPatches.B1_MPIy[ifield+nMPIy]->extract_fields_exch( 1, iNeighbor, oversize );
}
}
vecPatches( ipatch )->initExchange( vecPatches.B1_MPIy[ifield      ], 1, smpi ); 
vecPatches( ipatch )->initExchange( vecPatches.B1_MPIy[ifield+nMPIy], 1, smpi ); 
}

unsigned int h0, n_space;
double *pt1, *pt2;
h0 = vecPatches( 0 )->hindex;

n_space = vecPatches( 0 )->EMfields->n_space[1];

int nPatches( vecPatches.size() );
int nDim = vecPatches( 0 )->EMfields->Bx_->dims_.size();

int nFieldLocaly = vecPatches.B1_localy.size()/2;
for( int icomp=0 ; icomp<2 ; icomp++ ) {
if( nFieldLocaly==0 ) {
continue;
}

unsigned int nx_, ny_, nz_( 1 ), gsp;
nx_ = vecPatches.B1_localy[icomp*nFieldLocaly]->dims_[0];
ny_ = vecPatches.B1_localy[icomp*nFieldLocaly]->dims_[1];
if( nDim>2 ) {
nz_ = vecPatches.B1_localy[icomp*nFieldLocaly]->dims_[2];
}
gsp = ( oversize + 1 + vecPatches.B1_localy[icomp*nFieldLocaly]->isDual_[1] ); 

unsigned int istart =  icomp   *nFieldLocaly;
unsigned int iend    = ( icomp+1 )*nFieldLocaly;
#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=istart ; ifield<iend ; ifield++ ) {

int ipatch = vecPatches.LocalyIdx[ ifield-icomp*nFieldLocaly ];
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[1][0] ) {
pt1 = &( fields[vecPatches( ipatch )->neighbor_[1][0]-h0+icomp*nPatches]->data_[n_space*nz_] );
pt2 = &( vecPatches.B1_localy[ifield]->data_[0] );
for( unsigned int i = 0 ; i < nx_*ny_*nz_ ; i += ny_*nz_ ) {
for( unsigned int j = 0 ; j < oversize*nz_ ; j++ ) {
pt2[i+j] = pt1[i+j] ;
pt1[i+j+gsp*nz_] = pt2[i+j+gsp*nz_] ;
} 
}
} 

} 
}
}


void SyncVectorPatch::finalizeExchangeAllComponentsAlongY( std::vector<Field *> &fields, VectorPatch &vecPatches )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[1];

unsigned int nMPIy = vecPatches.MPIyIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nMPIy ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIyIdx[ifield];
vecPatches( ipatch )->finalizeExchange( vecPatches.B1_MPIy[ifield      ], 1 ); 
vecPatches( ipatch )->finalizeExchange( vecPatches.B1_MPIy[ifield+nMPIy], 1 ); 
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, ( iNeighbor+1 )%2 ) ) {
vecPatches.B1_MPIy[ifield      ]->inject_fields_exch( 1, iNeighbor, oversize );
vecPatches.B1_MPIy[ifield+nMPIy]->inject_fields_exch( 1, iNeighbor, oversize );
}
}
}

}


void SyncVectorPatch::exchangeAllComponentsAlongZ( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[2];

unsigned int nMPIz = vecPatches.MPIzIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nMPIz ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIzIdx[ifield];
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, iNeighbor ) ) {
vecPatches.B2_MPIz[ifield      ]->create_sub_fields  ( 2, iNeighbor, oversize );
vecPatches.B2_MPIz[ifield      ]->extract_fields_exch( 2, iNeighbor, oversize );
vecPatches.B2_MPIz[ifield+nMPIz]->create_sub_fields  ( 2, iNeighbor, oversize );
vecPatches.B2_MPIz[ifield+nMPIz]->extract_fields_exch( 2, iNeighbor, oversize );
}
}
vecPatches( ipatch )->initExchange( vecPatches.B2_MPIz[ifield],       2, smpi ); 
vecPatches( ipatch )->initExchange( vecPatches.B2_MPIz[ifield+nMPIz], 2, smpi ); 
}

unsigned int h0, n_space;
double *pt1, *pt2;
h0 = vecPatches( 0 )->hindex;

n_space = vecPatches( 0 )->EMfields->n_space[2];

int nPatches( vecPatches.size() );

int nFieldLocalz = vecPatches.B2_localz.size()/2;
for( int icomp=0 ; icomp<2 ; icomp++ ) {
if( nFieldLocalz==0 ) {
continue;
}

unsigned int nx_, ny_, nz_, gsp;
nx_ = vecPatches.B2_localz[icomp*nFieldLocalz]->dims_[0];
ny_ = vecPatches.B2_localz[icomp*nFieldLocalz]->dims_[1];
nz_ = vecPatches.B2_localz[icomp*nFieldLocalz]->dims_[2];
gsp = ( oversize + 1 + vecPatches.B2_localz[icomp*nFieldLocalz]->isDual_[2] ); 

unsigned int istart  =  icomp   *nFieldLocalz;
unsigned int iend    = ( icomp+1 )*nFieldLocalz;
#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ifield=istart ; ifield<iend ; ifield++ ) {

int ipatch = vecPatches.LocalzIdx[ ifield-icomp*nFieldLocalz ];
if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[2][0] ) {
pt1 = &( fields[vecPatches( ipatch )->neighbor_[2][0]-h0+icomp*nPatches]->data_[n_space] );
pt2 = &( vecPatches.B2_localz[ifield]->data_[0] );
for( unsigned int i = 0 ; i < nx_*ny_*nz_ ; i += ny_*nz_ ) {
for( unsigned int j = 0 ; j < ny_*nz_ ; j += nz_ ) {
for( unsigned int k = 0 ; k < oversize ; k++ ) {
pt2[i+j+k] = pt1[i+j+k] ;
pt1[i+j+k+gsp] = pt2[i+j+k+gsp] ;
}
}
}
} 

} 
}
}

void SyncVectorPatch::finalizeExchangeAllComponentsAlongZ( std::vector<Field *> fields, VectorPatch &vecPatches )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[2];

unsigned int nMPIz = vecPatches.MPIzIdx.size();
#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ifield=0 ; ifield<nMPIz ; ifield++ ) {
unsigned int ipatch = vecPatches.MPIzIdx[ifield];
vecPatches( ipatch )->finalizeExchange( vecPatches.B2_MPIz[ifield      ], 2 ); 
vecPatches( ipatch )->finalizeExchange( vecPatches.B2_MPIz[ifield+nMPIz], 2 ); 
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, ( iNeighbor+1 )%2 ) ) {
vecPatches.B2_MPIz[ifield      ]->inject_fields_exch( 2, iNeighbor, oversize );
vecPatches.B2_MPIz[ifield+nMPIz]->inject_fields_exch( 2, iNeighbor, oversize );
}
}
}

}



template<typename T, typename F>
void SyncVectorPatch::exchangeAlongX( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[0];

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
if (fields[ipatch]!=NULL) {
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, iNeighbor ) ) {
fields[ipatch]->create_sub_fields  ( 0, iNeighbor, oversize );
fields[ipatch]->extract_fields_exch( 0, iNeighbor, oversize );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initExchange( fields[ipatch], 0, smpi );
else
vecPatches( ipatch )->initExchangeComplex( fields[ipatch], 0, smpi );
}
}

unsigned int ny_( 1 ), nz_( 1 ), h0, neighbour_n_space, gsp;
T *pt1, *pt2;
h0 = vecPatches( 0 )->hindex;

int IPATCH = 0;
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ )
if (fields[ipatch]!=NULL) {
IPATCH = ipatch;
break;
}

if( fields[IPATCH]==NULL ) return;

gsp = ( oversize + 1 + fields[IPATCH]->isDual_[0] ); 

#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {

if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[0][0] ) {
if (fields[ipatch]!=NULL) {
Field* f = fields[vecPatches( ipatch )->neighbor_[0][0]-h0];
neighbour_n_space = f->dims_[0] - 1 - 2*oversize - f->isDual_[0]; 
if( fields[IPATCH]->dims_.size()>1 ) {
ny_ = f->dims_[1]; 
if( fields[IPATCH]->dims_.size()>2 ) {
nz_ = f->dims_[2]; 
}
}
pt1 = &( *static_cast<F*>( f              ) )( neighbour_n_space*ny_*nz_ ); 
pt2 = &( *static_cast<F*>( fields[ipatch] ) )( 0 ); 

memcpy( pt2, pt1, oversize*ny_*nz_*sizeof( T ) ); 
memcpy( pt1+gsp*ny_*nz_, pt2+gsp*ny_*nz_, oversize*ny_*nz_*sizeof( T ) );


} 
}
} 

}

void SyncVectorPatch::finalizeExchangeAlongX( std::vector<Field *> fields, VectorPatch &vecPatches )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[0];

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
if (fields[ipatch]!=NULL) {
vecPatches( ipatch )->finalizeExchange( fields[ipatch], 0 );
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 0, ( iNeighbor+1 )%2 ) ) {
fields[ipatch]->inject_fields_exch( 0, iNeighbor, oversize );
}
}
}
}

}

template<typename T, typename F>
void SyncVectorPatch::exchangeAlongY( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[1];

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
if (fields[ipatch]!=NULL) {
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, iNeighbor ) ) {
fields[ipatch]->create_sub_fields  ( 1, iNeighbor, oversize );
fields[ipatch]->extract_fields_exch( 1, iNeighbor, oversize );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initExchange( fields[ipatch], 1, smpi );
else
vecPatches( ipatch )->initExchangeComplex( fields[ipatch], 1, smpi );
}
}

unsigned int nx_, ny_, nz_( 1 ), h0, gsp, neighbour_n_space, my_ny;
T *pt1, *pt2;
h0 = vecPatches( 0 )->hindex;

int IPATCH = 0;
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ )
if (fields[ipatch]!=NULL) {
IPATCH = ipatch;
break;
}

if( fields[IPATCH]==NULL ) return;

gsp = ( oversize + 1 + fields[IPATCH]->isDual_[1] ); 

#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {

if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[1][0] ) {
if (fields[ipatch]!=NULL) {
Field* f = fields[vecPatches( ipatch )->neighbor_[1][0]-h0];
nx_ = f->dims_[0]; 
ny_ =                       f->dims_[1]; 
my_ny        = fields[ipatch]->dims_[1];
neighbour_n_space = ny_ - 1 - 2*oversize - f->isDual_[1]; 
if( f->dims_.size()>2 ) {
nz_ = f->dims_[2]; 
}
pt1 = &( *static_cast<F*>( f              ) )( neighbour_n_space * nz_ );
pt2 = &( *static_cast<F*>( fields[ipatch] ) )( 0 );
for( unsigned int i = 0 ; i < nx_ ; i ++ ) {
for( unsigned int j = 0 ; j < oversize*nz_ ; j++ ) {
pt2[i*my_ny*nz_ + j] = pt1[i*ny_*nz_ + j] ; 
pt1[i*ny_*nz_ + j + gsp*nz_] = pt2[i*my_ny*nz_ + j + gsp*nz_] ; 
}
}
}
}
}

}

void SyncVectorPatch::finalizeExchangeAlongY( std::vector<Field *> fields, VectorPatch &vecPatches )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[1];

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
if (fields[ipatch]!=NULL) {
vecPatches( ipatch )->finalizeExchange( fields[ipatch], 1 );
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 1, ( iNeighbor+1 )%2 ) ) {
fields[ipatch]->inject_fields_exch( 1, iNeighbor, oversize );
}
}
}
}

}

template<typename T, typename F>
void SyncVectorPatch::exchangeAlongZ( std::vector<Field *> fields, VectorPatch &vecPatches, SmileiMPI *smpi )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[2];

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
if (fields[ipatch]!=NULL) {
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, iNeighbor ) ) {
fields[ipatch]->create_sub_fields  ( 2, iNeighbor, oversize );
fields[ipatch]->extract_fields_exch( 2, iNeighbor, oversize );
}
}
if ( !dynamic_cast<cField*>( fields[ipatch] ) )
vecPatches( ipatch )->initExchange( fields[ipatch], 2, smpi );
else
vecPatches( ipatch )->initExchangeComplex( fields[ipatch], 2, smpi );
}
}

unsigned int nx_, ny_, nz_, h0, gsp;
T *pt1, *pt2;
h0 = vecPatches( 0 )->hindex;

int IPATCH = 0;
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ )
if (fields[ipatch]!=NULL) {
IPATCH = ipatch;
break;
}

if( fields[IPATCH]==NULL ) return;

gsp = ( oversize + 1 + fields[IPATCH]->isDual_[2] ); 

#pragma omp for schedule(static) private(pt1,pt2)
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {

if( vecPatches( ipatch )->MPI_me_ == vecPatches( ipatch )->MPI_neighbor_[2][0] ) {
if (fields[ipatch]!=NULL) {
Field* f = fields[vecPatches( ipatch )->neighbor_[2][0]-h0];
nx_ = f->dims_[0]; 
ny_ = f->dims_[1]; 
nz_ = f->dims_[2]; 
pt1 = &( *static_cast<F*>( f              ) )( nz_ - 1 - 2*oversize - f->isDual_[2] ); 
pt2 = &( *static_cast<F*>( fields[ipatch] ) )( 0 ); 
for( unsigned int i = 0 ; i < nx_*ny_*nz_ ; i += ny_*nz_ ) {
for( unsigned int j = 0 ; j < ny_*nz_ ; j += nz_ ) {
for( unsigned int k = 0 ; k < oversize ; k++ ) {
pt2[i+j+k] = pt1[i+j+k] ; 
pt1[i+j+k+gsp] = pt2[i+j+k+gsp] ; 
}
}
}
}
}
}

}

void SyncVectorPatch::finalizeExchangeAlongZ( std::vector<Field *> fields, VectorPatch &vecPatches )
{
unsigned oversize = vecPatches( 0 )->EMfields->oversize[2];

#ifndef _NO_MPI_TM
#pragma omp for schedule(static)
#else
#pragma omp single
#endif
for( unsigned int ipatch=0 ; ipatch<fields.size() ; ipatch++ ) {
if (fields[ipatch]!=NULL) {
vecPatches( ipatch )->finalizeExchange( fields[ipatch], 2 );
for (int iNeighbor=0 ; iNeighbor<2 ; iNeighbor++) {
if ( vecPatches( ipatch )->is_a_MPI_neighbor( 2, ( iNeighbor+1 )%2 ) ) {
fields[ipatch]->inject_fields_exch( 2, iNeighbor, oversize );
}
}
}
}

}

void SyncVectorPatch::exchangeForPML( Params &params, VectorPatch &vecPatches, SmileiMPI *smpi )
{
if ( ( params.geometry != "AMcylindrical" ) ) {
if (params.nDim_field==1) {
return;
}
else {
int iDim = 0;
for ( int min_max=0 ; min_max<2 ; min_max++ ) {
#pragma omp single
vecPatches.buildPMLList( "Bx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "By", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hy", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );
}
if (params.nDim_field>2) {
for ( int min_max=0 ; min_max<2 ; min_max++ ) {
#pragma omp single
vecPatches.buildPMLList( "Bx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "By", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hy", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );
}
}
iDim = 1;
for ( int min_max=0 ; min_max<2 ; min_max++ ) {
#pragma omp single
vecPatches.buildPMLList( "Bx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "By", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hy", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );
} 
if (params.nDim_field>2) {
for ( int min_max=0 ; min_max<2 ; min_max++ ) {
#pragma omp single
vecPatches.buildPMLList( "Bx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "By", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hy", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongZ<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongZ( vecPatches.listForPML_, vecPatches );
}
}

if (params.nDim_field>2) {
int iDim = 2;
for ( int min_max=0 ; min_max<2 ; min_max++ ) {
#pragma omp single
vecPatches.buildPMLList( "Bx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "By", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hy", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongX<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "By", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hx", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hy", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hz", iDim, min_max, smpi  );

SyncVectorPatch::exchangeAlongY<double,Field>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );
}
} 
} 
} 
else { 
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++) {
int iDim = 0;
for ( int min_max=0 ; min_max<2 ; min_max++ ) {
#pragma omp single
vecPatches.buildPMLList( "Bl", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongY<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Br", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongY<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bt", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongY<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hl", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongY<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hr", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongY<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Ht", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongY<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongY( vecPatches.listForPML_, vecPatches );
}

iDim = 1;
for ( int min_max=0 ; min_max<2 ; min_max++ ) {
#pragma omp single
vecPatches.buildPMLList( "Bl", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongX<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Br", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongX<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Bt", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongX<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hl", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongX<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Hr", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongX<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );

#pragma omp single
vecPatches.buildPMLList( "Ht", iDim, min_max, smpi, imode );

SyncVectorPatch::exchangeAlongX<complex<double>,cField>( vecPatches.listForPML_, vecPatches, smpi );
SyncVectorPatch::finalizeExchangeAlongX( vecPatches.listForPML_, vecPatches );
}
} 
} 
}
