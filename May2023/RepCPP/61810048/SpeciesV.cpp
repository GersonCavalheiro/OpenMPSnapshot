#include "SpeciesV.h"

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


SpeciesV::SpeciesV( Params &params, Patch *patch ) :
Species( params, patch )
{
initCluster( params );
npack_ = 0 ;
packsize_ = 0;

for (unsigned int idim=0; idim < params.nDim_field; idim++){
distance[idim] = &Species::cartesian_distance;
}
if (params.geometry == "AMcylindrical"){
distance[1] = &Species::radial_distance;
}


}

SpeciesV::~SpeciesV()
{
}


void SpeciesV::initCluster( Params &params )
{
int ncells = 1;
for( unsigned int iDim=0 ; iDim<nDim_field ; iDim++ ) {
ncells *= ( params.n_space[iDim]+1 );
}
particles->last_index.resize( ncells, 0 );
particles->first_index.resize( ncells, 0 );
count.resize( ncells, 0 );


f_dim0 =  params.n_space[0] + 2 * oversize[0] +1;
f_dim1 =  params.n_space[1] + 2 * oversize[1] +1;
f_dim2 =  params.n_space[2] + 2 * oversize[2] +1;

b_dim.resize( params.nDim_field, 1 );
if( nDim_field == 1 ) {
b_dim[0] = ( 1 + cluster_width_ ) + 2 * oversize[0];
f_dim1 = 1;
f_dim2 = 1;
}
if( nDim_field == 2 ) {
b_dim[0] = ( 1 + cluster_width_ ) + 2 * oversize[0]; 
b_dim[1] =  f_dim1;
f_dim2 = 1;
}
if( nDim_field == 3 ) {
b_dim[0] = ( 1 + cluster_width_ ) + 2 * oversize[0]; 
b_dim[1] = f_dim1;
b_dim[2] = f_dim2;
}

MPI_buffer_.allocate( nDim_field );

nrj_bc_lost = 0.;
nrj_mw_out = 0.;
nrj_mw_inj = 0.;
nrj_new_part_ = 0.;
nrj_radiated_ = 0.;

}


void SpeciesV::dynamics( double time_dual, unsigned int ispec,
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

if( npack_==0 ) {
npack_    = 1;
packsize_ = ( f_dim1-2*oversize[1] );

packsize_ *= ( f_dim0-2*oversize[0] );

if( nDim_field == 3 ) {
packsize_ *= ( f_dim2-2*oversize[2] );
}
}

unsigned int iPart;

int tid( 0 );
std::vector<double> nrj_lost_per_thd( 1, 0. );

if( time_dual>time_frozen_ || Ionize ) { 

vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );


for( unsigned int ipack = 0 ; ipack < npack_ ; ipack++ ) {

int nparts_in_pack = particles->last_index[( ipack+1 ) * packsize_-1 ];
smpi->dynamics_resize( ithread, nDim_field, nparts_in_pack, params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif

for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ){
Interp->fieldsWrapper( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ),
&( particles->last_index[ipack*packsize_+scell] ),
ithread, scell, particles->first_index[ipack*packsize_]);
}

#ifdef  __DETAILED_TIMERS
patch->patch_timers[0] += MPI_Wtime() - timer;
#endif

if( Ionize ) {
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {
( *Ionize )( particles, particles->first_index[scell], particles->last_index[scell], Epart, patch, Proj );
}
#ifdef  __DETAILED_TIMERS
patch->patch_timers[4] += MPI_Wtime() - timer;
#endif
}

if ( time_dual <= time_frozen_ ) continue;

for( unsigned int i=0; i<count.size(); i++ ) {
count[i] = 0;
}

if( Radiate ) {
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif

for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {

( *Radiate )( *particles,
radiated_photons_,
smpi,
RadiationTables, nrj_radiated_,
particles->first_index[scell], particles->last_index[scell], ithread );

}
#ifdef  __DETAILED_TIMERS
patch->patch_timers[5] += MPI_Wtime() - timer;
#endif
}

if( Multiphoton_Breit_Wheeler_process ) {
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
for( unsigned int scell = 0 ; scell < particles->first_index.size() ; scell++ ) {

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

}
#ifdef  __DETAILED_TIMERS
patch->patch_timers[6] += MPI_Wtime() - timer;
#endif
}

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Push )( *particles, smpi, particles->first_index[ipack*packsize_],
particles->last_index[ipack*packsize_+packsize_-1],
ithread, particles->first_index[ipack*packsize_] );

#ifdef  __DETAILED_TIMERS
patch->patch_timers[1] += MPI_Wtime() - timer;
timer = MPI_Wtime();
#endif


double energy_lost( 0. );

if( mass_>0 ) {

for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {

energy_lost = 0;

for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
( *partWalls )[iwall]->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += mass_ * energy_lost;
}

partBoundCond->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += mass_ * energy_lost;


}

} else if( mass_==0 ) {

for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {

energy_lost = 0;

for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
( *partWalls )[iwall]->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += energy_lost;
}

partBoundCond->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += energy_lost;


}
} 

computeParticleCellKeys( params,
particles,
&particles->cell_keys[0],
&count[0],
particles->first_index[ipack*packsize_],
particles->last_index[ipack*packsize_+packsize_-1] );




#ifdef  __DETAILED_TIMERS
patch->patch_timers[3] += MPI_Wtime() - timer;
#endif

if( ( !particles->is_test ) && ( mass_ > 0 ) )
#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif

for( unsigned int scell = 0 ; scell < packsize_ ; scell++ )
Proj->currentsAndDensityWrapper(
EMfields, *particles, smpi, particles->first_index[ipack*packsize_+scell],
particles->last_index[ipack*packsize_+scell],
ithread,
diag_flag, params.is_spectral,
ispec, ipack*packsize_+scell, particles->first_index[ipack*packsize_]
);

#ifdef  __DETAILED_TIMERS
patch->patch_timers[2] += MPI_Wtime() - timer;
#endif

for( unsigned int ithd=0 ; ithd<nrj_lost_per_thd.size() ; ithd++ ) {
nrj_bc_lost += nrj_lost_per_thd[tid];
}
} 
} 

if(time_dual <= time_frozen_ && diag_flag &&( !particles->is_test ) ) { 

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


void SpeciesV::computeCharge( unsigned int ispec, ElectroMagn *EMfields, bool old  )
{
if( ( !particles->is_test ) ) {
if( !dynamic_cast<ElectroMagnAM *>( EMfields ) ) {
double *b_rho=&( *EMfields->rho_ )( 0 );
for( unsigned int iPart=particles->first_index[0] ; ( int )iPart<particles->last_index[particles->last_index.size()-1]; iPart++ ) {
Proj->basic( b_rho, ( *particles ), iPart, 0 );
}
} else {
ElectroMagnAM *emAM = static_cast<ElectroMagnAM *>( EMfields );
unsigned int Nmode = emAM->rho_AM_.size();
for( unsigned int imode=0; imode<Nmode; imode++ ) {
complex<double> *b_rho = old ? &( *emAM->rho_old_AM_[imode] )( 0 ) : &( *emAM->rho_AM_[imode] )( 0 );
for( unsigned int iPart=particles->first_index[0] ; ( int )iPart<particles->last_index[particles->last_index.size()-1]; iPart++ ) {
Proj->basicForComplex( b_rho, ( *particles ), iPart, 0, imode );
}
}
}
}

}


void SpeciesV::sortParticles( Params &params, Patch *patch )
{
unsigned int npart, ncell;
int ip_dest, cell_target;
vector<int> buf_cell_keys[3][2];
std::vector<unsigned int> cycle;
unsigned int ip_src;

ncell = ( params.n_space[0]+1 );
for( unsigned int i=1; i < nDim_field; i++ ) {
ncell *= length_[i];
}

npart = particles->size();

for( unsigned int idim=0; idim < nDim_field ; idim++ ) {
for( unsigned int ineighbor=0 ; ineighbor < 2 ; ineighbor++ ) {
buf_cell_keys[idim][ineighbor].resize( MPI_buffer_.part_index_recv_sz[idim][ineighbor] );


computeParticleCellKeys( params,
&MPI_buffer_.partRecv[idim][ineighbor],
&buf_cell_keys[idim][ineighbor][0],
&count[0],
0,
MPI_buffer_.part_index_recv_sz[idim][ineighbor] );

}
}

particles->first_index[0]=0;
for( unsigned int ic=1; ic < ncell; ic++ ) {
particles->first_index[ic] = particles->first_index[ic-1] + count[ic-1];
particles->last_index[ic-1]= particles->first_index[ic];
}

particles->last_index[ncell-1] = particles->last_index[ncell-2] + count.back() ;


if( MPI_buffer_.partRecv[0][0].size() == 0 ) {
MPI_buffer_.partRecv[0][0].initialize( 0, *particles );    
}

if( ( unsigned int )particles->last_index.back() > npart ) {
particles->resize( particles->last_index.back(), nDim_particle, params.keep_position_old );
for (int ip = npart ; ip < particles->last_index.back() ; ip ++) {
particles->cell_keys[ip] = -1;
}
}

for( unsigned int idim=0; idim < nDim_field ; idim++ ) {
for( unsigned int ineighbor=0 ; ineighbor < 2 ; ineighbor++ ) {
for( unsigned int ip=0; ip < MPI_buffer_.part_index_recv_sz[idim][ineighbor]; ip++ ) {
cycle.resize( 1 );
cell_target = buf_cell_keys[idim][ineighbor][ip];
ip_dest = particles->first_index[cell_target];
while( particles->cell_keys[ip_dest] == cell_target ) {
ip_dest++;
}
particles->first_index[cell_target] = ip_dest + 1 ;
cycle[0] = ip_dest;
cell_target = particles->cell_keys[ip_dest];
while( cell_target != -1 ) {
ip_dest = particles->first_index[cell_target];
while( particles->cell_keys[ip_dest] == cell_target ) {
ip_dest++;
}
particles->first_index[cell_target] = ip_dest + 1 ;
cycle.push_back( ip_dest );
cell_target = particles->cell_keys[ip_dest];
}
particles->translateParticles( cycle );
MPI_buffer_.partRecv[idim][ineighbor].overwriteParticle( ip, *particles, cycle[0] );
}
}
}

for( unsigned int ip=( unsigned int )particles->last_index.back(); ip < npart; ip++ ) {
cell_target = particles->cell_keys[ip];

if( cell_target == -1 ) {
continue;
}
cycle.resize( 0 );
cycle.push_back( ip );

while( cell_target != -1 ) {

ip_dest = particles->first_index[cell_target];

while( particles->cell_keys[ip_dest] == cell_target ) {
ip_dest++;
}
particles->first_index[cell_target] = ip_dest + 1 ;
cycle.push_back( ip_dest );
cell_target = particles->cell_keys[ip_dest];
}
particles->translateParticles( cycle );
}

if( ( unsigned int )particles->last_index.back() < npart ) {
particles->resize( particles->last_index.back(), nDim_particle, params.keep_position_old );
}

for( int icell = 0 ; icell < ( int )ncell; icell++ ) {
for( unsigned int ip=( unsigned int )particles->first_index[icell]; ip < ( unsigned int )particles->last_index[icell] ; ip++ ) {
if( particles->cell_keys[ip] != icell ) {
cycle.resize( 1 );
cycle[0] = ip;
ip_src = ip;
while( particles->cell_keys[ip_src] != icell ) {
ip_dest = particles->first_index[particles->cell_keys[ip_src]];
while( particles->cell_keys[ip_dest] == particles->cell_keys[ip_src] ) {
ip_dest++;
}
particles->first_index[particles->cell_keys[ip_src]] = ip_dest + 1 ;
cycle.push_back( ip_dest );
ip_src = ip_dest; 
}
particles->swapParticles( cycle );
}
}
} 
particles->first_index[0]=0;
for( unsigned int ic=1; ic < ncell; ic++ ) {
particles->first_index[ic] = particles->last_index[ic-1];
}
}

void SpeciesV::computeParticleCellKeys( Params    & params,
Particles * particles,
int       * __restrict__ cell_keys,
int       * __restrict__ count,
unsigned int istart,
unsigned int iend ) {

unsigned int iPart;

double * __restrict__ position_x = particles->getPtrPosition(0);
double * __restrict__ position_y = particles->getPtrPosition(1);
double * __restrict__ position_z = particles->getPtrPosition(2);

if (params.geometry == "AMcylindrical"){

double min_loc_l = round(min_loc_vec[0]*dx_inv_[0]);
double min_loc_r = round(min_loc_vec[1]*dx_inv_[1]);

#pragma omp simd
for( iPart=istart; iPart < iend ; iPart++ ) {
if ( cell_keys[iPart] != -1 ) {
cell_keys[iPart]  = round( position_x[iPart] * dx_inv_[0]) - min_loc_l ;
cell_keys[iPart] *= length_[1];
cell_keys[iPart] += round( sqrt(position_y[iPart]*position_y[iPart]+position_z[iPart]*position_z[iPart]) * dx_inv_[1] ) - min_loc_r;
}
}

} else if (nDim_field == 3) {

double min_loc_x = round (min_loc_vec[0] * dx_inv_[0]);
double min_loc_y = round (min_loc_vec[1] * dx_inv_[1]);
double min_loc_z = round (min_loc_vec[2] * dx_inv_[2]);

#pragma omp simd
for( iPart=istart; iPart < iend ; iPart++  ) {
if ( cell_keys[iPart] != -1 ) {
cell_keys[iPart]  = round(position_x[iPart] * dx_inv_[0] )- min_loc_x ;
cell_keys[iPart] *= length_[1];                                         
cell_keys[iPart] += round(position_y[iPart] * dx_inv_[1] )- min_loc_y ;
cell_keys[iPart] *= length_[2];                                         
cell_keys[iPart] += round(position_z[iPart] * dx_inv_[2] )- min_loc_z ;
}
}

} else if (nDim_field == 2) {

double min_loc_x = round (min_loc_vec[0] * dx_inv_[0]);
double min_loc_y = round (min_loc_vec[1] * dx_inv_[1]);

#pragma omp simd
for( iPart=istart; iPart < iend ; iPart++  ) {
if ( cell_keys[iPart] != -1 ) {
cell_keys[iPart]  = round(position_x[iPart] * dx_inv_[0] )- min_loc_x ;
cell_keys[iPart] *= length_[1];
cell_keys[iPart] += round(position_y[iPart] * dx_inv_[1] )- min_loc_y ;

}
}
} else if (nDim_field == 1) {

double min_loc_x = round (min_loc_vec[0] * dx_inv_[0]);

#pragma omp simd
for( iPart=istart; iPart < iend ; iPart++  ) {
if ( cell_keys[iPart] != -1 ) {
cell_keys[iPart]  = round(position_x[iPart] * dx_inv_[0] )- min_loc_x ;
}
}

}

for( iPart=istart; iPart < iend ; iPart++  ) {
if ( cell_keys[iPart] != -1 ) {
count[cell_keys[iPart]] ++;
}
}
}

void SpeciesV::computeParticleCellKeys( Params &params )
{

unsigned int npart;

npart = particles->size(); 

int * __restrict__ cell_keys  = particles->getPtrCellKeys();

for( unsigned int ic=0; ic < count.size() ; ic++ ) {
count[ic] = 0 ;
}


computeParticleCellKeys( params, particles, cell_keys, &count[0], 0, npart );

}

void SpeciesV::importParticles( Params &params, Patch *patch, Particles &source_particles, vector<Diagnostic *> &localDiags )
{

unsigned int npart = source_particles.size(), ncells=particles->first_index.size();

if( particles->tracked ) {
dynamic_cast<DiagnosticTrack *>( localDiags[tracking_diagnostic] )->setIDs( source_particles );
}

vector<int> src_cell_keys( npart, 0 );
vector<int> src_count( ncells, 0 );


computeParticleCellKeys( params,
&source_particles,
&src_cell_keys[0],
&src_count[0],
0,
npart );

int istart = 0;
int istop  = src_count[0];

for ( int icell = 0 ; icell < (int)ncells ; icell++ ) {
if (src_count[icell]!=0) {
for( int ip=istart; ip < istop ; ip++ ) {
if ( src_cell_keys[ip] == icell )
continue;
else { 
int ip_swap = istop;
while (( src_cell_keys[ip_swap] != icell ) && (ip_swap<(int)npart))
ip_swap++;
source_particles.swapParticle(ip, ip_swap);
int tmp = src_cell_keys[ip];
src_cell_keys[ip] = src_cell_keys[ip_swap];
src_cell_keys[ip_swap] = tmp;
} 
} 

source_particles.copyParticles( istart, src_count[icell],
*particles,
particles->first_index[icell] );
particles->last_index[icell] += src_count[icell];
for ( unsigned int idx=icell+1 ; idx<particles->last_index.size() ; idx++ ) {
particles->first_index[idx] += src_count[icell];
particles->last_index[idx]  += src_count[icell];
}
count[icell] += src_count[icell];

}
istart += src_count[icell];
if ( icell != (int)ncells-1  )
istop  += src_count[icell+1];
else
istop = npart;

} 

for (unsigned int ip=0;ip<npart ; ip++ )
addSpaceForOneParticle();

source_particles.clear();

}

void SpeciesV::mergeParticles( double time_dual, unsigned int ispec,
Params &params,
Patch *patch, SmileiMPI *smpi,
std::vector<Diagnostic *> &localDiags )
{


if( time_dual>time_frozen_ ) {

unsigned int scell ;
std::vector <int> mask(particles->last_index.back(), 1);



for( scell = 0 ; scell < particles->first_index.size() ; scell++ ) {

( *Merge )( mass_, *particles, mask, smpi, particles->first_index[scell],
particles->last_index[scell], count[scell]);

}

particles->eraseParticlesWithMask(0, particles->last_index.back(), mask);

particles->first_index[0] = 0;
particles->last_index[0] = count[0];
for( scell = 1 ; scell < particles->first_index.size(); scell++ ) {
particles->first_index[scell] = particles->last_index[scell-1];
particles->last_index[scell] = particles->first_index[scell] + count[scell];
}







}
}


void SpeciesV::ponderomotiveUpdateSusceptibilityAndMomentum( double time_dual, unsigned int ispec,
ElectroMagn *EMfields,
Params &params, bool diag_flag,
Patch *patch, SmileiMPI *smpi,
std::vector<Diagnostic *> &localDiags )
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

if( npack_==0 ) {
npack_    = 1;
packsize_ = ( f_dim1-2*oversize[1] );

packsize_ *= ( f_dim0-2*oversize[0] );

if( nDim_particle == 3 ) {
packsize_ *= ( f_dim2-2*oversize[2] );
}
}


if( time_dual>time_frozen_ || Ionize ) { 

for( unsigned int ipack = 0 ; ipack < npack_ ; ipack++ ) {

int nparts_in_pack = particles->last_index[( ipack+1 ) * packsize_-1 ];
smpi->dynamics_resize( ithread, nDim_field, nparts_in_pack, params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
Interp->fieldsAndEnvelope( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ), &( particles->last_index[ipack*packsize_+scell] ), ithread, particles->first_index[ipack*packsize_] );
}
#ifdef  __DETAILED_TIMERS
patch->patch_timers[7] += MPI_Wtime() - timer;
#endif

if( Ionize ) {

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
vector<double> *Epart = &( smpi->dynamics_Epart[ithread] );
vector<double> *EnvEabs_part  = &( smpi->dynamics_EnvEabs_part[ithread] );
vector<double> *EnvExabs_part = &( smpi->dynamics_EnvExabs_part[ithread] );
vector<double> *Phipart = &( smpi->dynamics_PHIpart[ithread] );
for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
Interp->envelopeFieldForIonization( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ), &( particles->last_index[ipack*packsize_+scell] ), ithread );
Ionize->envelopeIonization( particles, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], Epart, EnvEabs_part, EnvExabs_part, Phipart, patch, Proj );
}
#ifdef  __DETAILED_TIMERS
patch->patch_timers[4] += MPI_Wtime() - timer;
#endif
}

if( time_dual<=time_frozen_ ) continue; 

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
Proj->susceptibility( EMfields, *particles, mass_, smpi, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], ithread, ipack*packsize_+scell, particles->first_index[ipack*packsize_] );
}

#ifdef  __DETAILED_TIMERS
patch->patch_timers[8] += MPI_Wtime() - timer;
#endif

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Push )( *particles, smpi, particles->first_index[ipack*packsize_], particles->last_index[ipack*packsize_+packsize_-1], ithread, particles->first_index[ipack*packsize_] );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[9] += MPI_Wtime() - timer;
#endif
}

} else { 

}

} 

void SpeciesV::ponderomotiveProjectSusceptibility( double time_dual, unsigned int ispec,
ElectroMagn *EMfields,
Params &params, bool diag_flag,
Patch *patch, SmileiMPI *smpi,
std::vector<Diagnostic *> &localDiags )
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

if( npack_==0 ) {
npack_    = 1;
packsize_ = ( f_dim1-2*oversize[1] );

packsize_ *= ( f_dim0-2*oversize[0] );

if( nDim_particle == 3 ) {
packsize_ *= ( f_dim2-2*oversize[2] );
}
}


if( time_dual>time_frozen_ ) { 

for( unsigned int ipack = 0 ; ipack < npack_ ; ipack++ ) {

int nparts_in_pack = particles->last_index[( ipack+1 ) * packsize_-1 ];
smpi->dynamics_resize( ithread, nDim_field, nparts_in_pack, params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
Interp->fieldsAndEnvelope( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ), &( particles->last_index[ipack*packsize_+scell] ), ithread, particles->first_index[ipack*packsize_] );
}
#ifdef  __DETAILED_TIMERS
patch->patch_timers[4] += MPI_Wtime() - timer;
#endif

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
Proj->susceptibility( EMfields, *particles, mass_, smpi, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], ithread, ipack*packsize_+scell, particles->first_index[ipack*packsize_] );
}

#ifdef  __DETAILED_TIMERS
patch->patch_timers[8] += MPI_Wtime() - timer;
#endif

}

} else { 

}

} 

void SpeciesV::ponderomotiveUpdatePositionAndCurrents( double time_dual, unsigned int ispec,
ElectroMagn *EMfields,
Params &params, bool diag_flag, PartWalls *partWalls,
Patch *patch, SmileiMPI *smpi,
std::vector<Diagnostic *> &localDiags )
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

for( unsigned int i=0; i<count.size(); i++ ) {
count[i] = 0;
}

for( unsigned int ipack = 0 ; ipack < npack_ ; ipack++ ) {

int nparts_in_pack = particles->last_index[( ipack+1 ) * packsize_-1 ];
smpi->dynamics_resize( ithread, nDim_field, nparts_in_pack, params.geometry=="AMcylindrical" );

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
Interp->timeCenteredEnvelope( EMfields, *particles, smpi, &( particles->first_index[ipack*packsize_+scell] ), &( particles->last_index[ipack*packsize_+scell] ), ithread, particles->first_index[ipack*packsize_] );
}
#ifdef  __DETAILED_TIMERS
patch->patch_timers[10] += MPI_Wtime() - timer;
#endif

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
( *Push_ponderomotive_position )( *particles, smpi, particles->first_index[ipack*packsize_], particles->last_index[ipack*packsize_+packsize_-1], ithread, particles->first_index[ipack*packsize_] );
#ifdef  __DETAILED_TIMERS
patch->patch_timers[11] += MPI_Wtime() - timer;
timer = MPI_Wtime();
#endif

for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
double energy_lost( 0. );
if( mass_>0 ) { 
for( unsigned int iwall=0; iwall<partWalls->size(); iwall++ ) {
( *partWalls )[iwall]->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += mass_ * energy_lost;
}


partBoundCond->apply( this, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], smpi->dynamics_invgf[ithread], patch->rand_, energy_lost );
nrj_lost_per_thd[tid] += mass_ * energy_lost;

for( iPart=particles->first_index[ipack*packsize_+scell] ; ( int )iPart<particles->last_index[ipack*packsize_+scell]; iPart++ ) {
if ( particles->cell_keys[iPart] != -1 ) {
for( int i = 0 ; i<( int )nDim_field; i++ ) {
particles->cell_keys[iPart] *= length_[i];
particles->cell_keys[iPart] += round( ((this)->*(distance[i]))(particles, i, iPart) * dx_inv_[i] );
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
patch->patch_timers[3] += MPI_Wtime() - timer;
#endif

#ifdef  __DETAILED_TIMERS
timer = MPI_Wtime();
#endif
if( ( !particles->is_test ) && ( mass_ > 0 ) )
for( unsigned int scell = 0 ; scell < packsize_ ; scell++ ) {
Proj->currentsAndDensityWrapper( EMfields, *particles, smpi, particles->first_index[ipack*packsize_+scell], particles->last_index[ipack*packsize_+scell], ithread, diag_flag, params.is_spectral, ispec, ipack*packsize_+scell, particles->first_index[ipack*packsize_] );
}

#ifdef  __DETAILED_TIMERS
patch->patch_timers[12] += MPI_Wtime() - timer;
#endif
}

for( unsigned int ithd=0 ; ithd<nrj_lost_per_thd.size() ; ithd++ ) {
nrj_bc_lost += nrj_lost_per_thd[tid];
}

} else { 
if( diag_flag &&( !particles->is_test ) ) {
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

} 
