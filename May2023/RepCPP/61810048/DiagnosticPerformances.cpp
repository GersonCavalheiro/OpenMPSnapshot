#include "PyTools.h"
#include <iomanip>

#include "DiagnosticPerformances.h"


using namespace std;

const unsigned int n_quantities_double = 19;
const unsigned int n_quantities_uint   = 4;

DiagnosticPerformances::DiagnosticPerformances( Params &params, SmileiMPI *smpi )
: mpi_size_( smpi->getSize() ),
mpi_rank_( smpi->getRank() ),
filespace_double( {n_quantities_double, mpi_size_}, {0, mpi_rank_}, {n_quantities_double, 1} ),
filespace_uint  ( {n_quantities_uint  , mpi_size_}, {0, mpi_rank_}, {n_quantities_uint  , 1} ),
memspace_double( { n_quantities_double, 1 }, {}, {} ),
memspace_uint  ( { n_quantities_uint  , 1 }, {}, {} )
{
timestep = params.timestep;
cell_load = params.cell_load;
frozen_particle_load = params.frozen_particle_load;
tot_number_of_patches = params.tot_number_of_patches;

ostringstream name( "" );
name << "Diagnostic performances";
string errorPrefix = name.str();

timeSelection = new TimeSelection(
PyTools::extract_py( "every", "DiagPerformances" ),
name.str()
);

flush_timeSelection = new TimeSelection(
PyTools::extract_py( "flush_every", "DiagPerformances" ),
name.str()
);

PyTools::extract( "patch_information", patch_information, "DiagPerformances"  );

if( smpi->isMaster() ) {
MESSAGE( 1, "Created performances diagnostic" );
}
filename = "Performances.h5";

ndim     = params.nDim_field;
has_adaptive_vectorization = params.has_adaptive_vectorization;

ncells_per_patch = 1;
for( unsigned int idim = 0; idim < params.nDim_field; idim++ ) {
ncells_per_patch *= params.n_space[idim]+2*params.oversize[idim];
}

} 


DiagnosticPerformances::~DiagnosticPerformances()
{
delete timeSelection;
delete flush_timeSelection;
} 


void DiagnosticPerformances::openFile( Params &params, SmileiMPI *smpi )
{
if( file_ ) {
return;
}

file_ = new H5Write( filename, &smpi->world() );

file_->attr( "MPI_SIZE", smpi->getSize() );
file_->attr( "patch_arrangement", params.patch_arrangement );

vector<string> quantities_uint( n_quantities_uint );
quantities_uint[0] = "hindex"                    ;
quantities_uint[1] = "number_of_cells"           ;
quantities_uint[2] = "number_of_particles"       ;
quantities_uint[3] = "number_of_frozen_particles";
file_->attr( "quantities_uint", quantities_uint );

vector<string> quantities_double( n_quantities_double );
quantities_double[ 0] = "total_load"      ;
quantities_double[ 1] = "timer_global"    ;
quantities_double[ 2] = "timer_particles" ;
quantities_double[ 3] = "timer_maxwell"   ;
quantities_double[ 4] = "timer_densities" ;
quantities_double[ 5] = "timer_collisions";
quantities_double[ 6] = "timer_movWindow" ;
quantities_double[ 7] = "timer_loadBal"   ;
quantities_double[ 8] = "timer_syncPart"  ;
quantities_double[ 9] = "timer_syncField" ;
quantities_double[10] = "timer_syncDens"  ;
quantities_double[11] = "timer_diags"     ;
quantities_double[12] = "timer_grids"     ;
quantities_double[13] = "timer_total"     ;
quantities_double[14] = "memory_total"    ;
quantities_double[15] = "memory_peak"    ;
quantities_double[16] = "timer_envelope"     ;
quantities_double[17] = "timer_syncSusceptibility"     ;
quantities_double[18] = "timer_partMerging"     ;
file_->attr( "quantities_double", quantities_double );

file_->flush();
}


void DiagnosticPerformances::closeFile()
{
if( file_ ) {
delete file_;
file_ = NULL;
}
} 



void DiagnosticPerformances::init( Params &params, SmileiMPI *smpi, VectorPatch &vecPatches )
{
openFile( params, smpi );
}


bool DiagnosticPerformances::prepare( int itime )
{
if( timeSelection->theTimeIsNow( itime ) ) {
return true;
} else {
return false;
}
} 


void DiagnosticPerformances::run( SmileiMPI *smpi, VectorPatch &vecPatches, int itime, SimWindow *simWindow, Timers &timers )
{

#pragma omp master
{
ostringstream name_t;
name_t.str( "" );
name_t << setfill( '0' ) << setw( 10 ) << itime;
group_name = name_t.str();
has_group = file_->has( group_name );
}
#pragma omp barrier

if( has_group ) {
return;
}

#pragma omp master
{
H5Write iteration_group = file_->group( group_name );

unsigned int number_of_patches = vecPatches.size();
unsigned int number_of_cells = ncells_per_patch * number_of_patches;
unsigned int number_of_species = vecPatches( 0 )->vecSpecies.size();
unsigned int number_of_particles=0, number_of_frozen_particles=0;
double time = itime * timestep;
for( unsigned int ipatch=0; ipatch < number_of_patches; ipatch++ ) {
for( unsigned int ispecies = 0; ispecies < number_of_species; ispecies++ ) {
if( time < vecPatches( ipatch )->vecSpecies[ispecies]->time_frozen_ ) {
number_of_frozen_particles += vecPatches( ipatch )->vecSpecies[ispecies]->getNbrOfParticles();
} else {
number_of_particles += vecPatches( ipatch )->vecSpecies[ispecies]->getNbrOfParticles();
}
}
}
double total_load =
( ( double )number_of_particles )
+ ( ( double )number_of_frozen_particles ) * frozen_particle_load
+ ( ( double )number_of_cells ) * cell_load;

vector<unsigned int> quantities_uint( n_quantities_uint );
quantities_uint[0] = vecPatches( 0 )->Hindex()   ;
quantities_uint[1] = number_of_cells           ;
quantities_uint[2] = number_of_particles       ;
quantities_uint[3] = number_of_frozen_particles;

iteration_group.array( "quantities_uint", quantities_uint[0], &filespace_uint, &memspace_uint );

vector<double> quantities_double( n_quantities_double );
quantities_double[ 0] = total_load                 ;
quantities_double[ 1] = timers.global    .getTime();
quantities_double[ 2] = timers.particles .getTime();
quantities_double[ 3] = timers.maxwell   .getTime();
quantities_double[ 4] = timers.densities .getTime();
quantities_double[ 5] = timers.collisions.getTime();
quantities_double[ 6] = timers.movWindow .getTime();
quantities_double[ 7] = timers.loadBal   .getTime();
quantities_double[ 8] = timers.syncPart  .getTime();
quantities_double[ 9] = timers.syncField .getTime();
quantities_double[10] = timers.syncDens  .getTime();
double timer_diags = MPI_Wtime() - timers.diags.last_start_ + timers.diags.time_acc_;
quantities_double[11] = timer_diags;
quantities_double[12] = timers.grids     .getTime();
double timer_total =
quantities_double[ 2] + quantities_double[3] + quantities_double[ 4]
+ quantities_double[ 5] + quantities_double[6] + quantities_double[ 7]
+ quantities_double[ 8] + quantities_double[9] + quantities_double[10]
+ quantities_double[11] + quantities_double[12];
quantities_double[13] = timer_total;

quantities_double[14] = Tools::getMemFootPrint(0);
quantities_double[15] = Tools::getMemFootPrint(1);
quantities_double[16] = timers.envelope         .getTime();
quantities_double[17] = timers.susceptibility   .getTime();
quantities_double[18] = timers.particleMerging  .getTime();

iteration_group.array( "quantities_double", quantities_double[0], &filespace_double, &memspace_double );

if( patch_information ) {

H5Write patch_group = iteration_group.group( "patches" );

hsize_t size = tot_number_of_patches;
hsize_t offset = vecPatches(0)->hindex;
hsize_t npoints = vecPatches.size();

vector <unsigned int> buffer( number_of_patches );
for( unsigned int ipatch=0; ipatch < number_of_patches; ipatch++ ) {
buffer[ipatch] = vecPatches( ipatch )->Pcoordinates[0];
}
patch_group.vect( "x", buffer[0], size, H5T_NATIVE_UINT, offset, npoints );

if( ndim > 1 ) {
for( unsigned int ipatch=0; ipatch < number_of_patches; ipatch++ ) {
buffer[ipatch] = vecPatches( ipatch )->Pcoordinates[1];
}
patch_group.vect( "y", buffer[0], size, H5T_NATIVE_UINT, offset, npoints );
}

if( ndim > 2 ) {
for( unsigned int ipatch=0; ipatch < number_of_patches; ipatch++ ) {
buffer[ipatch] = vecPatches( ipatch )->Pcoordinates[2];
}
patch_group.vect( "z", buffer[0], size, H5T_NATIVE_UINT, offset, npoints );
}

for( unsigned int ipatch=0; ipatch < number_of_patches; ipatch++ ) {
buffer[ipatch] = vecPatches( ipatch )->hindex;
}
patch_group.vect( "index", buffer[0], size, H5T_NATIVE_UINT, offset, npoints );

for( unsigned int ispecies = 0; ispecies < number_of_species; ispecies++ ) {
H5Write species_group = patch_group.group( vecPatches( 0 )->vecSpecies[ispecies]->name_ );

if( has_adaptive_vectorization ) {
for( unsigned int ipatch=0; ipatch < number_of_patches; ipatch++ ) {
buffer[ipatch] = ( unsigned int )( vecPatches( ipatch )->vecSpecies[ispecies]->vectorized_operators );
}
species_group.vect( "vecto", buffer[0], size, H5T_NATIVE_UINT, offset, npoints );
}
}

for( unsigned int ipatch=0; ipatch < number_of_patches; ipatch++ ) {
buffer[ipatch] = mpi_rank_;
}
patch_group.vect( "mpi_rank", buffer[0], size, H5T_NATIVE_UINT, offset, npoints );

}

if( flush_timeSelection->theTimeIsNow( itime ) ) {
file_->flush();
}
}

} 


uint64_t DiagnosticPerformances::getDiskFootPrint( int istart, int istop, Patch *patch )
{
uint64_t footprint = 0;

uint64_t ndumps = timeSelection->howManyTimesBefore( istop ) - timeSelection->howManyTimesBefore( istart );

footprint += 1000;

footprint += ndumps * 800;

footprint += ndumps * 2 * 600;

footprint += ndumps * ( uint64_t )( mpi_size_ ) * ( uint64_t )( n_quantities_double * sizeof( double ) + n_quantities_uint * sizeof( unsigned int ) );

return footprint;
}
