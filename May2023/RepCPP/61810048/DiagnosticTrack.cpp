#include "PyTools.h"

#include <string>
#include <sstream>

#include "ParticleData.h"
#include "PeekAtSpecies.h"
#include "DiagnosticTrack.h"
#include "VectorPatch.h"
#include "Params.h"

using namespace std;

DiagnosticTrack::DiagnosticTrack( Params &params, SmileiMPI *smpi, VectorPatch &vecPatches, unsigned int iDiagTrackParticles, unsigned int idiag, OpenPMDparams &oPMD ) :
Diagnostic( &oPMD, "DiagTrackParticles", iDiagTrackParticles ),
IDs_done( params.restart ),
nDim_particle( params.nDim_particle )
{

string species_name;
PyTools::extract( "species", species_name, "DiagTrackParticles", iDiagTrackParticles );
vector<string> species_names = {species_name};
vector<unsigned int> species_ids = Params::FindSpecies( vecPatches( 0 )->vecSpecies, species_names );
if( species_ids.size() > 1 ) {
ERROR( "DiagTrackParticles #" << iDiagTrackParticles << " corresponds to more than 1 species" );
}
if( species_ids.size() < 1 ) {
ERROR( "DiagTrackParticles #" << iDiagTrackParticles << " does not correspond to any existing species" );
}
speciesId_ = species_ids[0];

ostringstream name( "" );
name << "Tracking species '" << species_name << "'";

timeSelection = new TimeSelection( PyTools::extract_py( "every", "DiagTrackParticles", iDiagTrackParticles ), name.str() );

flush_timeSelection = new TimeSelection( PyTools::extract_py( "flush_every", "DiagTrackParticles", iDiagTrackParticles ), name.str() );

for( unsigned int ipatch=0; ipatch<vecPatches.size(); ipatch++ ) {
vecPatches( ipatch )->vecSpecies[speciesId_]->tracking_diagnostic = idiag;
}

filter = PyTools::extract_py( "filter", "DiagTrackParticles", iDiagTrackParticles );
has_filter = ( filter != Py_None );
if( has_filter ) {
#ifdef SMILEI_USE_NUMPY
name << " filter:";
bool *dummy = NULL;
ParticleData test( nDim_particle, filter, name.str(), dummy );
#else
ERROR( name.str() << " with a filter requires the numpy package" );
#endif
}

vector<string> attributes( 0 );
if( !PyTools::extractV( "attributes", attributes, "DiagTrackParticles", iDiagTrackParticles ) ) {
ERROR( "DiagTrackParticles #" << iDiagTrackParticles << ": argument `attribute` must be a list of strings" );
}
if( attributes.size() == 0 ) {
ERROR( "DiagTrackParticles #" << iDiagTrackParticles << ": argument `attribute` must have at least one element" );
}
ostringstream attr_list( "" );
attr_list << "id";
write_position.resize( 3, false );
write_momentum.resize( 3, false );
write_charge = false;
write_weight = false;
write_chi    = false;
write_E.resize( 3, false );
write_B.resize( 3, false );
interpolate = false;
for( unsigned int i=0; i<attributes.size(); i++ ) {
if( attributes[i] == "x" ) {
write_position[0] = true;
} else if( attributes[i] == "y" ) {
if( nDim_particle>1 ) {
write_position[1] = true;
} else {
continue;
}
} else if( attributes[i] == "z" ) {
if( nDim_particle>2 ) {
write_position[2] = true;
} else {
continue;
}
} else if( attributes[i] == "px" ) {
write_momentum[0] = true;
} else if( attributes[i] == "py" ) {
write_momentum[1] = true;
} else if( attributes[i] == "pz" ) {
write_momentum[2] = true;
} else if( attributes[i] == "charge" || attributes[i] == "q" ) {
write_charge      = true;
} else if( attributes[i] == "weight" || attributes[i] == "w" ) {
write_weight      = true;
} else if( attributes[i] == "chi" ) {
write_chi         = true;
} else if( attributes[i] == "Ex" ) {
write_E[0]        = true;
interpolate = true;
} else if( attributes[i] == "Ey" ) {
write_E[1]        = true;
interpolate = true;
} else if( attributes[i] == "Ez" ) {
write_E[2]        = true;
interpolate = true;
} else if( attributes[i] == "Bx" ) {
write_B[0]        = true;
interpolate = true;
} else if( attributes[i] == "By" ) {
write_B[1]        = true;
interpolate = true;
} else if( attributes[i] == "Bz" ) {
write_B[2]        = true;
interpolate = true;
} else {
ERROR( "DiagTrackParticles #" << iDiagTrackParticles << ": attribute `" << attributes[i] << "` unknown" );
}
attr_list << "," << attributes[i];
}
write_any_position = write_position[0] || write_position[1] || write_position[2];
write_any_momentum = write_momentum[0] || write_momentum[1] || write_momentum[2];
write_any_E = write_E[0] || write_E[1] || write_E[2];
write_any_B = write_B[0] || write_B[1] || write_B[2];
if( write_chi && ! vecPatches( 0 )->vecSpecies[speciesId_]->particles->isQuantumParameter ) {
ERROR( "DiagTrackParticles #" << iDiagTrackParticles << ": attribute `chi` not available for this species" );
}

ostringstream hdf_filename( "" );
hdf_filename << "TrackParticlesDisordered_" << species_name  << ".h5" ;
filename = hdf_filename.str();

if( smpi->isMaster() ) {
MESSAGE( 1, "Created TrackParticles #" << iDiagTrackParticles << ": species " << species_name );
MESSAGE( 2, attr_list.str() );
}

if( params.print_expected_disk_usage ) {
PeekAtSpecies peek( params, speciesId_ );
npart_total = peek.totalNumberofParticles();
} else {
npart_total = 0;
}
}

DiagnosticTrack::~DiagnosticTrack()
{
delete timeSelection;
delete flush_timeSelection;
Py_DECREF( filter );
closeFile();
}


void DiagnosticTrack::openFile( Params &params, SmileiMPI *smpi )
{
file_ = new H5Write( filename, &smpi->world() );

file_->attr( "name", diag_name_ );

openPMD_->writeRootAttributes( *file_, "no_meshes", "particles/" );

data_group = new H5Write( file_, "data" );

file_->flush();
}


void DiagnosticTrack::closeFile()
{
if( file_ ) {
delete data_group;
delete file_;
file_ = NULL;
}
}


void DiagnosticTrack::init( Params &params, SmileiMPI *smpi, VectorPatch &vecPatches )
{
if( ! IDs_done ) {
latest_Id = smpi->getRank() * 4294967296; 

for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
setIDs( vecPatches( ipatch ) );
}

IDs_done = true;
}

openFile( params, smpi );

}


bool DiagnosticTrack::prepare( int itime )
{
return timeSelection->theTimeIsNow( itime );
}


void DiagnosticTrack::run( SmileiMPI *smpi, VectorPatch &vecPatches, int itime, SimWindow *simWindow, Timers &timers )
{
uint64_t nParticles_global = 0;
string xyz = "xyz";

H5Write *momentum_group=NULL, *position_group=NULL, *species_group=NULL;
H5Space *file_space=NULL, *mem_space=NULL;
#pragma omp master
{
nParticles_local = 0;
patch_start.resize( vecPatches.size() );

if( has_filter ) {

#ifdef SMILEI_USE_NUMPY
patch_selection.resize( vecPatches.size() );
PyArrayObject *ret;
ParticleData particleData( 0 );
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
patch_selection[ipatch].resize( 0 );
Particles *p = vecPatches( ipatch )->vecSpecies[speciesId_]->particles;
unsigned int npart = p->size();
if( npart > 0 ) {
particleData.resize( npart );
particleData.set( p );
ret = ( PyArrayObject * )PyObject_CallFunctionObjArgs( filter, particleData.get(), NULL );
PyTools::checkPyError();
particleData.clear();
if( ret == NULL ) {
ERROR( "A DiagTrackParticles filter has not provided a correct result" );
}
bool *arr = ( bool * ) PyArray_GETPTR1( ret, 0 );
for( unsigned int i=0; i<npart; i++ ) {
if( arr[i] ) {
patch_selection[ipatch].push_back( i );
if( (p->id( i ) & 72057594037927935) == 0 ) {
p->id( i ) += ++latest_Id;
}
}
}
Py_DECREF( ret );
}
patch_start[ipatch] = nParticles_local;
nParticles_local += patch_selection[ipatch].size();
}
#endif

} else {
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
patch_start[ipatch] = nParticles_local;
nParticles_local += vecPatches( ipatch )->vecSpecies[speciesId_]->getNbrOfParticles();
}
}

mem_space = new H5Space( (hsize_t)nParticles_local );

uint64_t np_local = nParticles_local, offset;
MPI_Scan( &np_local, &offset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
nParticles_global = offset;
offset -= np_local;
MPI_Bcast( &nParticles_global, 1, MPI_UNSIGNED_LONG_LONG, smpi->getSize()-1, MPI_COMM_WORLD );

ostringstream t( "" );
t << setfill( '0' ) << setw( 10 ) << itime;
H5Write iteration_group = data_group->group( t.str() );
H5Write particles_group = iteration_group.group( "particles" );
species_group = new H5Write( &particles_group, vecPatches( 0 )->vecSpecies[speciesId_]->name_ );

openPMD_->writeBasePathAttributes( iteration_group, itime );
openPMD_->writeParticlesAttributes( particles_group );
openPMD_->writeSpeciesAttributes( *species_group );

iteration_group.attr( "x_moved", simWindow ? simWindow->getXmoved() : 0. );

hsize_t chunk = 0;
if( nParticles_global>0 ) {
unsigned int maximum_chunk_size = 100000000;
unsigned int number_of_chunks = nParticles_global/maximum_chunk_size;
if( nParticles_global%maximum_chunk_size != 0 ) {
number_of_chunks++;
}
if( number_of_chunks <= 1 ) {
chunk = 0;
} else {
unsigned int chunk_size = nParticles_global/number_of_chunks;
if( nParticles_global%number_of_chunks != 0 ) {
chunk_size++;
}
chunk = chunk_size;
}
}
file_space = new H5Space( nParticles_global, offset, nParticles_local, chunk );

iteration_group.vect( "latest_IDs", latest_Id, smpi->getSize(), H5T_NATIVE_UINT64, smpi->getRank(), 1 );

}

#pragma omp master
data_uint64.resize( nParticles_local, 1 );
#pragma omp barrier
fill_buffer( vecPatches, 0, data_uint64 );
#pragma omp master
{
write_scalar( species_group, "id", data_uint64[0], H5T_NATIVE_UINT64, file_space, mem_space, SMILEI_UNIT_NONE );
data_uint64.resize( 0 );
}

if( write_charge ) {
#pragma omp master
data_short.resize( nParticles_local, 0 );
#pragma omp barrier
fill_buffer( vecPatches, 0, data_short );
#pragma omp master
{
write_scalar( species_group, "charge", data_short[0], H5T_NATIVE_SHORT, file_space, mem_space, SMILEI_UNIT_CHARGE );
data_short.resize( 0 );
}
}

#pragma omp master
data_double.resize( nParticles_local, 0 );

if( write_weight ) {
#pragma omp barrier
fill_buffer( vecPatches, nDim_particle+3, data_double );
#pragma omp master
write_scalar( species_group, "weight", data_double[0], H5T_NATIVE_DOUBLE, file_space, mem_space, SMILEI_UNIT_DENSITY );
}

if( write_any_momentum ) {
#pragma omp master
{
momentum_group = new H5Write( species_group, "momentum" );
openPMD_->writeRecordAttributes( *momentum_group, SMILEI_UNIT_MOMENTUM );
}
for( unsigned int idim=0; idim<3; idim++ ) {
if( write_momentum[idim] ) {
#pragma omp barrier
fill_buffer( vecPatches, nDim_particle+idim, data_double );
#pragma omp master
{
if( vecPatches( 0 )->vecSpecies[speciesId_]->mass_ != 1. &&
vecPatches( 0 )->vecSpecies[speciesId_]->mass_ > 0) {
for( unsigned int ip=0; ip<nParticles_local; ip++ ) {
data_double[ip] *= vecPatches( 0 )->vecSpecies[speciesId_]->mass_;
}
}
write_component( momentum_group, xyz.substr( idim, 1 ).c_str(), data_double[0], H5T_NATIVE_DOUBLE, file_space, mem_space, SMILEI_UNIT_MOMENTUM );
}
}
}
#pragma omp master
delete momentum_group;
}

if( write_any_position ) {
#pragma omp master
{
position_group = new H5Write( species_group, "position" );
openPMD_->writeRecordAttributes( *position_group, SMILEI_UNIT_POSITION );
}
for( unsigned int idim=0; idim<nDim_particle; idim++ ) {
if( write_position[idim] ) {
#pragma omp barrier
fill_buffer( vecPatches, idim, data_double );
#pragma omp master
write_component( position_group, xyz.substr( idim, 1 ).c_str(), data_double[0], H5T_NATIVE_DOUBLE, file_space, mem_space, SMILEI_UNIT_POSITION );
}
}
#pragma omp master
delete position_group;
}

if( write_chi ) {
#pragma omp barrier
#ifdef  __DEBUG
fill_buffer( vecPatches, nDim_particle+3+3+1, data_double );
#else
fill_buffer( vecPatches, nDim_particle+3+1, data_double );
#endif
#pragma omp master
write_scalar( species_group, "chi", data_double[0], H5T_NATIVE_DOUBLE, file_space, mem_space, SMILEI_UNIT_NONE );
}

#pragma omp barrier

if( interpolate ) {


#pragma omp master
data_double.resize( nParticles_local*6 );

unsigned int nPatches=vecPatches.size();
#pragma omp barrier

if( has_filter ) {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<nPatches ; ipatch++ ) {
vecPatches.species( ipatch, speciesId_ )->Interp->fieldsSelection(
vecPatches.emfields( ipatch ),
*( vecPatches.species( ipatch, speciesId_ )->particles ),
&data_double[patch_start[ipatch]],
( int ) nParticles_local,
&patch_selection[ipatch]
);
}
} else {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<nPatches ; ipatch++ ) {
vecPatches.species( ipatch, speciesId_ )->Interp->fieldsSelection(
vecPatches.emfields( ipatch ),
*( vecPatches.species( ipatch, speciesId_ )->particles ),
&data_double[patch_start[ipatch]],
( int ) nParticles_local,
NULL
);
}
}
#pragma omp barrier

#pragma omp master
{
if( write_any_E ) {
H5Write Efield_group = species_group->group( "E" );
openPMD_->writeRecordAttributes( Efield_group, SMILEI_UNIT_EFIELD );
for( unsigned int idim=0; idim<3; idim++ ) {
if( write_E[idim] ) {
write_component( &Efield_group, xyz.substr( idim, 1 ).c_str(), data_double[idim*nParticles_local], H5T_NATIVE_DOUBLE, file_space, mem_space, SMILEI_UNIT_EFIELD );
}
}
}

if( write_any_B ) {
H5Write Bfield_group = species_group->group( "B" );
openPMD_->writeRecordAttributes( Bfield_group, SMILEI_UNIT_BFIELD );
for( unsigned int idim=0; idim<3; idim++ ) {
if( write_B[idim] ) {
write_component( &Bfield_group, xyz.substr( idim, 1 ).c_str(), data_double[( 3+idim )*nParticles_local], H5T_NATIVE_DOUBLE, file_space, mem_space, SMILEI_UNIT_BFIELD );
}
}
}
}
} 

#pragma omp master
{
data_double.resize( 0 );

H5Write positionoffset_group = species_group->group( "positionOffset" );
openPMD_->writeRecordAttributes( positionoffset_group, SMILEI_UNIT_POSITION );
vector<uint64_t> np = {nParticles_global};
for( unsigned int idim=0; idim<nDim_particle; idim++ ) {
H5Write xyz_group = positionoffset_group.group( xyz.substr( idim, 1 ) );
openPMD_->writeComponentAttributes( xyz_group, SMILEI_UNIT_POSITION );
xyz_group.attr( "value", 0. );
xyz_group.attr( "shape", np, H5T_NATIVE_UINT64 );
}

patch_selection.resize( 0 );

delete file_space;
delete mem_space;
delete species_group;

if( flush_timeSelection->theTimeIsNow( itime ) ) {
file_->flush();
}
}
#pragma omp barrier
}


void DiagnosticTrack::setIDs( Patch *patch )
{
if( has_filter ) {
return;
}
unsigned int s = patch->vecSpecies[speciesId_]->particles->size();
for( unsigned int iPart=0; iPart<s; iPart++ ) {
patch->vecSpecies[speciesId_]->particles->id( iPart ) = ++latest_Id;
}
}


void DiagnosticTrack::setIDs( Particles &particles )
{
if( has_filter ) {
return;
}
unsigned int s = particles.size();
#pragma omp critical
{
for( unsigned int iPart=0; iPart<s; iPart++ ) {
particles.id( iPart ) = ++latest_Id;
}
}
}


template<typename T>
void DiagnosticTrack::fill_buffer( VectorPatch &vecPatches, unsigned int iprop, vector<T> &buffer )
{
unsigned int patch_nParticles, i, j, nPatches=vecPatches.size();
vector<T> *property = NULL;

if( has_filter ) {
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<nPatches ; ipatch++ ) {
patch_nParticles = patch_selection[ipatch].size();
vecPatches( ipatch )->vecSpecies[speciesId_]->particles->getProperty( iprop, property );
i=0;
j=patch_start[ipatch];
while( i<patch_nParticles ) {
buffer[j] = ( *property )[patch_selection[ipatch][i]];
i++;
j++;
}
}
} else {
#pragma omp for schedule(runtime)
for( unsigned int ipatch=0 ; ipatch<nPatches ; ipatch++ ) {
patch_nParticles = vecPatches( ipatch )->vecSpecies[speciesId_]->particles->size();
vecPatches( ipatch )->vecSpecies[speciesId_]->particles->getProperty( iprop, property );
i=0;
j=patch_start[ipatch];
while( i<patch_nParticles ) {
buffer[j] = ( *property )[i];
i++;
j++;
}
}
}
}


template<typename T>
void DiagnosticTrack::write_scalar( H5Write * location, string name, T &buffer, hid_t dtype, H5Space *file_space, H5Space *mem_space, unsigned int unit_type )
{
H5Write a = location->array( name, buffer, dtype, file_space, mem_space );
openPMD_->writeRecordAttributes( a, unit_type );
openPMD_->writeComponentAttributes( a, unit_type );
}

template<typename T>
void DiagnosticTrack::write_component( H5Write * location, string name, T &buffer, hid_t dtype, H5Space *file_space, H5Space *mem_space, unsigned int unit_type )
{
H5Write a = location->array( name, buffer, dtype, file_space, mem_space );
openPMD_->writeComponentAttributes( a, unit_type );
}



uint64_t DiagnosticTrack::getDiskFootPrint( int istart, int istop, Patch *patch )
{
uint64_t footprint = 0;

uint64_t ndumps = timeSelection->howManyTimesBefore( istop ) - timeSelection->howManyTimesBefore( istart );

int nparams = 6 + nDim_particle;

footprint += 2500;

footprint += ndumps * 11250;

footprint += ndumps * ( uint64_t )( nparams * npart_total * 8 );

return footprint;
}
