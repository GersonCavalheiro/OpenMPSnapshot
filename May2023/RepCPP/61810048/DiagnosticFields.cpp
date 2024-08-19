
#include <algorithm>

#include "DiagnosticFields.h"
#include "VectorPatch.h"

using namespace std;

DiagnosticFields::DiagnosticFields( Params &params, SmileiMPI *smpi, VectorPatch &vecPatches, int ndiag, OpenPMDparams &oPMD ):
Diagnostic( &oPMD, "DiagFields", ndiag )
{
diag_n = ndiag;

filespace = NULL;
memspace = NULL;

time_average = 1;
PyTools::extract( "time_average", time_average, "DiagFields", ndiag );
if( time_average < 1 ) {
time_average = 1;
}
time_average_inv = 1./( ( double )time_average );

ostringstream fn( "" );
fn << "Fields"<< ndiag <<".h5";
filename = fn.str();

vector<string> fieldsToDump( 0 );
PyTools::extractV( "fields", fieldsToDump, "DiagFields", ndiag );

if (params.geometry == "AMcylindrical") {
vector<string> fieldsToAdd( 0 );
for( unsigned int ifield = 0; ifield < fieldsToDump.size(); ifield++ ){
if( fieldsToDump[ifield].find("_mode_") ==  std::string::npos ) {
for( unsigned int imode = 1; imode < params.nmodes; imode ++ ){
fieldsToAdd.push_back(fieldsToDump[ifield]+"_mode_"+to_string(imode));
}
fieldsToDump[ifield] = fieldsToDump[ifield] + "_mode_0" ;
}
}
for( unsigned int ifield = 0; ifield < fieldsToAdd.size(); ifield++ ){
fieldsToDump.push_back(fieldsToAdd[ifield]);
}
}

ostringstream ss( "" );
fields_indexes.resize( 0 );
fields_names  .resize( 0 );
hasRhoJs = false;
vector<string> allFields( vecPatches.emfields( 0 )->allFields.size() );
for( unsigned int i=0; i<allFields.size(); i++ ) {
allFields[i] = vecPatches.emfields( 0 )->allFields[i]->name;
}
if( fieldsToDump.size()==0 ) {
fieldsToDump = allFields;
}
for( unsigned int j=0; j<fieldsToDump.size(); j++ ) {
bool hasfield = false;
for( unsigned int i=0; i<allFields.size(); i++ ) {
if( fieldsToDump[j] == allFields[i] ) {
ss << allFields[i] << " ";
fields_indexes.push_back( i );
fields_names  .push_back( allFields[i] );
if( allFields[i].at( 0 )=='J' || allFields[i].at( 0 )=='R' ) {
hasRhoJs = true;
}
if( params.speciesField( allFields[i] ) != "" ) {
vecPatches.allocateField( i, params );
}
hasfield = true;
break;
}
}
if( ! hasfield ) {
ERROR_NAMELIST( 
"Diagnostic Fields #"<<ndiag
<<": field `"<<fieldsToDump[j]
<<"` does not exist",
LINK_NAMELIST + std::string("#particle-merging")
);
}
}

PyObject *subgrid = PyTools::extract_py( "subgrid", "DiagFields", ndiag );
vector<PyObject *> subgrids;
if( subgrid == Py_None ) {
subgrids.resize( params.nDim_field, Py_None );
} else if( ! PySequence_Check( subgrid ) ) {
subgrids.push_back( subgrid );
} else {
Py_ssize_t ns = PySequence_Length( subgrid );
for( Py_ssize_t is=0; is<ns; is++ ) {
subgrids.push_back( PySequence_Fast_GET_ITEM( subgrid, is ) );
}
}
Py_DECREF( subgrid );
unsigned int nsubgrid = subgrids.size();
if( nsubgrid != params.nDim_field ) {
ERROR( "Diagnostic Fields #"<<ndiag<<" `subgrid` containing "<<nsubgrid<<" axes whereas simulation dimension is "<<params.nDim_field );
}
for( unsigned int isubgrid=0; isubgrid<nsubgrid; isubgrid++ ) {
unsigned int n;
if( subgrids[isubgrid] == Py_None ) {
subgrid_start_.push_back( 0 );
subgrid_stop_ .push_back( params.n_space_global[isubgrid]+2 );
subgrid_step_ .push_back( 1 );
} else if( PyTools::py2scalar( subgrids[isubgrid], n ) ) {
subgrid_start_.push_back( n );
subgrid_stop_ .push_back( n + 1 );
subgrid_step_ .push_back( 1 );
} else if( PySlice_Check( subgrids[isubgrid] ) ) {
Py_ssize_t start, stop, step, slicelength;
#if PY_MAJOR_VERSION == 2
if( PySlice_GetIndicesEx( ( PySliceObject * )subgrids[isubgrid], params.n_space_global[isubgrid]+1, &start, &stop, &step, &slicelength ) < 0 ) {
#else
if( PySlice_GetIndicesEx( subgrids[isubgrid], params.n_space_global[isubgrid]+1, &start, &stop, &step, &slicelength ) < 0 ) {
#endif
PyTools::checkPyError();
ERROR( "Diagnostic Fields #"<<ndiag<<" `subgrid` axis #"<<isubgrid<<" not understood" );
}
subgrid_start_.push_back( start );
subgrid_stop_ .push_back( stop );
subgrid_step_ .push_back( step );
if( slicelength < 1 ) {
ERROR( "Diagnostic Fields #"<<ndiag<<" `subgrid` axis #"<<isubgrid<<" is an empty selection" );
}
} else {
ERROR( "Diagnostic Fields #"<<ndiag<<" `subgrid` axis #"<<isubgrid<<" must be an integer or a slice" );
}
}

ostringstream p( "" );
p << "(time average = " << time_average << ")";
MESSAGE( 1, "Diagnostic Fields #"<<ndiag<<" "<<( time_average>1?p.str():"" )<<" :" );
MESSAGE( 2, ss.str() );

if( ! smpi->test_mode ) {
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
vecPatches( ipatch )->EMfields->allFields_avg.resize( diag_n+1 );
if( time_average > 1 ) {
for( unsigned int ifield=0; ifield<fields_names.size(); ifield++ )
vecPatches( ipatch )->EMfields->allFields_avg[diag_n].push_back(
vecPatches( ipatch )->EMfields->createField( fields_names[ifield],params )
);
}
}
}

timeSelection = new TimeSelection( PyTools::extract_py( "every", "DiagFields", ndiag ), "DiagFields" );

if( timeSelection->smallestInterval() < time_average ) {
ERROR( "Diagnostic Fields #"<<ndiag<<" has a time average too large compared to its time-selection interval ('every')" );
}

flush_timeSelection = new TimeSelection( PyTools::extract_py( "flush_every", "DiagFields", ndiag ), "DiagFields flush_every" );

tot_number_of_patches = params.tot_number_of_patches;

field_type.resize( fields_names.size() );
for( unsigned int ifield=0; ifield<fields_names.size(); ifield++ ) {
string first_char = fields_names[ifield].substr( 0, 1 );
if( first_char == "E" ) {
field_type[ifield] = SMILEI_UNIT_EFIELD;
} else if( first_char == "B" ) {
field_type[ifield] = SMILEI_UNIT_BFIELD;
} else if( first_char == "J" ) {
field_type[ifield] = SMILEI_UNIT_CURRENT;
} else if( first_char == "R" ) {
field_type[ifield] = SMILEI_UNIT_DENSITY;
} else {
ERROR( " impossible field name " );
}
}
}


DiagnosticFields::~DiagnosticFields()
{
closeFile();
if( filespace ) {
delete filespace;
}
if( memspace ) {
delete memspace;
}
delete timeSelection;
delete flush_timeSelection;
}

void DiagnosticFields::openFile( Params &params, SmileiMPI *smpi )
{
if( file_ ) {
return;
}

file_ = new H5Write( filename, &smpi->world() );

file_->attr( "name", diag_name_ );

openPMD_->writeRootAttributes( *file_, "", "no_particles" );

data_group_ = new H5Write( file_, "data" );

file_->flush();
}

void DiagnosticFields::closeFile()
{
if( data_group_ ) {
delete data_group_;
data_group_ = NULL;
}
if( file_ ) {
delete file_;
file_ = NULL;
}
}



void DiagnosticFields::init( Params &params, SmileiMPI *smpi, VectorPatch &vecPatches )
{
openFile( params, smpi );
}

bool DiagnosticFields::prepare( int itime )
{

if( itime - timeSelection->previousTime( itime ) >= time_average ) {
return false;
}

return true;
}


void DiagnosticFields::run( SmileiMPI *smpi, VectorPatch &vecPatches, int itime, SimWindow *simWindow, Timers &timers )
{
if( time_average>1 ) {
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<vecPatches.size() ; ipatch++ ) {
for( unsigned int ifield=0; ifield<fields_names.size(); ifield++ ) {
vecPatches( ipatch )->EMfields->incrementAvgField(
vecPatches( ipatch )->EMfields->allFields[fields_indexes[ifield]], 
vecPatches( ipatch )->EMfields->allFields_avg[diag_n][ifield]    
);
}
}
}

if( itime - timeSelection->previousTime( itime ) != time_average-1 ) {
return;
}

#pragma omp master
{
refHindex = ( unsigned int )( vecPatches.refHindex_ );
setFileSplitting( smpi, vecPatches );

ostringstream name_t;
name_t << setfill( '0' ) << setw( 10 ) << itime;
status = data_group_->has( name_t.str() );
if( ! status ) {
iteration_group_ = new H5Write( data_group_, name_t.str() );
openPMD_->writeBasePathAttributes( *iteration_group_, itime );
openPMD_->writeMeshesAttributes( *iteration_group_ );
}
}
#pragma omp barrier

if( status ) {
return;
}

unsigned int nPatches( vecPatches.size() );

for( unsigned int ifield=0; ifield < fields_indexes.size(); ifield++ ) {

#pragma omp barrier
#pragma omp for schedule(static)
for( unsigned int ipatch=0 ; ipatch<nPatches ; ipatch++ ) {
getField( vecPatches( ipatch ), ifield );
}

#pragma omp master
{
H5Write dset = writeField( iteration_group_, fields_names[ifield], itime );
openPMD_->writeFieldAttributes( dset, subgrid_start_, subgrid_step_ );
openPMD_->writeRecordAttributes( dset, field_type[ifield] );
openPMD_->writeFieldRecordAttributes( dset );
openPMD_->writeComponentAttributes( dset, field_type[ifield] );
}
#pragma omp barrier 
}

#pragma omp master
{

double x_moved = simWindow ? simWindow->getXmoved() : 0.;
iteration_group_->attr( "x_moved", x_moved );
delete iteration_group_;
if( flush_timeSelection->theTimeIsNow( itime ) ) {
file_->flush();
}
}
#pragma omp barrier
}

bool DiagnosticFields::needsRhoJs( int itime )
{

return hasRhoJs && (itime - timeSelection->previousTime( itime ) < time_average);
}

uint64_t DiagnosticFields::getDiskFootPrint( int istart, int istop, Patch *patch )
{
uint64_t footprint = 0;
uint64_t nfields = fields_indexes.size();

uint64_t ndumps = timeSelection->howManyTimesBefore( istop ) - timeSelection->howManyTimesBefore( istart );

footprint += 2500;

footprint += ndumps * 2200;

footprint += ndumps * nfields * 1200;

footprint += ndumps * nfields * ( uint64_t )( total_dataset_size * 8 );

return footprint;
}

void DiagnosticFields::findSubgridIntersection(
unsigned int subgrid_start,
unsigned int subgrid_stop,
unsigned int subgrid_step,
unsigned int zone_begin,
unsigned int zone_end,
unsigned int &istart_in_zone,  
unsigned int &istart_in_file,  
unsigned int &nsteps  
)
{
unsigned int start, stop;
if( zone_begin <= subgrid_start ) {
istart_in_zone = subgrid_start - zone_begin;
istart_in_file = 0;
if( zone_end <= subgrid_start ) {
nsteps = 0;
} else {
stop = min( zone_end, subgrid_stop );
if( stop <= subgrid_start ) {
stop = subgrid_start + 1;
}
nsteps = ( stop - subgrid_start - 1 ) / subgrid_step + 1;
}
} else {
if( zone_begin >= subgrid_stop ) {
istart_in_zone = 0;
istart_in_file = 0;
nsteps = 0;
} else {
istart_in_file = ( zone_begin - subgrid_start - 1 ) / subgrid_step + 1;
start = subgrid_start + istart_in_file * subgrid_step;
istart_in_zone = start - zone_begin;
stop = min( zone_end, subgrid_stop );
if( stop <= start ) {
nsteps = 0;
} else {
nsteps = ( stop - start - 1 ) / subgrid_step + 1;
}
}
}
}

void DiagnosticFields::findSubgridIntersection1(
hsize_t idim,
hsize_t &zone_offset,  
hsize_t &zone_npoints, 
hsize_t &start_in_zone 
)
{
unsigned int istart_in_zone, istart_in_file, nsteps;
findSubgridIntersection(
subgrid_start_[idim], subgrid_stop_[idim], subgrid_step_[idim],
zone_offset, zone_offset + zone_npoints,
istart_in_zone, istart_in_file, nsteps
);
zone_offset = istart_in_file;
zone_npoints = nsteps;
start_in_zone = istart_in_zone;
}