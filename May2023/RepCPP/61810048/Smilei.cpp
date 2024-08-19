
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <omp.h>

#include "Smilei.h"
#include "SmileiMPI_test.h"
#include "Params.h"
#include "PatchesFactory.h"
#include "SyncVectorPatch.h"
#include "Checkpoint.h"
#include "Solver.h"
#include "SimWindow.h"
#include "Diagnostic.h"
#include "Region.h"
#include "DoubleGrids.h"
#include "DoubleGridsAM.h"
#include "Timers.h"

using namespace std;

int main( int argc, char *argv[] )
{
cout.setf( ios::fixed,  ios::floatfield ); 



#ifdef SMILEI_TESTMODE
SmileiMPI_test smpi( &argc, &argv );
#else
SmileiMPI smpi( &argc, &argv );
#endif

MESSAGE( "                   _            _" );
MESSAGE( " ___           _  | |        _  \\ \\   Version : " << __VERSION );
MESSAGE( "/ __|  _ __   (_) | |  ___  (_)  | |   " );
MESSAGE( "\\__ \\ | '  \\   _  | | / -_)  _   | |" );
MESSAGE( "|___/ |_|_|_| |_| |_| \\___| |_|  | |  " );
MESSAGE( "                                /_/    " );
MESSAGE( "" );

TITLE( "Reading the simulation parameters" );
Params params( &smpi, vector<string>( argv + 1, argv + argc ) );
OpenPMDparams openPMD( params );
PyTools::setIteration( 0 );

VectorPatch vecPatches( params );
Region region( params );

TITLE( "Initializing MPI" );
smpi.init( params, vecPatches.domain_decomposition_ );

Timers timers( &smpi );

params.print_parallelism_params( &smpi );

TITLE( "Initializing the restart environment" );
Checkpoint checkpoint( params, &smpi );


double time_prim = 0;
double time_dual = 0.5 * params.timestep;

SimWindow *simWindow = new SimWindow( params );

RadiationTables radiation_tables_;

MultiphotonBreitWheelerTables multiphoton_Breit_Wheeler_tables_;

if( smpi.test_mode ) {
executeTestMode( vecPatches, region, &smpi, simWindow, params, checkpoint, openPMD, &radiation_tables_ );
return 0;
}

radiation_tables_.initialization( params, &smpi);

multiphoton_Breit_Wheeler_tables_.initialization( params, &smpi );

if( params.restart ) {
checkpoint.readPatchDistribution( &smpi, simWindow );
PatchesFactory::createVector( vecPatches, params, &smpi, openPMD, &radiation_tables_, checkpoint.this_run_start_step+1, simWindow->getNmoved() );

if( params.multiple_decomposition ) {
TITLE( "Create SDMD grids" );
checkpoint.readRegionDistribution( region );

int target_map[smpi.getSize()];
MPI_Allgather(&(region.vecPatch_.refHindex_), 1, MPI_INT,
target_map, 1, MPI_INT,
MPI_COMM_WORLD);
region.define_regions_map(target_map, &smpi, params);

region.build( params, &smpi, vecPatches, openPMD, false, simWindow->getNmoved() );
region.identify_additional_patches( &smpi, vecPatches, params, simWindow );
region.identify_missing_patches( &smpi, vecPatches, params );
}






checkpoint.restartAll( vecPatches, region, &smpi, simWindow, params, openPMD );
vecPatches.sortAllParticles( params );

TITLE( "Minimum memory consumption (does not include all temporary buffers)" );
vecPatches.checkMemoryConsumption( &smpi, &region.vecPatch_ );

if( params.has_adaptive_vectorization ) {
vecPatches.configuration( params, timers, 0 );
}

time_prim = checkpoint.this_run_start_step * params.timestep;
time_dual = ( checkpoint.this_run_start_step +0.5 ) * params.timestep;

TITLE( "Open files & initialize diagnostics" );
vecPatches.initAllDiags( params, &smpi );

} else {

PatchesFactory::createVector( vecPatches, params, &smpi, openPMD, &radiation_tables_, 0 );
vecPatches.sortAllParticles( params );

if( params.multiple_decomposition ) {
TITLE( "Create SDMD grids" );
region.vecPatch_.refHindex_ = smpi.getRank();
region.build( params, &smpi, vecPatches, openPMD, false, simWindow->getNmoved() );
region.identify_additional_patches( &smpi, vecPatches, params, simWindow );
region.identify_missing_patches( &smpi, vecPatches, params );

region.reset_fitting( &smpi, params );
region.clean();
region.reset_mapping();

region.build( params, &smpi, vecPatches, openPMD, false, simWindow->getNmoved() );
region.identify_additional_patches( &smpi, vecPatches, params, simWindow );
region.identify_missing_patches( &smpi, vecPatches, params );
}

TITLE( "Minimum memory consumption (does not include all temporary buffers)" );
vecPatches.checkMemoryConsumption( &smpi, &region.vecPatch_ );

TITLE( "Initial fields setup" );

if( params.solve_relativistic_poisson == true ) {
MESSAGE( 1, "Solving relativistic Poisson at time t = 0" );
vecPatches.runRelativisticModule( time_prim, params, &smpi,  timers );
}

vecPatches.computeCharge();
vecPatches.sumDensities( params, time_dual, timers, 0, simWindow, &smpi );

if( params.solve_poisson == true && !vecPatches.isRhoNull( &smpi ) ) {
MESSAGE( 1, "Solving Poisson at time t = 0" );
vecPatches.runNonRelativisticPoissonModule( params, &smpi,  timers );
}

MESSAGE( 1, "Applying external fields at time t = 0" );
vecPatches.applyExternalFields();
vecPatches.saveExternalFields( params );

MESSAGE( 1, "Applying prescribed fields at time t = 0" );
vecPatches.applyPrescribedFields( time_prim );

MESSAGE( 1, "Applying antennas at time t = 0" );
vecPatches.applyAntennas( 0.5 * params.timestep );

if( params.has_adaptive_vectorization ) {
vecPatches.configuration( params, timers, 0 );
}

if( params.Laser_Envelope_model ) {
MESSAGE( 1, "Initialize envelope" );
vecPatches.initNewEnvelope( params );
}

vecPatches.projectionForDiags( params, &smpi, simWindow, time_dual, timers, 0 );

if( params.Laser_Envelope_model ) {
vecPatches.sumSusceptibility( params, time_dual, timers, 0, simWindow, &smpi );
}

vecPatches.sumDensities( params, time_dual, timers, 0, simWindow, &smpi );

if( params.multiple_decomposition ) {
if ( params.geometry != "AMcylindrical" ) {
DoubleGrids::syncFieldsOnRegion( vecPatches, region, params, &smpi );
SyncVectorPatch::exchangeE( params, region.vecPatch_, &smpi );
SyncVectorPatch::finalizeexchangeE( params, region.vecPatch_);
SyncVectorPatch::exchangeB( params, region.vecPatch_, &smpi );
SyncVectorPatch::finalizeexchangeB( params, region.vecPatch_);
} else {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  ) {
DoubleGridsAM::syncFieldsOnRegion( vecPatches, region, params, &smpi, imode );
SyncVectorPatch::exchangeE( params, region.vecPatch_, imode, &smpi );
SyncVectorPatch::exchangeB( params, region.vecPatch_, imode, &smpi );
}
}
}

if( params.initial_rotational_cleaning ) {
TITLE( "Rotational cleaning" );
Region region_global( params );
region_global.build( params, &smpi, vecPatches, openPMD, true, simWindow->getNmoved() );
region_global.identify_additional_patches( &smpi, vecPatches, params, simWindow );
region_global.identify_missing_patches( &smpi, vecPatches, params );
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  ) {
DoubleGridsAM::syncFieldsOnRegion( vecPatches, region_global, params, &smpi, imode );
}
if( params.is_pxr && smpi.isMaster()) {
region_global.coupling( params, true );
}
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  ) {
DoubleGridsAM::syncFieldsOnPatches( region_global, vecPatches, params, &smpi, timers, 0, imode );
}
vecPatches.setMagneticFieldsForDiagnostic( params );
region_global.clean();

if( params.multiple_decomposition ) {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  ) {
DoubleGridsAM::syncFieldsOnRegion( vecPatches, region, params, &smpi, imode );
SyncVectorPatch::exchangeE( params, region.vecPatch_, imode, &smpi );
SyncVectorPatch::exchangeB( params, region.vecPatch_, imode, &smpi );
}
}
}


TITLE( "Open files & initialize diagnostics" );
vecPatches.initAllDiags( params, &smpi );
TITLE( "Running diags at time t = 0" );
vecPatches.runAllDiags( params, &smpi, 0, timers, simWindow );
}

TITLE( "Species creation summary" );
vecPatches.printGlobalNumberOfParticlesPerSpecies( &smpi );

if( params.is_pxr ){
if( params.multiple_decomposition ) {
region.coupling( params, false );
} else {
vecPatches( 0 )->EMfields->MaxwellAmpereSolver_->coupling( params, vecPatches( 0 )->EMfields );
}
}

if( params.is_spectral && params.geometry != "AMcylindrical") {
vecPatches.saveOldRho( params );
}

timers.reboot();
timers.global.reboot();

TITLE( "Expected disk usage (approximate)" );
vecPatches.checkExpectedDiskUsage( &smpi, params, checkpoint );

TITLE( "Keeping or closing the python runtime environment" );
params.cleanup( &smpi );



TITLE( "Time-Loop started: number of time-steps n_time = " << params.n_time );
if( smpi.isMaster() ) {
params.print_timestep_headers( &smpi );
}

int count_dlb = 0;

unsigned int itime=checkpoint.this_run_start_step+1;
while( ( itime <= params.n_time ) && ( !checkpoint.exit_asap ) ) {

#pragma omp parallel shared (time_dual,smpi,params, vecPatches, region, simWindow, checkpoint, itime)
{

#pragma omp single
{
time_prim += params.timestep;
time_dual += params.timestep;
if( params.keep_python_running_ ) {
PyTools::setIteration( itime ); 
}
}
#pragma omp barrier

if( params.has_adaptive_vectorization && params.adaptive_vecto_time_selection->theTimeIsNow( itime ) ) {
vecPatches.reconfiguration( params, timers, itime );
}

vecPatches.applyBinaryProcesses( params, itime, timers );

if( params.solve_relativistic_poisson == true ) {
vecPatches.runRelativisticModule( time_prim, params, &smpi,  timers );
}

if ( params.geometry == "AMcylindrical" && params.is_spectral )
vecPatches.computeCharge(true);

vecPatches.dynamics( params, &smpi, simWindow, radiation_tables_,
multiphoton_Breit_Wheeler_tables_,
time_dual, timers, itime );

if( params.Laser_Envelope_model ) {
vecPatches.runEnvelopeModule( params, &smpi, simWindow, time_dual, timers, itime );
} 

vecPatches.sumDensities( params, time_dual, timers, itime, simWindow, &smpi );

vecPatches.applyAntennas( time_dual );

} 

if (!params.multiple_decomposition) {
if( time_dual > params.time_fields_frozen ) {
#pragma omp parallel shared (time_dual,smpi,params, vecPatches, region, simWindow, checkpoint, itime)
{
if ( vecPatches(0)->EMfields->prescribedFields.size() ) {
vecPatches.resetPrescribedFields();
}
vecPatches.solveMaxwell( params, simWindow, itime, time_dual, timers, &smpi );
}

}
}
else { 
if( time_dual > params.time_fields_frozen ) {
if ( params.geometry != "AMcylindrical" )
DoubleGrids::syncCurrentsOnRegion( vecPatches, region, params, &smpi, timers, itime );
else {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  )
DoubleGridsAM::syncCurrentsOnRegion( vecPatches, region, params, &smpi, timers, itime, imode );
}
region.vecPatch_.diag_flag = false;

if ( params.is_spectral && params.geometry == "AMcylindrical") {
timers.densitiesCorrection.restart();
region.vecPatch_( 0 )->EMfields->MaxwellAmpereSolver_->densities_correction( region.vecPatch_( 0 )->EMfields );
timers.densitiesCorrection.update();
}


timers.syncDens.restart();
if( params.geometry != "AMcylindrical" )
SyncVectorPatch::sumRhoJ( params, region.vecPatch_, &smpi, timers, itime ); 
else
for( unsigned int imode = 0 ; imode < params.nmodes ; imode++ ) {
SyncVectorPatch::sumRhoJ( params, region.vecPatch_, imode, &smpi, timers, itime );
}
timers.syncDens.update( params.printNow( itime ) );


if( region.vecPatch_(0)->EMfields->prescribedFields.size() ) {
region.vecPatch_.applyPrescribedFields( time_prim );
}

region.solveMaxwell( params, simWindow, itime, time_dual, timers, &smpi );
if ( params.geometry != "AMcylindrical" )
DoubleGrids::syncFieldsOnPatches( region, vecPatches, params, &smpi, timers, itime );
else {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  )
DoubleGridsAM::syncFieldsOnPatches( region, vecPatches, params, &smpi, timers, itime, imode );
}
}
if( vecPatches.diag_flag ) {

if (!params.is_spectral) {
if ( params.geometry != "AMcylindrical" )
DoubleGrids::syncBOnPatches( region, vecPatches, params, &smpi, timers, itime );
else {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  )
DoubleGridsAM::syncBOnPatches( region, vecPatches, params, &smpi, timers, itime, imode );
}

#pragma omp parallel shared (time_dual,smpi,params, vecPatches, region, simWindow, checkpoint, itime)
{
if( params.geometry != "AMcylindrical" ) {
SyncVectorPatch::sumRhoJ( params, vecPatches, &smpi, timers, itime ); 
}
else {
for( unsigned int imode = 0 ; imode < params.nmodes ; imode++ ) {
SyncVectorPatch::sumRhoJ( params, vecPatches, imode, &smpi, timers, itime );
}
}
}
}
else {
vecPatches.setMagneticFieldsForDiagnostic( params );

if ( params.geometry != "AMcylindrical" ) {
DoubleGrids::syncCurrentsOnPatches( region, vecPatches, params, &smpi, timers, itime );
}
else {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  )
DoubleGridsAM::syncCurrentsOnPatches( region, vecPatches, params, &smpi, timers, itime, imode );
}
}
}
bool old = (params.geometry == "AMcylindrical" && params.is_spectral);
region.vecPatch_.resetRhoJ(old);
}

#pragma omp parallel shared (time_dual,smpi,params, vecPatches, region, simWindow, checkpoint, itime)
{
vecPatches.finalizeAndSortParticles( params, &smpi, simWindow,
time_dual, timers, itime );

vecPatches.mergeParticles(params, &smpi, time_dual,timers, itime );

vecPatches.injectParticlesFromBoundaries(params, timers, itime );

vecPatches.cleanParticlesOverhead(params, timers, itime );

vecPatches.finalizeSyncAndBCFields( params, &smpi, simWindow, time_dual, timers, itime );

if( !params.multiple_decomposition ) {
if( time_dual > params.time_fields_frozen ) {
if( vecPatches(0)->EMfields->prescribedFields.size() ) {
#pragma omp single
vecPatches.applyPrescribedFields( time_prim );
#pragma omp barrier
}
}
}

vecPatches.runAllDiags( params, &smpi, itime, timers, simWindow );

timers.movWindow.restart();
simWindow->shift( vecPatches, &smpi, params, itime, time_dual, region );

if (itime == simWindow->getAdditionalShiftsIteration() ) {
int adjust = simWindow->isMoving(time_dual)?0:1;
for (unsigned int n=0;n < simWindow->getNumberOfAdditionalShifts()-adjust; n++)
simWindow->shift( vecPatches, &smpi, params, itime, time_dual, region );
}
timers.movWindow.update();
#pragma omp master
checkpoint.dump( vecPatches, region, itime, &smpi, simWindow, params );
#pragma omp barrier

} 

if( params.has_load_balancing && params.load_balancing_time_selection->theTimeIsNow( itime ) ) {
count_dlb++;
if (params.multiple_decomposition && count_dlb%5 ==0 ) {
if ( params.geometry != "AMcylindrical" ) {
DoubleGrids::syncBOnPatches( region, vecPatches, params, &smpi, timers, itime );
} else {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  ) {
DoubleGridsAM::syncBOnPatches( region, vecPatches, params, &smpi, timers, itime, imode );
}
}
}

timers.loadBal.restart();
#pragma omp single
vecPatches.loadBalance( params, time_dual, &smpi, simWindow, itime );
timers.loadBal.update( params.printNow( itime ) );

if( params.multiple_decomposition ) {

if( count_dlb%5 == 0 ) {
region.reset_fitting( &smpi, params );
region.clean();
region.reset_mapping();
region.build( params, &smpi, vecPatches, openPMD, false, simWindow->getNmoved() );
if( params.is_pxr ) {
region.coupling( params, false );
}
region.identify_additional_patches( &smpi, vecPatches, params, simWindow );
region.identify_missing_patches( &smpi, vecPatches, params );

if ( params.geometry != "AMcylindrical" ) {
DoubleGrids::syncFieldsOnRegion( vecPatches, region, params, &smpi );
} else {
for (unsigned int imode = 0 ; imode < params.nmodes ; imode++  ) {
DoubleGridsAM::syncFieldsOnRegion( vecPatches, region, params, &smpi, imode );
}
}

} else {
region.reset_mapping();
region.identify_additional_patches( &smpi, vecPatches, params, simWindow );
region.identify_missing_patches( &smpi, vecPatches, params );
}
}
}

if( params.printNow( itime ) ) {
double npart = vecPatches.getGlobalNumberOfParticles( &smpi );
params.print_timestep( &smpi, itime, time_dual, timers.global, npart ); 

#pragma omp master
timers.consolidate( &smpi );
#pragma omp barrier
}

itime++;

}

smpi.barrier();

TITLE( "End time loop, time dual = " << time_dual );
timers.global.update();

TITLE( "Time profiling : (print time > 0.001%)" );
timers.profile( &smpi );

smpi.barrier();



if (params.multiple_decomposition) {
region.clean();
}
vecPatches.close( &smpi );
smpi.barrier(); 
delete simWindow;
PyTools::closePython();
TITLE( "END" );

return 0;

}



int executeTestMode( VectorPatch &vecPatches,
Region &region,
SmileiMPI *smpi,
SimWindow *simWindow,
Params &params,
Checkpoint &checkpoint,
OpenPMDparams &openPMD,
RadiationTables * radiation_tables_ )
{
int itime = 0;
int moving_window_movement = 0;

if( params.restart ) {
checkpoint.readPatchDistribution( smpi, simWindow );
itime = checkpoint.this_run_start_step+1;
moving_window_movement = simWindow->getNmoved();
}

PatchesFactory::createVector( vecPatches, params, smpi, openPMD, radiation_tables_, itime, moving_window_movement );

if( params.restart ) {
if (params.multiple_decomposition) {
checkpoint.readRegionDistribution( region );
region.build( params, smpi, vecPatches, openPMD, false, simWindow->getNmoved() );
}
checkpoint.restartAll( vecPatches, region, smpi, simWindow, params, openPMD );
}

if( params.print_expected_disk_usage ) {
TITLE( "Expected disk usage (approximate)" );
vecPatches.checkExpectedDiskUsage( smpi, params, checkpoint );
}

TITLE( "Keeping or closing the python runtime environment" );
params.cleanup( smpi );
delete simWindow;
PyTools::closePython();
TITLE( "END TEST MODE" );

return 0;
}
