#include "dem/mappings/MoveParticles.h"

#include "peano/utils/Loop.h"

#include "tarch/la/Matrix.h"

#include "dem/mappings/Collision.h"

#include "dem/mappings/CreateGrid.h"

#include "delta/dynamics/dynamics.h"

peano::CommunicationSpecification   dem::mappings::MoveParticles::communicationSpecification() const {
return peano::CommunicationSpecification(peano::CommunicationSpecification::ExchangeMasterWorkerData::SendDataAndStateBeforeFirstTouchVertexFirstTime,peano::CommunicationSpecification::ExchangeWorkerMasterData::SendDataAndStateAfterLastTouchVertexLastTime,false);
}


peano::MappingSpecification   dem::mappings::MoveParticles::touchVertexFirstTimeSpecification(int level) const {
return peano::MappingSpecification(
peano::MappingSpecification::WholeTree,
dem::mappings::Collision::RunGridTraversalInParallel ?
peano::MappingSpecification::RunConcurrentlyOnFineGrid :
peano::MappingSpecification::Serial,
true
);
}


peano::MappingSpecification   dem::mappings::MoveParticles::enterCellSpecification(int level) const {
return peano::MappingSpecification(
peano::MappingSpecification::WholeTree,
dem::mappings::Collision::RunGridTraversalInParallel ?
peano::MappingSpecification::AvoidFineGridRaces :
peano::MappingSpecification::Serial,
true);
}

peano::MappingSpecification   dem::mappings::MoveParticles::leaveCellSpecification(int level) const {
return peano::MappingSpecification(peano::MappingSpecification::Nop,peano::MappingSpecification::AvoidFineGridRaces,true);
}


peano::MappingSpecification   dem::mappings::MoveParticles::touchVertexLastTimeSpecification(int level) const {
return peano::MappingSpecification(peano::MappingSpecification::Nop,peano::MappingSpecification::AvoidFineGridRaces,true);
}

peano::MappingSpecification   dem::mappings::MoveParticles::ascendSpecification(int level) const {
return peano::MappingSpecification(peano::MappingSpecification::Nop,peano::MappingSpecification::AvoidCoarseGridRaces,true);
}
peano::MappingSpecification   dem::mappings::MoveParticles::descendSpecification(int level) const {
return peano::MappingSpecification(peano::MappingSpecification::Nop,peano::MappingSpecification::AvoidCoarseGridRaces,true);
}

tarch::logging::Log		dem::mappings::MoveParticles::_log( "dem::mappings::MoveParticles" );
tarch::multicore::BooleanSemaphore  dem::mappings::MoveParticles::_MoveParticlesSemaphore;

void dem::mappings::MoveParticles::moveAllParticlesAssociatedToVertex(dem::Vertex&	fineGridVertex, State& state)
{
iREAL timeStepSize = state.getTimeStepSize();

for(int i=0; i<fineGridVertex.getNumberOfParticles(); i++)
{
records::Particle&  particle = fineGridVertex.getParticle(i);

if(particle.getIsObstacle()) continue;

particle._persistentRecords._velocity(1) += timeStepSize*(dem::mappings::Collision::gravity*-9.8); 

particle._persistentRecords._centre(0) += timeStepSize*particle._persistentRecords._velocity(0);
particle._persistentRecords._centre(1) += timeStepSize*particle._persistentRecords._velocity(1);
particle._persistentRecords._centre(2) += timeStepSize*particle._persistentRecords._velocity(2);

particle._persistentRecords._centreOfMass(0) += timeStepSize*particle._persistentRecords._velocity(0);
particle._persistentRecords._centreOfMass(1) += timeStepSize*particle._persistentRecords._velocity(1);
particle._persistentRecords._centreOfMass(2) += timeStepSize*particle._persistentRecords._velocity(2);


iREAL v = sqrt(particle._persistentRecords._velocity(0)*particle._persistentRecords._velocity(0) + particle._persistentRecords._velocity(1) * particle._persistentRecords._velocity(1) + particle._persistentRecords._velocity(2) * particle._persistentRecords._velocity(2));
if(v > state.getMaximumVelocityTravel())
{
state.setMaximumVelocityTravel(v);
}

delta::dynamics::updateRotationMatrix(&particle._persistentRecords._angular(0),
&particle._persistentRecords._referentialAngular(0),
&particle._persistentRecords._orientation(0), timeStepSize);

iREAL* x = fineGridVertex.getXCoordinates(i);
iREAL* y = fineGridVertex.getYCoordinates(i);
iREAL* z = fineGridVertex.getZCoordinates(i);

iREAL* refx = fineGridVertex.getXRefCoordinates(i);
iREAL* refy = fineGridVertex.getYRefCoordinates(i);
iREAL* refz = fineGridVertex.getZRefCoordinates(i);

for(int j=0; j<particle.getNumberOfTriangles()*DIMENSIONS; j++)
{
delta::dynamics::updateVertices(&x[j], &y[j], &z[j], &refx[j], &refy[j], &refz[j],
&particle._persistentRecords._orientation(0),
&particle._persistentRecords._centreOfMass(0),
&particle._persistentRecords._referentialCentreOfMass(0));
}
}
}

void dem::mappings::MoveParticles::reassignParticles(
dem::Vertex * const                        fineGridVertices,
const peano::grid::VertexEnumerator&       fineGridVerticesEnumerator,
State& state)
{
int numberOfReassignments = 0;
dfor2(k) 
if(!fineGridVertices[fineGridVerticesEnumerator(k)].isHangingNode())
for(int i=0; i<fineGridVertices[fineGridVerticesEnumerator(k)].getNumberOfParticles(); i++)
{
records::Particle&  particle = fineGridVertices[fineGridVerticesEnumerator(k)].getParticle(i);
tarch::la::Vector<DIMENSIONS,int> correctVertex;

for(int d=0; d<DIMENSIONS; d++)
{
correctVertex(d) = particle._persistentRecords._centre(d) < fineGridVerticesEnumerator.getCellCenter()(d) ? 0 : 1;
}

if(correctVertex != k)
{
fineGridVertices[fineGridVerticesEnumerator(correctVertex)].appendParticle(particle);
fineGridVertices[fineGridVerticesEnumerator(k)].releaseParticle(i);
numberOfReassignments++;


}
}
enddforx
state.incNumberOfParticleReassignments(numberOfReassignments);
}

void dem::mappings::MoveParticles::reflectParticles(dem::Vertex& fineGridVertex)
{
assertion(fineGridVertex.isBoundary());

for (int i=0; i<fineGridVertex.getNumberOfParticles(); i++)
{
records::Particle&  particle = fineGridVertex.getParticle(i);

if (particle.getCentre()[0]-particle.getDiameter()/2.0 < 0.0)
{
particle._persistentRecords._velocity(0) = std::abs(particle.getCentre()[0]-particle.getDiameter()/2.0) * std::abs(particle.getVelocity()[0]);
}

if (particle.getCentre()[1]-particle.getDiameter()/2.0 < 0.0)
{
particle._persistentRecords._velocity(1) = std::abs(particle.getCentre()[1]-particle.getDiameter()/2.0) * std::abs(particle.getVelocity()[1]);
}

if (particle.getCentre()[2]-particle.getDiameter()/2.0 < 0.0)
{
particle._persistentRecords._velocity(2) = std::abs(particle.getCentre()[2]-particle.getDiameter()/2.0) * std::abs(particle.getVelocity()[2]);
}


if (particle.getCentre()[0]-particle.getDiameter()/2.0 > 1.0)
{
particle._persistentRecords._velocity(0) = std::abs(particle.getCentre()[0]-particle.getDiameter()/2.0) * -std::abs(particle.getVelocity()[0]);
}

if (particle.getCentre()[1]-particle.getDiameter()/2.0 > 1.0)
{
particle._persistentRecords._velocity(1) = std::abs(particle.getCentre()[1]-particle.getDiameter()/2.0) * -std::abs(particle.getVelocity()[1]);
}

if (particle.getCentre()[2]-particle.getDiameter()/2.0 > 1.0)
{
particle._persistentRecords._velocity(2) = std::abs(particle.getCentre()[2]-particle.getDiameter()/2.0) * -std::abs(particle.getVelocity()[2]);
}
}
}

void dem::mappings::MoveParticles::beginIteration(
dem::State&  solverState
) {
logTraceInWith1Argument( "beginIteration(State)", solverState );

_state = solverState;
_state.clearAccumulatedData();

logTraceOutWith1Argument( "beginIteration(State)", solverState);
}

void dem::mappings::MoveParticles::endIteration(
dem::State&  solverState
) {
logTraceInWith1Argument( "endIteration(State)", solverState );

solverState.merge(_state);

logTraceOutWith1Argument( "endIteration(State)", solverState);
}

dem::mappings::MoveParticles::MoveParticles() {
}

dem::mappings::MoveParticles::~MoveParticles() {
}

#if defined(SharedMemoryParallelisation)
dem::mappings::MoveParticles::MoveParticles(const MoveParticles&  masterThread):
_state(masterThread._state) {
_state.clearAccumulatedData();
}

void dem::mappings::MoveParticles::mergeWithWorkerThread(const MoveParticles& workerThread) {
logTraceIn( "mergeWithWorkerThread(MoveParticles)" );

_state.merge(workerThread._state);

logTraceOut( "mergeWithWorkerThread(MoveParticles)" );
}
#endif

void dem::mappings::MoveParticles::touchVertexFirstTime(
dem::Vertex&               fineGridVertex,
const tarch::la::Vector<DIMENSIONS,iREAL>&                          fineGridX,
const tarch::la::Vector<DIMENSIONS,iREAL>&                          fineGridH,
dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
dem::Cell&                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
logTraceInWith6Arguments( "touchVertexFirstTime(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

if (fineGridVertex.isBoundary())
{
reflectParticles(fineGridVertex);
}

moveAllParticlesAssociatedToVertex(fineGridVertex, _state);

logTraceOutWith1Argument( "touchVertexFirstTime(...)", fineGridVertex );
}

void dem::mappings::MoveParticles::enterCell(
dem::Cell&                                 fineGridCell,
dem::Vertex * const                        fineGridVertices,
const peano::grid::VertexEnumerator&       fineGridVerticesEnumerator,
dem::Vertex * const                        coarseGridVertices,
const peano::grid::VertexEnumerator&       coarseGridVerticesEnumerator,
dem::Cell&                                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&   fineGridPositionOfCell
) {
logTraceInWith4Arguments( "enterCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );

reassignParticles(fineGridVertices, fineGridVerticesEnumerator, _state);

logTraceOutWith1Argument( "enterCell(...)", fineGridCell );
}

void dem::mappings::MoveParticles::leaveCell(
dem::Cell&           fineGridCell,
dem::Vertex * const  fineGridVertices,
const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
dem::Vertex * const  coarseGridVertices,
const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
dem::Cell&           coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfCell
) {
logTraceInWith4Arguments( "leaveCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );
logTraceOutWith1Argument( "leaveCell(...)", fineGridCell );
}

void dem::mappings::MoveParticles::touchVertexLastTime(
dem::Vertex&         fineGridVertex,
const tarch::la::Vector<DIMENSIONS,iREAL>&                    fineGridX,
const tarch::la::Vector<DIMENSIONS,iREAL>&                    fineGridH,
dem::Vertex * const  coarseGridVertices,
const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
dem::Cell&           coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
logTraceInWith6Arguments( "touchVertexLastTime(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

logTraceOutWith1Argument( "touchVertexLastTime(...)", fineGridVertex );
}

void dem::mappings::MoveParticles::createHangingVertex(
dem::Vertex&     fineGridVertex,
const tarch::la::Vector<DIMENSIONS,iREAL>&                fineGridX,
const tarch::la::Vector<DIMENSIONS,iREAL>&                fineGridH,
dem::Vertex * const   coarseGridVertices,
const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
dem::Cell&       coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                   fineGridPositionOfVertex
) {

fineGridVertex.init();

}

void dem::mappings::MoveParticles::destroyHangingVertex(
const dem::Vertex&   fineGridVertex,
const tarch::la::Vector<DIMENSIONS,iREAL>&                    fineGridX,
const tarch::la::Vector<DIMENSIONS,iREAL>&                    fineGridH,
dem::Vertex * const  coarseGridVertices,
const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
dem::Cell&           coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {

tarch::multicore::Lock lock(_MoveParticlesSemaphore);
liftAllParticles(fineGridVertex,coarseGridVertices,coarseGridVerticesEnumerator);
lock.free();
fineGridVertex.destroy();

}

void dem::mappings::MoveParticles::createInnerVertex(
dem::Vertex&               fineGridVertex,
const tarch::la::Vector<DIMENSIONS,iREAL>&                          fineGridX,
const tarch::la::Vector<DIMENSIONS,iREAL>&                          fineGridH,
dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
dem::Cell&                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {

fineGridVertex.init();

}

void dem::mappings::MoveParticles::createBoundaryVertex(
dem::Vertex&               fineGridVertex,
const tarch::la::Vector<DIMENSIONS,iREAL>&                          fineGridX,
const tarch::la::Vector<DIMENSIONS,iREAL>&                          fineGridH,
dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
dem::Cell&                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
) {
logTraceInWith6Arguments( "createBoundaryVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

fineGridVertex.init();

logTraceOutWith1Argument( "createBoundaryVertex(...)", fineGridVertex );
}

void dem::mappings::MoveParticles::destroyVertex(
const dem::Vertex&   fineGridVertex,
const tarch::la::Vector<DIMENSIONS,iREAL>&                    fineGridX,
const tarch::la::Vector<DIMENSIONS,iREAL>&                    fineGridH,
dem::Vertex * const  coarseGridVertices,
const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
dem::Cell&           coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
) {
logTraceInWith6Arguments( "destroyVertex(...)", fineGridVertex, fineGridX, fineGridH, coarseGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfVertex );

liftAllParticles(fineGridVertex,coarseGridVertices,coarseGridVerticesEnumerator);

assertion( fineGridVertex.getNumberOfParticles()==0 );
fineGridVertex.destroy();

logTraceOutWith1Argument( "destroyVertex(...)", fineGridVertex );
}

void dem::mappings::MoveParticles::createCell(
dem::Cell&                 fineGridCell,
dem::Vertex * const        fineGridVertices,
const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
dem::Cell&                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
logTraceInWith4Arguments( "createCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );
logTraceOutWith1Argument( "createCell(...)", fineGridCell );
}

void dem::mappings::MoveParticles::destroyCell(
const dem::Cell&           fineGridCell,
dem::Vertex * const        fineGridVertices,
const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
dem::Cell&                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
) {
logTraceInWith4Arguments( "destroyCell(...)", fineGridCell, fineGridVerticesEnumerator.toString(), coarseGridCell, fineGridPositionOfCell );
logTraceOutWith1Argument( "destroyCell(...)", fineGridCell );
}

#ifdef Parallel
void dem::mappings::MoveParticles::mergeWithNeighbour(
dem::Vertex&  vertex,
const dem::Vertex&  neighbour,
int                                           fromRank,
const tarch::la::Vector<DIMENSIONS,iREAL>&   fineGridX,
const tarch::la::Vector<DIMENSIONS,iREAL>&   fineGridH,
int                                           level
) {
logTraceInWith6Arguments( "mergeWithNeighbour(...)", vertex, neighbour, fromRank, fineGridX, fineGridH, level );
logTraceOut( "mergeWithNeighbour(...)" );
}

void dem::mappings::MoveParticles::prepareSendToNeighbour(
dem::Vertex&  vertex,
int                                           toRank,
const tarch::la::Vector<DIMENSIONS,iREAL>&   x,
const tarch::la::Vector<DIMENSIONS,iREAL>&   h,
int                                           level
) {
logTraceInWith3Arguments( "prepareSendToNeighbour(...)", vertex, toRank, level );
logTraceOut( "prepareSendToNeighbour(...)" );
}

void dem::mappings::MoveParticles::prepareCopyToRemoteNode(
dem::Vertex&  localVertex,
int                                           toRank,
const tarch::la::Vector<DIMENSIONS,iREAL>&   x,
const tarch::la::Vector<DIMENSIONS,iREAL>&   h,
int                                           level
) {
logTraceInWith5Arguments( "prepareCopyToRemoteNode(...)", localVertex, toRank, x, h, level );
logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void dem::mappings::MoveParticles::prepareCopyToRemoteNode(
dem::Cell&  localCell,
int                                           toRank,
const tarch::la::Vector<DIMENSIONS,iREAL>&   cellCentre,
const tarch::la::Vector<DIMENSIONS,iREAL>&   cellSize,
int                                           level
) {
logTraceInWith5Arguments( "prepareCopyToRemoteNode(...)", localCell, toRank, cellCentre, cellSize, level );
logTraceOut( "prepareCopyToRemoteNode(...)" );
}

void dem::mappings::MoveParticles::mergeWithRemoteDataDueToForkOrJoin(
dem::Vertex&  localVertex,
const dem::Vertex&  masterOrWorkerVertex,
int                                       fromRank,
const tarch::la::Vector<DIMENSIONS,iREAL>&  x,
const tarch::la::Vector<DIMENSIONS,iREAL>&  h,
int                                       level
) {
logTraceInWith6Arguments( "mergeWithRemoteDataDueToForkOrJoin(...)", localVertex, masterOrWorkerVertex, fromRank, x, h, level );
logTraceOut( "mergeWithRemoteDataDueToForkOrJoin(...)" );
}

void dem::mappings::MoveParticles::mergeWithRemoteDataDueToForkOrJoin(
dem::Cell&  localCell,
const dem::Cell&  masterOrWorkerCell,
int                                       fromRank,
const tarch::la::Vector<DIMENSIONS,iREAL>&  cellCentre,
const tarch::la::Vector<DIMENSIONS,iREAL>&  cellSize,
int                                       level
) {
logTraceInWith3Arguments( "mergeWithRemoteDataDueToForkOrJoin(...)", localCell, masterOrWorkerCell, fromRank );
logTraceOut( "mergeWithRemoteDataDueToForkOrJoin(...)" );
}

bool dem::mappings::MoveParticles::prepareSendToWorker(
dem::Cell&                 fineGridCell,
dem::Vertex * const        fineGridVertices,
const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
dem::Cell&                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
int                                                                  worker
) {
logTraceIn( "prepareSendToWorker(...)" );
logTraceOutWith1Argument( "prepareSendToWorker(...)", true );
return true;
}

void dem::mappings::MoveParticles::prepareSendToMaster(
dem::Cell&                       localCell,
dem::Vertex *                    vertices,
const peano::grid::VertexEnumerator&       verticesEnumerator, 
const dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&       coarseGridVerticesEnumerator,
const dem::Cell&                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&   fineGridPositionOfCell
) {
logTraceInWith2Arguments( "prepareSendToMaster(...)", localCell, verticesEnumerator.toString() );
logTraceOut( "prepareSendToMaster(...)" );
}

void dem::mappings::MoveParticles::mergeWithMaster(
const dem::Cell&           workerGridCell,
dem::Vertex * const        workerGridVertices,
const peano::grid::VertexEnumerator& workerEnumerator,
dem::Cell&                 fineGridCell,
dem::Vertex * const        fineGridVertices,
const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
dem::Cell&                 coarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
int                                                                  worker,
const dem::State&          workerState,
dem::State&                masterState
) {
logTraceIn( "mergeWithMaster(...)" );
logTraceOut( "mergeWithMaster(...)" );
}

void dem::mappings::MoveParticles::receiveDataFromMaster(
dem::Cell&                        receivedCell, 
dem::Vertex *                     receivedVertices,
const peano::grid::VertexEnumerator&        receivedVerticesEnumerator,
dem::Vertex * const               receivedCoarseGridVertices,
const peano::grid::VertexEnumerator&        receivedCoarseGridVerticesEnumerator,
dem::Cell&                        receivedCoarseGridCell,
dem::Vertex * const               workersCoarseGridVertices,
const peano::grid::VertexEnumerator&        workersCoarseGridVerticesEnumerator,
dem::Cell&                        workersCoarseGridCell,
const tarch::la::Vector<DIMENSIONS,int>&    fineGridPositionOfCell
) {
logTraceIn( "receiveDataFromMaster(...)" );
logTraceOut( "receiveDataFromMaster(...)" );
}

void dem::mappings::MoveParticles::mergeWithWorker(
dem::Cell&           localCell, 
const dem::Cell&     receivedMasterCell,
const tarch::la::Vector<DIMENSIONS,iREAL>&  cellCentre,
const tarch::la::Vector<DIMENSIONS,iREAL>&  cellSize,
int                                          level
) {
logTraceInWith2Arguments( "mergeWithWorker(...)", localCell.toString(), receivedMasterCell.toString() );
logTraceOutWith1Argument( "mergeWithWorker(...)", localCell.toString() );
}

void dem::mappings::MoveParticles::mergeWithWorker(
dem::Vertex&        localVertex,
const dem::Vertex&  receivedMasterVertex,
const tarch::la::Vector<DIMENSIONS,iREAL>&   x,
const tarch::la::Vector<DIMENSIONS,iREAL>&   h,
int                                           level
) {
logTraceInWith2Arguments( "mergeWithWorker(...)", localVertex.toString(), receivedMasterVertex.toString() );
logTraceOutWith1Argument( "mergeWithWorker(...)", localVertex.toString() );
}
#endif

void dem::mappings::MoveParticles::descend(
dem::Cell * const          fineGridCells,
dem::Vertex * const        fineGridVertices,
const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
dem::Vertex * const        coarseGridVertices,
const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
dem::Cell&                 coarseGridCell
) {
logTraceInWith2Arguments( "descend(...)", coarseGridCell.toString(), coarseGridVerticesEnumerator.toString() );
logTraceOut( "descend(...)" );
}

void dem::mappings::MoveParticles::ascend(
dem::Cell * const    fineGridCells,
dem::Vertex * const  fineGridVertices,
const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
dem::Vertex * const  coarseGridVertices,
const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
dem::Cell&           coarseGridCell
) {
logTraceInWith2Arguments( "ascend(...)", coarseGridCell.toString(), coarseGridVerticesEnumerator.toString() );
logTraceOut( "ascend(...)" );
}
