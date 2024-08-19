

#include "../contactpoint.h"
#include <vector>
#include <complex>
#include <limits>
#include <iostream>
#include "../../core/algo.h"

#ifdef peanoCall
#include "peano/utils/Loop.h"
#include "tarch/multicore/Lock.h"
#include "tarch/multicore/BooleanSemaphore.h"
#endif

namespace delta {
namespace contact {
namespace detection {


#if defined(SharedTBB) && defined(peanoCall)
std::vector<delta::contact::contactpoint> bf(
const iREAL*    xCoordinatesOfPointsOfGeometryA,
const iREAL*    yCoordinatesOfPointsOfGeometryA,
const iREAL*    zCoordinatesOfPointsOfGeometryA,
int             numberOfPointsOfGeometryA,
iREAL           epsilonA,
bool            frictionA,
int 	          particleA,

const iREAL*    xCoordinatesOfPointsOfGeometryB,
const iREAL*    yCoordinatesOfPointsOfGeometryB,
const iREAL*    zCoordinatesOfPointsOfGeometryB,
int             numberOfPointsOfGeometryB,
iREAL           epsilonB,
bool            frictionB,
int 	          particleB,
tarch::multicore::BooleanSemaphore &semaphore
);
#else
std::vector<delta::contact::contactpoint> bf(
const iREAL*    xCoordinatesOfPointsOfGeometryA,
const iREAL*    yCoordinatesOfPointsOfGeometryA,
const iREAL*    zCoordinatesOfPointsOfGeometryA,
int             numberOfPointsOfGeometryA,
iREAL           epsilonA,
bool            frictionA,
int 	          particleA,

const iREAL*    xCoordinatesOfPointsOfGeometryB,
const iREAL*    yCoordinatesOfPointsOfGeometryB,
const iREAL*    zCoordinatesOfPointsOfGeometryB,
int             numberOfPointsOfGeometryB,
iREAL           epsilonB,
bool            frictionB,
int 	          particleB
);
#endif

#ifdef OMPProcess
#pragma omp declare simd
#endif
void bfSolver(
const iREAL   *xCoordinatesOfTriangleA,
const iREAL   *yCoordinatesOfTriangleA,
const iREAL   *zCoordinatesOfTriangleA,
const iREAL   *xCoordinatesOfTriangleB,
const iREAL   *yCoordinatesOfTriangleB,
const iREAL   *zCoordinatesOfTriangleB,
iREAL&  xPA,
iREAL&  yPA,
iREAL&  zPA,
iREAL&  xPB,
iREAL&  yPB,
iREAL&  zPB
);
}
}
}
