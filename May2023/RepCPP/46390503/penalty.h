

#include "../contactpoint.h"
#include <vector>

#include <limits>
#include <float.h>
#include "../../core/algo.h"

#ifdef peanoCall
#include "peano/utils/Loop.h"
#include "tarch/multicore/Lock.h"
#include "tarch/multicore/BooleanSemaphore.h"
#endif

#define MaxNumberOfNewtonIterations 16

namespace delta {
namespace contact {
namespace detection {

void cleanPenaltyStatistics();


std::vector<int> getPenaltyIterationStatistics();

std::vector<std::vector<iREAL>> getPenaltyErrorStatistics();



std::vector<contactpoint> penaltyStat(
const iREAL*    xCoordinatesOfPointsOfGeometryA,
const iREAL*    yCoordinatesOfPointsOfGeometryA,
const iREAL*    zCoordinatesOfPointsOfGeometryA,
const int       numberOfTrianglesOfGeometryA,
const iREAL     epsilonA,
const bool      frictionA,
const int 	    particleA,

const iREAL*    xCoordinatesOfPointsOfGeometryB,
const iREAL*    yCoordinatesOfPointsOfGeometryB,
const iREAL*    zCoordinatesOfPointsOfGeometryB,
const int       numberOfTrianglesOfGeometryB,
const iREAL     epsilonB,
const bool      frictionB,
const int 	    particleB);

#if defined(SharedTBB) && defined(peanoCall)
std::vector<contactpoint> penalty(
const iREAL*    xCoordinatesOfPointsOfGeometryA,
const iREAL*    yCoordinatesOfPointsOfGeometryA,
const iREAL*    zCoordinatesOfPointsOfGeometryA,
const int       numberOfTrianglesOfGeometryA,
const iREAL     epsilonA,
const bool      frictionA,
const int 	    particleA,

const iREAL*    xCoordinatesOfPointsOfGeometryB,
const iREAL*    yCoordinatesOfPointsOfGeometryB,
const iREAL*    zCoordinatesOfPointsOfGeometryB,
const int       numberOfTrianglesOfGeometryB,
const iREAL     epsilonB,
const bool      frictionB,
const int       particleB,
tarch::multicore::BooleanSemaphore &semaphore
);
#else
std::vector<contactpoint> penalty(
const iREAL*    xCoordinatesOfPointsOfGeometryA,
const iREAL*    yCoordinatesOfPointsOfGeometryA,
const iREAL*    zCoordinatesOfPointsOfGeometryA,
const int       numberOfTrianglesOfGeometryA,
const iREAL     epsilonA,
const bool      frictionA,
const int 	    particleA,

const iREAL*    xCoordinatesOfPointsOfGeometryB,
const iREAL*    yCoordinatesOfPointsOfGeometryB,
const iREAL*    zCoordinatesOfPointsOfGeometryB,
const int       numberOfTrianglesOfGeometryB,
const iREAL     epsilonB,
const bool      frictionB,
const int       particleB
);
#endif

#pragma omp declare simd
#pragma omp declare simd linear(xCoordinatesOfTriangleA:3) linear(yCoordinatesOfTriangleA:3) linear(zCoordinatesOfTriangleA:3) linear(xCoordinatesOfTriangleB:3) linear(yCoordinatesOfTriangleB:3) linear(zCoordinatesOfTriangleB:3) nomask notinbranch
extern void penaltySolver(
const iREAL   *xCoordinatesOfTriangleA,
const iREAL   *yCoordinatesOfTriangleA,
const iREAL   *zCoordinatesOfTriangleA,
const iREAL   *xCoordinatesOfTriangleB,
const iREAL   *yCoordinatesOfTriangleB,
const iREAL   *zCoordinatesOfTriangleB,
iREAL&        xPA,
iREAL&        yPA,
iREAL&        zPA,
iREAL&        xPB,
iREAL&        yPB,
iREAL&        zPB,
iREAL         MaxErrorOfPenaltyMethod,
bool&         failed);


void penaltySolver(
const iREAL			*xCoordinatesOfTriangleA,
const iREAL			*yCoordinatesOfTriangleA,
const iREAL			*zCoordinatesOfTriangleA,
const iREAL			*xCoordinatesOfTriangleB,
const iREAL			*yCoordinatesOfTriangleB,
const iREAL			*zCoordinatesOfTriangleB,
iREAL&				xPA,
iREAL&				yPA,
iREAL&				zPA,
iREAL&				xPB,
iREAL&				yPB,
iREAL&				zPB,
iREAL					maxError,
int&          numberOfNewtonIterationsRequired);
}
}
}
