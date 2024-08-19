
#include <cmath>
#include <limits>
#include <string>
#include <cassert>
#include <fstream>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#else
#ifndef NO_OMP
#define NO_OMP
inline void omp_set_num_threads(int) {           }
inline int  omp_get_num_threads(   ) { return 1; }
inline int  omp_get_max_threads(   ) { return 1; }
inline int  omp_get_thread_num (   ) { return 0; }
#endif
#endif

#include "SimulationNBodyCollisionV1.h"

template <typename T>
SimulationNBodyCollisionV1<T>::SimulationNBodyCollisionV1(const unsigned long nBodies)
: SimulationNBodyCollisionLocal<T>(nBodies)
{
this->init();
}

template <typename T>
SimulationNBodyCollisionV1<T>::SimulationNBodyCollisionV1(const std::string inputFileName)
: SimulationNBodyCollisionLocal<T>(inputFileName)
{
this->init();
}

template <typename T>
void SimulationNBodyCollisionV1<T>::init()
{
this->flopsPerIte = 20.f * ((float)this->bodies->getN() -1.f) * (float)this->bodies->getN();
}

template <typename T>
SimulationNBodyCollisionV1<T>::~SimulationNBodyCollisionV1()
{
}

template <typename T>
void SimulationNBodyCollisionV1<T>::initIteration()
{
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
{
this->accelerations.x[iBody] = 0.0;
this->accelerations.y[iBody] = 0.0;
this->accelerations.z[iBody] = 0.0;

this->closestNeighborDist[iBody] = std::numeric_limits<T>::infinity();

this->collisions[iBody].clear();
}
}

template <typename T>
void SimulationNBodyCollisionV1<T>::computeLocalBodiesAcceleration()
{
const T *masses     = this->bodies->getMasses();
const T *radiuses   = this->bodies->getRadiuses();
const T *positionsX = this->bodies->getPositionsX();
const T *positionsY = this->bodies->getPositionsY();
const T *positionsZ = this->bodies->getPositionsZ();

#pragma omp parallel for schedule(runtime)
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
for(unsigned long jBody = 0; jBody < this->bodies->getN(); jBody++)
if(iBody != jBody)
{
T dist = SimulationNBodyCollisionV1<T>::computeAccelerationBetweenTwoBodies(this->G,
radiuses                 [iBody],
positionsX               [iBody],
positionsY               [iBody],
positionsZ               [iBody],
this->accelerations.x    [iBody],
this->accelerations.y    [iBody],
this->accelerations.z    [iBody],
this->closestNeighborDist[iBody],
masses                   [jBody],
radiuses                 [jBody],
positionsX               [jBody],
positionsY               [jBody],
positionsZ               [jBody]);
if(dist <= 0)
this->collisions[iBody].push_back(jBody);
}
}

template <typename T>
T SimulationNBodyCollisionV1<T>::computeAccelerationBetweenTwoBodies(const T &G,
const T &ri,
const T &qiX, const T &qiY, const T &qiZ,
T &aiX,       T &aiY,       T &aiZ,
T &closNeighi,
const T &mj,
const T &rj,
const T &qjX, const T &qjY, const T &qjZ)
{
const T rijX = qjX - qiX; 
const T rijY = qjY - qiY; 
const T rijZ = qjZ - qiZ; 

const T rijSquared = (rijX * rijX) + (rijY * rijY) + (rijZ * rijZ); 

const T rij = std::sqrt(rijSquared); 

assert(rij != 0);

const T ai = G * mj / (rijSquared * rij); 

aiX += ai * rijX; 
aiY += ai * rijY; 
aiZ += ai * rijZ; 

closNeighi = std::min(closNeighi, rij);

return rij - (ri + rj); 
}
