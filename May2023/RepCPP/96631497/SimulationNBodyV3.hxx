
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

#include "SimulationNBodyV3.h"

template <typename T>
SimulationNBodyV3<T>::SimulationNBodyV3(const unsigned long nBodies, T softening)
: SimulationNBodyLocal<T>(nBodies), softeningSquared(softening * softening)
{
this->init();
}

template <typename T>
SimulationNBodyV3<T>::SimulationNBodyV3(const std::string inputFileName, T softening)
: SimulationNBodyLocal<T>(inputFileName), softeningSquared(softening * softening)
{
this->init();
}

template <typename T>
void SimulationNBodyV3<T>::init()
{
this->flopsPerIte = 19.f * ((float)this->bodies->getN() -1.f) * (float)this->bodies->getN();
}

template <typename T>
SimulationNBodyV3<T>::~SimulationNBodyV3()
{
}

template <typename T>
void SimulationNBodyV3<T>::initIteration()
{
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
{
this->accelerations.x[iBody] = 0.0;
this->accelerations.y[iBody] = 0.0;
this->accelerations.z[iBody] = 0.0;

this->closestNeighborDist[iBody] = std::numeric_limits<T>::infinity();
}
}

template <typename T>
void SimulationNBodyV3<T>::computeLocalBodiesAcceleration()
{
const T *masses = this->getBodies()->getMasses();

const T *positionsX = this->getBodies()->getPositionsX();
const T *positionsY = this->getBodies()->getPositionsY();
const T *positionsZ = this->getBodies()->getPositionsZ();

#pragma omp parallel for schedule(runtime)
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
for(unsigned long jBody = 0; jBody < this->bodies->getN(); jBody++)
SimulationNBodyV3<T>::computeAccelerationBetweenTwoBodies(this->G, this->softeningSquared,
positionsX               [iBody],
positionsY               [iBody],
positionsZ               [iBody],
this->accelerations.x    [iBody],
this->accelerations.y    [iBody],
this->accelerations.z    [iBody],
this->closestNeighborDist[iBody],
masses                   [jBody],
positionsX               [jBody],
positionsY               [jBody],
positionsZ               [jBody]);
}

template <typename T>
void SimulationNBodyV3<T>::computeAccelerationBetweenTwoBodies(const T &G,   const T &softSquared,
const T &qiX, const T &qiY, const T &qiZ,
T &aiX,       T &aiY,       T &aiZ,
T &closNeighi,
const T &mj,
const T &qjX, const T &qjY, const T &qjZ)
{
const T rijX = qjX - qiX; 
const T rijY = qjY - qiY; 
const T rijZ = qjZ - qiZ; 

const T rijSquared = (rijX * rijX) + (rijY * rijY) + (rijZ * rijZ) + softSquared; 

const T rij = std::sqrt(rijSquared); 

const T ai = G * mj / (rijSquared * rij); 

aiX += ai * rijX; 
aiY += ai * rijY; 
aiZ += ai * rijZ; 

closNeighi = std::min(closNeighi, rij);
}
