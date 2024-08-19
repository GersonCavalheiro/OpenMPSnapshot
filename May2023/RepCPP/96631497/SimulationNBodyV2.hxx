
#include <cmath>
#include <limits>
#include <string>
#include <cassert>
#include <fstream>
#include <iostream>
#include <mipp.h>

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

#include "SimulationNBodyV2.h"

template <typename T>
SimulationNBodyV2<T>::SimulationNBodyV2(const unsigned long nBodies)
: SimulationNBodyLocal<T>(nBodies)
{
this->reAllocateBuffers();
}

template <typename T>
SimulationNBodyV2<T>::SimulationNBodyV2(const std::string inputFileName)
: SimulationNBodyLocal<T>(inputFileName)
{
this->reAllocateBuffers();
}

template <typename T>
SimulationNBodyV2<T>::~SimulationNBodyV2()
{
if(this->accelerations.x != nullptr) {
mipp::free(this->accelerations.x);
this->accelerations.x = nullptr;
}
if(this->accelerations.y != nullptr) {
mipp::free(this->accelerations.y);
this->accelerations.y = nullptr;
}
if(this->accelerations.z != nullptr) {
mipp::free(this->accelerations.z);
this->accelerations.z = nullptr;
}
}

template <typename T>
void SimulationNBodyV2<T>::reAllocateBuffers()
{
if(this->nMaxThreads > 1)
{
if(this->accelerations.x != nullptr)
mipp::free(this->accelerations.x);
if(this->accelerations.y != nullptr)
mipp::free(this->accelerations.y);
if(this->accelerations.z != nullptr)
mipp::free(this->accelerations.z);

this->accelerations.x = mipp::malloc<T>((this->bodies->getN() + this->bodies->getPadding()) * this->nMaxThreads);
this->accelerations.y = mipp::malloc<T>((this->bodies->getN() + this->bodies->getPadding()) * this->nMaxThreads);
this->accelerations.z = mipp::malloc<T>((this->bodies->getN() + this->bodies->getPadding()) * this->nMaxThreads);

this->allocatedBytes += (this->bodies->getN() + this->bodies->getPadding()) *
sizeof(T) * (this->nMaxThreads - 1) * 3;
}

this->flopsPerIte = 25.f * ((float)this->bodies->getN() * 0.5f) * (float)this->bodies->getN();
}

template <typename T>
void SimulationNBodyV2<T>::initIteration()
{
for(unsigned long iBody = 0; iBody < this->bodies->getN() * this->nMaxThreads; iBody++)
{
this->accelerations.x[iBody] = 0.0;
this->accelerations.y[iBody] = 0.0;
this->accelerations.z[iBody] = 0.0;
}

for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
this->closestNeighborDist[iBody] = std::numeric_limits<T>::infinity();
}

template <typename T>
void SimulationNBodyV2<T>::computeLocalBodiesAcceleration()
{
const T *masses     = this->bodies->getMasses();
const T *positionsX = this->bodies->getPositionsX();
const T *positionsY = this->bodies->getPositionsY();
const T *positionsZ = this->bodies->getPositionsZ();

#pragma omp parallel
{
const unsigned int  tid     = omp_get_thread_num();
const unsigned long tStride = tid * this->bodies->getN();

#pragma omp for schedule(runtime)
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
for(unsigned long jBody = iBody +1; jBody < this->bodies->getN(); jBody++)
SimulationNBodyV2<T>::computeAccelerationBetweenTwoBodies(this->G,
masses                   [iBody          ],
positionsX               [iBody          ],
positionsY               [iBody          ],
positionsZ               [iBody          ],
this->accelerations.x    [iBody + tStride],
this->accelerations.y    [iBody + tStride],
this->accelerations.z    [iBody + tStride],
this->closestNeighborDist[iBody          ],
masses                   [jBody          ],
positionsX               [jBody          ],
positionsY               [jBody          ],
positionsZ               [jBody          ],
this->accelerations.x    [jBody + tStride],
this->accelerations.y    [jBody + tStride],
this->accelerations.z    [jBody + tStride],
this->closestNeighborDist[jBody          ]);
}

if(this->nMaxThreads > 1)
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
for(unsigned iThread = 1; iThread < this->nMaxThreads; iThread++)
{
this->accelerations.x[iBody] += this->accelerations.x[iBody + iThread * this->bodies->getN()];
this->accelerations.y[iBody] += this->accelerations.y[iBody + iThread * this->bodies->getN()];
this->accelerations.z[iBody] += this->accelerations.z[iBody + iThread * this->bodies->getN()];
}
}

template <typename T>
void SimulationNBodyV2<T>::computeAccelerationBetweenTwoBodies(const T &G,
const T &mi,
const T &qiX, const T &qiY, const T &qiZ,
T &aiX,       T &aiY,       T &aiZ,
T &closNeighi,
const T &mj,
const T &qjX, const T &qjY, const T &qjZ,
T &ajX,       T &ajY,       T &ajZ,
T &closNeighj)
{
const T rijX = qjX - qiX; 
const T rijY = qjY - qiY; 
const T rijZ = qjZ - qiZ; 

const T rijSquared = (rijX * rijX) + (rijY * rijY) + (rijZ * rijZ); 

const T rij = std::sqrt(rijSquared); 

assert(rij != 0);

const T aTmp = G / (rijSquared * rij); 
const T ai = aTmp * mj; 

aiX += ai * rijX; 
aiY += ai * rijY; 
aiZ += ai * rijZ; 

const T aj = aTmp * mi; 

ajX -= aj * rijX; 
ajY -= aj * rijY; 
ajZ -= aj * rijZ; 

closNeighi = std::min(closNeighi, rij);
if(rij < closNeighj)
#pragma omp critical
closNeighj = std::min(closNeighj, rij);
}
