
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

#include "SimulationNBodyV3IntrinsicsBH.h"

template <typename T>
SimulationNBodyV3IntrinsicsBH<T>::SimulationNBodyV3IntrinsicsBH(const unsigned long nBodies, T softening)
: SimulationNBodyV3Intrinsics<T>(nBodies, softening)
{
this->init();
}

template <typename T>
SimulationNBodyV3IntrinsicsBH<T>::SimulationNBodyV3IntrinsicsBH(const std::string inputFileName, T softening)
: SimulationNBodyV3Intrinsics<T>(inputFileName, softening)
{
this->init();
}

template <typename T>
void SimulationNBodyV3IntrinsicsBH<T>::init()
{
this->flopsPerIte = 19.f * ((float)this->bodies->getN() -1.f) * (float)this->bodies->getN();
}

template <>
void SimulationNBodyV3IntrinsicsBH<float>::init()
{
this->flopsPerIte = 20.f * ((float)this->bodies->getN() -1.f) * (float)this->bodies->getN();
}

template <typename T>
SimulationNBodyV3IntrinsicsBH<T>::~SimulationNBodyV3IntrinsicsBH()
{
}

template <typename T>
void SimulationNBodyV3IntrinsicsBH<T>::initIteration()
{
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
{
this->accelerations.x[iBody] = 0.0;
this->accelerations.y[iBody] = 0.0;
this->accelerations.z[iBody] = 0.0;

this->closestNeighborDist[iBody] = std::numeric_limits<T>::infinity();
}
}

template <>
void SimulationNBodyV3IntrinsicsBH<float>::initIteration()
{
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
{
this->accelerations.x[iBody] = 0.0;
this->accelerations.y[iBody] = 0.0;
this->accelerations.z[iBody] = 0.0;

this->closestNeighborDist[iBody] = 0.0;
}
}

template <typename T>
void SimulationNBodyV3IntrinsicsBH<T>::_computeLocalBodiesAcceleration()
{
assert(this->dtConstant || (this->bodies->getN() % mipp::N<T>() == 0));

const auto mass  = (mipp::Reg<T>*)this->getBodies()->getMasses();
const auto posX  = (mipp::Reg<T>*)this->getBodies()->getPositionsX();
const auto posY  = (mipp::Reg<T>*)this->getBodies()->getPositionsY();
const auto posZ  = (mipp::Reg<T>*)this->getBodies()->getPositionsZ();
auto accX  = (mipp::Reg<T>*)this->accelerations.x;
auto accY  = (mipp::Reg<T>*)this->accelerations.y;
auto accZ  = (mipp::Reg<T>*)this->accelerations.z;
auto closN = (mipp::Reg<T>*)this->closestNeighborDist;

const mipp::Reg<T> rG           = (T)this->G;
const mipp::Reg<T> rSoftSquared = (T)this->softeningSquared;

#pragma omp parallel for schedule(runtime) firstprivate(rG)
for(unsigned long i = 0; i < this->bodies->getNVecs(); i++)
{
const auto rqiX = posX[i];
const auto rqiY = posY[i];
const auto rqiZ = posZ[i];

auto raiX = accX[i];
auto raiY = accY[i];
auto raiZ = accZ[i];

mipp::Reg<T> rclosNeighi = (T)0;
if(!this->dtConstant)
rclosNeighi = closN[i];

for(unsigned long j = 0; j < this->bodies->getNVecs(); j++)
{
auto rmj  = mass[j];
auto rqjX = posX[j];
auto rqjY = posY[j];
auto rqjZ = posZ[j];

for(unsigned short iRot = 0; iRot < mipp::N<T>(); iRot++)
{
SimulationNBodyV3Intrinsics<T>::computeAccelerationBetweenTwoBodies(rG, rSoftSquared,
rqiX, rqiY, rqiZ,
raiX, raiY, raiZ,
rclosNeighi,
rmj,
rqjX, rqjY, rqjZ);

rmj  = mipp::rrot(rmj);
rqjX = mipp::rrot(rqjX); rqjY = mipp::rrot(rqjY); rqjZ = mipp::rrot(rqjZ);
}
}

raiX.store(this->accelerations.x + i * mipp::N<T>());
raiY.store(this->accelerations.y + i * mipp::N<T>());
raiZ.store(this->accelerations.z + i * mipp::N<T>());

if(!this->dtConstant)
rclosNeighi.store(this->closestNeighborDist + i * mipp::N<T>());
}
}

template <typename T>
void SimulationNBodyV3IntrinsicsBH<T>::computeLocalBodiesAccelerationWithBlackHole()
{
const T *masses = this->getBodies()->getMasses();

const T *positionsX = this->getBodies()->getPositionsX();
const T *positionsY = this->getBodies()->getPositionsY();
const T *positionsZ = this->getBodies()->getPositionsZ();

T mbhGained = 0;
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
{
T dist = SimulationNBodyV3IntrinsicsBH<T>::computeAccelerationBetweenBodyAndBlackHole(
this->G, this->softeningSquared,
positionsX           [iBody],
positionsY           [iBody],
positionsZ           [iBody],
this->accelerations.x[iBody],
this->accelerations.y[iBody],
this->accelerations.z[iBody],
this->mbh,
this->rbh,
this->qbhX,
this->qbhY,
this->qbhZ);

if(dist <= 0)
mbhGained += masses[iBody];
}

this->mbh += mbhGained;
}

template <typename T>
void SimulationNBodyV3IntrinsicsBH<T>::computeLocalBodiesAcceleration()
{
this->_computeLocalBodiesAcceleration();
this->computeLocalBodiesAccelerationWithBlackHole();
}

template <>
void SimulationNBodyV3IntrinsicsBH<float>::computeLocalBodiesAcceleration()
{
this->_computeLocalBodiesAcceleration();
this->computeLocalBodiesAccelerationWithBlackHole();

for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
this->closestNeighborDist[iBody] = 1.0 / this->closestNeighborDist[iBody];
}


template <typename T>
T SimulationNBodyV3IntrinsicsBH<T>::computeAccelerationBetweenBodyAndBlackHole(const T &G,   const T &softSquared,
const T &qiX, const T &qiY, const T &qiZ,
T &aiX,       T &aiY,       T &aiZ,
const T &mbh,
const T &rbh,
const T &qbhX, const T &qbhY, const T &qbhZ)
{
const T ribhX = qbhX - qiX; 
const T ribhY = qbhY - qiY; 
const T ribhZ = qbhZ - qiZ; 

const T ribhSquared = (ribhX * ribhX) + (ribhY * ribhY) + (ribhZ * ribhZ) + softSquared; 

const T ribh = std::sqrt(ribhSquared); 

const T ai = G * mbh / (ribhSquared * ribh); 

aiX += ai * ribhX; 
aiY += ai * ribhY; 
aiZ += ai * ribhZ; 

return ribh - rbh; 
}
