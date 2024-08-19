
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

#include "SimulationNBodyMPIV1Intrinsics.h"

template <typename T>
SimulationNBodyMPIV1Intrinsics<T>::SimulationNBodyMPIV1Intrinsics(const unsigned long nBodies)
: SimulationNBodyMPI<T>(nBodies)
{
this->init();
}

template <typename T>
SimulationNBodyMPIV1Intrinsics<T>::SimulationNBodyMPIV1Intrinsics(const std::string inputFileName)
: SimulationNBodyMPI<T>(inputFileName)
{
this->init();
}

template <typename T>
void SimulationNBodyMPIV1Intrinsics<T>::init()
{
this->flopsPerIte = 19.f * (((float)this->bodies->getN() * (float)this->MPISize) -1) * ((float)this->bodies->getN() * (float)this->MPISize);
}

template <>
void SimulationNBodyMPIV1Intrinsics<float>::init()
{
this->flopsPerIte = 20.f * (((float)this->bodies->getN() * (float)this->MPISize) -1) * ((float)this->bodies->getN() * (float)this->MPISize);
}

template <typename T>
SimulationNBodyMPIV1Intrinsics<T>::~SimulationNBodyMPIV1Intrinsics()
{
}

template <typename T>
void SimulationNBodyMPIV1Intrinsics<T>::initIteration()
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
void SimulationNBodyMPIV1Intrinsics<float>::initIteration()
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
void SimulationNBodyMPIV1Intrinsics<T>::_computeLocalBodiesAcceleration()
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

const mipp::Reg<T> rG = (T)this->G;

#pragma omp parallel for schedule(runtime) firstprivate(rG)
for(unsigned long iVec = 0; iVec < this->bodies->getNVecs(); iVec++)
{
const auto rqiX = posX[iVec]; 
const auto rqiY = posY[iVec]; 
const auto rqiZ = posZ[iVec]; 

auto raiX = accX[iVec]; 
auto raiY = accY[iVec]; 
auto raiZ = accZ[iVec]; 

mipp::Reg<T> rclosNeighi = (T)0;
if(!this->dtConstant)
rclosNeighi = closN[iVec]; 

for(unsigned long jVec = 0; jVec < this->bodies->getNVecs(); jVec++)
{
auto rmj  = mass[jVec]; 
auto rqjX = posX[jVec]; 
auto rqjY = posY[jVec]; 
auto rqjZ = posZ[jVec]; 

if(iVec != jVec)
{
for(unsigned short iRot = 0; iRot < mipp::N<T>(); iRot++)
{
this->computeAccelerationBetweenTwoBodies(rG,
rqiX, rqiY, rqiZ,
raiX, raiY, raiZ,
rclosNeighi,
rmj,
rqjX, rqjY, rqjZ);

rmj  = mipp::rrot(rmj);
rqjX = mipp::rrot(rqjX);
rqjY = mipp::rrot(rqjY);
rqjZ = mipp::rrot(rqjZ);
}
}
else
{
for(unsigned short iRot = 1; iRot < mipp::N<T>(); iRot++)
{
rmj  = mipp::rrot(rmj);
rqjX = mipp::rrot(rqjX);
rqjY = mipp::rrot(rqjY);
rqjZ = mipp::rrot(rqjZ);

this->computeAccelerationBetweenTwoBodies(rG,
rqiX, rqiY, rqiZ,
raiX, raiY, raiZ,
rclosNeighi,
rmj,
rqjX, rqjY, rqjZ);
}
}
}

raiX.store(this->accelerations.x + iVec * mipp::N<T>());
raiY.store(this->accelerations.y + iVec * mipp::N<T>());
raiZ.store(this->accelerations.z + iVec * mipp::N<T>());

if(!this->dtConstant)
rclosNeighi.store(this->closestNeighborDist + iVec * mipp::N<T>());
}
}

template <typename T>
void SimulationNBodyMPIV1Intrinsics<T>::computeLocalBodiesAcceleration()
{
this->_computeLocalBodiesAcceleration();
}

template <>
void SimulationNBodyMPIV1Intrinsics<float>::computeLocalBodiesAcceleration()
{
this->_computeLocalBodiesAcceleration();

for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
this->closestNeighborDist[iBody] = 1.0 / this->closestNeighborDist[iBody];
}

template <typename T>
void SimulationNBodyMPIV1Intrinsics<T>::_computeNeighborBodiesAcceleration()
{
assert(this->dtConstant || (this->bodies->getN() % mipp::N<T>() == 0));

const auto posX      = (mipp::Reg<T>*)this->getBodies()->getPositionsX();
const auto posY      = (mipp::Reg<T>*)this->getBodies()->getPositionsY();
const auto posZ      = (mipp::Reg<T>*)this->getBodies()->getPositionsZ();
const auto massNeigh = (mipp::Reg<T>*)this->neighborBodies->getMasses();
const auto posNeighX = (mipp::Reg<T>*)this->neighborBodies->getPositionsX();
const auto posNeighY = (mipp::Reg<T>*)this->neighborBodies->getPositionsY();
const auto posNeighZ = (mipp::Reg<T>*)this->neighborBodies->getPositionsZ();
auto accX      = (mipp::Reg<T>*)this->accelerations.x;
auto accY      = (mipp::Reg<T>*)this->accelerations.y;
auto accZ      = (mipp::Reg<T>*)this->accelerations.z;
auto closN     = (mipp::Reg<T>*)this->closestNeighborDist;

const mipp::Reg<T> rG = (T)this->G;

#pragma omp parallel for schedule(runtime) firstprivate(rG)
for(unsigned long iVec = 0; iVec < this->bodies->getNVecs(); iVec++)
{
const auto rqiX = posX[iVec]; 
const auto rqiY = posY[iVec]; 
const auto rqiZ = posZ[iVec]; 

auto raiX = accX[iVec]; 
auto raiY = accY[iVec]; 
auto raiZ = accZ[iVec]; 

mipp::Reg<T> rclosNeighi = (T)0;
if(!this->dtConstant)
rclosNeighi = closN[iVec]; 

for(unsigned long jVec = 0; jVec < this->bodies->getNVecs(); jVec++)
{
auto rmj  = massNeigh[jVec]; 
auto rqjX = posNeighX[jVec]; 
auto rqjY = posNeighY[jVec]; 
auto rqjZ = posNeighZ[jVec]; 

for(unsigned short iRot = 0; iRot < mipp::N<T>(); iRot++)
{
this->computeAccelerationBetweenTwoBodies(rG,
rqiX, rqiY, rqiZ,
raiX, raiY, raiZ,
rclosNeighi,
rmj,
rqjX, rqjY, rqjZ);

rmj  = mipp::rrot(rmj);
rqjX = mipp::rrot(rqjX);
rqjY = mipp::rrot(rqjY);
rqjZ = mipp::rrot(rqjZ);
}
}

raiX.store(this->accelerations.x + iVec * mipp::N<T>());
raiY.store(this->accelerations.y + iVec * mipp::N<T>());
raiZ.store(this->accelerations.z + iVec * mipp::N<T>());

if(!this->dtConstant)
rclosNeighi.store(this->closestNeighborDist + iVec * mipp::N<T>());
}
}

template <typename T>
void SimulationNBodyMPIV1Intrinsics<T>::computeNeighborBodiesAcceleration()
{
this->_computeNeighborBodiesAcceleration();
}

template <>
void SimulationNBodyMPIV1Intrinsics<float>::computeNeighborBodiesAcceleration()
{
this->_computeNeighborBodiesAcceleration();

for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
this->closestNeighborDist[iBody] = 1.0 / this->closestNeighborDist[iBody];
}

template <typename T>
void SimulationNBodyMPIV1Intrinsics<T>::computeAccelerationBetweenTwoBodies(const mipp::Reg<T> &rG,
const mipp::Reg<T> &rqiX,
const mipp::Reg<T> &rqiY,
const mipp::Reg<T> &rqiZ,
mipp::Reg<T> &raiX,
mipp::Reg<T> &raiY,
mipp::Reg<T> &raiZ,
mipp::Reg<T> &rclosNeighi,
const mipp::Reg<T> &rmj,
const mipp::Reg<T> &rqjX,
const mipp::Reg<T> &rqjY,
const mipp::Reg<T> &rqjZ)
{
auto rrijX = rqjX - rqiX; 
auto rrijY = rqjY - rqiY; 
auto rrijZ = rqjZ - rqiZ; 

auto rrijSquared = mipp::Reg<T>((T)0);
rrijSquared = mipp::fmadd(rrijX, rrijX, rrijSquared); 
rrijSquared = mipp::fmadd(rrijY, rrijY, rrijSquared); 
rrijSquared = mipp::fmadd(rrijZ, rrijZ, rrijSquared); 

auto rrij = mipp::sqrt(rrijSquared); 

auto rai = (rG * rmj) / (rrij * rrijSquared); 

raiX = mipp::fmadd(rai, rrijX, raiX); 
raiY = mipp::fmadd(rai, rrijY, raiY); 
raiZ = mipp::fmadd(rai, rrijZ, raiZ); 

rclosNeighi = mipp::min(rclosNeighi, rrij);
}

template <>
void SimulationNBodyMPIV1Intrinsics<float>::computeAccelerationBetweenTwoBodies(const mipp::Reg<float> &rG,
const mipp::Reg<float> &rqiX,
const mipp::Reg<float> &rqiY,
const mipp::Reg<float> &rqiZ,
mipp::Reg<float> &raiX,
mipp::Reg<float> &raiY,
mipp::Reg<float> &raiZ,
mipp::Reg<float> &rclosNeighi,
const mipp::Reg<float> &rmj,
const mipp::Reg<float> &rqjX,
const mipp::Reg<float> &rqjY,
const mipp::Reg<float> &rqjZ)
{
auto rrijX = rqjX - rqiX; 
auto rrijY = rqjY - rqiY; 
auto rrijZ = rqjZ - rqiZ; 

auto rrijSquared = mipp::Reg<float>(0.f);
rrijSquared = mipp::fmadd(rrijX, rrijX, rrijSquared); 
rrijSquared = mipp::fmadd(rrijY, rrijY, rrijSquared); 
rrijSquared = mipp::fmadd(rrijZ, rrijZ, rrijSquared); 

auto rrijInv = mipp::rsqrt(rrijSquared); 

auto rai = rG * rmj * rrijInv * rrijInv * rrijInv; 

raiX = mipp::fmadd(rai, rrijX, raiX); 
raiY = mipp::fmadd(rai, rrijY, raiY); 
raiZ = mipp::fmadd(rai, rrijZ, raiZ); 

rclosNeighi = mipp::max(rclosNeighi, rrijInv);
}
