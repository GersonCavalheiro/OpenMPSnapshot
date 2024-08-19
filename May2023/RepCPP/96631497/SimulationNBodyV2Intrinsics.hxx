
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

#include "SimulationNBodyV2Intrinsics.h"

template <typename T>
SimulationNBodyV2Intrinsics<T>::SimulationNBodyV2Intrinsics(const unsigned long nBodies)
: SimulationNBodyV2<T>(nBodies)
{
this->reAllocateBuffers();
}

template <typename T>
SimulationNBodyV2Intrinsics<T>::SimulationNBodyV2Intrinsics(const std::string inputFileName)
: SimulationNBodyV2<T>(inputFileName)
{
this->reAllocateBuffers();
}

template <typename T>
SimulationNBodyV2Intrinsics<T>::~SimulationNBodyV2Intrinsics()
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
void SimulationNBodyV2Intrinsics<T>::_reAllocateBuffers()
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
}


template <typename T>
void SimulationNBodyV2Intrinsics<T>::reAllocateBuffers()
{
this->_reAllocateBuffers();
this->flopsPerIte = 26.f * ((float)this->bodies->getN() * 0.5f) * (float)this->bodies->getN();
}

template <>
void SimulationNBodyV2Intrinsics<float>::reAllocateBuffers()
{
this->_reAllocateBuffers();
this->flopsPerIte = 27.f * ((float)this->bodies->getN() * 0.5f) * (float)this->bodies->getN();
}

template <typename T>
void SimulationNBodyV2Intrinsics<T>::_initIteration()
{
for(unsigned long iBody = 0; iBody < this->bodies->getN() * this->nMaxThreads; iBody++)
{
this->accelerations.x[iBody] = 0.0;
this->accelerations.y[iBody] = 0.0;
this->accelerations.z[iBody] = 0.0;
}
}

template <typename T>
void SimulationNBodyV2Intrinsics<T>::initIteration()
{
this->_initIteration();

for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
this->closestNeighborDist[iBody] = std::numeric_limits<T>::infinity();
}

template <>
void SimulationNBodyV2Intrinsics<float>::initIteration()
{
this->_initIteration();

for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
this->closestNeighborDist[iBody] = 0;
}

template <typename T>
void SimulationNBodyV2Intrinsics<T>::_computeLocalBodiesAcceleration()
{
const auto mass  = (mipp::Reg<T>*)this->getBodies()->getMasses();
const auto posX  = (mipp::Reg<T>*)this->getBodies()->getPositionsX();
const auto posY  = (mipp::Reg<T>*)this->getBodies()->getPositionsY();
const auto posZ  = (mipp::Reg<T>*)this->getBodies()->getPositionsZ();
auto accX  = (mipp::Reg<T>*)this->accelerations.x;
auto accY  = (mipp::Reg<T>*)this->accelerations.y;
auto accZ  = (mipp::Reg<T>*)this->accelerations.z;
auto closN = (mipp::Reg<T>*)this->closestNeighborDist;

const T *masses = this->bodies->getMasses();

const T *positionsX = this->bodies->getPositionsX();
const T *positionsY = this->bodies->getPositionsY();
const T *positionsZ = this->bodies->getPositionsZ();

const mipp::Reg<T> rG = (T)this->G;

#pragma omp parallel firstprivate(rG)
{
const unsigned int  tid     = omp_get_thread_num();
const unsigned long tStride = tid * (this->bodies->getN() + this->bodies->getPadding());
const unsigned long tSt     = tStride / mipp::N<T>();

#pragma omp for schedule(runtime)
for(unsigned long iVec = 0; iVec < this->bodies->getNVecs(); iVec++)
{
for(unsigned short iPos = 0; iPos < mipp::N<T>(); iPos++)
{
const unsigned long iBody = iPos + iVec * mipp::N<T>();
for(unsigned short jPos = iPos +1; jPos < mipp::N<T>(); jPos++)
{
const unsigned long jBody = jPos + iVec * mipp::N<T>();
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
}

const auto rmi  = mass[iVec];
const auto rqiX = posX[iVec];
const auto rqiY = posY[iVec];
const auto rqiZ = posZ[iVec];

auto raiX = accX[iVec + tSt];
auto raiY = accY[iVec + tSt];
auto raiZ = accZ[iVec + tSt];

mipp::Reg<T> rclosNeighi = (T)0;
if(!this->dtConstant)
rclosNeighi = closN[iVec];

for(unsigned long jVec = iVec +1; jVec < this->bodies->getNVecs(); jVec++)
{
auto rmj  = mass[jVec];
auto rqjX = posX[jVec];
auto rqjY = posY[jVec];
auto rqjZ = posZ[jVec];

auto rajX = accX[jVec + tSt];
auto rajY = accY[jVec + tSt];
auto rajZ = accZ[jVec + tSt];

mipp::Reg<T> rclosNeighj = (T)0;
if(!this->dtConstant)
rclosNeighj = closN[jVec];

for(unsigned short iRot = 0; iRot < mipp::N<T>(); iRot++)
{
SimulationNBodyV2Intrinsics<T>::computeAccelerationBetweenTwoBodies(rG,
rmi,
rqiX, rqiY, rqiZ,
raiX, raiY, raiZ,
rclosNeighi,
rmj,
rqjX, rqjY, rqjZ,
rajX, rajY, rajZ,
rclosNeighj);

rmj  = mipp::rrot(rmj);
rqjX = mipp::rrot(rqjX); rqjY = mipp::rrot(rqjY); rqjZ = mipp::rrot(rqjZ);
rajX = mipp::rrot(rajX); rajY = mipp::rrot(rajY); rajZ = mipp::rrot(rajZ);
rclosNeighj = mipp::rrot<T>(rclosNeighj);
}

rajX.store(this->accelerations.x + jVec * mipp::N<T>() + tStride);
rajY.store(this->accelerations.y + jVec * mipp::N<T>() + tStride);
rajZ.store(this->accelerations.z + jVec * mipp::N<T>() + tStride);

if(!this->dtConstant)
#pragma omp critical 
rclosNeighj.store(this->closestNeighborDist + jVec * mipp::N<T>());
}

raiX.store(this->accelerations.x + iVec * mipp::N<T>() + tStride);
raiY.store(this->accelerations.y + iVec * mipp::N<T>() + tStride);
raiZ.store(this->accelerations.z + iVec * mipp::N<T>() + tStride);

if(!this->dtConstant)
rclosNeighi.store(this->closestNeighborDist + iVec * mipp::N<T>());
}
}

if(this->nMaxThreads > 1)
for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
for(unsigned iThread = 1; iThread < this->nMaxThreads; iThread++)
{
this->accelerations.x[iBody] +=
this->accelerations.x[iBody + iThread * (this->bodies->getN() + this->bodies->getPadding())];
this->accelerations.y[iBody] +=
this->accelerations.y[iBody + iThread * (this->bodies->getN() + this->bodies->getPadding())];
this->accelerations.z[iBody] +=
this->accelerations.z[iBody + iThread * (this->bodies->getN() + this->bodies->getPadding())];
}
}


template <typename T>
void SimulationNBodyV2Intrinsics<T>::computeLocalBodiesAcceleration()
{
this->_computeLocalBodiesAcceleration();
}

template <>
void SimulationNBodyV2Intrinsics<float>::computeLocalBodiesAcceleration()
{
this->_computeLocalBodiesAcceleration();

for(unsigned long iBody = 0; iBody < this->bodies->getN(); iBody++)
this->closestNeighborDist[iBody] = 1.0 / this->closestNeighborDist[iBody];
}

template <typename T>
void SimulationNBodyV2Intrinsics<T>::computeAccelerationBetweenTwoBodies(const mipp::Reg<T> &rG,
const mipp::Reg<T> &rmi,
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
const mipp::Reg<T> &rqjZ,
mipp::Reg<T> &rajX,
mipp::Reg<T> &rajY,
mipp::Reg<T> &rajZ,
mipp::Reg<T> &rclosNeighj)
{
auto rrijX = rqjX - rqiX;
auto rrijY = rqjY - rqiY;
auto rrijZ = rqjZ - rqiZ;

auto rrijSquared = mipp::Reg<T>((T)0);
rrijSquared = mipp::fmadd(rrijX, rrijX, rrijSquared); 
rrijSquared = mipp::fmadd(rrijY, rrijY, rrijSquared); 
rrijSquared = mipp::fmadd(rrijZ, rrijZ, rrijSquared); 

auto rrij = mipp::sqrt(rrijSquared); 

auto raTmp = rG / (rrij * rrijSquared); 

auto rai = raTmp * rmj; 

raiX = mipp::fmadd(rai, rrijX, raiX); 
raiY = mipp::fmadd(rai, rrijY, raiY); 
raiZ = mipp::fmadd(rai, rrijZ, raiZ); 

auto raj = raTmp * rmi; 

rajX = mipp::fnmadd(raj, rrijX, rajX); 
rajY = mipp::fnmadd(raj, rrijY, rajY); 
rajZ = mipp::fnmadd(raj, rrijZ, rajZ); 

rclosNeighi = mipp::min(rrij, rclosNeighi);
rclosNeighj = mipp::min(rrij, rclosNeighj); 
}

template <>
void SimulationNBodyV2Intrinsics<float>::computeAccelerationBetweenTwoBodies(const mipp::Reg<float> &rG,
const mipp::Reg<float> &rmi,
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
const mipp::Reg<float> &rqjZ,
mipp::Reg<float> &rajX,
mipp::Reg<float> &rajY,
mipp::Reg<float> &rajZ,
mipp::Reg<float> &rclosNeighj)
{
auto rrijX = rqjX - rqiX; 
auto rrijY = rqjY - rqiY; 
auto rrijZ = rqjZ - rqiZ; 

auto rrijSquared = mipp::Reg<float>(0.f);
rrijSquared = mipp::fmadd(rrijX, rrijX, rrijSquared); 
rrijSquared = mipp::fmadd(rrijY, rrijY, rrijSquared); 
rrijSquared = mipp::fmadd(rrijZ, rrijZ, rrijSquared); 

auto rrijInv = mipp::rsqrt(rrijSquared); 

auto raTmp = rG * rrijInv * rrijInv * rrijInv; 

auto rai = raTmp * rmj; 

raiX = mipp::fmadd(rai, rrijX, raiX); 
raiY = mipp::fmadd(rai, rrijY, raiY); 
raiZ = mipp::fmadd(rai, rrijZ, raiZ); 

auto raj = raTmp * rmi; 

rajX = mipp::fnmadd(raj, rrijX, rajX); 
rajY = mipp::fnmadd(raj, rrijY, rajY); 
rajZ = mipp::fnmadd(raj, rrijZ, rajZ); 

rclosNeighi = mipp::max(rrijInv, rclosNeighi);
rclosNeighj = mipp::max(rrijInv, rclosNeighj); 
}
