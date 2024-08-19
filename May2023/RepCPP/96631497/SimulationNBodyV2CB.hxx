
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

#include "SimulationNBodyV2CB.h"

template <typename T>
SimulationNBodyV2CB<T>::SimulationNBodyV2CB(const unsigned long nBodies)
: SimulationNBodyV2<T>(nBodies)
{
}

template <typename T>
SimulationNBodyV2CB<T>::SimulationNBodyV2CB(const std::string inputFileName)
: SimulationNBodyV2<T>(inputFileName)
{
}

template <typename T>
SimulationNBodyV2CB<T>::~SimulationNBodyV2CB()
{
}

template <typename T>
void SimulationNBodyV2CB<T>::computeLocalBodiesAcceleration()
{
const T *masses     = this->bodies->getMasses();
const T *positionsX = this->bodies->getPositionsX();
const T *positionsY = this->bodies->getPositionsY();
const T *positionsZ = this->bodies->getPositionsZ();

unsigned long blockSize = 512;
for(unsigned long jOff = 0; jOff < this->bodies->getN(); jOff += blockSize)
{
blockSize = std::min(blockSize, this->bodies->getN() - jOff);
#pragma omp parallel
{
const unsigned int  tid     = omp_get_thread_num();
const unsigned long tStride = tid * this->bodies->getN();

#pragma omp for schedule(runtime)
for(unsigned long iBody = jOff +1; iBody < this->bodies->getN(); iBody++)
{
unsigned long jEnd = std::min(jOff + blockSize, iBody);
for(unsigned long jBody = jOff; jBody < jEnd; jBody++)
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
