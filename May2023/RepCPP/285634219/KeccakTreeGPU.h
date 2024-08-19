

#ifndef KECCAKTREEGPU_H_INCLUDED
#define KECCAKTREEGPU_H_INCLUDED

#include <omp.h>
#include "KeccakTree.h"
#include "KeccakTypes.h"
#include "KeccakF.h"


#pragma omp declare target 
void KeccakTreeGPU(tKeccakLane * h_inBuffer, tKeccakLane * h_outBuffer,  const tKeccakLane *h_KeccakF_RoundConstants);
#pragma omp end declare target 



#endif 
