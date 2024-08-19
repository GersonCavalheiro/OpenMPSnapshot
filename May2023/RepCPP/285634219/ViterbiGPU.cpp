

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <omp.h>

int ViterbiGPU(float &viterbiProb,
int   *__restrict viterbiPath,
int   *__restrict obs, 
const int nObs, 
float *__restrict initProb,
float *__restrict mtState, 
const int nState,
const int nEmit,
float *__restrict mtEmit)
{
float maxProbNew[nState];
int path[(nObs-1)*nState];
float *maxProbOld = initProb;

#pragma omp target data map(to:initProb[0:nState], \
mtState[0:nState*nState], \
mtEmit[0:nEmit*nState], \
obs[0:nObs],\
maxProbOld[0:nState]) \
map(from: maxProbNew[0:nState], path[0:(nObs-1)*nState])
{
auto start = std::chrono::steady_clock::now();

for (int t = 1; t < nObs; t++) 
{ 
#pragma omp target teams distribute parallel for thread_limit(256)
for (int iState = 0; iState < nState; iState++) 
{
float maxProb = 0.0;
int maxState = -1;
for (int preState = 0; preState < nState; preState++) 
{
float p = maxProbOld[preState] + mtState[iState*nState + preState];
if (p > maxProb) 
{
maxProb = p;
maxState = preState;
}
}
maxProbNew[iState] = maxProb + mtEmit[obs[t]*nState+iState];
path[(t-1)*nState+iState] = maxState;
}

#pragma omp target teams distribute parallel for thread_limit(256)
for (int iState = 0; iState < nState; iState++) 
{
maxProbOld[iState] = maxProbNew[iState];
}
}

auto end = std::chrono::steady_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
printf("Device execution time of Viterbi iterations %f (s)\n", time * 1e-9f);
}

float maxProb = 0.0;
int maxState = -1;
for (int i = 0; i < nState; i++) 
{
if (maxProbNew[i] > maxProb) 
{
maxProb = maxProbNew[i];
maxState = i;
}
}
viterbiProb = maxProb;

viterbiPath[nObs-1] = maxState;
for (int t = nObs-2; t >= 0; t--) 
{
viterbiPath[t] = path[t*nState+viterbiPath[t+1]];
}

return 1;
}
