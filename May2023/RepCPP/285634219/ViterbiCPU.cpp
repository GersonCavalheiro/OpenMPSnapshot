

#include <cstdlib>
#include <cstdio>

int ViterbiCPU(float &viterbiProb,
int *viterbiPath,
int *obs, 
const int &nObs, 
float *initProb,
float *mtState, 
const int &nState,
float *mtEmit)
{
float *maxProbNew = (float*)malloc(sizeof(float)*nState);
float *maxProbOld = (float*)malloc(sizeof(float)*nState);
int **path = (int**)malloc(sizeof(int*)*(nObs-1));
for (int i = 0; i < nObs-1; i++)
path[i] = (int*)malloc(sizeof(int)*nState);

#pragma omp parallel for
for (int i = 0; i < nState; i++)
{
maxProbOld[i] = initProb[i];
}

for (int t = 1; t < nObs; t++) 
{ 
#pragma omp parallel for
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
path[t-1][iState] = maxState;
}

#pragma omp parallel for
for (int iState = 0; iState < nState; iState++) 
{
maxProbOld[iState] = maxProbNew[iState];
}
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
viterbiPath[t] = path[t][viterbiPath[t+1]];
}

free(maxProbNew);
free(maxProbOld);
for (int i = 0; i < nObs-1; i++) free(path[i]);
free(path);
return 1;
}
