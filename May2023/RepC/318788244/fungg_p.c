#include <math.h>
#include <float.h>
#include "definegg.h" 
double gendist(float *elem1, float *elem2)
{
float dist = 0;
int i;
for (i = 0; i < NFEAT; i++)
{
dist = dist + powl((elem1[i] - elem2[i]), 2);
}
return sqrt(dist);
}
void closestgroup(int nelem, float elem[][NFEAT], float cent[][NFEAT], int *grind)
{
int i, n;
double dist, closest = FLT_MAX;
#pragma omp parallel for default(none) shared(nelem, elem, cent, grind) private(i, n, dist, closest) num_threads(NUM_THREADS) schedule(dynamic, 3)
for (i = 0; i < nelem; i++)
{
closest = FLT_MAX;
for (n = 0; n < NGROUPS; n++)
{
dist = gendist(elem[i], cent[n]);
if (dist < closest)
{
closest = dist;
grind[i] = n;
}
}
}
}
void compactness(float elem[][NFEAT], struct ginfo *iingrs, float *compact)
{
int num, j, i, e;
float sum;
#pragma omp parallel for default(none) shared(elem, iingrs, compact) private(i, j, e, sum, num) num_threads(NUM_THREADS) schedule(static, 1)
for (i = 0; i < NGROUPS; i++)
{
num = 0;
sum = 0.0;
for (j = 0; j < iingrs[i].size; j++)
{
for (e = j + 1; e < iingrs[i].size; e++)
{
sum = sum + gendist(elem[iingrs[i].members[j]], elem[iingrs[i].members[e]]);
num++;
}
}
compact[i] = sum / num > 0 ? sum / num : 0;
}
}
void diseases(struct ginfo *iingrs, float dise[][TDISEASE], struct analysis *disepro)
{
int i, j, m;
float sum = 0;
for (i = 0; i < TDISEASE; i++)
{
disepro[i].max = FLT_MIN;
disepro[i].min = FLT_MAX;
}
#pragma omp parallel for default(none) shared(dise, iingrs, disepro, i) private(j, sum, m) num_threads(NUM_THREADS) schedule(static, 1)
for (i = 0; i < NGROUPS; i++)
{
for (j = 0; j < TDISEASE; j++)
{
sum = 0;
for (m = 0; m < iingrs[i].size; m++)
{
sum += dise[iingrs[i].members[m]][j];
}
sum /= iingrs[i].size;
if (sum > disepro[j].max)
{
disepro[j].max = sum;
disepro[j].gmax = i;
}
if (sum < disepro[j].min)
{
disepro[j].min = sum;
disepro[j].gmin = i;
}
}
}
}