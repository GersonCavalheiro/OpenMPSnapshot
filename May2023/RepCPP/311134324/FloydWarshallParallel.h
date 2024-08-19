#include <iostream>
#include "FloydWarshallBase.h"
#include <valarray>
#include <omp.h>

using namespace std;

#ifndef FLOYD_WARSHALL_PARALLEL_H
#define FLOYD_WARSHALL_PARALLEL_H

class FloydWarshallParallel : public FloydWarshallBase
{
private:
valarray<omp_lock_t> lock_objects;

void safe_set(int i, int j, double value)
{
auto n = matrix->size();
omp_set_lock(&lock_objects[i * n + j]);
(*matrix)[i * n + j] = value;
omp_unset_lock(&lock_objects[i * n + j]);
}

public:
virtual string GetLabel() { return "floydwarshall_parallel"; }

virtual void UpdateParam(int n)
{
FloydWarshallBase::UpdateParam(n);
lock_objects = valarray<omp_lock_t>(n * n);
for (int i = 0; i < n * n; i++)
{
lock_objects[i] = omp_lock_t();
}
}

virtual void Run()
{
for (int k = 0; k < n; k++)
{
#pragma omp parallel for
for (int i = 0; i < n; i++)
{
auto v = (*matrix)[i * n + k];
for (int j = 0; j < n; ++j)
{
auto val = v + (*matrix)[k * n + j];
if ((*matrix)[i * n + j] > val)
{
safe_set(i, j, val);
}
}
}
}
}
};

#endif