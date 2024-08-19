
#include <iostream>
#include <cstring>
#include <cstring>

#include "sniffle.h"

#include "cpuinfo.h"

using namespace sniffle;


template<typename Rep, uint Dimension, uint Population>
struct Schwefel
{
typedef Rep StateType[Dimension];

static float_t Eval(const StateType& state)
{
float_t sum = 0.f;
for(int d=0; d<Dimension; d++)
{
float_t x = (float_t)1000.f * (float_t)state[d] / (float_t)((Rep)~0) - (float_t)500.f;
sum += x * sin(sqrt(abs(x)));
}
return sum - 418.9829 * Dimension;
}

static void Solve(int solns)
{
printf("Minimze Schwefel<%d> : https:

float_t f[Population];
Maximizer<StateType, Population, ByteAnalyser> solver;

float_t best = 0.f;
uint i = 0;

uint t = 0;
solver.reset();
while( t < 1e6 ) {
#pragma omp for
for( int p=0; p<Population; p++ )
f[p] = Eval( solver.GetStateArr()[p] );

if( t == 10000 || best != f[0] ) {
float ff = solver.stateAnalyser.calcSmallestChannelDifference();
best = f[0];
if( t == 10000 || ff >= .99f ) {
printf("%u, %8.4f, %8.4f, [", t, best, ff);
for( int d=0; d<Dimension; d++ ) {
StateType &state = *solver.GetStateArr();
double_t val = (float_t) 1000. * (float_t) state[d] / (float_t) ((Rep) ~0) - (float_t) 500.;
printf("%8.8f%s ", val, d < Dimension - 1 ? "," : "]\n" );
}
fflush(stdout);
t = 0;
solver.reset();
if( ++i == solns ) break;
continue;
}
}

solver.crank(f);
t++;
}
}
};

int main()
{
srand(int(time(NULL)));

#if defined(NDEBUG)
{
int cores = EnumCores();
printf("OpenMP using %d threads\n", cores);
omp_set_num_threads(cores);
}
#endif

Schwefel<uint16_t, 20, 400>::Solve(10);

return 0;
}
