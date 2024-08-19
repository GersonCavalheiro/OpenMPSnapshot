
#ifndef PSYCHICSNIFFLE_SNIFFLE_H
#define PSYCHICSNIFFLE_SNIFFLE_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <functional>
#include <assert.h>

#include "nselector.h"
#include "samplertable.h"
#include "splice.h"
#include "taus88.h"


using namespace util;

namespace sniffle {

template<typename StateType>
struct ByteAnalyser {
const static int StateSize = sizeof(StateType);

uint8_t *GetByteArr(StateType &state) { return (uint8_t *) &state; }

uint8_t distr[StateSize][256];
uint8_t dSampler[StateSize][65535];
uint16_t dSamplerN[StateSize];

int iteration;
bool negated;

void dumpStats() {
for (int ss = 0; ss < StateSize; ss++) {
for (int b = 0; b < 256; b++) putchar('A' + 25 * (distr[ss][b]) / 255);
putchar('\n');
}
putchar('\n');
}

float_t calcSmallestChannelDifference()
{
uint8_t min[StateSize], max[StateSize];

#pragma omp parallel for
for (int ss = 0; ss < StateSize; ss++)
{
min[ss] = 255;
max[ss] = 0;
for (int b = 0; b < 256; b++) {
min[ss] = std::min(min[ss], distr[ss][b]);
max[ss] = std::max(max[ss], distr[ss][b]);
}
}

uint8_t deltaE = (uint8_t)~0;
for (int s1 = 0; s1 < StateSize; s1++)
for (int s2 = 0; s2 < StateSize; s2++)
deltaE = std::min(deltaE, (uint8_t)(max[s1] - min[s2]));

return (float_t)deltaE / 255.f;
}

void crank(StateType *stateArr, int *eliteArr, const int eliteSamples) {
#if 0
dumpStats();
#endif

if( ++iteration % 10 == 0 ) {
negated = true;

#pragma omp parallel for
for (int ss = 0; ss < StateSize; ss++)
for (int b = 0; b < 256; b++)
distr[ss][b] = ~distr[ss][b];
} else {
if( negated ) {
negated = !negated;
#pragma omp parallel for
for (int ss = 0; ss < StateSize; ss++)
for (int b = 0; b < 256; b++)
distr[ss][b] = ~distr[ss][b];
}

#pragma omp parallel for
for (int ss = 0; ss < StateSize; ss++)
for (int b = 0; b < 256; b++)
if (distr[ss][b] > 1) distr[ss][b] -= 1;

#pragma omp parallel for
for (int i = 0; i < eliteSamples; i++) {
for (int ss = 0; ss < StateSize; ss++) {
int b = GetByteArr(stateArr[eliteArr[i]])[ss];
if (distr[ss][b] < 250) distr[ss][b] += 5;

for (int bo = 1; bo < 4; bo++) {
if (b > 0 && distr[ss][b - bo] < 250) distr[ss][b - bo] += 1;
if (b < 255 && distr[ss][b + bo] < 250) distr[ss][b + bo] += 1;
}
}
}
}

#pragma omp parallel for
for (int ss = 0; ss < StateSize; ss++) {
dSamplerN[ss] = buildSamplerTable<uint8_t, 65535, uint8_t, 256>(&(dSampler[ss][0]), &(distr[ss][0]));
}

}

void reset() {
iteration = 0;
negated = false;

memset(distr, UINT8_MAX >> 2, StateSize * 256); 

#pragma omp parallel for
for (int ss = 0; ss < StateSize; ss++) {
for (int i = 0; i < 256; i++)
dSampler[ss][i] = i; 
dSamplerN[ss] = ~0;
}
}

void mutatebyte(uint8_t *p, Taus88& fnRand) {
int byte = fnRand() % StateSize;
p[byte] = dSampler[byte][ fnRand() % dSamplerN[byte] ]; 
}

void randomize(uint8_t *p, Taus88& fnRand) {
for (int ss = 0; ss < StateSize; ss++) {
p[ss] = dSampler[ss][ fnRand() % dSamplerN[ss] ];
}
}
};


template<typename StateType>
struct NullAnalyser {
const static int StateSize = sizeof(StateType);

void crank(StateType *stateArr, int *eliteArr, const int eliteSamples) { }

void reset() {}

void mutatebyte(uint8_t *p, Taus88& fnRand) {
int byte = fnRand() % StateSize;
p[byte] = fnRand();
}

void randomize(uint8_t *p, Taus88& fnRand) {
for (int ss = 0; ss < StateSize; ss++) {
p[ss] = fnRand();
}
}
};


template<typename StateType, uint Population, template <typename ST> typename StateAnalyser>
struct Maximizer {
const static int StateSize = sizeof(StateType);

const static int Group2End = Population * .30;
const static int Group3End = Population * .50;
const static int Group4End = Population * .70;
const static int Group5End = Population * .80;
const static int Group6End = Population * .90;

const static int EliteSamples = 5 + Group3End * .05;
int eliteSamples[EliteSamples];

int pa, pb;

StateType state[2][Population];

float_t e[Population];
uint16_t eSampler[65535];
uint16_t eSamplerN;

StateAnalyser<StateType> stateAnalyser;
Taus88State taus88State;

uint8_t *GetByteArr(StateType &state) { return (uint8_t *) &state; }

StateType *GetStateArr() { return &(state[pa][0]); }

uint8_t *newPop(int i = 0) { return GetByteArr(state[pb][i]); }

uint8_t *oldPop(int i = 0) { return GetByteArr(state[pa][i]); }

Maximizer() {
taus88State.seed();
}

void dumpStats() {
for (int ss = 0; ss < StateSize; ss++) {
for (int p = 0; p < Population; p++)
putchar('A' + 25 * oldPop(p)[ss] / 255);
putchar('\n');
}
putchar('\n');
}

void reset(int preserve = 0) {
pa = 0;
pb = 1;
stateAnalyser.reset();

#pragma omp parallel
{
Taus88 taus88(taus88State);
#pragma omp for
for (int i = preserve; i < Population; i++) {
stateAnalyser.randomize(oldPop(i), taus88);
}
}
}

void crank(float *f) {
#if 0
dumpStats();
#endif

int imax = 0;
int imin = 0;
for (int i = 1; i < Population; i++) {
if (f[i] > f[imax]) imax = i;
if (f[i] < f[imin]) imin = i;
}
for (int i = 1; i < Population; i++) {
if (f[i] == f[imax]) f[i] = f[imin];
}

eSamplerN = buildSamplerTable<uint16_t, 65535, float_t, Population>(eSampler, f);

eliteSamples[0] = imax; 
#pragma omp parallel
{
Taus88 taus88(taus88State);
#pragma omp for
for (int i = 1; i < EliteSamples; i++)
eliteSamples[i] = eSampler[ taus88() % eSamplerN ];
}

stateAnalyser.crank(GetStateArr(), eliteSamples, EliteSamples);


memcpy(newPop(0), oldPop(imax), (uint) StateSize);

#pragma omp parallel
{
Taus88 taus88(taus88State);
NSelector<2> nselector( eSamplerN );

#pragma omp for nowait
for (int i = 1; i < Group2End; i++) {
int p = eSampler[ taus88() % eSamplerN ];
memcpy(newPop(i), oldPop(p), (uint) StateSize);
}
#pragma omp for nowait
for (int i = Group2End +1; i < Group3End; i++) {
int p = eSampler[ taus88() % eSamplerN ];
memcpy(newPop(i), oldPop(p), (uint) StateSize);
stateAnalyser.mutatebyte(newPop(i), taus88);
}
#pragma omp for nowait
for (int i = Group3End +1; i < Group4End; i++) {
int b = eSampler[ taus88() % eSamplerN ];
splice<StateSize>(newPop(i), oldPop(0), oldPop(b), (uint) taus88());
}
#pragma omp for nowait
for (int i = Group4End +1; i < Group5End; i++) {
int a = eSampler[ taus88() % eSamplerN ];
splice<StateSize>(newPop(i), oldPop(a), oldPop(0), (uint) taus88());
}
#pragma omp for nowait
for (int i = Group5End +1; i < Group6End; i++) {
nselector.reset();
int a = eSampler[ nselector.select(taus88) ];
int b = eSampler[ nselector.select(taus88) ];
splice<StateSize>(newPop(i), oldPop(a), oldPop(b), (uint) taus88());
}
#pragma omp for nowait
for (int i = Group6End +1; i < Population; i++) {
stateAnalyser.randomize(newPop(i), taus88);
}
}

std::swap(pa, pb);
}

};

}

#endif 
