
#ifndef MULTIPARTICLEBROWNIANMOTION_H
#define MULTIPARTICLEBROWNIANMOTION_H

#include "MoveBase.h"
#include "System.h"
#include "StaticVals.h"
#include <cmath>
#include "Random123Wrapper.h"
#ifdef GOMC_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDAMemoryManager.cuh"
#include "TransformParticlesCUDAKernel.cuh"
#include "VariablesCUDA.cuh"
#endif

class MultiParticleBrownian : public MoveBase
{
public:
MultiParticleBrownian(System &sys, StaticVals const& statV);
~MultiParticleBrownian() {
#ifdef GOMC_CUDA
cudaVars = NULL;
cudaFreeHost(kill);
kill = NULL;
#endif
}

virtual uint Prep(const double subDraw, const double movPerc);
virtual uint PrepNEMTMC(const uint box, const uint midx = 0, const uint kidx = 0);
virtual void CalcEn();
virtual uint Transform();
virtual void Accept(const uint rejectState, const ulong step);
virtual void PrintAcceptKind();

private:
uint bPick;
bool initMol;
SystemPotential sysPotNew;
XYZArray molTorqueRef;
XYZArray molTorqueNew;
XYZArray atomForceRecNew;
XYZArray molForceRecNew;
XYZArray t_k;
XYZArray r_k;
Coordinates newMolsPos;
COM newCOMs;
int moveType;
std::vector<uint> moleculeIndex;
const MoleculeLookup& molLookup;
Random123Wrapper &r123wrapper;
bool allTranslate;
#ifdef GOMC_CUDA
VariablesCUDA *cudaVars;
bool isOrthogonal;
int *kill; 
#endif

double GetCoeff();
void CalculateTrialDistRot();
void RotateForceBiased(uint molIndex);
void TranslateForceBiased(uint molIndex);
void SetMolInBox(uint box);
XYZ CalcRandomTransform(XYZ const &lb, double const max, uint molIndex);
double CalculateWRatio(XYZ const &lb_new, XYZ const &lb_old, XYZ const &k,
double max4);
};

inline MultiParticleBrownian::MultiParticleBrownian(System &sys, StaticVals const &statV) :
MoveBase(sys, statV),
newMolsPos(sys.boxDimRef, newCOMs, sys.molLookupRef, sys.prng, statV.mol),
newCOMs(sys.boxDimRef, newMolsPos, sys.molLookupRef, statV.mol),
molLookup(sys.molLookup), r123wrapper(sys.r123wrapper)
{
molTorqueNew.Init(sys.com.Count());
molTorqueRef.Init(sys.com.Count());
atomForceRecNew.Init(sys.coordinates.Count());
molForceRecNew.Init(sys.com.Count());

t_k.Init(sys.com.Count());
r_k.Init(sys.com.Count());
newMolsPos.Init(sys.coordinates.Count());
newCOMs.Init(sys.com.Count());

initMol = false;

allTranslate = false;
uint numAtomsPerKind = 0;
for (uint k = 0; k < molLookup.GetNumKind(); ++k) {
numAtomsPerKind += molRef.NumAtoms(k);
}
allTranslate = (numAtomsPerKind == molLookup.GetNumKind());

#ifdef GOMC_CUDA
cudaVars = sys.statV.forcefield.particles->getCUDAVars();
isOrthogonal = statV.isOrthogonal;
cudaMallocHost((void**) &kill, sizeof(int));
checkLastErrorCUDA(__FILE__, __LINE__);
#endif
}

inline void MultiParticleBrownian::PrintAcceptKind()
{
printf("%-37s", "% Accepted MultiParticle-Brownian ");
for(uint b = 0; b < BOX_TOTAL; b++) {
printf("%10.5f ", 100.0 * moveSetRef.GetAccept(b, mv::MULTIPARTICLE_BM));
}
std::cout << std::endl;
}


inline void MultiParticleBrownian::SetMolInBox(uint box)
{
#if ENSEMBLE == GCMC || ENSEMBLE == GEMC
moleculeIndex.clear();
MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
while(thisMol != end) {
if(!molLookup.IsFix(*thisMol)) {
moleculeIndex.push_back(*thisMol);
}
thisMol++;
}
#else
if(!initMol) {
moleculeIndex.clear();
MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
while(thisMol != end) {
if(!molLookup.IsFix(*thisMol)) {
moleculeIndex.push_back(*thisMol);
}
thisMol++;
}
initMol = true;
}
#endif
}

inline uint MultiParticleBrownian::Prep(const double subDraw, const double movPerc)
{
GOMC_EVENT_START(1, GomcProfileEvent::PREP_MULTIPARTICLE_BM);
uint state = mv::fail_state::NO_FAIL;
#if ENSEMBLE == GCMC
bPick = mv::BOX0;
#else
prng.PickBox(bPick, subDraw, movPerc);
#endif

if(allTranslate) {
moveType = mp::MPDISPLACE;
} else {
moveType = prng.randIntExc(mp::MPTOTALTYPES);
}

SetMolInBox(bPick);
if (moleculeIndex.size() == 0) {
std::cout << "Warning: MultiParticleBrownian move can't move any molecules. Skipping..." << std::endl;
state = mv::fail_state::NO_MOL_OF_KIND_IN_BOX;
return state;
}

if(moveSetRef.GetSingleMoveAccepted(bPick)) {
GOMC_EVENT_START(1, GomcProfileEvent::CALC_EN_MULTIPARTICLE_BM);
calcEwald->CopyRecip(bPick);

calcEwald->BoxForceReciprocal(coordCurrRef, atomForceRecRef, molForceRecRef, bPick);

calcEnRef.BoxForce(sysPotRef, coordCurrRef, atomForceRef, molForceRef,
boxDimRef, bPick);

calcEnRef.CalculateTorque(moleculeIndex, coordCurrRef, comCurrRef,
atomForceRef, atomForceRecRef, molTorqueRef, bPick);

sysPotRef.Total();
GOMC_EVENT_STOP(1, GomcProfileEvent::CALC_EN_MULTIPARTICLE_BM);
}
coordCurrRef.CopyRange(newMolsPos, 0, 0, coordCurrRef.Count());
comCurrRef.CopyRange(newCOMs, 0, 0, comCurrRef.Count());
#if ENSEMBLE == GCMC || ENSEMBLE == GEMC
atomForceRef.CopyRange(atomForceNew, 0, 0, atomForceNew.Count());
molForceRef.CopyRange(molForceNew, 0, 0, molForceNew.Count());
atomForceRecRef.CopyRange(atomForceRecNew, 0, 0, atomForceRecNew.Count());
molForceRecRef.CopyRange(molForceRecNew, 0, 0, molForceRecNew.Count());
molTorqueRef.CopyRange(molTorqueNew, 0, 0, molTorqueNew.Count());
#endif
GOMC_EVENT_STOP(1, GomcProfileEvent::PREP_MULTIPARTICLE_BM);
return state;
}

inline uint MultiParticleBrownian::PrepNEMTMC(const uint box, const uint midx, const uint kidx)
{
GOMC_EVENT_START(1, GomcProfileEvent::PREP_MULTIPARTICLE_BM);
bPick = box;
uint state = mv::fail_state::NO_FAIL;
if(allTranslate) {
moveType = mp::MPDISPLACE;
} else {
moveType = prng.randIntExc(mp::MPTOTALTYPES);
}

SetMolInBox(bPick);
if (moleculeIndex.size() == 0) {
std::cout << "Warning: MultiParticleBrownian move can't move any molecules. Skipping..." << std::endl;
state = mv::fail_state::NO_MOL_OF_KIND_IN_BOX;
return state;
}

if(moveSetRef.GetSingleMoveAccepted(bPick)) {
calcEwald->CopyRecip(bPick);

calcEwald->BoxForceReciprocal(coordCurrRef, atomForceRecRef, molForceRecRef, bPick);

calcEnRef.BoxForce(sysPotRef, coordCurrRef, atomForceRef, molForceRef,
boxDimRef, bPick);

calcEnRef.CalculateTorque(moleculeIndex, coordCurrRef, comCurrRef,
atomForceRef, atomForceRecRef, molTorqueRef, bPick);

sysPotRef.Total();
}
coordCurrRef.CopyRange(newMolsPos, 0, 0, coordCurrRef.Count());
comCurrRef.CopyRange(newCOMs, 0, 0, comCurrRef.Count());
#if ENSEMBLE == GCMC || ENSEMBLE == GEMC
atomForceRef.CopyRange(atomForceNew, 0, 0, atomForceNew.Count());
molForceRef.CopyRange(molForceNew, 0, 0, molForceNew.Count());
atomForceRecRef.CopyRange(atomForceRecNew, 0, 0, atomForceRecNew.Count());
molForceRecRef.CopyRange(molForceRecNew, 0, 0, molForceRecNew.Count());
molTorqueRef.CopyRange(molTorqueNew, 0, 0, molTorqueNew.Count());
#endif
GOMC_EVENT_STOP(1, GomcProfileEvent::PREP_MULTIPARTICLE_BM);
return state;
}

inline uint MultiParticleBrownian::Transform()
{
GOMC_EVENT_START(1, GomcProfileEvent::TRANS_MULTIPARTICLE_BM);
uint state = mv::fail_state::NO_FAIL;

#ifdef GOMC_CUDA
kill[0] = 0;
if(moveType == mp::MPROTATE) {
double r_max = moveSetRef.GetRMAX(bPick);
BrownianMotionRotateParticlesGPU(
cudaVars,
moleculeIndex,
molTorqueRef,
newMolsPos, 
newCOMs, 
r_k, 
boxDimRef.GetAxis(bPick),
BETA, 
r_max,
r123wrapper.GetStep(), 
r123wrapper.GetKeyValue(), 
r123wrapper.GetSeedValue(),
bPick,
isOrthogonal,
kill);
} else {
double t_max = moveSetRef.GetTMAX(bPick);
BrownianMotionTranslateParticlesGPU(
cudaVars,
moleculeIndex,
molForceRef,
molForceRecRef,
newMolsPos, 
newCOMs, 
t_k, 
boxDimRef.GetAxis(bPick),
BETA, 
t_max,
r123wrapper.GetStep(), 
r123wrapper.GetKeyValue(), 
r123wrapper.GetSeedValue(),
bPick,
isOrthogonal,
kill);
}
if(kill[0]) {
std::cout << "Error: Transformation of " << kill[0] << " molecules in Multiparticle Brownian Motion move failed!" << std::endl;
if(moveType == mp::MPROTATE) {
std::cout << "       Trial rotation is not a finite number!" << std::endl << std::endl;
}
else {
std::cout << "       Either trial translation is not a finite number or the translation" << std::endl
<< "       exceeded half of the box length!" << std::endl << std::endl;
}
std::cout << "This might be due to a bad initial configuration, where atoms of the molecules" << std::endl
<< "are too close to each other or overlap. Please equilibrate your system using" << std::endl
<< "rigid body translation or rotation MC moves before using the Multiparticle" << std::endl
<< "Brownian Motion move." << std::endl << std::endl;
exit(EXIT_FAILURE);
} 
#else
CalculateTrialDistRot();
#endif
GOMC_EVENT_STOP(1, GomcProfileEvent::TRANS_MULTIPARTICLE_BM);
return state;
}

inline void MultiParticleBrownian::CalcEn()
{
GOMC_EVENT_START(1, GomcProfileEvent::CALC_EN_MULTIPARTICLE_BM);
cellList.GridBox(boxDimRef, newMolsPos, molLookup, bPick);

calcEwald->backupMolCache();

calcEwald->BoxReciprocalSums(bPick, newMolsPos);

sysPotNew = sysPotRef;
sysPotNew = calcEnRef.BoxForce(sysPotNew, newMolsPos, atomForceNew,
molForceNew, boxDimRef, bPick);
sysPotNew.boxEnergy[bPick].recip = calcEwald->BoxReciprocal(bPick, false);
calcEwald->BoxForceReciprocal(newMolsPos, atomForceRecNew, molForceRecNew,
bPick);

calcEnRef.CalculateTorque(moleculeIndex, newMolsPos, newCOMs, atomForceNew,
atomForceRecNew, molTorqueNew, bPick);

sysPotNew.Total();
GOMC_EVENT_STOP(1, GomcProfileEvent::CALC_EN_MULTIPARTICLE);
}

inline double MultiParticleBrownian::CalculateWRatio(XYZ const &lb_new, XYZ const &lb_old,
XYZ const &k, double max4)
{
double w_ratio = 0.0;
XYZ old_var = lb_old - k;
XYZ new_var = lb_new + k;

w_ratio -= (new_var.LengthSq() / max4);
w_ratio += (old_var.LengthSq() / max4);

return w_ratio;
}

inline double MultiParticleBrownian::GetCoeff()
{
double w_ratio = 0.0;
double r_max = moveSetRef.GetRMAX(bPick);
double t_max = moveSetRef.GetTMAX(bPick);
double r_max4 = 4.0 * r_max;
double t_max4 = 4.0 * t_max;

if(moveType == mp::MPROTATE) {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(r_max, t_max, r_max4, t_max4) reduction(+:w_ratio)
#endif
for(uint m = 0; m < moleculeIndex.size(); m++) {
uint molNumber = moleculeIndex[m];
XYZ bt_old = molTorqueRef.Get(molNumber) * BETA * r_max;
XYZ bt_new = molTorqueNew.Get(molNumber) * BETA * r_max;
w_ratio += CalculateWRatio(bt_new, bt_old, r_k.Get(molNumber), r_max4);
} 
} else {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(r_max, t_max, r_max4, t_max4) reduction(+:w_ratio)
#endif
for(uint m = 0; m < moleculeIndex.size(); m++) {
uint molNumber = moleculeIndex[m];
XYZ bf_old = (molForceRef.Get(molNumber) + molForceRecRef.Get(molNumber)) *
BETA * t_max;
XYZ bf_new = (molForceNew.Get(molNumber) + molForceRecNew.Get(molNumber)) *
BETA * t_max;
w_ratio += CalculateWRatio(bf_new, bf_old, t_k.Get(molNumber), t_max4);
}
}

return w_ratio;
}

inline void MultiParticleBrownian::Accept(const uint rejectState, const ulong step)
{
GOMC_EVENT_START(1, GomcProfileEvent::ACC_MULTIPARTICLE_BM);
double MPCoeff = GetCoeff();
double accept = exp(-BETA * (sysPotNew.Total() - sysPotRef.Total()) + MPCoeff);
bool result = (rejectState == mv::fail_state::NO_FAIL) && prng() < accept;
if(result) {
sysPotRef = sysPotNew;
swap(coordCurrRef, newMolsPos);
swap(comCurrRef, newCOMs);
swap(molForceRef, molForceNew);
swap(atomForceRef, atomForceNew);
swap(molForceRecRef, molForceRecNew);
swap(atomForceRecRef, atomForceRecNew);
swap(molTorqueRef, molTorqueNew);
calcEwald->UpdateRecip(bPick);
velocity.UpdateBoxVelocity(bPick);
} else {
cellList.GridAll(boxDimRef, coordCurrRef, molLookup);
calcEwald->exgMolCache();
}

moveSetRef.UpdateMoveSettingMultiParticle(bPick, result, moveType);
moveSetRef.Update(mv::MULTIPARTICLE_BM, result, bPick);
GOMC_EVENT_STOP(1, GomcProfileEvent::ACC_MULTIPARTICLE_BM);
}

inline XYZ MultiParticleBrownian::CalcRandomTransform(XYZ const &lb, double const max, uint molIndex)
{
XYZ lbmax = lb * max;
XYZ num, randnums;
double stdDev = sqrt(2.0 * max);

randnums = r123wrapper.GetGaussianCoords(molIndex, 0.0, stdDev);
num.x = lbmax.x + randnums.x;
num.y = lbmax.y + randnums.y;
num.z = lbmax.z + randnums.z;


if (!std::isfinite(num.Length())) {
std::cout << "Error: Trial transform is not a finite number in Brownian Motion Multiparticle move." << std::endl;
std::cout << "       Trial transform: " << num << std::endl;
exit(EXIT_FAILURE);
}
return num;
}

inline void MultiParticleBrownian::CalculateTrialDistRot()
{
double r_max = moveSetRef.GetRMAX(bPick);
double t_max = moveSetRef.GetTMAX(bPick);
if(moveType == mp::MPROTATE) { 
double *x = r_k.x;
double *y = r_k.y;
double *z = r_k.z;
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(r_max, x, y, z)
#endif
for(uint m = 0; m < moleculeIndex.size(); m++) {
uint molIndex = moleculeIndex[m];
XYZ bt = molTorqueRef.Get(molIndex) * BETA;
XYZ val = CalcRandomTransform(bt, r_max, molIndex);
x[molIndex] = val.x; 
y[molIndex] = val.y; 
z[molIndex] = val.z;
RotateForceBiased(molIndex);
}
} else { 
double *x = t_k.x;
double *y = t_k.y;
double *z = t_k.z;
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(t_max, x, y, z)
#endif
for(int m = 0; m < (int) moleculeIndex.size(); m++) {
uint molIndex = moleculeIndex[m];
XYZ bf = (molForceRef.Get(molIndex) + molForceRecRef.Get(molIndex)) * BETA;
XYZ val = CalcRandomTransform(bf, t_max, molIndex);
x[molIndex] = val.x; 
y[molIndex] = val.y; 
z[molIndex] = val.z; 
TranslateForceBiased(molIndex);
}
}
}

inline void MultiParticleBrownian::RotateForceBiased(uint molIndex)
{
XYZ rot = r_k.Get(molIndex);
double rotLen = rot.Length();
RotationMatrix matrix;

matrix = RotationMatrix::FromAxisAngle(rotLen, rot.Normalize());

XYZ center = comCurrRef.Get(molIndex);
uint start, stop, len;
molRef.GetRange(start, stop, len, molIndex);

XYZArray temp(len);
newMolsPos.CopyRange(temp, start, 0, len);
boxDimRef.UnwrapPBC(temp, bPick, center);

for(uint p = 0; p < len; p++) {
temp.Add(p, -center);
temp.Set(p, matrix.Apply(temp[p]));
temp.Add(p, center);
}
boxDimRef.WrapPBC(temp, bPick);
temp.CopyRange(newMolsPos, 0, start, len);
}

inline void MultiParticleBrownian::TranslateForceBiased(uint molIndex)
{
XYZ shift = t_k.Get(molIndex);
if(shift > boxDimRef.GetHalfAxis(bPick)) {
std::cout << "Error: Trial Displacement exceeds half of the box length in Multiparticle" << std::endl
<< "       Brownian Motion move!" << std::endl;
std::cout << "       Trial transformation vector: " << shift << std::endl;
std::cout << "       Box Dimensions: " << boxDimRef.GetAxis(bPick) << std::endl << std::endl;
std::cout << "This might be due to a bad initial configuration, where atoms of the molecules" << std::endl 
<< "are too close to each other or overlap. Please equilibrate your system using" << std::endl
<< "rigid body translation or rotation MC moves before using the Multiparticle" << std::endl
<< "Brownian Motion move." << std::endl << std::endl;
exit(EXIT_FAILURE);
} 

XYZ newcom = comCurrRef.Get(molIndex);
uint stop, start, len;
molRef.GetRange(start, stop, len, molIndex);
XYZArray temp(len);
newMolsPos.CopyRange(temp, start, 0, len);
temp.AddAll(shift);
newcom += shift;
boxDimRef.WrapPBC(temp, bPick);
newcom = boxDimRef.WrapPBC(newcom, bPick);
temp.CopyRange(newMolsPos, 0, start, len);
newCOMs.Set(molIndex, newcom);
}

#endif
