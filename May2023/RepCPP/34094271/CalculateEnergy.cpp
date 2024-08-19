
#include "CalculateEnergy.h"        
#include "EwaldCached.h"            
#include "Ewald.h"                  
#include "NoEwald.h"                
#include "EnergyTypes.h"            
#include "EnsemblePreprocessor.h"   
#include "BasicTypes.h"             
#include "System.h"                 
#include "StaticVals.h"             
#include "Forcefield.h"             
#include "MoleculeLookup.h"
#include "MoleculeKind.h"
#include "Coordinates.h"
#include "BoxDimensions.h"
#include "BoxDimensionsNonOrth.h"
#include "TrialMol.h"
#include "GeomLib.h"
#include "NumLib.h"
#include <cassert>
#ifdef GOMC_CUDA
#include "CalculateEnergyCUDAKernel.cuh"
#include "CalculateForceCUDAKernel.cuh"
#include "ConstantDefinitionsCUDAKernel.cuh"
#endif
#include "GOMCEventsProfile.h"
#define NUMBER_OF_NEIGHBOR_CELL 27


using namespace geom;

CalculateEnergy::CalculateEnergy(StaticVals & stat, System & sys) :
forcefield(stat.forcefield), mols(stat.mol), currentCoords(sys.coordinates),
currentCOM(sys.com),
lambdaRef(sys.lambdaRef),
atomForceRef(sys.atomForceRef),
molForceRef(sys.molForceRef),
#ifdef VARIABLE_PARTICLE_NUMBER
molLookup(sys.molLookup),
#else
molLookup(stat.molLookup),
#endif
currentAxes(sys.boxDimRef),
cellList(sys.cellList)
{
}


void CalculateEnergy::Init(System & sys)
{
uint maxAtomInMol = 0;
calcEwald = sys.GetEwald();
electrostatic = forcefield.electrostatic;
ewald = forcefield.ewald;
multiParticleEnabled = sys.statV.multiParticleEnabled;
for(uint m = 0; m < mols.count; ++m) {
const MoleculeKind& molKind = mols.GetKind(m);
if(molKind.NumAtoms() > maxAtomInMol)
maxAtomInMol = molKind.NumAtoms();
for(uint a = 0; a < molKind.NumAtoms(); ++a) {
particleKind.push_back(molKind.AtomKind(a));
particleMol.push_back(m);
particleCharge.push_back(molKind.AtomCharge(a));
particleIndex.push_back(int(a));
}
}
#ifdef GOMC_CUDA
InitCoordinatesCUDA(forcefield.particles->getCUDAVars(),
currentCoords.Count(), maxAtomInMol, currentCOM.Count());
#endif
}

SystemPotential CalculateEnergy::SystemTotal()
{
GOMC_EVENT_START(1, GomcProfileEvent::EN_SYSTEM_TOTAL);
SystemPotential pot =
SystemInter(SystemPotential(), currentCoords, currentAxes);

for (uint b = 0; b < BOX_TOTAL; ++b) {
GOMC_EVENT_START(1, GomcProfileEvent::EN_BOX_INTRA);
double bondEnergy[2] = {0};
double bondEn = 0.0, nonbondEn = 0.0, correction = 0.0;
MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(b);
MoleculeLookup::box_iterator end = molLookup.BoxEnd(b);
std::vector<uint> molID;

while (thisMol != end) {
molID.push_back(*thisMol);
++thisMol;
}

#ifdef _OPENMP
#pragma omp parallel for default(none) private(bondEnergy) shared(b, molID) \
reduction(+:bondEn, nonbondEn, correction)
#endif
for (int i = 0; i < (int) molID.size(); i++) {
MoleculeIntra(molID[i], b, bondEnergy);
bondEn += bondEnergy[0];
nonbondEn += bondEnergy[1];
correction += calcEwald->MolCorrection(molID[i], b);
}

pot.boxEnergy[b].intraBond = bondEn;
pot.boxEnergy[b].intraNonbond = nonbondEn;
pot.boxEnergy[b].self = calcEwald->BoxSelf(b);
pot.boxEnergy[b].correction = correction;

GOMC_EVENT_STOP(1, GomcProfileEvent::EN_BOX_INTRA);
pot.boxVirial[b] = VirialCalc(b);
}

pot.Total();

if(pot.totalEnergy.total > 1.0e12) {
std::cout << "\nWarning: Large energy detected due to the overlap in "
"initial configuration.\n"
"         The total energy will be recalculated at EqStep to "
"ensure the accuracy \n"
"         of the computed running energies.\n";
}

GOMC_EVENT_STOP(1, GomcProfileEvent::EN_SYSTEM_TOTAL);
return pot;
}


SystemPotential CalculateEnergy::SystemInter(SystemPotential potential,
XYZArray const& coords,
BoxDimensions const& boxAxes)
{
for (uint b = 0; b < BOXES_WITH_U_NB; ++b) {
potential = BoxInter(potential, coords, boxAxes, b);
potential.boxEnergy[b].recip = calcEwald->BoxReciprocal(b, false);
}

potential.Total();

return potential;
}


SystemPotential CalculateEnergy::BoxInter(SystemPotential potential,
XYZArray const& coords,
BoxDimensions const& boxAxes,
const uint box)
{
if (box >= BOXES_WITH_U_NB)
return potential;

GOMC_EVENT_START(1, GomcProfileEvent::EN_BOX_INTER);
double tempREn = 0.0, tempLJEn = 0.0;

std::vector<int> cellVector, cellStartIndex, mapParticleToCell;
std::vector< std::vector<int> > neighborList;
cellList.GetCellListNeighbor(box, currentCoords.Count(),
cellVector, cellStartIndex, mapParticleToCell);
neighborList = cellList.GetNeighborList(box);

#ifdef GOMC_CUDA
UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
boxAxes.cellBasis[box].x, boxAxes.cellBasis[box].y,
boxAxes.cellBasis[box].z);

if(!boxAxes.orthogonal[box]) {
const BoxDimensionsNonOrth *NonOrthAxes = static_cast<const BoxDimensionsNonOrth*>(&boxAxes);
UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
NonOrthAxes->cellBasis_Inv[box].x,
NonOrthAxes->cellBasis_Inv[box].y,
NonOrthAxes->cellBasis_Inv[box].z);
}

CallBoxInterGPU(forcefield.particles->getCUDAVars(), cellVector, cellStartIndex,
neighborList, coords, boxAxes, electrostatic, particleCharge,
particleKind, particleMol, tempREn, tempLJEn, forcefield.sc_coul,
forcefield.sc_sigma_6, forcefield.sc_alpha,
forcefield.sc_power, box);
#else
#ifdef _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(boxAxes, cellStartIndex, \
cellVector, coords, mapParticleToCell, box, neighborList) \
reduction(+:tempREn, tempLJEn)
#else
#pragma omp parallel for default(none) shared(boxAxes, cellStartIndex, \
cellVector, coords, mapParticleToCell, neighborList) \
reduction(+:tempREn, tempLJEn)
#endif
#endif
for(int currParticleIdx = 0; currParticleIdx < (int) cellVector.size(); currParticleIdx++) {
int currParticle = cellVector[currParticleIdx];
int currCell = mapParticleToCell[currParticle];
for(int nCellIndex = 0; nCellIndex < NUMBER_OF_NEIGHBOR_CELL; nCellIndex++) {
int neighborCell = neighborList[currCell][nCellIndex];

int endIndex = cellStartIndex[neighborCell + 1];
for(int nParticleIndex = cellStartIndex[neighborCell];
nParticleIndex < endIndex; nParticleIndex++) {
int nParticle = cellVector[nParticleIndex];

if(currParticle < nParticle && particleMol[currParticle] != particleMol[nParticle]) {
double distSq;
XYZ virComponents;
if(boxAxes.InRcut(distSq, virComponents, coords, currParticle, nParticle, box)) {
double lambdaVDW = GetLambdaVDW(particleMol[currParticle], particleMol[nParticle], box);
if (electrostatic) {
double lambdaCoulomb = GetLambdaCoulomb(particleMol[currParticle],
particleMol[nParticle], box);
double qi_qj_fact = particleCharge[currParticle] *
particleCharge[nParticle] * num::qqFact;
if (qi_qj_fact != 0.0) {
tempREn += forcefield.particles->CalcCoulomb(distSq,
particleKind[currParticle], particleKind[nParticle],
qi_qj_fact, lambdaCoulomb, box);
}
}
tempLJEn += forcefield.particles->CalcEn(distSq,
particleKind[currParticle], particleKind[nParticle], lambdaVDW);
}
}
}
}
}
#endif

potential.boxEnergy[box].inter = tempLJEn;
potential.boxEnergy[box].real = tempREn;

GOMC_EVENT_STOP(1, GomcProfileEvent::EN_BOX_INTER);
if (forcefield.useLRC) {
EnergyCorrection(potential, boxAxes, box);
}

potential.Total();
return potential;
}

SystemPotential CalculateEnergy::BoxForce(SystemPotential potential,
XYZArray const& coords,
XYZArray& atomForce,
XYZArray& molForce,
BoxDimensions const& boxAxes,
const uint box)
{
if (box >= BOXES_WITH_U_NB)
return potential;

GOMC_EVENT_START(1, GomcProfileEvent::EN_BOX_FORCE);

double tempREn = 0.0, tempLJEn = 0.0;
double *aForcex = atomForce.x;
double *aForcey = atomForce.y;
double *aForcez = atomForce.z;
double *mForcex = molForce.x;
double *mForcey = molForce.y;
double *mForcez = molForce.z;
int atomCount = atomForce.Count();
int molCount = molForce.Count();

ResetForce(atomForce, molForce, box);

std::vector<int> cellVector, cellStartIndex, mapParticleToCell;
std::vector<std::vector<int> > neighborList;
cellList.GetCellListNeighbor(box, coords.Count(), cellVector, cellStartIndex, mapParticleToCell);
neighborList = cellList.GetNeighborList(box);

#ifdef GOMC_CUDA
UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
boxAxes.cellBasis[box].x, boxAxes.cellBasis[box].y,
boxAxes.cellBasis[box].z);

if(!boxAxes.orthogonal[box]) {
const BoxDimensionsNonOrth *NonOrthAxes = static_cast<const BoxDimensionsNonOrth*>(&boxAxes);
UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
NonOrthAxes->cellBasis_Inv[box].x,
NonOrthAxes->cellBasis_Inv[box].y,
NonOrthAxes->cellBasis_Inv[box].z);
}

CallBoxForceGPU(forcefield.particles->getCUDAVars(), cellVector,
cellStartIndex, neighborList, mapParticleToCell,
coords, boxAxes, electrostatic, particleCharge,
particleKind, particleMol, tempREn, tempLJEn,
aForcex, aForcey, aForcez, mForcex, mForcey, mForcez,
atomCount, molCount, forcefield.sc_coul,
forcefield.sc_sigma_6, forcefield.sc_alpha,
forcefield.sc_power, box);

#else
#if defined _OPENMP && _OPENMP >= 201511 
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(boxAxes, cellStartIndex, \
cellVector, coords, mapParticleToCell, box, neighborList) \
reduction(+:tempREn, tempLJEn, aForcex[:atomCount], aForcey[:atomCount], \
aForcez[:atomCount], mForcex[:molCount], mForcey[:molCount], mForcez[:molCount])
#else
#pragma omp parallel for default(none) shared(boxAxes, cellStartIndex, \
cellVector, coords, mapParticleToCell, neighborList) \
reduction(+:tempREn, tempLJEn, aForcex[:atomCount], aForcey[:atomCount], \
aForcez[:atomCount], mForcex[:molCount], mForcey[:molCount], mForcez[:molCount])
#endif
#endif
for(int currParticleIdx = 0; currParticleIdx < (int) cellVector.size(); currParticleIdx++) {
int currParticle = cellVector[currParticleIdx];
int currCell = mapParticleToCell[currParticle];

for(int nCellIndex = 0; nCellIndex < NUMBER_OF_NEIGHBOR_CELL; nCellIndex++) {
int neighborCell = neighborList[currCell][nCellIndex];

int endIndex = cellStartIndex[neighborCell + 1];
for(int nParticleIndex = cellStartIndex[neighborCell];
nParticleIndex < endIndex; nParticleIndex++) {
int nParticle = cellVector[nParticleIndex];

if(currParticle < nParticle && particleMol[currParticle] != particleMol[nParticle]) {
double distSq;
XYZ virComponents, forceLJ, forceReal;
if(boxAxes.InRcut(distSq, virComponents, coords, currParticle, nParticle, box)) {
double lambdaVDW = GetLambdaVDW(particleMol[currParticle], particleMol[nParticle], box);
if (electrostatic) {
double lambdaCoulomb = GetLambdaCoulomb(particleMol[currParticle],
particleMol[nParticle], box);
double qi_qj_fact = particleCharge[currParticle] * particleCharge[nParticle] *
num::qqFact;
if (qi_qj_fact != 0.0) {
tempREn += forcefield.particles->CalcCoulomb(distSq, particleKind[currParticle],
particleKind[nParticle], qi_qj_fact, lambdaCoulomb, box);
forceReal = virComponents * forcefield.particles->CalcCoulombVir(distSq,
particleKind[currParticle], particleKind[nParticle], qi_qj_fact, lambdaCoulomb, box);
}
}
tempLJEn += forcefield.particles->CalcEn(distSq, particleKind[currParticle],
particleKind[nParticle], lambdaVDW);
forceLJ = virComponents * forcefield.particles->CalcVir(distSq, particleKind[currParticle],
particleKind[nParticle], lambdaVDW);
aForcex[currParticle] += forceLJ.x + forceReal.x;
aForcey[currParticle] += forceLJ.y + forceReal.y;
aForcez[currParticle] += forceLJ.z + forceReal.z;
aForcex[nParticle] += -(forceLJ.x + forceReal.x);
aForcey[nParticle] += -(forceLJ.y + forceReal.y);
aForcez[nParticle] += -(forceLJ.z + forceReal.z);
mForcex[particleMol[currParticle]] += (forceLJ.x + forceReal.x);
mForcey[particleMol[currParticle]] += (forceLJ.y + forceReal.y);
mForcez[particleMol[currParticle]] += (forceLJ.z + forceReal.z);
mForcex[particleMol[nParticle]] += -(forceLJ.x + forceReal.x);
mForcey[particleMol[nParticle]] += -(forceLJ.y + forceReal.y);
mForcez[particleMol[nParticle]] += -(forceLJ.z + forceReal.z);
}
}
}
}
}
#endif

potential.boxEnergy[box].inter = tempLJEn;
potential.boxEnergy[box].real = tempREn;

GOMC_EVENT_STOP(1, GomcProfileEvent::EN_BOX_FORCE);
return potential;
}


Virial CalculateEnergy::VirialCalc(const uint box)
{
Virial tempVir;
if (box >= BOXES_WITH_U_NB)
return tempVir;

GOMC_EVENT_START(1, GomcProfileEvent::EN_BOX_VIRIAL);

double vT11 = 0.0, vT12 = 0.0, vT13 = 0.0;
double vT22 = 0.0, vT23 = 0.0, vT33 = 0.0;
double rT11 = 0.0, rT12 = 0.0, rT13 = 0.0;
double rT22 = 0.0, rT23 = 0.0, rT33 = 0.0;

std::vector<int> cellVector, cellStartIndex, mapParticleToCell;
std::vector<std::vector<int> > neighborList;
cellList.GetCellListNeighbor(box, currentCoords.Count(), cellVector,
cellStartIndex, mapParticleToCell);
neighborList = cellList.GetNeighborList(box);

#ifdef GOMC_CUDA
UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
currentAxes.cellBasis[box].x,
currentAxes.cellBasis[box].y,
currentAxes.cellBasis[box].z);

if(!currentAxes.orthogonal[box]) {
const BoxDimensionsNonOrth *NonOrthAxes = static_cast<const BoxDimensionsNonOrth*>(&currentAxes);
UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
NonOrthAxes->cellBasis_Inv[box].x,
NonOrthAxes->cellBasis_Inv[box].y,
NonOrthAxes->cellBasis_Inv[box].z);
}

CallBoxInterForceGPU(forcefield.particles->getCUDAVars(),
cellVector, cellStartIndex, neighborList, mapParticleToCell,
currentCoords, currentCOM, currentAxes,
electrostatic, particleCharge, particleKind,
particleMol, rT11, rT12, rT13, rT22, rT23, rT33,
vT11, vT12, vT13, vT22, vT23, vT33,
forcefield.sc_coul,
forcefield.sc_sigma_6, forcefield.sc_alpha,
forcefield.sc_power, box);
#else
#ifdef _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(cellStartIndex, cellVector, \
mapParticleToCell, neighborList, box) \
reduction(+:vT11, vT12, vT13, vT22, vT23, vT33, rT11, rT12, rT13, rT22, rT23, rT33)
#else
#pragma omp parallel for default(none) shared(cellStartIndex, cellVector, \
mapParticleToCell, neighborList) \
reduction(+:vT11, vT12, vT13, vT22, vT23, vT33, rT11, rT12, rT13, rT22, rT23, rT33)
#endif
#endif
for(int currParticleIdx = 0; currParticleIdx < (int) cellVector.size(); currParticleIdx++) {
int currParticle = cellVector[currParticleIdx];
int currCell = mapParticleToCell[currParticle];

for(int nCellIndex = 0; nCellIndex < NUMBER_OF_NEIGHBOR_CELL; nCellIndex++) {
int neighborCell = neighborList[currCell][nCellIndex];

int endIndex = cellStartIndex[neighborCell + 1];
for(int nParticleIndex = cellStartIndex[neighborCell];
nParticleIndex < endIndex; nParticleIndex++) {
int nParticle = cellVector[nParticleIndex];

if(currParticle < nParticle && particleMol[currParticle] != particleMol[nParticle]) {
double distSq;
XYZ virC;
if (currentAxes.InRcut(distSq, virC, currentCoords, currParticle,
nParticle, box)) {

XYZ comC = currentCOM.Difference(particleMol[currParticle], particleMol[nParticle]);
comC = currentAxes.MinImage(comC, box);
double lambdaVDW = GetLambdaVDW(particleMol[currParticle], particleMol[nParticle], box);

if (electrostatic) {
double lambdaCoulomb = GetLambdaCoulomb(particleMol[currParticle],
particleMol[nParticle], box);
double qi_qj = particleCharge[currParticle] * particleCharge[nParticle];

if (qi_qj != 0.0) {
double pRF = forcefield.particles->CalcCoulombVir(distSq, particleKind[currParticle],
particleKind[nParticle], qi_qj, lambdaCoulomb, box);
rT11 += pRF * (virC.x * comC.x);

rT22 += pRF * (virC.y * comC.y);

rT33 += pRF * (virC.z * comC.z);
}
}

double pVF = forcefield.particles->CalcVir(distSq, particleKind[currParticle],
particleKind[nParticle], lambdaVDW);
vT11 += pVF * (virC.x * comC.x);

vT22 += pVF * (virC.y * comC.y);

vT33 += pVF * (virC.z * comC.z);
}
}
}
}
}
#endif

tempVir.interTens[0][0] = vT11;
tempVir.interTens[0][1] = vT12;
tempVir.interTens[0][2] = vT13;

tempVir.interTens[1][0] = vT12;
tempVir.interTens[1][1] = vT22;
tempVir.interTens[1][2] = vT23;

tempVir.interTens[2][0] = vT13;
tempVir.interTens[2][1] = vT23;
tempVir.interTens[2][2] = vT33;

if (electrostatic) {
tempVir.realTens[0][0] = rT11 * num::qqFact;
tempVir.realTens[0][1] = rT12 * num::qqFact;
tempVir.realTens[0][2] = rT13 * num::qqFact;

tempVir.realTens[1][0] = rT12 * num::qqFact;
tempVir.realTens[1][1] = rT22 * num::qqFact;
tempVir.realTens[1][2] = rT23 * num::qqFact;

tempVir.realTens[2][0] = rT13 * num::qqFact;
tempVir.realTens[2][1] = rT23 * num::qqFact;
tempVir.realTens[2][2] = rT33 * num::qqFact;
}

tempVir.inter = vT11 + vT22 + vT33;
tempVir.real = (rT11 + rT22 + rT33) * num::qqFact;

GOMC_EVENT_STOP(1, GomcProfileEvent::EN_BOX_VIRIAL);

if (forcefield.useLRC || forcefield.useIPC) {
VirialCorrection(tempVir, currentAxes, box);
}

tempVir = calcEwald->VirialReciprocal(tempVir, box);

tempVir.Total();
return tempVir;
}


bool CalculateEnergy::MoleculeInter(Intermolecular &inter_LJ,
Intermolecular &inter_coulomb,
XYZArray const& molCoords,
const uint molIndex,
const uint box) const
{
double tempREn = 0.0, tempLJEn = 0.0;
bool overlap = false;

if (box < BOXES_WITH_U_NB) {
GOMC_EVENT_START(1, GomcProfileEvent::EN_MOL_INTER);
uint length = mols.GetKind(molIndex).NumAtoms();
uint start = mols.MolStart(molIndex);

for (uint p = 0; p < length; ++p) {
uint atom = start + p;
CellList::Neighbors n = cellList.EnumerateLocal(currentCoords[atom],
box);

std::vector<uint> nIndex;
while (!n.Done()) {
nIndex.push_back(*n);
n.Next();
}

#ifdef _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(atom, nIndex, box, molIndex) \
reduction(+:tempREn, tempLJEn)
#else
#pragma omp parallel for default(none) shared(atom, nIndex) \
reduction(+:tempREn, tempLJEn)
#endif
#endif
for(int i = 0; i < (int) nIndex.size(); i++) {
double distSq = 0.0;
XYZ virComponents;
if (currentAxes.InRcut(distSq, virComponents, currentCoords, atom,
nIndex[i], box)) {
double lambdaVDW = GetLambdaVDW(molIndex, particleMol[nIndex[i]], box);

if (electrostatic) {
double lambdaCoulomb = GetLambdaCoulomb(molIndex, particleMol[nIndex[i]],
box);
double qi_qj_fact = particleCharge[atom] * particleCharge[nIndex[i]] *
num::qqFact;

if (qi_qj_fact != 0.0) {
tempREn += -forcefield.particles->CalcCoulomb(distSq, particleKind[atom],
particleKind[nIndex[i]], qi_qj_fact, lambdaCoulomb, box);
}
}

tempLJEn += -forcefield.particles->CalcEn(distSq, particleKind[atom],
particleKind[nIndex[i]], lambdaVDW);
}
}

n = cellList.EnumerateLocal(molCoords[p], box);
nIndex.clear();
while (!n.Done()) {
nIndex.push_back(*n);
n.Next();
}

#ifdef _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(atom, molCoords, nIndex, overlap, p, molIndex, box) \
reduction(+:tempREn, tempLJEn)
#else
#pragma omp parallel for default(none) shared(atom, molCoords, nIndex, overlap, p) \
reduction(+:tempREn, tempLJEn)
#endif
#endif
for(int i = 0; i < (int) nIndex.size(); i++) {
double distSq = 0.0;
XYZ virComponents;
if (currentAxes.InRcut(distSq, virComponents, molCoords, p,
currentCoords, nIndex[i], box)) {
double lambdaVDW = GetLambdaVDW(molIndex, particleMol[nIndex[i]], box);

if(distSq < forcefield.rCutLowSq) {
overlap |= true;
}

if (electrostatic) {
double lambdaCoulomb = GetLambdaCoulomb(molIndex, particleMol[nIndex[i]],
box);
double qi_qj_fact = particleCharge[atom] *
particleCharge[nIndex[i]] * num::qqFact;

if (qi_qj_fact != 0.0) {
tempREn += forcefield.particles->CalcCoulomb(distSq,
particleKind[atom], particleKind[nIndex[i]],
qi_qj_fact, lambdaCoulomb, box);
}
}

tempLJEn += forcefield.particles->CalcEn(distSq,
particleKind[atom],
particleKind[nIndex[i]], lambdaVDW);
}
}
}
GOMC_EVENT_STOP(1, GomcProfileEvent::EN_MOL_INTER);
}

inter_LJ.energy = tempLJEn;
inter_coulomb.energy = tempREn;
return overlap;
}

void CalculateEnergy::ParticleNonbonded(double* inter,
cbmc::TrialMol const& trialMol,
XYZArray const& trialPos,
const uint partIndex,
const uint box,
const uint trials) const
{
if (box >= BOXES_WITH_U_B)
return;

GOMC_EVENT_START(1, GomcProfileEvent::EN_CBMC_INTRA_NB);
const MoleculeKind& kind = trialMol.GetKind();
const uint* partner = kind.sortedNB.Begin(partIndex);
const uint* end = kind.sortedNB.End(partIndex);
while (partner != end) {
if (trialMol.AtomExists(*partner)) {
for (uint t = 0; t < trials; ++t) {
double distSq;
if (currentAxes.InRcut(distSq, trialPos, t, trialMol.GetCoords(), *partner, box)) {
inter[t] += forcefield.particles->CalcEn(distSq,
kind.AtomKind(partIndex),
kind.AtomKind(*partner), 1.0);
if (electrostatic) {
double qi_qj_fact = kind.AtomCharge(partIndex) *
kind.AtomCharge(*partner) * num::qqFact;

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(inter[t], distSq,
qi_qj_fact, true);
}
}
}
}
}
++partner;
}
GOMC_EVENT_STOP(1, GomcProfileEvent::EN_CBMC_INTRA_NB);
}

void CalculateEnergy::ParticleInter(double* en, double *real,
XYZArray const& trialPos,
bool* overlap,
const uint partIndex,
const uint molIndex,
const uint box,
const uint trials) const
{
if(box >= BOXES_WITH_U_NB)
return;

GOMC_EVENT_START(1, GomcProfileEvent::EN_CBMC_INTER);
double tempLJ, tempReal;
MoleculeKind const& thisKind = mols.GetKind(molIndex);
uint kindI = thisKind.AtomKind(partIndex);
double kindICharge = thisKind.AtomCharge(partIndex);
std::vector<uint> nIndex;

for(uint t = 0; t < trials; ++t) {
nIndex.clear();
tempReal = 0.0;
tempLJ = 0.0;
CellList::Neighbors n = cellList.EnumerateLocal(trialPos[t], box);
while (!n.Done()) {
nIndex.push_back(*n);
n.Next();
}

#ifdef _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(kindI, kindICharge, nIndex, \
overlap, t, trialPos, box, molIndex) \
reduction(+:tempLJ, tempReal)
#else
#pragma omp parallel for default(none) shared(kindI, kindICharge, nIndex, \
overlap, t, trialPos) \
reduction(+:tempLJ, tempReal)
#endif
#endif
for(int i = 0; i < (int) nIndex.size(); i++) {
double distSq = 0.0;
if(currentAxes.InRcut(distSq, trialPos, t, currentCoords, nIndex[i], box)) {
double lambdaVDW = GetLambdaVDW(molIndex, particleMol[nIndex[i]], box);

if(distSq < forcefield.rCutLowSq) {
overlap[t] |= true;
}
tempLJ += forcefield.particles->CalcEn(distSq, kindI,
particleKind[nIndex[i]],
lambdaVDW);
if(electrostatic) {
double lambdaCoulomb = GetLambdaCoulomb(molIndex, particleMol[nIndex[i]],
box);
double qi_qj_fact = particleCharge[nIndex[i]] * kindICharge * num::qqFact;

if (qi_qj_fact != 0.0) {
tempReal += forcefield.particles->CalcCoulomb(distSq, kindI,
particleKind[nIndex[i]], qi_qj_fact, lambdaCoulomb, box);
}
}
}
}
en[t] += tempLJ;
real[t] += tempReal;
}
GOMC_EVENT_STOP(1, GomcProfileEvent::EN_CBMC_INTER);
}


Intermolecular CalculateEnergy::MoleculeTailChange(const uint box,
const uint kind,
const bool add) const
{
Intermolecular delta;

if (box < BOXES_WITH_U_NB) {

double sign = (add ? 1.0 : -1.0);
uint mkIdxII = kind * mols.GetKindsCount() + kind;
for (uint j = 0; j < mols.GetKindsCount(); ++j) {
uint mkIdxIJ = j * mols.GetKindsCount() + kind;
double rhoDeltaIJ_2 = sign * 2.0 * (double)(molLookup.NumKindInBox(j, box)) *
currentAxes.volInv[box];
delta.energy += mols.pairEnCorrections[mkIdxIJ] * rhoDeltaIJ_2;
}

delta.energy += mols.pairEnCorrections[mkIdxII] *
currentAxes.volInv[box];
}
return delta;
}

Intermolecular CalculateEnergy::MoleculeTailVirChange(const uint box,
const uint kind,
const bool add) const
{
Intermolecular delta;

if (box < BOXES_WITH_U_NB) {

double sign = (add ? 1.0 : -1.0);
uint mkIdxII = kind * mols.GetKindsCount() + kind;
for (uint j = 0; j < mols.GetKindsCount(); ++j) {
uint mkIdxIJ = j * mols.GetKindsCount() + kind;
double rhoDeltaIJ_2 = sign * 2.0 * (double)(molLookup.NumKindInBox(j, box)) *
currentAxes.volInv[box];
delta.virial += mols.pairVirCorrections[mkIdxIJ] * rhoDeltaIJ_2;
}

delta.virial += mols.pairVirCorrections[mkIdxII] *
currentAxes.volInv[box];
}
return delta;
}


void CalculateEnergy::MoleculeIntra(const uint molIndex,
const uint box, double *bondEn) const
{
GOMC_EVENT_START(1, GomcProfileEvent::EN_MOL_INTRA);
bondEn[0] = 0.0, bondEn[1] = 0.0;

MoleculeKind& molKind = mols.kinds[mols.kIndex[molIndex]];
XYZArray bondVec(molKind.bondList.count * 2);

BondVectors(bondVec, molKind, molIndex, box);
MolBond(bondEn[0], molKind, bondVec, molIndex, box);
MolAngle(bondEn[0], molKind, bondVec, box);
MolDihedral(bondEn[0], molKind, bondVec, box);
MolNonbond(bondEn[1], molKind, molIndex, box);
MolNonbond_1_4(bondEn[1], molKind, molIndex, box);
MolNonbond_1_3(bondEn[1], molKind, molIndex, box);
GOMC_EVENT_STOP(1, GomcProfileEvent::EN_MOL_INTRA);
}

Energy CalculateEnergy::MoleculeIntra(cbmc::TrialMol const &mol) const
{
GOMC_EVENT_START(1, GomcProfileEvent::EN_MOL_INTRA);
double bondEn = 0.0, intraNonbondEn = 0.0;
const MoleculeKind& molKind = mol.GetKind();
uint count = molKind.bondList.count;
XYZArray bondVec(count * 2);
std::vector<bool> bondExist(count * 2, false);

BondVectors(bondVec, mol, bondExist, molKind);
MolBond(bondEn, mol, bondVec, bondExist, molKind);
MolAngle(bondEn, mol, bondVec, bondExist, molKind);
MolDihedral(bondEn, mol, bondVec, bondExist, molKind);
MolNonbond(intraNonbondEn, mol, molKind);
MolNonbond_1_4(intraNonbondEn, mol, molKind);
MolNonbond_1_3(intraNonbondEn, mol, molKind);
GOMC_EVENT_STOP(1, GomcProfileEvent::EN_MOL_INTRA);
return Energy(bondEn, intraNonbondEn, 0.0, 0.0, 0.0, 0.0, 0.0);
}

void CalculateEnergy::BondVectors(XYZArray & vecs,
MoleculeKind const& molKind,
const uint molIndex,
const uint box) const
{
for (uint i = 0; i < molKind.bondList.count; ++i) {
uint p1 = mols.start[molIndex] + molKind.bondList.part1[i];
uint p2 = mols.start[molIndex] + molKind.bondList.part2[i];
XYZ dist = currentCoords.Difference(p2, p1);
dist = currentAxes.MinImage(dist, box);

vecs.Set(i, dist);
vecs.Set(i + molKind.bondList.count, -dist.x, -dist.y, -dist.z);
}
}

void CalculateEnergy::BondVectors(XYZArray & vecs,
cbmc::TrialMol const &mol,
std::vector<bool> & bondExist,
MoleculeKind const& molKind) const
{
uint box = mol.GetBox();
uint count = molKind.bondList.count;
for (uint i = 0; i < count; ++i) {
uint p1 = molKind.bondList.part1[i];
uint p2 = molKind.bondList.part2[i];
if(mol.AtomExists(p1) && mol.AtomExists(p2)) {
bondExist[i] = true;
bondExist[i + count] = true;
XYZ dist = mol.GetCoords().Difference(p2, p1);
dist = currentAxes.MinImage(dist, box);
vecs.Set(i, dist);
vecs.Set(i + count, -dist.x, -dist.y, -dist.z);
}
}
}


void CalculateEnergy::MolBond(double & energy,
MoleculeKind const& molKind,
XYZArray const& vecs,
const uint molIndex,
const uint box) const
{
if (box >= BOXES_WITH_U_B)
return;

for (uint b = 0; b < molKind.bondList.count; ++b) {
double molLength = vecs.Get(b).Length();
energy += forcefield.bonds.Calc(molKind.bondList.kinds[b], molLength);

}
}

void CalculateEnergy::MolBond(double & energy,
cbmc::TrialMol const &mol,
XYZArray const& vecs,
std::vector<bool> const & bondExist,
MoleculeKind const& molKind)const
{
if (mol.GetBox() >= BOXES_WITH_U_B)
return;

uint count = molKind.bondList.count;
for (uint b = 0; b < count; ++b) {
if(bondExist[b]) {
energy += forcefield.bonds.Calc(molKind.bondList.kinds[b],
vecs.Get(b).Length());
}
}
}

void CalculateEnergy::MolAngle(double & energy,
MoleculeKind const& molKind,
XYZArray const& vecs,
const uint box) const
{
if (box >= BOXES_WITH_U_B)
return;
for (uint a = 0; a < molKind.angles.Count(); ++a) {
double theta = Theta(vecs.Get(molKind.angles.GetBond(a, 0)),
-vecs.Get(molKind.angles.GetBond(a, 1)));
energy += forcefield.angles->Calc(molKind.angles.GetKind(a), theta);
}
}

void CalculateEnergy::MolAngle(double & energy,
cbmc::TrialMol const &mol,
XYZArray const& vecs,
std::vector<bool> const & bondExist,
MoleculeKind const& molKind) const
{
if (mol.GetBox() >= BOXES_WITH_U_B)
return;

uint count = molKind.angles.Count();
for (uint a = 0; a < count; ++a) {
if(bondExist[molKind.angles.GetBond(a, 0)] &&
bondExist[molKind.angles.GetBond(a, 1)]) {
double theta = Theta(vecs.Get(molKind.angles.GetBond(a, 0)),
-vecs.Get(molKind.angles.GetBond(a, 1)));
energy += forcefield.angles->Calc(molKind.angles.GetKind(a), theta);
}
}
}

void CalculateEnergy::MolDihedral(double & energy,
MoleculeKind const& molKind,
XYZArray const& vecs,
const uint box) const
{
if (box >= BOXES_WITH_U_B)
return;
for (uint d = 0; d < molKind.dihedrals.Count(); ++d) {
double phi = Phi(vecs.Get(molKind.dihedrals.GetBond(d, 0)),
vecs.Get(molKind.dihedrals.GetBond(d, 1)),
vecs.Get(molKind.dihedrals.GetBond(d, 2)));
energy += forcefield.dihedrals.Calc(molKind.dihedrals.GetKind(d), phi);
}
}

void CalculateEnergy::MolDihedral(double & energy,
cbmc::TrialMol const &mol,
XYZArray const& vecs,
std::vector<bool> const & bondExist,
MoleculeKind const& molKind) const
{
if (mol.GetBox() >= BOXES_WITH_U_B)
return;

uint count =  molKind.dihedrals.Count();
for (uint d = 0; d < count; ++d) {
if(bondExist[molKind.dihedrals.GetBond(d, 0)] &&
bondExist[molKind.dihedrals.GetBond(d, 1)] &&
bondExist[molKind.dihedrals.GetBond(d, 2)]) {
double phi = Phi(vecs.Get(molKind.dihedrals.GetBond(d, 0)),
vecs.Get(molKind.dihedrals.GetBond(d, 1)),
vecs.Get(molKind.dihedrals.GetBond(d, 2)));
energy += forcefield.dihedrals.Calc(molKind.dihedrals.GetKind(d), phi);
}
}
}

void CalculateEnergy::MolNonbond(double & energy,
MoleculeKind const& molKind,
const uint molIndex,
const uint box) const
{
if (box >= BOXES_WITH_U_B)
return;

double distSq;
double qi_qj_fact;

for (uint i = 0; i < molKind.nonBonded.count; ++i) {
uint p1 = mols.start[molIndex] + molKind.nonBonded.part1[i];
uint p2 = mols.start[molIndex] + molKind.nonBonded.part2[i];
if (currentAxes.InRcut(distSq, currentCoords, p1, p2, box)) {
energy += forcefield.particles->CalcEn(distSq, molKind.AtomKind
(molKind.nonBonded.part1[i]),
molKind.AtomKind
(molKind.nonBonded.part2[i]), 1.0);
if (electrostatic) {
qi_qj_fact = num::qqFact *
molKind.AtomCharge(molKind.nonBonded.part1[i]) *
molKind.AtomCharge(molKind.nonBonded.part2[i]);

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(energy, distSq,
qi_qj_fact, true);
}
}
}
}

}

void CalculateEnergy::MolNonbond(double & energy, cbmc::TrialMol const &mol,
MoleculeKind const& molKind) const
{
if (mol.GetBox() >= BOXES_WITH_U_B)
return;

double distSq;
double qi_qj_fact;
uint count = molKind.nonBonded.count;

for (uint i = 0; i < count; ++i) {
uint p1 = molKind.nonBonded.part1[i];
uint p2 = molKind.nonBonded.part2[i];
if(mol.AtomExists(p1) && mol.AtomExists(p2)) {
if (currentAxes.InRcut(distSq, mol.GetCoords(), p1, p2, mol.GetBox())) {
energy += forcefield.particles->CalcEn(distSq, molKind.AtomKind(p1),
molKind.AtomKind(p2), 1.0);
if (electrostatic) {
qi_qj_fact = num::qqFact * molKind.AtomCharge(1) *
molKind.AtomCharge(p2);

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(energy, distSq,
qi_qj_fact, true);
}
}
}
}
}

}

void CalculateEnergy::MolNonbond_1_4(double & energy,
MoleculeKind const& molKind,
const uint molIndex,
const uint box) const
{
if (box >= BOXES_WITH_U_B)
return;

double distSq;
double qi_qj_fact;

for (uint i = 0; i < molKind.nonBonded_1_4.count; ++i) {
uint p1 = mols.start[molIndex] + molKind.nonBonded_1_4.part1[i];
uint p2 = mols.start[molIndex] + molKind.nonBonded_1_4.part2[i];
if (currentAxes.InRcut(distSq, currentCoords, p1, p2, box)) {
forcefield.particles->CalcAdd_1_4(energy, distSq,
molKind.AtomKind
(molKind.nonBonded_1_4.part1[i]),
molKind.AtomKind
(molKind.nonBonded_1_4.part2[i]));
if (electrostatic) {
qi_qj_fact = num::qqFact *
molKind.AtomCharge(molKind.nonBonded_1_4.part1[i]) *
molKind.AtomCharge(molKind.nonBonded_1_4.part2[i]);

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(energy, distSq,
qi_qj_fact, false);
}
}
}
}
}

void CalculateEnergy::MolNonbond_1_4(double & energy,
cbmc::TrialMol const &mol,
MoleculeKind const& molKind) const
{
if (mol.GetBox() >= BOXES_WITH_U_B)
return;

double distSq;
double qi_qj_fact;
uint count = molKind.nonBonded_1_4.count;

for (uint i = 0; i < count; ++i) {
uint p1 = molKind.nonBonded_1_4.part1[i];
uint p2 = molKind.nonBonded_1_4.part2[i];
if(mol.AtomExists(p1) && mol.AtomExists(p2)) {
if (currentAxes.InRcut(distSq, mol.GetCoords(), p1, p2, mol.GetBox())) {
forcefield.particles->CalcAdd_1_4(energy, distSq,
molKind.AtomKind(p1),
molKind.AtomKind(p2));
if (electrostatic) {
qi_qj_fact = num::qqFact * molKind.AtomCharge(p1) *
molKind.AtomCharge(p2);

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(energy, distSq,
qi_qj_fact, false);
}
}
}
}
}
}

void CalculateEnergy::MolNonbond_1_3(double & energy,
MoleculeKind const& molKind,
const uint molIndex,
const uint box) const
{
if (box >= BOXES_WITH_U_B)
return;

double distSq;
double qi_qj_fact;

for (uint i = 0; i < molKind.nonBonded_1_3.count; ++i) {
uint p1 = mols.start[molIndex] + molKind.nonBonded_1_3.part1[i];
uint p2 = mols.start[molIndex] + molKind.nonBonded_1_3.part2[i];
if (currentAxes.InRcut(distSq, currentCoords, p1, p2, box)) {
forcefield.particles->CalcAdd_1_4(energy, distSq,
molKind.AtomKind
(molKind.nonBonded_1_3.part1[i]),
molKind.AtomKind
(molKind.nonBonded_1_3.part2[i]));
if (electrostatic) {
qi_qj_fact = num::qqFact *
molKind.AtomCharge(molKind.nonBonded_1_3.part1[i]) *
molKind.AtomCharge(molKind.nonBonded_1_3.part2[i]);

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(energy, distSq,
qi_qj_fact, false);
}
}
}
}
}

void CalculateEnergy::MolNonbond_1_3(double & energy,
cbmc::TrialMol const &mol,
MoleculeKind const& molKind) const
{
if (mol.GetBox() >= BOXES_WITH_U_B)
return;

double distSq;
double qi_qj_fact;
uint count = molKind.nonBonded_1_3.count;

for (uint i = 0; i < count; ++i) {
uint p1 = molKind.nonBonded_1_3.part1[i];
uint p2 = molKind.nonBonded_1_3.part2[i];
if(mol.AtomExists(p1) && mol.AtomExists(p2)) {
if (currentAxes.InRcut(distSq, mol.GetCoords(), p1, p2, mol.GetBox())) {
forcefield.particles->CalcAdd_1_4(energy, distSq,
molKind.AtomKind(p1),
molKind.AtomKind(p2));
if (electrostatic) {
qi_qj_fact = num::qqFact * molKind.AtomCharge(p1) *
molKind.AtomCharge(p2);

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(energy, distSq,
qi_qj_fact, false);
}
}
}
}
}
}

double CalculateEnergy::IntraEnergy_1_3(const double distSq, const uint atom1,
const uint atom2, const uint molIndex) const
{
if(!forcefield.OneThree)
return 0.0;

double eng = 0.0;

MoleculeKind const& thisKind = mols.GetKind(molIndex);
uint kind1 = thisKind.AtomKind(atom1);
uint kind2 = thisKind.AtomKind(atom2);

if (electrostatic) {
double qi_qj_fact =  num::qqFact * thisKind.AtomCharge(atom1) *
thisKind.AtomCharge(atom2);

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(eng, distSq, qi_qj_fact, false);
}
}
forcefield.particles->CalcAdd_1_4(eng, distSq, kind1, kind2);

if(std::isnan(eng))
eng = num::BIGNUM;

return eng;

}

double CalculateEnergy::IntraEnergy_1_4(const double distSq, const uint atom1,
const uint atom2, const uint molIndex) const
{
if(!forcefield.OneFour)
return 0.0;

double eng = 0.0;


MoleculeKind const& thisKind = mols.GetKind(molIndex);
uint kind1 = thisKind.AtomKind(atom1);
uint kind2 = thisKind.AtomKind(atom2);

if (electrostatic) {
double qi_qj_fact =  num::qqFact * thisKind.AtomCharge(atom1) *
thisKind.AtomCharge(atom2);

if (qi_qj_fact != 0.0) {
forcefield.particles->CalcCoulombAdd_1_4(eng, distSq, qi_qj_fact, false);
}
}
forcefield.particles->CalcAdd_1_4(eng, distSq, kind1, kind2);

if(std::isnan(eng))
eng = num::BIGNUM;

return eng;

}

void CalculateEnergy::EnergyCorrection(SystemPotential& pot,
BoxDimensions const& boxAxes,
const uint box) const
{
if(box >= BOXES_WITH_U_NB) {
return;
}

double en = 0.0;
for (uint i = 0; i < mols.GetKindsCount(); ++i) {
uint numI = molLookup.NumKindInBox(i, box);
for (uint j = 0; j < mols.GetKindsCount(); ++j) {
uint numJ = molLookup.NumKindInBox(j, box);
en += mols.pairEnCorrections[i * mols.GetKindsCount() + j] * numI * numJ
* boxAxes.volInv[box];
}
}

if(!forcefield.freeEnergy) {
pot.boxEnergy[box].tailCorrection = en;
}
#if ENSEMBLE == NVT || ENSEMBLE == NPT
else {
uint fk = mols.GetMolKind(lambdaRef.GetMolIndex(box));
double lambdaVDW = lambdaRef.GetLambdaVDW(lambdaRef.GetMolIndex(box), box);
en += MoleculeTailChange(box, fk, false).energy;

for (uint i = 0; i < mols.GetKindsCount(); ++i) {
uint molNum = molLookup.NumKindInBox(i, box);
if(i == fk) {
--molNum; 
}
double rhoDeltaIJ_2 = 2.0 * (double)(molNum) * currentAxes.volInv[box];
en += lambdaVDW * mols.pairEnCorrections[fk * mols.GetKindsCount() + i] *
rhoDeltaIJ_2;
}
en += lambdaVDW * mols.pairEnCorrections[fk * mols.GetKindsCount() + fk] *
currentAxes.volInv[box];
pot.boxEnergy[box].tailCorrection = en;
}
#endif
}

double CalculateEnergy::EnergyCorrection(const uint box,
const uint *kCount) const
{
if (box >= BOXES_WITH_U_NB) {
return 0.0;
}

double tailCorrection = 0.0;
for (uint i = 0; i < mols.kindsCount; ++i) {
for (uint j = 0; j < mols.kindsCount; ++j) {
tailCorrection += mols.pairEnCorrections[i * mols.kindsCount + j] *
kCount[i] * kCount[j] * currentAxes.volInv[box];
}
}
return tailCorrection;
}

void CalculateEnergy::VirialCorrection(Virial& virial,
BoxDimensions const& boxAxes,
const uint box) const
{
if(box >= BOXES_WITH_U_NB) {
return;
}
double vir = 0.0;

for (uint i = 0; i < mols.GetKindsCount(); ++i) {
uint numI = molLookup.NumKindInBox(i, box);
for (uint j = 0; j < mols.GetKindsCount(); ++j) {
uint numJ = molLookup.NumKindInBox(j, box);
vir += mols.pairVirCorrections[i * mols.GetKindsCount() + j] *
numI * numJ * boxAxes.volInv[box];
}
}

if(!forcefield.freeEnergy) {
virial.tailCorrection = vir;
}
#if ENSEMBLE == NVT || ENSEMBLE == NPT
else {
uint fk = mols.GetMolKind(lambdaRef.GetMolIndex(box));
double lambdaVDW = lambdaRef.GetLambdaVDW(lambdaRef.GetMolIndex(box), box);
vir += MoleculeTailVirChange(box, fk, false).virial;

for (uint i = 0; i < mols.GetKindsCount(); ++i) {
uint molNum = molLookup.NumKindInBox(i, box);
if(i == fk) {
--molNum; 
}
double rhoDeltaIJ_2 = 2.0 * (double)(molNum) * currentAxes.volInv[box];
vir += lambdaVDW * mols.pairVirCorrections[fk * mols.GetKindsCount() + i] *
rhoDeltaIJ_2;
}
vir += lambdaVDW * mols.pairVirCorrections[fk * mols.GetKindsCount() + fk] *
currentAxes.volInv[box];
virial.tailCorrection = vir;
}
#endif
}

void CalculateEnergy::CalculateTorque(std::vector<uint>& moleculeIndex,
XYZArray const& coordinates,
XYZArray const& com,
XYZArray const& atomForce,
XYZArray const& atomForceRec,
XYZArray& molTorque,
const uint box)
{
if(multiParticleEnabled && (box < BOXES_WITH_U_NB)) {
GOMC_EVENT_START(1, GomcProfileEvent::BOX_TORQUE);
double *torquex = molTorque.x;
double *torquey = molTorque.y;
double *torquez = molTorque.z;

#if defined _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(atomForce, atomForceRec, com, coordinates,\
moleculeIndex, torquex, torquey, torquez, box)
#else
#pragma omp parallel for default(none) shared(atomForce, atomForceRec, com, coordinates,\
moleculeIndex, torquex, torquey, torquez)
#endif
#endif
for(int m = 0; m < (int) moleculeIndex.size(); m++) {
int mIndex = moleculeIndex[m];
int length = mols.GetKind(mIndex).NumAtoms();
int start = mols.MolStart(mIndex);
double tx = 0.0; double ty = 0.0; double tz = 0.0;
for(int p = start; p < start + length; p++) {
XYZ distFromCOM = coordinates.Difference(p, com, mIndex);
distFromCOM = currentAxes.MinImage(distFromCOM, box);
XYZ tempTorque = Cross(distFromCOM, atomForce[p] + atomForceRec[p]);

tx += tempTorque.x;
ty += tempTorque.y;
tz += tempTorque.z;
}
torquex[mIndex] = tx;
torquey[mIndex] = ty;
torquez[mIndex] = tz;
}
}
GOMC_EVENT_STOP(1, GomcProfileEvent::BOX_TORQUE);
}

void CalculateEnergy::ResetForce(XYZArray& atomForce, XYZArray& molForce,
uint box)
{
if(multiParticleEnabled) {
uint length, start;

MoleculeLookup::box_iterator thisMol = molLookup.BoxBegin(box);
MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);

while(thisMol != end) {
length = mols.GetKind(*thisMol).NumAtoms();
start = mols.MolStart(*thisMol);

molForce.Set(*thisMol, 0.0, 0.0, 0.0);
for(uint p = start; p < start + length; p++) {
atomForce.Set(p, 0.0, 0.0, 0.0);
}
thisMol++;
}
}
}

uint CalculateEnergy::NumberOfParticlesInsideBox(uint box)
{
uint numberOfAtoms = 0;

for(int k = 0; k < (int) mols.GetKindsCount(); k++) {
MoleculeKind const& thisKind = mols.kinds[k];
numberOfAtoms += thisKind.NumAtoms() * molLookup.NumKindInBox(k, box);
}

return numberOfAtoms;
}

bool CalculateEnergy::FindMolInCavity(std::vector< std::vector<uint> > &mol,
const XYZ& center, const XYZ& cavDim,
const XYZArray& invCav, const uint box,
const uint kind, const uint exRatio)
{
uint k;
mol.clear();
mol.resize(molLookup.GetNumKind());
double maxLength = cavDim.Max();

if(maxLength <= currentAxes.rCut[box]) {
CellList::Neighbors n = cellList.EnumerateLocal(center, box);
while (!n.Done()) {
if(currentAxes.InCavity(currentCOM.Get(particleMol[*n]), center, cavDim,
invCav, box)) {
uint molIndex = particleMol[*n];
if(!molLookup.IsNoSwap(molIndex)) {
k = mols.GetMolKind(molIndex);
bool exist = std::find(mol[k].begin(), mol[k].end(), molIndex) !=
mol[k].end();
if(!exist)
mol[k].push_back(molIndex);
}
}
n.Next();
}
} else {
MoleculeLookup::box_iterator n = molLookup.BoxBegin(box);
MoleculeLookup::box_iterator end = molLookup.BoxEnd(box);
while (n != end) {
if(currentAxes.InCavity(currentCOM.Get(*n), center, cavDim, invCav, box)) {
uint molIndex = *n;
if(!molLookup.IsNoSwap(molIndex)) {
k = mols.GetMolKind(molIndex);
bool exist = std::find(mol[k].begin(), mol[k].end(), molIndex) !=
mol[k].end();
if(!exist)
mol[k].push_back(molIndex);
}
}
n++;
}
}

if(mol[kind].size() >= exRatio)
return true;
else
return false;
}

void CalculateEnergy::SingleMoleculeInter(Energy &interEnOld,
Energy &interEnNew,
const double lambdaOldVDW,
const double lambdaNewVDW,
const double lambdaOldCoulomb,
const double lambdaNewCoulomb,
const uint molIndex,
const uint box) const
{
double tempREnOld = 0.0, tempLJEnOld = 0.0;
double tempREnNew = 0.0, tempLJEnNew = 0.0;
if (box < BOXES_WITH_U_NB) {
uint length = mols.GetKind(molIndex).NumAtoms();
uint start = mols.MolStart(molIndex);

for (uint p = 0; p < length; ++p) {
uint atom = start + p;
CellList::Neighbors n = cellList.EnumerateLocal(currentCoords[atom],
box);

std::vector<uint> nIndex;
while (!n.Done()) {
if(particleMol[*n] != (int) molIndex) {
nIndex.push_back(*n);
}
n.Next();
}

#ifdef _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(atom, nIndex, box, \
lambdaNewCoulomb, lambdaOldCoulomb, lambdaOldVDW, lambdaNewVDW) \
reduction(+:tempREnOld, tempLJEnOld, tempREnNew, tempLJEnNew)
#else
#pragma omp parallel for default(none) shared(atom, nIndex) \
reduction(+:tempREnOld, tempLJEnOld, tempREnNew, tempLJEnNew)
#endif
#endif
for(int i = 0; i < (int) nIndex.size(); i++) {
double distSq = 0.0;
XYZ virComponents;
if (currentAxes.InRcut(distSq, virComponents, currentCoords, atom,
nIndex[i], box)) {

if (electrostatic) {
double qi_qj_fact = particleCharge[atom] * particleCharge[nIndex[i]] *
num::qqFact;
if (qi_qj_fact != 0.0) {
tempREnNew += forcefield.particles->CalcCoulomb(distSq, particleKind[atom],
particleKind[nIndex[i]], qi_qj_fact, lambdaNewCoulomb, box);
tempREnOld += forcefield.particles->CalcCoulomb(distSq, particleKind[atom],
particleKind[nIndex[i]], qi_qj_fact, lambdaOldCoulomb, box);
}
}

tempLJEnNew += forcefield.particles->CalcEn(distSq, particleKind[atom],
particleKind[nIndex[i]], lambdaNewVDW);
tempLJEnOld += forcefield.particles->CalcEn(distSq, particleKind[atom],
particleKind[nIndex[i]], lambdaOldVDW);
}
}
}
}

interEnNew.inter = tempLJEnNew;
interEnNew.real = tempREnNew;
interEnOld.inter = tempLJEnOld;
interEnOld.real = tempREnOld;
}

double CalculateEnergy::GetLambdaVDW(uint molA, uint molB, uint box) const
{
double lambda = 1.0;
lambda *= lambdaRef.GetLambdaVDW(molA, box);
lambda *= lambdaRef.GetLambdaVDW(molB, box);
return lambda;
}

double CalculateEnergy::GetLambdaCoulomb(uint molA, uint molB, uint box) const
{
double lambda = 1.0;
lambda *= lambdaRef.GetLambdaCoulomb(molA, box);
lambda *= lambdaRef.GetLambdaCoulomb(molB, box);
return lambda;
}

double CalculateEnergy::MoleculeTailChange(const uint box, const uint kind,
const std::vector <uint> &kCount,
const double lambdaOld,
const double lambdaNew) const
{
if (box >= BOXES_WITH_U_NB) {
return 0.0;
}

double tcDiff = 0.0;
uint ktot = mols.GetKindsCount();
for (uint i = 0; i < ktot; ++i) {
double rhoDeltaIJ_2 = 2.0 * (double)(kCount[i]) * currentAxes.volInv[box];
uint index = kind * ktot + i;
tcDiff += (lambdaNew - lambdaOld) * mols.pairEnCorrections[index] *
rhoDeltaIJ_2;
}
uint index = kind * ktot + kind;
tcDiff += (lambdaNew - lambdaOld) * mols.pairEnCorrections[index] *
currentAxes.volInv[box];

return tcDiff;
}

void CalculateEnergy::EnergyChange(Energy *energyDiff, Energy &dUdL_VDW,
Energy &dUdL_Coul,
const std::vector<double> &lambda_VDW,
const std::vector<double> &lambda_Coul,
const uint iState, const uint molIndex,
const uint box) const
{
if (box >= BOXES_WITH_U_NB) {
return;
}

GOMC_EVENT_START(1, GomcProfileEvent::FREE_ENERGY);
uint length = mols.GetKind(molIndex).NumAtoms();
uint start = mols.MolStart(molIndex);
uint lambdaSize = lambda_VDW.size();
double *tempLJEnDiff = new double [lambdaSize];
double *tempREnDiff = new double [lambdaSize];
double dudl_VDW = 0.0, dudl_Coul = 0.0;
std::fill_n(tempLJEnDiff, lambdaSize, 0.0);
std::fill_n(tempREnDiff, lambdaSize, 0.0);

for (uint p = 0; p < length; ++p) {
uint atom = start + p;
CellList::Neighbors n = cellList.EnumerateLocal(currentCoords[atom], box);

std::vector<uint> nIndex;
while (!n.Done()) {
if(particleMol[*n] != (int) molIndex) {
nIndex.push_back(*n);
}
n.Next();
}

#if defined _OPENMP && _OPENMP >= 201511 
#if GCC_VERSION >= 90000
#pragma omp parallel for default(none) shared(atom, lambda_Coul, lambda_VDW, \
lambdaSize, nIndex, box, iState) \
reduction(+:dudl_VDW, dudl_Coul, tempREnDiff[:lambdaSize], tempLJEnDiff[:lambdaSize])
#else
#pragma omp parallel for default(none) shared(atom, lambda_Coul, lambda_VDW, \
lambdaSize, nIndex) \
reduction(+:dudl_VDW, dudl_Coul, tempREnDiff[:lambdaSize], tempLJEnDiff[:lambdaSize])
#endif
#endif
for(int i = 0; i < (int) nIndex.size(); i++) {
double distSq = 0.0;
XYZ virComponents;
if(currentAxes.InRcut(distSq, virComponents, currentCoords, atom,
nIndex[i], box)) {
double qi_qj_fact = 0.0, energyOldCoul = 0.0;
double energyOldVDW = forcefield.particles->CalcEn(distSq, particleKind[atom],
particleKind[nIndex[i]],
lambda_VDW[iState]);
dudl_VDW += forcefield.particles->CalcdEndL(distSq, particleKind[atom],
particleKind[nIndex[i]], lambda_VDW[iState]);

if(electrostatic) {
qi_qj_fact = particleCharge[atom] * particleCharge[nIndex[i]] *
num::qqFact;
if (qi_qj_fact != 0.0) {
energyOldCoul = forcefield.particles->CalcCoulomb(distSq, particleKind[atom],
particleKind[nIndex[i]], qi_qj_fact,
lambda_Coul[iState], box);
dudl_Coul += forcefield.particles->CalcCoulombdEndL(distSq, particleKind[atom],
particleKind[nIndex[i]], qi_qj_fact,
lambda_Coul[iState], box);
}
}

for(int s = 0; s < (int) lambdaSize; s++) {
tempLJEnDiff[s] += forcefield.particles->CalcEn(distSq, particleKind[atom],
particleKind[nIndex[i]], lambda_VDW[s]);
tempLJEnDiff[s] += -energyOldVDW;
if(electrostatic && qi_qj_fact != 0.0) {
tempREnDiff[s] += forcefield.particles->CalcCoulomb(distSq, particleKind[atom],
particleKind[nIndex[i]], qi_qj_fact, lambda_Coul[s], box);
tempREnDiff[s] += -energyOldCoul;
}
}
}
}
}

dUdL_VDW.inter = dudl_VDW;
dUdL_Coul.real = dudl_Coul;
for(int s = 0; s < (int) lambdaSize; s++) {
energyDiff[s].inter += tempLJEnDiff[s];
energyDiff[s].real += tempREnDiff[s];
}
delete [] tempLJEnDiff;
delete [] tempREnDiff;

if (forcefield.useLRC) {
ChangeLRC(energyDiff, dUdL_VDW, lambda_VDW, iState, molIndex, box);
}
calcEwald->ChangeSelf(energyDiff, dUdL_Coul, lambda_Coul, iState, molIndex,
box);
calcEwald->ChangeCorrection(energyDiff, dUdL_Coul, lambda_Coul, iState,
molIndex, box);
calcEwald->ChangeRecip(energyDiff, dUdL_Coul, lambda_Coul, iState, molIndex,
box);
GOMC_EVENT_STOP(1, GomcProfileEvent::FREE_ENERGY);
}

void CalculateEnergy::ChangeLRC(Energy *energyDiff, Energy &dUdL_VDW,
const std::vector<double> &lambda_VDW,
const uint iState, const uint molIndex,
const uint box) const
{
uint fk = mols.GetMolKind(molIndex);
double lambda_istate = lambda_VDW[iState];

for(size_t s = 0; s < lambda_VDW.size(); s++) {
double lambdaVDW = lambda_VDW[s];
for (uint i = 0; i < mols.GetKindsCount(); ++i) {
uint molNum = molLookup.NumKindInBox(i, box);
if(i == fk) {
--molNum; 
}
double rhoDeltaIJ_2 = 2.0 * (double)(molNum) * currentAxes.volInv[box];
energyDiff[s].tailCorrection += mols.pairEnCorrections[fk * mols.GetKindsCount() + i] *
rhoDeltaIJ_2 * (lambdaVDW - lambda_istate);
if(s == iState) {
dUdL_VDW.tailCorrection += mols.pairEnCorrections[fk * mols.GetKindsCount() + i] *
rhoDeltaIJ_2;
}
}
energyDiff[s].tailCorrection += mols.pairEnCorrections[fk * mols.GetKindsCount() + fk] *
currentAxes.volInv[box] * (lambdaVDW - lambda_istate);
if(s == iState) {
dUdL_VDW.tailCorrection += mols.pairEnCorrections[fk * mols.GetKindsCount() + fk] *
currentAxes.volInv[box];
}
}
}
