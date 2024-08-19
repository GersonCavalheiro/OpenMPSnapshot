
#ifdef _OPENMP
#include <omp.h>
#endif  

#include "DfOverlapX.h"
#include "TlOrbitalInfo.h"
#include "TlOrbitalInfo_Density.h"
#include "TlOrbitalInfo_XC.h"

DfOverlapX::DfOverlapX(TlSerializeData* pPdfParam)
: DfObject(pPdfParam), pEngines_(NULL) {}

DfOverlapX::~DfOverlapX() {}

void DfOverlapX::createEngines() {
assert(this->pEngines_ == NULL);
this->log_.info(TlUtils::format("create threads: %d", this->numOfThreads_));
this->pEngines_ = new DfOverlapEngine[this->numOfThreads_];
}

void DfOverlapX::destroyEngines() {
this->log_.info("delete threads");
if (this->pEngines_ != NULL) {
delete[] this->pEngines_;
this->pEngines_ = NULL;
}
}

DfTaskCtrl* DfOverlapX::getDfTaskCtrlObject() const {
DfTaskCtrl* pDfTaskCtrl = new DfTaskCtrl(this->pPdfParam_);
return pDfTaskCtrl;
}

void DfOverlapX::finalize(TlDenseGeneralMatrix_Lapack* pMtx) {
}

void DfOverlapX::finalize(TlDenseSymmetricMatrix_Lapack* pMtx) {
}

void DfOverlapX::finalize(TlDenseVector_Lapack* pVct) {
}

void DfOverlapX::getSpq(TlDenseSymmetricMatrix_Lapack* pSpq) {
assert(pSpq != NULL);
const index_type numOfAOs = this->m_nNumOfAOs;
pSpq->resize(numOfAOs);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
this->calcOverlap(orbitalInfo, pSpq);
this->finalize(pSpq);
}

void DfOverlapX::getSab(TlDenseSymmetricMatrix_Lapack* pSab) {
assert(pSab != NULL);
const index_type numOfAuxDens = this->m_nNumOfAux;
pSab->resize(numOfAuxDens);

const TlOrbitalInfo_Density orbitalInfo_Density(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_j"]);
this->calcOverlap(orbitalInfo_Density, pSab);
this->finalize(pSab);
}

void DfOverlapX::getSgd(TlDenseSymmetricMatrix_Lapack* pSgd) {
assert(pSgd != NULL);
const index_type numOfAuxXC = this->numOfAuxXC_;
pSgd->resize(numOfAuxXC);

const TlOrbitalInfo_XC orbitalInfo_XC(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_xc"]);
this->calcOverlap(orbitalInfo_XC, pSgd);
this->finalize(pSgd);
}

void DfOverlapX::getNalpha(TlDenseVector_Lapack* pNalpha) {
assert(pNalpha != NULL);
const index_type numOfAuxDens = this->m_nNumOfAux;
pNalpha->resize(numOfAuxDens);

const TlOrbitalInfo_Density orbitalInfo_Density(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_j"]);
this->calcOverlap(orbitalInfo_Density, pNalpha);
this->finalize(pNalpha);
}

void DfOverlapX::getOvpMat(const TlOrbitalInfoObject& orbitalInfo,
TlDenseSymmetricMatrix_Lapack* pS) {
assert(pS != NULL);
pS->resize(orbitalInfo.getNumOfOrbitals());

this->calcOverlap(orbitalInfo, pS);
this->finalize(pS);
}

void DfOverlapX::getTransMat(const TlOrbitalInfoObject& orbitalInfo1,
const TlOrbitalInfoObject& orbitalInfo2,
TlDenseGeneralMatrix_Lapack* pTransMat) {
assert(pTransMat != NULL);
pTransMat->resize(orbitalInfo1.getNumOfOrbitals(),
orbitalInfo2.getNumOfOrbitals());

this->calcOverlap(orbitalInfo1, orbitalInfo2, pTransMat);
this->finalize(pTransMat);
}

void DfOverlapX::get_pqg(const TlDenseVector_Lapack& myu,
TlDenseSymmetricMatrix_Lapack* pF) {
assert(pF != NULL);
const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlOrbitalInfo_XC orbitalInfo_XC(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_xc"]);
pF->resize(orbitalInfo.getNumOfOrbitals());
this->calcOverlap(orbitalInfo_XC, myu, orbitalInfo, pF);
this->finalize(pF);
}

void DfOverlapX::get_pqg(const TlDenseVector_Lapack& myu,
const TlDenseVector_Lapack& eps,
TlDenseSymmetricMatrix_Lapack* pF,
TlDenseSymmetricMatrix_Lapack* pE) {
assert(pF != NULL);
assert(pE != NULL);
const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlOrbitalInfo_XC orbitalInfo_XC(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_xc"]);
pF->resize(orbitalInfo.getNumOfOrbitals());
this->calcOverlap(orbitalInfo_XC, myu, eps, orbitalInfo, pF, pE);
this->finalize(pF);
this->finalize(pE);
}

void DfOverlapX::calcOverlap(const TlOrbitalInfoObject& orbitalInfo,
TlMatrixObject* pMatrix) {
this->createEngines();
DfTaskCtrl* pTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList, true);
while (hasTask == true) {
this->calcOverlap_part(orbitalInfo, taskList, pMatrix);

hasTask = pTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList);
}

pTaskCtrl->cutoffReport();
delete pTaskCtrl;
pTaskCtrl = NULL;
this->destroyEngines();
}

void DfOverlapX::calcOverlap(const TlOrbitalInfoObject& orbitalInfo1,
const TlOrbitalInfoObject& orbitalInfo2,
TlMatrixObject* pMatrix) {
this->createEngines();
DfTaskCtrl* pTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pTaskCtrl->getQueue2(orbitalInfo1, orbitalInfo2, true,
this->grainSize_, &taskList, true);
while (hasTask == true) {
this->calcOverlap_part(orbitalInfo1, orbitalInfo2, taskList, pMatrix);

hasTask = pTaskCtrl->getQueue2(orbitalInfo1, orbitalInfo2, true,
this->grainSize_, &taskList);
}

pTaskCtrl->cutoffReport();
delete pTaskCtrl;
pTaskCtrl = NULL;
this->destroyEngines();
}

void DfOverlapX::calcOverlap(const TlOrbitalInfoObject& orbitalInfo,
TlDenseVectorObject* pVector) {
this->createEngines();
DfTaskCtrl* pTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task> taskList;
bool hasTask =
pTaskCtrl->getQueue(orbitalInfo, this->grainSize_, &taskList, true);
while (hasTask == true) {
this->calcOverlap_part(orbitalInfo, taskList, pVector);

hasTask = pTaskCtrl->getQueue(orbitalInfo, this->grainSize_, &taskList);
}

pTaskCtrl->cutoffReport();
delete pTaskCtrl;
pTaskCtrl = NULL;
this->destroyEngines();
}

void DfOverlapX::calcOverlap(const TlOrbitalInfoObject& orbitalInfo_XC,
const TlDenseVector_Lapack& myu,
const TlOrbitalInfoObject& orbitalInfo,
TlMatrixObject* pF) {
this->createEngines();
DfTaskCtrl* pTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList, true);
while (hasTask == true) {
this->calcOverlap_part(orbitalInfo_XC, myu, orbitalInfo, taskList, pF);

hasTask = pTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList);
}

pTaskCtrl->cutoffReport();
delete pTaskCtrl;
pTaskCtrl = NULL;
this->destroyEngines();
}

void DfOverlapX::calcOverlap(const TlOrbitalInfoObject& orbitalInfo_XC,
const TlDenseVector_Lapack& myu,
const TlDenseVector_Lapack& eps,
const TlOrbitalInfoObject& orbitalInfo,
TlMatrixObject* pF, TlMatrixObject* pE) {
this->createEngines();
DfTaskCtrl* pTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList, true);
while (hasTask == true) {
this->calcOverlap_part(orbitalInfo_XC, myu, eps, orbitalInfo, taskList,
pF, pE);

hasTask = pTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList);
}

pTaskCtrl->cutoffReport();
delete pTaskCtrl;
pTaskCtrl = NULL;
this->destroyEngines();
}

void DfOverlapX::calcOverlap_part(
const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task2>& taskList, TlMatrixObject* pMatrix) {

const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  
assert(threadID < this->numOfThreads_);

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;

const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;

this->pEngines_[threadID].calc(0, orbitalInfo, shellIndexP, 0,
orbitalInfo, shellIndexQ, 0,
orbitalInfo, -1, 0, orbitalInfo, -1);

int index = 0;
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type globalShellIndexP = shellIndexP + stepP;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type globalShellIndexQ = shellIndexQ + stepQ;

if ((shellIndexP != shellIndexQ) ||
(globalShellIndexP >= globalShellIndexQ)) {
pMatrix->add(globalShellIndexP, globalShellIndexQ,
this->pEngines_[threadID].WORK[index]);
}
++index;
}
}
}
}
}

void DfOverlapX::calcOverlap_part(
const TlOrbitalInfoObject& orbitalInfo1,
const TlOrbitalInfoObject& orbitalInfo2,
const std::vector<DfTaskCtrl::Task2>& taskList, TlMatrixObject* pMatrix) {

const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  
assert(threadID < this->numOfThreads_);

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;

const int shellTypeP = orbitalInfo1.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo2.getShellType(shellIndexQ);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;

this->pEngines_[threadID].calc(
0, orbitalInfo1, shellIndexP, 0, orbitalInfo2, shellIndexQ, 0,
orbitalInfo1, -1, 0, orbitalInfo1, -1);

int index = 0;
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type globalShellIndexP = shellIndexP + stepP;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type globalShellIndexQ = shellIndexQ + stepQ;

pMatrix->add(globalShellIndexP, globalShellIndexQ,
this->pEngines_[threadID].WORK[index]);
++index;
}
}
}
}
}

void DfOverlapX::calcOverlap_part(
const TlOrbitalInfoObject& orbitalInfo_XC, const TlDenseVector_Lapack& myu,
const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task2>& taskList, TlMatrixObject* pMatrix) {

const index_type numOfAuxXC = orbitalInfo_XC.getNumOfOrbitals();
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;

const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;

for (index_type shellIndexR = 0; shellIndexR < numOfAuxXC;) {
const int shellTypeR = orbitalInfo_XC.getShellType(shellIndexR);
const int maxStepsR = 2 * shellTypeR + 1;


this->pEngines_[threadID].calc(
0, orbitalInfo, shellIndexP, 0, orbitalInfo, shellIndexQ, 0,
orbitalInfo_XC, shellIndexR, 0, orbitalInfo_XC, -1);

int index = 0;
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type globalShellIndexP = shellIndexP + stepP;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type globalShellIndexQ =
shellIndexQ + stepQ;

for (int stepR = 0; stepR < maxStepsR; ++stepR) {
const index_type globalShellIndexR =
shellIndexR + stepR;
if ((shellIndexP != shellIndexQ) ||
(globalShellIndexP >= globalShellIndexQ)) {
pMatrix->add(
globalShellIndexP, globalShellIndexQ,
myu.get(globalShellIndexR) *
this->pEngines_[threadID].WORK[index]);
}
}
++index;
}
}

shellIndexR += maxStepsR;
}
}
}
}

void DfOverlapX::calcOverlap_part(
const TlOrbitalInfoObject& orbitalInfo_XC, const TlDenseVector_Lapack& myu,
const TlDenseVector_Lapack& eps, const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task2>& taskList, TlMatrixObject* pF,
TlMatrixObject* pE) {

const index_type numOfAuxXC = orbitalInfo_XC.getNumOfOrbitals();
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;

const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;

for (index_type shellIndexR = 0; shellIndexR < numOfAuxXC;) {
const int shellTypeR = orbitalInfo_XC.getShellType(shellIndexR);
const int maxStepsR = 2 * shellTypeR + 1;


this->pEngines_[threadID].calc(
0, orbitalInfo, shellIndexP, 0, orbitalInfo, shellIndexQ, 0,
orbitalInfo_XC, shellIndexR, 0, orbitalInfo_XC, -1);

int index = 0;
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type globalShellIndexP = shellIndexP + stepP;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type globalShellIndexQ =
shellIndexQ + stepQ;

for (int stepR = 0; stepR < maxStepsR; ++stepR) {
const index_type globalShellIndexR =
shellIndexR + stepR;
if ((shellIndexP != shellIndexQ) ||
(globalShellIndexP >= globalShellIndexQ)) {
const double value =
this->pEngines_[threadID].WORK[index];
pF->add(globalShellIndexP, globalShellIndexQ,
myu.get(globalShellIndexR) * value);
pE->add(globalShellIndexP, globalShellIndexQ,
eps.get(globalShellIndexR) * value);
}
}
++index;
}
}

shellIndexR += maxStepsR;
}
}
}
}

void DfOverlapX::calcOverlap_part(const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task>& taskList,
TlDenseVectorObject* pVector) {

const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;

const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int maxStepsP = 2 * shellTypeP + 1;

this->pEngines_[threadID].calc(0, orbitalInfo, shellIndexP, 0,
orbitalInfo, -1, 0, orbitalInfo, -1,
0, orbitalInfo, -1);

int index = 0;
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type globalShellIndexP = shellIndexP + stepP;

pVector->add(globalShellIndexP,
this->pEngines_[threadID].WORK[index]);
++index;
}
}
}
}

void DfOverlapX::getForce(const TlDenseSymmetricMatrix_Lapack& W,
TlDenseGeneralMatrix_Lapack* pForce) {
assert(pForce != NULL);
pForce->resize(this->m_nNumOfAtoms, 3);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const ShellArrayTable shellArrayTable =
this->makeShellArrayTable(orbitalInfo);

this->createEngines();

const int maxShellType = TlOrbitalInfoObject::getMaxShellType();
for (int shellTypeP = maxShellType - 1; shellTypeP >= 0; --shellTypeP) {
const ShellArray shellArrayP = shellArrayTable[shellTypeP];
const index_type shellArraySizeP = shellArrayP.size();

for (int shellTypeQ = maxShellType - 1; shellTypeQ >= 0; --shellTypeQ) {
const ShellArray shellArrayQ = shellArrayTable[shellTypeQ];

for (index_type p = 0; p < shellArraySizeP; ++p) {
const index_type shellIndexP = shellArrayP[p];

this->getForce_partProc(orbitalInfo, shellTypeP, shellTypeQ,
shellIndexP, shellArrayQ, W, pForce);
}
}
}

this->destroyEngines();
}

void DfOverlapX::getForce_partProc(const TlOrbitalInfoObject& orbitalInfo,
const int shellTypeP, const int shellTypeQ,
const index_type shellIndexP,
const ShellArray& shellArrayQ,
const TlDenseSymmetricMatrix_Lapack& W,
TlDenseGeneralMatrix_Lapack* pForce) {

const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;
const std::size_t shellArraySizeQ = shellArrayQ.size();

const int atomIndexA = orbitalInfo.getAtomIndex(shellIndexP);

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

#pragma omp for schedule(runtime)
for (std::size_t q = 0; q < shellArraySizeQ; ++q) {
const index_type shellIndexQ = shellArrayQ[q];

const int atomIndexB = orbitalInfo.getAtomIndex(shellIndexQ);

this->pEngines_[threadID].calc(1, orbitalInfo, shellIndexP, 0,
orbitalInfo, shellIndexQ, 0,
orbitalInfo, -1, 0, orbitalInfo, -1);

int index = 0;
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type indexP = shellIndexP + stepP;
for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type indexQ = shellIndexQ + stepQ;

double coef = W.get(indexP, indexQ);
const double dSdA = this->pEngines_[threadID].WORK[index];
const double dSdB = -dSdA;
#pragma omp critical(DfOverlapX__getForce)
{
pForce->add(atomIndexA, X, coef * dSdA);
pForce->add(atomIndexB, X, coef * dSdB);
}
++index;
}
}
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type indexP = shellIndexP + stepP;
for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type indexQ = shellIndexQ + stepQ;

double coef = W.get(indexP, indexQ);
const double dSdA = this->pEngines_[threadID].WORK[index];
const double dSdB = -dSdA;
#pragma omp critical(DfOverlapX__getForce)
{
pForce->add(atomIndexA, Y, coef * dSdA);
pForce->add(atomIndexB, Y, coef * dSdB);
}
++index;
}
}
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type indexP = shellIndexP + stepP;
for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type indexQ = shellIndexQ + stepQ;

double coef = W.get(indexP, indexQ);
const double dSdA = this->pEngines_[threadID].WORK[index];
const double dSdB = -dSdA;
#pragma omp critical(DfOverlapX__getForce)
{
pForce->add(atomIndexA, Z, coef * dSdA);
pForce->add(atomIndexB, Z, coef * dSdB);
}
++index;
}
}
}
}
}

DfOverlapX::ShellArrayTable DfOverlapX::makeShellArrayTable(
const TlOrbitalInfoObject& orbitalInfo) {
const int maxShellType = TlOrbitalInfoObject::getMaxShellType();
ShellArrayTable shellArrayTable(maxShellType);
const int maxShellIndex = orbitalInfo.getNumOfOrbitals();

int shellIndex = 0;
while (shellIndex < maxShellIndex) {
const int shellType = orbitalInfo.getShellType(shellIndex);
const int steps = 2 * shellType + 1;

shellArrayTable[shellType].push_back(shellIndex);

shellIndex += steps;
}

return shellArrayTable;
}




void DfOverlapX::getGradient(const TlOrbitalInfoObject& orbitalInfo,
TlDenseGeneralMatrix_Lapack* pMatX,
TlDenseGeneralMatrix_Lapack* pMatY,
TlDenseGeneralMatrix_Lapack* pMatZ) {
assert(pMatX != NULL);
assert(pMatY != NULL);
assert(pMatZ != NULL);

const index_type numOfAOs = orbitalInfo.getNumOfOrbitals();
pMatX->resize(numOfAOs, numOfAOs);
pMatY->resize(numOfAOs, numOfAOs);
pMatZ->resize(numOfAOs, numOfAOs);

const ShellArrayTable shellArrayTable =
this->makeShellArrayTable(orbitalInfo);

this->createEngines();
DfTaskCtrl* pTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList, true);
while (hasTask == true) {
this->getGradient_partProc(orbitalInfo, taskList, pMatX, pMatY, pMatZ);

hasTask = pTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList);
}

pTaskCtrl->cutoffReport();
delete pTaskCtrl;
pTaskCtrl = NULL;
this->destroyEngines();

this->finalize(pMatX);
this->finalize(pMatY);
this->finalize(pMatZ);
}

void DfOverlapX::getGradient_partProc(
const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task2>& taskList, TlMatrixObject* pMatX,
TlMatrixObject* pMatY, TlMatrixObject* pMatZ) {

const int taskListSize = taskList.size();
#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;

const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;

this->pEngines_[threadID].calc(1, orbitalInfo, shellIndexP, 0,
orbitalInfo, shellIndexQ, 0,
orbitalInfo, -1, 0, orbitalInfo, -1);

int index = 0;
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type globalShellIndexP = shellIndexP + stepP;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type globalShellIndexQ = shellIndexQ + stepQ;

if ((shellIndexP != shellIndexQ) ||
(globalShellIndexP >= globalShellIndexQ)) {
const double dSdA =
this->pEngines_[threadID].WORK[index];
const double dSdB = -dSdA;

pMatX->add(globalShellIndexP, globalShellIndexQ, dSdA);
pMatX->add(globalShellIndexQ, globalShellIndexP, dSdB);
}
++index;
}
}
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type globalShellIndexP = shellIndexP + stepP;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type globalShellIndexQ = shellIndexQ + stepQ;

if ((shellIndexP != shellIndexQ) ||
(globalShellIndexP >= globalShellIndexQ)) {
const double dSdA =
this->pEngines_[threadID].WORK[index];
const double dSdB = -dSdA;

pMatY->add(globalShellIndexP, globalShellIndexQ, dSdA);
pMatY->add(globalShellIndexQ, globalShellIndexP, dSdB);
}
++index;
}
}
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type globalShellIndexP = shellIndexP + stepP;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type globalShellIndexQ = shellIndexQ + stepQ;

if ((shellIndexP != shellIndexQ) ||
(globalShellIndexP >= globalShellIndexQ)) {
const double dSdA =
this->pEngines_[threadID].WORK[index];
const double dSdB = -dSdA;

pMatZ->add(globalShellIndexP, globalShellIndexQ, dSdA);
pMatZ->add(globalShellIndexQ, globalShellIndexP, dSdB);
}
++index;
}
}
}
}
}

void DfOverlapX::getM(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseSymmetricMatrix_Lapack* pM) {
assert(pM != NULL);
pM->resize(this->m_nNumOfAOs);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);

const TlSparseSymmetricMatrix schwarzTable =
this->makeSchwarzTable(orbitalInfo);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();
pDfTaskCtrl->setCutoffEpsilon_distribution(0.0); 

std::vector<DfTaskCtrl::Task4> taskList;
bool hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, &taskList, true);
while (hasTask == true) {
this->getM_part(orbitalInfo, taskList, P, pM);
hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, &taskList);
}

this->finalize(pM);

delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();
}

TlSparseSymmetricMatrix DfOverlapX::makeSchwarzTable(
const TlOrbitalInfoObject& orbitalInfo) {
this->log_.info("make Schwartz cutoff table: start");
const index_type maxShellIndex = orbitalInfo.getNumOfOrbitals();
TlSparseSymmetricMatrix schwarz(maxShellIndex);

DfOverlapEngine engine;

for (index_type shellIndexP = 0; shellIndexP < maxShellIndex;) {
const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int maxStepsP = 2 * shellTypeP + 1;
const TlPosition posP = orbitalInfo.getPosition(shellIndexP);

for (index_type shellIndexQ = 0; shellIndexQ < maxShellIndex;) {
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);
const int maxStepsQ = 2 * shellTypeQ + 1;

const TlPosition posQ = orbitalInfo.getPosition(shellIndexQ);

engine.calc(0, orbitalInfo, shellIndexP, 0, orbitalInfo,
shellIndexQ, 0, orbitalInfo, shellIndexP, 0,
orbitalInfo, shellIndexQ);

double maxValue = 0.0;
const int maxIndex = maxStepsP * maxStepsQ;
for (int index = 0; index < maxIndex; ++index) {
maxValue = std::max(maxValue, std::fabs(engine.WORK[index]));
}
schwarz.set(shellIndexP, shellIndexQ, std::sqrt(maxValue));

shellIndexQ += maxStepsQ;
}
shellIndexP += maxStepsP;
}

this->log_.info("make Schwartz cutoff table: end");
return schwarz;
}

void DfOverlapX::getM_part(const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task4>& taskList,
const TlMatrixObject& P, TlMatrixObject* pM) {
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  


#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;
const index_type shellIndexR = taskList[i].shellIndex3;
const index_type shellIndexS = taskList[i].shellIndex4;
const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);
const int shellTypeR = orbitalInfo.getShellType(shellIndexR);
const int shellTypeS = orbitalInfo.getShellType(shellIndexS);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;
const int maxStepsR = 2 * shellTypeR + 1;
const int maxStepsS = 2 * shellTypeS + 1;

this->pEngines_[threadID].calc(
0, orbitalInfo, shellIndexP, 0, orbitalInfo, shellIndexQ, 0,
orbitalInfo, shellIndexR, 0, orbitalInfo, shellIndexS);

this->storeM(shellIndexP, maxStepsP, shellIndexQ, maxStepsQ,
shellIndexR, maxStepsR, shellIndexS, maxStepsS,
this->pEngines_[threadID], P, pM);
}
}
}

void DfOverlapX::storeM(const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const index_type shellIndexS, const int maxStepsS,
const DfOverlapEngine& engine, const TlMatrixObject& P,
TlMatrixObject* pM) {
int index = 0;
for (int i = 0; i < maxStepsP; ++i) {
const index_type indexP = shellIndexP + i;

for (int j = 0; j < maxStepsQ; ++j) {
const index_type indexQ = shellIndexQ + j;
const double P_pq = P.get(indexP, indexQ);

for (int k = 0; k < maxStepsR; ++k) {
const index_type indexR = shellIndexR + k;

for (int l = 0; l < maxStepsS; ++l) {
const index_type indexS = shellIndexS + l;
const double P_rs = P.get(indexR, indexS);

const double value = engine.WORK[index];
const index_type maxIndexS =
(indexP == indexR) ? indexQ : indexR;

if ((indexP >= indexQ) && (maxIndexS >= indexS)) {
const double coefEq1 = (indexR != indexS) ? 2.0 : 1.0;
pM->add(indexP, indexQ, coefEq1 * P_rs * value);

if ((shellIndexP != shellIndexR) ||
(shellIndexQ != shellIndexS) ||
(indexP == indexR)) {
if (((indexP + indexQ) != (indexR + indexS)) ||
((indexP * indexQ) != (indexR * indexS))) {

const double coefEq2 =
(indexP != indexQ) ? 2.0 : 1.0;
pM->add(indexR, indexS, coefEq2 * P_pq * value);
}
}
}
++index;
}
}
}
}
}

void DfOverlapX::getM_A(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseSymmetricMatrix_Lapack* pM) {
this->log_.info("DfGridFreeXC::getM_A() in");
assert(pM != NULL);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlOrbitalInfo orbitalInfo_GF(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_gridfree"]);  
pM->resize(orbitalInfo_GF.getNumOfOrbitals());

const TlSparseSymmetricMatrix schwarzTable_PQ =
this->makeSchwarzTable(orbitalInfo_GF);
const TlSparseSymmetricMatrix schwarzTable_RS =
this->makeSchwarzTable(orbitalInfo);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();
pDfTaskCtrl->setCutoffEpsilon_distribution(0.0); 

std::vector<DfTaskCtrl::Task4> taskList;
bool hasTask = pDfTaskCtrl->getQueue4(orbitalInfo_GF, orbitalInfo,
schwarzTable_PQ, schwarzTable_RS,
this->grainSize_, &taskList, true);
while (hasTask == true) {
this->getM_part(orbitalInfo_GF, orbitalInfo, taskList, P, pM);
hasTask = pDfTaskCtrl->getQueue4(orbitalInfo_GF, orbitalInfo,
schwarzTable_PQ, schwarzTable_RS,
this->grainSize_, &taskList);
}

this->finalize(pM);

delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();
}

void DfOverlapX::getM_part(const TlOrbitalInfoObject& orbitalInfo_PQ,
const TlOrbitalInfoObject& orbitalInfo_RS,
const std::vector<DfTaskCtrl::Task4>& taskList,
const TlMatrixObject& P, TlMatrixObject* pM) {
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  


#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;
const index_type shellIndexR = taskList[i].shellIndex3;
const index_type shellIndexS = taskList[i].shellIndex4;
const int shellTypeP = orbitalInfo_PQ.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo_PQ.getShellType(shellIndexQ);
const int shellTypeR = orbitalInfo_RS.getShellType(shellIndexR);
const int shellTypeS = orbitalInfo_RS.getShellType(shellIndexS);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;
const int maxStepsR = 2 * shellTypeR + 1;
const int maxStepsS = 2 * shellTypeS + 1;

this->pEngines_[threadID].calc(
0, orbitalInfo_PQ, shellIndexP, 0, orbitalInfo_PQ, shellIndexQ,
0, orbitalInfo_RS, shellIndexR, 0, orbitalInfo_RS, shellIndexS);

this->storeM_A(shellIndexP, maxStepsP, shellIndexQ, maxStepsQ,
shellIndexR, maxStepsR, shellIndexS, maxStepsS,
this->pEngines_[threadID], P, pM);
}
}
}

void DfOverlapX::storeM_A(const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const index_type shellIndexS, const int maxStepsS,
const DfOverlapEngine& engine,
const TlMatrixObject& P, TlMatrixObject* pM) {
int index = 0;
for (int i = 0; i < maxStepsP; ++i) {
const index_type indexP = shellIndexP + i;

for (int j = 0; j < maxStepsQ; ++j) {
const index_type indexQ = shellIndexQ + j;

if (indexP >= indexQ) {
double value = 0.0;
for (int k = 0; k < maxStepsR; ++k) {
const index_type indexR = shellIndexR + k;

for (int l = 0; l < maxStepsS; ++l) {
const index_type indexS = shellIndexS + l;

if (indexR >= indexS) {
const double coefEq1 =
(indexR != indexS) ? 2.0 : 1.0;
const double P_rs = P.get(indexR, indexS);
value += coefEq1 * P_rs * engine.WORK[index];
}
++index;
}
}
pM->add(indexP, indexQ, value);
} else {
index += maxStepsR * maxStepsS;
}
}
}
}
