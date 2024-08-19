
#ifdef _OPENMP
#include <omp.h>
#endif  

#include "DfEriX.h"
#include "TlFmt.h"
#include "TlMath.h"
#include "TlTime.h"

const int DfEriX::FORCE_K_BUFFER_SIZE = 3 * 7 * 7 * 7;  

DfEriX::DfEriX(TlSerializeData* pPdfParam)
: DfObject(pPdfParam), pEriEngines_(NULL) {
this->lengthScaleParameter_ = 1.0;
if ((*pPdfParam)["length_scale_parameter"].getStr() != "") {
this->lengthScaleParameter_ =
(*pPdfParam)["length_scale_parameter"].getDouble();
}

this->cutoffThreshold_ = 1.0E-10;
if ((*pPdfParam)["cut_value"].getStr().empty() != true) {
this->cutoffThreshold_ = (*pPdfParam)["cut_value"].getDouble();
}

this->cutoffEpsilon_density_ = this->cutoffThreshold_;
if ((*pPdfParam)["cutoff_density"].getStr().empty() != true) {
this->cutoffEpsilon_density_ =
(*pPdfParam)["cutoff_density"].getDouble();
}

this->cutoffEpsilon_distribution_ = this->cutoffThreshold_;
if ((*pPdfParam)["cutoff_distribution"].getStr().empty() != true) {
this->cutoffEpsilon_distribution_ =
(*pPdfParam)["cutoff_distribution"].getDouble();
}



this->cutoffThreshold_primitive_ = this->cutoffThreshold_ * 0.01;
if ((*pPdfParam)["cutoff_threshold_primitive"].getStr().empty() != true) {
this->cutoffThreshold_primitive_ =
(*pPdfParam)["cutoff_threshold_primitive"].getDouble();
}

this->cutoffThreshold_P_grad_ = 1.0E-5;

this->isDebugOutJ_ = false;
if ((*pPdfParam)["debug/eri/output_J"].getStr().empty() != true) {
this->isDebugOutJ_ = (*pPdfParam)["debug/eri/output_J"].getBoolean();
}

this->isDebugOutK_ = false;
if ((*pPdfParam)["debug/eri/output_K"].getStr().empty() != true) {
this->isDebugOutK_ = (*pPdfParam)["debug/eri/output_K"].getBoolean();
}

this->isDebugExactJ_ = false;
if ((*pPdfParam)["debug/eri/exact_J"].getStr().empty() != true) {
this->isDebugExactJ_ = (*pPdfParam)["debug/eri/exact_J"].getBoolean();
}
this->isDebugExactK_ = false;
if ((*pPdfParam)["debug/eri/exact_K"].getStr().empty() != true) {
this->isDebugExactK_ = (*pPdfParam)["debug/eri/exact_K"].getBoolean();
}

}

DfEriX::~DfEriX() {}

void DfEriX::createEngines() {
assert(this->pEriEngines_ == NULL);

static const int maxSizeOfElement =
5 * 5 * 5 * 5 * 4;  
const int numOfThreads = this->numOfThreads_;

this->log_.info(TlUtils::format("create ERI engine: %d", numOfThreads));
this->pEriEngines_ = new DfEriEngine[numOfThreads];

this->pThreadIndexPairs_ = new std::vector<index_type>[numOfThreads];
this->pThreadValues_ = new std::vector<double>[numOfThreads];
for (int i = 0; i < numOfThreads; ++i) {
this->pThreadIndexPairs_[i].resize(
(maxSizeOfElement * this->grainSize_ / numOfThreads + 1) * 2);
this->pThreadValues_[i].resize(
maxSizeOfElement * this->grainSize_ / numOfThreads + 1);
}
}

void DfEriX::destroyEngines() {
const int numOfThreads = this->numOfThreads_;

this->log_.info("delete ERI engine");
delete[] this->pEriEngines_;
this->pEriEngines_ = NULL;

for (int i = 0; i < numOfThreads; ++i) {
this->pThreadIndexPairs_[i].clear();
this->pThreadValues_[i].clear();
}
delete[] this->pThreadIndexPairs_;
this->pThreadIndexPairs_ = NULL;
delete[] this->pThreadValues_;
this->pThreadValues_ = NULL;
}

DfTaskCtrl* DfEriX::getDfTaskCtrlObject() const {
DfTaskCtrl* pDfTaskCtrl = new DfTaskCtrl(this->pPdfParam_);
return pDfTaskCtrl;
}

void DfEriX::finalize(TlDenseGeneralMatrix_Lapack* pMtx) {
}

void DfEriX::finalize(TlDenseSymmetricMatrix_Lapack* pMtx) {
}

void DfEriX::finalize(TlDenseVector_Lapack* pVct) {
}

void DfEriX::getJ(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseVector_Lapack* pRho) {
assert(pRho != NULL);

const double maxDeltaP = P.getMaxAbsoluteElement();
if (maxDeltaP < 1.0) {
this->cutoffThreshold_ /= std::fabs(maxDeltaP);
this->log_.info(TlUtils::format(" new cutoff threshold = % e\n",
this->cutoffThreshold_));
}

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlOrbitalInfo_Density orbitalInfo_Density(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_j"]);
const ShellArrayTable shellArrayTable_Density =
this->makeShellArrayTable(orbitalInfo_Density);

*pRho = TlDenseVector_Lapack(this->m_nNumOfAux);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();
pDfTaskCtrl->setCutoffThreshold(this->cutoffThreshold_);

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pDfTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList, true);
while (hasTask == true) {
this->getJ_part(orbitalInfo, orbitalInfo_Density,
shellArrayTable_Density, taskList, P, pRho);

hasTask = pDfTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList);
}
this->finalize(pRho);

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();

{
}
P.save("dP.mat");
pRho->save("rho.vtr");
}

void DfEriX::getJ_part(const TlOrbitalInfo& orbitalInfo,
const TlOrbitalInfo_Density& orbitalInfo_Density,
const ShellArrayTable& shellArrayTable_Density,
const std::vector<DfTaskCtrl::Task2>& taskList,
const TlMatrixObject& P, TlDenseVector_Lapack* pRho) {
const int taskListSize = taskList.size();

#pragma omp parallel
{

std::vector<double> local_rho(this->m_nNumOfAux);

int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  
this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;
const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);

const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;

const int maxShellType = TlOrbitalInfoObject::getMaxShellType();
for (int shellTypeR = maxShellType - 1; shellTypeR >= 0;
--shellTypeR) {
const int maxStepsR = 2 * shellTypeR + 1;


const std::size_t numOfShellArrayR =
shellArrayTable_Density[shellTypeR].size();
for (std::size_t indexR = 0; indexR < numOfShellArrayR;
++indexR) {
const index_type shellIndexR =
shellArrayTable_Density[shellTypeR][indexR];

this->pEriEngines_[threadID].calc(
0, orbitalInfo, shellIndexP, 0, orbitalInfo,
shellIndexQ, 0, orbitalInfo_Density, shellIndexR, 0,
orbitalInfo_Density, -1);

int index = 0;
for (int i = 0; i < maxStepsP; ++i) {
const index_type indexP = shellIndexP + i;

for (int j = 0; j < maxStepsQ; ++j) {
const index_type indexQ = shellIndexQ + j;

if ((shellIndexP != shellIndexQ) ||
(indexP >= indexQ)) {
const double coef =
(indexP != indexQ) ? 2.0 : 1.0;
const double P_pq =
coef * P.get(indexP, indexQ);

for (int k = 0; k < maxStepsR; ++k) {
const index_type indexR = shellIndexR + k;

const double value =
this->pEriEngines_[threadID]
.WORK[index];
local_rho[indexR] += P_pq * value;
++index;
}
} else {
index += maxStepsR;
}
}
}
}
}
}

#pragma omp critical(DfEriX__getJ_P_to_rho)
{
const int numOfAux = this->m_nNumOfAux;
for (int i = 0; i < numOfAux; ++i) {
pRho->add(i, local_rho[i]);
}
}
#pragma omp barrier
}

}

void DfEriX::getJ(const TlDenseVector_Lapack& rho,
TlDenseSymmetricMatrix_Lapack* pJ) {
assert(pJ != NULL);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlOrbitalInfo_Density orbitalInfo_Density(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_j"]);
const ShellArrayTable shellArrayTable =
this->makeShellArrayTable(orbitalInfo);
const ShellArrayTable shellArrayTable_Density =
this->makeShellArrayTable(orbitalInfo_Density);


pJ->resize(this->m_nNumOfAOs);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pDfTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList, true);
while (hasTask == true) {
this->getJ_part(orbitalInfo, orbitalInfo_Density,
shellArrayTable_Density, taskList, rho, pJ);

hasTask = pDfTaskCtrl->getQueue2(orbitalInfo, true, this->grainSize_,
&taskList);
}

this->finalize(pJ);

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();

{
}
}

void DfEriX::getJ_part(const TlOrbitalInfo& orbitalInfo,
const TlOrbitalInfo_Density& orbitalInfo_Density,
const ShellArrayTable& shellArrayTable_Density,
const std::vector<DfTaskCtrl::Task2>& taskList,
const TlDenseVector_Lapack& rho, TlMatrixObject* pJ) {

const int maxShellType = orbitalInfo.getMaxShellType();
const int taskListSize = taskList.size();

#pragma omp parallel
{

std::vector<index_type> local_indexP;
std::vector<index_type> local_indexQ;
std::vector<double> local_values;
local_indexP.reserve(taskListSize);
local_indexQ.reserve(taskListSize);
local_values.reserve(taskListSize);

int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  
this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;
const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);

const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;

for (int shellTypeR = maxShellType - 1; shellTypeR >= 0;
--shellTypeR) {
const int maxStepsR = 2 * shellTypeR + 1;


const std::size_t numOfShellArrayR =
shellArrayTable_Density[shellTypeR].size();
for (std::size_t indexR = 0; indexR < numOfShellArrayR;
++indexR) {
const index_type shellIndexR =
shellArrayTable_Density[shellTypeR][indexR];

this->pEriEngines_[threadID].calc(
0, orbitalInfo, shellIndexP, 0, orbitalInfo,
shellIndexQ, 0, orbitalInfo_Density, shellIndexR, 0,
orbitalInfo_Density, -1);

int index = 0;
for (int i = 0; i < maxStepsP; ++i) {
const index_type indexP = shellIndexP + i;

for (int j = 0; j < maxStepsQ; ++j) {
const index_type indexQ = shellIndexQ + j;

if ((shellIndexP != shellIndexQ) ||
(indexP >= indexQ)) {
double value = 0.0;
for (int k = 0; k < maxStepsR; ++k) {
const index_type indexR = shellIndexR + k;

value += rho.get(indexR) *
this->pEriEngines_[threadID]
.WORK[index];
++index;
}

local_indexP.push_back(indexP);
local_indexQ.push_back(indexQ);
local_values.push_back(value);
} else {
index += maxStepsR;
}
}
}
}
}
}

#pragma omp critical(DfEriX__getJ_rho_to_J)
{
const int local_size = local_values.size();
assert(local_size == static_cast<int>(local_indexP.size()));
assert(local_size == static_cast<int>(local_indexQ.size()));
for (int i = 0; i < local_size; ++i) {
pJ->add(local_indexP[i], local_indexQ[i], local_values[i]);
}
}
}

}

void DfEriX::getJpq(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseSymmetricMatrix_Lapack* pJ) {
assert(pJ != NULL);

const double maxDeltaP = P.getMaxAbsoluteElement();
if ((maxDeltaP > 0.0) && (maxDeltaP < 1.0)) {
this->cutoffThreshold_ /= maxDeltaP;
this->log_.info(TlUtils::format(" new cutoff threshold = % e\n",
this->cutoffThreshold_));
}

if (this->isDebugExactJ_ == true) {
this->log_.info("calculate J using DEBUG engine.");
this->getJpq_exact(P, pJ);
} else {
this->getJpq_integralDriven(P, pJ);
}

}

void DfEriX::getJpq_exact(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseSymmetricMatrix_Lapack* pJ) {
assert(pJ != NULL);

assert(pJ != NULL);
pJ->resize(this->m_nNumOfAOs);

DfEriEngine engine;
engine.setPrimitiveLevelThreshold(0.0);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const ShellArrayTable shellArrayTable =
this->makeShellArrayTable(orbitalInfo);
const ShellPairArrayTable shellPairArrayTable =
this->getShellPairArrayTable(shellArrayTable);

const int maxShellType = TlOrbitalInfoObject::getMaxShellType();
for (int shellTypeP = maxShellType - 1; shellTypeP >= 0; --shellTypeP) {
const int maxStepsP = 2 * shellTypeP + 1;
const ShellArray shellArrayP = shellArrayTable[shellTypeP];
ShellArray::const_iterator pItEnd = shellArrayP.end();

for (int shellTypeQ = maxShellType - 1; shellTypeQ >= 0; --shellTypeQ) {
const int maxStepsQ = 2 * shellTypeQ + 1;
const ShellArray shellArrayQ = shellArrayTable[shellTypeQ];
ShellArray::const_iterator qItEnd = shellArrayQ.end();


for (int shellTypeR = maxShellType - 1; shellTypeR >= 0;
--shellTypeR) {
const int maxStepsR = 2 * shellTypeR + 1;
const ShellArray shellArrayR = shellArrayTable[shellTypeR];
ShellArray::const_iterator rItEnd = shellArrayR.end();

for (int shellTypeS = maxShellType - 1; shellTypeS >= 0;
--shellTypeS) {
const int maxStepsS = 2 * shellTypeS + 1;
const ShellArray shellArrayS = shellArrayTable[shellTypeS];
ShellArray::const_iterator sItEnd = shellArrayS.end();


for (ShellArray::const_iterator pIt = shellArrayP.begin();
pIt != pItEnd; ++pIt) {
const index_type shellIndexP = *pIt;
for (ShellArray::const_iterator qIt =
shellArrayQ.begin();
qIt != qItEnd; ++qIt) {
const index_type shellIndexQ = *qIt;


for (ShellArray::const_iterator rIt =
shellArrayR.begin();
rIt != rItEnd; ++rIt) {
const index_type shellIndexR = *rIt;
for (ShellArray::const_iterator sIt =
shellArrayS.begin();
sIt != sItEnd; ++sIt) {
const index_type shellIndexS = *sIt;

engine.calc(0, orbitalInfo, shellIndexP, 0,
orbitalInfo, shellIndexQ, 0,
orbitalInfo, shellIndexR, 0,
orbitalInfo, shellIndexS);

int index = 0;
for (int i = 0; i < maxStepsP; ++i) {
const int indexP = shellIndexP + i;
for (int j = 0; j < maxStepsQ; ++j) {
const int indexQ = shellIndexQ + j;

for (int k = 0; k < maxStepsR;
++k) {
const int indexR =
shellIndexR + k;
for (int l = 0; l < maxStepsS;
++l) {
const int indexS =
shellIndexS + l;

if (indexP >= indexQ) {
const double P_rs =
P.get(indexR,
indexS);
const double value =
engine.WORK[index];
pJ->add(indexP, indexQ,
P_rs * value);
}
++index;
}
}
}
}
}
}
}
}
}
}
}
}
}

void DfEriX::getJpq_integralDriven(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseSymmetricMatrix_Lapack* pJ) {
assert(pJ != NULL);
pJ->resize(this->m_nNumOfAOs);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlSparseSymmetricMatrix schwarzTable =
this->makeSchwarzTable(orbitalInfo);

#ifdef DEBUG_J
const index_type numOfAOs = this->m_nNumOfAOs;
this->IA_J_ID1_.resize(numOfAOs);
this->IA_J_ID2_.resize(numOfAOs);
#endif  

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();
pDfTaskCtrl->setCutoffThreshold(this->cutoffThreshold_);
pDfTaskCtrl->setCutoffEpsilon_density(0.0);  
pDfTaskCtrl->setCutoffEpsilon_distribution(
this->cutoffEpsilon_distribution_);

static const int maxElements =
5 * 5 * 5 * 5 * 4;  
index_type* pIndexPairs =
new index_type[maxElements * this->grainSize_ * 2];
double* pValues = new double[maxElements * this->grainSize_];

bool hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, NULL, true);

std::vector<DfTaskCtrl::Task4> taskList;

while (hasTask) {
const int numOfTaskElements = this->getJ_integralDriven_part(
orbitalInfo, taskList, P, pIndexPairs, pValues);
assert(numOfTaskElements <= (maxElements * this->grainSize_));

for (int i = 0; i < numOfTaskElements; ++i) {
const index_type p = pIndexPairs[i * 2];
const index_type q = pIndexPairs[i * 2 + 1];
pJ->add(p, q, pValues[i]);
}

hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, &taskList);
}
this->finalize(pJ);

delete[] pIndexPairs;
delete[] pValues;
pIndexPairs = NULL;
pValues = NULL;

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();

#ifdef DEBUG_J
if (this->isDebugOutJ_ == true) {
for (index_type i = 0; i < numOfAOs; ++i) {
for (index_type j = 0; j <= i; ++j) {
std::cerr << TlUtils::format(">>>>J(%2d,%2d)", i, j)
<< std::endl;
for (index_type k = 0; k < numOfAOs; ++k) {
for (index_type l = 0; l <= k; ++l) {
const int counter1 =
this->IA_J_ID1_.getCount(i, j, k, l);
const int counter2 =
this->IA_J_ID2_.getCount(i, j, k, l);
const int counter = counter1 + counter2;
std::string YN = "  ";
if (counter != ((k == l) ? 1 : 2)) {
YN = "NG";
}
std::cerr
<< TlUtils::format(
"J(%2d,%2d) <= (%2d,%2d) %2d(%2d,%2d) %s", i,
j, k, l, counter, counter1, counter2,
YN.c_str())
<< std::endl;
}
}
std::cerr << std::endl;
}
}
}
#endif  
}

int DfEriX::getJ_integralDriven_part(
const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task4>& taskList, const TlMatrixObject& P,
index_type* pIndexPairs, double* pValues) {
int numOfElements = 0;
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  
this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

const std::size_t maxThreadElements =
taskListSize * 5 * 5 * 5 * 5 * 4;  
index_type* pThreadIndexPairs = new index_type[maxThreadElements * 2];
double* pThreadValues = new double[maxThreadElements];
int numOfThreadElements = 0;

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


this->pEriEngines_[threadID].calc(
0, orbitalInfo, shellIndexP, 0, orbitalInfo, shellIndexQ, 0,
orbitalInfo, shellIndexR, 0, orbitalInfo, shellIndexS);

const int stores = this->storeJ_integralDriven(
shellIndexP, maxStepsP, shellIndexQ, maxStepsQ, shellIndexR,
maxStepsR, shellIndexS, maxStepsS, this->pEriEngines_[threadID],
P, pThreadIndexPairs + numOfThreadElements * 2,
pThreadValues + numOfThreadElements);
numOfThreadElements += stores;
assert(numOfThreadElements < static_cast<int>(maxThreadElements));
}

#pragma omp critical(DfEriX__getJ_integralDriven_part)
{
for (int i = 0; i < numOfThreadElements; ++i) {
pIndexPairs[numOfElements * 2] = pThreadIndexPairs[i * 2];
pIndexPairs[numOfElements * 2 + 1] =
pThreadIndexPairs[i * 2 + 1];
pValues[numOfElements] = pThreadValues[i];
++numOfElements;
}
}

delete[] pThreadIndexPairs;
delete[] pThreadValues;
pThreadIndexPairs = NULL;
pThreadValues = NULL;
}
assert(numOfElements < (this->grainSize_ * 5 * 5 * 5 * 5 * 4));

return numOfElements;
}

int DfEriX::storeJ_integralDriven(
const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const index_type shellIndexS, const int maxStepsS,
const DfEriEngine& engine, const TlMatrixObject& P, index_type* pIndexPairs,
double* pValues) {
assert(pIndexPairs != NULL);
assert(pValues != NULL);

int numOfElements = 0;
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
pIndexPairs[numOfElements * 2] = indexP;
pIndexPairs[numOfElements * 2 + 1] = indexQ;
pValues[numOfElements] = coefEq1 * P_rs * value;
++numOfElements;
#ifdef DEBUG_J
this->IA_J_ID1_.countUp(indexP, indexQ, indexR, indexS,
coefEq1);
#endif  

if ((shellIndexP != shellIndexR) ||
(shellIndexQ != shellIndexS) ||
(indexP == indexR)) {
if (((indexP + indexQ) != (indexR + indexS)) ||
((indexP * indexQ) != (indexR * indexS))) {

const double coefEq2 =
(indexP != indexQ) ? 2.0 : 1.0;
pIndexPairs[numOfElements * 2] = indexR;
pIndexPairs[numOfElements * 2 + 1] = indexS;
pValues[numOfElements] = coefEq2 * P_pq * value;
++numOfElements;
#ifdef DEBUG_J
this->IA_J_ID2_.countUp(indexR, indexS, indexP,
indexQ, coefEq2);
#endif  
}
}
}
++index;
}
}
}
}

return numOfElements;
}

void DfEriX::getJab(TlDenseSymmetricMatrix_Lapack* pJab) {
assert(pJab != NULL);
const index_type numOfAuxDens = this->m_nNumOfAux;
pJab->resize(numOfAuxDens);

const TlOrbitalInfo_Density orbitalInfo_Density(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_j"]);

const ShellArrayTable shellArrayTable =
this->makeShellArrayTable(orbitalInfo_Density);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task2> taskList;
pDfTaskCtrl->setCutoffThreshold(this->cutoffThreshold_);
pDfTaskCtrl->setCutoffEpsilon_density(0.0);  
pDfTaskCtrl->setCutoffEpsilon_distribution(
this->cutoffEpsilon_distribution_);

bool hasTask = pDfTaskCtrl->getQueue2(orbitalInfo_Density, false,
this->grainSize_, &taskList, true);
while (hasTask == true) {
this->getJab_part(orbitalInfo_Density, taskList, pJab);

hasTask = pDfTaskCtrl->getQueue2(orbitalInfo_Density, false,
this->grainSize_, &taskList);
}

this->finalize(pJab);

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();
}

void DfEriX::getJab_part(const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task2>& taskList,
TlMatrixObject* pJab) {
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexR = taskList[i].shellIndex2;
const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeR = orbitalInfo.getShellType(shellIndexR);


const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsR = 2 * shellTypeR + 1;

this->pEriEngines_[threadID].calc(0, orbitalInfo, shellIndexP, 0,
orbitalInfo, -1, 0, orbitalInfo,
shellIndexR, 0, orbitalInfo, -1);

int index = 0;
for (int p = 0; p < maxStepsP; ++p) {
const index_type indexP = shellIndexP + p;

for (int r = 0; r < maxStepsR; ++r) {
const index_type indexR = shellIndexR + r;

if ((shellIndexP != shellIndexR) || (indexP >= indexR)) {
const double value =
this->pEriEngines_[threadID].WORK[index];
pJab->add(indexP, indexR, value);
}
++index;
}
}
}
}
}

void DfEriX::getForceJ(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseGeneralMatrix_Lapack* pForce) {
assert(pForce != NULL);
pForce->resize(this->m_nNumOfAtoms, 3);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlSparseSymmetricMatrix schwarzTable =
this->makeSchwarzTable(orbitalInfo);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task4> taskList;
pDfTaskCtrl->setCutoffThreshold(this->cutoffThreshold_);
pDfTaskCtrl->setCutoffEpsilon_density(0.0);  
pDfTaskCtrl->setCutoffEpsilon_distribution(
this->cutoffEpsilon_distribution_);

bool hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, &taskList, true);
while (hasTask == true) {
this->getForceJ_part(orbitalInfo, taskList, P, pForce);
hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, &taskList);
}

this->finalize(pForce);

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();
}

void DfEriX::getForceJ_part(const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task4>& taskList,
const TlMatrixObject& P, TlMatrixObject* pForce) {
const int taskListSize = taskList.size();
const double pairwisePGTO_cutoffThreshold =
this->cutoffThreshold_primitive_;

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

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
const index_type atomIndexA = orbitalInfo.getAtomIndex(shellIndexP);
const index_type atomIndexB = orbitalInfo.getAtomIndex(shellIndexQ);
const index_type atomIndexC = orbitalInfo.getAtomIndex(shellIndexR);
const index_type atomIndexD = orbitalInfo.getAtomIndex(shellIndexS);

if ((atomIndexA == atomIndexB) && (atomIndexB == atomIndexC) &&
(atomIndexC == atomIndexD) && (atomIndexD == atomIndexA)) {
continue;
}

const DfEriEngine::AngularMomentum2 queryPQ00(0, 0, shellTypeP,
shellTypeQ);
const DfEriEngine::AngularMomentum2 queryRS00(0, 0, shellTypeR,
shellTypeS);
const DfEriEngine::CGTO_Pair PQ =
this->pEriEngines_[threadID].getCGTO_pair(
orbitalInfo, shellIndexP, shellIndexQ,
pairwisePGTO_cutoffThreshold);
const DfEriEngine::CGTO_Pair RS =
this->pEriEngines_[threadID].getCGTO_pair(
orbitalInfo, shellIndexR, shellIndexS,
pairwisePGTO_cutoffThreshold);

this->pEriEngines_[threadID].calcGrad(queryPQ00, queryRS00, PQ, RS);

this->storeForceJ_integralDriven(
atomIndexA, atomIndexB, atomIndexC, atomIndexD, shellIndexP,
maxStepsP, shellIndexQ, maxStepsQ, shellIndexR, maxStepsR,
shellIndexS, maxStepsS, this->pEriEngines_[threadID], P,
pForce);
}
}
}

void DfEriX::storeForceJ_integralDriven(
const int atomIndexA, const int atomIndexB, const int atomIndexC,
const int atomIndexD, const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const index_type shellIndexS, const int maxStepsS,
const DfEriEngine& engine, const TlMatrixObject& P,
TlMatrixObject* pForce) {
int index = 0;
this->storeForceJ_integralDriven(
atomIndexA, atomIndexB, atomIndexC, atomIndexD, shellIndexP, maxStepsP,
shellIndexQ, maxStepsQ, shellIndexR, maxStepsR, shellIndexS, maxStepsS,
engine, P, pForce, X, &index);
this->storeForceJ_integralDriven(
atomIndexA, atomIndexB, atomIndexC, atomIndexD, shellIndexP, maxStepsP,
shellIndexQ, maxStepsQ, shellIndexR, maxStepsR, shellIndexS, maxStepsS,
engine, P, pForce, Y, &index);
this->storeForceJ_integralDriven(
atomIndexA, atomIndexB, atomIndexC, atomIndexD, shellIndexP, maxStepsP,
shellIndexQ, maxStepsQ, shellIndexR, maxStepsR, shellIndexS, maxStepsS,
engine, P, pForce, Z, &index);
}

void DfEriX::storeForceJ_integralDriven(
const int atomIndexA, const int atomIndexB, const int atomIndexC,
const int atomIndexD, const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const index_type shellIndexS, const int maxStepsS,
const DfEriEngine& engine, const TlMatrixObject& P, TlMatrixObject* pForce,
const int target, int* pIndex) {
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type indexP = shellIndexP + stepP;
const index_type iw = indexP * (indexP + 1) / 2;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type indexQ = shellIndexQ + stepQ;
const double Ppq = P.get(indexP, indexQ);

if ((indexP >= indexQ) &&
(std::fabs(Ppq) > this->cutoffThreshold_P_grad_)) {
const index_type ij = iw + indexQ;

for (int stepR = 0; stepR < maxStepsR; ++stepR) {
const index_type indexR = shellIndexR + stepR;
const index_type kw = indexR * (indexR + 1) / 2;

for (int stepS = 0; stepS < maxStepsS; ++stepS) {
const index_type indexS = shellIndexS + stepS;
const double Prs = P.get(indexR, indexS);
const double PpqPrs = Ppq * Prs;

const index_type maxIndexS =
(indexP == indexR) ? indexQ : indexR;
if ((indexS <= maxIndexS) &&
(std::fabs(PpqPrs) >
this->cutoffThreshold_P_grad_)) {
const double vA = engine.WORK_A[*pIndex];
const double vB = engine.WORK_B[*pIndex];
const double vC = engine.WORK_C[*pIndex];
const double vD = engine.WORK_D[*pIndex];

if (indexP >= indexQ) {
double coef = 2.0;  
coef *= (indexR != indexS) ? 2.0 : 1.0;
const double v1 = coef * PpqPrs * vA;
pForce->add(atomIndexA, target, v1);

if (indexP != indexQ) {
const double v2 = coef * PpqPrs * vB;
pForce->add(atomIndexB, target, v2);
}

if ((shellIndexP != shellIndexR) ||
(shellIndexQ != shellIndexS) ||
(indexP == indexR)) {
const index_type kl = kw + indexS;
if (ij != kl) {  
double coef = 2.0;
coef *= (indexP != indexQ) ? 2.0 : 1.0;
{
const double v3 =
coef * PpqPrs * vC;
pForce->add(atomIndexC, target, v3);
if (indexR != indexS) {
const double v4 =
coef * PpqPrs * vD;
pForce->add(atomIndexD, target,
v4);
}
}
}
}
}
}
++(*pIndex);
}
}
} else {
*pIndex += (maxStepsR * maxStepsS);
}
}
}
}

void DfEriX::getForceJ(const TlDenseSymmetricMatrix_Lapack& P,
const TlDenseVector_Lapack& rho,
TlDenseGeneralMatrix_Lapack* pForce) {
assert(pForce != NULL);

const double maxDeltaP = P.getMaxAbsoluteElement();
if (maxDeltaP < 1.0) {
this->cutoffThreshold_ /= std::fabs(maxDeltaP);
this->log_.info(TlUtils::format(" new cutoff threshold = % e",
this->cutoffThreshold_));
}

pForce->resize(this->m_nNumOfAtoms, 3);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlOrbitalInfo_Density orbitalInfo_Density(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_j"]);
const ShellArrayTable shellArrayTable =
this->makeShellArrayTable(orbitalInfo);
const ShellArrayTable shellArrayTable_Density =
this->makeShellArrayTable(orbitalInfo_Density);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();
pDfTaskCtrl->setCutoffThreshold(this->cutoffThreshold_);
pDfTaskCtrl->setCutoffEpsilon_density(0.0);  
pDfTaskCtrl->setCutoffEpsilon_distribution(
this->cutoffEpsilon_distribution_);

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pDfTaskCtrl->getQueue2(orbitalInfo, false, this->grainSize_,
&taskList, true);
while (hasTask == true) {
this->getForceJ_part(orbitalInfo, orbitalInfo_Density,
shellArrayTable_Density, taskList, P, rho, pForce);

hasTask = pDfTaskCtrl->getQueue2(orbitalInfo, false, this->grainSize_,
&taskList);
}

this->finalize(pForce);

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();
}

void DfEriX::getForceJ_part(const TlOrbitalInfoObject& orbitalInfo,
const TlOrbitalInfoObject& orbitalInfo_Density,
const ShellArrayTable& shellArrayTable_Density,
std::vector<DfTaskCtrl::Task2>& taskList,
const TlDenseSymmetricMatrix_Lapack& P,
const TlDenseVector_Lapack& rho,
TlDenseGeneralMatrix_Lapack* pForce) {
static const int BUFFER_SIZE = 3 * 5 * 5 * 5;  
const int maxShellType = orbitalInfo.getMaxShellType();
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

double* p_dJdA = new double[BUFFER_SIZE];
double* p_dJdB = new double[BUFFER_SIZE];

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexQ = taskList[i].shellIndex2;
const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsQ = 2 * shellTypeQ + 1;
const TlPosition posP = orbitalInfo.getPosition(shellIndexP);
const TlPosition posQ = orbitalInfo.getPosition(shellIndexQ);
const index_type atomIndexA = orbitalInfo.getAtomIndex(shellIndexP);
const index_type atomIndexB = orbitalInfo.getAtomIndex(shellIndexQ);

for (int shellTypeR = maxShellType - 1; shellTypeR >= 0;
--shellTypeR) {
const int maxStepsR = 2 * shellTypeR + 1;
const ShellArray& shellArrayR =
shellArrayTable_Density[shellTypeR];
const DfEriEngine::AngularMomentum2 queryRS(0, 0, shellTypeR,
0);

const std::size_t shellArraySizeR = shellArrayR.size();
for (std::size_t r = 0; r < shellArraySizeR; ++r) {
const index_type shellIndexR = shellArrayR[r];
const index_type atomIndexC =
orbitalInfo_Density.getAtomIndex(shellIndexR);

if ((atomIndexA == atomIndexB) &&
(atomIndexB == atomIndexC) &&
(atomIndexC == atomIndexA)) {
continue;
}

this->pEriEngines_[threadID].calc(
1, orbitalInfo, shellIndexP, 0, orbitalInfo,
shellIndexQ, 0, orbitalInfo_Density, shellIndexR, 0,
orbitalInfo_Density, -1);
std::copy(this->pEriEngines_[threadID].WORK,
this->pEriEngines_[threadID].WORK + BUFFER_SIZE,
p_dJdA);

this->pEriEngines_[threadID].calc(
0, orbitalInfo, shellIndexP, 1, orbitalInfo,
shellIndexQ, 0, orbitalInfo_Density, shellIndexR, 0,
orbitalInfo_Density, -1);
std::copy(this->pEriEngines_[threadID].WORK,
this->pEriEngines_[threadID].WORK + BUFFER_SIZE,
p_dJdB);

int index = 0;
this->storeForceJ(atomIndexA, atomIndexB, atomIndexC,
shellIndexP, maxStepsP, shellIndexQ,
maxStepsQ, shellIndexR, maxStepsR, p_dJdA,
p_dJdB, P, rho, pForce, X, &index);
this->storeForceJ(atomIndexA, atomIndexB, atomIndexC,
shellIndexP, maxStepsP, shellIndexQ,
maxStepsQ, shellIndexR, maxStepsR, p_dJdA,
p_dJdB, P, rho, pForce, Y, &index);
this->storeForceJ(atomIndexA, atomIndexB, atomIndexC,
shellIndexP, maxStepsP, shellIndexQ,
maxStepsQ, shellIndexR, maxStepsR, p_dJdA,
p_dJdB, P, rho, pForce, Z, &index);
}
}
}

delete[] p_dJdA;
p_dJdA = NULL;
delete[] p_dJdB;
p_dJdB = NULL;
}
}

void DfEriX::storeForceJ(const index_type atomIndexA,
const index_type atomIndexB,
const index_type atomIndexC,
const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const double* p_dJdA, const double* p_dJdB,
const TlMatrixObject& P,
const TlDenseVectorObject& rho, TlMatrixObject* pForce,
const int target, int* pIndex) {
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type indexP = shellIndexP + stepP;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type indexQ = shellIndexQ + stepQ;

if ((shellIndexP != shellIndexQ) || (indexP >= indexQ)) {
double coef_PQ = P.get(indexP, indexQ);
coef_PQ *= (indexP != indexQ) ? 2.0 : 1.0;

for (int stepR = 0; stepR < maxStepsR; ++stepR) {
const double gradIntA = p_dJdA[*pIndex];
const double gradIntB = p_dJdB[*pIndex];
const double gradIntC = -(gradIntA + gradIntB);
const double coef_PQR =
coef_PQ * rho.get(shellIndexR + stepR);

pForce->add(atomIndexA, target, coef_PQR * gradIntA);
pForce->add(atomIndexB, target, coef_PQR * gradIntB);
pForce->add(atomIndexC, target, coef_PQR * gradIntC);
++(*pIndex);
}
} else {
*pIndex += maxStepsR;
}
}
}
}

void DfEriX::getForceJ(const TlDenseVector_Lapack& rho,
TlDenseGeneralMatrix_Lapack* pForce) {
assert(pForce != NULL);
pForce->resize(this->m_nNumOfAtoms, 3);

const TlOrbitalInfo_Density orbitalInfo_Density(
(*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set_j"]);
const ShellArrayTable shellArray_Density =
this->makeShellArrayTable(orbitalInfo_Density);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();
pDfTaskCtrl->setCutoffThreshold(this->cutoffThreshold_);
pDfTaskCtrl->setCutoffEpsilon_density(0.0);  
pDfTaskCtrl->setCutoffEpsilon_distribution(
this->cutoffEpsilon_distribution_);

std::vector<DfTaskCtrl::Task2> taskList;
bool hasTask = pDfTaskCtrl->getQueue2(orbitalInfo_Density, false,
this->grainSize_, &taskList, true);
while (hasTask == true) {
this->getForceJ_part(orbitalInfo_Density, taskList, rho, pForce);

hasTask = pDfTaskCtrl->getQueue2(orbitalInfo_Density, false,
this->grainSize_, &taskList);
}

this->finalize(pForce);

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();
}

void DfEriX::getForceJ_part(const TlOrbitalInfoObject& orbitalInfo_Density,
std::vector<DfTaskCtrl::Task2>& taskList,
const TlDenseVector_Lapack& rho,
TlDenseGeneralMatrix_Lapack* pForce) {
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

#pragma omp for schedule(runtime)
for (int i = 0; i < taskListSize; ++i) {
const index_type shellIndexP = taskList[i].shellIndex1;
const index_type shellIndexR = taskList[i].shellIndex2;
const index_type atomIndexA =
orbitalInfo_Density.getAtomIndex(shellIndexP);
const index_type atomIndexC =
orbitalInfo_Density.getAtomIndex(shellIndexR);
if (atomIndexA == atomIndexC) {
continue;
}

const int shellTypeP =
orbitalInfo_Density.getShellType(shellIndexP);
const int shellTypeR =
orbitalInfo_Density.getShellType(shellIndexR);
const int maxStepsP = 2 * shellTypeP + 1;
const int maxStepsR = 2 * shellTypeR + 1;

this->pEriEngines_[threadID].calc(
1, orbitalInfo_Density, shellIndexP, 0, orbitalInfo_Density, -1,
0, orbitalInfo_Density, shellIndexR, 0, orbitalInfo_Density,
-1);

int index = 0;
this->storeForceJ(atomIndexA, atomIndexC, shellIndexP, maxStepsP,
shellIndexR, maxStepsR,
this->pEriEngines_[threadID], rho, pForce, X,
&index);
this->storeForceJ(atomIndexA, atomIndexC, shellIndexP, maxStepsP,
shellIndexR, maxStepsR,
this->pEriEngines_[threadID], rho, pForce, Y,
&index);
this->storeForceJ(atomIndexA, atomIndexC, shellIndexP, maxStepsP,
shellIndexR, maxStepsR,
this->pEriEngines_[threadID], rho, pForce, Z,
&index);
}
}
}

void DfEriX::storeForceJ(const index_type atomIndexA,
const index_type atomIndexC,
const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexR, const int maxStepsR,
const DfEriEngine& engine,
const TlDenseVectorObject& rho, TlMatrixObject* pForce,
const int target, int* pIndex) {
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type indexP = shellIndexP + stepP;
const double coef_P = rho.get(indexP);

for (int stepR = 0; stepR < maxStepsR; ++stepR) {
const index_type indexR = shellIndexR + stepR;

if ((shellIndexP != shellIndexR) || (indexP >= indexR)) {
double coef = coef_P * rho.get(indexR);
coef *= (indexP != indexR) ? 2.0 : 1.0;
const double gradA = engine.WORK[*pIndex];
const double gradC = -gradA;

pForce->add(atomIndexA, target, coef * gradA);
pForce->add(atomIndexC, target, coef * gradC);
}
++(*pIndex);
}
}
}

void DfEriX::getK(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseSymmetricMatrix_Lapack* pK) {

const double maxDeltaP = P.getMaxAbsoluteElement();
if ((maxDeltaP > 0.0) && (maxDeltaP < 1.0)) {
this->cutoffThreshold_ /= maxDeltaP;
this->log_.info(TlUtils::format(" new cutoff threshold = % e",
this->cutoffThreshold_));
}

if (this->isDebugExactK_ == true) {
this->log_.info("calculate K using DEBUG engine.");
this->getK_exact(P, pK);
} else {
this->getK_integralDriven(P, pK);
}

}

void DfEriX::getK_exact(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseSymmetricMatrix_Lapack* pK) {
assert(pK != NULL);
pK->resize(this->m_nNumOfAOs);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const ShellArrayTable shellArrayTable =
this->makeShellArrayTable(orbitalInfo);
const ShellPairArrayTable shellPairArrayTable =
this->getShellPairArrayTable(shellArrayTable);

DfEriEngine engine;
engine.setPrimitiveLevelThreshold(0.0);

const int maxShellType = TlOrbitalInfoObject::getMaxShellType();
for (int shellTypeP = maxShellType - 1; shellTypeP >= 0; --shellTypeP) {
const int maxStepsP = 2 * shellTypeP + 1;
const ShellArray shellArrayP = shellArrayTable[shellTypeP];
ShellArray::const_iterator pItEnd = shellArrayP.end();

for (int shellTypeQ = maxShellType - 1; shellTypeQ >= 0; --shellTypeQ) {
const int maxStepsQ = 2 * shellTypeQ + 1;
const ShellArray shellArrayQ = shellArrayTable[shellTypeQ];
ShellArray::const_iterator qItEnd = shellArrayQ.end();


for (int shellTypeR = maxShellType - 1; shellTypeR >= 0;
--shellTypeR) {
const int maxStepsR = 2 * shellTypeR + 1;
const ShellArray shellArrayR = shellArrayTable[shellTypeR];
ShellArray::const_iterator rItEnd = shellArrayR.end();

for (int shellTypeS = maxShellType - 1; shellTypeS >= 0;
--shellTypeS) {
const int maxStepsS = 2 * shellTypeS + 1;
const ShellArray shellArrayS = shellArrayTable[shellTypeS];
ShellArray::const_iterator sItEnd = shellArrayS.end();


for (ShellArray::const_iterator pIt = shellArrayP.begin();
pIt != pItEnd; ++pIt) {
const index_type shellIndexP = *pIt;
for (ShellArray::const_iterator qIt =
shellArrayQ.begin();
qIt != qItEnd; ++qIt) {
const index_type shellIndexQ = *qIt;


for (ShellArray::const_iterator rIt =
shellArrayR.begin();
rIt != rItEnd; ++rIt) {
const index_type shellIndexR = *rIt;
for (ShellArray::const_iterator sIt =
shellArrayS.begin();
sIt != sItEnd; ++sIt) {
const index_type shellIndexS = *sIt;

engine.calc(0, orbitalInfo, shellIndexP, 0,
orbitalInfo, shellIndexQ, 0,
orbitalInfo, shellIndexR, 0,
orbitalInfo, shellIndexS);

int index = 0;
for (int i = 0; i < maxStepsP; ++i) {
const int indexP = shellIndexP + i;
for (int j = 0; j < maxStepsQ; ++j) {
const int indexQ = shellIndexQ + j;

for (int k = 0; k < maxStepsR;
++k) {
const int indexR =
shellIndexR + k;
for (int l = 0; l < maxStepsS;
++l) {
const int indexS =
shellIndexS + l;

if (indexP >= indexR) {
const double P_qs =
P.get(indexQ,
indexS);
const double value =
engine.WORK[index];
pK->add(indexP, indexR,
-1.0 * P_qs *
value);
}
++index;
}
}
}
}
}
}
}
}
}
}
}
}
}

void DfEriX::getK_integralDriven(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseSymmetricMatrix_Lapack* pK) {
assert(pK != NULL);
pK->resize(this->m_nNumOfAOs);

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlSparseSymmetricMatrix schwarzTable =
this->makeSchwarzTable(orbitalInfo);

#ifdef DEBUG_K
this->IA_K_ID1_.resize(numOfAOs);
this->IA_K_ID2_.resize(numOfAOs);
this->IA_K_ID3_.resize(numOfAOs);
this->IA_K_ID4_.resize(numOfAOs);
#endif  

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();
pDfTaskCtrl->setCutoffThreshold(this->cutoffThreshold_);
pDfTaskCtrl->setCutoffEpsilon_density(this->cutoffEpsilon_density_);
pDfTaskCtrl->setCutoffEpsilon_distribution(
this->cutoffEpsilon_distribution_);

static const int maxElements =
5 * 5 * 5 * 5 * 4;  
index_type* pIndexPairs =
new index_type[maxElements * this->grainSize_ * 2];
double* pValues = new double[maxElements * this->grainSize_];

bool hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, NULL, true);
std::vector<DfTaskCtrl::Task4> taskList;

while (hasTask) {
const int numOfTaskElements = this->getK_integralDriven_part(
orbitalInfo, taskList, P, pIndexPairs, pValues);
assert(numOfTaskElements <= (maxElements * this->grainSize_));

for (int i = 0; i < numOfTaskElements; ++i) {
const index_type p = pIndexPairs[i * 2];
const index_type q = pIndexPairs[i * 2 + 1];
pK->add(p, q, pValues[i]);
}

hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, &taskList);
}
this->finalize(pK);

delete[] pIndexPairs;
delete[] pValues;
pIndexPairs = NULL;
pValues = NULL;

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();

#ifdef DEBUG_K
if (this->isDebugOutK_ == true) {
this->debugoutK_integralDriven();
}
#endif  


}

int DfEriX::getK_integralDriven_part(
const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task4>& taskList, const TlMatrixObject& P,
index_type* pIndexPairs, double* pValues) {
int numOfElements = 0;
const int taskListSize = taskList.size();

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  
this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

const std::size_t maxThreadElements =
taskListSize * 5 * 5 * 5 * 5 * 4;  
index_type* pThreadIndexPairs = new index_type[maxThreadElements * 2];
double* pThreadValues = new double[maxThreadElements];
int numOfThreadElements = 0;

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


this->pEriEngines_[threadID].calc(
0, orbitalInfo, shellIndexP, 0, orbitalInfo, shellIndexQ, 0,
orbitalInfo, shellIndexR, 0, orbitalInfo, shellIndexS);

const int stores = this->storeK_integralDriven(
shellIndexP, maxStepsP, shellIndexQ, maxStepsQ, shellIndexR,
maxStepsR, shellIndexS, maxStepsS, this->pEriEngines_[threadID],
P, pThreadIndexPairs + numOfThreadElements * 2,
pThreadValues + numOfThreadElements);
numOfThreadElements += stores;
assert(numOfThreadElements < static_cast<int>(maxThreadElements));
}

#pragma omp critical(DfEriX__getK_integralDriven_part)
{
for (int i = 0; i < numOfThreadElements; ++i) {
pIndexPairs[numOfElements * 2] = pThreadIndexPairs[i * 2];
pIndexPairs[numOfElements * 2 + 1] =
pThreadIndexPairs[i * 2 + 1];
pValues[numOfElements] = pThreadValues[i];
++numOfElements;
}
}

delete[] pThreadIndexPairs;
delete[] pThreadValues;
pThreadIndexPairs = NULL;
pThreadValues = NULL;
}
assert(numOfElements < (this->grainSize_ * 5 * 5 * 5 * 5 * 4));

return numOfElements;
}

int DfEriX::storeK_integralDriven(
const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const index_type shellIndexS, const int maxStepsS,
const DfEriEngine& engine, const TlMatrixObject& P, index_type* pIndexPairs,
double* pValues) {
assert(pIndexPairs != NULL);
assert(pValues != NULL);

int numOfElements = 0;
int index = 0;
for (int i = 0; i < maxStepsP; ++i) {
const index_type indexP = shellIndexP + i;

for (int j = 0; j < maxStepsQ; ++j) {
const index_type indexQ = shellIndexQ + j;
if (indexP < indexQ) {
index += (maxStepsR * maxStepsS);
continue;
}

for (int k = 0; k < maxStepsR; ++k) {
const index_type indexR = shellIndexR + k;
if (shellIndexQ == shellIndexS) {
if (indexP < indexR) {
index += maxStepsS;
continue;
}
}

for (int l = 0; l < maxStepsS; ++l) {
const index_type indexS = shellIndexS + l;
if (indexS > ((indexP == indexR) ? indexQ : indexR)) {
++index;
continue;
}

const double value = engine.WORK[index];

const double coefEq1 =
((indexP == indexR) && (indexQ != indexS)) ? 2.0 : 1.0;
pIndexPairs[numOfElements * 2] = indexP;
pIndexPairs[numOfElements * 2 + 1] = indexR;
pValues[numOfElements] =
-coefEq1 * P.getLocal(indexQ, indexS) * value;
++numOfElements;
#ifdef DEBUG_K
this->IA_K_ID1_.countUp(indexP, indexR, indexQ, indexS,
coefEq1);
#endif  

if (indexR != indexS) {  
const double coefEq2 =
((indexP == indexS) && (indexQ != indexR)) ? 2.0
: 1.0;
pIndexPairs[numOfElements * 2] = indexP;
pIndexPairs[numOfElements * 2 + 1] = indexS;
pValues[numOfElements] =
-coefEq2 * P.getLocal(indexQ, indexR) * value;
++numOfElements;
#ifdef DEBUG_K
this->IA_K_ID2_.countUp(indexP, indexS, indexQ, indexR,
coefEq2);
#endif  
}

if (indexP != indexQ) {  
if ((indexP != indexR) ||
(indexQ !=
indexS)) {  
const double coefEq3 =
((indexQ == indexR) && (indexP != indexS))
? 2.0
: 1.0;
pIndexPairs[numOfElements * 2] = indexQ;
pIndexPairs[numOfElements * 2 + 1] = indexR;
pValues[numOfElements] =
-coefEq3 * P.getLocal(indexP, indexS) * value;
++numOfElements;
#ifdef DEBUG_K
this->IA_K_ID3_.countUp(indexQ, indexR, indexP,
indexS, coefEq3);
#endif  
}

if (indexR != indexS) {  
const double coefEq4 =
((indexQ == indexS) && (indexP != indexR))
? 2.0
: 1.0;
pIndexPairs[numOfElements * 2] = indexQ;
pIndexPairs[numOfElements * 2 + 1] = indexS;
pValues[numOfElements] =
-coefEq4 * P.getLocal(indexP, indexR) * value;
++numOfElements;
#ifdef DEBUG_K
this->IA_K_ID4_.countUp(indexQ, indexS, indexP,
indexR, coefEq4);
#endif  
}
}
++index;
}
}
}
}

return numOfElements;
}

void DfEriX::debugoutK_integralDriven() const {
#ifdef DEBUG_K
const index_type numOfAOs = this->m_nNumOfAOs;
for (index_type i = 0; i < numOfAOs; ++i) {
for (index_type j = 0; j <= i; ++j) {
std::cerr << TlUtils::format(">>>>K(%2d,%2d)", i, j) << std::endl;
for (index_type k = 0; k < numOfAOs; ++k) {
for (index_type l = 0; l <= k; ++l) {
const int counter1 = this->IA_K_ID1_.getCount(i, j, k, l);
const int counter2 = this->IA_K_ID2_.getCount(i, j, k, l);
const int counter3 = this->IA_K_ID3_.getCount(i, j, k, l);
const int counter4 = this->IA_K_ID4_.getCount(i, j, k, l);
const int counter =
counter1 + counter2 + counter3 + counter4;
std::string YN = "  ";
if (counter != ((k == l) ? 1 : 2)) {
YN = "NG";
}
std::cerr << TlUtils::format(
"K(%2d,%2d) <= (%2d,%2d) "
"%2d(%2d,%2d,%2d,%2d) %s",
i, j, k, l, counter, counter1, counter2,
counter3, counter4, YN.c_str())
<< std::endl;
}
}
std::cerr << std::endl;
}
}
#endif  
}

void DfEriX::getForceK(const TlDenseSymmetricMatrix_Lapack& P,
TlDenseGeneralMatrix_Lapack* pForce) {
assert(pForce != NULL);
pForce->resize(this->m_nNumOfAtoms, 3);

const double maxDeltaP = P.getMaxAbsoluteElement();
if (maxDeltaP < 1.0) {
this->cutoffThreshold_ /= std::fabs(maxDeltaP);
this->log_.info(TlUtils::format(" new cutoff threshold = % e",
this->cutoffThreshold_));
}

const TlOrbitalInfo orbitalInfo((*(this->pPdfParam_))["coordinates"],
(*(this->pPdfParam_))["basis_set"]);
const TlSparseSymmetricMatrix schwarzTable =
this->makeSchwarzTable(orbitalInfo);

this->createEngines();
DfTaskCtrl* pDfTaskCtrl = this->getDfTaskCtrlObject();

std::vector<DfTaskCtrl::Task4> taskList;
pDfTaskCtrl->setCutoffThreshold(this->cutoffThreshold_);
pDfTaskCtrl->setCutoffEpsilon_density(this->cutoffEpsilon_density_);
pDfTaskCtrl->setCutoffEpsilon_distribution(
this->cutoffEpsilon_distribution_);

bool hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, &taskList, true);
while (hasTask == true) {
this->getForceK_part(orbitalInfo, taskList, P, pForce);
hasTask = pDfTaskCtrl->getQueue4(orbitalInfo, schwarzTable,
this->grainSize_, &taskList);
}

this->finalize(pForce);

pDfTaskCtrl->cutoffReport();
delete pDfTaskCtrl;
pDfTaskCtrl = NULL;
this->destroyEngines();
}

void DfEriX::getForceK_part(const TlOrbitalInfoObject& orbitalInfo,
const std::vector<DfTaskCtrl::Task4>& taskList,
const TlMatrixObject& P, TlMatrixObject* pForce) {
const int taskListSize = taskList.size();
const double pairwisePGTO_cutoffThreshold =
this->cutoffThreshold_primitive_;

#pragma omp parallel
{
int threadID = 0;
#ifdef _OPENMP
threadID = omp_get_thread_num();
#endif  

this->pEriEngines_[threadID].setPrimitiveLevelThreshold(
this->cutoffThreshold_primitive_);

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
const index_type atomIndexA = orbitalInfo.getAtomIndex(shellIndexP);
const index_type atomIndexB = orbitalInfo.getAtomIndex(shellIndexQ);
const index_type atomIndexC = orbitalInfo.getAtomIndex(shellIndexR);
const index_type atomIndexD = orbitalInfo.getAtomIndex(shellIndexS);

if ((atomIndexA == atomIndexB) && (atomIndexB == atomIndexC) &&
(atomIndexC == atomIndexD) && (atomIndexD == atomIndexA)) {
continue;
}

const DfEriEngine::AngularMomentum2 queryRS00(0, 0, shellTypeR,
shellTypeS);
const DfEriEngine::AngularMomentum2 queryPQ00(0, 0, shellTypeP,
shellTypeQ);
const DfEriEngine::CGTO_Pair PQ =
this->pEriEngines_[threadID].getCGTO_pair(
orbitalInfo, shellIndexP, shellIndexQ,
pairwisePGTO_cutoffThreshold);
const DfEriEngine::CGTO_Pair RS =
this->pEriEngines_[threadID].getCGTO_pair(
orbitalInfo, shellIndexR, shellIndexS,
pairwisePGTO_cutoffThreshold);

this->pEriEngines_[threadID].calcGrad(queryPQ00, queryRS00, PQ, RS);

this->storeForceK_integralDriven(
atomIndexA, atomIndexB, atomIndexC, atomIndexD, shellIndexP,
maxStepsP, shellIndexQ, maxStepsQ, shellIndexR, maxStepsR,
shellIndexS, maxStepsS, this->pEriEngines_[threadID], P,
pForce);
}
}
}

DfEriX::ShellArrayTable DfEriX::makeShellArrayTable(
const TlOrbitalInfoObject& orbitalInfo) {
const int maxShellType = TlOrbitalInfoObject::getMaxShellType();
ShellArrayTable shellArrayTable(maxShellType);
const index_type maxShellIndex = orbitalInfo.getNumOfOrbitals();

index_type shellIndex = 0;
while (shellIndex < maxShellIndex) {
const int shellType = orbitalInfo.getShellType(shellIndex);
const int steps = 2 * shellType + 1;

shellArrayTable[shellType].push_back(shellIndex);

shellIndex += steps;
}

return shellArrayTable;
}

void DfEriX::storeForceK_integralDriven(
const int atomIndexA, const int atomIndexB, const int atomIndexC,
const int atomIndexD, const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const index_type shellIndexS, const int maxStepsS,
const DfEriEngine& engine, const TlMatrixObject& P,
TlMatrixObject* pForce) {
int index = 0;
this->storeForceK_integralDriven(
atomIndexA, atomIndexB, atomIndexC, atomIndexD, shellIndexP, maxStepsP,
shellIndexQ, maxStepsQ, shellIndexR, maxStepsR, shellIndexS, maxStepsS,
engine, P, pForce, X, &index);
this->storeForceK_integralDriven(
atomIndexA, atomIndexB, atomIndexC, atomIndexD, shellIndexP, maxStepsP,
shellIndexQ, maxStepsQ, shellIndexR, maxStepsR, shellIndexS, maxStepsS,
engine, P, pForce, Y, &index);
this->storeForceK_integralDriven(
atomIndexA, atomIndexB, atomIndexC, atomIndexD, shellIndexP, maxStepsP,
shellIndexQ, maxStepsQ, shellIndexR, maxStepsR, shellIndexS, maxStepsS,
engine, P, pForce, Z, &index);
}

void DfEriX::storeForceK_integralDriven(
const int atomIndexA, const int atomIndexB, const int atomIndexC,
const int atomIndexD, const index_type shellIndexP, const int maxStepsP,
const index_type shellIndexQ, const int maxStepsQ,
const index_type shellIndexR, const int maxStepsR,
const index_type shellIndexS, const int maxStepsS,
const DfEriEngine& engine, const TlMatrixObject& P, TlMatrixObject* pForce,
const int target, int* pIndex) {
for (int stepP = 0; stepP < maxStepsP; ++stepP) {
const index_type indexP = shellIndexP + stepP;
const index_type iw = indexP * (indexP + 1) / 2;

for (int stepQ = 0; stepQ < maxStepsQ; ++stepQ) {
const index_type indexQ = shellIndexQ + stepQ;

if (indexP < indexQ) {
*pIndex += (maxStepsR * maxStepsS);
continue;
}

const index_type ij = iw + indexQ;

for (int stepR = 0; stepR < maxStepsR; ++stepR) {
const index_type indexR = shellIndexR + stepR;
if (shellIndexQ == shellIndexS) {
if (indexP < indexR) {
*pIndex += maxStepsS;
continue;
}
}

const index_type kw = indexR * (indexR + 1) / 2;
const double Ppr = P.get(indexP, indexR);
const double Pqr = P.get(indexQ, indexR);

for (int stepS = 0; stepS < maxStepsS; ++stepS) {
const index_type indexS = shellIndexS + stepS;
const index_type kl = kw + indexS;

const int maxIndexS = (indexP == indexR) ? indexQ : indexR;
if (indexS > maxIndexS) {
++(*pIndex);
continue;
}

const double Pps = P.get(indexP, indexS);
const double Pqs = P.get(indexQ, indexS);
const double vA = engine.WORK_A[*pIndex];
const double vB = engine.WORK_B[*pIndex];
const double vC = engine.WORK_C[*pIndex];
const double vD = engine.WORK_D[*pIndex];

{
double coef = 1.0;
coef *= (indexR != indexS) ? 2.0 : 1.0;

pForce->add(atomIndexA, target, coef * Ppr * Pqs * vA);

pForce->add(atomIndexA, target, coef * Pps * Pqr * vA);
}

if (indexP != indexQ) {
double coef = 1.0;
coef *= (indexR != indexS) ? 2.0 : 1.0;

pForce->add(atomIndexB, target, coef * Ppr * Pqs * vB);
pForce->add(atomIndexB, target, coef * Pps * Pqr * vB);
}

if (ij != kl) {
{
double coef = 1.0;
coef *= (indexP != indexQ) ? 2.0 : 1.0;

pForce->add(atomIndexC, target,
coef * Ppr * Pqs * vC);
pForce->add(atomIndexC, target,
coef * Pps * Pqr * vC);
}

if (indexR != indexS) {
double coef = 1.0;
coef *= (indexP != indexQ) ? 2.0 : 1.0;

pForce->add(atomIndexD, target,
coef * Ppr * Pqs * vD);
pForce->add(atomIndexD, target,
coef * Pps * Pqr * vD);
}
}

++(*pIndex);
}
}
}
}
}

DfEriX::ShellPairArrayTable DfEriX::getShellPairArrayTable(
const ShellArrayTable& shellArrayTable) {
const int maxShellType = TlOrbitalInfoObject::getMaxShellType();
ShellPairArrayTable shellPairArrayTable(maxShellType * maxShellType);

for (int shellTypeP = maxShellType - 1; shellTypeP >= 0; --shellTypeP) {
const ShellArray& shellArrayP = shellArrayTable[shellTypeP];
ShellArray::const_iterator pItEnd = shellArrayP.end();

for (int shellTypeR = maxShellType - 1; shellTypeR >= 0; --shellTypeR) {
const ShellArray& shellArrayR = shellArrayTable[shellTypeR];
ShellArray::const_iterator rItEnd = shellArrayR.end();

const int shellPairType_PR = shellTypeP * maxShellType + shellTypeR;
for (ShellArray::const_iterator pIt = shellArrayP.begin();
pIt != pItEnd; ++pIt) {
const index_type indexP = *pIt;

for (ShellArray::const_iterator rIt = shellArrayR.begin();
rIt != rItEnd; ++rIt) {
const index_type indexR = *rIt;

if (indexP >= indexR) {
ShellPair shellPair(indexP, indexR);
shellPairArrayTable[shellPairType_PR].push_back(
shellPair);
}
}
}
}
}

return shellPairArrayTable;
}


























TlSparseSymmetricMatrix DfEriX::makeSchwarzTable(
const TlOrbitalInfoObject& orbitalInfo) {
this->log_.info("make Schwartz cutoff table: start");
const index_type maxShellIndex = orbitalInfo.getNumOfOrbitals();
TlSparseSymmetricMatrix schwarz(maxShellIndex);

DfEriEngine engine;
engine.setPrimitiveLevelThreshold(0.0);

for (index_type shellIndexP = 0; shellIndexP < maxShellIndex;) {
const int shellTypeP = orbitalInfo.getShellType(shellIndexP);
const int maxStepsP = 2 * shellTypeP + 1;

for (index_type shellIndexQ = 0; shellIndexQ < maxShellIndex;) {
const int shellTypeQ = orbitalInfo.getShellType(shellIndexQ);
const int maxStepsQ = 2 * shellTypeQ + 1;

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






