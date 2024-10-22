
#ifdef _OPENMP
#include <omp.h>
#endif  

#include <algorithm>
#include <iostream>

#include "CnError.h"
#include "DfLocalize.h"
#include "TlTime.h"


DfLocalize::DfLocalize(TlSerializeData* pPdfParam)
: DfObject(pPdfParam),
log_(TlLogging::getInstance()),
isRestart_(false),
CMatrixPath_(""),
pairingOrder_(PO_BIG_SMALL),
orbInfo_((*pPdfParam)["coordinates"], (*pPdfParam)["basis_set"]) {
this->maxIteration_ = 100;
if ((*pPdfParam)["lo/max_iteration"].getStr().empty() != true) {
this->maxIteration_ = (*pPdfParam)["lo/max_iteration"].getInt();
}

this->lo_iteration_ = 0;
if ((*pPdfParam_)["lo/num_of_iterations"].getStr().empty() != true) {
this->lo_iteration_ = (*pPdfParam_)["lo/num_of_iterations"].getInt();
}

this->threshold_ = 1.0E-4;
if ((*pPdfParam)["lo/threshold"].getStr().empty() != true) {
this->threshold_ = (*pPdfParam)["lo/threshold"].getDouble();
}

this->numOfOcc_ = (this->m_nNumOfElectrons + 1) / 2;

this->initialize();
}

DfLocalize::~DfLocalize() {
}

void DfLocalize::setRestart(bool yn) {
this->isRestart_ = yn;
}

void DfLocalize::setPairingOrder(DfLocalize::PairingOrder po) {
if (po == UNDEFINED) {
po = PO_BIG_SMALL;
}
this->pairingOrder_ = po;
}

std::string DfLocalize::getPairingOrderStr() const {
std::string answer = "UNDEFINED";
switch (this->pairingOrder_) {
case PO_BIG_BIG:
answer = "BIG-BIG";
break;

case PO_BIG_SMALL:
answer = "BIG-SMALL";
break;

case PO_SMALL_SMALL:
answer = "SMALL-SMALL";
break;

case PO_ORDERED:
answer = "ORDERED";
break;

default:
answer = "UNDEFINED";
break;
}

return answer;
}

void DfLocalize::setCMatrixPath(const std::string& path) {
this->CMatrixPath_ = path;
}

void DfLocalize::setGroup() {

const int numOfAtoms = this->m_nNumOfAtoms;
this->atomGroupList_.resize(numOfAtoms);

for (int atomIndex = 0; atomIndex < numOfAtoms; ++atomIndex) {
this->atomGroupList_[atomIndex] = atomIndex;
}
}

void DfLocalize::setGroup(const TlSerializeData& groupData) {
const int numOfAtoms = this->m_nNumOfAtoms;
this->atomGroupList_.resize(numOfAtoms);

const int numOfGroups = groupData.getSize();
for (int groupId = 0; groupId < numOfGroups; ++groupId) {
const TlSerializeData& group = groupData.getAt(groupId);

const int numOfGroupAtoms = group.getSize();
for (int atomId = 0; atomId < numOfGroupAtoms; ++atomId) {
const int atomIndex = group.getAt(atomId).getInt();
this->atomGroupList_[atomIndex] = groupId;
}
}
}

void DfLocalize::initialize() {
this->checkOpenMP();

this->startOrb_ = 0;
this->endOrb_ = this->numOfOcc_;

this->G_ = 0.0;

this->log_.info(TlUtils::format("number of Atoms: %d", this->m_nNumOfAtoms));
this->log_.info(TlUtils::format("number of AOs: %d", this->m_nNumOfAOs));
this->log_.info(TlUtils::format("number of MOs: %d", this->m_nNumOfMOs));
this->log_.info(TlUtils::format("max iteration: %d", this->maxIteration_));
this->log_.info(TlUtils::format("threshold: %10.5e", this->threshold_));
this->log_.info(TlUtils::format("processing MO: %d -> %d", this->startOrb_, this->endOrb_));

this->log_.info("load S matrix");
const bool isLoaded = this->getSMatrix(&(this->S_));
if (!isLoaded) {
const std::string path = this->getSpqMatrixPath();
this->log_.critical(TlUtils::format("could not load: %s", path.c_str()));
abort();
}
assert(this->S_.getNumOfRows() == this->m_nNumOfAOs);

this->setGroup();
}

void DfLocalize::checkOpenMP() {
#ifdef _OPENMP
{
this->log_.info(">>>> OpenMP info ");
const int numOfOmpThreads = omp_get_max_threads();
this->log_.info(TlUtils::format("OpenMP max threads: %d", numOfOmpThreads));

omp_sched_t kind;
int modifier = 0;
omp_get_schedule(&kind, &modifier);
switch (kind) {
case omp_sched_static:
this->log_.info("OpenMP schedule: static");
break;
case omp_sched_dynamic:
this->log_.info("OpenMP schedule: dynamic");
break;
case omp_sched_guided:
this->log_.info("OpenMP schedule: guided");
break;
case omp_sched_auto:
this->log_.info("OpenMP schedule: auto");
break;
default:
this->log_.info(TlUtils::format("OpenMP schedule: unknown: %d", static_cast<int>(kind)));
break;
}

this->log_.info("<<<<");
}
#else
{ this->log_.info("OpenMP is not enabled."); }
#endif  
}

bool DfLocalize::getSMatrix(TlDenseSymmetricMatrix_Lapack* pS) {
const std::string path = this->getSpqMatrixPath();
const bool isLoaded = pS->load(path);

return isLoaded;
}

std::string DfLocalize::getCMatrixPath() {
std::string path = "";
if (this->CMatrixPath_.empty()) {
if (this->isRestart_) {
path = DfObject::getCloMatrixPath(DfObject::RUN_RKS, this->lo_iteration_);
} else {
const int scf_iteration = (*(this->pPdfParam_))["num_of_iterations"].getInt();
path = DfObject::getCMatrixPath(DfObject::RUN_RKS, scf_iteration);
this->lo_iteration_ = 0;
}
} else {
path = this->CMatrixPath_;
}

return path;
}

void DfLocalize::getCMatrix(TlDenseGeneralMatrix_Lapack* pC) {
const std::string path = this->getCMatrixPath();

this->log_.info(TlUtils::format("load C matrix: %s", path.c_str()));
pC->load(path);
}

void DfLocalize::exec() {
this->log_.info(TlUtils::format("pairing order: %s", this->getPairingOrderStr().c_str()));

TlDenseGeneralMatrix_Lapack C;
{
TlDenseGeneralMatrix_Lapack C_full;
this->getCMatrix(&C_full);
assert(this->startOrb_ < C_full.getNumOfCols());
assert(this->endOrb_ < C_full.getNumOfCols());

C_full.block(0, this->startOrb_, C_full.getNumOfRows(), this->endOrb_ - this->startOrb_ + 1, &C);
}

this->log_.info("make group");
this->makeGroup();

const int maxIteration = this->maxIteration_;
for (int i = this->lo_iteration_ + 1; i <= maxIteration; ++i) {
this->log_.info(TlUtils::format("localize iteration: %d", i));
const double sumDeltaG = this->localize_byPop(&C);

const std::string msg = TlUtils::format("%d th: G=%10.5e, delta_G=%10.5e", i, this->G_, sumDeltaG);
this->log_.info(msg);
std::cout << msg << std::endl;

DfObject::saveCloMatrix(RUN_RKS, i, C);

if (sumDeltaG < this->threshold_) {
(*(this->pPdfParam_))["lo/num_of_iterations"] = i;
(*(this->pPdfParam_))["lo/satisfied"] = "yes";
break;
}
}
}

double DfLocalize::localize(TlDenseGeneralMatrix_Lapack* pC) {
this->G_ = this->calcG(*pC, this->startOrb_, this->endOrb_);
const double sumDeltaG = this->localize_core(pC, this->startOrb_, this->endOrb_, this->startOrb_, this->endOrb_);

return sumDeltaG;
}

double DfLocalize::localize_core(TlDenseGeneralMatrix_Lapack* pC, const index_type startMO1, const index_type endMO1,
const index_type startMO2, const index_type endMO2) {
assert(startMO1 < endMO1);
assert(startMO2 < endMO2);

const index_type numOfAOs = this->m_nNumOfAOs;
assert(pC->getNumOfRows() == numOfAOs);

this->log_.info("localize: start");

std::vector<TaskItem> taskList;
if ((startMO1 == startMO2) && (endMO1 == endMO2)) {
taskList = this->getTaskList(startMO1, endMO1);
} else {
taskList = this->getTaskList(startMO1, endMO1, startMO2, endMO2);
}

const double rotatingThreshold = this->threshold_ * 0.01;
double sumDeltaG = 0.0;

assert(endMO1 <= endMO2);
this->initLockMO(endMO2 + 1);

const std::size_t numOfTasks = taskList.size();
for (std::size_t i = 0; i < numOfTasks; ++i) {
const TaskItem& task = taskList[i];
const index_type orb_i = task.first;
const index_type orb_j = task.second;

double sleep = 100.0;
while (this->isLockedMO(orb_i, orb_j)) {
TlTime::sleep(sleep);
sleep = std::min(4000.0, sleep * 2.0);
}
{
this->lockMO(orb_i);
this->lockMO(orb_j);
}

TlDenseVector_Lapack Cpi = pC->getColVector(orb_i);
TlDenseVector_Lapack Cpj = pC->getColVector(orb_j);

double A_ij = 0.0;
double B_ij = 0.0;
this->calcQA_ij(Cpi, Cpj, &A_ij, &B_ij);

const double normAB = std::sqrt(A_ij * A_ij + B_ij * B_ij);
const double deltaG = A_ij + normAB;

if (std::fabs(deltaG) > rotatingThreshold) {
sumDeltaG += deltaG;
TlDenseGeneralMatrix_Lapack rot(2, 2);
this->getRotatingMatrix(A_ij, B_ij, normAB, &rot);
this->rotateVectors(&Cpi, &Cpj, rot);

pC->setColVector(orb_i, numOfAOs, Cpi.data());
pC->setColVector(orb_j, numOfAOs, Cpj.data());
}

{
this->unlockMO(orb_i);
this->unlockMO(orb_j);
}
}

this->log_.info("localize: end");
return sumDeltaG;
}





void DfLocalize::makeGroup() {
const index_type numOfAOs = this->m_nNumOfAOs;

std::vector<int>::const_iterator itMax = std::max_element(this->atomGroupList_.begin(), this->atomGroupList_.end());
const int numOfGroups = *itMax + 1;

this->group_.resize(numOfGroups);
for (int i = 0; i < numOfGroups; ++i) {
this->group_[i].resize(numOfAOs);
}

for (index_type orb = 0; orb < numOfAOs; ++orb) {
const index_type atomIndex = this->orbInfo_.getAtomIndex(orb);
const int groupIndex = this->atomGroupList_[atomIndex];
this->group_[groupIndex].set(orb, 1.0);
}
}

double DfLocalize::calcG(const TlDenseGeneralMatrix_Lapack& C, const index_type startMO, const index_type endMO) {
this->log_.info("calc G");
double G = 0.0;

const index_type size = endMO - startMO;
#pragma omp parallel for schedule(runtime) reduction(+: G)
for (index_type i = 0; i < size; ++i) {
const index_type orb = startMO + i;
const double QAii2 = this->calcQA_ii(C, orb);
G += QAii2;
}

return G;
}

double DfLocalize::calcG_sort(const TlDenseGeneralMatrix_Lapack& C, const index_type startMO, const index_type endMO) {
this->log_.info("calc G");
double G = 0.0;

const index_type size = endMO - startMO;
this->groupMoPops_.resize(size);
#pragma omp parallel for schedule(runtime) reduction(+: G)
for (index_type i = 0; i < size; ++i) {
const index_type mo = startMO + i;
const double QAii2 = this->calcQA_ii(C, mo);
G += QAii2;

this->groupMoPops_[i] = MoPop(mo, QAii2);
}

switch (this->pairingOrder_) {
case PO_BIG_BIG: {
this->log_.info("sort QA table (greater)");
std::sort(this->groupMoPops_.begin(), this->groupMoPops_.end(), MoPop::MoPop_sort_functor_cmp_grater());
} break;

case PO_BIG_SMALL: {
this->log_.info("sort QA table (greater)");
std::sort(this->groupMoPops_.begin(), this->groupMoPops_.end(), MoPop::MoPop_sort_functor_cmp_grater());
} break;

case PO_SMALL_SMALL: {
this->log_.info("sort QA table (lesser)");
std::sort(this->groupMoPops_.begin(), this->groupMoPops_.end(), MoPop::MoPop_sort_functor_cmp_lesser());
} break;

default:
this->log_.info("no sort QA table");
break;
}

return G;
}

void DfLocalize::rotateVectors(TlDenseVector_Lapack* pCpi, TlDenseVector_Lapack* pCpj,
const TlDenseGeneralMatrix_Lapack& rot) {
const index_type numOfAOs = this->m_nNumOfAOs;
assert(pCpi->getSize() == numOfAOs);
assert(pCpj->getSize() == numOfAOs);

TlDenseGeneralMatrix_Lapack A(2, numOfAOs);
A.setRowVector(0, numOfAOs, pCpi->data());
A.setRowVector(1, numOfAOs, pCpj->data());

const TlDenseGeneralMatrix_Lapack B = rot * A;  

*pCpi = B.getRowVector(0);
*pCpj = B.getRowVector(1);
assert(pCpi->getSize() == numOfAOs);
assert(pCpj->getSize() == numOfAOs);
}

void DfLocalize::getRotatingMatrix(const double A_ij, const double B_ij, const double normAB,
TlDenseGeneralMatrix_Lapack* pRot) {
assert(pRot != NULL);
pRot->resize(2, 2);

const double cos4a = -A_ij / normAB;

const double cos2a = std::sqrt(0.5 * (cos4a + 1.0));
const double cos_a_2 = 0.5 * (cos2a + 1.0);  
const double cos_a = std::sqrt(cos_a_2);
double sin_a = std::sqrt(1.0 - cos_a_2);

if (B_ij < 0.0) {
sin_a = -sin_a;
}

pRot->set(0, 0, cos_a);
pRot->set(0, 1, sin_a);
pRot->set(1, 0, -sin_a);
pRot->set(1, 1, cos_a);
}

void DfLocalize::calcQA_ij(const TlDenseVector_Lapack& Cpi, const TlDenseVector_Lapack& Cpj, double* pA_ij,
double* pB_ij) {
assert(pA_ij != NULL);
assert(pB_ij != NULL);

const index_type numOfGrps = this->group_.size();

const TlDenseVector_Lapack SCpi = this->S_ * Cpi;
const TlDenseVector_Lapack SCpj = this->S_ * Cpj;

double sumAij = 0.0;
double sumBij = 0.0;

#pragma omp parallel for schedule(dynamic) reduction(+: sumAij, sumBij)
for (index_type grp = 0; grp < numOfGrps; ++grp) {
TlDenseVector_Lapack tCqi = Cpi;
TlDenseVector_Lapack tCqj = Cpj;
tCqi.dotInPlace(this->group_[grp]);
tCqj.dotInPlace(this->group_[grp]);

const double QAii = tCqi * SCpi;
const double QAjj = tCqj * SCpj;
const double QAij = 0.5 * (tCqj * SCpi + tCqi * SCpj);

const double QAii_QAjj = QAii - QAjj;

sumAij += QAij * QAij - 0.25 * QAii_QAjj * QAii_QAjj;
sumBij += QAij * QAii_QAjj;
}

*pA_ij += sumAij;
*pB_ij += sumBij;
}

double DfLocalize::calcQA_ii_0(const TlDenseGeneralMatrix_Lapack& C, const index_type orb_i) {
const index_type numOfGrps = this->group_.size();
double sumQAii2 = 0.0;

const TlDenseVector_Lapack Cpi = C.getColVector(orb_i);
const TlDenseVector_Lapack SCpi = this->S_ * Cpi;

for (index_type grp = 0; grp < numOfGrps; ++grp) {
TlDenseVector_Lapack tCqi = Cpi;
tCqi.dotInPlace(this->group_[grp]);

const double QAii = tCqi * SCpi;
sumQAii2 += QAii;
}

return sumQAii2;
}

double DfLocalize::calcQA_ii(const TlDenseGeneralMatrix_Lapack& C, const index_type orb_i) {
const index_type numOfGrps = this->group_.size();
double sumQAii2 = 0.0;

const TlDenseVector_Lapack Cpi = C.getColVector(orb_i);
const TlDenseVector_Lapack SCpi = this->S_ * Cpi;

for (index_type grp = 0; grp < numOfGrps; ++grp) {
TlDenseVector_Lapack tCqi = Cpi;
tCqi.dotInPlace(this->group_[grp]);

const double QAii = tCqi * SCpi;
sumQAii2 += QAii * QAii;
}

return sumQAii2;
}

std::vector<DfLocalize::TaskItem> DfLocalize::getTaskList(const index_type startMO, const index_type endMO) {
const index_type dim = endMO - startMO + 1;
const index_type pairs = dim * (dim - 1) / 2;
std::vector<TaskItem> taskList(pairs);

index_type taskIndex = 0;
TaskItem item;
for (int index1 = 1; index1 < dim; ++index1) {
const int max_index2 = dim - index1;
for (int index2 = 0; index2 < max_index2; ++index2) {
const index_type orb_i = startMO + index2;
const index_type orb_j = startMO + index1 + index2;

taskList[taskIndex] = std::make_pair(orb_i, orb_j);
++taskIndex;
}
}
assert(taskIndex == pairs);

return taskList;
}

std::vector<DfLocalize::TaskItem> DfLocalize::getTaskList(const index_type startMO1, const index_type endMO1,
const index_type startMO2, const index_type endMO2) {
assert(startMO1 < endMO1);
assert(endMO1 < startMO2);
assert(startMO2 < endMO2);

const index_type pairs = (endMO1 - startMO1 + 1) * (endMO2 - startMO2 + 1);
std::vector<TaskItem> taskList(pairs);

index_type taskIndex = 0;
for (index_type mo1 = startMO1; mo1 <= endMO1; ++mo1) {
for (index_type mo2 = startMO2; mo2 <= endMO2; ++mo2) {
taskList[taskIndex] = std::make_pair(mo1, mo2);
++taskIndex;
}
}
assert(taskIndex == pairs);

return taskList;
}










void DfLocalize::initLockMO(const index_type numOfMOs) {
this->lockMOs_.resize(numOfMOs);
std::fill(this->lockMOs_.begin(), this->lockMOs_.end(), 0);
}

void DfLocalize::lockMO(const index_type mo) {
assert(mo < static_cast<index_type>(this->lockMOs_.size()));
this->lockMOs_[mo] = 1;
}

void DfLocalize::unlockMO(const index_type mo) {
assert(mo < static_cast<index_type>(this->lockMOs_.size()));
this->lockMOs_[mo] = 0;
}

bool DfLocalize::isLockedMO(const index_type mo1, const index_type mo2) const {
assert(mo1 < static_cast<index_type>(this->lockMOs_.size()));
assert(mo2 < static_cast<index_type>(this->lockMOs_.size()));
bool answer = true;

if ((this->lockMOs_[mo1] + this->lockMOs_[mo2] == 0)) {
answer = false;
}

return answer;
}

double DfLocalize::localize_byPop(TlDenseGeneralMatrix_Lapack* pC) {
this->G_ = this->calcG_sort(*pC, this->startOrb_, this->endOrb_);
const double sumDeltaG = this->localize_core_byPop(pC, this->startOrb_, this->endOrb_);

return sumDeltaG;
}

double DfLocalize::localize_core_byPop(TlDenseGeneralMatrix_Lapack* pC, const index_type startMO1,
const index_type endMO1) {
assert(startMO1 < endMO1);

const index_type numOfAOs = this->m_nNumOfAOs;
assert(pC->getNumOfRows() == numOfAOs);

this->log_.info("localize: start");

const std::vector<TaskItem> taskList = this->getTaskList_byPop();

const double rotatingThreshold = this->threshold_ * 0.01;
double sumDeltaG = 0.0;

this->initLockMO(endMO1 + 1);

const std::size_t numOfTasks = taskList.size();
this->log_.info(TlUtils::format("#tasks: %ld", numOfTasks));
std::size_t progress = 0;
const std::size_t period = numOfTasks * 0.1;

for (std::size_t i = 0; i < numOfTasks; ++i) {
index_type orb_i, orb_j;

{
const TaskItem& task = taskList[i];
orb_i = task.first;
orb_j = task.second;
}



++progress;
if (progress > period) {
const double ratio = double(i) / double(numOfTasks) * 100.0;
this->log_.info(TlUtils::format("calc progress: %16ld/%16ld; %5.2f%%", i, numOfTasks, ratio));
progress = 0;
}

TlDenseVector_Lapack Cpi = pC->getColVector(orb_i);
TlDenseVector_Lapack Cpj = pC->getColVector(orb_j);

double A_ij = 0.0;
double B_ij = 0.0;
this->calcQA_ij(Cpi, Cpj, &A_ij, &B_ij);

const double normAB = std::sqrt(A_ij * A_ij + B_ij * B_ij);
const double deltaG = A_ij + normAB;

if (std::fabs(deltaG) > rotatingThreshold) {
TlDenseGeneralMatrix_Lapack rot(2, 2);
this->getRotatingMatrix(A_ij, B_ij, normAB, &rot);
this->rotateVectors(&Cpi, &Cpj, rot);

sumDeltaG += deltaG;
pC->setColVector(orb_i, numOfAOs, Cpi.data());
pC->setColVector(orb_j, numOfAOs, Cpj.data());
}

}

this->log_.info("localize: end");
return sumDeltaG;
}

std::vector<DfLocalize::TaskItem> DfLocalize::getTaskList_byPop() {
const index_type dim = this->groupMoPops_.size();
const index_type pairs = dim * (dim - 1) / 2;
std::vector<TaskItem> taskList(pairs);

index_type taskIndex = 0;
TaskItem item;

switch (this->pairingOrder_) {
case PO_ORDERED:
for (int i = 0; i < dim; ++i) {
const index_type mo1 = this->groupMoPops_[i].mo;
for (int j = i + 1; j < dim; ++j) {
const index_type mo2 = this->groupMoPops_[j].mo;

taskList[taskIndex] = std::make_pair(mo1, mo2);
++taskIndex;
}
}
break;

case PO_BIG_BIG:
for (int i = 1; i < dim; ++i) {  
const int max_j = dim - i;
for (int j = 0; j < max_j; ++j) {  
const index_type mo1 = this->groupMoPops_[i + j].mo;
const index_type mo2 = this->groupMoPops_[j].mo;

taskList[taskIndex] = std::make_pair(mo1, mo2);
++taskIndex;
}
}

break;

case PO_BIG_SMALL:
for (int i = 0; i < dim - 1; ++i) {  
const int max_j = dim - 1 - i;
for (int j = 0; j < max_j; ++j) {  
const index_type mo1 = this->groupMoPops_[i + j].mo;
const index_type mo2 = this->groupMoPops_[dim - 1 - j].mo;

taskList[taskIndex] = std::make_pair(mo1, mo2);
++taskIndex;
}
}
break;

case PO_SMALL_SMALL:
for (int i = 1; i < dim; ++i) {  
const int max_j = dim - i;
for (int j = 0; j < max_j; ++j) {  
const index_type mo1 = this->groupMoPops_[i + j].mo;
const index_type mo2 = this->groupMoPops_[j].mo;

taskList[taskIndex] = std::make_pair(mo1, mo2);
++taskIndex;
}
}
break;

default:
std::cerr << TlUtils::format("unknown pairing order algorithm: %d", static_cast<int>(this->pairingOrder_)) << std::endl;
CnErr.abort();
}

assert(taskIndex == pairs);

return taskList;
}
