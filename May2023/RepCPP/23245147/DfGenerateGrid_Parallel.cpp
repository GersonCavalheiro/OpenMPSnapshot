
#include "DfGenerateGrid_Parallel.h"

#include <cassert>

#include "DfXCFunctional.h"
#include "TlCommunicate.h"
#include "TlUtils.h"

DfGenerateGrid_Parallel::DfGenerateGrid_Parallel(TlSerializeData* pPdfParam) : DfGenerateGrid(pPdfParam) {
}

DfGenerateGrid_Parallel::~DfGenerateGrid_Parallel() {
}

void DfGenerateGrid_Parallel::logger(const std::string& str) const {
TlCommunicate& rComm = TlCommunicate::getInstance();

if (rComm.isMaster() == true) {
DfGenerateGrid::logger(str);
}
}

void DfGenerateGrid_Parallel::makeTable() {
TlCommunicate& rComm = TlCommunicate::getInstance();
if (rComm.isMaster() == true) {
DfGenerateGrid::makeTable();
}

rComm.broadcast(this->maxRadii_);
}

TlDenseGeneralMatrix_Lapack DfGenerateGrid_Parallel::getOMatrix() {
TlCommunicate& rComm = TlCommunicate::getInstance();

TlDenseGeneralMatrix_Lapack O;
if (rComm.isMaster() == true) {
O = DfGenerateGrid::getOMatrix();
}
rComm.broadcast(&O);

return O;
}

void DfGenerateGrid_Parallel::generateGrid(const TlDenseGeneralMatrix_Lapack& O) {
this->generateGrid_DC(O);

this->gatherGridData();
}

void DfGenerateGrid_Parallel::generateGrid_DC(const TlDenseGeneralMatrix_Lapack& O) {
this->logger("generate grid by DC");
TlCommunicate& rComm = TlCommunicate::getInstance();

const int nEndAtomNumber = this->m_nNumOfAtoms;

const int nProc = rComm.getNumOfProc();
const int nRank = rComm.getRank();
const int nRange = nEndAtomNumber;
const int nInterval = (nRange + (nProc - 1)) / nProc;  
const int nLocalStart = nRank * nInterval;             
const int nLocalEnd = std::min((nLocalStart + nInterval), nEndAtomNumber);

std::size_t numOfGrids = 0;
#pragma omp parallel for schedule(runtime)
for (int atom = nLocalStart; atom < nLocalEnd; ++atom) {
std::vector<double> coordX;
std::vector<double> coordY;
std::vector<double> coordZ;
std::vector<double> weight;

DfGenerateGrid::generateGrid_atom(O, atom, &coordX, &coordY, &coordZ, &weight);

const std::size_t numOfAtomGrids = weight.size();
if (numOfAtomGrids == 0) {
continue;
}

#pragma omp critical(DfGenerateGrid__generateGrid)
{
this->grdMat_.resize(numOfGrids + numOfAtomGrids, this->numOfColsOfGrdMat_);
for (std::size_t i = 0; i < numOfAtomGrids; ++i) {
this->grdMat_.set(numOfGrids, 0, coordX[i]);
this->grdMat_.set(numOfGrids, 1, coordY[i]);
this->grdMat_.set(numOfGrids, 2, coordZ[i]);
this->grdMat_.set(numOfGrids, 3, weight[i]);
this->grdMat_.set(numOfGrids, 4, atom);
++numOfGrids;
}
}
}
}

void DfGenerateGrid_Parallel::gatherGridData() {
this->logger("gather grid data");

TlCommunicate& rComm = TlCommunicate::getInstance();
const int numOfProcs = rComm.getNumOfProc();

index_type numOfRowsOfGlobalGridMatrix = this->grdMat_.getNumOfRows();

const int tag = TAG_GENGRID_GATHER_GRID_DATA;
if (rComm.isMaster() == true) {
const index_type numOfColsOfGlobalGridMatrix = this->grdMat_.getNumOfCols();
this->log_.info(
TlUtils::format("grid matrix size: %d, %d", numOfRowsOfGlobalGridMatrix, numOfRowsOfGlobalGridMatrix));
TlDenseGeneralMatrix_Lapack grdMat(numOfRowsOfGlobalGridMatrix, numOfColsOfGlobalGridMatrix);

index_type currentNumOfRows = 0;

grdMat.block(currentNumOfRows, 0, this->grdMat_);
currentNumOfRows += this->grdMat_.getNumOfRows();

std::vector<bool> recvCheck(numOfProcs, false);
for (int i = 1; i < numOfProcs; ++i) {
int proc = 0;
TlDenseGeneralMatrix_Lapack tmpGrdMat;
rComm.receiveDataFromAnySource(tmpGrdMat, &proc, tag);
if (recvCheck[proc] != false) {
this->log_.warn(TlUtils::format("already received grid data from %d", proc));
}
recvCheck[proc] = true;
this->log_.debug(TlUtils::format("recv grid data from %d", proc));

this->log_.info(TlUtils::format("recv block mat (%d, %d) from %d", tmpGrdMat.getNumOfRows(),
tmpGrdMat.getNumOfCols(), proc));
numOfRowsOfGlobalGridMatrix += tmpGrdMat.getNumOfRows();
grdMat.resize(numOfRowsOfGlobalGridMatrix, numOfColsOfGlobalGridMatrix);

grdMat.block(currentNumOfRows, 0, tmpGrdMat);
currentNumOfRows += tmpGrdMat.getNumOfRows();
}

this->saveGridMatrix(0, grdMat);
} else {
rComm.sendData(this->grdMat_, 0, tag);
this->log_.debug("send grid data");
}
}
