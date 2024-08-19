#include "tl_dense_general_matrix_arrays_mmap_roworiented.h"

#include <cassert>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif  

#include "TlFile.h"
#include "TlSystem.h"
#include "TlUtils.h"
#include "tl_dense_general_matrix_eigen.h"
#include "tl_dense_general_matrix_lapack.h"
#include "tl_matrix_utils.h"


TlDenseGeneralMatrix_arrays_mmap_RowOriented::TlDenseGeneralMatrix_arrays_mmap_RowOriented(
const std::string& baseFilePath, const index_type row, const index_type col, const int numOfSubunits,
const int subunitID, const int reservedCols, bool forceCreateNewFile)
: TlDenseMatrix_arrays_mmap_Object(baseFilePath, row, col, numOfSubunits, subunitID, reservedCols, forceCreateNewFile),
tempCsfdMatPath_(""),
isRemoveTempCsfdMat_(false) {
this->tempDir_ = "/tmp";
}

TlDenseGeneralMatrix_arrays_mmap_RowOriented::TlDenseGeneralMatrix_arrays_mmap_RowOriented(const std::string& filePath)
: TlDenseMatrix_arrays_mmap_Object(filePath), tempCsfdMatPath_(""), isRemoveTempCsfdMat_(false) {
this->tempDir_ = "/tmp";
}



TlDenseGeneralMatrix_arrays_mmap_RowOriented::~TlDenseGeneralMatrix_arrays_mmap_RowOriented() {
if ((!this->tempCsfdMatPath_.empty()) && (this->isRemoveTempCsfdMat_)) {
TlFile::remove(this->tempCsfdMatPath_);
this->tempCsfdMatPath_ = "";
}
}

void TlDenseGeneralMatrix_arrays_mmap_RowOriented::resize(const index_type row, const index_type col) {
TlDenseMatrix_arrays_mmap_Object::resize(row, col);
}

void TlDenseGeneralMatrix_arrays_mmap_RowOriented::reserveColSize(const index_type reserveColSize) {
TlDenseMatrix_arrays_mmap_Object::reserveVectorSize(reserveColSize);
}

void TlDenseGeneralMatrix_arrays_mmap_RowOriented::set(const index_type row, const index_type col, const double value) {
TlDenseMatrix_arrays_mmap_Object::set_to_vm(row, col, value);
}

void TlDenseGeneralMatrix_arrays_mmap_RowOriented::add(const index_type row, const index_type col, const double value) {
TlDenseMatrix_arrays_mmap_Object::add_to_vm(row, col, value);
}

double TlDenseGeneralMatrix_arrays_mmap_RowOriented::get(const index_type row, const index_type col) const {
return TlDenseMatrix_arrays_mmap_Object::get_from_vm(row, col);
}

std::vector<double> TlDenseGeneralMatrix_arrays_mmap_RowOriented::getRowVector(const index_type row) const {
return TlDenseMatrix_arrays_mmap_Object::getVector(row);
}

std::size_t TlDenseGeneralMatrix_arrays_mmap_RowOriented::getRowVector(const index_type row, double* pBuf,
std::size_t maxCount) const {
return TlDenseMatrix_arrays_mmap_Object::getVector(row, pBuf, maxCount);
}

void TlDenseGeneralMatrix_arrays_mmap_RowOriented::setColVector(const index_type col, const std::valarray<double>& v) {
TlDenseMatrix_arrays_mmap_Object::setAcrossMultipleVectors(col, v);
}

TlDenseGeneralMatrix_Lapack TlDenseGeneralMatrix_arrays_mmap_RowOriented::getTlMatrixObject() const {
const index_type numOfRows = this->getNumOfRows();
const index_type numOfCols = this->getNumOfCols();
TlDenseGeneralMatrix_Lapack answer(numOfRows, numOfCols);

for (index_type r = 0; r < numOfRows; ++r) {
TlDenseVector_Lapack v = TlDenseMatrix_arrays_mmap_Object::getVector(r);
assert(v.getSize() == numOfCols);
for (index_type c = 0; c < numOfCols; ++c) {
answer.set(r, c, v.get(c));
}
}

return answer;
}

void TlDenseGeneralMatrix_arrays_mmap_RowOriented::convertMemoryLayout(const std::string& tempCsfdMatPath, const bool verbose,
const bool showProgress) const {
const TlMatrixObject::index_type numOfRows = this->getNumOfRows();
const TlMatrixObject::index_type numOfCols = this->getNumOfCols();
const int numOfSubunits = this->getNumOfSubunits();
const int sizeOfChunk = this->getSizeOfChunk();
const int unit = this->getSubunitID();

if (tempCsfdMatPath.empty()) {
this->tempCsfdMatPath_ = TlUtils::format("./tmp.csfd.mat.%d", unit);
this->isRemoveTempCsfdMat_ = true;
} else {
this->tempCsfdMatPath_ = tempCsfdMatPath;
}

const int numOfLocalChunks =
TlDenseMatrix_arrays_mmap_Object::getNumOfLocalChunks(numOfRows, numOfSubunits, sizeOfChunk);

const TlMatrixObject::index_type localNumOfRows = numOfLocalChunks * sizeOfChunk;

static const bool forceCreateNew = true;
TlDenseGeneralMatrix_mmap outMat(this->tempCsfdMatPath_, localNumOfRows, numOfCols, forceCreateNew);
assert(outMat.getNumOfRows() == localNumOfRows);
assert(outMat.getNumOfCols() == numOfCols);

if (verbose) {
std::cerr << "output matrix has been prepared by mmap." << std::endl;
}

int progress = 0;
#pragma omp parallel
{
std::vector<double> chunkBuf(numOfCols * sizeOfChunk);
std::vector<double> transBuf(numOfCols * sizeOfChunk);

int threadId = 0;
#ifdef _OPENMP
threadId = omp_get_thread_num();
#endif  

#pragma omp for schedule(runtime)
for (int chunk = 0; chunk < numOfLocalChunks; ++chunk) {
const TlMatrixObject::index_type chunkStartRow = sizeOfChunk * (numOfSubunits * chunk + unit);

if (chunkStartRow < numOfRows) {
this->getChunk(chunkStartRow, &(chunkBuf[0]), numOfCols * sizeOfChunk);

const TlMatrixObject::index_type readRowChunks = std::min(sizeOfChunk, numOfRows - chunkStartRow);
TlUtils::changeMemoryLayout(&(chunkBuf[0]), readRowChunks, numOfCols, &(transBuf[0]));

TlDenseGeneralMatrix_Eigen tmpMat(readRowChunks, numOfCols, &(transBuf[0]));

outMat.block(chunk * sizeOfChunk, 0, tmpMat);
}

#pragma omp atomic
++progress;

if ((threadId == 0) && (showProgress)) {
TlUtils::progressbar(float(progress) / numOfLocalChunks);
}
}
}

if (showProgress) {
TlUtils::progressbar(1.0);
std::cout << std::endl;
}
}

void TlDenseGeneralMatrix_arrays_mmap_RowOriented::set2csfd(TlDenseGeneralMatrix_mmap* pOutMat, const bool verbose,
const bool showProgress) const {
const TlMatrixObject::index_type numOfRows = this->getNumOfRows();
const TlMatrixObject::index_type numOfCols = this->getNumOfCols();
const int numOfSubunits = this->getNumOfSubunits();
const int sizeOfChunk = this->getSizeOfChunk();
const int unit = this->getSubunitID();

if (this->tempCsfdMatPath_.empty()) {
std::cerr << "please call convertMemoryLayout() before this function. stop." << std::endl;
abort();
}

TlDenseGeneralMatrix_mmap inMat(this->tempCsfdMatPath_);

const int numOfLocalChunks =
TlDenseMatrix_arrays_mmap_Object::getNumOfLocalChunks(numOfRows, numOfSubunits, sizeOfChunk);

int progress = 0;
#pragma omp parallel
{
int threadId = 0;
#ifdef _OPENMP
threadId = omp_get_thread_num();
#endif  

#pragma omp for schedule(runtime)
for (int chunk = 0; chunk < numOfLocalChunks; ++chunk) {
const TlMatrixObject::index_type row = sizeOfChunk * chunk;
const TlMatrixObject::index_type chunkStartRow = sizeOfChunk * (numOfSubunits * chunk + unit);
if (chunkStartRow >= numOfRows) {
continue;
}
const TlMatrixObject::index_type rowDistance = std::min(sizeOfChunk, numOfRows - chunkStartRow);
TlDenseGeneralMatrix_Eigen tmpMat;
inMat.block(row, 0, rowDistance, numOfCols, &tmpMat);

pOutMat->block(chunkStartRow, 0, tmpMat);

#pragma omp atomic
++progress;

if ((threadId == 0) && (showProgress)) {
TlUtils::progressbar(float(progress) / numOfLocalChunks);
}
}
}

if (showProgress) {
TlUtils::progressbar(1.0);
std::cout << std::endl;
}
}

std::ostream& operator<<(std::ostream& stream, const TlDenseGeneralMatrix_arrays_mmap_RowOriented& mat) {
const TlMatrixObject::index_type numOfRows = mat.getNumOfRows();
const TlMatrixObject::index_type numOfCols = mat.getNumOfCols();

for (TlMatrixObject::index_type ord = 0; ord < numOfCols; ord += 10) {
stream << "       ";
for (TlMatrixObject::index_type j = ord; ((j < ord + 10) && (j < numOfCols)); ++j) {
stream << TlUtils::format("   %5d th", j + 1);
}
stream << "\n ----";

for (TlMatrixObject::index_type j = ord; ((j < ord + 10) && (j < numOfCols)); ++j) {
stream << "-----------";
}
stream << "----\n";

for (TlMatrixObject::index_type i = 0; i < numOfRows; ++i) {
stream << TlUtils::format(" %5d  ", i + 1);

for (TlMatrixObject::index_type j = ord; ((j < ord + 10) && (j < numOfCols)); ++j) {
if (mat.getSubunitID(i) == mat.getSubunitID()) {
stream << TlUtils::format(" %10.6lf", mat.get(i, j));

} else {
stream << " ----------";
}
}
stream << "\n";
}
stream << "\n\n";
}

return stream;
}

bool convert2csfd(const std::string& rvmBasePath, const int unit, const std::string& outputPath, const bool verbose,
const bool showProgress) {
bool answer = false;

TlMatrixObject::index_type numOfRows = 0;
TlMatrixObject::index_type numOfCols = 0;
int numOfSubunits = 0;
int sizeOfChunk = 0;
{
const std::string inputPath0 = TlDenseMatrix_arrays_mmap_Object::getFileName(rvmBasePath, unit);

TlMatrixObject::HeaderInfo headerInfo;
const bool isLoadable = TlMatrixUtils::getHeaderInfo(inputPath0, &headerInfo);

if (isLoadable != true) {
std::cerr << "can not open file: " << inputPath0 << std::endl;
answer = false;
}

numOfRows = headerInfo.numOfVectors;
numOfCols = headerInfo.sizeOfVector;
}

{
TlDenseGeneralMatrix_arrays_mmap_RowOriented partMat(rvmBasePath, 1, 1, numOfSubunits, unit);
if (verbose) {
std::cerr << "load partial matrix " << std::endl;
}

std::vector<double> chunkBuf(numOfCols * sizeOfChunk);
std::vector<double> transBuf(numOfCols * sizeOfChunk);
const int numOfLocalChunks =
TlDenseMatrix_arrays_mmap_Object::getNumOfLocalChunks(numOfRows, numOfSubunits, sizeOfChunk);

const TlMatrixObject::index_type localNumOfRows = numOfLocalChunks * sizeOfChunk;
TlDenseGeneralMatrix_mmap outMat(outputPath, localNumOfRows, numOfCols);
if (verbose) {
std::cerr << "output matrix has been prepared by mmap." << std::endl;
}

for (int chunk = 0; chunk < numOfLocalChunks; ++chunk) {
const TlMatrixObject::index_type chunkStartRow = sizeOfChunk * (numOfSubunits * chunk + unit);

if (chunkStartRow < numOfRows) {
partMat.getChunk(chunkStartRow, &(chunkBuf[0]), numOfCols * sizeOfChunk);

const TlMatrixObject::index_type readRowChunks = std::min(sizeOfChunk, numOfRows - chunkStartRow);
TlUtils::changeMemoryLayout(&(chunkBuf[0]), readRowChunks, numOfCols, &(transBuf[0]));

TlDenseGeneralMatrix_Eigen tmpMat(readRowChunks, numOfCols, &(transBuf[0]));
outMat.block(chunk * sizeOfChunk, 0, tmpMat);
}

if (showProgress) {
TlUtils::progressbar(float(chunk) / numOfLocalChunks);
}
}

if (showProgress) {
TlUtils::progressbar(1.0);
std::cout << std::endl;
}
}

return answer;
}

void copy2csfd(const TlMatrixObject::index_type numOfRows, const TlMatrixObject::index_type numOfCols,
const int numOfSubunits, const int sizeOfChunk, const std::string& inMatPath, const int unit,
TlDenseGeneralMatrix_mmap* pOutMat, const bool verbose) {
TlDenseGeneralMatrix_mmap inMat(inMatPath);

const int numOfLocalChunks =
TlDenseMatrix_arrays_mmap_Object::getNumOfLocalChunks(numOfRows, numOfSubunits, sizeOfChunk);

TlDenseGeneralMatrix_Eigen tmpMat;
for (int chunk = 0; chunk < numOfLocalChunks; ++chunk) {
TlMatrixObject::index_type row = sizeOfChunk * chunk;
const TlMatrixObject::index_type chunkStartRow = sizeOfChunk * (numOfSubunits * chunk + unit);
if (chunkStartRow >= numOfRows) {
continue;
}

TlMatrixObject::index_type rowDistance = std::min(sizeOfChunk, numOfRows - chunkStartRow);

inMat.block(row, 0, rowDistance, numOfCols, &tmpMat);

pOutMat->block(chunkStartRow, 0, tmpMat);
}
}

bool transpose2CSFD(const std::string& rvmBasePath, const std::string& outputMatrixPath, const std::string& tmpPath,
const bool verbose, const bool showProgress) {
bool answer = false;

TlMatrixObject::index_type numOfRows = 0;
TlMatrixObject::index_type numOfCols = 0;
int numOfSubunits = 0;
int sizeOfChunk = 0;
{
int subunitID = 0;
const std::string inputPath0 = TlDenseMatrix_arrays_mmap_Object::getFileName(rvmBasePath, subunitID);

TlMatrixObject::HeaderInfo headerInfo;
const bool isLoadable = TlMatrixUtils::getHeaderInfo(inputPath0, &headerInfo);

if (isLoadable != true) {
std::cerr << "can not open file: " << inputPath0 << std::endl;
return false;
}

numOfRows = headerInfo.numOfVectors;
numOfCols = headerInfo.sizeOfVector;
numOfSubunits = headerInfo.numOfSubunits;
sizeOfChunk = headerInfo.sizeOfChunk;

if (verbose) {
std::cerr << "rows: " << numOfRows << std::endl;
std::cerr << "cols: " << numOfCols << std::endl;
std::cerr << "units: " << numOfSubunits << std::endl;
std::cerr << "chunk: " << sizeOfChunk << std::endl;
}
}

if (TlFile::isExistFile(outputMatrixPath)) {
if (verbose) {
std::cerr << "file overwrite: " << outputMatrixPath << std::endl;
}
TlFile::remove(outputMatrixPath);
}
TlDenseGeneralMatrix_mmap outMat(outputMatrixPath, numOfRows, numOfCols);
if (verbose) {
std::cerr << "output matrix has been prepared by mmap." << std::endl;
}

const int pid = TlSystem::getPID();
for (int unit = 0; unit < numOfSubunits; ++unit) {


const std::string inputPath = TlDenseMatrix_arrays_mmap_Object::getFileName(rvmBasePath, unit);
TlDenseGeneralMatrix_arrays_mmap_RowOriented inMat(inputPath);

const std::string tempCsfdPath = TlUtils::format("%s/csfd.%d.%d.mat", tmpPath.c_str(), pid, unit);
inMat.convertMemoryLayout(tempCsfdPath, verbose, showProgress);
inMat.set2csfd(&outMat, verbose, showProgress);
TlFile::remove(tempCsfdPath);
}

return answer;
}
