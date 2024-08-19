
#include <cstdlib>
#include <iostream>

#include "TlGetopt.h"
#include "TlMath.h"
#include "TlUtils.h"
#include "tl_dense_general_matrix_arrays_mmap_roworiented.h"
#include "tl_dense_general_matrix_mmap.h"
#include "tl_dense_symmetric_matrix_lapack.h"
#include "tl_matrix_utils.h"

int main(int argc, char* argv[]) {
TlGetopt opt(argc, argv, "cghv");

const bool bVerbose = (opt["v"] == "defined");
const bool bCsvMode = (opt["c"] == "defined");
const bool bGuessMode = (opt["g"] == "defined");

std::string path = opt[1];
if (bVerbose) {
std::cerr << "loading... " << path << std::endl;
}

TlMatrixObject::HeaderInfo headerInfo;
const bool isLoadable = TlMatrixUtils::getHeaderInfo(path, &headerInfo);
if (isLoadable == true) {
bool isFound = false;

switch (headerInfo.matrixType) {
case TlMatrixObject::RLHD: {
std::cout << "type: symmetric" << std::endl;
TlDenseSymmetricMatrix_Lapack M;
M.load(path);

if (bCsvMode == true) {
M.saveCsv(std::cout);
} else if (bGuessMode == true) {
M.saveText(std::cout);
} else {
std::cout << M << std::endl;
}
} break;

case TlMatrixObject::CSFD: {
std::cout << "type: normal (column-major)" << std::endl;
TlDenseGeneralMatrix_mmap M(path);
const TlMatrixObject::index_type numOfRows = M.getNumOfRows();
const TlMatrixObject::index_type numOfCols = M.getNumOfCols();

for (TlMatrixObject::index_type c = 0; c < numOfCols; ++c) {
TlUtils::progressbar(float(c) / numOfCols);
for (TlMatrixObject::index_type r = 0; r < numOfRows; ++r) {
const double v = M.get(r, c);
if (!TlMath::isfinite(v)) {
isFound = true;
std::cerr << TlUtils::format("not finite: [%ld, %ld]=%f", r, c, v) << std::endl;
}
}
}
TlUtils::progressbar(1.0);
std::cout << std::endl;
} break;

case TlMatrixObject::RSFD: {
std::cout << "type: normal (row-major)" << std::endl;
break;
}

case TlMatrixObject::ABGD: {
std::cout << "type: General Dens-matrix stored by Arrays Blocks" << std::endl;
std::cout << "#vectors: " << headerInfo.numOfVectors << std::endl;
std::cout << "sizeOfVector: " << headerInfo.sizeOfVector << std::endl;
std::cout << TlUtils::format("unit: %d/%d", headerInfo.subunitId + 1, headerInfo.numOfSubunits)
<< std::endl;
std::cout << "chunk size: " << headerInfo.sizeOfChunk << std::endl;

TlDenseGeneralMatrix_arrays_mmap_RowOriented M(path);
const TlMatrixObject::index_type numOfRows = M.getNumOfRows();
const TlMatrixObject::index_type numOfCols = M.getNumOfCols();
const int id = M.getSubunitID();
for (TlMatrixObject::index_type r = 0; r < numOfRows; ++r) {
TlUtils::progressbar(float(r) / numOfRows);
if (id == M.getSubunitID(r)) {
const std::vector<double> cols = M.getRowVector(r);
for (TlMatrixObject::index_type c = 0; c < numOfCols; ++c) {
const double v = cols[c];
if (!TlMath::isfinite(v)) {
isFound = true;
std::cerr << TlUtils::format("not finite: [%ld, %ld]=%f", r, c, v) << std::endl;
}
}
}
}
TlUtils::progressbar(1.0);
std::cout << std::endl;
break;
}

default:
std::cout << "sorry, not supported matrix." << std::endl;
break;
}

if (isFound) {
return EXIT_FAILURE;
}
} else {
std::cerr << "cannot open file: " << path << std::endl;
return EXIT_FAILURE;
}

return EXIT_SUCCESS;
}
