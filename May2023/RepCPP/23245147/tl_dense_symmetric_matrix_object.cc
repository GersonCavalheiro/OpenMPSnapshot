#include "tl_dense_symmetric_matrix_object.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

#include "TlUtils.h"
#include "tl_dense_general_matrix_object.h"
#include "tl_dense_matrix_impl_object.h"
#include "tl_matrix_utils.h"

#ifdef HAVE_HDF5
#include "TlHdf5Utils.h"
#endif  

TlDenseSymmetricMatrixObject::TlDenseSymmetricMatrixObject(TlDenseMatrix_ImplObject* pImpl)
: pImpl_(pImpl) {
}

TlDenseSymmetricMatrixObject::~TlDenseSymmetricMatrixObject() {
}

void TlDenseSymmetricMatrixObject::vtr2mat(const std::vector<double>& vtr) {
const TlMatrixObject::index_type dim = this->getNumOfRows();
assert(dim == this->getNumOfCols());
assert(static_cast<TlMatrixObject::index_type>(vtr.size()) == dim * (dim + 1) / 2);

std::size_t i = 0;
for (TlMatrixObject::index_type c = 0; c < dim; ++c) {
for (TlMatrixObject::index_type r = 0; r <= c; ++r) {
double v = vtr[i];
this->set(r, c, v);
++i;
}
}
}


TlMatrixObject::index_type TlDenseSymmetricMatrixObject::getNumOfRows() const {
return this->pImpl_->getNumOfRows();
}

TlMatrixObject::index_type TlDenseSymmetricMatrixObject::getNumOfCols() const {
return this->pImpl_->getNumOfCols();
}

void TlDenseSymmetricMatrixObject::resize(const TlMatrixObject::index_type dim) {
this->pImpl_->resize(dim, dim);
}

double TlDenseSymmetricMatrixObject::get(const TlMatrixObject::index_type row,
const TlMatrixObject::index_type col) const {
return this->pImpl_->get(row, col);
}

void TlDenseSymmetricMatrixObject::set(const TlMatrixObject::index_type row, const TlMatrixObject::index_type col,
const double value) {
this->pImpl_->set(row, col, value);
}

void TlDenseSymmetricMatrixObject::add(const TlMatrixObject::index_type row, const TlMatrixObject::index_type col,
const double value) {
this->pImpl_->add(row, col, value);
}

std::vector<double> TlDenseSymmetricMatrixObject::getRowVector(const TlMatrixObject::index_type row) const {
const TlMatrixObject::index_type size = this->getNumOfCols();
std::vector<double> v(size);
for (TlMatrixObject::index_type i = 0; i < size; ++i) {
v[i] = this->get(row, i);
}

return v;
}

std::vector<double> TlDenseSymmetricMatrixObject::getColVector(const TlMatrixObject::index_type col) const {
const TlMatrixObject::index_type size = this->getNumOfRows();
std::vector<double> v(size);
for (TlMatrixObject::index_type i = 0; i < size; ++i) {
v[i] = this->get(i, col);
}

return v;
}

void TlDenseSymmetricMatrixObject::setRowVector(const TlMatrixObject::index_type row, const std::vector<double>& v) {
const TlMatrixObject::index_type size =
std::min(this->getNumOfCols(), static_cast<TlMatrixObject::index_type>(v.size()));

#pragma omp parallel for
for (TlMatrixObject::index_type i = 0; i < size; ++i) {
this->set(row, i, v[i]);
}
}

void TlDenseSymmetricMatrixObject::setColVector(const TlMatrixObject::index_type col, const std::vector<double>& v) {
const TlMatrixObject::index_type size =
std::min(this->getNumOfRows(), static_cast<TlMatrixObject::index_type>(v.size()));

#pragma omp parallel for
for (TlMatrixObject::index_type i = 0; i < size; ++i) {
this->set(i, col, v[i]);
}
}





double TlDenseSymmetricMatrixObject::sum() const {
return this->pImpl_->sum();
}

double TlDenseSymmetricMatrixObject::trace() const {
return this->pImpl_->trace();
}

double TlDenseSymmetricMatrixObject::getRMS() const {
return this->pImpl_->getRMS();
}

double TlDenseSymmetricMatrixObject::getMaxAbsoluteElement(TlMatrixObject::index_type* outRow,
TlMatrixObject::index_type* outCol) const {
return this->pImpl_->getMaxAbsoluteElement(outRow, outCol);
}

void TlDenseSymmetricMatrixObject::pivotedCholeskyDecomposition(TlDenseGeneralMatrixObject* pL,
const double threshold) const {
struct argmax_pivot {
std::size_t operator()(const std::vector<double>& diagonals, const std::vector<int>& pivot,
const int pivotBegin) {
std::size_t maxPivotIndex = pivotBegin;
double maxVal = 0.0;
const std::size_t end = pivot.size();
for (std::size_t pivotIndex = pivotBegin; pivotIndex < end; ++pivotIndex) {
std::size_t diagonal_index = pivot[pivotIndex];
assert(diagonal_index < diagonals.size());
const double v = diagonals[pivot[pivotIndex]];
if (maxVal < v) {
maxVal = v;
maxPivotIndex = pivotIndex;
}
}

return maxPivotIndex;
};
};

struct accumulate {
double operator()(const std::vector<double>& diagonals, const std::vector<int>& pivot, const int pivotBegin) {
double sum = 0.0;
const std::size_t end = pivot.size();
for (std::size_t pivotIndex = pivotBegin; pivotIndex < end; ++pivotIndex) {
sum += diagonals[pivot[pivotIndex]];
}

return sum;
};
};


const TlMatrixObject::index_type N = this->getNumOfRows();
assert(N == this->getNumOfCols());

std::vector<double> diagonals(N);
double error = 0.0;
std::vector<TlMatrixObject::index_type> pivot(N);
for (TlMatrixObject::index_type i = 0; i < N; ++i) {
const double v = this->get(i, i);
diagonals[i] = v;
pivot[i] = i;
}
error = accumulate()(diagonals, pivot, 0);

pL->resize(N, N);
TlMatrixObject::index_type m = 0;
while (error > threshold) {
const TlMatrixObject::index_type argmax = argmax_pivot()(diagonals, pivot, m);
std::swap(pivot[m], pivot[argmax]);

assert(diagonals[pivot[m]] >= 0.0);
const double l_m_pm = std::sqrt(diagonals[pivot[m]]);
pL->set(m, pivot[m], l_m_pm);

const double inv_l_m_pm = 1.0 / l_m_pm;

for (TlMatrixObject::index_type i = m + 1; i < N; ++i) {
double sum_ll = 0.0;
for (TlMatrixObject::index_type j = 0; j < m; ++j) {
sum_ll += pL->get(j, pivot[m]) * pL->get(j, pivot[i]);
}
const double l_m_pi = (this->get(pivot[m], pivot[i]) - sum_ll) * inv_l_m_pm;
pL->set(m, pivot[i], l_m_pi);

diagonals[pivot[i]] -= l_m_pi * l_m_pi;
}

error = accumulate()(diagonals, pivot, m);
++m;
}

pL->resize(m, N);
pL->transposeInPlace();
}

bool TlDenseSymmetricMatrixObject::load(const std::string& filePath) {
bool answer = false;
TlMatrixObject::HeaderInfo headerInfo;

const bool isLoadable = TlMatrixUtils::getHeaderInfo(filePath, &headerInfo);
const std::size_t headerSize = headerInfo.headerSize;
if (isLoadable == true) {
const TlMatrixObject::index_type row = headerInfo.numOfRows;
const TlMatrixObject::index_type col = headerInfo.numOfCols;

if (row != col) {
this->log_.critical(TlUtils::format("illegal format: @%s.%d", __FILE__, __LINE__));
}
this->resize(row);

std::fstream fs;
fs.open(filePath.c_str(), std::ios::in | std::ios::binary);
if (!fs.fail()) {
fs.seekg(headerSize);

switch (headerInfo.matrixType) {
case TlMatrixObject::RLHD: {
double v;
for (TlMatrixObject::index_type r = 0; r < row; ++r) {
for (TlMatrixObject::index_type c = 0; c <= r; ++c) {
fs.read(reinterpret_cast<char*>(&v), sizeof(double));
this->set(r, c, v);
}
}
answer = true;
} break;

default:
this->log_.critical(TlUtils::format("not supported format: @%s:%d", __FILE__, __LINE__));
break;
}
} else {
this->log_.critical(TlUtils::format("cannot open matrix file: %s @%s:%d", filePath.c_str(), __FILE__, __LINE__));
throw;
}

fs.close();
} else {
this->log_.critical(TlUtils::format("illegal matrix format: %s @%s:%d", filePath.c_str(), __FILE__, __LINE__));
throw;
}

return answer;
}

bool TlDenseSymmetricMatrixObject::save(const std::string& filePath) const {
bool answer = false;
std::ofstream fs;
fs.open(filePath.c_str(), std::ofstream::out | std::ofstream::binary);
if (!fs.fail()) {
const char nType = static_cast<char>(TlMatrixObject::RLHD);
const TlMatrixObject::index_type dim = this->getNumOfRows();

fs.write(&nType, sizeof(char));
fs.write(reinterpret_cast<const char*>(&dim), sizeof(TlMatrixObject::index_type));
fs.write(reinterpret_cast<const char*>(&dim), sizeof(TlMatrixObject::index_type));

for (TlMatrixObject::index_type r = 0; r < dim; ++r) {
for (TlMatrixObject::index_type c = 0; c <= r; ++c) {
const double v = this->get(r, c);
fs.write(reinterpret_cast<const char*>(&v), sizeof(double));
}
}
fs.flush();
answer = true;
} else {
this->log_.critical(TlUtils::format("cannot write matrix: %s @%s:%d", filePath.c_str(), __FILE__, __LINE__));
}
fs.close();

return answer;
}

bool TlDenseSymmetricMatrixObject::saveText(const std::string& filePath) const {
bool answer = true;

std::ofstream ofs;
ofs.open(filePath.c_str(), std::ofstream::out);

if (ofs.good()) {
this->saveText(ofs);
}

ofs.close();

return answer;
}

void TlDenseSymmetricMatrixObject::saveText(std::ostream& os) const {
const TlMatrixObject::index_type rows = this->getNumOfRows();
const TlMatrixObject::index_type cols = this->getNumOfCols();

os << "TEXT\n";
os << rows << "\n";
os << cols << "\n";

for (TlMatrixObject::index_type i = 0; i < rows; ++i) {
for (TlMatrixObject::index_type j = 0; j < cols; ++j) {
os << TlUtils::format(" %10.6lf", this->get(i, j));
}
os << "\n";
}
os << "\n";
}

bool TlDenseSymmetricMatrixObject::saveCsv(const std::string& filePath) const {
bool answer = true;

std::ofstream ofs;
ofs.open(filePath.c_str(), std::ofstream::out);

if (ofs.good()) {
this->saveCsv(ofs);
}

ofs.close();

return answer;
}

void TlDenseSymmetricMatrixObject::saveCsv(std::ostream& os) const {
const TlMatrixObject::index_type rows = this->getNumOfRows();
const TlMatrixObject::index_type cols = this->getNumOfCols();

for (TlMatrixObject::index_type i = 0; i < rows; ++i) {
for (TlMatrixObject::index_type j = 0; j < cols - 1; ++j) {
os << TlUtils::format(" %16.10e, ", this->get(i, j));
}
os << TlUtils::format(" %16.10e", this->get(i, cols - 1));
os << "\n";
}
os << "\n";
}

#ifdef HAVE_HDF5
bool TlDenseSymmetricMatrixObject::loadHdf5(const std::string& filepath, const std::string& h5path) {
TlHdf5Utils h5(filepath);

int mat_type;
h5.getAttr(h5path, "type", &mat_type);

index_type row = 0;
index_type col = 0;
h5.getAttr(h5path, "row", &row);
h5.getAttr(h5path, "col", &col);
if (row != col) {
this->log_.critical(
TlUtils::format("illegal parameter: row(%d) != col(%d) @%s:%d", row, col, __FILE__, __LINE__));
}
this->resize(row);
const TlMatrixObject::index_type dim = row;

const std::size_t numOfElements = this->getNumOfElements_RLHD();
std::vector<double> buf(numOfElements);
h5.get(h5path, &(buf[0]), numOfElements);

switch (mat_type) {
case RLHD: {
std::size_t i = 0;
for (TlMatrixObject::index_type r = 0; r < dim; ++r) {
for (TlMatrixObject::index_type c = 0; c <= r; ++c) {
this->set(r, c, buf[i]);
i++;
}
}
} break;
case CUHD: {
std::size_t i = 0;
for (TlMatrixObject::index_type c = 0; c < dim; ++c) {
for (TlMatrixObject::index_type r = 0; r <= c; ++r) {
this->set(r, c, buf[i]);
i++;
}
}
} break;

default:
this->log_.critical(
TlUtils::format("illegal matrix type for TlDenseSymmetricMatrix_BLAS_Old: %d", mat_type));
break;
}

return true;
}

bool TlDenseSymmetricMatrixObject::saveHdf5(const std::string& filepath, const std::string& h5path) const {
TlHdf5Utils h5(filepath);

const index_type row = this->getNumOfRows();
const index_type col = this->getNumOfCols();
const TlMatrixObject::index_type dim = row;
assert(dim == col);

{
const std::size_t numOfElements = this->getNumOfElements_RLHD();
std::vector<double> buf(numOfElements);
std::size_t i = 0;
for (TlMatrixObject::index_type r = 0; r < dim; ++r) {
for (TlMatrixObject::index_type c = 0; c <= r; ++c) {
buf[i] = this->get(r, c);
i++;
}
}

h5.write(h5path, &(buf[0]), numOfElements);
}

h5.setAttr(h5path, "type", static_cast<int>(TlMatrixObject::RLHD));
h5.setAttr(h5path, "row", row);
h5.setAttr(h5path, "col", col);

return true;
}

#endif  

void TlDenseSymmetricMatrixObject::loadSerializeData(const TlSerializeData& data) {
const TlMatrixObject::index_type row = std::max(data["row"].getInt(), 1);
const TlMatrixObject::index_type col = std::max(data["col"].getInt(), 1);
assert(row == col);
this->resize(row);

size_type index = 0;
const TlMatrixObject::index_type maxRow = this->getNumOfRows();
for (index_type row = 0; row < maxRow; ++row) {
for (index_type col = 0; col <= row; ++col) {
this->set(row, col, data["data"].getAt(index).getDouble());
++index;
}
}
}

TlSerializeData TlDenseSymmetricMatrixObject::getSerializeData() const {
TlSerializeData data;
data["row"] = this->getNumOfRows();
data["col"] = this->getNumOfCols();
data["type"] = "RLHD";

TlSerializeData tmp;
const TlMatrixObject::index_type maxRow = this->getNumOfRows();
for (index_type row = 0; row < maxRow; ++row) {
for (index_type col = 0; col <= row; ++col) {
tmp.pushBack(this->get(row, col));
}
}
data["data"] = tmp;

return data;
}

std::ostream& operator<<(std::ostream& stream, const TlDenseSymmetricMatrixObject& mat) {
const TlMatrixObject::index_type nNumOfDim = mat.getNumOfRows();  

stream << "\n\n";
for (TlMatrixObject::index_type ord = 0; ord < nNumOfDim; ord += 10) {
stream << "       ";
for (TlMatrixObject::index_type j = ord; ((j < ord + 10) && (j < nNumOfDim)); ++j) {
stream << TlUtils::format("   %5d th", j + 1);
}
stream << "\n"
<< " ----";

for (TlMatrixObject::index_type j = ord; ((j < ord + 10) && (j < nNumOfDim)); ++j) {
stream << "-----------";
}
stream << "----\n";

for (TlMatrixObject::index_type i = 0; i < nNumOfDim; ++i) {
stream << TlUtils::format(" %5d  ", i + 1);

for (TlMatrixObject::index_type j = ord; ((j < ord + 10) && (j < nNumOfDim)); ++j) {
if (j > i) {
stream << "    ----   ";
} else {
stream << TlUtils::format(" %10.6lf", mat.get(i, j));
}
}
stream << "\n";
}
stream << "\n\n";
}

return stream;
}
