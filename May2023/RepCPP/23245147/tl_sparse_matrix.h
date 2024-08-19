
#ifndef TLSPARSEMATRIX_H
#define TLSPARSEMATRIX_H

#ifdef HAVE_CONFIG_H
#include "config.h"  
#endif               

#include <cassert>
#include <functional>
#include <iostream>

#ifdef HAVE_UNORDERED_MAP
#include <unordered_map>
#elifdef HAVE_TR1_UNORDERED_MAP
#include <tr1/unordered_map>
#elifdef HAVE_GOOGLE_SPARSE_HASH_MAP
#include <google/sparse_hash_map>
#else
#include <map>
#endif

#include "TlUtils.h"
#include "tl_dense_vector_lapack.h"
#include "tl_matrix_object.h"

struct TlMatrixElement {
typedef TlMatrixObject::index_type index_type;

public:
index_type row;
index_type col;
double value;
};

class TlSparseMatrix : public TlMatrixObject {
public:
explicit TlSparseMatrix(index_type row = 1, index_type col = 1);

TlSparseMatrix(const TlSparseMatrix& rhs);

virtual ~TlSparseMatrix();

public:
struct Index2 {
public:
Index2(index_type r = 0, index_type c = 0) : row(r), col(c) {}

bool operator<(const Index2& rhs) const {
bool answer = false;
if ((this->row < rhs.row) ||
((this->row == rhs.row) && (this->col < rhs.col))) {
answer = true;
}

return answer;
}

bool operator>(const Index2& rhs) const {
bool answer = false;
if ((this->row > rhs.row) ||
((this->row == rhs.row) && (this->col > rhs.col))) {
answer = true;
}

return answer;
}

bool operator==(const Index2& rhs) const {
return ((this->row == rhs.row) && (this->col == rhs.col));
}

bool operator!=(const Index2& rhs) const {
return !(this->operator==(rhs));
}

public:
index_type row;
index_type col;
};

public:
typedef Index2 KeyType;

typedef std::map<KeyType, double> SparseMatrixData;

typedef SparseMatrixData::const_iterator const_iterator;
typedef SparseMatrixData::iterator iterator;

public:
const_iterator begin() const { return m_aMatrix.begin(); }

iterator begin() { return m_aMatrix.begin(); }

const_iterator end() const { return m_aMatrix.end(); }

iterator end() { return m_aMatrix.end(); }

virtual void clear();

virtual void zeroClear() { this->clear(); }

virtual void erase(index_type row, index_type col);
virtual void erase(iterator p);

virtual index_type getNumOfRows() const;

virtual index_type getNumOfCols() const;

virtual size_type getSize() const;

virtual std::size_t getMemSize() const;

virtual void resize(index_type row, index_type col);

double pop(index_type* pRow, index_type* pCol);

virtual double get(index_type row, index_type col) const;


virtual void set(const index_type row, const index_type col,
const double value);

virtual void add(const index_type row, const index_type col,
const double value);

void add(const std::vector<MatrixElement>& elements);


virtual bool hasKey(index_type row, index_type col) {
return (this->m_aMatrix.find(KeyType(row, col)) !=
this->m_aMatrix.end());
}

virtual void merge(const TlSparseMatrix& rhs);

TlSparseMatrix& operator=(const TlSparseMatrix& rhs);

TlSparseMatrix& operator*=(const double& rhs);

TlSparseMatrix& operator/=(const double& rhs);

virtual TlDenseVector_Lapack getRowVector(index_type row) const;

virtual TlDenseVector_Lapack getColVector(index_type col) const;

const TlSparseMatrix& dot(const TlSparseMatrix& X);

double sum() const;

std::vector<int> getRowIndexList() const;
std::vector<int> getColIndexList() const;

std::vector<MatrixElement> getMatrixElements() const;

public:
template <typename T>
void print(T& out) const;

public:

public:
virtual bool load(const std::string& path);
virtual bool save(const std::string& path) const;

protected:
bool load(std::ifstream& ifs);
bool save(std::ofstream& ofs) const;

protected:
index_type m_nRows;                  
index_type m_nCols;                  
mutable SparseMatrixData m_aMatrix;  

friend class TlCommunicate;
};


inline void TlSparseMatrix::set(const index_type row, const index_type col,
const double value) {
assert((0 <= row) && (row < this->getNumOfRows()));
assert((0 <= col) && (col < this->getNumOfCols()));

#pragma omp critical(TlSparseMatrix__set)
{ this->m_aMatrix[KeyType(row, col)] = value; }
}

inline void TlSparseMatrix::add(const index_type row, const index_type col,
const double value) {
assert((0 <= row) && (row < this->getNumOfRows()));
assert((0 <= col) && (col < this->getNumOfCols()));

#pragma omp critical(TlSparseMatrix__add)
{ this->m_aMatrix[KeyType(row, col)] += value; }
}

inline double TlSparseMatrix::get(const index_type row,
const index_type col) const {
assert((0 <= row) && (row < this->m_nRows));
assert((0 <= col) && (col < this->m_nCols));

double answer = 0.0;
const_iterator p = this->m_aMatrix.find(KeyType(row, col));
if (p != this->m_aMatrix.end()) {
answer = p->second;
}

return answer;
}

template <typename T>
void TlSparseMatrix::print(T& out) const {
const int nNumOfRows = this->getNumOfRows();
const int nNumOfCols = this->getNumOfCols();

for (int ord = 0; ord < nNumOfCols; ord += 10) {
out << "       ";
for (int j = ord; j < ord + 10 && j < nNumOfCols; j++) {
out << TlUtils::format("   %5d th", j + 1);
}
out << "\n ----";

for (int j = ord; ((j < ord + 10) && (j < nNumOfCols)); j++) {
out << "-----------";
}
out << "----\n";

for (int i = 0; i < nNumOfRows; i++) {
out << TlUtils::format(" %5d  ", i + 1);

for (int j = ord; ((j < ord + 10) && (j < nNumOfCols)); j++) {
out << TlUtils::format(" %10.6lf", this->get(i, j));
}
out << "\n";
}
out << "\n\n";
}
out.flush();
}

#endif  
