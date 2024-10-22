#ifndef TL_DENSE_GENERAL_MATRIX_OBJECT_H
#define TL_DENSE_GENERAL_MATRIX_OBJECT_H

#include <cassert>
#include <valarray>

#include "tl_dense_matrix_impl_object.h"
#include "tl_dense_vector_object.h"
#include "tl_matrix_object.h"

class TlDenseGeneralMatrixObject : public TlMatrixObject {
public:
TlDenseGeneralMatrixObject(TlDenseMatrix_ImplObject* pImpl = NULL);
virtual ~TlDenseGeneralMatrixObject();

public:
virtual TlMatrixObject::index_type getNumOfRows() const;
virtual TlMatrixObject::index_type getNumOfCols() const;
virtual void resize(const TlMatrixObject::index_type row, const TlMatrixObject::index_type col);

virtual double get(const TlMatrixObject::index_type row, const TlMatrixObject::index_type col) const;
virtual void set(const TlMatrixObject::index_type row, const TlMatrixObject::index_type col, const double value);
virtual void add(const TlMatrixObject::index_type row, const TlMatrixObject::index_type col, const double value);

void block(const TlMatrixObject::index_type row, const TlMatrixObject::index_type col,
const TlMatrixObject::index_type rowDistance, const TlMatrixObject::index_type colDistance,
TlDenseGeneralMatrixObject* pOut) const;

void block(const TlMatrixObject::index_type row, const TlMatrixObject::index_type col,
const TlDenseGeneralMatrixObject& ref);

virtual std::vector<double> getRowVector(const TlMatrixObject::index_type row) const;
virtual std::vector<double> getColVector(const TlMatrixObject::index_type col) const;
virtual void setRowVector(const TlMatrixObject::index_type row, const std::vector<double>& v);
virtual void setColVector(const TlMatrixObject::index_type row, const std::vector<double>& v);

template <class VectorType>
VectorType getRowVector_tmpl(const TlMatrixObject::index_type row) const {
const TlMatrixObject::index_type size = this->getNumOfCols();
VectorType v(size);
#pragma omp parallel for
for (TlMatrixObject::index_type i = 0; i < size; ++i) {
v.set(i, this->get(row, i));
}

return v;
}

template <class VectorType>
VectorType getColVector_tmpl(const TlMatrixObject::index_type col) const {
const TlMatrixObject::index_type size = this->getNumOfRows();
VectorType v(size);
#pragma omp parallel for
for (TlMatrixObject::index_type i = 0; i < size; ++i) {
v.set(i, this->get(i, col));
}

return v;
}

virtual TlMatrixObject::index_type setRowVector(const TlMatrixObject::index_type row, const double* p,
const TlMatrixObject::index_type length) {
const index_type copiedLength = std::min(length, this->getNumOfCols());
#pragma omp parallel for
for (index_type i = 0; i < copiedLength; ++i) {
this->set(row, i, p[i]);
}

return copiedLength;
}

virtual TlMatrixObject::index_type setColVector(const TlMatrixObject::index_type col, const double* p,
const TlMatrixObject::index_type length) {
const index_type copiedLength = std::min(length, this->getNumOfRows());
#pragma omp parallel for
for (index_type i = 0; i < copiedLength; ++i) {
this->set(i, col, p[i]);
}

return copiedLength;
}

virtual std::vector<double> diagonals() const;
virtual double sum() const;
virtual double trace() const;
virtual double getRMS() const;
virtual double getMaxAbsoluteElement(TlMatrixObject::index_type* outRow = NULL,
TlMatrixObject::index_type* outCol = NULL) const;
virtual void transposeInPlace();

virtual bool load(const std::string& filePath);
virtual bool save(const std::string& filePath) const;

virtual bool saveText(const std::string& filePath) const;
virtual void saveText(std::ostream& os) const;

virtual bool saveCsv(const std::string& filePath) const;
virtual void saveCsv(std::ostream& os) const;

#ifdef HAVE_HDF5
virtual bool loadHdf5(const std::string& filepath, const std::string& h5path);
virtual bool saveHdf5(const std::string& filepath, const std::string& h5path) const;
#endif  

virtual void dump(double* buf, const std::size_t size) const;
virtual void restore(const double* buf, const std::size_t size);

protected:
TlDenseMatrix_ImplObject* pImpl_;
};

std::ostream& operator<<(std::ostream& stream, const TlDenseGeneralMatrixObject& mat);

#endif  
