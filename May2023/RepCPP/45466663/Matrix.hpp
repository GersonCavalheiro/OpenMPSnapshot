#pragma once





#include <stdexcept>
#include <vector>


template <typename T>
class Matrix
{
public:


using value_type    = T;
using iterator      = typename std::vector<T>::iterator;
using const_iterator= typename std::vector<T>::const_iterator;


Matrix()
: rows_(0)
, cols_(0)
, data_()
{}

Matrix (size_t rows, size_t cols)
: rows_(rows)
, cols_(cols)
, data_(rows * cols)
{}

Matrix (size_t rows, size_t cols, T init)
: rows_(rows)
, cols_(cols)
, data_(rows * cols, init)
{}

Matrix (size_t rows, size_t cols, std::initializer_list<T> const& init_list)
: rows_(rows)
, cols_(cols)
, data_(rows * cols)
{
if (init_list.size() != size()) {
throw std::length_error("__FUNCTION__: length_error");
}

size_t i = 0;
for (T const& v : init_list) {
data_[i] = v;
++i;
}
}

~Matrix() = default;

Matrix(Matrix const&) = default;
Matrix(Matrix&&)      = default;

Matrix& operator= (Matrix const&) = default;
Matrix& operator= (Matrix&&)      = default;

void swap (Matrix& other)
{
using std::swap;
swap(rows_, other.rows_);
swap(cols_, other.cols_);
swap(data_, other.data_);
}


size_t rows() const
{
return rows_;
}

size_t cols() const
{
return cols_;
}

size_t size() const
{
return rows_ * cols_;
}


inline size_t coord(const size_t row, const size_t col) const
{
return row * cols_ + col;
}

T& at (const size_t row, const size_t col)
{
if (row >= rows_ || col >= cols_) {
throw std::out_of_range("__FUNCTION__: out_of_range");
}
return data_[row * cols_ + col];
}

const T at (const size_t row, const size_t col) const
{
if (row >= rows_ || col >= cols_) {
throw std::out_of_range("__FUNCTION__: out_of_range");
}
return data_[row * cols_ + col];
}

T& operator () (const size_t row, const size_t col)
{
return data_[row * cols_ + col];
}

inline const T operator () (const size_t row, const size_t col) const
{
return data_[row * cols_ + col];
}

const std::vector<T>& get_array() const
{
return data_;
}


std::vector<T> row( size_t index ) const
{
if( index >= rows_ ) {
throw std::out_of_range("__FUNCTION__: out_of_range");
}

auto result = std::vector<T>( cols() );
for( size_t i = 0; i < cols(); ++i ) {
result[i] = operator()( index, i );
}
return result;
}

std::vector<T> col( size_t index ) const
{
if( index >= cols_ ) {
throw std::out_of_range("__FUNCTION__: out_of_range");
}

auto result = std::vector<T>( rows() );
for( size_t i = 0; i < rows(); ++i ) {
result[i] = operator()( i, index );
}
return result;
}


iterator begin()
{
return data_.begin();
}

iterator end()
{
return data_.end();
}

const_iterator begin() const
{
return data_.begin();
}

const_iterator end() const
{
return data_.end();
}

const_iterator cbegin() const
{
return data_.cbegin();
}

const_iterator cend() const
{
return data_.cend();
}


bool operator == (const Matrix<T>& rhs) const
{
return rows_ == rhs.rows_
&& cols_ == rhs.cols_
&& data_ == rhs.data_;
}

bool operator != (const Matrix<T>& rhs) const
{
return !(*this == rhs);
}


private:

size_t         rows_;
size_t         cols_;
std::vector<T> data_;
};
