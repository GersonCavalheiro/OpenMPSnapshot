#ifndef AMGCL_IO_BINARY_HPP
#define AMGCL_IO_BINARY_HPP





#include <vector>
#include <string>
#include <fstream>

#include <amgcl/util.hpp>
#include <amgcl/detail/sort_row.hpp>

namespace amgcl {
namespace io {

template <class T>
bool read(std::ifstream &f, T &val) {
return static_cast<bool>(f.read((char*)&val, sizeof(T)));
}

template <class T>
bool read(std::ifstream &f, std::vector<T> &vec) {
return static_cast<bool>(f.read((char*)&vec[0], sizeof(T) * vec.size()));
}

template <typename IndexType>
IndexType crs_size(const std::string &fname) {
std::ifstream f(fname.c_str(), std::ios::binary);
IndexType n;

precondition(f, "Failed to open matrix file");
precondition(read(f, n), "File I/O error");

return n;
}

template <typename SizeT, typename Ptr, typename Col, typename Val>
void read_crs(
const std::string &fname,
SizeT &n,
std::vector<Ptr> &ptr,
std::vector<Col> &col,
std::vector<Val> &val,
ptrdiff_t row_beg = -1,
ptrdiff_t row_end = -1
)
{
std::ifstream f(fname.c_str(), std::ios::binary);
precondition(f, "Failed to open matrix file");

precondition(read(f, n), "File I/O error");

if (row_beg < 0) row_beg = 0;
if (row_end < 0) row_end = n;

precondition(row_beg >= 0 && row_end <= static_cast<ptrdiff_t>(n),
"Wrong subset of rows is requested");

ptrdiff_t chunk = row_end - row_beg;

ptr.resize(chunk + 1);

size_t ptr_beg = sizeof(SizeT);
f.seekg(ptr_beg + row_beg * sizeof(Ptr));
precondition(read(f, ptr), "File I/O error");

Ptr nnz;
f.seekg(ptr_beg + n * sizeof(Ptr));
precondition(read(f, nnz), "File I/O error");

SizeT nnz_beg = ptr.front();
if (nnz_beg) for(auto &p : ptr) p -= nnz_beg;

col.resize(ptr.back());
val.resize(ptr.back());

size_t col_beg = ptr_beg + (n + 1) * sizeof(Ptr);
f.seekg(col_beg + nnz_beg * sizeof(Col));
precondition(read(f, col), "File I/O error");

f.seekg(col_beg + nnz * sizeof(Col) + nnz_beg * sizeof(Val));
precondition(read(f, val), "File I/O error");

#pragma omp parallel for
for(ptrdiff_t i = 0; i < chunk; ++i) {
Ptr beg = ptr[i];
Ptr end = ptr[i + 1];
amgcl::detail::sort_row(&col[beg], &val[beg], end - beg);
}
}

template <typename SizeT>
void dense_size(const std::string &fname, SizeT &n, SizeT &m) {
std::ifstream f(fname.c_str(), std::ios::binary);
precondition(f, "Failed to open matrix file");

precondition(read(f, n), "File I/O error");
precondition(read(f, m), "File I/O error");
}

template <typename SizeT, typename Val>
void read_dense(const std::string &fname,
SizeT &n, SizeT &m, std::vector<Val> &v,
ptrdiff_t row_beg = -1, ptrdiff_t row_end = -1)
{
std::ifstream f(fname.c_str(), std::ios::binary);
precondition(f, "Failed to open matrix file");

precondition(read(f, n), "File I/O error");
precondition(read(f, m), "File I/O error");

if (row_beg < 0) row_beg = 0;
if (row_end < 0) row_end = n;

precondition(row_beg >= 0 && row_end <= static_cast<ptrdiff_t>(n),
"Wrong subset of rows is requested");

ptrdiff_t chunk = row_end - row_beg;

v.resize(chunk * m);

f.seekg(2 * sizeof(SizeT) + row_beg * m * sizeof(Val));
precondition(read(f, v), "File I/O error");
}

template <class T>
bool write(std::ofstream &f, const T &val) {
return static_cast<bool>(f.write((char*)&val, sizeof(T)));
}

template <class T>
bool write(std::ofstream &f, const std::vector<T> &vec) {
return static_cast<bool>(f.write((char*)&vec[0], sizeof(T) * vec.size()));
}

} 
} 

#endif
