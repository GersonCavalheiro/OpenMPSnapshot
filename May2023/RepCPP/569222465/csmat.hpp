#pragma once

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <vector>

template <typename T>
using DenseMatrix = std::vector<T>;


enum class CsKind {
Csr,
Csc,
};


template <typename T>
class CsMat {
public:
~CsMat() = default;

static auto new_csr(size_t ncols, size_t nrows, DenseMatrix<T> const& mtx) -> CsMat<T> {
std::vector<T> data;
std::vector<size_t> indices;
std::vector<size_t> indptr(nrows + 1, 0);

for (size_t i = 0; i < nrows; ++i) {
for (size_t j = 0; j < ncols; ++j) {
if (mtx[i * ncols + j] != 0.0) {
indices.push_back(j);
data.push_back(mtx[i * ncols + j]);
}
}
indptr[i + 1] = indices.size();
}

return CsMat(CsKind::Csr, ncols, nrows, data, indices, indptr);
}

static auto csr_parse(std::string filename) {
std::ifstream file(filename);
if (!file.is_open()) {
throw std::runtime_error("Error: could not open file");
}
std::string line;

size_t line_count = 0;
size_t n = 0;
size_t ncols = 0;
size_t nrows = 0;

std::vector<double> data;
std::vector<size_t> cols;
std::vector<size_t> rows;
double tmp;
size_t tmp2;

while (std::getline(file, line)) {
std::istringstream iss(line);

switch (line_count) {
case 0:
iss >> n >> ncols >> nrows;
data.resize(n);
cols.resize(ncols);
rows.resize(nrows);
break;

case 1:
for (size_t i = 0; i < n; ++i) {
iss >> tmp;
data[i] = tmp;
}
break;
case 2:
for (size_t i = 0; i < ncols; ++i) {
iss >> tmp2;
cols[i] = tmp2;
}
break;
case 3:
for (size_t i = 0; i < nrows; ++i) {
iss >> tmp2;
rows[i] = tmp2;
}
break;
}
line_count++;
}

return CsMat(CsKind::Csr, ncols, nrows, data, cols, rows);
}

static auto new_csc(size_t ncols, size_t nrows, DenseMatrix<T> const& mtx) -> CsMat {
std::vector<T> data;
std::vector<size_t> indices;
std::vector<size_t> indptr(ncols + 1, 0);

for (size_t j = 0; j < ncols; ++j) {
for (size_t i = 0; i < nrows; ++i) {
if (mtx[i * ncols + j] != 0.0) {
indices.push_back(i);
data.push_back(mtx[i * ncols + j]);
}
}
indptr[j + 1] = indices.size();
}

return CsMat(CsKind::Csc, ncols, nrows, data, indices, indptr);
}

[[nodiscard]] auto transposed() const -> CsMat<T> {
auto transposed_indptr = std::vector<size_t>(get_inner_dim() + 1, 0);
auto transposed_indices = std::vector<size_t>(get_nnz());
auto transposed_data = std::vector<T>(get_nnz());

auto const& outer_indptr = get_indptr();
auto const& outer_indices = get_indices();
auto const& outer_data = get_data();

for (size_t i = 0; i < get_nnz(); ++i) {
++transposed_indptr[outer_indices[i] + 1];
}

for (size_t i = 0; i < get_inner_dim(); ++i) {
transposed_indptr[i + 1] += transposed_indptr[i];
}

for (size_t i = 0; i < get_outer_dim(); ++i) {
for (size_t j = outer_indptr[i]; j < outer_indptr[i + 1]; ++j) {
auto col = outer_indices[j];
auto dest = transposed_indptr[col];
transposed_indices[dest] = i;
transposed_data[dest] = outer_data[j];
++transposed_indptr[col];
}
}

for (int i = get_inner_dim() - 1; i > 0; --i) {
transposed_indptr[i] = transposed_indptr[i - 1];
}
transposed_indptr[0] = 0;

return CsMat<T>(
this->storage,
this->nrows,
this->ncols,
transposed_data,
transposed_indices,
transposed_indptr
);
}

[[nodiscard]] inline auto get_storage() const -> CsKind {
return this->storage;
};

[[nodiscard]] inline auto get_ncols() const -> size_t {
return this->ncols;
};

[[nodiscard]] inline auto get_nrows() const -> size_t {
return this->nrows;
};

[[nodiscard]] inline auto get_nnz() const -> size_t {
return this->data.size();
}

[[nodiscard]] inline auto get_outer_dim() const -> size_t {
return this->storage == CsKind::Csr ? this->nrows : this->ncols;
}

[[nodiscard]] inline auto get_inner_dim() const -> size_t {
return this->storage == CsKind::Csc ? this->nrows : this->ncols;
}

[[nodiscard]] inline auto get_data() const -> std::vector<T> const& {
return this->data;
};

[[nodiscard]] inline auto get_indices() const -> std::vector<size_t> const& {
return this->indices;
};

[[nodiscard]] inline auto get_indptr() const -> std::vector<size_t> const& {
return this->indptr;
};

[[nodiscard]] inline auto get_mut_storage() -> CsKind& {
return this->storage;
};

[[nodiscard]] inline auto get_mut_ncols() -> size_t& {
return this->ncols;
};

[[nodiscard]] inline auto get_mut_nrows() -> size_t& {
return this->nrows;
};

[[nodiscard]] inline auto get_mut_data() -> std::vector<T>& {
return this->data;
};

[[nodiscard]] inline auto get_mut_indices() -> std::vector<size_t>& {
return this->indices;
};

[[nodiscard]] inline auto get_mut_indptr() -> std::vector<size_t>& {
return this->indptr;
};

auto inline operator==(CsMat const& rhs) const -> bool {
assert(this->data.size() == rhs.data.size());
assert(this->indices.size() == rhs.indices.size());
assert(this->indptr.size() == rhs.indptr.size());

bool is_eq = true;

for (size_t i = 0; i < this->data.size(); ++i) {
if (this->data[i] != rhs.data[i]) {
is_eq = false;
}
}
if (is_eq != true) { return false; }

for (size_t i = 0; i < this->indices.size(); ++i) {
if (this->indices[i] != rhs.indices[i]) {
is_eq = false;
}
}
if (is_eq != true) { return false; }

for (size_t i = 0; i < this->indptr.size(); ++i) {
if (this->indptr[i] != rhs.indptr[i]) {
is_eq = false;
}
}
if (is_eq != true) { return false; }

return true;
}

friend auto operator<<(std::ostream& stream, CsMat const& mat) -> std::ostream& {
stream << "CsMat { storage: " << mat.get_storage_as_str();
stream << ", ncols: " << mat.ncols << ", nrows: " << mat.nrows;
stream << ", data: [";
for (auto val : mat.data) {
stream << val << ", ";
}
stream << "\b\b], indices: [";
for (auto val : mat.indices) {
stream << val << ", ";
}
stream << "\b\b], indptr: [";
for (auto val : mat.indptr) {
stream << val << ", ";
}
stream << "\b\b] }";
return stream;
}

CsMat() : storage(CsKind::Csr), ncols(0), nrows(0) {}

private:
CsKind storage;
size_t ncols;
size_t nrows;
std::vector<T> data;
std::vector<size_t> indices;
std::vector<size_t> indptr;

CsMat(
CsKind storage,
size_t ncols,
size_t nrows,
std::vector<T> const& data,
std::vector<size_t> const& indices,
std::vector<size_t> const& indptr
) : storage(storage),
ncols(ncols),
nrows(nrows),
data(data),
indices(indices),
indptr(indptr)
{}

[[nodiscard]] inline auto get_storage_as_str() const -> std::string {
switch (this->storage) {
case CsKind::Csr:
return "CSR";
case CsKind::Csc:
return "CSC";
default:
return "unknown";
}
};
};
