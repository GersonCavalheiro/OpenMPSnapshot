#pragma once

#include <cstdio> 
#include <cstdint> 
#include <cassert> 
#include <algorithm> 
#include <vector> 
#include <type_traits> 

#include "status.hxx" 
#include "print_tools.hxx" 
#include "green_memory.hxx" 
#include "simple_stats.hxx" 


#ifdef  DEBUG
#define debug_printf(...) std::printf(__VA_ARGS__)
#else  
#define debug_printf(...)
#endif 

namespace green_sparse {

double const MByte = 1e-6; char const *const _MByte = "MByte";

char const int_t_names[][10] = {"uint8_t", "uint16_t", "uint32_t", "uint64_t", "uint128_t"};

template <typename int_t>
char const * int_t_name() {
int const b = sizeof(int_t); 
int const log2b = (b < 2) ? 0 : ((b < 4) ? 1 : ((b < 8) ? 2 : ((b < 16) ? 3 : 4 ))); 
return int_t_names[log2b] + std::is_signed<int_t>::value; 
} 

template <typename ColIndex_t=uint32_t, typename RowIndex_t=uint32_t>
class sparse_t { 
public:

static bool constexpr row_signed = std::is_signed<RowIndex_t>::value;
static bool constexpr col_signed = std::is_signed<ColIndex_t>::value;

sparse_t() : rowStart_(nullptr), colIndex_(nullptr), rowIndex_(nullptr), nRows_(0) {
debug_printf("# sparse_t default constructor\n");
} 

sparse_t(
std::vector<std::vector<ColIndex_t>> const & list
, bool const with_rowIndex=false  
, char const *matrix_name=nullptr 
, int const echo=0 
) {
auto const name = matrix_name ? matrix_name : "constructor";
if (echo > 7) std::printf("# sparse_t<ColIndex_t=%s,RowIndex_t=%s> %s\n", int_t_name<ColIndex_t>(), int_t_name<RowIndex_t>(), name);
nRows_ = list.size();
assert(list.size() == nRows_ && "RowIndex_t cannot hold number of rows");
auto const rowStart_nc = get_memory<RowIndex_t>(nRows_ + 1, echo, "rowStart_"); 
assert(rowStart_nc && "sparse_t failed to get_memory for rowStart");
size_t nnz{0}; 
simple_stats::Stats<> st;
rowStart_nc[0] = 0;
for (RowIndex_t iRow = 0; iRow < nRows_; ++iRow) {
auto const n = list[iRow].size(); 
st.add(n);
rowStart_nc[iRow + 1] = rowStart_nc[iRow] + n; 
nnz += n;
} 
rowStart_ = rowStart_nc;
assert(nnz == rowStart_[nRows_] && "Counting error");

rowIndex_ = with_rowIndex ? rowIndex() : nullptr;

auto const colIndex_nc = get_memory<ColIndex_t>(nnz, echo, "colIndex_"); 
assert(rowStart_ && "sparse_t failed to get_memory for colIndex");
for (RowIndex_t iRow = 0; iRow < nRows_; ++iRow) {
auto const n = list[iRow].size(); 
assert(rowStart_[iRow] + n == rowStart_[iRow + 1]);
for (size_t j = 0; j < n; ++j) {
auto const jnz = rowStart_[iRow] + j;
colIndex_nc[jnz] = list[iRow][j];
} 
} 
colIndex_ = colIndex_nc;

if (echo > 7) std::printf("# sparse_t constructed with %d rows and %ld non-zero elements\n", uint32_t(nRows_), size_t(nnz));
if (echo > 5) std::printf("# sparse_t %s columns per row stats [%g, %g +/- %g, %g]\n", name, st.min(), st.mean(), st.dev(), st.max());
if (echo > 9) { 
std::printf("# sparse_t.rowStart"); if (echo > 19) std::printf("(%p)", (void*)rowStart_); 
std::printf("[0..%d]=", nRows_);  printf_vector(" %d", rowStart_, nRows_ + 1);
std::printf("# sparse_t.colIndex"); if (echo > 19) std::printf("(%p)", (void*)colIndex_); 
std::printf("[0..%ld-1]= ", nnz); printf_vector(" %d", colIndex_, nnz);
} 
} 

~sparse_t() {
debug_printf("# sparse_t destructor\n");
if (rowStart_) free_memory(rowStart_);
if (colIndex_) free_memory(colIndex_);
if (rowIndex_) free_memory(rowIndex_);
nRows_ = 0;
} 

sparse_t(sparse_t && rhs)                   = delete; 
sparse_t(sparse_t const & rhs)              = delete; 
sparse_t & operator= (sparse_t       & rhs) = delete; 
sparse_t & operator= (sparse_t const & rhs) = delete; 

sparse_t & operator= (sparse_t && rhs) {
debug_printf("# sparse_t move assignment sparse_t<ColIndex_t=%s,RowIndex_t=%s>\n", int_t_name<ColIndex_t>(), int_t_name<RowIndex_t>());
rowStart_ = rhs.rowStart_; rhs.rowStart_ = nullptr;
colIndex_ = rhs.colIndex_; rhs.colIndex_ = nullptr;
rowIndex_ = rhs.rowIndex_; rhs.rowIndex_ = nullptr;
nRows_    = rhs.nRows_;    rhs.nRows_    = 0;
return *this;
} 

__host__ __device__ RowIndex_t const * rowStart() const { return rowStart_; };
__host__ __device__ ColIndex_t const * colIndex() const { return colIndex_; };
__host__ __device__ RowIndex_t            nRows() const { return nRows_; }
__host__ __device__ RowIndex_t        nNonzeros() const { return (row_signed && nRows_ < 0) ? 0 : (rowStart_ ? rowStart_[nRows_] : 0); }

private:

__host__ bool invalid_row_index_(RowIndex_t const iRow) const {
if (iRow >= nRows_)  return true;  
else if (iRow < 0)   return true;  
else                 return false; 
} 

public:

#if 0 

RowIndex_t nNonzeroCols(RowIndex_t const iRow) const {
if (invalid_row_index_(iRow)) return 0;
return rowStart_ ? rowStart_[iRow] : 0;
} 

ColIndex_t const * nonzeroCols(RowIndex_t const iRow) const {
if (invalid_row_index_(iRow)) return nullptr;
auto const row_start = rowStart_ ? rowStart_[iRow] : 0;
return colIndex_ ? &colIndex_[row_start] : nullptr;
} 

#endif 

__host__ bool is_in(RowIndex_t const iRow, ColIndex_t const jCol, RowIndex_t *index=nullptr) const {
if (invalid_row_index_(iRow)) return false;
if (nullptr == rowStart_ || nullptr == colIndex_) return false;
for (auto jnz = rowStart_[iRow]; jnz < rowStart_[iRow]; ++jnz) {
if (jCol == colIndex_[jnz]) {
if (index) *index = jnz; 
return true;
} 
} 
return false;
} 

__host__ RowIndex_t const * rowIndex() { 
if (nullptr == rowIndex_) {
auto const nnz = nNonzeros();
if (nnz < 1) return nullptr;
auto const rowIndex_nc = get_memory<RowIndex_t>(nnz); 
for (RowIndex_t iRow = 0; iRow < nRows_; ++iRow) {
for (auto jnz = rowStart_[iRow]; jnz < rowStart_[iRow + 1]; ++jnz) {
rowIndex_nc[jnz] = iRow;
} 
} 
rowIndex_ = rowIndex_nc;
} 
return rowIndex_;
} 

private: 
RowIndex_t const *rowStart_;
ColIndex_t const *colIndex_;
RowIndex_t const *rowIndex_; 
RowIndex_t nRows_;

}; 



#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_sparse(int const echo=0, int const n=7) {
sparse_t<> s0; 
sparse_t<int> sint; 
sparse_t<int,int> sintint; 
std::vector<std::vector<uint8_t>> list(n);
for (int i = 0; i < n; ++i) {
list[i].resize(i + 1); 
for (int j = 0; j <= i; ++j) list[i][j] = j;
} 
sparse_t<uint8_t> s(list, false, "lowerTriangularMatrix", echo); 
if (echo > 3) std::printf("# %s nRows= %d, nNonzeros= %d\n", __func__, s.nRows(), s.nNonzeros());
if (echo > 6) std::printf("# %s last element = %d\n", __func__, int(s.colIndex()[s.nNonzeros() - 1]));
if (echo > 6) { std::printf("# rowIndex "); printf_vector(" %d", s.rowIndex(), s.nNonzeros()); }
if (echo > 3) std::printf("# sizeof(sparse_t) = %ld Byte\n", sizeof(sparse_t<>));
return 0;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_sparse(echo);
return stat;
} 

#endif 

} 

#ifdef DEBUG
#undef DEBUG
#endif 

#undef debug_printf
