#ifndef _CSRMatrix_hpp_
#define _CSRMatrix_hpp_


#include <cstddef>
#include <vector>
#include <algorithm>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace miniFE {

template<typename Scalar,
typename LocalOrdinal,
typename GlobalOrdinal>
struct
CSRMatrix {
CSRMatrix()
: has_local_indices(false),
rows(), row_offsets(), row_offsets_external(),
packed_cols(), packed_coefs(),
num_cols(0)
#ifdef HAVE_MPI
,external_index(), external_local_index(), elements_to_send(),
neighbors(), recv_length(), send_length(), send_buffer(), request()
#endif
{
}

~CSRMatrix()
{}

typedef Scalar        ScalarType;
typedef LocalOrdinal  LocalOrdinalType;
typedef GlobalOrdinal GlobalOrdinalType;

bool                       has_local_indices;
std::vector<GlobalOrdinal> rows;
std::vector<LocalOrdinal>  row_offsets;
std::vector<LocalOrdinal>  row_offsets_external;
std::vector<GlobalOrdinal> packed_cols;
std::vector<Scalar>        packed_coefs;
LocalOrdinal               num_cols;

#ifdef HAVE_MPI
std::vector<GlobalOrdinal> external_index;
std::vector<GlobalOrdinal>  external_local_index;
std::vector<GlobalOrdinal> elements_to_send;
std::vector<int>           neighbors;
std::vector<LocalOrdinal>  recv_length;
std::vector<LocalOrdinal>  send_length;
std::vector<Scalar>        send_buffer;
std::vector<MPI_Request>   request;
#endif

size_t num_nonzeros() const
{
return row_offsets[row_offsets.size()-1];
}

void reserve_space(unsigned nrows, unsigned ncols_per_row)
{
rows.resize(nrows);
row_offsets.resize(nrows+1);
packed_cols.reserve(nrows * ncols_per_row);
packed_coefs.reserve(nrows * ncols_per_row);

#pragma omp parallel for
for(MINIFE_GLOBAL_ORDINAL i = 0; i < nrows; ++i) {
rows[i] = 0;
row_offsets[i] = 0;
}

#pragma omp parallel for
for(MINIFE_GLOBAL_ORDINAL i = 0; i < (nrows * ncols_per_row); ++i) {
packed_cols[i] = 0;
packed_coefs[i] = 0;
}
}

void get_row_pointers(GlobalOrdinalType row, size_t& row_length,
GlobalOrdinalType*& cols,
ScalarType*& coefs)
{
ptrdiff_t local_row = -1;
if (rows.size() >= 1) {
ptrdiff_t idx = row - rows[0];
if (idx < rows.size() && rows[idx] == row) {
local_row = idx;
}
}

if (local_row == -1) {
typename std::vector<GlobalOrdinal>::iterator row_iter =
std::lower_bound(rows.begin(), rows.end(), row);

if (row_iter == rows.end() || *row_iter != row) {
row_length = 0;
return;
}

local_row = row_iter - rows.begin();
}

LocalOrdinalType offset = row_offsets[local_row];
row_length = row_offsets[local_row+1] - offset;
cols = &packed_cols[offset];
coefs = &packed_coefs[offset];
}
};

}

#endif

