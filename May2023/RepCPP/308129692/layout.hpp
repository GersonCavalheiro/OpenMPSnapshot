#pragma once
#include <costa/grid2grid/grid_layout.hpp>
#include <costa/grid2grid/scalapack_layout.hpp>

namespace costa {

struct block_t {
void *data;
int ld;
int row;
int col;
};


template <typename T>
grid_layout<T> custom_layout(const int rowblocks,
const int colblocks,
const int* rowsplit,
const int* colsplit,
const int* owners,
const int nlocalblocks,
const block_t* localblocks,
const char ordering);

assigned_grid2D custom_grid(const int rowblocks,
const int colblocks,
const int* rowsplit,
const int* colsplit,
const int* owners);


template <typename T>
grid_layout<T> block_cyclic_layout(
const int m, const int n, 
const int block_m, const int block_n, 
const int i, const int j, 
const int sub_m, const int sub_n, 
const int p_m, const int p_n, 
const char order, 
const int rsrc, const int csrc, 
T* ptr, 
const int lld, 
const char ordering, 
const int rank 
);

assigned_grid2D block_cyclic_grid(
const int m, const int n, 
const int block_m, const int block_n, 
const int i, const int j, 
const int sub_m, const int sub_n, 
const int proc_m, const int proc_n, 
const char rank_grid_ordering, 
const int rsrc, const int csrc 
);
}
