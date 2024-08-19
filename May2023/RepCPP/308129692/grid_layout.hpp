#pragma once
#include <costa/grid2grid/block.hpp>
#include <costa/grid2grid/grid2D.hpp>
#include <costa/grid2grid/mpi_type_wrapper.hpp>
#include <mpi.h>

namespace costa {
template <typename T>
class grid_layout {
public:
grid_layout() = default;

grid_layout(assigned_grid2D &&g, local_blocks<T> &&b, char ordering)
: grid(std::forward<assigned_grid2D>(g))
, blocks(std::forward<local_blocks<T>>(b))
{
this->ordering = std::toupper(ordering);
assert(this->ordering == 'R' || this->ordering == 'C');

for (size_t i = 0; i < blocks.num_blocks(); ++i) {
blocks.get_block(i).set_ordering(this->ordering);
}
}

int num_ranks() const { return grid.num_ranks(); }

void transpose() {
grid.transpose();
blocks.transpose();
}

void reorder_ranks(std::vector<int>& reordering) {
grid.reorder_ranks(reordering);
}

int reordered_rank(int rank) const {
return grid.reordered_rank(rank);
}

bool ranks_reordered() {
return grid.ranks_reordered();
}

int num_cols() const noexcept { return grid.num_cols(); }
int num_rows() const noexcept { return grid.num_rows(); }

int num_blocks_col() const noexcept { return grid.num_blocks_col(); }
int num_blocks_row() const noexcept { return grid.num_blocks_row(); }

void scale_by(const T beta) {
if (beta == T{1}) return;
for (unsigned i = 0u; i < blocks.num_blocks(); ++i) {
auto& block = blocks.get_block(i);
block.scale_by(beta);
}
}

template <typename Function>
void initialize(Function f) {
for (size_t i = 0; i < blocks.num_blocks(); ++i) {
auto& b = blocks.get_block(i);

for (int li = 0; li < b.n_rows(); ++li) {
for (int lj = 0; lj < b.n_cols(); ++lj) {
int gi, gj;
std::tie(gi, gj) = b.local_to_global(li, lj);
assert(gi >= 0 && gj >= 0);
assert(gi < num_rows() && gj < num_cols());
b.local_element(li, lj) = (T) f(gi, gj);

}
}
}
}

template <typename Function>
void apply(Function f) {
for (size_t i = 0; i < blocks.num_blocks(); ++i) {
auto& b = blocks.get_block(i);

for (int li = 0; li < b.n_rows(); ++li) {
for (int lj = 0; lj < b.n_cols(); ++lj) {
int gi, gj;
std::tie(gi, gj) = b.local_to_global(li, lj);
assert(gi >= 0 && gj >= 0);
assert(gi < num_rows() && gj < num_cols());
auto prev_value = b.local_element(li, lj);
b.local_element(li, lj) = (T) f(gi, gj, prev_value);

}
}
}
}

template <typename Function>
bool validate(Function f, double tolerance = 1e-12) {
bool ok = true;

for (size_t i = 0; i < blocks.num_blocks(); ++i) {
block<T> b = blocks.get_block(i);

for (int li = 0; li < b.n_rows(); ++li) {
for (int lj = 0; lj < b.n_cols(); ++lj) {
int gi, gj;
std::tie(gi, gj) = b.local_to_global(li, lj);
assert(gi >= 0 && gj >= 0);
assert(gi < num_rows() && gj < num_cols());
auto diff = std::abs(b.local_element(li, lj) - (T)f(gi, gj));
if (diff > tolerance) { 
std::cout << "[ERROR] mat(" << gi << ", " << gj << ")"
<<  " = " << b.local_element(li, lj)
<< " instead of " << (T) f(gi, gj)
<< std::endl;
ok = false;
}
}
}
}
return ok;
}

template <typename Function>
T accumulate(Function f, T initial_value) {
T result = initial_value;

for (size_t i = 0; i < blocks.num_blocks(); ++i) {
auto& b = blocks.get_block(i);
for (int li = 0; li < b.n_rows(); ++li) {
for (int lj = 0; lj < b.n_cols(); ++lj) {
auto el = b.local_element(li, lj);
result = f(result, el);
}
}
}

return result;
}

assigned_grid2D grid;
local_blocks<T> blocks;
char ordering = 'C';
};

template <typename T>
using layout_ref = std::reference_wrapper<grid_layout<T>>;

} 
