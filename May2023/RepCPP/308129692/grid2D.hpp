#pragma once
#include <costa/grid2grid/interval.hpp>

#include <vector>

namespace costa {

struct grid2D {
int n_rows = 0;
int n_cols = 0;
std::vector<int> rows_split;
std::vector<int> cols_split;

grid2D() = default;
grid2D(std::vector<int> &&r_split, std::vector<int> &&c_split);

interval row_interval(int index) const;

interval col_interval(int index) const;

void transpose();
};


class assigned_grid2D {
public:
assigned_grid2D() = default;
assigned_grid2D(grid2D &&g,
std::vector<std::vector<int>> &&proc,
int n_ranks);

int owner(int i, int j) const;

const grid2D &grid() const;

int num_ranks() const;

interval rows_interval(int index) const;

interval cols_interval(int index) const;

int block_size(int row_index, int col_index);

void transpose();

void reorder_ranks(std::vector<int>& reordering);

int reordered_rank(int rank) const;

bool ranks_reordered() const;

friend std::ostream &operator<<(std::ostream &os, const assigned_grid2D &other) {
for (int i = 0; i < other.num_blocks_row(); ++i) {
for (int j = 0; j < other.num_blocks_col(); ++j) {
os << "block " << other.rows_interval(i) << " x " << other.cols_interval(j) <<  " is owned by rank " 
<< other.owner(i, j) << std::endl;
}
}
return os;
}

int num_blocks_row() const noexcept { 
return g.n_rows;
}

int num_blocks_col() const noexcept {
return g.n_cols;
}

int num_rows() const noexcept { 
return g.rows_split.back();
}

int num_cols() const noexcept { 
return g.cols_split.back();
}

private:
friend bool operator==(assigned_grid2D const &,
assigned_grid2D const &) noexcept;

bool transposed = false;

grid2D g;
std::vector<std::vector<int>> ranks;
int n_ranks = 0;

std::vector<int> ranks_reordering;
};

bool operator==(assigned_grid2D const &, assigned_grid2D const &) noexcept;

} 
