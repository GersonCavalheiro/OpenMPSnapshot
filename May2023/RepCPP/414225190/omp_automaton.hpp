
#ifndef PARALLEL_CELLULAR_AUTOMATA_OMPL_AUTOMATON_HPP
#define PARALLEL_CELLULAR_AUTOMATA_OMPL_AUTOMATON_HPP
#include <cstddef>
#include <functional>
#include <grid.hpp>
#include <iostream>
#include <tuple>

namespace ca
{

namespace omp
{

template <typename T>
class CellularAutomaton
{
public:

CellularAutomaton(ca::Grid<T> &grid, std::function<T(T, T, T, T, T, T, T, T, T)> update_function,
unsigned workers = 0)
: grid(grid), generation(0), update_function(update_function)
{
nw = (workers) ? workers : std::thread::hardware_concurrency();
};


CellularAutomaton(CellularAutomaton &&other)
{
grid = std::move(other.grid);
update_function = std::move(other.update_function);
}


CellularAutomaton(const CellularAutomaton &other) = delete;


virtual void simulate(unsigned steps = 1)
{
if (steps == 0)
return;
Grid<T> new_grid = ca::Grid<T>::newWithSameSize(grid);

while (steps > 0)
{
#pragma omp parallel for collapse(2) num_threads(nw)
for (size_t r = 0; r < grid.rows(); ++r)
{
for (size_t c = 0; c < grid.columns(); ++c)
{
auto cell = std::make_tuple(grid(r, c));
new_grid(r, c) = std::apply(update_function, std::tuple_cat(cell, get_neighborhood(r, c)));
}
}
grid.swap(new_grid);
++generation;
--steps;
}
}


size_t get_generation() const
{
return generation;
}

friend std::ostream &operator<<(std::ostream &os, const CellularAutomaton &ca)
{

return os << ca.grid;
}

protected:

Grid<T> &grid;


size_t generation;

std::function<T(T, T, T, T, T, T, T, T, T)> update_function;


unsigned nw;


virtual std::tuple<T, T, T, T, T, T, T, T> get_neighborhood(int row, int col) const

{
unsigned rows = grid.rows();
unsigned columns = grid.columns();
T top_left, top, top_right, left, right, bottom_left, bottom, bottom_right;
top_left = grid((row - 1 + rows) % rows, (col - 1 + columns) % columns);
top = grid((row - 1 + rows) % rows, col);
top_right = grid((row - 1 + rows) % rows, (col + 1) % columns);
left = grid(row, (col - 1 + columns) % columns);
right = grid(row, (col + 1) % columns);
bottom_left = grid((row + 1) % rows, (col - 1 + columns) % columns);
bottom = grid((row + 1) % rows, col);
bottom_right = grid((row + 1) % rows, (col + 1) % columns);
return std::make_tuple(top_left, top, top_right, left, right, bottom_left, bottom, bottom_right);
};
};
} 
} 

#endif