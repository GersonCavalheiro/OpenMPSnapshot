#include "GridUtils.h"
#include "GridFind.h"

#include <stdexcept>

void make_grid(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,	
heap_darray<rr_uint>& grid,
heap_darray<rr_uint>& cells_start_in_grid) 
{
printlog_debug(__func__)();

static heap_darray<rr_uint> unsorted_grid(params.maxn);

cells_start_in_grid.fill(0);

for (rr_uint i = 0; i < ntotal; ++i) {
rr_uint cell_idx = get_cell_idx(r(i));
unsorted_grid(i) = cell_idx;

if (params.enable_check_consistency) {
if (cell_idx >= params.max_cells) {
printlog("cell_idx: ")(cell_idx)();
printlog("max_cells: ")(params.max_cells)();
printlog("particle_idx: ")(i)();
printlog("x: ")(r(i).x)(" -> (")(params.x_mingeom)(";")(params.x_maxgeom)(")")();
printlog("y: ")(r(i).y)(" -> (")(params.y_mingeom)(";")(params.y_maxgeom)(")")();
throw std::runtime_error{ "cell_idx was >= params.max_cells" };
}
}
cells_start_in_grid(cell_idx)++;
}

for (rr_uint i = 1; i < params.max_cells; ++i) {
cells_start_in_grid(i) += cells_start_in_grid(i - 1ull);
}

#pragma omp parallel for
for (rr_iter i = ntotal; i > 0; --i) {
rr_uint j = i - 1;

rr_uint cell_idx = unsorted_grid(j);
grid(cells_start_in_grid(cell_idx) - 1ull) = j;
cells_start_in_grid(cell_idx)--;
}
}

void find_neighbours(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,
const heap_darray<rr_uint>& grid,
const heap_darray<rr_uint>& cell_starts_in_grid,
heap_darray_md<rr_uint>& neighbours) 
{
printlog_debug(__func__)();

const rr_float max_dist = sqr(grid_cell_size());

bool err = false;

#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; j++) { 
rr_uint neighbour_id = 0;
rr_uint center_cell_idx = get_cell_idx(r(j));

rr_uint neighbour_cells[9];
get_neighbouring_cells(center_cell_idx, neighbour_cells);
for (rr_uint cell_i = 0; cell_i < 9; ++cell_i) { 
rr_uint cell_idx = neighbour_cells[cell_i];
if (cell_idx == GRID_INVALID_CELL) continue; 

for (rr_uint grid_i = cell_starts_in_grid(cell_idx); 
grid_i < cell_starts_in_grid(cell_idx + 1ull);
++grid_i)
{
rr_uint i = grid(grid_i); 
if (i == j) continue; 

rr_float2 diff = r(i) - r(j);
rr_float dist_sqr = length_sqr(diff);

if (dist_sqr < max_dist) {
if (params.enable_check_consistency) {
if (neighbour_id == params.max_neighbours - 1) {
#pragma omp critical
{
printlog("neighbour_id: ")(neighbour_id)(" / ")(params.max_neighbours)();
printlog("j: ")(j)(" / ")(ntotal)();
printlog("x: ")(r(j).x)();
printlog("y: ")(r(j).y)();
printlog("cell: ")(center_cell_idx)();
printlog("cell_x: ")(get_cell_x(center_cell_idx))();
printlog("cell_y: ")(get_cell_y(center_cell_idx))();
err = true;
}
--neighbour_id;
}
}
neighbours(neighbour_id, j) = i;
++neighbour_id;
}
} 
} 

rr_uint n = std::min(neighbour_id, params.max_neighbours - 1);
neighbours(n, j) = ntotal;
} 


if (err) {
throw std::runtime_error{ "making neighbours grid: error occured" };
}
}

void grid_find(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,
heap_darray_md<rr_uint>& neighbours) 
{
printlog_debug(__func__)();

static heap_darray<rr_uint> grid(params.maxn);
static heap_darray<rr_uint> cell_starts_in_grid(params.max_cells);

make_grid(ntotal, 
r, 
grid, 
cell_starts_in_grid);

find_neighbours(ntotal,
r,
grid,
cell_starts_in_grid,
neighbours);
}
