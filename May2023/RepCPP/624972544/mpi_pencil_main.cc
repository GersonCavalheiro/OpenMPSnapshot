#define ROW_MAJOR_IDX(i, j) ((i) * (nbr_of_col)) + (j)

#include <mpi.h>
#include <cstddef>          
#include <initializer_list> 

#include <hwloc.h>
#include <unistd.h> 

#include <boost/program_options.hpp>
#if defined(_OPENMP)
#include <cstdlib>
#include <omp.h>
#endif

#if defined(PRINT_PERF) or not defined(NDEBUG)
#include <iostream>
#include <sstream>
#include <string_view>
#include <chrono>
static constexpr const std::string_view LineInfo[] =
{"Bcast END",
"ALLOCATION END",
"LOOP START",
"EXCHANGE END",
"COMPUTE END",
"LOOP END",
"FREE END"};
typedef std::chrono::high_resolution_clock::time_point TimeVar;
#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()
#endif

#include "success_or_die.h"

constexpr size_t col_block_size = 64;
constexpr size_t row_block_size = col_block_size;

typedef const enum {
UP,
DOWN,
} DIRECTIONS;

typedef const enum {
ARG_NBR_OF_COLUMN,
ARG_NBR_OF_ROW,
ARG_ENERGY_INIT,
ARG_NBR_ITERS,
ARG_OMPTHREAD_NBR,
} PARAMS_ARGS;

inline void compute_vectorized_N(
double *in_host,
double *out_host,
const size_t &nbr_of_col,
const size_t &nbr_of_row,
const size_t &col_block_size, const size_t &row_block_size,
const size_t &col_block, const size_t &row_block,
const std::initializer_list<size_t> col_idx_slice_idx,
const std::initializer_list<size_t> row_idx_slice_idx)
{
#define not_limit_left (col_idx > 0)
#define not_limit_right (col_idx < nbr_of_col - 1)
#define not_limit_up row_idx

const std::initializer_list<size_t>::const_iterator
col_slice_idx = col_idx_slice_idx.begin(),
row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
for (size_t col_block_size_idx = col_slice_idx[0];
col_block_size_idx < col_slice_idx[1];
col_block_size_idx += col_block_size)
for (size_t row_block_size_idx = row_slice_idx[0];
row_block_size_idx < row_slice_idx[1];
row_block_size_idx += row_block_size)
for (size_t row_idx = row_block_size_idx;
(row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
++row_idx)
if (not_limit_up)
for (size_t col_idx = col_block_size_idx;
(col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
if (not_limit_right and not_limit_left)
{
const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
out_host[idx] = 0.125 *
(in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
2 * in_host[idx] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
}

#undef not_limit_left
#undef not_limit_right
#undef not_limit_up
}

inline void compute_vectorized_S(
double *in_host,
double *out_host,
const size_t &nbr_of_col,
const size_t &nbr_of_row,
const size_t &col_block_size, const size_t &row_block_size,
const size_t &col_block, const size_t &row_block,
const std::initializer_list<size_t> col_idx_slice_idx,
const std::initializer_list<size_t> row_idx_slice_idx)
{
#define not_limit_left (col_idx > 0)
#define not_limit_right (col_idx < nbr_of_col - 1)
#define limit_down (row_idx > nbr_of_row - 2)

const std::initializer_list<size_t>::const_iterator
col_slice_idx = col_idx_slice_idx.begin(),
row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
for (size_t col_block_size_idx = col_slice_idx[0];
col_block_size_idx < col_slice_idx[1];
col_block_size_idx += col_block_size)
for (size_t row_block_size_idx = row_slice_idx[0];
row_block_size_idx < row_slice_idx[1];
row_block_size_idx += row_block_size)
for (size_t row_idx = row_block_size_idx;
(row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
++row_idx)
if (not limit_down)
for (size_t col_idx = col_block_size_idx;
(col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
if (not_limit_right and not_limit_left)
{
const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
out_host[idx] = 0.125 *
(in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
2 * in_host[idx] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
}

#undef limit_down
#undef not_limit_left
#undef not_limit_right
}

inline void compute_vectorized_E(
double *in_host,
double *out_host,
const size_t &nbr_of_col,
const size_t &nbr_of_row,
const size_t &col_block_size, const size_t &row_block_size,
const size_t &col_block, const size_t &row_block,
const std::initializer_list<size_t> col_idx_slice_idx,
const std::initializer_list<size_t> row_idx_slice_idx)
{
#define not_limit_right (col_idx < nbr_of_col - 1)

const std::initializer_list<size_t>::const_iterator
col_slice_idx = col_idx_slice_idx.begin(),
row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
for (size_t col_block_size_idx = col_slice_idx[0];
col_block_size_idx < col_slice_idx[1];
col_block_size_idx += col_block_size)
for (size_t row_block_size_idx = row_slice_idx[0];
row_block_size_idx < row_slice_idx[1];
row_block_size_idx += row_block_size)
for (size_t row_idx = row_block_size_idx;
(row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
++row_idx)
for (size_t col_idx = col_block_size_idx;
(col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
if (not_limit_right)
{
const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
out_host[idx] = 0.125 *
(in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
2 * in_host[idx] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
}
#undef not_limit_right
}

inline void compute_vectorized_W(
double *in_host,
double *out_host,
const size_t &nbr_of_col,
const size_t &nbr_of_row,
const size_t &col_block_size, const size_t &row_block_size,
const size_t &col_block, const size_t &row_block,
const std::initializer_list<size_t> col_idx_slice_idx,
const std::initializer_list<size_t> row_idx_slice_idx)
{
#define not_limit_left (col_idx > 0)

const std::initializer_list<size_t>::const_iterator
col_slice_idx = col_idx_slice_idx.begin(),
row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
for (size_t col_block_size_idx = col_slice_idx[0];
col_block_size_idx < col_slice_idx[1];
col_block_size_idx += col_block_size)
for (size_t row_block_size_idx = row_slice_idx[0];
row_block_size_idx < row_slice_idx[1];
row_block_size_idx += row_block_size)
for (size_t row_idx = row_block_size_idx;
(row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
++row_idx)
for (size_t col_idx = col_block_size_idx;
(col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
if (not_limit_left)
{
const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
out_host[idx] = 0.125 *
(in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
2 * in_host[idx] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
}

#undef not_limit_left
}

inline void compute_vectorized_I(
double *in_host,
double *out_host,
const size_t &nbr_of_col,
const size_t &nbr_of_row,
const size_t &col_block_size, const size_t &row_block_size,
const size_t &col_block, const size_t &row_block,
const std::initializer_list<size_t> col_idx_slice_idx,
const std::initializer_list<size_t> row_idx_slice_idx)
{

const std::initializer_list<size_t>::const_iterator
col_slice_idx = col_idx_slice_idx.begin(),
row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
for (size_t col_block_size_idx = col_slice_idx[0];
col_block_size_idx < col_slice_idx[1];
col_block_size_idx += col_block_size)
for (size_t row_block_size_idx = row_slice_idx[0];
row_block_size_idx < row_slice_idx[1];
row_block_size_idx += row_block_size)
for (size_t row_idx = row_block_size_idx;
(row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
++row_idx)
for (size_t col_idx = col_block_size_idx;
(col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
{
const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
out_host[idx] = 0.125 *
(in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
2 * in_host[idx] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
}
}

int main(int argc, char *argv[])
{
int provided = 0;
hwloc_topology_t topology;
hwloc_cpuset_t cpuset;
hwloc_obj_t obj;

SUCCESS_OR_DIE(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
if (provided < MPI_THREAD_FUNNELED)
{
std::cerr << "The threading support level is lesser than that demanded.\n"
<< "Asked = " << MPI_THREAD_FUNNELED << " provided = " << provided
<< std::endl;
MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}
MPI_Comm comm = MPI_COMM_WORLD;
int periods[] = {false};
int dims[] = {0};
int neighbours_ranks[] = { -2,
-2};
int comm_size, my_rank;

size_t args[5];
SUCCESS_OR_DIE(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));
SUCCESS_OR_DIE(MPI_Dims_create(comm_size, 1, dims));
SUCCESS_OR_DIE(MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, true, &comm));
SUCCESS_OR_DIE(MPI_Cart_shift( comm,
0,
1,
&neighbours_ranks[DIRECTIONS::UP],
&neighbours_ranks[DIRECTIONS::DOWN]));
SUCCESS_OR_DIE(MPI_Comm_rank(comm, &my_rank));


hwloc_topology_init(&topology);
hwloc_topology_load(topology);
hwloc_bitmap_t set = hwloc_bitmap_alloc();
int err = hwloc_get_proc_cpubind(topology, getpid(), set, hwloc_cpubind_flags_t::HWLOC_CPUBIND_PROCESS);
if (err)
{
std::cerr << "Error while using hwloc\n"
<< std::endl;
exit(1);
}
hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(topology, hwloc_bitmap_first(set));



SUCCESS_OR_DIE(MPI_Barrier(MPI_COMM_WORLD));
#if defined(PRINT_PERF)
TimeVar time_checkpoint_start = timeNow();
#endif
#if defined(PRINT_PERF)
std::stringstream stream_message_bench;
#if defined(PERF_ON_RANK_0_ONLY)
if (my_rank == 0)
#endif
stream_message_bench << "TimeCheckpointLocation,"
<< "TimeCheckpointInfo,"
<< "TimeCheckpoint,"
<< "ProcessNbr,"
<< "rankNbr_on_rankSize,"
<< "nbr_of_column,"
<< "nbr_of_row,"
<< "nbr_of_row_local,"
<< "segment_size,"
<< "nbr_iters,"
<< "ompthread_nbr,"
<< "init_val,"
<< "FuncName,"
<< "FileName"
<< "\n"
<< std::flush;
#endif
if (my_rank == 0)
{
boost::program_options::options_description options("Allowed options");
options.add_options()
("nbr_of_column", boost::program_options::value<std::size_t>()->required(), "nbr_of_column")
("nbr_of_row", boost::program_options::value<std::size_t>()->required(), "nbr_of_row")
("init_val", boost::program_options::value<std::size_t>()->required(), "init_val")
("nbr_iters", boost::program_options::value<std::size_t>()->required(), "nbr_iters")
("ompthread_nbr", boost::program_options::value<std::size_t>()->required(), "ompthread_nbr");

boost::program_options::variables_map variables_map_args;
boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), variables_map_args);
boost::program_options::notify(variables_map_args);
if (variables_map_args.count("nbr_of_column"))
args[PARAMS_ARGS::ARG_NBR_OF_COLUMN] = variables_map_args["nbr_of_column"].as<std::size_t>();
if (variables_map_args.count("nbr_of_row"))
args[PARAMS_ARGS::ARG_NBR_OF_ROW] = variables_map_args["nbr_of_row"].as<std::size_t>();
if (variables_map_args.count("init_val"))
args[PARAMS_ARGS::ARG_ENERGY_INIT] = variables_map_args["init_val"].as<std::size_t>();
if (variables_map_args.count("nbr_iters"))
args[PARAMS_ARGS::ARG_NBR_ITERS] = variables_map_args["nbr_iters"].as<std::size_t>();
if (variables_map_args.count("ompthread_nbr"))
args[PARAMS_ARGS::ARG_OMPTHREAD_NBR] = variables_map_args["ompthread_nbr"].as<std::size_t>();
SUCCESS_OR_DIE(MPI_Bcast(args, std::size(args), MPI_UNSIGNED_LONG, 0, comm));
}
else
SUCCESS_OR_DIE(MPI_Bcast(args, std::size(args), MPI_UNSIGNED_LONG, 0, comm));

SUCCESS_OR_DIE(MPI_Barrier(MPI_COMM_WORLD));

#define OMP_NUM_THREADS args[PARAMS_ARGS::ARG_OMPTHREAD_NBR]
#if defined(_OPENMP)
#pragma message("OMP_NUM_THREADS")
omp_set_dynamic(0); 
omp_set_num_threads(OMP_NUM_THREADS);
#endif

#define NBR_ITERS args[PARAMS_ARGS::ARG_NBR_ITERS]
#define ENERGY_INIT args[PARAMS_ARGS::ARG_ENERGY_INIT]
const size_t nbr_of_col = args[PARAMS_ARGS::ARG_NBR_OF_COLUMN];
const size_t nbr_of_row = args[PARAMS_ARGS::ARG_NBR_OF_ROW];
int coords[1] = {-2};
MPI_Cart_coords(comm, my_rank, 1, coords);
const size_t nbr_of_row_local = (nbr_of_row + comm_size - 1) / comm_size; 
#define MPI_coords_x coords[0]
#define offset_row (MPI_coords_x * nbr_of_row_local) 
MPI_Win windows[2];
double *windows_buffer[2];
const MPI_Aint segment_size =
( (nbr_of_col * nbr_of_row_local) +
(2 * nbr_of_col));
#if defined(PRINT_PERF)
stream_message_bench << __LINE__ << ','
<< LineInfo[__COUNTER__] << ','
<< duration(timeNow() - time_checkpoint_start) << ','
<< comm_size << ','
<< "rank_" + std::to_string(my_rank) + "_on_" + std::to_string(comm_size - 1) << ','
<< nbr_of_col << ','
<< nbr_of_row << ','
<< nbr_of_row_local << ','
<< segment_size << ','
<< NBR_ITERS << ','
<< args[PARAMS_ARGS::ARG_OMPTHREAD_NBR] << ','
<< ENERGY_INIT << ",\""
<< __PRETTY_FUNCTION__ << "\",\""
<< __FILE__ << '\"'
<< "\n"
<< std::flush;
#endif
SUCCESS_OR_DIE(MPI_Alloc_mem(segment_size * sizeof(double), MPI_INFO_NULL, &windows_buffer[0]));
SUCCESS_OR_DIE(MPI_Alloc_mem(segment_size * sizeof(double), MPI_INFO_NULL, &windows_buffer[1]));

SUCCESS_OR_DIE(MPI_Win_create(windows_buffer[0], segment_size, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &windows[0]));
SUCCESS_OR_DIE(MPI_Win_create(windows_buffer[1], segment_size, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &windows[1]));

MPI_Datatype north_south_type;
SUCCESS_OR_DIE(MPI_Type_contiguous(nbr_of_col - 2, MPI_DOUBLE, &north_south_type));
SUCCESS_OR_DIE(MPI_Type_commit(&north_south_type));
const MPI_Aint offset_array_north = nbr_of_col * (nbr_of_row_local);
const MPI_Aint offset_array_south = nbr_of_col * (nbr_of_row_local + 1) + 1;

std::fill(windows_buffer[0], windows_buffer[0] + segment_size, ENERGY_INIT);
std::fill(windows_buffer[1], windows_buffer[1] + segment_size, ENERGY_INIT);
SUCCESS_OR_DIE(MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[0]));
SUCCESS_OR_DIE(MPI_Win_fence(MPI_MODE_NOPRECEDE, windows[1]));
#if defined(PRINT_PERF)
#if defined(PERF_ON_RANK_0_ONLY)
if (my_rank == 0)
#endif
stream_message_bench << __LINE__ << ','
<< LineInfo[__COUNTER__] << ','
<< duration(timeNow() - time_checkpoint_start)
<< "\n"
<< std::flush;
#endif
#if not defined(NDEBUG)

MPI_Status status;
MPI_File fh;
SUCCESS_OR_DIE(MPI_File_open(MPI_COMM_SELF, ("rank_" + std::to_string(my_rank) + "_on_" + std::to_string(comm_size - 1) + ".txt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh));
std::stringstream stream_message_debug;
stream_message_debug << "my_rank = " << my_rank << " on " << comm_size << "\n"
<< std::flush;
for (size_t i = 0; i < nbr_of_row_local + 2; ++i)
{
for (size_t j = 0; j < nbr_of_col; ++j)
stream_message_debug << windows_buffer[0][ROW_MAJOR_IDX(i, j)] << "\t";
stream_message_debug << "\n"
<< std::flush;
}
stream_message_debug << "\n"
<< std::flush;

#endif
MPI_Request reqs[2];

const size_t col_block = (nbr_of_col + col_block_size - 1) / col_block_size;
const size_t row_block = (nbr_of_row_local + row_block_size - 1) / row_block_size;

for (size_t nb = 0; nb < NBR_ITERS; nb++)
{
#if defined(PRINT_PERF)
#endif

SUCCESS_OR_DIE(MPI_Rget(
&(windows_buffer[0][1]),
1,
north_south_type,
neighbours_ranks[DIRECTIONS::UP],
offset_array_north + 1,
1,
north_south_type,
*windows,
&reqs[0]));
SUCCESS_OR_DIE(MPI_Rget(
&(windows_buffer[0][offset_array_south]),
1,
north_south_type,
neighbours_ranks[DIRECTIONS::DOWN],
nbr_of_col + 1,
1,
north_south_type,
*windows,
&reqs[1]));










SUCCESS_OR_DIE(MPI_Waitall(std::size(windows), reqs, MPI_STATUS_IGNORE));
SUCCESS_OR_DIE(MPI_Win_fence(0, windows[0]));
SUCCESS_OR_DIE(MPI_Win_fence(0, windows[1]));









#if not defined(NDEBUG)

stream_message_debug << "my_rank = " << my_rank << "\n"
<< std::flush;
for (size_t i = 0; i < nbr_of_row_local + 2; ++i)
{
for (size_t j = 0; j < nbr_of_col; ++j)
stream_message_debug << windows_buffer[0][ROW_MAJOR_IDX(i, j)] << "\t";
stream_message_debug << "\n"
<< std::flush;
}
stream_message_debug << "\n"
<< std::flush;

#endif
std::swap(windows_buffer[0], windows_buffer[1]);
std::swap(windows[0], windows[1]);
}

#if defined(PRINT_PERF)
#if defined(PERF_ON_RANK_0_ONLY)
if (my_rank == 0)
#endif
stream_message_bench << __LINE__ << ','
<< LineInfo[__COUNTER__] << ','
<< duration(timeNow() - time_checkpoint_start)
<< "\n"
<< std::flush;
#endif
SUCCESS_OR_DIE(MPI_Win_fence(MPI_MODE_NOSUCCEED, windows[0]));
SUCCESS_OR_DIE(MPI_Win_fence(MPI_MODE_NOSUCCEED, windows[1]));
SUCCESS_OR_DIE(MPI_Free_mem(windows_buffer[0]));
SUCCESS_OR_DIE(MPI_Free_mem(windows_buffer[1]));
SUCCESS_OR_DIE(MPI_Win_free(&windows[0]));
SUCCESS_OR_DIE(MPI_Win_free(&windows[1]));
#if not defined(NDEBUG)

std::string message_debug = stream_message_debug.str();
stream_message_debug.clear();
MPI_Status status_debug;
MPI_File fh_debug;
SUCCESS_OR_DIE(MPI_File_open(MPI_COMM_SELF, ("MPI_rank_" + std::to_string(my_rank) + "_on_" + std::to_string(comm_size - 1) + ".txt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh));
SUCCESS_OR_DIE(MPI_File_write(fh_debug, message_debug.c_str(), message_debug.length(), MPI_CHAR, &status_debug));
SUCCESS_OR_DIE(MPI_File_close(&fh_debug));

#endif
#if defined(PRINT_PERF)
#if defined(PERF_ON_RANK_0_ONLY)
if (my_rank == 0)
#endif
{
stream_message_bench << __LINE__ << ','
<< LineInfo[__COUNTER__] << ','
<< duration(timeNow() - time_checkpoint_start)
<< "\n"
<< std::flush;
std::string message_bench = stream_message_bench.str();
stream_message_bench.clear();
MPI_Status status_bench;
MPI_File fh_bench;
SUCCESS_OR_DIE(MPI_File_open(MPI_COMM_SELF, ("MPI-" + std::to_string(nbr_of_row) + "-" + std::to_string(OMP_NUM_THREADS) + "-rank_" + std::to_string(my_rank) + "_on_" + std::to_string(comm_size - 1) + ".csv").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_bench));
SUCCESS_OR_DIE(MPI_File_write(fh_bench, message_bench.c_str(), message_bench.length(), MPI_CHAR, &status_bench));
SUCCESS_OR_DIE(MPI_File_close(&fh_bench));
}
#endif
hwloc_bitmap_free(set);
hwloc_topology_destroy(topology);
SUCCESS_OR_DIE(MPI_Finalize());
return EXIT_SUCCESS;
}
