#define ROW_MAJOR_IDX(i, j) ((i) * (nbr_of_col)) + (j)

#include <hwloc.h>

#if defined(PRINT_PERF) or not defined(NDEBUG)
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <string_view>
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

#include <GASPI.h>

#include <boost/program_options.hpp>

#include <success_or_die.hh>
#include <up_down.hh>

constexpr size_t col_block_size = 64;
constexpr size_t row_block_size = col_block_size;

#if defined(_OPENMP)
#include <cstdlib>
#include <omp.h>
#endif

typedef const enum {
ID_PARAMS,
ID_DOMAIN,
ID_DOMAIN_TMP,
} SEGMENT_ID;

typedef const enum {
ARG_NBR_OF_COLUMN,
ARG_NBR_OF_ROW,
ARG_ENERGY_INIT,
ARG_NBR_ITERS,
ARG_OMPTHREAD_NBR,
} PARAMS_ARGS;


void wait_if_queue_full(const gaspi_queue_id_t queue_id,
const gaspi_number_t request_size)
{
gaspi_number_t queue_size_max;
gaspi_number_t queue_size;

SUCCESS_OR_DIE(gaspi_queue_size_max(&queue_size_max));
SUCCESS_OR_DIE(gaspi_queue_size(queue_id, &queue_size));

if (queue_size + request_size >= queue_size_max)
SUCCESS_OR_DIE(gaspi_wait(queue_id, GASPI_BLOCK));
}

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
SUCCESS_OR_DIE(gaspi_proc_init(GASPI_BLOCK));
#if defined(PRINT_PERF)
TimeVar time_checkpoint_start = timeNow();
#endif
gaspi_rank_t iProc, nProc;
hwloc_topology_t topology;
hwloc_cpuset_t cpuset;
hwloc_obj_t obj;
SUCCESS_OR_DIE(gaspi_proc_rank(&iProc));
SUCCESS_OR_DIE(gaspi_proc_num(&nProc));
#if defined(PRINT_PERF)
std::stringstream stream_message_bench;
#if defined(PERF_ON_RANK_0_ONLY)
if (iProc == 0)
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
<< "energy_init,"
<< "FuncName,"
<< "FileName"
<< "\n"
<< std::flush;
#endif
gaspi_segment_id_t segment_id = SEGMENT_ID::ID_DOMAIN;
gaspi_segment_id_t segment_id_tmp = SEGMENT_ID::ID_DOMAIN_TMP;


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

SUCCESS_OR_DIE(gaspi_segment_create(SEGMENT_ID::ID_PARAMS, 5 * sizeof(size_t), GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));
gaspi_pointer_t args_ptr;
SUCCESS_OR_DIE(gaspi_segment_ptr(SEGMENT_ID::ID_PARAMS, &args_ptr));
size_t *args = (size_t *)(args_ptr);

if (iProc == 0)
{
boost::program_options::options_description options("Allowed options");
options.add_options()
("nbr_of_column", boost::program_options::value<std::size_t>()->required(), "nbr_of_column")
("nbr_of_row", boost::program_options::value<std::size_t>()->required(), "nbr_of_row")
("nbr_iters", boost::program_options::value<std::size_t>()->required(), "nbr_iters")
("energy_init", boost::program_options::value<std::size_t>()->required(), "energy_init")
("ompthread_nbr", boost::program_options::value<std::size_t>()->required(), "ompthread_nbr");

boost::program_options::variables_map variables_map_args;
boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), variables_map_args);
boost::program_options::notify(variables_map_args);
if (variables_map_args.count("nbr_of_column"))
args[PARAMS_ARGS::ARG_NBR_OF_COLUMN] = variables_map_args["nbr_of_column"].as<std::size_t>();
if (variables_map_args.count("nbr_of_row"))
args[PARAMS_ARGS::ARG_NBR_OF_ROW] = variables_map_args["nbr_of_row"].as<std::size_t>();
if (variables_map_args.count("energy_init"))
args[PARAMS_ARGS::ARG_ENERGY_INIT] = variables_map_args["energy_init"].as<std::size_t>();
if (variables_map_args.count("nbr_iters"))
args[PARAMS_ARGS::ARG_NBR_ITERS] = variables_map_args["nbr_iters"].as<std::size_t>();
if (variables_map_args.count("ompthread_nbr"))
args[PARAMS_ARGS::ARG_OMPTHREAD_NBR] = variables_map_args["ompthread_nbr"].as<std::size_t>();
for (gaspi_rank_t rank = 1; rank < nProc; rank++)
SUCCESS_OR_DIE(gaspi_passive_send(
SEGMENT_ID::ID_PARAMS,
0,
rank,
5 * sizeof(std::size_t),
GASPI_BLOCK));
}
else
{
gaspi_rank_t sender = 0;
SUCCESS_OR_DIE(gaspi_passive_receive(
SEGMENT_ID::ID_PARAMS,
0,
&sender,
5 * sizeof(size_t),
GASPI_BLOCK));
}
SUCCESS_OR_DIE(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

#define OMP_NUM_THREADS args[PARAMS_ARGS::ARG_OMPTHREAD_NBR]
#if defined(_OPENMP)
#pragma message("OMP_NUM_THREADS")
omp_set_dynamic(0); 
omp_set_num_threads(OMP_NUM_THREADS);
#endif

#define nbr_of_col args[PARAMS_ARGS::ARG_NBR_OF_COLUMN]
#define NBR_OF_ROW args[PARAMS_ARGS::ARG_NBR_OF_ROW]
#define NBR_ITERS args[PARAMS_ARGS::ARG_NBR_ITERS]
#define ENERGY_INIT args[PARAMS_ARGS::ARG_ENERGY_INIT]

const gaspi_size_t nbr_of_row_local = NBR_OF_ROW / nProc;
const gaspi_size_t segment_size =  (nbr_of_col * nbr_of_row_local) +
(2 * nbr_of_col);

#if defined(PRINT_PERF)
#if defined(PERF_ON_RANK_0_ONLY)
if (iProc == 0)
#endif
stream_message_bench << __LINE__ << ','
<< LineInfo[__COUNTER__] << ','
<< duration(timeNow() - time_checkpoint_start) << ','
<< nProc << ','
<< "rank_" + std::to_string(iProc) + "_on_" + std::to_string(nProc - 1) << ','
<< nbr_of_col << ','
<< NBR_OF_ROW << ','
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

SUCCESS_OR_DIE(gaspi_segment_create(segment_id, segment_size * sizeof(double), GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
SUCCESS_OR_DIE(gaspi_segment_create(segment_id_tmp, segment_size * sizeof(double), GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

gaspi_pointer_t array_ptr;
gaspi_pointer_t array_ptr_tmp;


SUCCESS_OR_DIE(gaspi_segment_ptr(segment_id, &array_ptr));
SUCCESS_OR_DIE(gaspi_segment_ptr(segment_id_tmp, &array_ptr_tmp));

double *array = (double *)(array_ptr);
double *array_tmp = (double *)(array_ptr_tmp);

std::fill(array, array + segment_size, ENERGY_INIT);
std::fill(array_tmp, array_tmp + segment_size, ENERGY_INIT);

const gaspi_queue_id_t queue_id = 0;

wait_if_queue_full(queue_id, 1);
const gaspi_offset_t offset_array_north = (nbr_of_col * nbr_of_row_local);
const gaspi_offset_t offset_array_south = offset_array_north + nbr_of_col;
#if defined(PRINT_PERF)
#if defined(PERF_ON_RANK_0_ONLY)
if (iProc == 0)
#endif
stream_message_bench << __LINE__ << ','
<< LineInfo[__COUNTER__] << ','
<< duration(timeNow() - time_checkpoint_start)
<< "\n"
<< std::flush;
#endif
#if not defined(NDEBUG)

std::stringstream stream_message_debug;
stream_message_debug << "my_rank = " << iProc << " on " << nProc << "\n"
<< std::flush;
for (size_t i = 0; i < nbr_of_row_local + 2; ++i)
{
for (size_t j = 0; j < nbr_of_col; ++j)
stream_message_debug << array[ROW_MAJOR_IDX(i, j)] << "\t";
stream_message_debug << "\n"
<< std::flush;
}
stream_message_debug << "\n"
<< std::flush;

#endif

const size_t col_block = (nbr_of_col + col_block_size - 1) / col_block_size;
const size_t row_block = (nbr_of_row_local + row_block_size - 1) / row_block_size;

for (size_t nb = 0; nb < NBR_ITERS; nb++)
{
SUCCESS_OR_DIE(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
if (UP(iProc, nProc) < iProc)
{
SUCCESS_OR_DIE(gaspi_read( segment_id,
0,
UP(iProc, nProc),
segment_id,
offset_array_north * sizeof(double),
nbr_of_col * sizeof(double),
queue_id,
GASPI_BLOCK));
}
if (DOWN(iProc, nProc) > 0)
{
SUCCESS_OR_DIE(gaspi_read( segment_id,
offset_array_south * sizeof(double),
DOWN(iProc, nProc),
segment_id,
nbr_of_col * sizeof(double),
nbr_of_col * sizeof(double),
queue_id,
GASPI_BLOCK));
}






SUCCESS_OR_DIE(gaspi_wait(queue_id, GASPI_BLOCK));





#if not defined(NDEBUG)

for (size_t i = 0; i < nbr_of_row_local + 2; ++i)
{
for (size_t j = 0; j < nbr_of_col; ++j)
stream_message_debug << array[ROW_MAJOR_IDX(i, j)] << "\t";
stream_message_debug << "\n"
<< std::flush;
}
stream_message_debug << "\n"
<< std::flush;

#endif
std::swap(segment_id, segment_id_tmp);
std::swap(array, array_tmp);
}
#if not defined(NDEBUG)

std::ofstream outfile_debug;
outfile_debug.open(("rank_" + std::to_string(iProc) + "_on_" + std::to_string(nProc - 1) + ".txt").c_str(), std::ios::out);
outfile_debug << stream_message_debug.str();
outfile_debug.close();
stream_message_debug.clear();

#endif
#if defined(PRINT_PERF)
#if defined(PERF_ON_RANK_0_ONLY)
if (iProc == 0)
#endif
stream_message_bench << __LINE__ << ','
<< LineInfo[__COUNTER__] << ','
<< duration(timeNow() - time_checkpoint_start)
<< "\n"
<< std::flush;
#endif
SUCCESS_OR_DIE(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
SUCCESS_OR_DIE(gaspi_segment_delete(segment_id));
SUCCESS_OR_DIE(gaspi_segment_delete(segment_id_tmp));
#if defined(PRINT_PERF)
#if defined(PERF_ON_RANK_0_ONLY)
if (iProc == 0)
#endif
{
stream_message_bench << __LINE__ << ','
<< LineInfo[__COUNTER__] << ','
<< duration(timeNow() - time_checkpoint_start)
<< "\n"
<< std::flush;
std::ofstream outfile_bench;
outfile_bench.open(("GPI2-" + std::to_string(NBR_OF_ROW) + "-" + std::to_string(OMP_NUM_THREADS) + "-rank_" + std::to_string(iProc) + "_on_" + std::to_string(nProc - 1) + ".csv").c_str(), std::ios::out);
outfile_bench << stream_message_bench.str();
outfile_bench.close();
stream_message_bench.clear();
}
#endif
hwloc_bitmap_free(set);
hwloc_topology_destroy(topology);
SUCCESS_OR_DIE(gaspi_proc_term(GASPI_BLOCK));
return EXIT_SUCCESS;
}