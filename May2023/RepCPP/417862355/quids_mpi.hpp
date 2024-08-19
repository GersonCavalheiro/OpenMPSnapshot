

#pragma once

typedef unsigned uint;

#include "quids.hpp"

#include <mpi.h>

#include "utils/mpi_utils.hpp"

#ifndef MIN_EQUALIZE_SIZE
#define MIN_EQUALIZE_SIZE 100
#endif
#ifndef GRANULARITY

#define GRANULARITY 64
#endif
#ifndef EQUALIZE_INBALANCE
#define EQUALIZE_INBALANCE 0.1
#endif
#ifndef MIN_INBALANCE_STEP
#define MIN_INBALANCE_STEP 0.2
#endif

#define MPI_SPECIFIC_SYMBOLIC_ITERATION_MEMORY_SIZE 2*sizeof(PROBA_TYPE) + sizeof(size_t)
#define MPI_SYMBOLIC_ITERATION_MEMORY_SIZE 2*sizeof(PROBA_TYPE) + sizeof(size_t) + sizeof(int)

namespace quids::mpi {
const static MPI_Datatype Proba_MPI_Datatype = utils::get_mpi_datatype((PROBA_TYPE)0);
const static MPI_Datatype mag_MPI_Datatype = utils::get_mpi_datatype((std::complex<PROBA_TYPE>)0);

size_t min_equalize_size = MIN_EQUALIZE_SIZE;
float equalize_inbalance = EQUALIZE_INBALANCE;
float min_equalize_step = MIN_INBALANCE_STEP;
#ifdef EQUALIZE_OBJECTS
bool equalize_children = false;
#else
bool equalize_children = true;
#endif

typedef class mpi_iteration mpi_it_t;
typedef class mpi_symbolic_iteration mpi_sy_it_t;

class mpi_iteration : public quids::iteration {
public:
PROBA_TYPE node_total_proba = 0;

mpi_iteration() {}

mpi_iteration(char* object_begin_, char* object_end_) : quids::iteration(object_begin_, object_end_) {}


size_t get_total_num_object(MPI_Comm communicator) const {

size_t total_num_object;
MPI_Allreduce(&num_object, &total_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

return total_num_object;
}

size_t get_total_num_symbolic_object(MPI_Comm communicator) const {
size_t total_num_child = get_num_symbolic_object();
MPI_Allreduce(MPI_IN_PLACE, &total_num_child, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
return total_num_child;
}

PROBA_TYPE average_value(const quids::observable_t observable) const {
return node_total_proba == 0 ? 0 : quids::iteration::average_value(observable) / node_total_proba * total_proba;
}

PROBA_TYPE average_value(const quids::observable_t 	observable, MPI_Comm communicator) const {

PROBA_TYPE avg = quids::iteration::average_value(observable);


MPI_Allreduce(MPI_IN_PLACE, &avg, 1, Proba_MPI_Datatype, MPI_SUM, communicator);
return avg;
}

void send_objects(size_t num_object_sent, int node, MPI_Comm communicator, bool send_num_child=false) {
const static size_t max_int = 1 << 31 - 1;


MPI_Send(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node, 0 , communicator);
if (num_object_sent == 0)
return;
size_t begin = num_object - num_object_sent;
size_t send_object_size = object_begin[num_object] - object_begin[begin];
MPI_Send(&send_object_size, 1, MPI_UNSIGNED_LONG_LONG, node, 0 , communicator);


bool send;
MPI_Recv(&send, 1, MPI_UNSIGNED_LONG_LONG, node, 0 , communicator, MPI_STATUS_IGNORE);

if (send) {

size_t send_object_begin = object_begin[begin];
#pragma omp parallel for 
for (size_t i = begin + 1; i <= num_object; ++i)
object_begin[i] -= send_object_begin;


MPI_Send(&magnitude[begin], num_object_sent, mag_MPI_Datatype, node, 0 , communicator);
MPI_Send(&object_begin[begin + 1], num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 , communicator);
MPI_Send(&object_size[begin], num_object_sent, MPI_UNSIGNED, node, 0 , communicator);


size_t send_object_size = object_begin[num_object];
while (send_object_size > max_int) {
MPI_Send(&objects[send_object_begin], max_int, MPI_CHAR, node, 0 , communicator);

send_object_size -= max_int;
send_object_begin += max_int;
}

MPI_Send(&objects[send_object_begin], send_object_size, MPI_CHAR, node, 0 , communicator);

if (send_num_child)

MPI_Send(&num_childs[begin], num_object_sent, MPI_UNSIGNED, node, 0 , communicator);


pop(num_object_sent, false);
}
}

void receive_objects(int node, MPI_Comm communicator, bool receive_num_child=false, size_t max_mem=-1) {
const static size_t max_int = 1 << 31 - 1;


size_t num_object_sent, send_object_size;
MPI_Recv(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node, 0 , communicator, MPI_STATUS_IGNORE);
if (num_object_sent == 0)
return;
MPI_Recv(&send_object_size, 1, MPI_UNSIGNED_LONG_LONG, node, 0 , communicator, MPI_STATUS_IGNORE);


static const size_t iteration_memory_size = ITERATION_MEMORY_SIZE;
bool recv = num_object_sent*iteration_memory_size + send_object_size < max_mem;
MPI_Send(&recv, 1, MPI_CHAR, node, 0 , communicator);

if (recv) {

size_t send_object_begin = object_begin[num_object];
resize(num_object + num_object_sent);
allocate(send_object_begin + send_object_size);


MPI_Recv(&magnitude[num_object], num_object_sent, mag_MPI_Datatype, node, 0 , communicator, MPI_STATUS_IGNORE);
MPI_Recv(&object_begin[num_object + 1], num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 , communicator, MPI_STATUS_IGNORE);
MPI_Recv(&object_size[num_object], num_object_sent, MPI_UNSIGNED, node, 0 , communicator, MPI_STATUS_IGNORE);


size_t object_offset = send_object_begin;
while (send_object_size > max_int) {
MPI_Recv(&objects[send_object_begin], max_int, MPI_CHAR, node, 0 , communicator, MPI_STATUS_IGNORE);

send_object_size -= max_int;
send_object_begin += max_int;
}

MPI_Recv(&objects[send_object_begin], send_object_size, MPI_CHAR, node, 0 , communicator, MPI_STATUS_IGNORE);


#pragma omp parallel for 
for (size_t i = num_object + 1; i <= num_object + num_object_sent; ++i)
object_begin[i] += object_offset;

if (receive_num_child) {

MPI_Recv(&num_childs[num_object], num_object_sent, MPI_UNSIGNED, node, 0 , communicator, MPI_STATUS_IGNORE);


num_childs[num_object] += child_begin[num_object];
__gnu_parallel::partial_sum(num_childs.begin() + num_object, num_childs.begin() + num_object + num_object_sent, child_begin.begin() + num_object + 1);
num_childs[num_object] -= child_begin[num_object];
}

num_object += num_object_sent;
}
}


void equalize(MPI_Comm communicator);

void distribute_objects(MPI_Comm communicator, int node_id);

void gather_objects(MPI_Comm communicator, int node_id);

private:
friend mpi_symbolic_iteration;
friend void inline simulate(mpi_it_t &iteration, quids::rule_t const *rule, mpi_it_t &next_iteration, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object, quids::debug_t mid_step_function);

void equalize_symbolic(MPI_Comm communicator);
void normalize(MPI_Comm communicator, quids::debug_t mid_step_function=[](const char*){});




size_t inline get_mem_size(MPI_Comm communicator) const {
static const size_t iteration_memory_size = ITERATION_MEMORY_SIZE;

size_t total_size, local_size = iteration_memory_size*magnitude.size() + objects.size();
MPI_Allreduce(&local_size, &total_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
return total_size;
}
size_t inline get_total_truncated_num_object(MPI_Comm communicator) const {
size_t total_truncated_num_object;
MPI_Allreduce(&truncated_num_object, &total_truncated_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
return total_truncated_num_object;
}


float get_avg_num_symbolic_object_per_task(MPI_Comm communicator) const {
size_t total_num_object_per_node = get_num_symbolic_object();
MPI_Allreduce(MPI_IN_PLACE, &total_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

int size;
MPI_Comm_size(communicator, &size);

return (float)total_num_object_per_node/size;
}
float get_avg_num_object_per_task(MPI_Comm communicator) const {
size_t max_num_object_per_node;
MPI_Allreduce(&num_object, &max_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

int size;
MPI_Comm_size(communicator, &size);

return (float)max_num_object_per_node/size;
}
size_t get_max_num_symbolic_object_per_task(MPI_Comm communicator) const {
size_t max_num_object_per_node = get_num_symbolic_object();
MPI_Allreduce(MPI_IN_PLACE, &max_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
return max_num_object_per_node;
}
size_t get_max_num_object_per_task(MPI_Comm communicator) const {
size_t max_num_object_per_node;
MPI_Allreduce(&num_object, &max_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
return max_num_object_per_node;
}


size_t get_truncated_mem_size(size_t begin_num_object=0) const;
};

class mpi_symbolic_iteration : public quids::symbolic_iteration {
public:
mpi_symbolic_iteration() {}


size_t inline get_total_num_object(MPI_Comm communicator) const {

size_t total_num_object;
MPI_Allreduce(&num_object, &total_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

return total_num_object;
}

size_t inline get_total_num_object_after_interferences(MPI_Comm communicator) const {

size_t total_num_object_after_interference;
MPI_Allreduce(&num_object_after_interferences, &total_num_object_after_interference, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

return total_num_object_after_interference;
}

private:
friend mpi_iteration;
friend void inline simulate(mpi_it_t &iteration, quids::rule_t const *rule, mpi_it_t &next_iteration, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object, quids::debug_t mid_step_function); 

quids::utils::fast_vector<mag_t> partitioned_mag;
quids::utils::fast_vector<size_t> partitioned_hash;

quids::utils::fast_vector<mag_t> mag_buffer;
quids::utils::fast_vector<size_t> hash_buffer;
quids::utils::fast_vector<int> node_id_buffer;

void compute_collisions(MPI_Comm communicator, quids::debug_t mid_step_function=[](const char*){});
void mpi_resize(size_t size) {
#pragma omp parallel sections
{
#pragma omp section
partitioned_mag.resize(size);

#pragma omp section
partitioned_hash.resize(size);
}
}
void buffer_resize(size_t size) {
#pragma omp parallel sections
{
#pragma omp section
mag_buffer.resize(size);

#pragma omp section
hash_buffer.resize(size);

#pragma omp section
node_id_buffer.resize(size);

#pragma omp section
if (size > next_oid_partitioner_buffer.size())
next_oid_partitioner_buffer.resize(size);
}
}




float inline get_avg_object_size(MPI_Comm communicator) const {
static const float hash_map_size = HASH_MAP_OVERHEAD*2*sizeof(size_t);

static const size_t symbolic_iteration_memory_size = SYMBOLIC_ITERATION_MEMORY_SIZE + MPI_SPECIFIC_SYMBOLIC_ITERATION_MEMORY_SIZE;

static const size_t mpi_symbolic_iteration_memory_size = MPI_SYMBOLIC_ITERATION_MEMORY_SIZE;
return (float)symbolic_iteration_memory_size + (float)mpi_symbolic_iteration_memory_size + hash_map_size;
}
size_t inline get_mem_size(MPI_Comm communicator) const {
static const size_t symbolic_iteration_memory_size = SYMBOLIC_ITERATION_MEMORY_SIZE + MPI_SPECIFIC_SYMBOLIC_ITERATION_MEMORY_SIZE;
size_t memory_size = magnitude.size()*symbolic_iteration_memory_size;

static const size_t mpi_symbolic_iteration_memory_size = MPI_SYMBOLIC_ITERATION_MEMORY_SIZE;
memory_size += mag_buffer.size()*mpi_symbolic_iteration_memory_size;

size_t total_size = mpi_symbolic_iteration_memory_size*magnitude.size();
MPI_Allreduce(MPI_IN_PLACE, &total_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
return total_size;
}
size_t inline get_total_next_iteration_num_object(MPI_Comm communicator) const {
size_t total_next_iteration_num_object;
MPI_Allreduce(&next_iteration_num_object, &total_next_iteration_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
return total_next_iteration_num_object;
}
};


void simulate(mpi_it_t &iteration, quids::rule_t const *rule, mpi_it_t &next_iteration, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object=0, quids::debug_t mid_step_function=[](const char*){}) {

MPI_Comm localComm;
int rank, size, local_size;
MPI_Comm_size(communicator, &size);
MPI_Comm_rank(communicator, &rank);
MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
MPI_Comm_size(localComm, &local_size);

const int max_equalize = quids::utils::log_2_upper_bound(size);






if (size == 1)
return quids::simulate(iteration, rule, next_iteration, symbolic_iteration, max_num_object, mid_step_function);



if (!equalize_children) {
mid_step_function("equalize_object");
float previous_diff, avg_n_object = iteration.get_avg_num_object_per_task(communicator);
for (int i = 0; i < max_equalize; ++i) {

size_t max_n_object = iteration.get_max_num_object_per_task(communicator);
float diff = (float)max_n_object - avg_n_object;
float inbalance = diff/max_n_object;

if (rank == 0)
std::cerr << "\ti=" << i << "/" << max_equalize << "\tmax=" << max_n_object << ", avg=" << avg_n_object << ", inbalance=" << inbalance << "\n";

if (max_n_object < min_equalize_size ||
inbalance < equalize_inbalance ||
(diff > previous_diff*(1 - min_equalize_step)) && i > 0)
break;


iteration.equalize(communicator);

previous_diff = diff;
}
}



iteration.compute_num_child(rule, mid_step_function);
iteration.truncated_num_object = iteration.num_object;



if (equalize_children) {
mid_step_function("equalize_child");
float previous_diff, avg_n_child = iteration.get_avg_num_symbolic_object_per_task(communicator);
for (int i = 0; i < max_equalize; ++i) {

size_t max_n_object = iteration.get_max_num_object_per_task(communicator);
size_t max_n_child = iteration.get_max_num_symbolic_object_per_task(communicator);
float diff = (float)max_n_child - avg_n_child;
float inbalance = diff/max_n_child;

if (rank == 0)
std::cerr << "\ti=" << i << "/" << max_equalize << "\tmax=" << max_n_child << ", avg=" << avg_n_child << ", inbalance=" << inbalance << "\n";

if (max_n_object < min_equalize_size ||
inbalance < equalize_inbalance ||
(diff > previous_diff*(1 - min_equalize_step)) && i > 0)
break;


iteration.equalize_symbolic(communicator);
iteration.truncated_num_object = iteration.num_object;

previous_diff = diff;
}
}



mid_step_function("truncate_symbolic - prepare");
iteration.prepare_truncate(mid_step_function);



mid_step_function("truncate_symbolic");
if (max_num_object == 0) {

size_t avail_memory = next_iteration.get_mem_size(localComm) + symbolic_iteration.get_mem_size(localComm) + quids::utils::get_free_mem();
size_t target_memory = avail_memory/local_size*(1 - quids::safety_margin);


if (iteration.get_truncated_mem_size() > target_memory) {
size_t begin = 0, end = iteration.num_object;
while (end > begin + 1) {
size_t middle = (end + begin) / 2;
iteration.truncate(begin, middle, mid_step_function);

size_t used_memory = iteration.get_truncated_mem_size(begin);
if (used_memory < target_memory) {
target_memory -= used_memory;
begin = middle;
} else
end = middle;
}
}
} else
iteration.truncate(0, max_num_object/local_size, mid_step_function);



if (iteration.num_object > 0) {
if (iteration.truncated_num_object < next_iteration.num_object)
next_iteration.resize(iteration.truncated_num_object);
size_t next_object_size = iteration.truncated_num_object*iteration.get_object_length()/iteration.num_object;
if (next_object_size < next_iteration.objects.size())
next_iteration.allocate(next_object_size);
}



iteration.generate_symbolic_iteration(rule, symbolic_iteration, mid_step_function);
symbolic_iteration.compute_collisions(communicator, mid_step_function);
symbolic_iteration.next_iteration_num_object = symbolic_iteration.num_object_after_interferences;



mid_step_function("truncate - prepare");
symbolic_iteration.prepare_truncate(mid_step_function);



mid_step_function("truncate");
if (max_num_object == 0) {

size_t avail_memory = next_iteration.get_mem_size(localComm) + quids::utils::get_free_mem();
size_t target_memory = avail_memory/local_size*(1 - quids::safety_margin);


if (symbolic_iteration.get_truncated_mem_size() > target_memory) {
size_t begin = 0, end = symbolic_iteration.num_object_after_interferences;
while (end > begin + 1) {
size_t middle = (end + begin) / 2;
symbolic_iteration.truncate(begin, middle, mid_step_function);

size_t used_memory = symbolic_iteration.get_truncated_mem_size(begin);
if (used_memory < target_memory) {
target_memory -= used_memory;
begin = middle;
} else
end = middle;
}
}
} else
symbolic_iteration.truncate(0, max_num_object/local_size, mid_step_function);



symbolic_iteration.finalize(rule, iteration, next_iteration, mid_step_function);
next_iteration.normalize(communicator, mid_step_function);

MPI_Comm_free(&localComm);
}


size_t mpi_iteration::get_truncated_mem_size(size_t begin_num_object) const {
static const size_t iteration_memory_size = ITERATION_MEMORY_SIZE;

static const float hash_map_size = HASH_MAP_OVERHEAD*2*sizeof(size_t);
static const size_t symbolic_iteration_memory_size = SYMBOLIC_ITERATION_MEMORY_SIZE + MPI_SPECIFIC_SYMBOLIC_ITERATION_MEMORY_SIZE;
static const size_t mpi_symbolic_iteration_memory_size = MPI_SYMBOLIC_ITERATION_MEMORY_SIZE;

size_t mem_size = iteration_memory_size*(truncated_num_object - begin_num_object);
for (size_t i = begin_num_object; i < truncated_num_object; ++i) {
size_t oid = truncated_oid[i];

mem_size += object_begin[oid + 1] - object_begin[oid];
mem_size += num_childs[oid]*(symbolic_iteration_memory_size + mpi_symbolic_iteration_memory_size + hash_map_size);
}

return mem_size;
}


void mpi_symbolic_iteration::compute_collisions(MPI_Comm communicator, quids::debug_t mid_step_function) {
int size, rank;
MPI_Comm_size(communicator, &size);
MPI_Comm_rank(communicator, &rank);

if (size == 1)
return quids::symbolic_iteration::compute_collisions(mid_step_function);

int num_threads;
#pragma omp parallel
#pragma omp single
num_threads = omp_get_num_threads();

int const n_segment = size*num_threads;
int const num_bucket = quids::utils::nearest_power_of_two(load_balancing_bucket_per_thread*n_segment);
size_t const offset = 8*sizeof(size_t) - quids::utils::log_2_upper_bound(num_bucket);

std::vector<int> load_balancing_begin(n_segment + 1);
std::vector<size_t> partition_begin(num_bucket + 1);
std::vector<size_t> total_partition_begin(num_bucket + 1);

std::vector<int> local_disp(n_segment + 1);
std::vector<int> local_count(n_segment);
std::vector<int> global_disp(n_segment + 1, 0);
std::vector<int> global_count(n_segment);

std::vector<int> send_disp(size + 1);
std::vector<int> send_count(size);
std::vector<int> receive_disp(size + 1);
std::vector<int> receive_count(size);

mid_step_function("compute_collisions - prepare");
mpi_resize(num_object);






quids::utils::parallel_generalized_partition_from_iota(&next_oid[0], &next_oid[0] + num_object, 0,
partition_begin.begin(), partition_begin.end(),
[&](size_t const oid) {
return hash[oid] >> offset;
});


#pragma omp parallel for
for (size_t id = 0; id < num_object; ++id) {
size_t oid = next_oid[id];

partitioned_mag[id] = magnitude[oid];
partitioned_hash[id] = hash[oid];
}









#ifndef SKIP_CCP
mid_step_function("compute_collisions - com");
MPI_Allreduce(&partition_begin[1], &total_partition_begin[1],
num_bucket, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

mid_step_function("compute_collisions - prepare");
total_partition_begin[0] = 0;
quids::utils::load_balancing_from_prefix_sum(total_partition_begin.begin(), total_partition_begin.end(),
load_balancing_begin.begin(), load_balancing_begin.end());
#else
for (size_t i = 0; i <= n_segment; ++i)
load_balancing_begin[i] = i*num_bucket/n_segment;
#endif


local_disp[0] = 0;
for (int i = 1; i <= n_segment; ++i) {
local_disp[i] = partition_begin[load_balancing_begin[i]];
local_count[i - 1] = local_disp[i] - local_disp[i - 1];
}







mid_step_function("compute_collisions - com");
MPI_Alltoall(&local_count [0], num_threads, MPI_INT, 
&global_count[0], num_threads, MPI_INT, communicator);

mid_step_function("compute_collisions - prepare");
std::partial_sum(&global_count[0], &global_count[0] + n_segment, &global_disp[1]);


send_disp[0] = 0; receive_disp[0] = 0;
for (int i = 1; i <= size; ++i) {
send_disp[i] = local_disp[i*num_threads];
send_count[i - 1] = send_disp[i] - send_disp[i - 1];

receive_disp[i] = global_disp[i*num_threads];
receive_count[i - 1] = receive_disp[i] - receive_disp[i - 1];
}


buffer_resize(receive_disp[size]);


mid_step_function("compute_collisions - com");
MPI_Alltoallv(&partitioned_hash[0], &send_count[0],    &send_disp[0],    MPI_UNSIGNED_LONG_LONG,
&hash_buffer[0],      &receive_count[0], &receive_disp[0], MPI_UNSIGNED_LONG_LONG, communicator);
MPI_Alltoallv(&partitioned_mag[0],  &send_count[0],    &send_disp[0],    mag_MPI_Datatype,
&mag_buffer[0],       &receive_count[0], &receive_disp[0], mag_MPI_Datatype,       communicator);







mid_step_function("compute_collisions - prepare");

for (int node = 0; node < size; ++node)
std::fill(&node_id_buffer[0] + receive_disp[node],
&node_id_buffer[0] + receive_disp[node + 1],
node);

if (rank == 0)
std::cerr << size << "=size\n";

mid_step_function("compute_collisions - insert");
#pragma omp parallel
{
std::vector<size_t> global_num_object_after_interferences(size, 0);
const auto work_steal = [&](int const node_id, int const other_node_id) {
#ifndef SKIP_WORK_STEALING
bool steal = global_num_object_after_interferences[other_node_id] >= global_num_object_after_interferences[node_id];

if (steal) {
--global_num_object_after_interferences[other_node_id];
} else
--global_num_object_after_interferences[node_id];

return steal;
#else
return false;
#endif
};

int const thread_id = omp_get_thread_num();
robin_hood::unordered_map<size_t, size_t> elimination_map;


size_t total_size = 0, max_count = 0;
for (int node_id = 0; node_id < size; ++node_id) {
size_t this_size = global_count[node_id*num_threads + thread_id];

total_size +=          this_size;
max_count   = std::max(this_size, max_count);
}
elimination_map.reserve(total_size);


for (size_t i = 0; GRANULARITY*i < max_count; ++i)
for (int j = 0; j < size; ++j) {
const int node_id = (rank + j)%size;

if (node_id >= size)
throw;

const size_t begin =                i*GRANULARITY        + global_disp[node_id*num_threads + thread_id    ];
const size_t end   = std::min(begin + GRANULARITY, (size_t)global_disp[node_id*num_threads + thread_id + 1]);

for (size_t oid = begin; oid < end; ++oid) {
++global_num_object_after_interferences[node_id];


auto [it, unique] = elimination_map.insert({hash_buffer[oid], oid});
if (!unique) {
const size_t other_oid = it->second;
const int other_node_id = node_id_buffer[other_oid];

if (work_steal(node_id, other_node_id)) {

it->second = oid;


mag_buffer[oid]      += mag_buffer[other_oid];
mag_buffer[other_oid] = 0;
} else {

mag_buffer[other_oid] += mag_buffer[oid];
mag_buffer[oid]        = 0;
}
}
}
}
}








mid_step_function("compute_collisions - com");
MPI_Alltoallv(&mag_buffer[0],      &receive_count[0], &receive_disp[0], mag_MPI_Datatype,
&partitioned_mag[0], &send_count[0],    &send_disp[0],    mag_MPI_Datatype, communicator);


mid_step_function("compute_collisions - finalize");
#pragma omp parallel for
for (size_t id = 0; id < num_object; ++id)
magnitude[next_oid[id]] = partitioned_mag[id];








size_t* partitioned_it = __gnu_parallel::partition(&next_oid[0], &next_oid[0] + num_object,
[&](size_t const &oid) {
return std::norm(magnitude[oid]) > tolerance;
});
num_object_after_interferences = std::distance(&next_oid[0], partitioned_it);
}


void mpi_iteration::normalize(MPI_Comm communicator, quids::debug_t mid_step_function) {

mid_step_function("normalize");

node_total_proba = 0;
total_proba = 0;

#pragma omp parallel for reduction(+:node_total_proba)
for (size_t oid = 0; oid < num_object; ++oid)
node_total_proba += std::norm(magnitude[oid]);


MPI_Allreduce(&node_total_proba, &total_proba, 1, Proba_MPI_Datatype, MPI_SUM, communicator);
PROBA_TYPE normalization_factor = std::sqrt(total_proba);

if (normalization_factor != 1)
#pragma omp parallel for 
for (size_t oid = 0; oid < num_object; ++oid)
magnitude[oid] /= normalization_factor;

node_total_proba /= total_proba;

mid_step_function("end");
}



void mpi_iteration::equalize(MPI_Comm communicator) {
MPI_Request request = MPI_REQUEST_NULL;

MPI_Comm localComm;
int rank, size, local_size;
MPI_Comm_size(communicator, &size);
MPI_Comm_rank(communicator, &rank);
MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
MPI_Comm_size(localComm, &local_size);

int this_pair_id;
if (rank == 0) {

std::vector<size_t> sizes(size, 0);
MPI_Gather(&num_object, 1, MPI_UNSIGNED_LONG_LONG, &sizes[0], 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);


std::vector<int> pair_id(size, 0);
utils::make_equal_pairs(&sizes[0], &sizes[0] + size, &pair_id[0]);


MPI_Scatter(&pair_id[0], 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
} else {

MPI_Gather(&num_object, 1, MPI_UNSIGNED_LONG_LONG, NULL, 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);


MPI_Scatter(NULL, 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
}


MPI_Barrier(localComm);
size_t total_iteration_size, iteration_size = quids::iteration::get_mem_size();
MPI_Allreduce(&iteration_size, &total_iteration_size, 1, Proba_MPI_Datatype, MPI_SUM, localComm);
size_t avail_memory = ((quids::utils::get_free_mem() + total_iteration_size)/local_size - iteration_size)*(1 - quids::safety_margin);
MPI_Barrier(localComm);


if (this_pair_id == rank)
return;


size_t other_num_object;
MPI_Isend(&num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 , communicator, &request);
MPI_Isend(&num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 , communicator, &request);

MPI_Recv(&other_num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 , communicator, MPI_STATUS_IGNORE);
MPI_Recv(&other_num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 , communicator, MPI_STATUS_IGNORE);


if (num_object > other_num_object) {
size_t num_object_sent = (num_object -  other_num_object) / 2;
send_objects(num_object_sent, this_pair_id, communicator, false);
} else if (num_object < other_num_object)
receive_objects(this_pair_id, communicator, false, avail_memory);
}


void mpi_iteration::equalize_symbolic(MPI_Comm communicator) {
MPI_Request request = MPI_REQUEST_NULL;

MPI_Comm localComm;
int rank, size, local_size;
MPI_Comm_size(communicator, &size);
MPI_Comm_rank(communicator, &rank);
MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
MPI_Comm_size(localComm, &local_size);


size_t num_symbolic_object = child_begin[num_object];

int this_pair_id;
if (rank == 0) {

std::vector<size_t> sizes(size, 0);
MPI_Gather(&num_symbolic_object, 1, MPI_UNSIGNED_LONG_LONG, &sizes[0], 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);


std::vector<int> pair_id(size, 0);
utils::make_equal_pairs(&sizes[0], &sizes[0] + size, &pair_id[0]);


MPI_Scatter(&pair_id[0], 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
} else {

MPI_Gather(&num_symbolic_object, 1, MPI_UNSIGNED_LONG_LONG, NULL, 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);


MPI_Scatter(NULL, 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
}


MPI_Barrier(localComm);
size_t total_iteration_size, iteration_size = quids::iteration::get_mem_size();
MPI_Allreduce(&iteration_size, &total_iteration_size, 1, Proba_MPI_Datatype, MPI_SUM, localComm);
size_t avail_memory = ((quids::utils::get_free_mem() + total_iteration_size)/local_size - iteration_size)*(1 - quids::safety_margin);
MPI_Barrier(localComm);


if (this_pair_id == rank)
return;


uint other_num_object;
uint other_ub_symbolic_object_size;

MPI_Isend(&num_symbolic_object,          1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 , communicator, &request);
MPI_Isend(&ub_symbolic_object_size,      1, MPI_UNSIGNED,           this_pair_id, 0 , communicator, &request);

MPI_Recv(&other_num_object,              1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 , communicator, MPI_STATUS_IGNORE);
MPI_Recv(&other_ub_symbolic_object_size, 1, MPI_UNSIGNED,           this_pair_id, 0 , communicator, MPI_STATUS_IGNORE);

ub_symbolic_object_size = std::max(ub_symbolic_object_size, other_ub_symbolic_object_size);


if (num_symbolic_object > other_num_object) {
size_t num_symbolic_object_to_send = (num_symbolic_object - other_num_object) / 2;


auto limit_it = std::lower_bound(child_begin.begin(), child_begin.begin() + num_object, num_symbolic_object - num_symbolic_object_to_send) - 1;
size_t num_object_sent = std::distance(limit_it, child_begin.begin() + num_object);

send_objects(num_object_sent, this_pair_id, communicator, true);
} else if (num_symbolic_object < other_num_object)
receive_objects(this_pair_id, communicator, true, avail_memory);
}


void mpi_iteration::distribute_objects(MPI_Comm communicator, int node_id=0) {
int size, rank;
MPI_Comm_size(communicator, &size);
MPI_Comm_rank(communicator, &rank);

size_t initial_num_object = num_object;
if (rank == node_id) {
for (int node = 1; node < size; ++node) {
int node_to_send = node <= node_id ? node - 1 : node; 
size_t num_object_sent = (initial_num_object * (node + 1)) / size - (initial_num_object * node) / size; 


send_objects(num_object_sent, node_to_send, communicator);
}

} else

receive_objects(node_id, communicator);
}


void mpi_iteration::gather_objects(MPI_Comm communicator, int node_id=0) {
int size, rank;
MPI_Comm_size(communicator, &size);
MPI_Comm_rank(communicator, &rank);

if (rank == node_id) {
for (int node = 1; node < size; ++node) {
int receive_node = node <= node_id ? node - 1 : node;


receive_objects(receive_node, communicator);
}

} else

send_objects(num_object, node_id, communicator);


node_total_proba = rank == node_id;
}
}