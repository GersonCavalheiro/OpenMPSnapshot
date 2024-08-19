#pragma once

typedef unsigned uint;

#include <parallel/algorithm>
#include <parallel/numeric>
#include <limits>

#include <complex>
#include <cstddef>
#include <vector>

#include "utils/libs/robin_hood.h"

#include "utils/vector.hpp"
#include "utils/load_balancing.hpp"
#include "utils/algorithm.hpp"
#include "utils/memory.hpp"
#include "utils/random.hpp"

#ifndef PROBA_TYPE
#define PROBA_TYPE double 
#endif
#ifndef HASH_MAP_OVERHEAD
#define HASH_MAP_OVERHEAD 1.7 
#endif
#ifndef ALIGNMENT_BYTE_LENGTH
#define ALIGNMENT_BYTE_LENGTH 8
#endif
#ifndef TOLERANCE
#define TOLERANCE 1e-30;
#endif
#ifndef SAFETY_MARGIN
#define SAFETY_MARGIN 0.2
#endif
#ifndef LOAD_BALANCING_BUCKET_PER_THREAD
#define LOAD_BALANCING_BUCKET_PER_THREAD 32
#endif

#define ITERATION_MEMORY_SIZE 2*sizeof(PROBA_TYPE) + 3*sizeof(size_t) + sizeof(float) + 2*sizeof(uint)
#define SYMBOLIC_ITERATION_MEMORY_SIZE 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t) + 2*sizeof(uint) + sizeof(float)


#ifndef _OPENMP
#define omp_set_nested(i)
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#else
#include <omp.h>
#endif

namespace quids {
uint align_byte_length = ALIGNMENT_BYTE_LENGTH;
PROBA_TYPE tolerance = TOLERANCE;
float safety_margin = SAFETY_MARGIN;
int load_balancing_bucket_per_thread = LOAD_BALANCING_BUCKET_PER_THREAD;
#ifdef SIMPLE_TRUNCATION
bool simple_truncation = true;
#else
bool simple_truncation = false;
#endif

typedef std::complex<PROBA_TYPE> mag_t;
typedef class iteration it_t;
typedef class symbolic_iteration sy_it_t;
typedef class rule rule_t;
typedef std::function<void(char* parent_begin, char* parent_end, mag_t &mag)> modifier_t;
typedef std::function<PROBA_TYPE(char const *object_begin, char const *object_end)> observable_t;
typedef std::function<void(const char* step)> debug_t;

uint inline get_alignment_offset(const uint size) {
if (align_byte_length <= 1)
return 0;

uint alignment_offset = align_byte_length - size%align_byte_length;
if (alignment_offset == align_byte_length)
alignment_offset = 0;

return alignment_offset;
}

class rule {
public:
rule() {};

virtual inline void get_num_child(char const *parent_begin, char const *parent_end, uint &num_child, uint &max_child_size) const = 0;

virtual inline void populate_child(char const *parent_begin, char const *parent_end, char* const child_begin, uint const child_id, uint &size, mag_t &mag) const = 0;

virtual inline void populate_child_simple(char const *parent_begin, char const *parent_end, char* const child_begin, uint const child_id) const { 
uint size_placeholder;
mag_t mag_placeholder;
populate_child(parent_begin, parent_end, child_begin, child_id,
size_placeholder, mag_placeholder);
}

virtual inline size_t hasher(char const *object_begin, char const *object_end) const {
return std::hash<std::string_view>()(std::string_view(object_begin, std::distance(object_begin, object_end)));
}
};

class iteration {
public:
size_t num_object = 0;
PROBA_TYPE total_proba = 1;

iteration() {
resize(0);
allocate(0);
object_begin[0] = 0;
}

iteration(char* object_begin_, char* object_end_) : iteration() {
append(object_begin_, object_end_);
}

void append(char const *object_begin_, char const *object_end_, mag_t const mag=1) {
size_t offset = object_begin[num_object];
size_t size = std::distance(object_begin_, object_end_);
uint alignment_offset = get_alignment_offset(size);

resize(++num_object);
allocate(offset + size + alignment_offset);

for (size_t i = 0; i < size; ++i)
objects[offset + i] = object_begin_[i];

magnitude[num_object - 1] = mag;
object_size[num_object - 1] = size;
object_begin[num_object] = offset + size + alignment_offset;
}

void pop(size_t n=1, bool normalize_=true) {
if (n < 1)
return;

num_object -= n;
allocate(object_begin[num_object]);
resize(num_object);

if (normalize_) normalize();
}

PROBA_TYPE average_value(const observable_t observable) const {
PROBA_TYPE avg = 0;
if (num_object == 0)
return avg;

#pragma omp parallel
{	

PROBA_TYPE local_avg = 0;
#pragma omp for 
for (size_t oid = 0; oid < num_object; ++oid) {
uint size;
std::complex<PROBA_TYPE> mag;


char const *this_object_begin;
get_object(oid, this_object_begin, size, mag);
local_avg += observable(this_object_begin, this_object_begin + size) * std::norm(mag);
}


#pragma omp critical
avg += local_avg;
}

return avg;
}

void get_object(size_t const object_id, char *& object_begin_, uint &object_size_, mag_t *&mag) {
object_size_ = object_size[object_id];
mag = &magnitude[object_id];
object_begin_ = &objects[object_begin[object_id]];
}

void get_object(size_t const object_id, char const *& object_begin_, uint &object_size_, mag_t &mag) const {
object_size_ = object_size[object_id];
mag = magnitude[object_id];
object_begin_ = &objects[object_begin[object_id]];
}

private:
friend symbolic_iteration;
friend void inline simulate(it_t &iteration, rule_t const *rule, it_t &next_iteration, sy_it_t &symbolic_iteration, size_t max_num_object, debug_t mid_step_function);  
friend void inline simulate(it_t &iteration, modifier_t const rule);

protected:
mutable size_t truncated_num_object = 0;
mutable uint ub_symbolic_object_size = 0;

mutable utils::fast_vector<mag_t> magnitude;
mutable utils::fast_vector<char> objects;
mutable utils::fast_vector<size_t> object_begin;
mutable utils::fast_vector<uint> object_size;
mutable utils::fast_vector<uint> num_childs;
mutable utils::fast_vector<size_t> child_begin;
mutable utils::fast_vector<size_t> truncated_oid;
mutable utils::fast_vector<float> random_selector;

void inline resize(size_t num_object) const {
#pragma omp parallel sections
{
#pragma omp section
magnitude.resize(num_object);

#pragma omp section
num_childs.resize(num_object);

#pragma omp section
object_size.resize(num_object);

#pragma omp section
object_begin.resize(num_object + 1);

#pragma omp section
child_begin.resize(num_object + 1);

#pragma omp section
{
truncated_oid.resize(num_object);
utils::parallel_iota(&truncated_oid[0], &truncated_oid[0] + num_object, 0);
}

#pragma omp section
random_selector.resize(num_object);
}
}
void inline allocate(size_t size) const {
objects.resize(size, align_byte_length);
}



size_t get_mem_size() const {
static const size_t iteration_memory_size = ITERATION_MEMORY_SIZE;
return magnitude.size()*iteration_memory_size + objects.size();
}
size_t get_object_length() const {
return object_begin[num_object];
}
size_t get_num_symbolic_object() const {
return __gnu_parallel::accumulate(&num_childs[0], &num_childs[0] + num_object, (size_t)0);
}


void compute_num_child(rule_t const *rule, debug_t mid_step_function=[](const char*){}) const;
void prepare_truncate(debug_t mid_step_function=[](const char*){}) const;
size_t get_truncated_mem_size(size_t begin_num_object=0) const;
void truncate(size_t begin_num_object, size_t max_num_object, debug_t mid_step_function=[](const char*){}) const;
void generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, debug_t mid_step_function=[](const char*){}) const;
void apply_modifier(modifier_t const rule);
void normalize(debug_t mid_step_function=[](const char*){});
};

class symbolic_iteration {
public:
symbolic_iteration() {}

size_t num_object = 0;
size_t num_object_after_interferences = 0;

private:
friend iteration;
friend void inline simulate(it_t &iteration, rule_t const *rule, it_t &next_iteration, sy_it_t &symbolic_iteration, size_t max_num_object, debug_t mid_step_function); 

protected:
size_t next_iteration_num_object = 0;

std::vector<char*> placeholder;

utils::fast_vector<mag_t> magnitude;
utils::fast_vector<size_t> next_oid;
utils::fast_vector<uint> size;
utils::fast_vector<size_t> hash;
utils::fast_vector<size_t> parent_oid;
utils::fast_vector<uint> child_id;
utils::fast_vector<float> random_selector;
utils::fast_vector<size_t> next_oid_partitioner_buffer;

void inline resize(size_t num_object) {
#pragma omp parallel sections
{
#pragma omp section
magnitude.resize(num_object);

#pragma omp section
next_oid.resize(num_object);

#pragma omp section
size.resize(num_object);

#pragma omp section
hash.resize(num_object);

#pragma omp section
parent_oid.resize(num_object);

#pragma omp section
child_id.resize(num_object);

#pragma omp section
random_selector.resize(num_object);

#pragma omp section
next_oid_partitioner_buffer.resize(num_object);
}

utils::parallel_iota(&next_oid[0], &next_oid[num_object], 0);
}
void inline reserve(size_t max_size) {
int num_threads;
#pragma omp parallel
#pragma omp single
num_threads = omp_get_num_threads();

placeholder.resize(num_threads);

#pragma omp parallel
{
auto &buffer = placeholder[omp_get_thread_num()];
if (buffer == NULL)
free(buffer);
buffer = new char[max_size];
}
}



size_t get_mem_size() const {
static const size_t symbolic_iteration_memory_size = SYMBOLIC_ITERATION_MEMORY_SIZE;
return magnitude.size()*symbolic_iteration_memory_size;
}

void compute_collisions(debug_t mid_step_function=[](const char*){});
size_t get_truncated_mem_size(size_t begin_num_object=0) const;
void truncate(size_t begin_num_object, size_t max_num_object, debug_t mid_step_function=[](const char*){});
void prepare_truncate(debug_t mid_step_function=[](const char*){});
void finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, debug_t mid_step_function=[](const char*){});
};


void inline simulate(it_t &iteration, modifier_t const rule) {
iteration.apply_modifier(rule);
}

void inline simulate(it_t &iteration, rule_t const *rule, it_t &next_iteration, sy_it_t &symbolic_iteration, size_t max_num_object=0, debug_t mid_step_function=[](const char*){}) {	

iteration.compute_num_child(rule, mid_step_function);
iteration.truncated_num_object = iteration.num_object;


mid_step_function("truncate_symbolic - prepare");
iteration.prepare_truncate(mid_step_function);


mid_step_function("truncate_symbolic");
if (max_num_object == 0) {


size_t avail_memory =  next_iteration.get_mem_size() + symbolic_iteration.get_mem_size() + utils::get_free_mem();
size_t target_memory = avail_memory*(1 - safety_margin);


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
iteration.truncate(0, max_num_object, mid_step_function);


if (iteration.num_object > 0) {
if (iteration.truncated_num_object < next_iteration.num_object)
next_iteration.resize(iteration.truncated_num_object);
size_t next_object_size = iteration.truncated_num_object*iteration.get_object_length()/iteration.num_object;
if (next_object_size < next_iteration.objects.size())
next_iteration.allocate(next_object_size);
}


iteration.generate_symbolic_iteration(rule, symbolic_iteration, mid_step_function);
symbolic_iteration.compute_collisions(mid_step_function);
symbolic_iteration.next_iteration_num_object = symbolic_iteration.num_object_after_interferences;



mid_step_function("truncate - prepare");
symbolic_iteration.prepare_truncate(mid_step_function);


mid_step_function("truncate");
if (max_num_object == 0) {


size_t avail_memory = next_iteration.get_mem_size() + utils::get_free_mem();
size_t target_memory = avail_memory*(1 - safety_margin);


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
symbolic_iteration.truncate(0, max_num_object, mid_step_function);


symbolic_iteration.finalize(rule, iteration, next_iteration, mid_step_function);
next_iteration.normalize(mid_step_function);
}


void iteration::compute_num_child(rule_t const *rule, debug_t mid_step_function) const {

mid_step_function("num_child");

if (num_object == 0)
return;

ub_symbolic_object_size = 0;

#pragma omp parallel for  reduction(max:ub_symbolic_object_size)
for (size_t oid = 0; oid < num_object; ++oid) {
uint size;
rule->get_num_child(&objects[object_begin[oid]],
&objects[object_begin[oid] + object_size[oid]],
num_childs[oid], size);
ub_symbolic_object_size = std::max(ub_symbolic_object_size, size);
}

__gnu_parallel::partial_sum(num_childs.begin(), num_childs.begin() + num_object, child_begin.begin() + 1);
}


size_t iteration::get_truncated_mem_size(size_t begin_num_object) const {
static const size_t iteration_memory_size = ITERATION_MEMORY_SIZE;

static const float hash_map_size = HASH_MAP_OVERHEAD*2*sizeof(size_t);
static const size_t symbolic_iteration_memory_size = SYMBOLIC_ITERATION_MEMORY_SIZE;

size_t mem_size = iteration_memory_size*(truncated_num_object - begin_num_object);
for (size_t i = begin_num_object; i < truncated_num_object; ++i) {
size_t oid = truncated_oid[i];

mem_size += object_begin[oid + 1] - object_begin[oid];
mem_size += num_childs[oid]*(symbolic_iteration_memory_size + hash_map_size);
}

return mem_size;
}


void iteration::prepare_truncate(debug_t mid_step_function) const {


if (!simple_truncation)
#pragma omp parallel
{
utils::random_generator rng;

#pragma omp for
for (size_t oid = 0; oid < num_object; ++oid)
random_selector[oid] = rng() / std::norm(magnitude[oid]); 
}
}


void iteration::truncate(size_t begin_num_object, size_t max_num_object, debug_t mid_step_function) const {


if (max_num_object >= num_object) {
truncated_num_object = num_object;
return;
}


auto begin = truncated_oid.begin() + begin_num_object;
auto middle = truncated_oid.begin() + max_num_object;
auto end = truncated_oid.begin() + truncated_num_object;

if (simple_truncation) {

__gnu_parallel::nth_element(begin, middle, end,
[&](size_t const &oid1, size_t const &oid2) {
return std::norm(magnitude[oid1]) > std::norm(magnitude[oid2]);
});
} else

__gnu_parallel::nth_element(begin, middle, end,
[&](size_t const &oid1, size_t const &oid2) {
return random_selector[oid1] < random_selector[oid2];
});

truncated_num_object = max_num_object;
}


void iteration::generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, debug_t mid_step_function) const {
if (truncated_num_object == 0) {
symbolic_iteration.num_object = 0;
mid_step_function("prepare_index");
mid_step_function("symbolic_iteration");
return;
}








mid_step_function("prepare_index");

child_begin[0] = 0;
for (size_t i = 0; i < truncated_num_object; ++i) {
size_t oid = truncated_oid[i];

child_begin[i + 1] = child_begin[i] + num_childs[oid];
}

symbolic_iteration.num_object = child_begin[truncated_num_object];


symbolic_iteration.resize(symbolic_iteration.num_object);
symbolic_iteration.reserve(ub_symbolic_object_size);

#pragma omp parallel
{
auto thread_id = omp_get_thread_num();

#pragma omp for 
for (size_t i = 0; i < truncated_num_object; ++i) {
size_t oid = truncated_oid[i];


std::fill(&symbolic_iteration.parent_oid[child_begin[i]],
&symbolic_iteration.parent_oid[child_begin[i + 1]],
oid);
std::iota(&symbolic_iteration.child_id[child_begin[i]],
&symbolic_iteration.child_id[child_begin[i + 1]],
0);
}





#pragma omp single
mid_step_function("symbolic_iteration");

#pragma omp for 
for (size_t oid = 0; oid < symbolic_iteration.num_object; ++oid) {
auto id = symbolic_iteration.parent_oid[oid];


symbolic_iteration.magnitude[oid] = magnitude[id];
rule->populate_child(&objects[object_begin[id]],
&objects[object_begin[id] + object_size[id]],
symbolic_iteration.placeholder[thread_id], symbolic_iteration.child_id[oid],
symbolic_iteration.size[oid], symbolic_iteration.magnitude[oid]);


symbolic_iteration.hash[oid] = rule->hasher(symbolic_iteration.placeholder[thread_id],
symbolic_iteration.placeholder[thread_id] + symbolic_iteration.size[oid]);
}
}
}


void symbolic_iteration::compute_collisions(debug_t mid_step_function) {
if (num_object == 0) {
num_object_after_interferences = 0;
mid_step_function("compute_collisions - prepare");
mid_step_function("compute_collisions - insert");
mid_step_function("compute_collisions - finalize");
return;
}

int num_threads;
#pragma omp parallel
#pragma omp single
num_threads = omp_get_num_threads();

int const num_bucket = utils::nearest_power_of_two(load_balancing_bucket_per_thread*num_threads);
size_t const offset = 8*sizeof(size_t) - utils::log_2_upper_bound(num_bucket);

std::vector<int> load_balancing_begin(num_threads + 1, 0);
std::vector<size_t> partition_begin(num_bucket + 1, 0);







mid_step_function("compute_collisions - prepare");
quids::utils::parallel_generalized_partition_from_iota(&next_oid[0], &next_oid[0] + num_object, 0,
&partition_begin[0], &partition_begin[num_bucket + 1],
[&](size_t const oid) {
return hash[oid] >> offset;
});






#ifndef SKIP_CCP
quids::utils::load_balancing_from_prefix_sum(partition_begin.begin(), partition_begin.end(),
load_balancing_begin.begin(), load_balancing_begin.end());
#else
for (size_t i = 0; i <= num_threads; ++i)
load_balancing_begin[i] = i*num_bucket/num_threads;
#endif







mid_step_function("compute_collisions - insert");
#pragma omp parallel
{
int thread_id = omp_get_thread_num();
int load_begin = load_balancing_begin[thread_id], load_end = load_balancing_begin[thread_id + 1];
for (int j = load_begin; j < load_end; ++j) {
robin_hood::unordered_map<size_t, size_t> elimination_map;

size_t begin = partition_begin[j], end = partition_begin[j + 1];

elimination_map.reserve(end - begin);
for (size_t i = begin; i < end; ++i) {
size_t oid = next_oid[i];


auto [it, unique] = elimination_map.insert({hash[oid], oid});
if (!unique) {
const size_t other_oid = it->second;


magnitude[other_oid] += magnitude[oid];
magnitude[oid]        = 0;
}
}
}
}
mid_step_function("compute_collisions - finalize");






size_t* partitioned_it = __gnu_parallel::partition(&next_oid[0], &next_oid[0] + num_object,
[&](size_t const &oid) {
return std::norm(magnitude[oid]) > tolerance;
});
num_object_after_interferences = std::distance(&next_oid[0], partitioned_it);
}


void symbolic_iteration::prepare_truncate(debug_t mid_step_function) {


if (!simple_truncation)
#pragma omp parallel
{
utils::random_generator rng;

#pragma omp for
for (size_t i = 0; i < num_object_after_interferences; ++i) {
size_t oid = next_oid[i];
random_selector[oid] = rng() / std::norm(magnitude[oid]); 
}
}
}


size_t symbolic_iteration::get_truncated_mem_size(size_t begin_num_object) const {
static const size_t iteration_memory_size = ITERATION_MEMORY_SIZE;

size_t mem_size = iteration_memory_size*(next_iteration_num_object - begin_num_object);
for (size_t i = begin_num_object; i < next_iteration_num_object; ++i) {
size_t this_mem_size = size[next_oid[i]];
uint alignment_offset = get_alignment_offset(this_mem_size);
mem_size += this_mem_size + alignment_offset;
}

return mem_size;
}


void symbolic_iteration::truncate(size_t begin_num_object, size_t max_num_object, debug_t mid_step_function) {
if (next_iteration_num_object == 0)
return;



if (max_num_object >= num_object_after_interferences) {
next_iteration_num_object = num_object_after_interferences;
return;
}


auto begin = next_oid.begin() + begin_num_object;
auto middle = next_oid.begin() + max_num_object;
auto end = next_oid.begin() + next_iteration_num_object;

if (simple_truncation) {

__gnu_parallel::nth_element(begin, middle, end,
[&](size_t const &oid1, size_t const &oid2) {
return std::norm(magnitude[oid1]) > std::norm(magnitude[oid2]);
});
} else {

__gnu_parallel::nth_element(begin, middle, end,
[&](size_t const &oid1, size_t const &oid2) {
return random_selector[oid1] < random_selector[oid2];
});
}


next_iteration_num_object = max_num_object;
}


void symbolic_iteration::finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, debug_t mid_step_function) {
if (next_iteration_num_object == 0) {
next_iteration.num_object = 0;
mid_step_function("prepare_final");
mid_step_function("final");
return;
}









mid_step_function("prepare_final");
next_iteration.num_object = next_iteration_num_object;


__gnu_parallel::sort(&next_oid[0], &next_oid[0] + next_iteration.num_object);


next_iteration.resize(next_iteration.num_object);


#pragma omp parallel for 
for (size_t oid = 0; oid < next_iteration.num_object; ++oid) {
auto id = next_oid[oid];


uint alignment_offset = get_alignment_offset(size[id]);
next_iteration.object_size[oid] = size[id];
next_iteration.object_begin[oid + 1] = size[id] + alignment_offset;
next_iteration.magnitude[oid] = magnitude[id];
}

__gnu_parallel::partial_sum(&next_iteration.object_begin[1],
&next_iteration.object_begin[1] + next_iteration.num_object + 1,
&next_iteration.object_begin[1]);

next_iteration.allocate(next_iteration.object_begin[next_iteration.num_object]);





mid_step_function("final");

#pragma omp parallel for 
for (size_t oid = 0; oid < next_iteration.num_object; ++oid) {
auto id = next_oid[oid];
auto this_parent_oid = parent_oid[id];

rule->populate_child_simple(&last_iteration.objects[last_iteration.object_begin[this_parent_oid]],
&last_iteration.objects[last_iteration.object_begin[this_parent_oid] + last_iteration.object_size[this_parent_oid]],
&next_iteration.objects[next_iteration.object_begin[oid]],
child_id[id]);
}
}


void iteration::apply_modifier(modifier_t const rule) {
#pragma omp parallel for 
for (size_t oid = 0; oid < num_object; ++oid)

rule(&objects[object_begin[oid]],
&objects[object_begin[oid] + object_size[oid]],
magnitude[oid]);
}


void iteration::normalize(debug_t mid_step_function) {
total_proba = 0;

if (num_object == 0) {
mid_step_function("normalize");
mid_step_function("end");
return;
}





mid_step_function("normalize");

#pragma omp parallel
{
#pragma omp for reduction(+:total_proba)
for (size_t oid = 0; oid < num_object; ++oid)
total_proba += std::norm(magnitude[oid]);

PROBA_TYPE normalization_factor = std::sqrt(total_proba);

if (normalization_factor != 1)
#pragma omp for
for (size_t oid = 0; oid < num_object; ++oid)
magnitude[oid] /= normalization_factor;
}

mid_step_function("end");
}
}