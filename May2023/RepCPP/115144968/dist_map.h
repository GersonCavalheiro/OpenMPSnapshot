#pragma once

#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <functional>
#include "bare_concurrent_map.h"
#include "dist_hasher.h"
#include "mpi_type.h"
#include "reducer.h"

namespace hpmr {

template <class K, class V, class H = std::hash<K>>
class DistMap {
public:
DistMap();

void reserve(const size_t n_keys_min);

size_t get_n_keys();

size_t get_n_buckets();

float get_load_factor();

float get_max_load_factor() const { return max_load_factor; }

void set_max_load_factor(const float max_load_factor);

void async_set(
const K& key,
const V& value,
const std::function<void(V&, const V&)>& reducer = Reducer<V>::overwrite);

void sync(
const std::function<void(V&, const V&)>& reducer = Reducer<V>::keep,
const bool verbose = false,
const int trunk_size = DEFAULT_TRUNK_SIZE);

V get(const K& key, const V& default_value = V());

void clear();

void clear_and_shrink();

template <class KR, class VR, class HR = std::hash<KR>>
DistMap<KR, VR, HR> mapreduce(
const std::function<
void(const K&, const V&, const std::function<void(const KR&, const VR&)>&)>& mapper,
const std::function<void(VR&, const VR&)>& reducer,
const bool verbose = false);

private:
int n_procs;

int proc_id;

H hasher;

float max_load_factor;

BareConcurrentMap<K, V, DistHasher<K, H>> local_map;

std::vector<BareConcurrentMap<K, V, DistHasher<K, H>>> remote_maps;

constexpr static int DEFAULT_TRUNK_SIZE = 1 << 20;

std::vector<int> generate_shuffled_procs();

int get_shuffled_id(const std::vector<int>& shuffled_procs);
};

template <class K, class V, class H>
DistMap<K, V, H>::DistMap() {
MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
remote_maps.resize(n_procs);
max_load_factor = local_map.get_max_load_factor();
}

template <class K, class V, class H>
void DistMap<K, V, H>::reserve(const size_t n_keys_min) {
local_map.reserve(n_keys_min / n_procs);
for (auto& remote_map : remote_maps) {
remote_map.reserve(n_keys_min / n_procs / n_procs);
}
}

template <class K, class V, class H>
size_t DistMap<K, V, H>::get_n_keys() {
const size_t local_n_keys = local_map.get_n_keys();
size_t n_keys;
MPI_Allreduce(&local_n_keys, &n_keys, 1, MpiType<size_t>::value, MPI_SUM, MPI_COMM_WORLD);
return n_keys;
}

template <class K, class V, class H>
size_t DistMap<K, V, H>::get_n_buckets() {
const size_t local_n_buckets = local_map.get_n_buckets();
size_t n_buckets;
MPI_Allreduce(&local_n_buckets, &n_buckets, 1, MpiType<size_t>::value, MPI_SUM, MPI_COMM_WORLD);
return n_buckets;
}

template <class K, class V, class H>
float DistMap<K, V, H>::get_load_factor() {
return static_cast<float>(get_n_buckets()) / get_n_keys();
}

template <class K, class V, class H>
void DistMap<K, V, H>::set_max_load_factor(const float max_load_factor) {
this->max_load_factor = max_load_factor;
local_map.set_max_load_factor(max_load_factor);
for (auto& remote_map : remote_maps) remote_map.set_max_load_factor(max_load_factor);
}

template <class K, class V, class H>
void DistMap<K, V, H>::async_set(
const K& key, const V& value, const std::function<void(V&, const V&)>& reducer) {
const size_t hash_value = hasher(key);
const size_t n_procs_u = static_cast<size_t>(n_procs);
const size_t dest_proc_id = hash_value % n_procs_u;
const size_t dist_hash_value = hash_value / n_procs_u;
if (dest_proc_id == static_cast<size_t>(proc_id)) {
local_map.async_set(key, dist_hash_value, value, reducer);
} else {
remote_maps[dest_proc_id].async_set(key, dist_hash_value, value, reducer);
}
}

template <class K, class V, class H>
V DistMap<K, V, H>::get(const K& key, const V& default_value) {
const size_t hash_value = hasher(key);
const size_t n_procs_u = static_cast<size_t>(n_procs);
const size_t dest_proc_id = hash_value % n_procs_u;
const size_t dist_hash_value = hash_value / n_procs_u;
V res;
if (dest_proc_id == static_cast<size_t>(proc_id)) {
res = local_map.get(key, dist_hash_value, default_value);
}
MPI_Bcast(&res, 1, MpiType<V>::value, dest_proc_id, MPI_COMM_WORLD);
return res;
}

template <class K, class V, class H>
void DistMap<K, V, H>::sync(
const std::function<void(V&, const V&)>& reducer, const bool verbose, const int trunk_size) {
assert(trunk_size > 0);
const bool report = proc_id == 0 && verbose;
if (report) printf("Syncing: ");

const auto& node_handler = [&](const K& key, const size_t hash_value, const V& value) {
local_map.async_set(key, hash_value, value, reducer);
};

const auto& shuffled_procs = generate_shuffled_procs();
const int shuffled_id = get_shuffled_id(shuffled_procs);

std::string send_buf;
std::string recv_buf;
char send_buf_char[trunk_size];
char recv_buf_char[trunk_size];
MPI_Request reqs[2];
MPI_Status stats[2];
for (int i = 1; i < n_procs; i++) {
const int dest_proc_id = shuffled_procs[(shuffled_id + i) % n_procs];
const int src_proc_id = shuffled_procs[(shuffled_id + n_procs - i) % n_procs];
remote_maps[dest_proc_id].sync(reducer);
send_buf = remote_maps[dest_proc_id].to_string();
remote_maps[dest_proc_id].clear();
size_t send_cnt = send_buf.size();
size_t recv_cnt;
MPI_Irecv(&recv_cnt, 1, MpiType<size_t>::value, src_proc_id, 0, MPI_COMM_WORLD, &reqs[0]);
MPI_Isend(&send_cnt, 1, MpiType<size_t>::value, dest_proc_id, 0, MPI_COMM_WORLD, &reqs[1]);
MPI_Waitall(2, reqs, stats);
size_t send_pos = 0;
size_t recv_pos = 0;
recv_buf.clear();
recv_buf.reserve(recv_cnt);
const size_t trunk_size_u = static_cast<size_t>(trunk_size);
while (send_pos < send_cnt || recv_pos < recv_cnt) {
const int recv_trunk_cnt =
(recv_cnt - recv_pos >= trunk_size_u) ? trunk_size : recv_cnt - recv_pos;
const int send_trunk_cnt =
(send_cnt - send_pos >= trunk_size_u) ? trunk_size : send_cnt - send_pos;
if (recv_trunk_cnt > 0) {
MPI_Irecv(
recv_buf_char, recv_trunk_cnt, MPI_CHAR, src_proc_id, 1, MPI_COMM_WORLD, &reqs[0]);
recv_pos += recv_trunk_cnt;
}
if (send_trunk_cnt > 0) {
send_buf.copy(send_buf_char, send_trunk_cnt, send_pos);
MPI_Issend(
send_buf_char, send_trunk_cnt, MPI_CHAR, dest_proc_id, 1, MPI_COMM_WORLD, &reqs[1]);
send_pos += send_trunk_cnt;
}
MPI_Waitall(2, reqs, stats);
recv_buf.append(recv_buf_char, recv_trunk_cnt);
}
remote_maps[dest_proc_id].from_string(recv_buf);
remote_maps[dest_proc_id].for_each(node_handler);
remote_maps[dest_proc_id].clear();
if (report) printf("%d/%d ", i - 1, n_procs - 1);
}
local_map.sync(reducer);
if (report) printf("#\n");
}

template <class K, class V, class H>
std::vector<int> DistMap<K, V, H>::generate_shuffled_procs() {
std::vector<int> res(n_procs);

if (proc_id == 0) {
for (int i = 0; i < n_procs; i++) res[i] = i;
srand(time(0));
for (int i = res.size() - 1; i > 0; i--) {
const int j = rand() % (i + 1);
if (i != j) {
const int tmp = res[i];
res[i] = res[j];
res[j] = tmp;
}
}
}

MPI_Bcast(res.data(), n_procs, MPI_INT, 0, MPI_COMM_WORLD);

return res;
}

template <class K, class V, class H>
int DistMap<K, V, H>::get_shuffled_id(const std::vector<int>& shuffled_procs) {
for (int i = 0; i < n_procs; i++) {
if (shuffled_procs[i] == proc_id) return i;
}
throw std::runtime_error("proc id does not exist in shuffled procs.");
}

template <class K, class V, class H>
void DistMap<K, V, H>::clear() {
local_map.clear();
for (auto& remote_map : remote_maps) remote_map.clear();
}

template <class K, class V, class H>
void DistMap<K, V, H>::clear_and_shrink() {
local_map.clear_and_shrink();
for (auto& remote_map : remote_maps) remote_map.clear_and_shrink();
}

template <class K, class V, class H>
template <class KR, class VR, class HR>
DistMap<KR, VR, HR> DistMap<K, V, H>::mapreduce(
const std::function<void(const K&, const V&, const std::function<void(const KR&, const VR&)>&)>&
mapper,
const std::function<void(VR&, const VR&)>& reducer,
const bool verbose) {
DistMap<KR, VR, HR> res;

const bool report = verbose && proc_id == 0;
if (report) {
const int n_threads = omp_get_max_threads();
printf("MapReduce on %d (%dx) node(s):\nMapping: ", n_procs, n_threads);
}

const auto& emit = [&](const KR& key, const VR& value) { res.async_set(key, value, reducer); };
const auto& node_handler = [&](const K& key, const size_t, const V& value) {
mapper(key, value, emit);
};
local_map.for_each(node_handler, verbose && proc_id == 0);

res.sync(reducer, verbose);

return res;
}

}  
