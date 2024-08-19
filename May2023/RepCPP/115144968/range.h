#pragma once

#include <functional>
#include "dist_map.h"

namespace hpmr {
template <class T>
class Range {
public:
Range(const T start, const T end, const T step = 1) : start(start), end(end), step(step) {}

template <class K, class V, class H = std::hash<K>>
DistMap<K, V, H> mapreduce(
const std::function<void(const T, const std::function<void(const K&, const V&)>&)>& mapper,
const std::function<void(V&, const V&)>& reducer,
const bool verbose = false);


private:
T start;

T end;

T step;
};

template <class T>
template <class K, class V, class H>
DistMap<K, V, H> Range<T>::mapreduce(
const std::function<void(const T, const std::function<void(const K&, const V&)>&)>& mapper,
const std::function<void(V&, const V&)>& reducer,
const bool verbose) {
DistMap<K, V, H> res;
int proc_id;
int n_procs;
MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
double target_progress = 0.1;

const auto& emit = [&](const K& key, const V& value) { res.async_set(key, value, reducer); };
if (verbose && proc_id == 0) {
const int n_threads = omp_get_max_threads();
printf("MapReduce on %d (%dx) node(s):\nMapping: ", n_procs, n_threads);
}

#pragma omp parallel for schedule(dynamic, 3)
for (T i = start + proc_id * step; i < end; i += step * n_procs) {
mapper(i, emit);
const int thread_id = omp_get_thread_num();
if (verbose && proc_id == 0 && thread_id == 0) {
const double current_progress = (i - start) * 100.0 / (end - start);
while (target_progress <= current_progress) {
printf("%.1f%% ", target_progress);
target_progress *= 2;
}
}
}
if (verbose && proc_id == 0) printf("\n");

res.sync(reducer, verbose);

return res;
}

}  
