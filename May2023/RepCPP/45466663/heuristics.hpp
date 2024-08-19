#pragma once

#include <algorithm>

#ifdef __OMP
#include <omp.h>
#endif

#include "core/Work.hpp"
#include "sample/Sample.hpp"
#include "util/Options.hpp"
#include "set_manipulators.hpp"


static inline size_t get_num_threads(const Options& options)
{
#ifdef __OMP
const size_t num_threads  = options.num_threads
? options.num_threads
: omp_get_max_threads();
omp_set_num_threads(num_threads);
#else
(void) options;
const size_t num_threads = 1;
#endif
return num_threads;
}

static inline size_t get_thread_id()
{
#ifdef __OMP
return omp_get_thread_num();
#else
return 0;
#endif
}

using getiter_t = pq_iter_t(PQuery<Placement>&,double const);

template< typename F >
static Work heuristic_( Sample<Placement>& sample,
const Options& options,
F filterstop)
{
Work result;
compute_and_set_lwr(sample);

const auto num_threads = get_num_threads(options);

std::vector<Work> workvec(num_threads);

#ifdef __OMP
#pragma omp parallel for schedule(dynamic)
#endif
for( size_t i = 0; i < sample.size(); ++i ) {
auto &pq = sample[i];
const auto tid = get_thread_id();

auto end = filterstop( pq, options.prescoring_threshold );

for( auto iter = pq.begin(); iter != end; ++iter ) {
workvec[tid].add(iter->branch_id(), pq.sequence_id());
}
}
merge(result, workvec);
return result;
}

Work dynamic_heuristic( Sample<Placement>& sample,
const Options& options)
{
return heuristic_<getiter_t>( sample, options, until_accumulated_reached );
}

Work fixed_heuristic( Sample<Placement>& sample,
const Options& options)
{
return heuristic_<getiter_t>( sample, options, until_top_percent );
}

Work baseball_heuristic( Sample<Placement>& sample,
const Options& options)
{
Work result;

const auto num_threads = get_num_threads(options);

const double strike_box = 3;
const size_t max_strikes = 6;
const size_t max_pitches = 40;

std::vector<Work> workvec(num_threads);
#ifdef __OMP
#pragma omp parallel for schedule(dynamic)
#endif
for (size_t i = 0; i < sample.size(); ++i) {
auto &pq = sample[i];
const auto tid = get_thread_id();

assert(pq.size());
sort_by_logl(pq);
const double best_logl = pq[0].likelihood();
const double thresh = best_logl - strike_box;
auto keep_iter = std::find_if(pq.begin(), pq.end(),
[thresh](const auto& p){
return (p.likelihood() < thresh);
}
);

const auto hits = std::distance(pq.begin(), keep_iter);

size_t to_add = std::min(max_pitches - hits, max_strikes);

std::advance(keep_iter, to_add);

for (auto iter = pq.begin(); iter != keep_iter; ++iter) {
workvec[tid].add(iter->branch_id(), pq.sequence_id());
}

}
merge(result, workvec);
return result;
}

Work apply_heuristic(Sample<Placement>& sample,
const Options& options)
{
if (options.baseball) {
return baseball_heuristic(sample, options);
} else if (options.prescoring_by_percentage) {
return fixed_heuristic(sample, options);
} else {
return dynamic_heuristic(sample, options);
}
}
