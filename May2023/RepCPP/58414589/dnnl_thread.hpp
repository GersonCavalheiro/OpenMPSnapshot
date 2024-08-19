

#ifndef COMMON_DNNL_THREAD_HPP
#define COMMON_DNNL_THREAD_HPP

#include <algorithm>
#include <functional>
#include <mutex>

#include "utils.hpp"
#include "z_magic.hpp"


#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "counting_barrier.hpp"
#endif

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "common/ittnotify.hpp"
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
#define DNNL_THR_SYNC 1
inline int dnnl_get_max_threads() {
return 1;
}
inline int dnnl_in_parallel() {
return 0;
}
inline void dnnl_thr_barrier() {}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#include "omp.h"
#define DNNL_THR_SYNC 1
inline int dnnl_get_max_threads() {
return omp_get_max_threads();
}
inline int dnnl_in_parallel() {
return omp_in_parallel();
}
inline void dnnl_thr_barrier() {
#pragma omp barrier
}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"
#define DNNL_THR_SYNC 0
inline int dnnl_get_max_threads() {
return tbb::this_task_arena::max_concurrency();
}
inline int dnnl_in_parallel() {
return 0;
}
inline void dnnl_thr_barrier() {
assert(!"no barrier in TBB");
}

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include <thread>
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#define DNNL_THR_SYNC 0

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace threadpool_utils {


void activate_threadpool(dnnl::threadpool_interop::threadpool_iface *tp);

void deactivate_threadpool();

dnnl::threadpool_interop::threadpool_iface *get_active_threadpool();

int get_max_concurrency();

int &get_threadlocal_max_concurrency();

} 
} 
} 

inline int dnnl_get_max_threads() {
using namespace dnnl::impl::threadpool_utils;
dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();

int max_concurrency = dnnl::impl::threadpool_utils::get_max_concurrency();

return tp ? std::max(1, tp->get_num_threads()) : max_concurrency;
}
inline int dnnl_in_parallel() {
using namespace dnnl::impl::threadpool_utils;
dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
return tp ? tp->get_in_parallel() : 0;
}
inline void dnnl_thr_barrier() {
assert(!"no barrier with THREADPOOL");
}
#endif


inline int dnnl_get_current_num_threads() {
if (dnnl_in_parallel()) return 1;
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
return omp_get_max_threads();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
return tbb::this_task_arena::max_concurrency();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
using namespace dnnl::impl::threadpool_utils;
dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
return (tp) ? dnnl_get_max_threads() : 1;
#else
return 1;
#endif
}

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#define PRAGMA_OMP(...) PRAGMA_MACRO(CHAIN2(omp, __VA_ARGS__))
#define OMP_GET_THREAD_NUM() omp_get_thread_num()
#define OMP_GET_NUM_THREADS() omp_get_num_threads()
#else
#define PRAGMA_OMP(...)
#define OMP_GET_THREAD_NUM() 0
#define OMP_GET_NUM_THREADS() 1
#endif

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#define PRAGMA_OMP_SIMD(...)
#else
#define PRAGMA_OMP_SIMD(...) PRAGMA_MACRO(CHAIN2(omp, simd __VA_ARGS__))
#endif 

#if (defined(__clang_major__) \
&& (__clang_major__ < 3 \
|| (__clang_major__ == 3 && __clang_minor__ < 9))) \
|| (defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1700) \
|| (!defined(__INTEL_COMPILER) && !defined(__clang__) \
&& (defined(_MSC_VER) || __GNUC__ < 6 \
|| (__GNUC__ == 6 && __GNUC_MINOR__ < 1)))
#define simdlen(x)
#endif 

namespace dnnl {
namespace impl {

inline bool dnnl_thr_syncable() {
return DNNL_THR_SYNC == 1;
}

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
T n_min = 1;
T &n_my = n_end;
if (team <= 1 || n == 0) {
n_start = 0;
n_my = n;
} else if (n_min == 1) {
T n1 = utils::div_up(n, (T)team);
T n2 = n1 - 1;
T T1 = n - n2 * (T)team;
n_my = (T)tid < T1 ? n1 : n2;
n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
}

n_end += n_start;
}

template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end, T nx, T &nx_start,
T &nx_end, T nx_divider) {
const T grp_count = nstl::min(nx_divider, static_cast<T>(nthr));
const int grp_size_big = nthr / static_cast<int>(grp_count) + 1;
const int grp_size_small = nthr / static_cast<int>(grp_count);
const int n_grp_big = nthr % static_cast<int>(grp_count);
const int threads_in_big_groups = n_grp_big * grp_size_big;

const int ithr_bound_distance = ithr - threads_in_big_groups;
T grp, grp_ithr, grp_nthr;
if (ithr_bound_distance < 0) { 
grp = ithr / grp_size_big;
grp_ithr = ithr % grp_size_big;
grp_nthr = grp_size_big;
} else { 
grp = n_grp_big + ithr_bound_distance / grp_size_small;
grp_ithr = ithr_bound_distance % grp_size_small;
grp_nthr = grp_size_small;
}

balance211(nx, grp_count, grp, nx_start, nx_end);
balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}




inline int adjust_num_threads(int nthr, dim_t work_amount) {
if (nthr == 0) nthr = dnnl_get_current_num_threads();
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
return (work_amount == 1 || omp_in_parallel()) ? 1 : nthr;
#else
return (int)std::min((dim_t)nthr, work_amount);
#endif
}

static inline void parallel(int nthr, const std::function<void(int, int)> &f) {
nthr = adjust_num_threads(nthr, INT64_MAX);
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
for (int i = 0; i < nthr; ++i) {
f(i, nthr);
}
#else
#if defined(DNNL_ENABLE_ITT_TASKS)
auto task_primitive_kind = itt::primitive_task_get_current_kind();
bool itt_enable = itt::get_itt(itt::__itt_task_level_high);
#endif
if (nthr == 1) {
f(0, 1);
return;
}
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#pragma omp parallel num_threads(nthr)
{
int nthr_ = omp_get_num_threads();
int ithr_ = omp_get_thread_num();
assert(nthr_ == nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
if (ithr_ && itt_enable) itt::primitive_task_start(task_primitive_kind);
#endif
f(ithr_, nthr_);
#if defined(DNNL_ENABLE_ITT_TASKS)
if (ithr_ && itt_enable) itt::primitive_task_end();
#endif
}
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB
tbb::parallel_for(
0, nthr,
[&](int ithr) {
#if defined(DNNL_ENABLE_ITT_TASKS)
bool mark_task = itt::primitive_task_get_current_kind()
== primitive_kind::undefined;
if (mark_task && itt_enable)
itt::primitive_task_start(task_primitive_kind);
#endif
f(ithr, nthr);
#if defined(DNNL_ENABLE_ITT_TASKS)
if (mark_task && itt_enable) itt::primitive_task_end();
#endif
},
tbb::static_partitioner());
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
using namespace dnnl::impl::threadpool_utils;
dnnl::threadpool_interop::threadpool_iface *tp = get_active_threadpool();
if (!tp || dnnl_in_parallel()) {
threadpool_utils::deactivate_threadpool();
for (int ithr = 0; ithr < nthr; ithr++) {
f(ithr, nthr);
}
threadpool_utils::activate_threadpool(tp);
} else {
bool async = tp->get_flags()
& dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS;
counting_barrier_t b;
if (async) b.init(nthr);
tp->parallel_for(nthr, [&, tp](int ithr, int nthr) {
bool is_master = threadpool_utils::get_active_threadpool() == tp;
if (!is_master) {
threadpool_utils::activate_threadpool(tp);
#if defined(DNNL_ENABLE_ITT_TASKS)
if (itt_enable) itt::primitive_task_start(task_primitive_kind);
#endif
}
f(ithr, nthr);
if (!is_master) {
#if defined(DNNL_ENABLE_ITT_TASKS)
if (itt_enable) itt::primitive_task_end();
#endif
threadpool_utils::deactivate_threadpool();
}
if (async) b.notify();
});
if (async) b.wait();
}
#endif
#endif
}



static inline void for_nd(const int ithr, const int nthr, dim_t D0,
const std::function<void(dim_t)> &f) {
dim_t start {0}, end {0};
balance211(D0, nthr, ithr, start, end);
for (dim_t d0 = start; d0 < end; ++d0)
f(d0);
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
const std::function<void(dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0};
utils::nd_iterator_init(start, d0, D0, d1, D1);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(d0, d1);
utils::nd_iterator_step(d0, D0, d1, D1);
}
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
dim_t D2, const std::function<void(dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0}, d2 {0};
utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(d0, d1, d2);
utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
}
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
dim_t D2, dim_t D3,
const std::function<void(dim_t, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2 * D3;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0};
utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(d0, d1, d2, d3);
utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
}
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
dim_t D2, dim_t D3, dim_t D4,
const std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0};
utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(d0, d1, d2, d3, d4);
utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
}
}
static inline void for_nd(const int ithr, const int nthr, dim_t D0, dim_t D1,
dim_t D2, dim_t D3, dim_t D4, dim_t D5,
const std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)>
&f) {
const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0}, d5 {0};
utils::nd_iterator_init(
start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(d0, d1, d2, d3, d4, d5);
utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
}
}


static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
const std::function<void(int, int, dim_t)> &f) {
dim_t start {0}, end {0};
balance211(D0, nthr, ithr, start, end);
for (dim_t d0 = start; d0 < end; ++d0)
f(ithr, nthr, d0);
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
dim_t D1, const std::function<void(int, int, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0};
utils::nd_iterator_init(start, d0, D0, d1, D1);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(ithr, nthr, d0, d1);
utils::nd_iterator_step(d0, D0, d1, D1);
}
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
dim_t D1, dim_t D2,
const std::function<void(int, int, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0}, d2 {0};
utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(ithr, nthr, d0, d1, d2);
utils::nd_iterator_step(d0, D0, d1, D1, d2, D2);
}
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
dim_t D1, dim_t D2, dim_t D3,
const std::function<void(int, int, dim_t, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2 * D3;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0};
utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(ithr, nthr, d0, d1, d2, d3);
utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3);
}
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
dim_t D1, dim_t D2, dim_t D3, dim_t D4,
const std::function<void(int, int, dim_t, dim_t, dim_t, dim_t, dim_t)>
&f) {
const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0};
utils::nd_iterator_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(ithr, nthr, d0, d1, d2, d3, d4);
utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
}
}
static inline void for_nd_ext(const int ithr, const int nthr, dim_t D0,
dim_t D1, dim_t D2, dim_t D3, dim_t D4, dim_t D5,
const std::function<void(
int, int, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
if (work_amount == 0) return;
dim_t start {0}, end {0};
balance211(work_amount, nthr, ithr, start, end);

dim_t d0 {0}, d1 {0}, d2 {0}, d3 {0}, d4 {0}, d5 {0};
utils::nd_iterator_init(
start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
for (dim_t iwork = start; iwork < end; ++iwork) {
f(ithr, nthr, d0, d1, d2, d3, d4, d5);
utils::nd_iterator_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
}
}


static inline void parallel_nd_ext(
int nthr, dim_t D0, const std::function<void(int, int, dim_t)> &f) {
const dim_t work_amount = D0;
nthr = adjust_num_threads(nthr, work_amount);
if (nthr)
parallel(nthr,
[&](int ithr, int nthr) { for_nd_ext(ithr, nthr, D0, f); });
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1,
const std::function<void(int, int, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1;
nthr = adjust_num_threads(nthr, work_amount);
if (nthr)
parallel(nthr,
[&](int ithr, int nthr) { for_nd_ext(ithr, nthr, D0, D1, f); });
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2,
const std::function<void(int, int, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2;
nthr = adjust_num_threads(nthr, work_amount);
if (nthr)
parallel(nthr, [&](int ithr, int nthr) {
for_nd_ext(ithr, nthr, D0, D1, D2, f);
});
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2,
dim_t D3,
const std::function<void(int, int, dim_t, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2 * D3;
nthr = adjust_num_threads(nthr, work_amount);
if (nthr)
parallel(nthr, [&](int ithr, int nthr) {
for_nd_ext(ithr, nthr, D0, D1, D2, D3, f);
});
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2,
dim_t D3, dim_t D4,
const std::function<void(int, int, dim_t, dim_t, dim_t, dim_t, dim_t)>
&f) {
const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
nthr = adjust_num_threads(nthr, work_amount);
if (nthr)
parallel(nthr, [&](int ithr, int nthr) {
for_nd_ext(ithr, nthr, D0, D1, D2, D3, D4, f);
});
}
static inline void parallel_nd_ext(int nthr, dim_t D0, dim_t D1, dim_t D2,
dim_t D3, dim_t D4, dim_t D5,
const std::function<void(
int, int, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
nthr = adjust_num_threads(nthr, work_amount);
if (nthr)
parallel(nthr, [&](int ithr, int nthr) {
for_nd_ext(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
});
}


static inline void parallel_nd(dim_t D0, const std::function<void(dim_t)> &f) {
int nthr = adjust_num_threads(dnnl_get_current_num_threads(), D0);
if (nthr)
parallel(nthr, [&](int ithr, int nthr) { for_nd(ithr, nthr, D0, f); });
}
static inline void parallel_nd(
dim_t D0, dim_t D1, const std::function<void(dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1;
int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
if (nthr)
parallel(nthr,
[&](int ithr, int nthr) { for_nd(ithr, nthr, D0, D1, f); });
}
static inline void parallel_nd(dim_t D0, dim_t D1, dim_t D2,
const std::function<void(dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2;
int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
if (nthr)
parallel(nthr,
[&](int ithr, int nthr) { for_nd(ithr, nthr, D0, D1, D2, f); });
}
static inline void parallel_nd(dim_t D0, dim_t D1, dim_t D2, dim_t D3,
const std::function<void(dim_t, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2 * D3;
int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
if (nthr)
parallel(nthr, [&](int ithr, int nthr) {
for_nd(ithr, nthr, D0, D1, D2, D3, f);
});
}
static inline void parallel_nd(dim_t D0, dim_t D1, dim_t D2, dim_t D3, dim_t D4,
const std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t)> &f) {
const dim_t work_amount = D0 * D1 * D2 * D3 * D4;
int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
if (nthr)
parallel(nthr, [&](int ithr, int nthr) {
for_nd(ithr, nthr, D0, D1, D2, D3, D4, f);
});
}
static inline void parallel_nd(dim_t D0, dim_t D1, dim_t D2, dim_t D3, dim_t D4,
dim_t D5,
const std::function<void(dim_t, dim_t, dim_t, dim_t, dim_t, dim_t)>
&f) {
const dim_t work_amount = D0 * D1 * D2 * D3 * D4 * D5;
int nthr = adjust_num_threads(dnnl_get_current_num_threads(), work_amount);
if (nthr)
parallel(nthr, [&](int ithr, int nthr) {
for_nd(ithr, nthr, D0, D1, D2, D3, D4, D5, f);
});
}



template <typename... Args>
void parallel_nd_in_omp(Args &&...args) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
for_nd(0, 1, utils::forward<Args>(args)...);
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
for_nd(omp_get_thread_num(), omp_get_num_threads(),
utils::forward<Args>(args)...);
#elif (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB \
|| DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL)
assert(!"parallel_nd_in_omp() is not supported by this DNNL_CPU_RUNTIME");
#endif
}

} 
} 

#endif

