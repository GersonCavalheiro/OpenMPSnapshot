#ifndef AMGCL_RELAXATION_GAUSS_SEIDEL_HPP
#define AMGCL_RELAXATION_GAUSS_SEIDEL_HPP





#include <numeric>

#include <memory>

#include <amgcl/backend/interface.hpp>
#include <amgcl/util.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace amgcl {
namespace relaxation {


template <class Backend>
struct gauss_seidel {
struct params {
bool serial;

params() : serial(false) {}

#ifndef AMGCL_NO_BOOST
params(const boost::property_tree::ptree &p)
: AMGCL_PARAMS_IMPORT_VALUE(p, serial)
{
check_params(p, {"serial"});
}

void get(boost::property_tree::ptree &p, const std::string &path) const {
AMGCL_PARAMS_EXPORT_VALUE(p, path, serial);
}
#endif
};

bool is_serial;

template <class Matrix>
gauss_seidel( const Matrix &A, const params &prm, const typename Backend::params&)
: is_serial(prm.serial || num_threads() < 4)
{
if(!is_serial) {
forward  = std::make_shared< parallel_sweep<true>  >(A);
backward = std::make_shared< parallel_sweep<false> >(A);
}
}

template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
void apply_pre(
const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP&
) const
{
if (is_serial)
serial_sweep(A, rhs, x, true);
else
forward->sweep(rhs, x);
}

template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
void apply_post(
const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP&
) const
{
if (is_serial)
serial_sweep(A, rhs, x, false);
else
backward->sweep(rhs, x);
}

template <class Matrix, class VectorRHS, class VectorX>
void apply(const Matrix &A, const VectorRHS &rhs, VectorX &x) const
{
backend::clear(x);
if (is_serial) {
serial_sweep(A, rhs, x, true);
serial_sweep(A, rhs, x, false);
} else {
forward->sweep(rhs, x);
backward->sweep(rhs, x);
}
}

size_t bytes() const {
size_t b = 0;
if (forward)  b += forward->bytes();
if (backward) b += backward->bytes();
return b;
}

private:
static int num_threads() {
#ifdef _OPENMP
return omp_get_max_threads();
#else
return 1;
#endif
}

static int thread_id() {
#ifdef _OPENMP
return omp_get_thread_num();
#else
return 0;
#endif
}

template <class Matrix, class VectorRHS, class VectorX>
static void serial_sweep(
const Matrix &A, const VectorRHS &rhs, VectorX &x, bool forward)
{
typedef typename backend::value_type<Matrix>::type val_type;
typedef typename math::rhs_of<val_type>::type rhs_type;

const ptrdiff_t n = backend::rows(A);

const ptrdiff_t beg = forward ? 0 : n-1;
const ptrdiff_t end = forward ? n : -1;
const ptrdiff_t inc = forward ? 1 : -1;

for(ptrdiff_t i = beg; i != end; i += inc) {
val_type D = math::identity<val_type>();
rhs_type X;
X = rhs[i];

for (auto a = backend::row_begin(A, i); a; ++a) {
ptrdiff_t c = a.col();
val_type  v = a.value();

if (c == i)
D = v;
else
X -= v * x[c];
}

x[i] = math::inverse(D) * X;
}
}

template <bool forward>
struct parallel_sweep {
typedef typename Backend::value_type value_type;
typedef typename math::rhs_of<value_type>::type rhs_type;

struct task {
ptrdiff_t beg, end;
task(ptrdiff_t beg, ptrdiff_t end) : beg(beg), end(end) {}
};

int nthreads;

std::vector< std::vector<task>       > tasks;
std::vector< std::vector<ptrdiff_t>  > ptr;
std::vector< std::vector<ptrdiff_t>  > col;
std::vector< std::vector<value_type> > val;
std::vector< std::vector<ptrdiff_t>  > ord;

template <class Matrix>
parallel_sweep(const Matrix &A)
: nthreads(num_threads()), tasks(nthreads),
ptr(nthreads), col(nthreads), val(nthreads), ord(nthreads)
{
ptrdiff_t n    = backend::rows(A);
ptrdiff_t nlev = 0;

std::vector<ptrdiff_t> level(n, 0);
std::vector<ptrdiff_t> order(n, 0);

ptrdiff_t beg = forward ? 0 : n-1;
ptrdiff_t end = forward ? n :  -1;
ptrdiff_t inc = forward ? 1 :  -1;

for(ptrdiff_t i = beg; i != end; i += inc) {
ptrdiff_t l = level[i];

for(auto a = row_begin(A, i); a; ++a) {
ptrdiff_t c = a.col();

if (forward) {
if (c >= i) continue;
} else {
if (c <= i) continue;
}

l = std::max(l, level[c]+1);
}

level[i] = l;
nlev = std::max(nlev, l+1);
}


std::vector<ptrdiff_t> start(nlev+1, 0);

for(ptrdiff_t i = 0; i < n; ++i)
++start[level[i]+1];

std::partial_sum(start.begin(), start.end(), start.begin());

for(ptrdiff_t i = 0; i < n; ++i)
order[start[level[i]]++] = i;

std::rotate(start.begin(), start.end() - 1, start.end());
start[0] = 0;


std::vector<ptrdiff_t> thread_rows(nthreads, 0);
std::vector<ptrdiff_t> thread_cols(nthreads, 0);

#pragma omp parallel
{
int tid = thread_id();
tasks[tid].reserve(nlev);

for(ptrdiff_t lev = 0; lev < nlev; ++lev) {
ptrdiff_t lev_size = start[lev+1] - start[lev];
ptrdiff_t chunk_size = (lev_size + nthreads - 1) / nthreads;

ptrdiff_t beg = std::min(tid * chunk_size, lev_size);
ptrdiff_t end = std::min(beg + chunk_size, lev_size);

beg += start[lev];
end += start[lev];

tasks[tid].push_back(task(beg, end));

thread_rows[tid] += end - beg;
for(ptrdiff_t i = beg; i < end; ++i) {
ptrdiff_t j = order[i];
thread_cols[tid] += row_nonzeros(A, j);
}
}
}

#pragma omp parallel
{
int tid = thread_id();

col[tid].reserve(thread_cols[tid]);
val[tid].reserve(thread_cols[tid]);
ord[tid].reserve(thread_rows[tid]);
ptr[tid].reserve(thread_rows[tid] + 1);
ptr[tid].push_back(0);

for(task &t : tasks[tid]) {
ptrdiff_t loc_beg = ptr[tid].size() - 1;
ptrdiff_t loc_end = loc_beg;

for(ptrdiff_t r = t.beg; r < t.end; ++r, ++loc_end) {
ptrdiff_t i = order[r];

ord[tid].push_back(i);

for(auto a = row_begin(A, i); a; ++a) {
col[tid].push_back(a.col());
val[tid].push_back(a.value());
}

ptr[tid].push_back(col[tid].size());
}

t.beg = loc_beg;
t.end = loc_end;
}
}
}

template <class Vector1, class Vector2>
void sweep(const Vector1 &rhs, Vector2 &x) const {
#pragma omp parallel
{
int tid = thread_id();

for(const task &t : tasks[tid]) {
for(ptrdiff_t r = t.beg; r < t.end; ++r) {
ptrdiff_t i   = ord[tid][r];
ptrdiff_t beg = ptr[tid][r];
ptrdiff_t end = ptr[tid][r+1];

value_type D = math::identity<value_type>();
rhs_type X;
X = rhs[i];

for(ptrdiff_t j = beg; j < end; ++j) {
ptrdiff_t  c = col[tid][j];
value_type v = val[tid][j];

if (c == i)
D = v;
else
X -= v * x[c];
}

x[i] = math::inverse(D) * X;
}

#pragma omp barrier
;
}
}
}

size_t bytes() const {
size_t b = 0;

for(int i = 0; i < nthreads; ++i) {
b += sizeof(task) * tasks[i].size();
b += backend::bytes(ptr[i]);
b += backend::bytes(col[i]);
b += backend::bytes(val[i]);
b += backend::bytes(ord[i]);
}

return b;
}
};

std::shared_ptr< parallel_sweep<true>  > forward;
std::shared_ptr< parallel_sweep<false> > backward;
};

} 

namespace backend {

template <class Backend>
struct relaxation_is_supported<
Backend,
relaxation::gauss_seidel,
typename std::enable_if<
!Backend::provides_row_iterator::value
>::type
> : std::false_type
{};

} 
} 


#endif
