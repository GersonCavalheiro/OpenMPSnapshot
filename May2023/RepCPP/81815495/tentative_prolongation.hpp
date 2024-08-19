#ifndef AMGCL_COARSENING_TENTATIVE_PROLONGATION_HPP
#define AMGCL_COARSENING_TENTATIVE_PROLONGATION_HPP





#include <vector>
#include <algorithm>

#include <memory>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/detail/qr.hpp>

namespace amgcl {
namespace coarsening {
namespace detail {
struct skip_negative {
const std::vector<ptrdiff_t> &key;
int block_size;

skip_negative(const std::vector<ptrdiff_t> &key, int block_size)
: key(key), block_size(block_size) { }

bool operator()(ptrdiff_t i, ptrdiff_t j) const {
return
static_cast<size_t>(key[i]) / block_size <
static_cast<size_t>(key[j]) / block_size;
}
};
} 


struct nullspace_params {
int cols;


std::vector<double> B;

nullspace_params() : cols(0) {}

#ifndef AMGCL_NO_BOOST
nullspace_params(const boost::property_tree::ptree &p)
: cols(p.get("cols", nullspace_params().cols))
{
double *b = 0;
b = p.get("B", b);

if (b) {
size_t rows = 0;
rows = p.get("rows", rows);

precondition(cols > 0,
"Error in nullspace parameters: "
"B is set, but cols is not"
);

precondition(rows > 0,
"Error in nullspace parameters: "
"B is set, but rows is not"
);

B.assign(b, b + rows * cols);
} else {
precondition(cols == 0,
"Error in nullspace parameters: "
"cols > 0, but B is empty"
);
}

check_params(p, {"cols", "rows", "B"});
}

void get(boost::property_tree::ptree&, const std::string&) const {}
#endif
};


template <class Matrix>
std::shared_ptr<Matrix> tentative_prolongation(
size_t n,
size_t naggr,
const std::vector<ptrdiff_t> aggr,
nullspace_params &nullspace,
int block_size
)
{
typedef typename backend::value_type<Matrix>::type value_type;
typedef typename backend::col_type<Matrix>::type col_type;

auto P = std::make_shared<Matrix>();

AMGCL_TIC("tentative");
if (nullspace.cols > 0) {
ptrdiff_t nba = naggr / block_size;

std::vector<ptrdiff_t> order(n);
for(size_t i = 0; i < n; ++i) order[i] = i;
std::stable_sort(order.begin(), order.end(), detail::skip_negative(aggr, block_size));
std::vector<ptrdiff_t> aggr_ptr(nba + 1, 0);
for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
ptrdiff_t a = aggr[order[i]];
if (a < 0) break;
++aggr_ptr[a / block_size + 1];
}
std::partial_sum(aggr_ptr.begin(), aggr_ptr.end(), aggr_ptr.begin());

P->set_size(n, nullspace.cols * nba);
P->ptr[0] = 0;

#pragma omp parallel for
for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
P->ptr[i+1] = aggr[i] < 0 ? 0 : nullspace.cols;

P->scan_row_sizes();
P->set_nonzeros();

std::vector<double> Bnew;
Bnew.resize(nba * nullspace.cols * nullspace.cols);

#pragma omp parallel
{
amgcl::detail::QR<double> qr;
std::vector<double> Bpart;

#pragma omp for
for(ptrdiff_t i = 0; i < nba; ++i) {
auto aggr_beg = aggr_ptr[i];
auto aggr_end = aggr_ptr[i+1];
auto d = aggr_end - aggr_beg;

Bpart.resize(d * nullspace.cols);

for(ptrdiff_t j = aggr_beg, jj = 0; j < aggr_end; ++j, ++jj) {
ptrdiff_t ib = nullspace.cols * order[j];
for(int k = 0; k < nullspace.cols; ++k)
Bpart[jj + d * k] = nullspace.B[ib + k];
}

qr.factorize(d, nullspace.cols, &Bpart[0], amgcl::detail::col_major);

for(int ii = 0, kk = 0; ii < nullspace.cols; ++ii)
for(int jj = 0; jj < nullspace.cols; ++jj, ++kk)
Bnew[i * nullspace.cols * nullspace.cols + kk] = qr.R(ii,jj);

for(ptrdiff_t j = aggr_beg, ii = 0; j < aggr_end; ++j, ++ii) {
col_type   *c = &P->col[P->ptr[order[j]]];
value_type *v = &P->val[P->ptr[order[j]]];

for(int jj = 0; jj < nullspace.cols; ++jj) {
c[jj] = i * nullspace.cols + jj;
v[jj] = qr.Q(ii,jj) * math::identity<value_type>();
}
}
}
}

std::swap(nullspace.B, Bnew);
} else {
P->set_size(n, naggr);
P->ptr[0] = 0;
#pragma omp parallel for
for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
P->ptr[i+1] = (aggr[i] >= 0);

P->set_nonzeros(P->scan_row_sizes());

#pragma omp parallel for
for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
if (aggr[i] >= 0) {
P->col[P->ptr[i]] = aggr[i];
P->val[P->ptr[i]] = math::identity<value_type>();
}
}
}
AMGCL_TOC("tentative");

return P;
}

} 
} 

#endif
