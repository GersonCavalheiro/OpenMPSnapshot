#ifndef AMGCL_COARSENING_POINTWISE_AGGREGATES_HPP
#define AMGCL_COARSENING_POINTWISE_AGGREGATES_HPP





#include <vector>
#include <cmath>

#include <amgcl/util.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>

namespace amgcl {
namespace coarsening {


class pointwise_aggregates {
public:
struct params : plain_aggregates::params {

unsigned block_size;

params() : block_size(1) {}

#ifndef AMGCL_NO_BOOST
params(const boost::property_tree::ptree &p)
: plain_aggregates::params(p),
AMGCL_PARAMS_IMPORT_VALUE(p, block_size)
{
check_params(p, {"eps_strong", "block_size"});
}

void get(boost::property_tree::ptree &p, const std::string &path) const {
plain_aggregates::params::get(p, path);
AMGCL_PARAMS_EXPORT_VALUE(p, path, block_size);
}
#endif
};

static const ptrdiff_t undefined = -1;
static const ptrdiff_t removed   = -2;

size_t count;

std::vector<char> strong_connection;

std::vector<ptrdiff_t> id;

template <class Matrix>
pointwise_aggregates(const Matrix &A, const params &prm, unsigned min_aggregate)
: count(0)
{
if (prm.block_size == 1) {
plain_aggregates aggr(A, prm);

remove_small_aggregates(A.nrows, 1, min_aggregate, aggr);

count = aggr.count;
strong_connection.swap(aggr.strong_connection);
id.swap(aggr.id);
} else {
strong_connection.resize( nonzeros(A) );
id.resize( rows(A) );

auto ap = backend::pointwise_matrix(A, prm.block_size);
auto &Ap = *ap;

plain_aggregates pw_aggr(Ap, prm);

remove_small_aggregates(
Ap.nrows, prm.block_size, min_aggregate, pw_aggr);

count = pw_aggr.count * prm.block_size;

#pragma omp parallel
{
std::vector<ptrdiff_t> j(prm.block_size);
std::vector<ptrdiff_t> e(prm.block_size);

#pragma omp for
for(ptrdiff_t ip = 0; ip < static_cast<ptrdiff_t>(Ap.nrows); ++ip) {
ptrdiff_t ia = ip * prm.block_size;

for(unsigned k = 0; k < prm.block_size; ++k, ++ia) {
id[ia] = prm.block_size * pw_aggr.id[ip] + k;

j[k] = A.ptr[ia];
e[k] = A.ptr[ia+1];
}

for(ptrdiff_t jp = Ap.ptr[ip], ep = Ap.ptr[ip+1]; jp < ep; ++jp) {
ptrdiff_t cp = Ap.col[jp];
bool      sp = (cp == ip) || pw_aggr.strong_connection[jp];

ptrdiff_t col_end = (cp + 1) * prm.block_size;

for(unsigned k = 0; k < prm.block_size; ++k) {
ptrdiff_t beg = j[k];
ptrdiff_t end = e[k];

while(beg < end && A.col[beg] < col_end) {
strong_connection[beg] = sp && A.col[beg] != (ia + k);
++beg;
}

j[k] = beg;
}
}
}
}
}
}

static void remove_small_aggregates(
size_t n, unsigned block_size, unsigned min_aggregate,
plain_aggregates &aggr
)
{
if (min_aggregate <= 1) return; 

std::vector<ptrdiff_t> count(aggr.count, 0);

for(size_t i = 0; i < n; ++i) {
ptrdiff_t id = aggr.id[i];
if (id != removed) ++count[id];
}

size_t m = 0;
for(size_t i = 0; i < aggr.count; ++i) {
if (block_size * count[i] < min_aggregate) {
count[i] = removed;
} else {
count[i] = m++;
}
}

aggr.count = m;

for(size_t i = 0; i < n; ++i) {
ptrdiff_t id = aggr.id[i];
if (id != removed) aggr.id[i] = count[id];
}
}
};

} 
} 


#endif
