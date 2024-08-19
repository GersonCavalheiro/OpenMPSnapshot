#ifndef AMGCL_COARSENING_PLAIN_AGGREGATES_HPP
#define AMGCL_COARSENING_PLAIN_AGGREGATES_HPP





#include <vector>
#include <numeric>

#include <amgcl/util.hpp>
#include <amgcl/backend/builtin.hpp>

namespace amgcl {
namespace coarsening {




struct plain_aggregates {
struct params {

float eps_strong;

params() : eps_strong(0.08f) {}

#ifndef AMGCL_NO_BOOST
params(const boost::property_tree::ptree &p)
: AMGCL_PARAMS_IMPORT_VALUE(p, eps_strong)
{
check_params(p, {"eps_strong", "block_size"});
}

void get(boost::property_tree::ptree &p, const std::string &path) const {
AMGCL_PARAMS_EXPORT_VALUE(p, path, eps_strong);
}
#endif
};

static const ptrdiff_t undefined = -1;
static const ptrdiff_t removed   = -2;

size_t count;


std::vector<char> strong_connection;


std::vector<ptrdiff_t> id;


template <class Matrix>
plain_aggregates(const Matrix &A, const params &prm)
: count(0),
strong_connection( backend::nonzeros(A) ),
id( backend::rows(A) )
{
typedef typename backend::value_type<Matrix>::type value_type;
typedef typename math::scalar_of<value_type>::type scalar_type;

scalar_type eps_squared = prm.eps_strong * prm.eps_strong;

const size_t n = rows(A);


auto dia = diagonal(A);
#pragma omp parallel for
for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
value_type eps_dia_i = eps_squared * (*dia)[i];

for(ptrdiff_t j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
ptrdiff_t c = A.col[j];
value_type v = A.val[j];

strong_connection[j] = (c != i) && (eps_dia_i * (*dia)[c] < v * v);
}
}



size_t max_neib = 0;
for(size_t i = 0; i < n; ++i) {
ptrdiff_t j = A.ptr[i], e = A.ptr[i+1];
max_neib    = std::max<size_t>(max_neib, e - j);

ptrdiff_t state = removed;
for(; j < e; ++j)
if (strong_connection[j]) {
state = undefined;
break;
}

id[i] = state;
}

std::vector<ptrdiff_t> neib;
neib.reserve(max_neib);

for(size_t i = 0; i < n; ++i) {
if (id[i] != undefined) continue;

ptrdiff_t cur_id = static_cast<ptrdiff_t>(count++);
id[i] = cur_id;

neib.clear();
for(ptrdiff_t j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
ptrdiff_t c = A.col[j];
if (strong_connection[j] && id[c] != removed) {
id[c] = cur_id;
neib.push_back(c);
}
}

for(ptrdiff_t c : neib) {
for(ptrdiff_t j = A.ptr[c], e = A.ptr[c+1]; j < e; ++j) {
ptrdiff_t cc = A.col[j];
if (strong_connection[j] && id[cc] == undefined)
id[cc] = cur_id;
}
}
}

if (!count) throw error::empty_level();

std::vector<ptrdiff_t> cnt(count, 0);
for(ptrdiff_t i : id)
if (i >= 0) cnt[i] = 1;
std::partial_sum(cnt.begin(), cnt.end(), cnt.begin());

if (static_cast<ptrdiff_t>(count) > cnt.back()) {
count = cnt.back();

for(size_t i = 0; i < n; ++i)
if (id[i] >= 0) id[i] = cnt[id[i]] - 1;
}
}
};

} 
} 

#endif
