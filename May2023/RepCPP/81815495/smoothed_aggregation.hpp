#ifndef AMGCL_MPI_COARSENING_SMOOTHED_AGGREGATION_HPP
#define AMGCL_MPI_COARSENING_SMOOTHED_AGGREGATION_HPP





#include <tuple>
#include <memory>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>
#include <amgcl/coarsening/detail/galerkin.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/mpi/coarsening/pmis.hpp>

namespace amgcl {
namespace mpi {
namespace coarsening {

template <class Backend>
struct smoothed_aggregation {
typedef typename Backend::value_type value_type;
typedef typename math::scalar_of<value_type>::type scalar_type;
typedef backend::crs<value_type> build_matrix;

struct params {
typedef typename pmis<Backend>::params aggr_params;
aggr_params aggr;

scalar_type relax;

bool estimate_spectral_radius;

int power_iters;

params()
: relax(1.0f), estimate_spectral_radius(false), power_iters(0)
{ }

#ifndef AMGCL_NO_BOOST
params(const boost::property_tree::ptree &p)
: AMGCL_PARAMS_IMPORT_CHILD(p, aggr),
AMGCL_PARAMS_IMPORT_VALUE(p, relax),
AMGCL_PARAMS_IMPORT_VALUE(p, estimate_spectral_radius),
AMGCL_PARAMS_IMPORT_VALUE(p, power_iters)
{
check_params(p, {"aggr", "relax", "estimate_spectral_radius", "power_iters"});
}

void get(boost::property_tree::ptree &p, const std::string &path) const {
AMGCL_PARAMS_EXPORT_CHILD(p, path, aggr);
AMGCL_PARAMS_EXPORT_VALUE(p, path, relax);
AMGCL_PARAMS_EXPORT_VALUE(p, path, estimate_spectral_radius);
AMGCL_PARAMS_EXPORT_VALUE(p, path, power_iters);
}
#endif
} prm;

smoothed_aggregation(const params &prm = params()) : prm(prm) {}

std::tuple<
std::shared_ptr< distributed_matrix<Backend> >,
std::shared_ptr< distributed_matrix<Backend> >
>
transfer_operators(const distributed_matrix<Backend> &A) {
typedef distributed_matrix<Backend> DM;
typedef backend::crs<char> bool_matrix;

pmis<Backend> aggr(A, prm.aggr);
prm.aggr.eps_strong *= 0.5;

communicator comm = A.comm();
const build_matrix &A_loc = *A.local();
const build_matrix &A_rem = *A.remote();

bool_matrix &S_loc = *aggr.conn->local();
bool_matrix &S_rem = *aggr.conn->remote();

AMGCL_TIC("filtered matrix");
ptrdiff_t n = A.loc_rows();

scalar_type omega = prm.relax;
if (prm.estimate_spectral_radius) {
omega *= static_cast<scalar_type>(4.0/3) / backend::spectral_radius<true>(A, prm.power_iters);
} else {
omega *= static_cast<scalar_type>(2.0/3);
}

auto af_loc = std::make_shared<build_matrix>();
auto af_rem = std::make_shared<build_matrix>();

build_matrix &Af_loc = *af_loc;
build_matrix &Af_rem = *af_rem;

backend::numa_vector<value_type> Af_loc_val(S_loc.nnz, false);
backend::numa_vector<value_type> Af_rem_val(S_rem.nnz, false);

Af_loc.own_data = false;
Af_loc.nrows = S_loc.nrows;
Af_loc.ncols = S_loc.ncols;
Af_loc.nnz   = S_loc.nnz;
Af_loc.ptr   = S_loc.ptr;
Af_loc.col   = S_loc.col;
Af_loc.val   = Af_loc_val.data();

Af_rem.own_data = false;
Af_rem.nrows = S_rem.nrows;
Af_rem.ncols = S_rem.ncols;
Af_rem.nnz   = S_rem.nnz;
Af_rem.ptr   = S_rem.ptr;
Af_rem.col   = S_rem.col;
Af_rem.val   = Af_rem_val.data();

backend::numa_vector<value_type> Df(n, false);

#pragma omp parallel for
for(ptrdiff_t i = 0; i < n; ++i) {
ptrdiff_t loc_head = Af_loc.ptr[i];
ptrdiff_t rem_head = Af_rem.ptr[i];

value_type dia_f = math::zero<value_type>();

for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j)
if (A_loc.col[j] == i || !S_loc.val[j]) dia_f += A_loc.val[j];

for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j)
if (!S_rem.val[j]) dia_f += A_rem.val[j];

dia_f = -omega * math::inverse(dia_f);

for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j) {
if (A_loc.col[j] == i) {
Af_loc.val[loc_head++] = (1 - omega) * math::identity<value_type>();
} else if(S_loc.val[j]) {
Af_loc.val[loc_head++] = dia_f * A_loc.val[j];
}
}

for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j) {
if (S_rem.val[j]) {
Af_rem.val[rem_head++] = dia_f * A_rem.val[j];
}
}
}

auto Af = std::make_shared<DM>(comm, af_loc, af_rem);
AMGCL_TOC("filtered matrix");

AMGCL_TIC("smoothing");
auto P = product(*Af, *aggr.p_tent);
AMGCL_TOC("smoothing");

return std::make_tuple(P, transpose(*P));
}

std::shared_ptr< distributed_matrix<Backend> >
coarse_operator(
const distributed_matrix<Backend> &A,
const distributed_matrix<Backend> &P,
const distributed_matrix<Backend> &R
) const
{
return amgcl::coarsening::detail::galerkin(A, P, R);
}

};

template <class Backend>
unsigned block_size(const smoothed_aggregation<Backend> &c) {
return c.prm.aggr.block_size;
}

} 
} 
} 

#endif
