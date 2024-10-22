#ifndef AMGCL_SOLVER_IDRS_HPP
#define AMGCL_SOLVER_IDRS_HPP





#include <vector>
#include <algorithm>
#include <iostream>

#include <tuple>
#include <random>

#include <amgcl/backend/interface.hpp>
#include <amgcl/solver/detail/default_inner_product.hpp>
#include <amgcl/util.hpp>

#ifdef MPI_VERSION
#  include <amgcl/mpi/util.hpp>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace amgcl {
namespace solver {

template <
class Backend,
class InnerProduct = detail::default_inner_product
>
class idrs {
public:
typedef Backend backend_type;

typedef typename Backend::vector     vector;
typedef typename Backend::value_type value_type;
typedef typename Backend::params     backend_params;

typedef typename math::scalar_of<value_type>::type scalar_type;
typedef typename math::rhs_of<value_type>::type rhs_type;

typedef typename math::inner_product_impl<
typename math::rhs_of<value_type>::type
>::return_type coef_type;

struct params {
unsigned s;


scalar_type omega;

bool smoothing;


bool replacement;

unsigned maxiter;

scalar_type tol;

scalar_type abstol;

/
bool ns_search;

bool verbose;

params()
: s(4), omega(0.7), smoothing(false),
replacement(false), maxiter(100), tol(1e-8),
abstol(std::numeric_limits<scalar_type>::min()),
ns_search(false), verbose(false)
{ }

#ifndef AMGCL_NO_BOOST
params(const boost::property_tree::ptree &p)
: AMGCL_PARAMS_IMPORT_VALUE(p, s),
AMGCL_PARAMS_IMPORT_VALUE(p, omega),
AMGCL_PARAMS_IMPORT_VALUE(p, smoothing),
AMGCL_PARAMS_IMPORT_VALUE(p, replacement),
AMGCL_PARAMS_IMPORT_VALUE(p, maxiter),
AMGCL_PARAMS_IMPORT_VALUE(p, tol),
AMGCL_PARAMS_IMPORT_VALUE(p, abstol),
AMGCL_PARAMS_IMPORT_VALUE(p, ns_search),
AMGCL_PARAMS_IMPORT_VALUE(p, verbose)
{
check_params(p, {"s", "omega", "smoothing", "replacement",
"maxiter", "tol", "abstol", "ns_search", "verbose"});
}

void get(boost::property_tree::ptree &p, const std::string &path) const {
AMGCL_PARAMS_EXPORT_VALUE(p, path, s);
AMGCL_PARAMS_EXPORT_VALUE(p, path, omega);
AMGCL_PARAMS_EXPORT_VALUE(p, path, smoothing);
AMGCL_PARAMS_EXPORT_VALUE(p, path, replacement);
AMGCL_PARAMS_EXPORT_VALUE(p, path, maxiter);
AMGCL_PARAMS_EXPORT_VALUE(p, path, tol);
AMGCL_PARAMS_EXPORT_VALUE(p, path, abstol);
AMGCL_PARAMS_EXPORT_VALUE(p, path, ns_search);
AMGCL_PARAMS_EXPORT_VALUE(p, path, verbose);
}
#endif
} prm;

idrs(
size_t n,
const params &prm = params(),
const backend_params &bprm = backend_params(),
const InnerProduct &inner_product = InnerProduct()
)
: prm(prm), n(n), inner_product(inner_product),
M(prm.s, prm.s),
f(prm.s), c(prm.s),
r(Backend::create_vector(n, bprm)),
v(Backend::create_vector(n, bprm)),
t(Backend::create_vector(n, bprm))
{
static const scalar_type one = math::identity<scalar_type>();
static const scalar_type zero = math::zero<scalar_type>();

if (prm.smoothing) {
x_s = Backend::create_vector(n, bprm);
r_s = Backend::create_vector(n, bprm);
}

G.reserve(prm.s);
U.reserve(prm.s);
for(unsigned i = 0; i < prm.s; ++i) {
G.push_back(Backend::create_vector(n, bprm));
U.push_back(Backend::create_vector(n, bprm));
}

P.reserve(prm.s);
{
std::vector<rhs_type> p(n);

int pid = inner_product.rank();

#pragma omp parallel
{
#ifdef _OPENMP
int tid = omp_get_thread_num();
int nt = omp_get_max_threads();
#else
int tid = 0;
int nt = 1;
#endif

std::mt19937 rng(pid * nt + tid);
std::uniform_real_distribution<scalar_type> rnd(-1, 1);

for(unsigned j = 0; j < prm.s; ++j) {
#pragma omp for
for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
p[i] = math::constant<rhs_type>(rnd(rng));

#pragma omp single
{
P.push_back(Backend::copy_vector(p, bprm));
}
}
}

for(unsigned j = 0; j < prm.s; ++j) {
for(unsigned k = 0; k < j; ++k) {
coef_type alpha = inner_product(*P[k], *P[j]);
backend::axpby(-alpha, *P[k], one, *P[j]);
}
scalar_type norm_pj = norm(*P[j]);
backend::axpby(math::inverse(norm_pj), *P[j], zero, *P[j]);
}
}
}


template <class Matrix, class Precond, class Vec1, class Vec2>
std::tuple<size_t, scalar_type> operator()(
Matrix  const &A,
Precond const &Prec,
Vec1    const &rhs,
Vec2          &x
) const
{
static const scalar_type one = math::identity<scalar_type>();
static const scalar_type zero = math::zero<scalar_type>();

ios_saver ss(std::cout);

scalar_type norm_rhs = norm(rhs);
if (norm_rhs < amgcl::detail::eps<scalar_type>(1)) {
if (prm.ns_search) {
norm_rhs = math::identity<scalar_type>();
} else {
backend::clear(x);
return std::make_tuple(0, norm_rhs);
}
}

scalar_type eps = std::max(prm.tol * norm_rhs, prm.abstol);

backend::residual(rhs, A, x, *r);

scalar_type res_norm = norm(*r);
if (res_norm <= eps) {
return std::make_tuple(0, res_norm / norm_rhs);
}

if (prm.smoothing) {
backend::copy( x, *x_s);
backend::copy(*r, *r_s);
}

coef_type om = math::identity<coef_type>();

for(unsigned i = 0; i < prm.s; ++i) {
backend::clear(*G[i]);
backend::clear(*U[i]);

for(unsigned j = 0; j < prm.s; ++j)
M(i, j) = (i == j);
}

size_t iter = 0;
while(iter < prm.maxiter && res_norm > eps) {
for(unsigned i = 0; i < prm.s; ++i)
f[i] = inner_product(*r, *P[i]);

for(unsigned k = 0; k < prm.s; ++k) {
backend::copy(*r, *v);

for(unsigned i = k; i < prm.s; ++i) {
c[i] = f[i];
for(unsigned j = k; j < i; ++j)
c[i] -= M(i, j) * c[j];
c[i] = math::inverse(M(i, i)) * c[i];

backend::axpby(-c[i], *G[i], one, *v);
}

Prec.apply(*v, *t);

backend::axpby(om, *t, c[k], *U[k]);
for(unsigned i = k+1; i < prm.s; ++i)
backend::axpby(c[i], *U[i], one, *U[k]);

backend::spmv(one, A, *U[k], zero, *G[k]);

for(unsigned i = 0; i < k; ++i) {
coef_type alpha = inner_product(*G[k], *P[i]) / M(i, i);

backend::axpby(-alpha, *G[i], one, *G[k]);
backend::axpby(-alpha, *U[i], one, *U[k]);
}

for(unsigned i = k; i < prm.s; ++i)
M(i, k) = inner_product(*G[k], *P[i]);

precondition(!math::is_zero(M(k, k)), "IDR(s) breakdown: zero M[k,k]");

coef_type beta = math::inverse(M(k, k)) * f[k];
backend::axpby(-beta, *G[k], one, *r);
backend::axpby( beta, *U[k], one,  x);

res_norm = norm(*r);

if (prm.smoothing) {
backend::axpbypcz(one, *r_s, -one, *r, zero, *t);
coef_type gamma = inner_product(*t, *r_s) / inner_product(*t, *t);
backend::axpby(-gamma, *t, one, *r_s);
backend::axpbypcz(-gamma, *x_s, gamma, x, one, *x_s);
res_norm = norm(*r_s);
}

if (prm.verbose && iter % 5 == 0)
std::cout << iter << "\t" << std::scientific << res_norm / norm_rhs << std::endl;
if (res_norm <= eps || ++iter >= prm.maxiter) break;

for(unsigned i = k + 1; i < prm.s; ++i)
f[i] -= beta * M(i, k);
}

if (res_norm <= eps || iter >= prm.maxiter) break;


Prec.apply(*r, *v);
backend::spmv(one, A, *v, zero, *t);

om = omega(*t, *r);
precondition(!math::is_zero(om), "IDR(s) breakdown: zero omega");

backend::axpby(-om, *t, one, *r);
backend::axpby( om, *v, one,  x);

if (prm.replacement) {
backend::residual(rhs, A, x, *r);
}
res_norm = norm(*r);

if (prm.smoothing) {
backend::axpbypcz(one, *r_s, -one, *r, zero, *t);
coef_type gamma = inner_product(*t, *r_s) / inner_product(*t, *t);
backend::axpby(-gamma, *t, one, *r_s);
backend::axpbypcz(-gamma, *x_s, gamma, x, one, *x_s);
res_norm = norm(*r_s);
}

++iter;
}

if (prm.smoothing)
backend::copy(*x_s, x);

return std::make_tuple(iter, res_norm / norm_rhs);
}


template <class Precond, class Vec1, class Vec2>
std::tuple<size_t, scalar_type> operator()(
Precond const &P,
Vec1    const &rhs,
Vec2          &x
) const
{
return (*this)(P.system_matrix(), P, rhs, x);
}

size_t bytes() const {
size_t b = 0;

b += M.size() * sizeof(coef_type);

b += backend::bytes(f);
b += backend::bytes(c);

b += backend::bytes(*r);
b += backend::bytes(*v);
b += backend::bytes(*t);

if (x_s) b += backend::bytes(*x_s);
if (r_s) b += backend::bytes(*r_s);

for(const auto &v : P) b += backend::bytes(*v);
for(const auto &v : G) b += backend::bytes(*v);
for(const auto &v : U) b += backend::bytes(*v);

return b;
}

friend std::ostream& operator<<(std::ostream &os, const idrs &s) {
return os
<< "Type:             IDR(" << s.prm.s << ")"
<< "\nUnknowns:         " << s.n
<< "\nMemory footprint: " << human_readable_memory(s.bytes())
<< std::endl;
}

private:
size_t n;

InnerProduct inner_product;

mutable multi_array<coef_type,2> M;
mutable std::vector<coef_type> f, c;

std::shared_ptr<vector> r, v, t;
std::shared_ptr<vector> x_s;
std::shared_ptr<vector> r_s;

std::vector< std::shared_ptr<vector> > P, G, U;


template <class Vec>
scalar_type norm(const Vec &x) const {
return std::abs(sqrt(inner_product(x, x)));
}

template <class Vector1, class Vector2>
coef_type omega(const Vector1 &t, const Vector2 &s) const {
scalar_type norm_t = norm(t);
scalar_type norm_s = norm(s);

coef_type   ts  = inner_product(t, s);
scalar_type rho = math::norm(ts / (norm_t * norm_s));
coef_type   om  = ts / (norm_t * norm_t);

if (rho < prm.omega)
om *= prm.omega/rho;

return om;
}
};

} 
} 

#endif
