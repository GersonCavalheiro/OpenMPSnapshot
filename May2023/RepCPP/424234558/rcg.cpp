#include "rcg.hpp"


#include <assert.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>

#include "openmp_config.hpp"
#include "mpi_config.hpp"

namespace rcgpar {
double digamma(double x) {
double result = 0, xx, xx2, xx4;
assert(x > 0);
for ( ; x < 7; ++x)
result -= 1/x;
x -= 1.0/2.0;
xx = 1.0/x;
xx2 = xx*xx;
xx4 = xx2*xx2;
result += std::log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
return result;
}

void logsumexp(seamat::Matrix<double> &gamma_Z) {
uint32_t n_obs = gamma_Z.get_cols();
uint16_t n_groups = gamma_Z.get_rows();

#pragma omp parallel for schedule(static)
for (uint32_t i = 0; i < n_obs; ++i) {
double m = gamma_Z.log_sum_exp_col<double>(i);
for (uint16_t j = 0; j < n_groups; ++j) {
gamma_Z(j, i) -= m;
}
}
}

void logsumexp(seamat::Matrix<double> &gamma_Z, std::vector<double> &m) {
uint32_t n_obs = gamma_Z.get_cols();
uint16_t n_groups = gamma_Z.get_rows();

#pragma omp parallel for schedule(static)
for (uint32_t i = 0; i < n_obs; ++i) {
m[i] = gamma_Z.log_sum_exp_col<double>(i);
}

#pragma omp parallel for schedule(static)
for (uint16_t i = 0; i < n_groups; ++i) {
for (uint32_t j = 0; j < n_obs; ++j) {
gamma_Z(i, j) -= m[j];
}
}
}

double mixt_negnatgrad(const seamat::Matrix<double> &gamma_Z, const std::vector<double> &N_k, const seamat::Matrix<double> &logl, seamat::Matrix<double> &dL_dphi, bool mpi_mode) {
uint32_t n_obs = gamma_Z.get_cols();
uint16_t n_groups = gamma_Z.get_rows();

std::vector<double> colsums(n_obs, 0.0);
#pragma omp parallel for schedule(static) reduction(vec_double_plus:colsums)
for (uint16_t i = 0; i < n_groups; ++i) {
double digamma_N_k = digamma(N_k[i]) - 1.0;
for (uint32_t j = 0; j < n_obs; ++j) {
dL_dphi(i, j) = logl(i, j);
dL_dphi(i, j) += digamma_N_k - gamma_Z(i, j);
colsums[j] += dL_dphi(i, j) * std::exp(gamma_Z(i, j));
}
}

double newnorm = 0.0;
#pragma omp parallel for schedule(static) reduction(+:newnorm)
for (uint16_t i = 0; i < n_groups; ++i) {
for (uint32_t j = 0; j < n_obs; ++j) {
newnorm += std::exp(gamma_Z(i, j)) * (dL_dphi(i, j) - colsums[j]) * dL_dphi(i, j);
}
}

if (mpi_mode) {
#if defined(RCGPAR_MPI_SUPPORT) && (RCGPAR_MPI_SUPPORT) == 1
long double newnorm_partial = newnorm;
MPI_Allreduce(&newnorm_partial, &newnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}
return newnorm;
}

void update_N_k(const seamat::Matrix<double> &gamma_Z, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, std::vector<double> &N_k, bool mpi_mode) {
gamma_Z.exp_right_multiply(log_times_observed, N_k);
if (mpi_mode) {
#if defined(RCGPAR_MPI_SUPPORT) && (RCGPAR_MPI_SUPPORT) == 1
std::vector<double> N_k_partial = N_k;
MPI_Allreduce(&N_k_partial.front(), &N_k.front(), N_k.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}
std::transform(N_k.begin(), N_k.end(), alpha0.begin(), N_k.begin(), std::plus<double>());
}

long double ELBO_rcg_mat(const seamat::Matrix<double> &logl, const seamat::Matrix<double> &gamma_Z, const std::vector<double> &counts, const std::vector<double> &N_k, const double bound_const, const bool mpi_mode) {
long double bound = 0.0;
uint16_t n_groups = gamma_Z.get_rows();
uint32_t n_obs = gamma_Z.get_cols();
#pragma omp parallel for schedule(static) reduction(+:bound)
for (uint16_t i = 0; i < n_groups; ++i) {
for (uint32_t j = 0; j < n_obs; ++j) {
bound += std::exp(gamma_Z(i, j) + counts[j])*(logl(i, j) - gamma_Z(i, j));
}
}

if (mpi_mode) {
#if defined(RCGPAR_MPI_SUPPORT) && (RCGPAR_MPI_SUPPORT) == 1
long double bound_partial = bound;
MPI_Allreduce(&bound_partial, &bound, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}
bound += std::accumulate(N_k.begin(), N_k.end(), (double)0.0, [](double acc, double elem){ return acc + std::lgamma(elem); });
bound += bound_const;

return bound;
}

void revert_step(seamat::Matrix<double> &gamma_Z, const std::vector<double> &oldm) {
uint16_t n_groups = gamma_Z.get_rows();
uint32_t n_obs = gamma_Z.get_cols();
#pragma omp parallel for schedule(static)
for (uint16_t i = 0; i < n_groups; ++i) {
for (uint32_t j = 0; j < n_obs; ++j) {
gamma_Z(i, j) += oldm[j];
}
}
}

double calc_bound_const(const std::vector<double> &log_times_observed, const std::vector<double> &alpha0) {
double counts_sum = 0.0;
uint32_t n_obs = log_times_observed.size();
#pragma omp parallel for schedule(static) reduction(+:counts_sum)
for (uint32_t i = 0; i < n_obs; ++i) {
counts_sum += std::exp(log_times_observed[i]);
}

double alpha0_sum = 0.0;
double lgamma_alpha0_sum = 0.0;
uint16_t n_groups = alpha0.size();
#pragma omp parallel for schedule(static) reduction(+:alpha0_sum) reduction(+:lgamma_alpha0_sum)
for (uint32_t i = 0; i < n_groups; ++i) {
alpha0_sum += alpha0[i];
lgamma_alpha0_sum += std::lgamma(alpha0[i]);
}
double bound_const = std::lgamma(alpha0_sum);
bound_const -= std::lgamma(alpha0_sum + counts_sum);
bound_const -= lgamma_alpha0_sum;
return bound_const;
}

void rcg_optl_mat(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed,
const std::vector<double> &alpha0,
const long double bound_const, const double tol, const uint16_t max_iters,
const bool mpi_mode, seamat::Matrix<double> &gamma_Z, std::ostream &log) {
uint16_t n_groups = alpha0.size();
uint32_t n_obs = log_times_observed.size();

seamat::DenseMatrix<double> step(n_groups, n_obs, 0.0);
seamat::DenseMatrix<double> oldstep(n_groups, n_obs, 0.0);
std::vector<double> oldm(n_obs, 0.0);
double oldnorm = 1.0;
long double bound = -100000.0;

std::vector<double> N_k(n_groups);
update_N_k(gamma_Z, log_times_observed, alpha0, N_k, mpi_mode);

bool didreset = false;
for (uint16_t k = 0; k < max_iters; ++k) {
double newnorm = mixt_negnatgrad(gamma_Z, N_k, logl, step);
double beta_FR = newnorm/oldnorm;
oldnorm = newnorm;

if (didreset) {
oldstep *= 0.0;
} else if (beta_FR > 0) {
oldstep *= beta_FR;
step += oldstep;
}
didreset = false;

gamma_Z += step;

logsumexp(gamma_Z, oldm);
update_N_k(gamma_Z, log_times_observed, alpha0, N_k, mpi_mode);

long double oldbound = bound;
bound = ELBO_rcg_mat(logl, gamma_Z, log_times_observed, N_k, bound_const, mpi_mode);

if (bound < oldbound) {
didreset = true;
revert_step(gamma_Z, oldm);
if (beta_FR > 0) {
gamma_Z -= oldstep;
}

logsumexp(gamma_Z, oldm);
update_N_k(gamma_Z, log_times_observed, alpha0, N_k, mpi_mode);

bound = ELBO_rcg_mat(logl, gamma_Z, log_times_observed, N_k, bound_const, mpi_mode);
} else {
oldstep = step;
}
if (k % 5 == 0) {
log << "  " <<  "iter: " << k << ", bound: " << bound << ", |g|: " << newnorm << '\n';
}
if (bound - oldbound < tol && !didreset) {
logsumexp(gamma_Z, oldm);
log << std::endl;
return;
}
}
logsumexp(gamma_Z, oldm);
log << std::endl;
return;
}
}
