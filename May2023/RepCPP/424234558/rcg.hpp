#ifndef RCGPAR_RCG_HPP
#define RCGPAR_RCG_HPP

#include <vector>
#include <cstddef>
#include <fstream>

#include "Matrix.hpp"

#include <omp.h>
#include <algorithm>
#pragma omp declare reduction(vec_double_plus : std::vector<double> :	\
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

namespace rcgpar {
double digamma(double in);
void logsumexp(seamat::Matrix<double> &gamma_Z);
void logsumexp(seamat::Matrix<double> &gamma_Z, std::vector<double> &m);

double mixt_negnatgrad(const seamat::Matrix<double> &gamma_Z,
const std::vector<double> &N_k,
const seamat::Matrix<double> &logl, seamat::Matrix<double> &dL_dphi, bool mpi_mode = false);

void update_N_k(const seamat::Matrix<double> &gamma_Z, const std::vector<double> &log_times_observed, const std::vector<double> &alpha0, std::vector<double> &N_k, bool mpi_mode = false);

long double ELBO_rcg_mat(const seamat::Matrix<double> &logl, const seamat::Matrix<double> &gamma_Z,
const std::vector<double> &counts, const std::vector<double> &N_k,
const double bound_const, bool mpi_mode = false);

void revert_step(seamat::Matrix<double> &gamma_Z, const std::vector<double> &oldm);

double calc_bound_const(const std::vector<double> &log_times_observed,
const std::vector<double> &alpha0);

void add_alpha0_to_Nk(const std::vector<double> &alpha0,
std::vector<double> &N_k);
void rcg_optl_mat(const seamat::Matrix<double> &logl, const std::vector<double> &log_times_observed,
const std::vector<double> &alpha0,
const long double bound_const, const double tol, const uint16_t max_iters,
const bool mpi_mode, seamat::Matrix<double> &gamma_Z, std::ostream &log);
}

#endif
