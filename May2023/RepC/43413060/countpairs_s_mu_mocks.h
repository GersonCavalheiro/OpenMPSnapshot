#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include "defs.h"
#include <stdint.h> 
typedef struct{
uint64_t *npairs;
double *supp;
double *savg;
double mu_max;
double mu_min;
double *weightavg;
int nsbin;
int nmu_bins;
} results_countpairs_mocks_s_mu;
int countpairs_mocks_s_mu(const int64_t ND1, void *theta1, void *phi1, void *czD1,
const int64_t ND2, void *theta2, void *phi2, void *czD2,
const int numthreads,
const int autocorr,
const char *sbinfile,
const double mu_max,
const int nmu_bins,
const int cosmology,
results_countpairs_mocks_s_mu *results,
struct config_options *options,
struct extra_options *extra);
void free_results_mocks_s_mu(results_countpairs_mocks_s_mu *results);
#ifdef __cplusplus
}
#endif
