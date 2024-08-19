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
} results_countpairs_s_mu;
extern int countpairs_s_mu(const int64_t ND1, void *X1, void *Y1, void *Z1,
const int64_t ND2, void *X2, void *Y2, void *Z2,
const int numthreads,
const int autocorr,
const char *sbinfile,
const double mu_max,
const int nmu_bins, 
results_countpairs_s_mu *results,
struct config_options *options,
struct extra_options *extra);
extern void free_results_s_mu(results_countpairs_s_mu *results);
#ifdef __cplusplus
}
#endif
