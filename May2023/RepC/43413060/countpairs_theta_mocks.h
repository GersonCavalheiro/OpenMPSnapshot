#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include "defs.h"
#include <stdint.h> 
typedef struct{
uint64_t *npairs;
double *theta_upp;
double *theta_avg;
double *weightavg;
int nbin;
} results_countpairs_theta;
extern int countpairs_theta_mocks(const int64_t ND1, void *phi1, void *theta1,
const int64_t ND2, void *phi2, void *theta2,
const int numthreads,
const int autocorr,
const char *binfile,
results_countpairs_theta *results,
struct config_options *options, struct extra_options *extra);
extern void free_results_countpairs_theta(results_countpairs_theta *results);
#ifdef __cplusplus
}
#endif
