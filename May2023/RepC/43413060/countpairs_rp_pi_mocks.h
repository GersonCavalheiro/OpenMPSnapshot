#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include "defs.h"
#include <stdint.h> 
typedef struct{
uint64_t *npairs;
double *rupp;
double *rpavg;
double *weightavg;
double pimax;
int nbin;
int npibin;
} results_countpairs_mocks;
int countpairs_mocks(const int64_t ND1, void *theta1, void *phi1, void *czD1,
const int64_t ND2, void *theta2, void *phi2, void *czD2,
const int numthreads,
const int autocorr,
const char *binfile,
const double pimax,
const int cosmology,
results_countpairs_mocks *results,
struct config_options *options, struct extra_options *extra);
void free_results_mocks(results_countpairs_mocks *results);
#ifdef __cplusplus
}
#endif
