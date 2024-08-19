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
} results_countpairs_rp_pi;
extern int countpairs_rp_pi(const int64_t ND1, void *X1, void *Y1, void *Z1,
const int64_t ND2, void *X2, void *Y2, void *Z2,
const int numthreads,
const int autocorr,
const char *binfile,
const double pimax,
results_countpairs_rp_pi *results,
struct config_options *options,
struct extra_options *extra);
extern void free_results_rp_pi(results_countpairs_rp_pi *results);
#ifdef __cplusplus
}
#endif
