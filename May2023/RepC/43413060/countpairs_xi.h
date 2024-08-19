#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include "defs.h" 
#include <stdint.h> 
typedef struct{
uint64_t *npairs;
double *xi;
double *rupp;
double *ravg;
double *weightavg;
int nbin;
} results_countpairs_xi;
extern void free_results_xi(results_countpairs_xi *results);
extern int countpairs_xi(const int64_t ND1, void * restrict X1, void * restrict Y1, void * restrict Z1,
const double boxsize,
const int numthreads,
const char *binfile,
results_countpairs_xi *results,
struct config_options *options,
struct extra_options *extra);
#ifdef __cplusplus
}
#endif
