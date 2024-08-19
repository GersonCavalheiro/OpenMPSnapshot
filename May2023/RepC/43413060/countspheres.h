#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include "defs.h" 
#include <stdint.h> 
typedef struct{
double **pN;
double rmax;
int nbin;
int nc;
int num_pN;
} results_countspheres;
extern int countspheres(const int64_t np, void * restrict X, void * restrict Y, void * restrict Z,
const double rmax, const int nbin, const int nc,
const int num_pN,
unsigned long seed,
results_countspheres *results,
struct config_options *options,
struct extra_options *extra);
extern void free_results_countspheres(results_countspheres *results);
#ifdef __cplusplus
}
#endif
