#pragma once
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
extern int cosmology_initialized;
extern double OMEGA_M;
extern double OMEGA_B;
extern double OMEGA_L;
extern double HUBBLE;
extern double LITTLE_H;
extern double SIGMA_8;
extern double NS;
extern int active_cosmology;
int init_cosmology(const int lasdamas_cosmology)__attribute__((warn_unused_result));
#ifdef __cplusplus
}
#endif
