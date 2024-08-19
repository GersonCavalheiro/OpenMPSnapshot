#pragma once
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
#include "cosmology_params.h"
double get_age(const double z);
double agefunc(double z,void *params);
double get_comoving_distance(const double zlow, const double z);
double comoving_distance_func(const double z, void *params);
double epeebles(const double z);
#ifdef __cplusplus
}
#endif
