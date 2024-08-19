
#pragma once

#include "astro_functions.hpp"

void propagateKEP(const double *, const double *, double, double,
double *, double *);

void IC2par(const double *, const double *, double, double *);

void par2IC(const double *, double, double *, double *);

void cross(const double *, const double *, double *);
