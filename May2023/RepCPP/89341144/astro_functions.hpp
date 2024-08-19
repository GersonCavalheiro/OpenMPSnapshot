#pragma once

#include <float.h>
#include <math.h>
#include <cctype>
#include <vector>
#include "zero_finder.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double Mean2Eccentric(const double, const double);

void Conversion(const double *, double *, double *, const double);

double norm(const double *, const double *);

double norm2(const double *);

void vett(const double *, const double *, double *);

double asinh(double);

double acosh(double);

double tofabn(const double &, const double &, const double &);

void vers(const double *, double *);

double x2tof(const double &, const double &, const double &, const int);
