

#pragma once

#include "header.h"

namespace trinity { namespace numeric {

void eigenDecompose(
const double* matrix, double* val, double* vec1, double* vec2
);
void interpolateTensor(
const double* matrix, double* result, int n
);
void doKroneckerProduct(
const double* vec1, const double* vec2, double* matrix
);
double computeQuality(
const double* point_a, const double* point_b, const double* point_c,
const double* tensor_a, const double* tensor_b, const double* tensor_c
);
void computeSteinerPoint(
const double* point_a, const double* point_b,
const double* tensor_a, const double* tensor_b,
double* result_point, double* result_tensor
);
double approxRiemannDist(
const double* point_a, const double* point_b, const double* tensor
);
double approxRiemannDist(
const double* point_a, const double* point_b,
const double* tensor_a, const double* tensor_b
);
void approxRiemannCircum(
const double* point_a, const double* point_b, const double* point_c,
const double* matrix, double* circum
);

}} 

