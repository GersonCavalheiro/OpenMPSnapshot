#include "../../monolish_internal.hpp"
#include "../monolish_lapack_float.hpp"
#ifndef MONOLISH_USE_MKL
#include "../lapack.h"
#endif

#include <vector>

namespace monolish {

int internal::lapack::sytrs(const matrix::Dense<float> &A, vector<float> &B,
const std::vector<int> &ipiv) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

int info = 0;
const int M = (int)A.get_row();
const int N = (int)A.get_col();
const int nrhs = 1;
const float *Ad = A.data();
float *Bd = B.data();
const int *ipivd = ipiv.data();
const char U = 'U';


ssytrs_(&U, &M, &nrhs, Ad, &N, ipivd, Bd, &M, &info);

logger.func_out();
return info;
}

} 
