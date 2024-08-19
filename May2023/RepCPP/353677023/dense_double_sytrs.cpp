#include "../../monolish_internal.hpp"
#include "../monolish_lapack_double.hpp"
#ifndef MONOLISH_USE_MKL
#include "../lapack.h"
#endif

#include <vector>

namespace monolish {

int internal::lapack::sytrs(const matrix::Dense<double> &A, vector<double> &B,
const std::vector<int> &ipiv) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

int info = 0;
const int M = (int)A.get_row();
const int N = (int)A.get_col();
const int nrhs = 1;
const double *Ad = A.data();
double *Bd = B.data();
const int *ipivd = ipiv.data();
const char U = 'U';


dsytrs_(&U, &M, &nrhs, Ad, &N, ipivd, Bd, &M, &info);

logger.func_out();
return info;
}

} 
