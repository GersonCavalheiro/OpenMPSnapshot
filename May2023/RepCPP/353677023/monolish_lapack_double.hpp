#pragma once
#include "../../../include/monolish/common/monolish_common.hpp"
#include <stdio.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif


namespace monolish::internal::lapack {



int syevd(matrix::Dense<double> &A, vector<double> &W, const char *jobz,
const char *uplo);


int sygvd(matrix::Dense<double> &A, matrix::Dense<double> &B, vector<double> &W,
const int itype, const char *jobz, const char *uplo);



int getrf(matrix::Dense<double> &A, std::vector<int> &ipiv);


int getrs(const matrix::Dense<double> &A, vector<double> &B,
const std::vector<int> &ipiv);


int sytrf(matrix::Dense<double> &A, std::vector<int> &ipiv);


int sytrs(const matrix::Dense<double> &A, vector<double> &B,
const std::vector<int> &ipiv);

} 
