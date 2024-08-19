#pragma once
#include "../../../include/monolish/common/monolish_common.hpp"
#include <stdio.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif


namespace monolish::internal::lapack {



int syevd(matrix::Dense<float> &A, vector<float> &W, const char *jobz,
const char *uplo);


int sygvd(matrix::Dense<float> &A, matrix::Dense<float> &B, vector<float> &W,
const int itype, const char *jobz, const char *uplo);



int getrf(matrix::Dense<float> &A, std::vector<int> &ipiv);


int getrs(const matrix::Dense<float> &A, vector<float> &B,
const std::vector<int> &ipiv);


int sytrf(matrix::Dense<float> &A, std::vector<int> &ipiv);


int sytrs(const matrix::Dense<float> &A, vector<float> &B,
const std::vector<int> &ipiv);

} 
