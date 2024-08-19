#pragma once

#include "defs.hpp"

namespace bnmf_algs {
namespace bld {


template <typename T> struct EMResult {
public:

matrix_t<T> S_pjk;

matrix_t<T> S_ipk;

matrix_t<T> X_full;

matrix_t<double> logW;

matrix_t<double> logH;

vector_t<double> log_PS;

public:

EMResult() = default;


EMResult(matrix_t<T> S_pjk, matrix_t<T> S_ipk, matrix_t<T> X_full,
matrix_t<T> logW, matrix_t<T> logH, vector_t<T> EM_bound)
: S_pjk(std::move(S_pjk)), S_ipk(std::move(S_ipk)),
X_full(std::move(X_full)), logW(std::move(logW)),
logH(std::move(logH)), log_PS(std::move(log_PS)) {}
};
} 
} 
