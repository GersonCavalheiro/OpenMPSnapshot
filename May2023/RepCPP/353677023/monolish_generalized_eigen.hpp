#pragma once
#include <omp.h>
#include <vector>

#include "./monolish_solver.hpp"
#include "monolish/common/monolish_common.hpp"

namespace monolish {

namespace generalized_eigen {





template <typename MATRIX, typename Float>
class LOBPCG : public solver::solver<MATRIX, Float> {
private:
[[nodiscard]] int monolish_LOBPCG(MATRIX &A, MATRIX &B, vector<Float> &lambda,
matrix::Dense<Float> &x, int itype);

public:
[[nodiscard]] int solve(MATRIX &A, MATRIX &B, vector<Float> &lambda,
matrix::Dense<Float> &x, int itype);

void create_precond(MATRIX &A) {
throw std::runtime_error("this precond. is not impl.");
}

void apply_precond(const vector<Float> &r, vector<Float> &z) {
throw std::runtime_error("this precond. is not impl.");
}


[[nodiscard]] std::string name() const {
return "monolish::generalized_eigen::LOBPCG";
}


[[nodiscard]] std::string solver_name() const { return "LOBPCG"; }
};




template <typename MATRIX, typename Float>
class DC : public solver::solver<MATRIX, Float> {
private:
[[nodiscard]] int LAPACK_DC(MATRIX &A, MATRIX &B, vector<Float> &lambda,
int itype);

public:
[[nodiscard]] int solve(MATRIX &A, MATRIX &B, vector<Float> &lambda,
int itype);

void create_precond(MATRIX &A) {
throw std::runtime_error("this precond. is not impl.");
}

void apply_precond(const vector<Float> &r, vector<Float> &z) {
throw std::runtime_error("this precond. is not impl.");
}


[[nodiscard]] std::string name() const {
return "monolish::generalized_eigen::DC";
}


[[nodiscard]] std::string solver_name() const { return "DC"; }
};

} 
} 
