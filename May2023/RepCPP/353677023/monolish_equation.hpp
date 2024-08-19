#pragma once
#include <vector>

#include "./monolish_solver.hpp"
#include "monolish/common/monolish_common.hpp"
#include <functional>

namespace monolish {

namespace equation {





template <typename MATRIX, typename Float>
class none : public monolish::solver::solver<MATRIX, Float> {
public:
void create_precond(MATRIX &A);
void apply_precond(const vector<Float> &r, vector<Float> &z);
[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);


[[nodiscard]] std::string name() const { return "monolish::equation::none"; }


[[nodiscard]] std::string solver_name() const { return "none"; }
};




template <typename MATRIX, typename Float>
class CG : public monolish::solver::solver<MATRIX, Float> {
private:
[[nodiscard]] int monolish_CG(MATRIX &A, vector<Float> &x, vector<Float> &b);

public:

[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);

void create_precond(MATRIX &A) {
throw std::runtime_error("this precond. is not impl.");
}

void apply_precond(const vector<Float> &r, vector<Float> &z) {
throw std::runtime_error("this precond. is not impl.");
}


[[nodiscard]] std::string name() const { return "monolish::equation::CG"; }


[[nodiscard]] std::string solver_name() const { return "CG"; }
};





template <typename MATRIX, typename Float>
class BiCGSTAB : public monolish::solver::solver<MATRIX, Float> {
private:
[[nodiscard]] int monolish_BiCGSTAB(MATRIX &A, vector<Float> &x,
vector<Float> &b);

public:

[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);

void create_precond(MATRIX &A) {
throw std::runtime_error("this precond. is not impl.");
}

void apply_precond(const vector<Float> &r, vector<Float> &z) {
throw std::runtime_error("this precond. is not impl.");
}


[[nodiscard]] std::string name() const {
return "monolish::equation::BiCGSTAB";
}


[[nodiscard]] std::string solver_name() const { return "BiCGSTAB"; }
};




template <typename MATRIX, typename Float>
class Jacobi : public monolish::solver::solver<MATRIX, Float> {
private:
[[nodiscard]] int monolish_Jacobi(MATRIX &A, vector<Float> &x,
vector<Float> &b);

public:

[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
void create_precond(MATRIX &A);
void apply_precond(const vector<Float> &r, vector<Float> &z);


[[nodiscard]] std::string name() const {
return "monolish::equation::Jacobi";
}


[[nodiscard]] std::string solver_name() const { return "Jacobi"; }
};




template <typename MATRIX, typename Float>
class SOR : public monolish::solver::solver<MATRIX, Float> {
private:
[[nodiscard]] int monolish_SOR(MATRIX &A, vector<Float> &x, vector<Float> &b);

public:

[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
void create_precond(MATRIX &A);
void apply_precond(const vector<Float> &r, vector<Float> &z);


[[nodiscard]] std::string name() const { return "monolish::equation::SOR"; }


[[nodiscard]] std::string solver_name() const { return "SOR"; }
};




template <typename MATRIX, typename Float>
class IC : public monolish::solver::solver<MATRIX, Float> {
private:
int cusparse_IC(MATRIX &A, vector<Float> &x, vector<Float> &b);
void *matM = 0, *matL = 0;
void *infoM = 0, *infoL = 0, *infoLt = 0;
void *cusparse_handle = nullptr;
int bufsize;
monolish::vector<double> buf;
monolish::vector<Float> zbuf;

public:
~IC();

[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
void create_precond(MATRIX &A);
void apply_precond(const vector<Float> &r, vector<Float> &z);


std::string name() const { return "monolish::equation::IC"; }


std::string solver_name() const { return "IC"; }
};




template <typename MATRIX, typename Float>
class ILU : public monolish::solver::solver<MATRIX, Float> {
private:
int cusparse_ILU(MATRIX &A, vector<Float> &x, vector<Float> &b);
void *matM = 0, *matL = 0, *matU = 0;
void *infoM = 0, *infoL = 0, *infoU = 0;
void *cusparse_handle = nullptr;
int bufsize;
monolish::vector<double> buf;
monolish::vector<Float> zbuf;

public:
~ILU();

[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
void create_precond(MATRIX &A);
void apply_precond(const vector<Float> &r, vector<Float> &z);


std::string name() const { return "monolish::equation::ILU"; }


std::string solver_name() const { return "ILU"; }
};




template <typename MATRIX, typename Float>
class LU : public monolish::solver::solver<MATRIX, Float> {
private:
int lib = 1; 
[[nodiscard]] int mumps_LU(MATRIX &A, vector<double> &x, vector<double> &b);
[[nodiscard]] int cusolver_LU(MATRIX &A, vector<double> &x,
vector<double> &b);

public:
[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
[[nodiscard]] int solve(MATRIX &A, vector<Float> &xb);

void create_precond(MATRIX &A) {
throw std::runtime_error("this precond. is not impl.");
}
void apply_precond(const vector<Float> &r, vector<Float> &z) {
throw std::runtime_error("this precond. is not impl.");
}


[[nodiscard]] std::string name() const { return "monolish::equation::LU"; }


[[nodiscard]] std::string solver_name() const { return "LU"; }
};




template <typename MATRIX, typename Float>
class QR : public monolish::solver::solver<MATRIX, Float> {
private:
int lib = 1; 
[[nodiscard]] int cusolver_QR(MATRIX &A, vector<double> &x,
vector<double> &b);
[[nodiscard]] int cusolver_QR(MATRIX &A, vector<float> &x, vector<float> &b);

public:

[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
void create_precond(MATRIX &A) {
throw std::runtime_error("this precond. is not impl.");
}
void apply_precond(const vector<Float> &r, vector<Float> &z) {
throw std::runtime_error("this precond. is not impl.");
}


[[nodiscard]] std::string name() const { return "monolish::equation::QR"; }


[[nodiscard]] std::string solver_name() const { return "QR"; }
};




template <typename MATRIX, typename Float>
class Cholesky : public monolish::solver::solver<MATRIX, Float> {
private:
int lib = 1; 
[[nodiscard]] int cusolver_Cholesky(MATRIX &A, vector<float> &x,
vector<float> &b);
[[nodiscard]] int cusolver_Cholesky(MATRIX &A, vector<double> &x,
vector<double> &b);

public:

[[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
[[nodiscard]] int solve(MATRIX &A, vector<Float> &xb);

void create_precond(matrix::CRS<Float> &A) {
throw std::runtime_error("this precond. is not impl.");
}
void apply_precond(const vector<Float> &r, vector<Float> &z) {
throw std::runtime_error("this precond. is not impl.");
}


[[nodiscard]] std::string name() const {
return "monolish::equation::Cholesky";
}


[[nodiscard]] std::string solver_name() const { return "Cholesky"; }
};



} 
} 
