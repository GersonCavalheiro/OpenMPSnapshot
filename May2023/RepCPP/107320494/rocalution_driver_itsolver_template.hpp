

#pragma once

#include "rocalution_bench_solver_parameters.hpp"
#include "rocalution_driver_itsolver_traits.hpp"

template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_driver_itsolver_template
{

protected:
static constexpr bool s_verbose = true;
using traits_t                  = rocalution_driver_itsolver_traits<ITSOLVER, T>;
using solver_t                  = typename traits_t::solver_t;
using preconditioner_t          = typename traits_t::preconditioner_t;
using params_t                  = rocalution_bench_solver_parameters;

protected:
virtual const preconditioner_t* GetPreconditioner() const = 0;

virtual preconditioner_t* GetPreconditioner() = 0;

virtual void SetPreconditioner(preconditioner_t* preconditioner) = 0;

virtual bool CreatePreconditioner(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
const params_t& parameters)
= 0;


protected:
virtual bool PreprocessSolverBuild(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
solver_t&       solver,
const params_t& parameters)
{
return true;
}

virtual bool PostprocessSolverBuild(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
solver_t&       solver,
const params_t& parameters)
{
return true;
}

public:
virtual bool ImportLinearSystem(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& S,
const params_t& parameters)
= 0;

public:
virtual bool PreprocessLinearSolve(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
solver_t&       solver,
const params_t& parameters)
{
return true;
}

virtual bool PostprocessLinearSolve(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
solver_t&       solver,
const params_t& parameters)
{
return true;
}

virtual bool PostprocessImportLinearSystem(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& S,
const params_t& parameters)
{
S.Allocate("s", A.GetN());
B.Allocate("b", A.GetM());

S.Ones();

A.Apply(S, &B);
return true;
}

bool ConfigureLinearSolver(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
solver_t&       solver,
const params_t& parameters)
{
const auto abs_tol  = parameters.Get(params_t::abs_tol);
const auto rel_tol  = parameters.Get(params_t::rel_tol);
const auto div_tol  = parameters.Get(params_t::div_tol);
const auto max_iter = parameters.Get(params_t::max_iter);

if(s_verbose)
{
std::cout << "ConfigureLinearSolver CreatePreconditioner ..." << std::endl;
}
bool success = this->CreatePreconditioner(A, B, X, parameters);
if(!success)
{
rocalution_bench_errmsg << "create preconditioner failed.." << std::endl;
return false;
}
if(s_verbose)
{
std::cout << "ConfigureLinearSolver CreatePreconditioner done." << std::endl;
}

solver.Verbose(0);
solver.SetOperator(A);
solver.Init(abs_tol, rel_tol, div_tol, max_iter);

auto* preconditioner = this->GetPreconditioner();
if(preconditioner != nullptr)
{
solver.SetPreconditioner(*preconditioner);
}

if(s_verbose)
{
std::cout << "ConfigureLinearSolver PreprocessSolverBuild ..." << std::endl;
}
success = this->PreprocessSolverBuild(A, B, X, solver, parameters);
if(!success)
{
rocalution_bench_errmsg << "preprocess solver build failed.." << std::endl;
return false;
}
if(s_verbose)
{
std::cout << "ConfigureLinearSolver PreprocessSolverBuild done." << std::endl;
}

solver.Build();

if(s_verbose)
{
std::cout << "ConfigureLinearSolver PostprocessSolverBuild ..." << std::endl;
}
success = this->PostprocessSolverBuild(A, B, X, solver, parameters);
if(!success)
{
rocalution_bench_errmsg << "postprocess solver build failed.." << std::endl;
return false;
}
if(s_verbose)
{
std::cout << "ConfigureLinearSolver PostprocessSolverBuild ..." << std::endl;
}

return success;
}
};
