

#pragma once

#include <rocalution/rocalution.hpp>
using namespace rocalution;

#include "rocalution_bench_itsolver.hpp"
#include "rocalution_driver_itsolver_fgmres.hpp"
#include "rocalution_driver_itsolver_gmres.hpp"
#include "rocalution_driver_itsolver_uaamg.hpp"

static constexpr bool s_verbose = true;

bool rocalution_bench_record_results(const rocalution_bench_solver_parameters&,
const rocalution_bench_solver_results&);

template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_bench_itsolver_impl : public rocalution_bench_itsolver<T>
{
using driver_traits_t = rocalution_driver_itsolver_traits<ITSOLVER, T>;
using solver_t        = typename driver_traits_t::solver_t;
using results_t       = rocalution_bench_solver_results;
using params_t        = rocalution_bench_solver_parameters;

private:
const rocalution_bench_solver_parameters* m_params{};

results_t* m_results{};

solver_t m_solver{};

rocalution_driver_itsolver<ITSOLVER, T> m_driver{};

public:
rocalution_bench_itsolver_impl(const params_t* parameters, results_t* results)
: m_params(parameters)
, m_results(results){};

virtual ~rocalution_bench_itsolver_impl()
{
this->m_solver.Clear();
};

virtual bool
ImportLinearSystem(LocalMatrix<T>& A, LocalVector<T>& B, LocalVector<T>& S) override
{
bool success = this->m_driver.ImportLinearSystem(A, B, S, *this->m_params);
if(!success)
{
rocalution_bench_errmsg << "import linear system failed." << std::endl;
return false;
}

A.MoveToAccelerator();
S.MoveToAccelerator();
B.MoveToAccelerator();

success = this->m_driver.PostprocessImportLinearSystem(A, B, S, *this->m_params);
if(!success)
{
rocalution_bench_errmsg << "post process Import linear system failed." << std::endl;
return false;
}

return true;
}

virtual bool LogBenchResults(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
LocalVector<T>& S,
results_t&      results) override
{
return rocalution_bench_record_results(*this->m_params, results);
}


virtual bool
AnalyzeLinearSystem(LocalMatrix<T>& A, LocalVector<T>& B, LocalVector<T>& X) override
{
{
if(s_verbose)
{
std::cout << "AnalyzeLinearSystem ConfigureLinearSolver ..." << std::endl;
std::cout << "A.M   = " << A.GetM() << std::endl;
std::cout << "A.N   = " << A.GetN() << std::endl;
std::cout << "A.NNZ = " << A.GetNnz() << std::endl;
}
{
bool success = this->m_driver.ConfigureLinearSolver(
A, B, X, this->m_solver, *this->m_params);
if(!success)
{
rocalution_bench_errmsg << "configure linear solver failed." << std::endl;
return false;
}
}
if(s_verbose)
{
std::cout << "AnalyzeLinearSystem ConfigureLinearSolver done." << std::endl;
}
}

{
if(s_verbose)
{
std::cout << "AnalyzeLinearSystem ConvertTo..." << std::endl;
}

{
auto format = this->m_params->Get(params_t::format);
if(s_verbose)
{
std::cout << "AnalyzeLinearSystem ConvertTo ... format = " << format << " ... "
<< std::endl;
}
const auto blockdim = this->m_params->Get(params_t::blockdim);
A.ConvertTo(format, format == BCSR ? blockdim : 1);
}

if(s_verbose)
{
std::cout << "AnalyzeLinearSystem ConvertTo done." << std::endl;
}
}

return true;
}

virtual bool SolveLinearSystem(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
results_t&      results) override
{
if(!this->m_driver.PreprocessLinearSolve(A, B, X, this->m_solver, *this->m_params))
{
rocalution_bench_errmsg << "preprocess linear solve failed." << std::endl;
return false;
}

this->m_solver.Solve(B, &X);

if(!this->m_driver.PostprocessLinearSolve(A, B, X, this->m_solver, *this->m_params))
{
rocalution_bench_errmsg << "postprocess linear solve failed." << std::endl;
return false;
}

results.Set(results_t::iter, this->m_solver.GetIterationCount());

results.Set(results_t::norm_residual, this->m_solver.GetCurrentResidual());

{
auto status = this->m_solver.GetSolverStatus();
results.Set(results_t::convergence, (status == 1) ? true : false);
}

return true;
}
};
