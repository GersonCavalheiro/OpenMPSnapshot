

#pragma once

#include "rocalution_bench_solver_results.hpp"
#include <chrono>

template <typename T>
struct rocalution_linear_system
{
LocalMatrix<T> A{};
LocalVector<T> B{};
LocalVector<T> X{};
LocalVector<T> S{};
};

template <typename T>
struct rocalution_bench_itsolver
{
public:
using results_t = rocalution_bench_solver_results;

private:
rocalution_linear_system<T> m_linsys;

protected:
static constexpr bool s_verbose = false;

virtual bool ImportLinearSystem(LocalMatrix<T>& A, LocalVector<T>& B, LocalVector<T>& S) = 0;

virtual bool AnalyzeLinearSystem(LocalMatrix<T>& A, LocalVector<T>& B, LocalVector<T>& X) = 0;

virtual bool SolveLinearSystem(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
results_t&      results)
= 0;

virtual bool LogBenchResults(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
LocalVector<T>& S,
results_t&      results)
= 0;

bool ComputeErrorApproximation(const LocalVector<T>& X,
const LocalVector<T>& S,
LocalVector<T>&       E,
results_t&            results)
{

const auto N = X.GetSize();
E.CopyFrom(X, 0, 0, N);
E.ScaleAdd(-1.0, S);

const T nrm2 = E.Norm();
results.Set(results_t::norm_residual, nrm2);
{
E.MoveToHost();
T* v = new T[N];
for(auto i = 0; i < N; ++i)
{
v[i] = T(std::abs(E[i]));
}
std::sort(v, v + N);
results.Set(results_t::nrmmax_err, v[N - 1]);
results.Set(results_t::nrmmax95_err, v[((N - 1) * 95) / 100]);
results.Set(results_t::nrmmax75_err, v[((N - 1) * 75) / 100]);
results.Set(results_t::nrmmax50_err, v[((N - 1) * 50) / 100]);
delete[] v;
E.MoveToAccelerator();
}

return true;
};

public:
virtual ~rocalution_bench_itsolver(){};

bool Run(results_t& results)
{

#define TIC(var_) double var_ = rocalution_time()
#define TAC(var_) var_ = ((rocalution_time() - var_) / 1e3)

TIC(t_run);

if(s_verbose)
{
std::cout << "Import linear system ..." << std::endl;
}
TIC(t_import_linear_system);
{
bool success
= this->ImportLinearSystem(this->m_linsys.A, this->m_linsys.B, this->m_linsys.S);
if(!success)
{
rocalution_bench_errmsg << "ImportLinearSystem failed." << std::endl;
return false;
}
}
TAC(t_import_linear_system);
if(s_verbose)
{
std::cout << "Import linear system done." << std::endl;
}

this->m_linsys.X.MoveToAccelerator();
this->m_linsys.X.Allocate("x", this->m_linsys.A.GetN());
this->m_linsys.X.Zeros();

if(s_verbose)
{
std::cout << "Analyze linear system ..." << std::endl;
}
TIC(t_analyze_linear_system);
{
bool success
= this->AnalyzeLinearSystem(this->m_linsys.A, this->m_linsys.B, this->m_linsys.X);
if(!success)
{
rocalution_bench_errmsg << "AnalyzeLinearSystem  failed." << std::endl;
return false;
}
}
TAC(t_analyze_linear_system);

if(s_verbose)
{
std::cout << "Analyze linear system done." << std::endl;
}
if(s_verbose)
{
std::cout << "Solve linear system ..." << std::endl;
}
TIC(t_solve_linear_system);
{
bool success = this->SolveLinearSystem(
this->m_linsys.A, this->m_linsys.B, this->m_linsys.X, results);
if(!success)
{
rocalution_bench_errmsg << "SolveLinearSystem  failed." << std::endl;
return false;
}
}
TAC(t_solve_linear_system);
if(s_verbose)
{
std::cout << "Solve linear system done." << std::endl;
}

if(s_verbose)
{
std::cout << "Compute error approximation ..." << std::endl;
}

{
LocalVector<T> E;
E.MoveToAccelerator();
E.Allocate("e", this->m_linsys.X.GetSize());

bool success
= this->ComputeErrorApproximation(this->m_linsys.X, this->m_linsys.S, E, results);
if(!success)
{
rocalution_bench_errmsg << "ComputeErrorApproximation failed." << std::endl;
return false;
}
}

if(s_verbose)
{
std::cout << "Compute error approximation done." << std::endl;
}
TAC(t_run);

results.Set(results_t::time_import, t_import_linear_system);
results.Set(results_t::time_analyze, t_analyze_linear_system);
results.Set(results_t::time_solve, t_solve_linear_system);
results.Set(results_t::time_global, t_run);

{
bool success = this->LogBenchResults(
this->m_linsys.A, this->m_linsys.B, this->m_linsys.X, this->m_linsys.S, results);
if(!success)
{
rocalution_bench_errmsg << "LogBenchResults failed." << std::endl;
return false;
}
}

return true;
}
};
