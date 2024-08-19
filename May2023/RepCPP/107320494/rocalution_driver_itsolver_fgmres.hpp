

#pragma once
#include "rocalution_driver_itsolver.hpp"

template <typename T>
struct rocalution_driver_itsolver<rocalution_enum_itsolver::fgmres, T>
: rocalution_driver_itsolver_default<rocalution_enum_itsolver::fgmres, T>
{

static constexpr auto ITSOLVER = rocalution_enum_itsolver::fgmres;
using traits_t                 = rocalution_driver_itsolver_traits<ITSOLVER, T>;
using solver_t                 = typename traits_t::solver_t;
using params_t                 = rocalution_bench_solver_parameters;

virtual bool PreprocessSolverBuild(LocalMatrix<T>& A,
LocalVector<T>& B,
LocalVector<T>& X,
solver_t&       solver,
const params_t& parameters) override
{
const auto krylov_basis = parameters.Get(params_t::krylov_basis);
solver.SetBasisSize(krylov_basis);
return true;
};
};
