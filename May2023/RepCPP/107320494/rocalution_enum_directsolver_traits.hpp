

#pragma once

#include "rocalution_bench_solver_template_base.hpp"
#include "rocalution_enum_directsolver.hpp"

using namespace rocalution;

template <rocalution_enum_directsolver::value_type DIRECTSOLVER, typename T>
struct rocalution_enum_directsolver_traits;

template <typename T>
struct rocalution_enum_directsolver_traits<rocalution_enum_directsolver::lu, T>
{
using solver_t = LU<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_enum_directsolver_traits<rocalution_enum_directsolver::qr, T>
{
using solver_t = QR<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_enum_directsolver_traits<rocalution_enum_directsolver::inversion, T>
{
using solver_t = Inversion<LocalMatrix<T>, LocalVector<T>, T>;
};
