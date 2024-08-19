

#pragma once

#include "rocalution_bench_itsolver.hpp"
#include "rocalution_bench_solver_parameters.hpp"

using namespace rocalution;

template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_driver_itsolver_traits;

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::ruge_stueben_amg, T>
{
using solver_t         = BiCGStab<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::saamg, T>
{
using solver_t         = FCG<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::uaamg, T>
{
using solver_t         = FCG<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = UAAMG<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::qmrcgstab, T>
{
using solver_t         = QMRCGStab<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::pairwise_amg, T>
{
using solver_t         = CG<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::idr, T>
{
using solver_t         = IDR<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::fcg, T>
{
using solver_t         = FCG<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::cr, T>
{
using solver_t         = CR<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::cg, T>
{
using solver_t         = CG<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::gmres, T>
{
using solver_t         = GMRES<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::fgmres, T>
{
using solver_t         = FGMRES<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};

template <typename T>
struct rocalution_driver_itsolver_traits<rocalution_enum_itsolver::bicgstab, T>
{
using solver_t         = BiCGStab<LocalMatrix<T>, LocalVector<T>, T>;
using preconditioner_t = Preconditioner<LocalMatrix<T>, LocalVector<T>, T>;
};
