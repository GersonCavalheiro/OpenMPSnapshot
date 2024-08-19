

#pragma once

#include "rocalution_enum_coarsening_strategy.hpp"
#include "rocalution_enum_directsolver.hpp"
#include "rocalution_enum_itsolver.hpp"
#include "rocalution_enum_matrix_init.hpp"
#include "rocalution_enum_preconditioner.hpp"
#include "rocalution_enum_smoother.hpp"

#include <iomanip>

struct rocalution_bench_solver_parameters
{
protected:
rocalution_enum_matrix_init m_enum_matrix_init{};

rocalution_enum_itsolver m_enum_itsolver{};

rocalution_enum_preconditioner m_enum_preconditioner{};

rocalution_enum_smoother m_enum_smoother{};

rocalution_enum_directsolver m_enum_directsolver{};

rocalution_enum_coarsening_strategy m_enum_coarsening_strategy{};

public:
rocalution_enum_directsolver GetEnumDirectSolver() const;
rocalution_enum_smoother GetEnumSmoother() const;
rocalution_enum_coarsening_strategy GetEnumCoarseningStrategy() const;
rocalution_enum_preconditioner GetEnumPreconditioner() const;
rocalution_enum_itsolver GetEnumIterativeSolver() const;
rocalution_enum_matrix_init GetEnumMatrixInit() const;


#define PBOOL_TRANSFORM_EACH			\
PBOOL_TRANSFORM(verbose)			\
PBOOL_TRANSFORM(mcilu_use_level)

#define PBOOL_TRANSFORM(x_) x_,
typedef enum e_bool_ : int
{
PBOOL_TRANSFORM_EACH
} e_bool;

static constexpr e_bool e_bool_all[] = {PBOOL_TRANSFORM_EACH};
#undef PBOOL_TRANSFORM


#define PINT_TRANSFORM_EACH						\
PINT_TRANSFORM(krylov_basis)						\
PINT_TRANSFORM(ndim)							\
PINT_TRANSFORM(ilut_n)						\
PINT_TRANSFORM(mcilu_p)						\
PINT_TRANSFORM(mcilu_q)						\
PINT_TRANSFORM(max_iter)						\
PINT_TRANSFORM(solver_pre_smooth)					\
PINT_TRANSFORM(solver_post_smooth)					\
PINT_TRANSFORM(solver_ordering)					\
PINT_TRANSFORM(rebuild_numeric)					\
PINT_TRANSFORM(cycle)							\
PINT_TRANSFORM(solver_coarsest_level)					\
PINT_TRANSFORM(blockdim)


#define PINT_TRANSFORM(x_) x_,
typedef enum e_int_ : int
{
PINT_TRANSFORM_EACH
} e_int;

static constexpr e_int e_int_all[] = {PINT_TRANSFORM_EACH};
#undef PINT_TRANSFORM


#define PSTRING_TRANSFORM_EACH						\
PSTRING_TRANSFORM(coarsening_strategy)				\
PSTRING_TRANSFORM(direct_solver)					\
PSTRING_TRANSFORM(iterative_solver)					\
PSTRING_TRANSFORM(matrix)						\
PSTRING_TRANSFORM(matrix_filename)					\
PSTRING_TRANSFORM(preconditioner)					\
PSTRING_TRANSFORM(smoother)

#define PSTRING_TRANSFORM(x_) x_,
typedef enum e_string_ : int
{
PSTRING_TRANSFORM_EACH
} e_string;

static constexpr e_string e_string_all[] = {PSTRING_TRANSFORM_EACH};
#undef PSTRING_TRANSFORM


#define PUINT_TRANSFORM_EACH					\
PUINT_TRANSFORM(format)

#define PUINT_TRANSFORM(x_) x_,
typedef enum e_uint_ : int
{
PUINT_TRANSFORM_EACH
} e_uint;

static constexpr e_uint e_uint_all[] = {PUINT_TRANSFORM_EACH};
#undef PUINT_TRANSFORM


#define PDOUBLE_TRANSFORM_EACH				\
PDOUBLE_TRANSFORM(abs_tol)				\
PDOUBLE_TRANSFORM(rel_tol)				\
PDOUBLE_TRANSFORM(div_tol)				\
PDOUBLE_TRANSFORM(residual_tol)			\
PDOUBLE_TRANSFORM(ilut_tol)				\
PDOUBLE_TRANSFORM(mcgs_relax)				\
PDOUBLE_TRANSFORM(solver_over_interp)			\
PDOUBLE_TRANSFORM(solver_coupling_strength) \

#define PDOUBLE_TRANSFORM(x_) x_,
typedef enum e_double_ : int
{
PDOUBLE_TRANSFORM_EACH
} e_double;
static constexpr e_double e_double_all[] = {PDOUBLE_TRANSFORM_EACH};
#undef PDOUBLE_TRANSFORM

private:
static constexpr std::size_t e_string_size = countof(e_string_all);
static constexpr std::size_t e_uint_size = countof(e_uint_all);
static constexpr std::size_t e_bool_size = countof(e_bool_all);
static constexpr std::size_t e_int_size = countof(e_int_all);
static constexpr std::size_t e_double_size = countof(e_double_all);

#define PBOOL_TRANSFORM(x_) #x_,
static constexpr const char* e_bool_names[e_bool_size]{PBOOL_TRANSFORM_EACH};
#undef PBOOL_TRANSFORM

#define PUINT_TRANSFORM(x_) #x_,
static constexpr const char* e_uint_names[e_uint_size]{PUINT_TRANSFORM_EACH};
#undef PUINT_TRANSFORM

#define PSTRING_TRANSFORM(x_) #x_,
static constexpr const char* e_string_names[e_string_size]{PSTRING_TRANSFORM_EACH};
#undef PSTRING_TRANSFORM

#define PINT_TRANSFORM(x_) #x_,
static constexpr const char* e_int_names[e_int_size]{PINT_TRANSFORM_EACH};
#undef PINT_TRANSFORM

#define PDOUBLE_TRANSFORM(x_) #x_,
static constexpr const char* e_double_names[e_double_size]{PDOUBLE_TRANSFORM_EACH};
#undef PDOUBLE_TRANSFORM

bool bool_values[e_bool_size]{};

unsigned int uint_values[e_uint_size]{};

int int_values[e_int_size]{};

std::string string_values[e_string_size]{};

double double_values[e_double_size]{};

public:
static const char* GetName(e_bool v)
{
return e_bool_names[v];
}
static const char* GetName(e_int v)
{
return e_int_names[v];
}
static const char* GetName(e_uint v)
{
return e_uint_names[v];
}
static const char* GetName(e_double v)
{
return e_double_names[v];
}
static const char* GetName(e_string v)
{
return e_string_names[v];
}

std::string* GetPointer(e_string v);

std::string Get(e_string v) const;

void Set(e_string v, const std::string& s);

unsigned int Get(e_uint v) const;
unsigned int* GetPointer(e_uint v);

void Set(e_uint v, unsigned int s);

bool Get(e_bool v) const;
bool* GetPointer(e_bool v);

void Set(e_bool v, bool s);

int Get(e_int v) const;

int* GetPointer(e_int v);

void Set(e_int v, int s);

double Get(e_double v) const;

double* GetPointer(e_double v);

void Set(e_double v, double s);

void Info(std::ostream& out) const
{
out.setf(std::ios::left);
out << "bool:  " << std::endl;
for(auto e : e_bool_all)
{
out << std::setw(20) << e_bool_names[e] << std::setw(20) << bool_values[e] << std::endl;
}
out << "int:  " << std::endl;
for(auto e : e_int_all)
{
out << std::setw(20) << e_int_names[e] << std::setw(20) << int_values[e] << std::endl;
}
out << "uint:  " << std::endl;
for(auto e : e_uint_all)
{
out << std::setw(20) << e_uint_names[e] << std::setw(20) << uint_values[e] << std::endl;
}
out << "double:  " << std::endl;
for(auto e : e_double_all)
{
out << std::setw(20) << e_double_names[e] << std::setw(20) << double_values[e]
<< std::endl;
}
out << "string:  " << std::endl;
for(auto e : e_string_all)
{
out << std::setw(20) << e_string_names[e] << "'" << string_values[e] << "'"
<< std::endl;
}
}

void WriteJson(std::ostream& out) const;
void WriteNicely(std::ostream& out) const;
};
