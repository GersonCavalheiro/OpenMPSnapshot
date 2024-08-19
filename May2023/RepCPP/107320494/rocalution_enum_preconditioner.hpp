

#pragma once
#include "rocalution/rocalution.hpp"
#include "utility.hpp"

#include <cstring>

struct rocalution_enum_preconditioner
{

#define LIST_ROCALUTION_ENUM_PRECONDITIONER \
ENUM_PRECONDITIONER(none)               \
ENUM_PRECONDITIONER(chebyshev)          \
ENUM_PRECONDITIONER(FSAI)               \
ENUM_PRECONDITIONER(SPAI)               \
ENUM_PRECONDITIONER(TNS)                \
ENUM_PRECONDITIONER(Jacobi)             \
ENUM_PRECONDITIONER(GS)                 \
ENUM_PRECONDITIONER(SGS)                \
ENUM_PRECONDITIONER(ILU)                \
ENUM_PRECONDITIONER(ILUT)               \
ENUM_PRECONDITIONER(IC)                 \
ENUM_PRECONDITIONER(MCGS)               \
ENUM_PRECONDITIONER(MCSGS)              \
ENUM_PRECONDITIONER(MCILU)

#define ENUM_PRECONDITIONER(x_) x_,

typedef enum rocalution_enum_preconditioner__ : int
{
LIST_ROCALUTION_ENUM_PRECONDITIONER
} value_type;

static constexpr value_type  all[]{LIST_ROCALUTION_ENUM_PRECONDITIONER};
static constexpr std::size_t size = countof(all);

#undef ENUM_PRECONDITIONER

#define ENUM_PRECONDITIONER(x_) #x_,

static constexpr const char* names[size]{LIST_ROCALUTION_ENUM_PRECONDITIONER};

#undef ENUM_PRECONDITIONER

bool is_invalid() const;
rocalution_enum_preconditioner();
rocalution_enum_preconditioner& operator()(const char* name_);
rocalution_enum_preconditioner(const char* name_);

value_type value{};
};

