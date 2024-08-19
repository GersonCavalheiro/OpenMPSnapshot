

#pragma once
#include "rocalution/rocalution.hpp"
#include "utility.hpp"

#include <cstring>

struct rocalution_enum_smoother
{

#define LIST_ROCALUTION_ENUM_SMOOTHER \
ENUM_SMOOTHER(FSAI)               \
ENUM_SMOOTHER(ILU)

#define ENUM_SMOOTHER(x_) x_,

typedef enum rocalution_enum_smoother__ : int
{
LIST_ROCALUTION_ENUM_SMOOTHER
} value_type;

static constexpr value_type  all[]{LIST_ROCALUTION_ENUM_SMOOTHER};
static constexpr std::size_t size = countof(all);

#undef ENUM_SMOOTHER

#define ENUM_SMOOTHER(x_) #x_,

static constexpr const char* names[size]{LIST_ROCALUTION_ENUM_SMOOTHER};

#undef ENUM_SMOOTHER

bool is_invalid() const;
rocalution_enum_smoother();
rocalution_enum_smoother& operator()(const char* name_);
rocalution_enum_smoother(const char* name_);

value_type value{};
};
