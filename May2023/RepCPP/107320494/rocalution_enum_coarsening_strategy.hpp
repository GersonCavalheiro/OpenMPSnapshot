

#pragma once
#include "rocalution/rocalution.hpp"
#include "utility.hpp"

#include <cstring>

struct rocalution_enum_coarsening_strategy
{

#define LIST_ROCALUTION_ENUM_COARSENING_STRATEGY \
ENUM_COARSENING_STRATEGY(Greedy)             \
ENUM_COARSENING_STRATEGY(PMIS)

#define ENUM_COARSENING_STRATEGY(x_) x_,

typedef enum rocalution_enum_coarsening_strategy__ : int
{
LIST_ROCALUTION_ENUM_COARSENING_STRATEGY
} value_type;

static constexpr value_type  all[]{LIST_ROCALUTION_ENUM_COARSENING_STRATEGY};
static constexpr std::size_t size = countof(all);

#undef ENUM_COARSENING_STRATEGY

#define ENUM_COARSENING_STRATEGY(x_) #x_,

static constexpr const char* names[size]{LIST_ROCALUTION_ENUM_COARSENING_STRATEGY};

#undef ENUM_COARSENING_STRATEGY

bool is_invalid() const;
rocalution_enum_coarsening_strategy();
rocalution_enum_coarsening_strategy& operator()(const char* name_);
rocalution_enum_coarsening_strategy(const char* name_);

value_type value{};
};
