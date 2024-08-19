

#pragma once
#include "rocalution/rocalution.hpp"
#include "utility.hpp"

#include <cstring>

struct rocalution_enum_matrix_init
{

#define LIST_ROCALUTION_ENUM_MATRIX_INIT \
ENUM_MATRIX_INIT(laplacian)          \
ENUM_MATRIX_INIT(permuted_identity)  \
ENUM_MATRIX_INIT(file)

#define ENUM_MATRIX_INIT(x_) x_,

typedef enum rocalution_enum_matrix_init__ : int
{
LIST_ROCALUTION_ENUM_MATRIX_INIT
} value_type;

static constexpr value_type  all[]{LIST_ROCALUTION_ENUM_MATRIX_INIT};
static constexpr std::size_t size = countof(all);

#undef ENUM_MATRIX_INIT

#define ENUM_MATRIX_INIT(x_) #x_,

static constexpr const char* names[size]{LIST_ROCALUTION_ENUM_MATRIX_INIT};

#undef ENUM_MATRIX_INIT

bool is_invalid() const;
rocalution_enum_matrix_init();
rocalution_enum_matrix_init& operator()(const char* name_);
rocalution_enum_matrix_init(const char* name_);

value_type value{};
};
