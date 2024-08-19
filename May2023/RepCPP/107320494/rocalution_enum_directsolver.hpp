

#pragma once
#include "utility.hpp"
#include <cstring>

#define ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM_EACH			\
ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM(inversion)			\
ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM(lu)				\
ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM(qr)

struct rocalution_enum_directsolver
{
private:
public:
#define ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM(x_) x_,
typedef enum rocalution_enum_directsolver__ : int
{
ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM_EACH
} value_type;
static constexpr value_type all[] = {ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM_EACH};
#undef ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM
static constexpr std::size_t size = countof(all);
value_type                   value{};

private:
#define ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM(x_) #x_,
static constexpr const char* names[size]{ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM_EACH};
#undef ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM
public:
operator value_type() const
{
return this->value;
};
rocalution_enum_directsolver();
rocalution_enum_directsolver& operator()(const char* function);
rocalution_enum_directsolver(const char* function);
const char*               to_string() const;
bool                      is_invalid() const;
static inline const char* to_string(rocalution_enum_directsolver::value_type value)
{
switch(value)
{
#define ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM(x_) \
case x_:                                       \
{                                              \
if(strcmp(#x_, names[value]))              \
return nullptr;                        \
break;                                     \
}

ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM_EACH;

#undef ROCALUTION_ENUM_DIRECTSOLVER_TRANSFORM
}

return names[value];
}
};
