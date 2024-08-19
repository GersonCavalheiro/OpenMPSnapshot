

#pragma once
#include "utility.hpp"
#include <cstring>


#define ROCALUTION_ENUM_ITSOLVER_TRANSFORM_EACH				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(gmres)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(bicgstab)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(fgmres)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(cg)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(cr)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(fcg)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(idr)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(pairwise_amg)			\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(qmrcgstab)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(ruge_stueben_amg)			\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(saamg)				\
ROCALUTION_ENUM_ITSOLVER_TRANSFORM(uaamg)

struct rocalution_enum_itsolver
{
private:
public:
#define ROCALUTION_ENUM_ITSOLVER_TRANSFORM(x_) x_,
typedef enum rocalution_enum_itsolver__ : int
{
ROCALUTION_ENUM_ITSOLVER_TRANSFORM_EACH
} value_type;
static constexpr value_type all[] = {ROCALUTION_ENUM_ITSOLVER_TRANSFORM_EACH};
#undef ROCALUTION_ENUM_ITSOLVER_TRANSFORM
static constexpr std::size_t size = countof(all);
value_type                   value{};

private:
#define ROCALUTION_ENUM_ITSOLVER_TRANSFORM(x_) #x_,
static constexpr const char* names[size]{ROCALUTION_ENUM_ITSOLVER_TRANSFORM_EACH};
#undef ROCALUTION_ENUM_ITSOLVER_TRANSFORM
public:
operator value_type() const
{
return this->value;
};
rocalution_enum_itsolver();
rocalution_enum_itsolver& operator()(const char* function);
rocalution_enum_itsolver(const char* function);
const char*               to_string() const;
bool                      is_invalid() const;
static inline const char* to_string(rocalution_enum_itsolver::value_type value)
{
switch(value)
{
#define ROCALUTION_ENUM_ITSOLVER_TRANSFORM(x_) \
case x_:                                   \
{                                          \
if(strcmp(#x_, names[value]))          \
return nullptr;                    \
break;                                 \
}

ROCALUTION_ENUM_ITSOLVER_TRANSFORM_EACH;

#undef ROCALUTION_ENUM_ITSOLVER_TRANSFORM
}

return names[value];
}
};
