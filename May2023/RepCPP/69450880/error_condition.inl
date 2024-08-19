


#pragma once

#include <hydra/detail/external/hydra_thrust/system/detail/error_condition.inl>
#include <hydra/detail/external/hydra_thrust/functional.h>

namespace hydra_thrust
{

namespace system
{

error_condition
::error_condition(void)
:m_val(0),m_cat(&generic_category())
{
;
} 


error_condition
::error_condition(int val, const error_category &cat)
:m_val(val),m_cat(&cat)
{
;
} 


template<typename ErrorConditionEnum>
error_condition
::error_condition(ErrorConditionEnum e
#if HYDRA_THRUST_HOST_COMPILER != HYDRA_THRUST_HOST_COMPILER_MSVC
, typename hydra_thrust::detail::enable_if<is_error_condition_enum<ErrorConditionEnum>::value>::type *
#endif 
)
{
*this = make_error_condition(e);
} 


void error_condition
::assign(int val, const error_category &cat)
{
m_val = val;
m_cat = &cat;
} 


template<typename ErrorConditionEnum>
#if HYDRA_THRUST_HOST_COMPILER != HYDRA_THRUST_HOST_COMPILER_MSVC
typename hydra_thrust::detail::enable_if<is_error_condition_enum<ErrorConditionEnum>::value, error_condition>::type &
#else
error_condition &
#endif 
error_condition
::operator=(ErrorConditionEnum e)
{
*this = make_error_condition(e);
return *this;
} 


void error_condition
::clear(void)
{
m_val = 0;
m_cat = &generic_category();
} 


int error_condition
::value(void) const
{
return m_val;
} 


const error_category &error_condition
::category(void) const
{
return *m_cat;
} 


std::string error_condition
::message(void) const
{
return category().message(value());
} 


error_condition
::operator bool (void) const
{
return value() != 0;
} 


error_condition make_error_condition(errc::errc_t e)
{
return error_condition(static_cast<int>(e), generic_category());
} 


bool operator<(const error_condition &lhs,
const error_condition &rhs)
{
return lhs.category().operator<(rhs.category()) || (lhs.category().operator==(rhs.category()) && (lhs.value() < rhs.value()));
} 


} 

} 

