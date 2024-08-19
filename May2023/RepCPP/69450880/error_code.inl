


#pragma once

#include <hydra/detail/external/hydra_thrust/system/error_code.h>

namespace hydra_thrust
{

namespace system
{

error_code
::error_code(void)
:m_val(0),m_cat(&system_category())
{
;
} 


error_code
::error_code(int val, const error_category &cat)
:m_val(val),m_cat(&cat)
{
;
} 


template <typename ErrorCodeEnum>
error_code
::error_code(ErrorCodeEnum e
#if HYDRA_THRUST_HOST_COMPILER != HYDRA_THRUST_HOST_COMPILER_MSVC
, typename hydra_thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value>::type *
#endif 
)
{
*this = make_error_code(e);
} 


void error_code
::assign(int val, const error_category &cat)
{
m_val = val;
m_cat = &cat;
} 


template <typename ErrorCodeEnum>
#if HYDRA_THRUST_HOST_COMPILER != HYDRA_THRUST_HOST_COMPILER_MSVC
typename hydra_thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value, error_code>::type &
#else
error_code &
#endif 
error_code
::operator=(ErrorCodeEnum e)
{
*this = make_error_code(e);
return *this;
} 


void error_code
::clear(void)
{
m_val = 0;
m_cat = &system_category();
} 


int error_code
::value(void) const
{
return m_val;
} 


const error_category &error_code
::category(void) const
{
return *m_cat;
} 


error_condition error_code
::default_error_condition(void) const
{
return category().default_error_condition(value());
} 


std::string error_code
::message(void) const
{
return category().message(value());
} 


error_code
::operator bool (void) const
{
return value() != 0;
} 


error_code make_error_code(errc::errc_t e)
{
return error_code(static_cast<int>(e), generic_category());
} 


bool operator<(const error_code &lhs, const error_code &rhs)
{
bool result = lhs.category().operator<(rhs.category());
result = result || lhs.category().operator==(rhs.category());
result = result || lhs.value() < rhs.value();
return result;
} 


template<typename charT, typename traits>
std::basic_ostream<charT,traits>&
operator<<(std::basic_ostream<charT,traits> &os, const error_code &ec)
{
return os << ec.category().name() << ':' << ec.value();
} 


bool operator==(const error_code &lhs, const error_code &rhs)
{
return lhs.category().operator==(rhs.category()) && lhs.value() == rhs.value();
} 


bool operator==(const error_code &lhs, const error_condition &rhs)
{
return lhs.category().equivalent(lhs.value(), rhs) || rhs.category().equivalent(lhs,rhs.value());
} 


bool operator==(const error_condition &lhs, const error_code &rhs)
{
return rhs.category().equivalent(lhs.value(), lhs) || lhs.category().equivalent(rhs, lhs.value());
} 


bool operator==(const error_condition &lhs, const error_condition &rhs)
{
return lhs.category().operator==(rhs.category()) && lhs.value() == rhs.value();
} 


bool operator!=(const error_code &lhs, const error_code &rhs)
{
return !(lhs == rhs);
} 


bool operator!=(const error_code &lhs, const error_condition &rhs)
{
return !(lhs == rhs);
} 


bool operator!=(const error_condition &lhs, const error_code &rhs)
{
return !(lhs == rhs);
} 


bool operator!=(const error_condition &lhs, const error_condition &rhs)
{
return !(lhs == rhs);
} 


} 

} 

