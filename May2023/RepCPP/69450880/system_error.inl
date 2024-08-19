


#pragma once

#include <hydra/detail/external/hydra_thrust/system/system_error.h>

namespace hydra_thrust
{

namespace system
{


system_error
::system_error(error_code ec, const std::string &what_arg)
: std::runtime_error(what_arg), m_error_code(ec)
{

} 


system_error
::system_error(error_code ec, const char *what_arg)
: std::runtime_error(what_arg), m_error_code(ec)
{
;
} 


system_error
::system_error(error_code ec)
: std::runtime_error(""), m_error_code(ec)
{
;
} 


system_error
::system_error(int ev, const error_category &ecat, const std::string &what_arg)
: std::runtime_error(what_arg), m_error_code(ev,ecat)
{
;
} 


system_error
::system_error(int ev, const error_category &ecat, const char *what_arg)
: std::runtime_error(what_arg), m_error_code(ev,ecat)
{
;
} 


system_error
::system_error(int ev, const error_category &ecat)
: std::runtime_error(""), m_error_code(ev,ecat)
{
;
} 


const error_code &system_error
::code(void) const throw()
{
return m_error_code;
} 


const char *system_error
::what(void) const throw()
{
if(m_what.empty())
{
try
{
m_what = this->std::runtime_error::what();
if(m_error_code)
{
if(!m_what.empty()) m_what += ": ";
m_what += m_error_code.message();
}
}
catch(...)
{
return std::runtime_error::what();
}
}

return m_what.c_str();
} 


} 

} 

