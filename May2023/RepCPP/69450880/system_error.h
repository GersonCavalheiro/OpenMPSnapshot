




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <stdexcept>
#include <string>

#include <hydra/detail/external/hydra_thrust/system/error_code.h>

namespace hydra_thrust
{

namespace system
{






class system_error
: public std::runtime_error
{
public:


inline system_error(error_code ec, const std::string &what_arg);


inline system_error(error_code ec, const char *what_arg);


inline system_error(error_code ec);


inline system_error(int ev, const error_category &ecat, const std::string &what_arg);


inline system_error(int ev, const error_category &ecat, const char *what_arg);


inline system_error(int ev, const error_category &ecat);


inline virtual ~system_error(void) throw () {};


inline const error_code &code(void) const throw();


inline const char *what(void) const throw();


private:
error_code          m_error_code;
mutable std::string m_what;


}; 

} 



using system::system_error;

} 

#include <hydra/detail/external/hydra_thrust/system/detail/system_error.inl>

