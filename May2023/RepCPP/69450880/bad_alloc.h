


#pragma once

#include <new>
#include <string>

namespace hydra_thrust
{
namespace system
{
namespace detail
{

class bad_alloc
: public std::bad_alloc
{
public:
inline bad_alloc(const std::string &w)
: std::bad_alloc(), m_what()
{
m_what = std::bad_alloc::what();
m_what += ": ";
m_what += w;
} 

inline virtual ~bad_alloc(void) throw () {};

inline virtual const char *what(void) const throw()
{
return m_what.c_str();
} 

private:
std::string m_what;
}; 

} 
} 
} 

