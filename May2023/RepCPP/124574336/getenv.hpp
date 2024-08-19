
#ifndef BOOST_COMPUTE_DETAIL_GETENV_HPP
#define BOOST_COMPUTE_DETAIL_GETENV_HPP

#include <cstdlib>

namespace boost {
namespace compute {
namespace detail {

inline const char* getenv(const char *env_var)
{
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4996)
#endif
return std::getenv(env_var);
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
}

} 
} 
} 

#endif 
