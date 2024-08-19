#ifndef BOOST_CURRENT_FUNCTION_HPP_INCLUDED
#define BOOST_CURRENT_FUNCTION_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


namespace boost
{

namespace detail
{

inline void current_function_helper()
{

#if defined( BOOST_DISABLE_CURRENT_FUNCTION )

# define BOOST_CURRENT_FUNCTION "(unknown)"

#elif defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__) || defined(__clang__)

# define BOOST_CURRENT_FUNCTION __PRETTY_FUNCTION__

#elif defined(__DMC__) && (__DMC__ >= 0x810)

# define BOOST_CURRENT_FUNCTION __PRETTY_FUNCTION__

#elif defined(__FUNCSIG__)

# define BOOST_CURRENT_FUNCTION __FUNCSIG__

#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))

# define BOOST_CURRENT_FUNCTION __FUNCTION__

#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)

# define BOOST_CURRENT_FUNCTION __FUNC__

#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)

# define BOOST_CURRENT_FUNCTION __func__

#elif defined(__cplusplus) && (__cplusplus >= 201103)

# define BOOST_CURRENT_FUNCTION __func__

#else

# define BOOST_CURRENT_FUNCTION "(unknown)"

#endif

}

} 

} 

#endif 
