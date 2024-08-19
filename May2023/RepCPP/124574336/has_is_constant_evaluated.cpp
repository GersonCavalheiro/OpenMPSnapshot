




#include <boost/config.hpp>
#include <boost/config/pragma_message.hpp>

#include <boost/multiprecision/number.hpp>

#ifdef __has_builtin
BOOST_PRAGMA_MESSAGE (" __has_builtin is defined.")

#  if __has_builtin(__builtin_is_constant_evaluated)
BOOST_PRAGMA_MESSAGE (" __has_builtin(__builtin_is_constant_evaluated), so BOOST_MP_NO_CONSTEXPR_DETECTION should NOT be defined.")
#  endif 

#endif 

#ifdef BOOST_MP_HAS_IS_CONSTANT_EVALUATED
BOOST_PRAGMA_MESSAGE ("BOOST_MP_HAS_IS_CONSTANT_EVALUATED defined.")
#else
BOOST_PRAGMA_MESSAGE ("BOOST_MP_HAS_IS_CONSTANT_EVALUATED is NOT defined, so no std::is_constant_evaluated() from std library.")
#endif

#ifdef BOOST_NO_CXX14_CONSTEXPR
BOOST_PRAGMA_MESSAGE ("BOOST_NO_CXX14_CONSTEXPR is defined.")
#endif

#ifdef BOOST_MP_NO_CONSTEXPR_DETECTION
#  error 1  "std::is_constant_evaluated is NOT available to determine if a calculation can use constexpr."
#endif
