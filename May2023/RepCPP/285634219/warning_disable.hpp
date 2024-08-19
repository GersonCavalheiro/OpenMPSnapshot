
#ifndef BOOST_CONFIG_WARNING_DISABLE_HPP
#define BOOST_CONFIG_WARNING_DISABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1400) 
#  pragma warning(disable:4996)
#endif
#if defined(__INTEL_COMPILER) || defined(__ICL)
#  pragma warning(disable:1786)
#endif

#endif 
