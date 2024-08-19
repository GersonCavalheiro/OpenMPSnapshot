

#if !defined(BOOST_LIST_INCLUDES_CONFIG_HPP_0DE80E47_8D50_4DFA_9C1C_0EECAA8A934A_INCLUDED)
#define BOOST_LIST_INCLUDES_CONFIG_HPP_0DE80E47_8D50_4DFA_9C1C_0EECAA8A934A_INCLUDED


#if defined(BOOST_SPIRIT_DEBUG)

#define BOOST_SPIRIT_DEBUG_FLAGS ( \
BOOST_SPIRIT_DEBUG_FLAGS_NODES | \
BOOST_SPIRIT_DEBUG_FLAGS_CLOSURES \
) \



#define BOOST_SPIRIT_DEBUG_FLAGS_CPP (\
\
) \

#endif 

#include <boost/wave/wave_config.hpp>

#if defined(BOOST_MSVC)
#pragma warning (disable: 4355) 
#pragma warning (disable: 4800) 
#pragma inline_depth(255)
#pragma inline_recursion(on)
#endif 

#endif 
