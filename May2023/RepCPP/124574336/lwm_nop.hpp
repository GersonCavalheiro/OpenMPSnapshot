
#ifndef BOOST_SIGNALS2_LWM_NOP_HPP
#define BOOST_SIGNALS2_LWM_NOP_HPP


#if defined(_MSC_VER)
# pragma once
#endif


#include <boost/signals2/dummy_mutex.hpp>

namespace boost
{

namespace signals2
{

class mutex: public dummy_mutex
{
};

} 

} 

#endif 
