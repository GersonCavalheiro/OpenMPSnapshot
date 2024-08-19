
#if !defined(BOOST_SPIRIT_KARMA_DELIMIT_FLAG_DEC_02_2009_1201PM)
#define BOOST_SPIRIT_KARMA_DELIMIT_FLAG_DEC_02_2009_1201PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/detail/scoped_enum_emulation.hpp>

namespace boost { namespace spirit { namespace karma
{
BOOST_SCOPED_ENUM_START(delimit_flag) 
{ 
predelimit,         
dont_predelimit     
};
BOOST_SCOPED_ENUM_END
}}}

#endif

