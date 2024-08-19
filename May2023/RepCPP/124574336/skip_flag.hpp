
#if !defined(BOOST_SPIRIT_SKIP_FLAG_DEC_02_2009_0412PM)
#define BOOST_SPIRIT_SKIP_FLAG_DEC_02_2009_0412PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/detail/scoped_enum_emulation.hpp>

namespace boost { namespace spirit { namespace qi 
{
BOOST_SCOPED_ENUM_START(skip_flag) 
{ 
postskip,           
dont_postskip       
};
BOOST_SCOPED_ENUM_END

}}}

#endif

