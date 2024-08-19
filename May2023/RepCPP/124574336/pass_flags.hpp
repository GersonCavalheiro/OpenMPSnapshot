
#if !defined(BOOST_SPIRIT_LEX_PASS_FLAGS_JUN_09_2009_0840PM)
#define BOOST_SPIRIT_LEX_PASS_FLAGS_JUN_09_2009_0840PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/detail/scoped_enum_emulation.hpp>

namespace boost { namespace spirit { namespace lex
{
BOOST_SCOPED_ENUM_START(pass_flags) 
{ 
pass_fail = 0,        
pass_normal = 1,      
pass_ignore = 2       
};
BOOST_SCOPED_ENUM_END

}}}

#endif
