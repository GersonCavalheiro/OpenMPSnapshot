
#ifndef BOOST_XPRESSIVE_REGEX_CONSTANTS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_REGEX_CONSTANTS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/identity.hpp>

#ifndef BOOST_XPRESSIVE_DOXYGEN_INVOKED
# define icase icase_
#endif

namespace boost { namespace xpressive { namespace regex_constants
{

enum syntax_option_type
{

ECMAScript  = 0,        
icase       = 1 << 1,   
nosubs      = 1 << 2,   
optimize    = 1 << 3,   
collate     = 1 << 4,   




single_line         = 1 << 10,  
not_dot_null        = 1 << 11,  
not_dot_newline     = 1 << 12,  
ignore_white_space  = 1 << 13   
};

enum match_flag_type
{
match_default           = 0,        
match_not_bol           = 1 << 1,   
match_not_eol           = 1 << 2,   
match_not_bow           = 1 << 3,   
match_not_eow           = 1 << 4,   
match_any               = 1 << 7,   
match_not_null          = 1 << 8,   
match_continuous        = 1 << 10,  
match_partial           = 1 << 11,  
match_prev_avail        = 1 << 12,  
format_default          = 0,        
format_sed              = 1 << 13,  
format_perl             = 1 << 14,  
format_no_copy          = 1 << 15,  
format_first_only       = 1 << 16,  
format_literal          = 1 << 17,  
format_all              = 1 << 18   
};

enum error_type
{
error_collate,              
error_ctype,                
error_escape,               
error_subreg,               
error_brack,                
error_paren,                
error_brace,                
error_badbrace,             
error_range,                
error_space,                
error_badrepeat,            
error_complexity,           
error_stack,                
error_badref,               
error_badmark,              
error_badlookbehind,        
error_badrule,              
error_badarg,               
error_badattr,              
error_internal              
};

inline syntax_option_type operator &(syntax_option_type b1, syntax_option_type b2)
{
return static_cast<syntax_option_type>(
static_cast<int>(b1) & static_cast<int>(b2));
}

inline syntax_option_type operator |(syntax_option_type b1, syntax_option_type b2)
{
return static_cast<syntax_option_type>(static_cast<int>(b1) | static_cast<int>(b2));
}

inline syntax_option_type operator ^(syntax_option_type b1, syntax_option_type b2)
{
return static_cast<syntax_option_type>(static_cast<int>(b1) ^ static_cast<int>(b2));
}

inline syntax_option_type operator ~(syntax_option_type b)
{
return static_cast<syntax_option_type>(~static_cast<int>(b));
}

inline match_flag_type operator &(match_flag_type b1, match_flag_type b2)
{
return static_cast<match_flag_type>(static_cast<int>(b1) & static_cast<int>(b2));
}

inline match_flag_type operator |(match_flag_type b1, match_flag_type b2)
{
return static_cast<match_flag_type>(static_cast<int>(b1) | static_cast<int>(b2));
}

inline match_flag_type operator ^(match_flag_type b1, match_flag_type b2)
{
return static_cast<match_flag_type>(static_cast<int>(b1) ^ static_cast<int>(b2));
}

inline match_flag_type operator ~(match_flag_type b)
{
return static_cast<match_flag_type>(~static_cast<int>(b));
}

}}} 

#ifndef BOOST_XPRESSIVE_DOXYGEN_INVOKED
# undef icase
#endif

#endif
