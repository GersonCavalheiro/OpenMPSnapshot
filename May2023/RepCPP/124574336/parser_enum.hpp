
#ifndef BOOST_XPRESSIVE_DETAIL_DYNAMIC_PARSER_ENUM_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_DYNAMIC_PARSER_ENUM_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

namespace boost { namespace xpressive { namespace regex_constants
{

enum compiler_token_type
{
token_literal,
token_any,                          
token_escape,                       
token_group_begin,                  
token_group_end,                    
token_alternate,                    
token_invalid_quantifier,           
token_charset_begin,                
token_charset_end,                  
token_charset_invert,               
token_charset_hyphen,               
token_charset_backspace,            
token_posix_charset_begin,          
token_posix_charset_end,            
token_equivalence_class_begin,      
token_equivalence_class_end,        
token_collation_element_begin,      
token_collation_element_end,        

token_quote_meta_begin,             
token_quote_meta_end,               

token_no_mark,                      
token_positive_lookahead,           
token_negative_lookahead,           
token_positive_lookbehind,          
token_negative_lookbehind,          
token_independent_sub_expression,   
token_comment,                      
token_recurse,                      
token_rule_assign,                  
token_rule_ref,                     
token_named_mark,                   
token_named_mark_ref,               

token_assert_begin_sequence,        
token_assert_end_sequence,          
token_assert_begin_line,            
token_assert_end_line,              
token_assert_word_begin,            
token_assert_word_end,              
token_assert_word_boundary,         
token_assert_not_word_boundary,     

token_escape_newline,               
token_escape_escape,                
token_escape_formfeed,              
token_escape_horizontal_tab,        
token_escape_vertical_tab,          
token_escape_bell,                  
token_escape_control,               

token_end_of_pattern
};

}}} 

#endif
