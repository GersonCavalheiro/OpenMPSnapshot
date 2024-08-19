



#ifndef BOOST_REGEX_V4_REGBASE_HPP
#define BOOST_REGEX_V4_REGBASE_HPP

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

namespace boost{
class BOOST_REGEX_DECL regbase
{
public:
enum flag_type_
{
perl_syntax_group = 0,                      
basic_syntax_group = 1,                     
literal = 2,                                
main_option_type = literal | basic_syntax_group | perl_syntax_group, 
no_bk_refs = 1 << 8,                        
no_perl_ex = 1 << 9,                        
no_mod_m = 1 << 10,                         
mod_x = 1 << 11,                            
mod_s = 1 << 12,                            
no_mod_s = 1 << 13,                         

no_char_classes = 1 << 8,                   
no_intervals = 1 << 9,                      
bk_plus_qm = 1 << 10,                       
bk_vbar = 1 << 11,                          
emacs_ex = 1 << 12,                         

no_escape_in_lists = 1 << 16,                     
newline_alt = 1 << 17,                            
no_except = 1 << 18,                              
failbit = 1 << 19,                                
icase = 1 << 20,                                  
nocollate = 0,                                    
collate = 1 << 21,                                
nosubs = 1 << 22,                                 
save_subexpression_location = 1 << 23,            
no_empty_expressions = 1 << 24,                   
optimize = 0,                                     



basic = basic_syntax_group | collate | no_escape_in_lists,
extended = no_bk_refs | collate | no_perl_ex | no_escape_in_lists,
normal = 0,
emacs = basic_syntax_group | collate | emacs_ex | bk_vbar,
awk = no_bk_refs | collate | no_perl_ex,
grep = basic | newline_alt,
egrep = extended | newline_alt,
sed = basic,
perl = normal,
ECMAScript = normal,
JavaScript = normal,
JScript = normal
};
typedef unsigned int flag_type;

enum restart_info
{
restart_any = 0,
restart_word = 1,
restart_line = 2,
restart_buf = 3,
restart_continue = 4,
restart_lit = 5,
restart_fixed_lit = 6, 
restart_count = 7
};
};

namespace regex_constants{

enum flag_type_
{

no_except = ::boost::regbase::no_except,
failbit = ::boost::regbase::failbit,
literal = ::boost::regbase::literal,
icase = ::boost::regbase::icase,
nocollate = ::boost::regbase::nocollate,
collate = ::boost::regbase::collate,
nosubs = ::boost::regbase::nosubs,
optimize = ::boost::regbase::optimize,
bk_plus_qm = ::boost::regbase::bk_plus_qm,
bk_vbar = ::boost::regbase::bk_vbar,
no_intervals = ::boost::regbase::no_intervals,
no_char_classes = ::boost::regbase::no_char_classes,
no_escape_in_lists = ::boost::regbase::no_escape_in_lists,
no_mod_m = ::boost::regbase::no_mod_m,
mod_x = ::boost::regbase::mod_x,
mod_s = ::boost::regbase::mod_s,
no_mod_s = ::boost::regbase::no_mod_s,
save_subexpression_location = ::boost::regbase::save_subexpression_location,
no_empty_expressions = ::boost::regbase::no_empty_expressions,

basic = ::boost::regbase::basic,
extended = ::boost::regbase::extended,
normal = ::boost::regbase::normal,
emacs = ::boost::regbase::emacs,
awk = ::boost::regbase::awk,
grep = ::boost::regbase::grep,
egrep = ::boost::regbase::egrep,
sed = basic,
perl = normal,
ECMAScript = normal,
JavaScript = normal,
JScript = normal
};
typedef ::boost::regbase::flag_type syntax_option_type;

} 

} 

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif

