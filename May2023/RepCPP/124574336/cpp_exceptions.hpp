

#if !defined(BOOST_CPP_EXCEPTIONS_HPP_5190E447_A781_4521_A275_5134FF9917D7_INCLUDED)
#define BOOST_CPP_EXCEPTIONS_HPP_5190E447_A781_4521_A275_5134FF9917D7_INCLUDED

#include <exception>
#include <string>
#include <limits>

#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include <boost/wave/wave_config.hpp>
#include <boost/wave/cpp_throw.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace wave {

namespace util {

enum severity {
severity_remark = 0,
severity_warning,
severity_error,
severity_fatal,
severity_commandline_error,
last_severity_code = severity_commandline_error
};

inline char const *
get_severity(int level)
{
static char const *severity_text[] =
{
"remark",               
"warning",              
"error",                
"fatal error",          
"command line error"    
};
BOOST_ASSERT(severity_remark <= level &&
level <= last_severity_code);
return severity_text[level];
}
}

class BOOST_SYMBOL_VISIBLE cpp_exception
:   public std::exception
{
public:
cpp_exception(std::size_t line_, std::size_t column_, char const *filename_) throw()
:   line(line_), column(column_)
{
unsigned int off = 0;
while (off < sizeof(filename)-1 && *filename_)
filename[off++] = *filename_++;
filename[off] = 0;
}
~cpp_exception() throw() {}

char const *what() const throw() BOOST_OVERRIDE = 0;    
virtual char const *description() const throw() = 0;
virtual int get_errorcode() const throw() = 0;
virtual int get_severity() const throw() = 0;
virtual char const* get_related_name() const throw() = 0;
virtual bool is_recoverable() const throw() = 0;

std::size_t line_no() const throw() { return line; }
std::size_t column_no() const throw() { return column; }
char const *file_name() const throw() { return filename; }

protected:
char filename[512];
std::size_t line;
std::size_t column;
};

class BOOST_SYMBOL_VISIBLE preprocess_exception :
public cpp_exception
{
public:
enum error_code {
no_error = 0,
unexpected_error,
macro_redefinition,
macro_insertion_error,
bad_include_file,
bad_include_statement,
bad_has_include_expression,
ill_formed_directive,
error_directive,
warning_directive,
ill_formed_expression,
missing_matching_if,
missing_matching_endif,
ill_formed_operator,
bad_define_statement,
bad_define_statement_va_args,
bad_define_statement_va_opt,
bad_define_statement_va_opt_parens,
bad_define_statement_va_opt_recurse,
too_few_macroarguments,
too_many_macroarguments,
empty_macroarguments,
improperly_terminated_macro,
bad_line_statement,
bad_line_number,
bad_line_filename,
bad_undefine_statement,
bad_macro_definition,
illegal_redefinition,
duplicate_parameter_name,
invalid_concat,
last_line_not_terminated,
ill_formed_pragma_option,
include_nesting_too_deep,
misplaced_operator,
alreadydefined_name,
undefined_macroname,
invalid_macroname,
unexpected_qualified_name,
division_by_zero,
integer_overflow,
illegal_operator_redefinition,
ill_formed_integer_literal,
ill_formed_character_literal,
unbalanced_if_endif,
character_literal_out_of_range,
could_not_open_output_file,
incompatible_config,
ill_formed_pragma_message,
pragma_message_directive,
last_error_number = pragma_message_directive
};

preprocess_exception(char const *what_, error_code code, std::size_t line_,
std::size_t column_, char const *filename_) throw()
:   cpp_exception(line_, column_, filename_),
code(code)
{
unsigned int off = 0;
while (off < sizeof(buffer) - 1 && *what_)
buffer[off++] = *what_++;
buffer[off] = 0;
}
~preprocess_exception() throw() {}

char const *what() const throw() BOOST_OVERRIDE
{
return "boost::wave::preprocess_exception";
}
char const *description() const throw() BOOST_OVERRIDE
{
return buffer;
}
int get_severity() const throw() BOOST_OVERRIDE
{
return severity_level(code);
}
int get_errorcode() const throw() BOOST_OVERRIDE
{
return code;
}
char const* get_related_name() const throw() BOOST_OVERRIDE
{
return "<unknown>";
}
bool is_recoverable() const throw() BOOST_OVERRIDE
{
switch (get_errorcode()) {
case preprocess_exception::no_error:        
case preprocess_exception::macro_redefinition:
case preprocess_exception::macro_insertion_error:
case preprocess_exception::bad_macro_definition:
case preprocess_exception::illegal_redefinition:
case preprocess_exception::duplicate_parameter_name:
case preprocess_exception::invalid_macroname:
case preprocess_exception::bad_include_file:
case preprocess_exception::bad_include_statement:
case preprocess_exception::bad_has_include_expression:
case preprocess_exception::ill_formed_directive:
case preprocess_exception::error_directive:
case preprocess_exception::warning_directive:
case preprocess_exception::ill_formed_expression:
case preprocess_exception::missing_matching_if:
case preprocess_exception::missing_matching_endif:
case preprocess_exception::unbalanced_if_endif:
case preprocess_exception::bad_define_statement:
case preprocess_exception::bad_define_statement_va_args:
case preprocess_exception::bad_define_statement_va_opt:
case preprocess_exception::bad_define_statement_va_opt_parens:
case preprocess_exception::bad_define_statement_va_opt_recurse:
case preprocess_exception::bad_line_statement:
case preprocess_exception::bad_line_number:
case preprocess_exception::bad_line_filename:
case preprocess_exception::bad_undefine_statement:
case preprocess_exception::division_by_zero:
case preprocess_exception::integer_overflow:
case preprocess_exception::ill_formed_integer_literal:
case preprocess_exception::ill_formed_character_literal:
case preprocess_exception::character_literal_out_of_range:
case preprocess_exception::last_line_not_terminated:
case preprocess_exception::include_nesting_too_deep:
case preprocess_exception::illegal_operator_redefinition:
case preprocess_exception::incompatible_config:
case preprocess_exception::ill_formed_pragma_option:
case preprocess_exception::ill_formed_pragma_message:
case preprocess_exception::pragma_message_directive:
return true;

case preprocess_exception::unexpected_error:
case preprocess_exception::ill_formed_operator:
case preprocess_exception::too_few_macroarguments:
case preprocess_exception::too_many_macroarguments:
case preprocess_exception::empty_macroarguments:
case preprocess_exception::improperly_terminated_macro:
case preprocess_exception::invalid_concat:
case preprocess_exception::could_not_open_output_file:
break;
}
return false;
}

static char const *error_text(int code)
{
static char const *preprocess_exception_errors[] = {
"no error",                                 
"unexpected error (should not happen)",     
"illegal macro redefinition",               
"macro definition failed (out of memory?)", 
"could not find include file",              
"ill formed #include directive",            
"ill formed __has_include expression",      
"ill formed preprocessor directive",        
"encountered #error directive or #pragma wave stop()", 
"encountered #warning directive",           
"ill formed preprocessor expression",       
"the #if for this directive is missing",    
"detected at least one missing #endif directive",   
"ill formed preprocessing operator",        
"ill formed #define directive",             
"__VA_ARGS__ can only appear in the "
"expansion of a C99 variadic macro",        
"__VA_OPT__ can only appear in the "
"expansion of a C++20 variadic macro",      
"__VA_OPT__ must be followed by a left "
"paren in a C++20 variadic macro",          
"__VA_OPT__() may not contain __VA_OPT__",  
"too few macro arguments",                  
"too many macro arguments",                 
"empty macro arguments are not supported in pure C++ mode, "
"use variadics mode to allow these",        
"improperly terminated macro invocation "
"or replacement-list terminates in partial "
"macro expansion (not supported yet)",      
"ill formed #line directive",               
"line number argument of #line directive "
"should consist out of decimal digits "
"only and must be in range of [1..INT_MAX]", 
"filename argument of #line directive should "
"be a narrow string literal",               
"#undef may not be used on this predefined name",   
"invalid macro definition",                 
"this predefined name may not be redefined",        
"duplicate macro parameter name",           
"pasting the following two tokens does not "
"give a valid preprocessing token",         
"last line of file ends without a newline", 
"unknown or illformed pragma option",       
"include files nested too deep",            
"misplaced operator defined()",             
"the name is already used in this scope as "
"a macro or scope name",                    
"undefined macro or scope name may not be imported", 
"ill formed macro name",                    
"qualified names are supported in C++11 mode only",  
"division by zero in preprocessor expression",       
"integer overflow in preprocessor expression",       
"this cannot be used as a macro name as it is "
"an operator in C++",                       
"ill formed integer literal or integer constant too large",   
"ill formed character literal",             
"unbalanced #if/#endif in include file",    
"expression contains out of range character literal", 
"could not open output file",               
"incompatible state information",           
"illformed pragma message",                 
"encountered #pragma message directive"     
};
BOOST_ASSERT(no_error <= code && code <= last_error_number);
return preprocess_exception_errors[code];
}

static util::severity severity_level(int code)
{
static util::severity preprocess_exception_severity[] = {
util::severity_remark,             
util::severity_fatal,              
util::severity_warning,            
util::severity_fatal,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_fatal,              
util::severity_warning,            
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_warning,            
util::severity_warning,            
util::severity_warning,            
util::severity_error,              
util::severity_warning,            
util::severity_warning,            
util::severity_warning,            
util::severity_warning,            
util::severity_commandline_error,  
util::severity_warning,            
util::severity_error,              
util::severity_error,              
util::severity_warning,            
util::severity_warning,            
util::severity_fatal,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_fatal,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_error,              
util::severity_warning,            
util::severity_warning,            
util::severity_error,              
util::severity_remark,             
util::severity_warning,            
util::severity_remark,             
};
BOOST_ASSERT(no_error <= code && code <= last_error_number);
return preprocess_exception_severity[code];
}
static char const *severity_text(int code)
{
return util::get_severity(severity_level(code));
}

private:
char buffer[512];
error_code code;
};

class BOOST_SYMBOL_VISIBLE macro_handling_exception :
public preprocess_exception
{
public:
macro_handling_exception(char const *what_, error_code code, std::size_t line_,
std::size_t column_, char const *filename_, char const *macroname) throw()
:   preprocess_exception(what_, code, line_, column_, filename_)
{
unsigned int off = 0;
while (off < sizeof(name) && *macroname)
name[off++] = *macroname++;
name[off] = 0;
}
~macro_handling_exception() throw() {}

char const *what() const throw() BOOST_OVERRIDE
{
return "boost::wave::macro_handling_exception";
}
char const* get_related_name() const throw() BOOST_OVERRIDE
{
return name;
}

private:
char name[512];
};

inline bool
is_recoverable(cpp_exception const& e)
{
return e.is_recoverable();
}

}   
}   

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif 
