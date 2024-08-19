

#if !defined(BOOST_DEFAULT_PREPROCESSING_HOOKS_HPP_INCLUDED)
#define BOOST_DEFAULT_PREPROCESSING_HOOKS_HPP_INCLUDED

#include <boost/wave/wave_config.hpp>
#include <boost/wave/util/cpp_include_paths.hpp>
#include <boost/wave/cpp_exceptions.hpp>

#include <vector>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace wave {
namespace context_policies {

struct default_preprocessing_hooks 
{

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename TokenT, typename ContainerT>
void expanding_function_like_macro(
TokenT const& macrodef, std::vector<TokenT> const& formal_args, 
ContainerT const& definition,
TokenT const& macrocall, std::vector<ContainerT> const& arguments) 
{}
#else
template <typename ContextT, typename TokenT, typename ContainerT, typename IteratorT>
bool 
expanding_function_like_macro(ContextT const& ctx,
TokenT const& macrodef, std::vector<TokenT> const& formal_args, 
ContainerT const& definition,
TokenT const& macrocall, std::vector<ContainerT> const& arguments,
IteratorT const& seqstart, IteratorT const& seqend) 
{ return false; }   
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename TokenT, typename ContainerT>
void expanding_object_like_macro(TokenT const& macro, 
ContainerT const& definition, TokenT const& macrocall)
{}
#else
template <typename ContextT, typename TokenT, typename ContainerT>
bool 
expanding_object_like_macro(ContextT const& ctx, TokenT const& macro, 
ContainerT const& definition, TokenT const& macrocall)
{ return false; }   
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename ContainerT>
void expanded_macro(ContainerT const& result)
{}
#else
template <typename ContextT, typename ContainerT>
void expanded_macro(ContextT const& ctx, ContainerT const& result)
{}
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename ContainerT>
void rescanned_macro(ContainerT const& result)
{}
#else
template <typename ContextT, typename ContainerT>
void rescanned_macro(ContextT const& ctx, ContainerT const& result)
{}
#endif

template <typename ContextT>
bool 
locate_include_file(ContextT& ctx, std::string &file_path, 
bool is_system, char const *current_name, std::string &dir_path, 
std::string &native_name) 
{
if (!ctx.find_include_file (file_path, dir_path, is_system, current_name))
return false;   

namespace fs = boost::filesystem;

fs::path native_path(wave::util::create_path(file_path));
if (!fs::exists(native_path)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_include_file, 
file_path.c_str(), ctx.get_main_pos());
return false;
}

native_name = wave::util::native_file_string(native_path);

return true;      
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
void 
found_include_directive(std::string const& filename, bool include_next) 
{}
#else
template <typename ContextT>
bool 
found_include_directive(ContextT const& ctx, std::string const& filename, 
bool include_next) 
{
return false;    
}
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
void 
opened_include_file(std::string const& relname, std::string const& absname, 
std::size_t include_depth, bool is_system_include) 
{}
#else
template <typename ContextT>
void 
opened_include_file(ContextT const& ctx, std::string const& relname, 
std::string const& absname, bool is_system_include) 
{}
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
void
returning_from_include_file() 
{}
#else
template <typename ContextT>
void
returning_from_include_file(ContextT const& ctx) 
{}
#endif

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
template <typename ContextT>
void
detected_include_guard(ContextT const& ctx, std::string const& filename,
std::string const& include_guard) 
{}

template <typename ContextT, typename TokenT>
void
detected_pragma_once(ContextT const& ctx, TokenT const& pragma_token,
std::string const& filename) 
{}
#endif 

template <typename ContextT, typename ContainerT>
bool 
interpret_pragma(ContextT const& ctx, ContainerT &pending, 
typename ContextT::token_type const& option, ContainerT const& values, 
typename ContextT::token_type const& act_token)
{
return false;
}

template <typename ContextT, typename ContainerT>
bool 
emit_line_directive(ContextT const& ctx, ContainerT &pending, 
typename ContextT::token_type const& act_token)
{
return false;
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename TokenT, typename ParametersT, typename DefinitionT>
void
defined_macro(TokenT const& macro_name, bool is_functionlike, 
ParametersT const& parameters, DefinitionT const& definition, 
bool is_predefined)
{}
#else
template <
typename ContextT, typename TokenT, typename ParametersT, 
typename DefinitionT
>
void
defined_macro(ContextT const& ctx, TokenT const& macro_name, 
bool is_functionlike, ParametersT const& parameters, 
DefinitionT const& definition, bool is_predefined)
{}
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename TokenT>
void
undefined_macro(TokenT const& macro_name)
{}
#else
template <typename ContextT, typename TokenT>
void
undefined_macro(ContextT const& ctx, TokenT const& macro_name)
{}
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename TokenT>
void
found_directive(TokenT const& directive)
{}
#else
template <typename ContextT, typename TokenT>
bool
found_directive(ContextT const& ctx, TokenT const& directive)
{ return false; }   
#endif

template <typename ContextT, typename ContainerT>
bool
found_unknown_directive(ContextT const& ctx, ContainerT const& line, 
ContainerT& pending)
{ return false; }   

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename ContainerT>
void
evaluated_conditional_expression(ContainerT const& expression, 
bool expression_value)
{}
#else
template <typename ContextT, typename TokenT, typename ContainerT>
bool
evaluated_conditional_expression(ContextT const& ctx, 
TokenT const& directive, ContainerT const& expression, 
bool expression_value)
{ return false; }         
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename TokenT>
void
skipped_token(TokenT const& token)
{}
#else
template <typename ContextT, typename TokenT>
void
skipped_token(ContextT const& ctx, TokenT const& token)
{}
#endif

template <typename ContextT, typename TokenT>
TokenT const&
generated_token(ContextT const& ctx, TokenT const& t)
{ return t; }

template <typename ContextT, typename TokenT>
bool
may_skip_whitespace(ContextT const& ctx, TokenT& token, bool& skipped_newline)
{ return false; }

#if BOOST_WAVE_SUPPORT_WARNING_DIRECTIVE != 0
template <typename ContextT, typename ContainerT>
bool
found_warning_directive(ContextT const& ctx, ContainerT const& message)
{ return false; }
#endif

template <typename ContextT, typename ContainerT>
bool
found_error_directive(ContextT const& ctx, ContainerT const& message)
{ return false; }

template <typename ContextT, typename ContainerT>
void
found_line_directive(ContextT const& ctx, ContainerT const& arguments,
unsigned int line, std::string const& filename)
{}

template <typename ContextT, typename ExceptionT>
void
throw_exception(ContextT const& ctx, ExceptionT const& e)
{
boost::throw_exception(e);
}
};

}   
}   
}   

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif 
