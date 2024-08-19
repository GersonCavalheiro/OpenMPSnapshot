

#if !defined(BOOST_CPP_CONTEXT_HPP_907485E2_6649_4A87_911B_7F7225F3E5B8_INCLUDED)
#define BOOST_CPP_CONTEXT_HPP_907485E2_6649_4A87_911B_7F7225F3E5B8_INCLUDED

#include <string>
#include <vector>
#include <stack>

#include <boost/concept_check.hpp>
#include <boost/noncopyable.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <boost/wave/wave_config.hpp>
#if BOOST_WAVE_SERIALIZATION != 0
#include <boost/serialization/serialization.hpp>
#include <boost/wave/wave_config_constant.hpp>
#endif
#include <boost/wave/token_ids.hpp>

#include <boost/wave/util/unput_queue_iterator.hpp>
#include <boost/wave/util/cpp_ifblock.hpp>
#include <boost/wave/util/cpp_include_paths.hpp>
#include <boost/wave/util/iteration_context.hpp>
#include <boost/wave/util/cpp_iterator.hpp>
#include <boost/wave/util/cpp_macromap.hpp>

#include <boost/wave/preprocessing_hooks.hpp>
#include <boost/wave/whitespace_handling.hpp>
#include <boost/wave/cpp_iteration_context.hpp>
#include <boost/wave/language_support.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace wave {


struct this_type {};

template <
typename IteratorT,
typename LexIteratorT,
typename InputPolicyT = iteration_context_policies::load_file_to_string,
typename HooksT = context_policies::eat_whitespace<typename LexIteratorT::token_type>,
typename DerivedT = this_type
>
class context : private boost::noncopyable
{
private:
typedef typename mpl::if_<
is_same<DerivedT, this_type>, context, DerivedT
>::type actual_context_type;

public:
BOOST_CLASS_REQUIRE(IteratorT, boost, ForwardIteratorConcept);

typedef typename LexIteratorT::token_type       token_type;
typedef typename token_type::string_type        string_type;

typedef IteratorT                               target_iterator_type;
typedef LexIteratorT                            lexer_type;
typedef pp_iterator<context>                    iterator_type;

typedef InputPolicyT                            input_policy_type;
typedef typename token_type::position_type      position_type;

typedef std::list<token_type, boost::fast_pool_allocator<token_type> >
token_sequence_type;
typedef HooksT                                  hook_policy_type;

private:
typedef boost::shared_ptr<base_iteration_context<context, lexer_type> >
iteration_ptr_type;
typedef boost::wave::util::iteration_context_stack<iteration_ptr_type>
iteration_context_stack_type;
typedef typename iteration_context_stack_type::size_type iter_size_type;

context *this_() { return this; }           

public:
context(target_iterator_type const &first_, target_iterator_type const &last_,
char const *fname = "<Unknown>", HooksT const &hooks_ = HooksT())
:   first(first_), last(last_), filename(fname)
, has_been_initialized(false)
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
, current_filename(fname)
#endif
, current_relative_filename(fname)
, macros(*this_())
, language(language_support(
support_cpp
| support_option_convert_trigraphs
| support_option_emit_line_directives
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
| support_option_include_guard_detection
#endif
#if BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES != 0
| support_option_emit_pragma_directives
#endif
| support_option_insert_whitespace
))
, hooks(hooks_)
{
macros.init_predefined_macros(fname);
}


iterator_type begin()
{
std::string fname(filename);
if (filename != "<Unknown>" && filename != "<stdin>") {
using namespace boost::filesystem;
path fpath(util::complete_path(path(filename)));
fname = fpath.string();
}
return iterator_type(*this, first, last, position_type(fname.c_str()));
}
iterator_type begin(
target_iterator_type const &first_,
target_iterator_type const &last_)
{
std::string fname(filename);
if (filename != "<Unknown>" && filename != "<stdin>") {
using namespace boost::filesystem;
path fpath(util::complete_path(path(filename)));
fname = fpath.string();
}
return iterator_type(*this, first_, last_, position_type(fname.c_str()));
}
iterator_type end() const
{ return iterator_type(); }

bool add_include_path(char const *path_)
{ return includes.add_include_path(path_, false);}
bool add_sysinclude_path(char const *path_)
{ return includes.add_include_path(path_, true);}
void set_sysinclude_delimiter() { includes.set_sys_include_delimiter(); }
typename iteration_context_stack_type::size_type get_iteration_depth() const
{ return iter_ctxs.size(); }

#if BOOST_WAVE_ENABLE_COMMANDLINE_MACROS != 0
template <typename StringT>
bool add_macro_definition(StringT macrostring, bool is_predefined = false)
{
return boost::wave::util::add_macro_definition(*this,
util::to_string<std::string>(macrostring), is_predefined,
get_language());
}
#endif
template <typename StringT>
bool add_macro_definition(StringT const &name, position_type const& pos,
bool has_params, std::vector<token_type> &parameters,
token_sequence_type &definition, bool is_predefined = false)
{
return macros.add_macro(
token_type(T_IDENTIFIER, util::to_string<string_type>(name), pos),
has_params, parameters, definition, is_predefined);
}
template <typename StringT>
bool is_defined_macro(StringT const &str) const
{
return macros.is_defined(util::to_string<string_type>(str));
}
template <typename StringT>
bool get_macro_definition(StringT const &name,
bool &has_params, bool &is_predefined, position_type &pos,
std::vector<token_type> &parameters,
token_sequence_type &definition) const
{
return macros.get_macro(util::to_string<string_type>(name),
has_params, is_predefined, pos, parameters, definition);
}
template <typename StringT>
bool remove_macro_definition(StringT const& undefname, bool even_predefined = false)
{
string_type name = util::to_string<string_type>(undefname);
typename string_type::size_type pos = name.find_first_not_of(" \t");
if (pos != string_type::npos) {
typename string_type::size_type endpos = name.find_last_not_of(" \t");
name = name.substr(pos, endpos-pos+1);
}

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
includes.remove_pragma_once_header(
util::to_string<std::string>(name));
#endif
return macros.remove_macro(name, macros.get_main_pos(), even_predefined);
}
void reset_macro_definitions()
{ macros.reset_macromap(); macros.init_predefined_macros(); }

typedef boost::wave::util::macromap<context> macromap_type;
typedef typename macromap_type::name_iterator name_iterator;
typedef typename macromap_type::const_name_iterator const_name_iterator;

name_iterator macro_names_begin() { return macros.begin(); }
name_iterator macro_names_end() { return macros.end(); }
const_name_iterator macro_names_begin() const { return macros.begin(); }
const_name_iterator macro_names_end() const { return macros.end(); }

bool add_macro_definition(token_type const &name, bool has_params,
std::vector<token_type> &parameters, token_sequence_type &definition,
bool is_predefined = false)
{
return macros.add_macro(name, has_params, parameters, definition,
is_predefined);
}

static std::string get_version()
{
boost::wave::util::predefined_macros p;
return util::to_string<std::string>(p.get_fullversion());
}
static std::string get_version_string()
{
boost::wave::util::predefined_macros p;
return util::to_string<std::string>(p.get_versionstr());
}

void set_language(boost::wave::language_support language_,
bool reset_macros = true)
{
language = language_;
if (reset_macros)
reset_macro_definitions();
}
boost::wave::language_support get_language() const { return language; }

position_type &get_main_pos() { return macros.get_main_pos(); }
position_type const& get_main_pos() const { return macros.get_main_pos(); }

void set_max_include_nesting_depth(iter_size_type new_depth)
{ iter_ctxs.set_max_include_nesting_depth(new_depth); }
iter_size_type get_max_include_nesting_depth() const
{ return iter_ctxs.get_max_include_nesting_depth(); }

hook_policy_type &get_hooks() { return hooks; }
hook_policy_type const &get_hooks() const { return hooks; }

actual_context_type& derived()
{ return *static_cast<actual_context_type*>(this); }
actual_context_type const& derived() const
{ return *static_cast<actual_context_type const*>(this); }

boost::filesystem::path get_current_directory() const
{ return includes.get_current_directory(); }

#if !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
protected:
friend class boost::wave::pp_iterator<context>;
friend class boost::wave::impl::pp_iterator_functor<context>;
friend class boost::wave::util::macromap<context>;
#endif

void init_context()
{
if (!has_been_initialized) {
std::string fname(filename);
if (filename != "<Unknown>" && filename != "<stdin>") {
using namespace boost::filesystem;
path fpath(util::complete_path(path(filename)));
fname = fpath.string();
includes.set_current_directory(fname.c_str());
}
has_been_initialized = true;  
}
}

template <typename IteratorT2>
bool is_defined_macro(IteratorT2 const &begin, IteratorT2 const &end) const
{ return macros.is_defined(begin, end); }

void set_current_directory(char const *path_)
{ includes.set_current_directory(path_); }

bool get_if_block_status() const { return ifblocks.get_status(); }
bool get_if_block_some_part_status() const
{ return ifblocks.get_some_part_status(); }
bool get_enclosing_if_block_status() const
{ return ifblocks.get_enclosing_status(); }
void enter_if_block(bool new_status)
{ ifblocks.enter_if_block(new_status); }
bool enter_elif_block(bool new_status)
{ return ifblocks.enter_elif_block(new_status); }
bool enter_else_block() { return ifblocks.enter_else_block(); }
bool exit_if_block() { return ifblocks.exit_if_block(); }
typename boost::wave::util::if_block_stack::size_type get_if_block_depth() const
{ return ifblocks.get_if_block_depth(); }

iteration_ptr_type pop_iteration_context()
{ iteration_ptr_type top = iter_ctxs.top(); iter_ctxs.pop(); return top; }
void push_iteration_context(position_type const &act_pos, iteration_ptr_type iter_ctx)
{ iter_ctxs.push(*this, act_pos, iter_ctx); }

template <typename IteratorT2>
token_type expand_tokensequence(IteratorT2 &first_, IteratorT2 const &last_,
token_sequence_type &pending, token_sequence_type &expanded,
bool& seen_newline, bool expand_defined = false,
bool expand_has_include = false)
{
return macros.expand_tokensequence(first_, last_, pending, expanded,
seen_newline, expand_defined, expand_has_include);
}

template <typename IteratorT2>
void expand_whole_tokensequence(IteratorT2 &first_, IteratorT2 const &last_,
token_sequence_type &expanded, bool expand_defined = true,
bool expand_has_include = true)
{
macros.expand_whole_tokensequence(
expanded, first_, last_,
expand_defined, expand_has_include);

boost::wave::util::impl::remove_placeholders(expanded);
}

public:
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
void set_current_filename(char const *real_name)
{ current_filename = real_name; }
std::string const &get_current_filename() const
{ return current_filename; }

bool has_pragma_once(std::string const &filename_)
{ return includes.has_pragma_once(filename_); }
bool add_pragma_once_header(std::string const &filename_,
std::string const& guard_name)
{
get_hooks().detected_include_guard(derived(), filename_, guard_name);
return includes.add_pragma_once_header(filename_, guard_name);
}
bool add_pragma_once_header(token_type const &pragma_,
std::string const &filename_)
{
get_hooks().detected_pragma_once(derived(), pragma_, filename_);
return includes.add_pragma_once_header(filename_,
"__BOOST_WAVE_PRAGMA_ONCE__");
}
#endif

void set_current_relative_filename(char const *real_name)
{ current_relative_filename = real_name; }
std::string const &get_current_relative_filename() const
{ return current_relative_filename; }

bool find_include_file (std::string &s, std::string &d, bool is_system,
char const *current_file) const
{ return includes.find_include_file(s, d, is_system, current_file); }

#if BOOST_WAVE_SERIALIZATION != 0
public:
BOOST_STATIC_CONSTANT(unsigned int, version = 0x10);
BOOST_STATIC_CONSTANT(unsigned int, version_mask = 0x0f);

private:
friend class boost::serialization::access;
template<class Archive>
void save(Archive & ar, const unsigned int version) const
{
using namespace boost::serialization;

string_type cfg(BOOST_PP_STRINGIZE(BOOST_WAVE_CONFIG));
string_type kwd(BOOST_WAVE_PRAGMA_KEYWORD);
string_type strtype(BOOST_PP_STRINGIZE((BOOST_WAVE_STRINGTYPE)));
ar & make_nvp("config", cfg);
ar & make_nvp("pragma_keyword", kwd);
ar & make_nvp("string_type", strtype);

ar & make_nvp("language_options", language);
ar & make_nvp("macro_definitions", macros);
ar & make_nvp("include_settings", includes);
}
template<class Archive>
void load(Archive & ar, const unsigned int loaded_version)
{
using namespace boost::serialization;
if (version != (loaded_version & ~version_mask)) {
BOOST_WAVE_THROW_CTX((*this), preprocess_exception,
incompatible_config, "cpp_context state version",
get_main_pos());
return;
}

string_type config, pragma_keyword, string_type_str;

ar & make_nvp("config", config);
if (config != BOOST_PP_STRINGIZE(BOOST_WAVE_CONFIG)) {
BOOST_WAVE_THROW_CTX((*this), preprocess_exception,
incompatible_config, "BOOST_WAVE_CONFIG", get_main_pos());
return;
}

ar & make_nvp("pragma_keyword", pragma_keyword);
if (pragma_keyword != BOOST_WAVE_PRAGMA_KEYWORD) {
BOOST_WAVE_THROW_CTX((*this), preprocess_exception,
incompatible_config, "BOOST_WAVE_PRAGMA_KEYWORD",
get_main_pos());
return;
}

ar & make_nvp("string_type", string_type_str);
if (string_type_str != BOOST_PP_STRINGIZE((BOOST_WAVE_STRINGTYPE))) {
BOOST_WAVE_THROW_CTX((*this), preprocess_exception,
incompatible_config, "BOOST_WAVE_STRINGTYPE", get_main_pos());
return;
}

try {
ar & make_nvp("language_options", language);
ar & make_nvp("macro_definitions", macros);
ar & make_nvp("include_settings", includes);
}
catch (boost::wave::preprocess_exception const& e) {
get_hooks().throw_exception(derived(), e);
}
}
BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

private:
target_iterator_type first;         
target_iterator_type last;
std::string filename;               
bool has_been_initialized;          
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
std::string current_filename;       
#endif
std::string current_relative_filename;        

boost::wave::util::if_block_stack ifblocks;   
boost::wave::util::include_paths includes;    
iteration_context_stack_type iter_ctxs;       
macromap_type macros;                         
boost::wave::language_support language;       
hook_policy_type hooks;                       
};

}   
}   

#if BOOST_WAVE_SERIALIZATION != 0
namespace boost { namespace serialization {

template<
typename Iterator, typename LexIterator,
typename InputPolicy, typename Hooks
>
struct tracking_level<boost::wave::context<Iterator, LexIterator, InputPolicy, Hooks> >
{
typedef mpl::integral_c_tag tag;
typedef mpl::int_<track_never> type;
BOOST_STATIC_CONSTANT(
int,
value = tracking_level::type::value
);
};

template<
typename Iterator, typename LexIterator,
typename InputPolicy, typename Hooks
>
struct version<boost::wave::context<Iterator, LexIterator, InputPolicy, Hooks> >
{
typedef boost::wave::context<Iterator, LexIterator, InputPolicy, Hooks>
target_type;
typedef mpl::int_<target_type::version> type;
typedef mpl::integral_c_tag tag;
BOOST_STATIC_CONSTANT(unsigned int, value = version::type::value);
};

}}  
#endif

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif 
