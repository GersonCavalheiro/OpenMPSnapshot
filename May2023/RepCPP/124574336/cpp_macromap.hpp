

#if !defined(BOOST_CPP_MACROMAP_HPP_CB8F51B0_A3F0_411C_AEF4_6FF631B8B414_INCLUDED)
#define BOOST_CPP_MACROMAP_HPP_CB8F51B0_A3F0_411C_AEF4_6FF631B8B414_INCLUDED

#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <list>
#include <map>
#include <set>
#include <vector>
#include <iterator>
#include <algorithm>

#include <boost/assert.hpp>
#include <boost/wave/wave_config.hpp>
#if BOOST_WAVE_SERIALIZATION != 0
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#endif

#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>

#include <boost/wave/util/time_conversion_helper.hpp>
#include <boost/wave/util/unput_queue_iterator.hpp>
#include <boost/wave/util/macro_helpers.hpp>
#include <boost/wave/util/macro_definition.hpp>
#include <boost/wave/util/symbol_table.hpp>
#include <boost/wave/util/cpp_macromap_utils.hpp>
#include <boost/wave/util/cpp_macromap_predef.hpp>
#include <boost/wave/util/filesystem_compatibility.hpp>
#include <boost/wave/grammars/cpp_defined_grammar_gen.hpp>
#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
#include <boost/wave/grammars/cpp_has_include_grammar_gen.hpp>
#endif

#include <boost/wave/wave_version.hpp>
#include <boost/wave/cpp_exceptions.hpp>
#include <boost/wave/language_support.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace boost { namespace wave { namespace util {

template <typename ContextT>
class macromap {

typedef macromap<ContextT>                      self_type;
typedef typename ContextT::token_type           token_type;
typedef typename token_type::string_type        string_type;
typedef typename token_type::position_type      position_type;

typedef typename ContextT::token_sequence_type  definition_container_type;
typedef std::vector<token_type>                 parameter_container_type;

typedef macro_definition<token_type, definition_container_type>
macro_definition_type;
typedef symbol_table<string_type, macro_definition_type>
defined_macros_type;
typedef typename defined_macros_type::value_type::second_type
macro_ref_type;

public:
macromap(ContextT &ctx_)
:   current_macros(0), defined_macros(new defined_macros_type(1)),
main_pos("", 0), ctx(ctx_), macro_uid(1)
{
current_macros = defined_macros.get();
}
~macromap() {}

bool add_macro(token_type const &name, bool has_parameters,
parameter_container_type &parameters,
definition_container_type &definition, bool is_predefined = false,
defined_macros_type *scope = 0);

bool is_defined(string_type const &name,
typename defined_macros_type::iterator &it,
defined_macros_type *scope = 0) const;

template <typename IteratorT>
bool is_defined(IteratorT const &begin, IteratorT const &end) const;

bool is_defined(string_type const &str) const;

#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
template <typename IteratorT>
bool has_include(IteratorT const &begin, IteratorT const &end,
bool is_quoted_filename, bool is_system) const;
#endif

bool get_macro(string_type const &name, bool &has_parameters,
bool &is_predefined, position_type &pos,
parameter_container_type &parameters,
definition_container_type &definition,
defined_macros_type *scope = 0) const;

bool remove_macro(string_type const &name, position_type const& pos,
bool even_predefined = false);

template <typename IteratorT, typename ContainerT>
token_type const &expand_tokensequence(IteratorT &first,
IteratorT const &last, ContainerT &pending, ContainerT &expanded,
bool& seen_newline, bool expand_operator_defined,
bool expand_operator_has_include);

template <typename IteratorT, typename ContainerT>
void expand_whole_tokensequence(ContainerT &expanded,
IteratorT &first, IteratorT const &last,
bool expand_operator_defined,
bool expand_operator_has_include);

void init_predefined_macros(char const *fname = "<Unknown>",
defined_macros_type *scope = 0, bool at_global_scope = true);
void predefine_macro(defined_macros_type *scope, string_type const &name,
token_type const &t);

void reset_macromap();

position_type &get_main_pos() { return main_pos; }
position_type const& get_main_pos() const { return main_pos; }

typedef typename defined_macros_type::name_iterator name_iterator;
typedef typename defined_macros_type::const_name_iterator const_name_iterator;

name_iterator begin()
{ return defined_macros_type::make_iterator(current_macros->begin()); }
name_iterator end()
{ return defined_macros_type::make_iterator(current_macros->end()); }
const_name_iterator begin() const
{ return defined_macros_type::make_iterator(current_macros->begin()); }
const_name_iterator end() const
{ return defined_macros_type::make_iterator(current_macros->end()); }

protected:
template <typename IteratorT, typename ContainerT>
token_type const &expand_tokensequence_worker(ContainerT &pending,
unput_queue_iterator<IteratorT, token_type, ContainerT> &first,
unput_queue_iterator<IteratorT, token_type, ContainerT> const &last,
bool& seen_newline, bool expand_operator_defined,
bool expand_operator_has_include,
boost::optional<position_type> expanding_pos);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename IteratorT, typename ContainerT, typename SizeT>
typename std::vector<ContainerT>::size_type collect_arguments (
token_type const curr_token, std::vector<ContainerT> &arguments,
IteratorT &next, IteratorT const &end, SizeT const &parameter_count,
bool& seen_newline);
#else
template <typename IteratorT, typename ContainerT, typename SizeT>
typename std::vector<ContainerT>::size_type collect_arguments (
token_type const curr_token, std::vector<ContainerT> &arguments,
IteratorT &next, IteratorT &endparen, IteratorT const &end,
SizeT const &parameter_count, bool& seen_newline);
#endif

template <typename IteratorT, typename ContainerT>
bool expand_macro(ContainerT &pending, token_type const &name,
typename defined_macros_type::iterator it,
IteratorT &first, IteratorT const &last,
bool& seen_newline, bool expand_operator_defined,
bool expand_operator_has_include,
boost::optional<position_type> expanding_pos,
defined_macros_type *scope = 0, ContainerT *queue_symbol = 0);

template <typename ContainerT>
bool expand_predefined_macro(token_type const &curr_token,
ContainerT &expanded);

template <typename ContainerT>
void expand_argument (typename std::vector<ContainerT>::size_type arg,
std::vector<ContainerT> &arguments,
std::vector<ContainerT> &expanded_args, bool expand_operator_defined,
bool expand_operator_has_include,
std::vector<bool> &has_expanded_args);

template <typename ContainerT>
void expand_replacement_list(
typename macro_definition_type::const_definition_iterator_t cbeg,
typename macro_definition_type::const_definition_iterator_t cend,
std::vector<ContainerT> &arguments,
bool expand_operator_defined,
bool expand_operator_has_include,
ContainerT &expanded);

template <typename IteratorT, typename ContainerT>
void rescan_replacement_list(token_type const &curr_token,
macro_definition_type &macrodef, ContainerT &replacement_list,
ContainerT &expanded, bool expand_operator_defined,
bool expand_operator_has_include,
IteratorT &nfirst, IteratorT const &nlast);

template <typename IteratorT, typename ContainerT>
token_type const &resolve_defined(IteratorT &first, IteratorT const &last,
ContainerT &expanded);

#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
template <typename IteratorT, typename ContainerT>
token_type const &resolve_has_include(IteratorT &first, IteratorT const &last,
ContainerT &expanded);
#endif

template <typename IteratorT, typename ContainerT>
bool resolve_operator_pragma(IteratorT &first,
IteratorT const &last, ContainerT &expanded, bool& seen_newline);

template <typename ContainerT>
bool concat_tokensequence(ContainerT &expanded);

template <typename ContainerT>
bool is_valid_concat(string_type new_value,
position_type const &pos, ContainerT &rescanned);

static bool is_space(char);

template <typename ContainerT>
static void set_expand_positions(ContainerT &tokens, position_type pos);

#if BOOST_WAVE_SERIALIZATION != 0
public:
BOOST_STATIC_CONSTANT(unsigned int, version = 0x10);
BOOST_STATIC_CONSTANT(unsigned int, version_mask = 0x0f);

private:
friend class boost::serialization::access;
template<typename Archive>
void save(Archive &ar, const unsigned int version) const
{
using namespace boost::serialization;
ar & make_nvp("defined_macros", defined_macros);
}
template<typename Archive>
void load(Archive &ar, const unsigned int loaded_version)
{
using namespace boost::serialization;
if (version != (loaded_version & ~version_mask)) {
BOOST_WAVE_THROW(preprocess_exception, incompatible_config,
"cpp_context state version", get_main_pos());
}
ar & make_nvp("defined_macros", defined_macros);
current_macros = defined_macros.get();
}
BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

private:
defined_macros_type *current_macros;                   
boost::shared_ptr<defined_macros_type> defined_macros; 

token_type act_token;       
position_type main_pos;     
string_type base_name;      
ContextT &ctx;              
long macro_uid;
predefined_macros predef;   
};

template <typename ContextT>
inline bool
macromap<ContextT>::add_macro(token_type const &name, bool has_parameters,
parameter_container_type &parameters, definition_container_type &definition,
bool is_predefined, defined_macros_type *scope)
{
if (!is_predefined && impl::is_special_macroname (ctx, name.get_value())) {
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
illegal_redefinition, name.get_value().c_str(), main_pos,
name.get_value().c_str());
return false;
}
if (boost::wave::need_variadics(ctx.get_language()) &&
"__VA_ARGS__" == name.get_value())
{
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
bad_define_statement_va_args, name.get_value().c_str(), main_pos,
name.get_value().c_str());
return false;
}
if (boost::wave::need_variadics(ctx.get_language()) &&
"__VA_OPT__" == name.get_value())
{
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
bad_define_statement_va_opt, name.get_value().c_str(), main_pos,
name.get_value().c_str());
return false;
}
#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
if (boost::wave::need_has_include(ctx.get_language()) &&
"__has_include" == name.get_value())
{
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
bad_define_statement_va_opt, name.get_value().c_str(), main_pos,
name.get_value().c_str());
return false;
}
#endif
if (AltExtTokenType == (token_id(name) & ExtTokenOnlyMask)) {
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
illegal_operator_redefinition, name.get_value().c_str(), main_pos,
name.get_value().c_str());
return false;
}

defined_macros_type* current_scope = scope ? scope : current_macros;
typename defined_macros_type::iterator it = current_scope->find(name.get_value());

if (it != current_scope->end()) {
macro_definition_type* macrodef = (*it).second.get();
if (macrodef->is_functionlike != has_parameters ||
!impl::parameters_equal(macrodef->macroparameters, parameters) ||
!impl::definition_equals(macrodef->macrodefinition, definition))
{
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
macro_redefinition, name.get_value().c_str(), main_pos,
name.get_value().c_str());
}
return false;
}

if (has_parameters) {
std::set<typename token_type::string_type> names;

typedef typename parameter_container_type::iterator
parameter_iterator_type;
typedef typename std::set<typename token_type::string_type>::iterator
name_iterator_type;

parameter_iterator_type end = parameters.end();
for (parameter_iterator_type itp = parameters.begin(); itp != end; ++itp)
{
name_iterator_type pit = names.find((*itp).get_value());

if (pit != names.end()) {
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
duplicate_parameter_name, (*pit).c_str(), main_pos,
name.get_value().c_str());
return false;
}
names.insert((*itp).get_value());
}
}

#if BOOST_WAVE_SUPPORT_VA_OPT != 0
if (boost::wave::need_va_opt(ctx.get_language())) {
typedef typename macro_definition_type::const_definition_iterator_t iter_t;
iter_t mdit = definition.begin();
iter_t mdend = definition.end();
for (; mdit != mdend; ++mdit) {
if ((IS_EXTCATEGORY((*mdit), OptParameterTokenType)) ||  
("__VA_OPT__" == (*mdit).get_value())) {             
iter_t va_opt_it = mdit;
if ((++mdit == mdend) ||                             
(T_LEFTPAREN != token_id(*mdit))) {              
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
bad_define_statement_va_opt_parens,
name.get_value().c_str(), main_pos,
name.get_value().c_str());
return false;
}
iter_t va_opt_end = va_opt_it;
if (!impl::find_va_opt_args(va_opt_end, mdend)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
improperly_terminated_macro, "missing ')' in __VA_OPT__",
main_pos);
return false;
}
++va_opt_it; ++va_opt_it;
for (;va_opt_it != va_opt_end; ++va_opt_it) {
if ((IS_EXTCATEGORY((*va_opt_it), OptParameterTokenType)) ||
("__VA_OPT__" == (*va_opt_it).get_value())) {
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
bad_define_statement_va_opt_recurse,
name.get_value().c_str(), (*va_opt_it).get_position(),
name.get_value().c_str());
}
}
}
}
}
#endif

std::pair<typename defined_macros_type::iterator, bool> p =
current_scope->insert(
typename defined_macros_type::value_type(
name.get_value(),
macro_ref_type(new macro_definition_type(name,
has_parameters, is_predefined, ++macro_uid)
)
)
);

if (!p.second) {
BOOST_WAVE_THROW_NAME_CTX(ctx, macro_handling_exception,
macro_insertion_error, name.get_value().c_str(), main_pos,
name.get_value().c_str());
return false;
}

std::swap((*p.first).second->macroparameters, parameters);
std::swap((*p.first).second->macrodefinition, definition);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().defined_macro(name, has_parameters,
(*p.first).second->macroparameters,
(*p.first).second->macrodefinition, is_predefined);
#else
ctx.get_hooks().defined_macro(ctx.derived(), name, has_parameters,
(*p.first).second->macroparameters,
(*p.first).second->macrodefinition, is_predefined);
#endif
return true;
}

template <typename ContextT>
inline bool
macromap<ContextT>::is_defined(typename token_type::string_type const &name,
typename defined_macros_type::iterator &it,
defined_macros_type *scope) const
{
if (0 == scope) scope = current_macros;

if ((it = scope->find(name)) != scope->end())
return true;        

if (name.size() < 8 || '_' != name[0] || '_' != name[1])
return false;       

if (name == "__LINE__" || name == "__FILE__" ||
name == "__INCLUDE_LEVEL__")
return true;

#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
return (boost::wave::need_has_include(ctx.get_language()) &&
(name == "__has_include"));
#else
return false;
#endif
}

template <typename ContextT>
template <typename IteratorT>
inline bool
macromap<ContextT>::is_defined(IteratorT const &begin,
IteratorT const &end) const
{
token_id id = token_id(*begin);

if (T_IDENTIFIER != id &&
!IS_CATEGORY(id, KeywordTokenType) &&
!IS_EXTCATEGORY(id, OperatorTokenType|AltExtTokenType) &&
!IS_CATEGORY(id, BoolLiteralTokenType))
{
std::string msg(impl::get_full_name(begin, end));
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, invalid_macroname,
msg.c_str(), main_pos);
return false;
}

IteratorT it = begin;
string_type name((*it).get_value());
typename defined_macros_type::iterator cit;

if (++it != end) {
std::string msg(impl::get_full_name(begin, end));
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, invalid_macroname,
msg.c_str(), main_pos);
return false;
}
return is_defined(name, cit, 0);
}

template <typename ContextT>
inline bool
macromap<ContextT>::is_defined(string_type const &str) const
{
typename defined_macros_type::iterator cit;
return is_defined(str, cit, 0);
}

#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
template <typename ContextT>
template <typename IteratorT>
inline bool
macromap<ContextT>::has_include(
IteratorT const &begin, IteratorT const &end,
bool is_quoted_filename, bool is_system) const
{
typename ContextT::token_sequence_type filetoks;

if (is_quoted_filename) {
filetoks = typename ContextT::token_sequence_type(begin, end);
} else {
IteratorT first = begin;
IteratorT last = end;
ctx.expand_whole_tokensequence(first, last, filetoks);
}

using namespace boost::wave::util::impl;
std::string fn(trim_whitespace(as_string(filetoks)).c_str());

if (!((fn.size() >= 3) &&
(((fn[0] == '"') && (*fn.rbegin() == '"')) ||
((fn[0] == '<') && (*fn.rbegin() == '>')))))
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_has_include_expression,
fn.c_str(), ctx.get_main_pos());

fn = fn.substr(1, fn.size() - 2);

std::string dir_path;
std::string native_path;
return ctx.get_hooks().locate_include_file(
ctx, fn, is_system, 0, dir_path, native_path);

}
#endif

template <typename ContextT>
inline bool
macromap<ContextT>::get_macro(string_type const &name, bool &has_parameters,
bool &is_predefined, position_type &pos,
parameter_container_type &parameters,
definition_container_type &definition,
defined_macros_type *scope) const
{
typename defined_macros_type::iterator it;
if (!is_defined(name, it, scope))
return false;

macro_definition_type& macro_def = *(*it).second.get();

has_parameters = macro_def.is_functionlike;
is_predefined = macro_def.is_predefined;
pos = macro_def.macroname.get_position();
parameters = macro_def.macroparameters;
definition = macro_def.macrodefinition;
return true;
}

template <typename ContextT>
inline bool
macromap<ContextT>::remove_macro(string_type const &name,
position_type const& pos, bool even_predefined)
{
typename defined_macros_type::iterator it = current_macros->find(name);

if (it != current_macros->end()) {
if ((*it).second->is_predefined) {
if (!even_predefined || impl::is_special_macroname(ctx, name)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
bad_undefine_statement, name.c_str(), main_pos);
return false;
}
}
current_macros->erase(it);

token_type tok(T_IDENTIFIER, name, pos);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().undefined_macro(tok);
#else
ctx.get_hooks().undefined_macro(ctx.derived(), tok);
#endif
return true;
}
else if (impl::is_special_macroname(ctx, name)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_undefine_statement,
name.c_str(), pos);
}
return false;       
}

template <typename ContextT>
template <typename IteratorT, typename ContainerT>
inline typename ContextT::token_type const &
macromap<ContextT>::expand_tokensequence(IteratorT &first,
IteratorT const &last, ContainerT &pending, ContainerT &expanded,
bool& seen_newline, bool expand_operator_defined,
bool expand_operator_has_include)
{
typedef impl::gen_unput_queue_iterator<IteratorT, token_type, ContainerT>
gen_type;
typedef typename gen_type::return_type iterator_type;

iterator_type first_it = gen_type::generate(expanded, first);
iterator_type last_it = gen_type::generate(last);

on_exit::assign<IteratorT, iterator_type> on_exit(first, first_it);

return expand_tokensequence_worker(pending, first_it, last_it,
seen_newline, expand_operator_defined, expand_operator_has_include,
boost::none);
}

template <typename ContextT>
template <typename IteratorT, typename ContainerT>
inline typename ContextT::token_type const &
macromap<ContextT>::expand_tokensequence_worker(
ContainerT &pending,
unput_queue_iterator<IteratorT, token_type, ContainerT> &first,
unput_queue_iterator<IteratorT, token_type, ContainerT> const &last,
bool& seen_newline, bool expand_operator_defined,
bool expand_operator_has_include,
boost::optional<position_type> expanding_pos)
{
if (!pending.empty()) {
on_exit::pop_front<definition_container_type> pop_front_token(pending);

return act_token = pending.front();
}

using namespace boost::wave;

if (first != last) {
token_id id = token_id(*first);

if (T_PLACEHOLDER == id) {
token_type placeholder = *first;

++first;
if (first == last)
return act_token = placeholder;
id = token_id(*first);
}

if (T_IDENTIFIER == id || IS_CATEGORY(id, KeywordTokenType) ||
IS_EXTCATEGORY(id, OperatorTokenType|AltExtTokenType) ||
IS_CATEGORY(id, BoolLiteralTokenType))
{
if (expand_operator_defined && (*first).get_value() == "defined") {
return resolve_defined(first, last, pending);
}
#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
else if (boost::wave::need_has_include(ctx.get_language()) &&
expand_operator_has_include &&
(*first).get_value() == "__has_include") {
return resolve_has_include(first, last, pending);
}
#endif
else if (boost::wave::need_variadics(ctx.get_language()) &&
(*first).get_value() == "_Pragma")
{
token_type curr_token = *first;

if (!resolve_operator_pragma(first, last, pending, seen_newline) ||
pending.size() > 0)
{
on_exit::pop_front<definition_container_type> pop_token(pending);

return act_token = pending.front();
}

return act_token = token_type(T_PLACEHOLDER, "_",
curr_token.get_position());
}

token_type name_token(*first);
typename defined_macros_type::iterator it;

if (is_defined(name_token.get_value(), it)) {
if (expand_macro(pending, name_token, it, first, last,
seen_newline, expand_operator_defined,
expand_operator_has_include,
expanding_pos))
{
if (first != last) {
typename ContainerT::reverse_iterator rit = pending.rbegin();

first.get_unput_queue().splice(
first.get_unput_queue().begin(), pending,
(++rit).base(), pending.end());
}

}
else if (!pending.empty()) {
on_exit::pop_front<definition_container_type> pop_queue(pending);

return act_token = pending.front();
}
else {
return act_token = token_type();
}

if (!expanding_pos)
expanding_pos = name_token.get_expand_position();

typename ContextT::token_type const & result =
expand_tokensequence_worker(
pending, first, last,
seen_newline, expand_operator_defined,
expand_operator_has_include,
expanding_pos);

return result;
}
else {
act_token = name_token;
++first;
return act_token;
}
}
else if (expand_operator_defined && IS_CATEGORY(*first, BoolLiteralTokenType)) {

return act_token = token_type(T_INTLIT, T_TRUE != id ? "0" : "1",
(*first++).get_position());
}
else {
act_token = *first;
++first;
return act_token;
}
}
return act_token = token_type();     
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename ContextT>
template <typename IteratorT, typename ContainerT, typename SizeT>
inline typename std::vector<ContainerT>::size_type
macromap<ContextT>::collect_arguments (token_type const curr_token,
std::vector<ContainerT> &arguments, IteratorT &next,
IteratorT const &end, SizeT const &parameter_count, bool& seen_newline)
#else
template <typename ContextT>
template <typename IteratorT, typename ContainerT, typename SizeT>
inline typename std::vector<ContainerT>::size_type
macromap<ContextT>::collect_arguments (token_type const curr_token,
std::vector<ContainerT> &arguments, IteratorT &next, IteratorT &endparen,
IteratorT const &end, SizeT const &parameter_count, bool& seen_newline)
#endif
{
using namespace boost::wave;

arguments.push_back(ContainerT());

typename std::vector<ContainerT>::size_type count_arguments = 0;
int nested_parenthesis_level = 1;
ContainerT* argument = &arguments[0];
bool was_whitespace = false;
token_type startof_argument_list = *next;

while (++next != end && nested_parenthesis_level) {
token_id id = token_id(*next);

if (0 == parameter_count &&
!IS_CATEGORY((*next), WhiteSpaceTokenType) && id != T_NEWLINE &&
id != T_RIGHTPAREN && id != T_LEFTPAREN)
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
too_many_macroarguments, curr_token.get_value().c_str(),
main_pos);
return 0;
}

switch (id) {
case T_LEFTPAREN:
++nested_parenthesis_level;
argument->push_back(*next);
was_whitespace = false;
break;

case T_RIGHTPAREN:
{
if (--nested_parenthesis_level >= 1)
argument->push_back(*next);
else {
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS == 0
endparen = next;
#endif
if (parameter_count > 0) {
if (argument->empty() ||
impl::is_whitespace_only(*argument))
{
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_variadics(ctx.get_language())) {
argument->push_back(token_type(T_PLACEMARKER, "\xA7",
(*next).get_position()));
++count_arguments;
}
#endif
}
else {
++count_arguments;
}
}
}
was_whitespace = false;
}
break;

case T_COMMA:
if (1 == nested_parenthesis_level) {
if (argument->empty() ||
impl::is_whitespace_only(*argument))
{
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_variadics(ctx.get_language())) {
argument->push_back(token_type(T_PLACEMARKER, "\xA7",
(*next).get_position()));
++count_arguments;
}
#endif
}
else {
++count_arguments;
}
arguments.push_back(ContainerT()); 
argument = &arguments[arguments.size()-1];
}
else {
argument->push_back(*next);
}
was_whitespace = false;
break;

case T_NEWLINE:
seen_newline = true;

case T_SPACE:
case T_SPACE2:
case T_CCOMMENT:
if (!was_whitespace)
argument->push_back(token_type(T_SPACE, " ", (*next).get_position()));
was_whitespace = true;
break;      

case T_PLACEHOLDER:
break;      

default:
argument->push_back(*next);
was_whitespace = false;
break;
}
}

if (nested_parenthesis_level >= 1) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
improperly_terminated_macro, "missing ')'", main_pos);
return 0;
}

if (0 == parameter_count && 0 == count_arguments) {
BOOST_ASSERT(1 == arguments.size());
arguments.clear();
}
return count_arguments;
}

template <typename ContextT>
template <typename IteratorT, typename ContainerT>
inline void
macromap<ContextT>::expand_whole_tokensequence(ContainerT &expanded,
IteratorT &first, IteratorT const &last,
bool expand_operator_defined,
bool expand_operator_has_include)
{
typedef impl::gen_unput_queue_iterator<IteratorT, token_type, ContainerT>
gen_type;
typedef typename gen_type::return_type iterator_type;

ContainerT empty;
iterator_type first_it = gen_type::generate(empty, first);
iterator_type last_it = gen_type::generate(last);

on_exit::assign<IteratorT, iterator_type> on_exit(first, first_it);
ContainerT pending_queue;
bool seen_newline;

while (!pending_queue.empty() || first_it != last_it) {
expanded.push_back(
expand_tokensequence_worker(
pending_queue, first_it,
last_it, seen_newline, expand_operator_defined,
expand_operator_has_include,
boost::none)
);
}

BOOST_ASSERT(pending_queue.empty());
}

template <typename ContextT>
template <typename ContainerT>
inline void
macromap<ContextT>::expand_argument (
typename std::vector<ContainerT>::size_type arg,
std::vector<ContainerT> &arguments, std::vector<ContainerT> &expanded_args,
bool expand_operator_defined, bool expand_operator_has_include,
std::vector<bool> &has_expanded_args)
{
if (!has_expanded_args[arg]) {
typedef typename std::vector<ContainerT>::value_type::iterator
argument_iterator_type;

argument_iterator_type begin_it = arguments[arg].begin();
argument_iterator_type end_it = arguments[arg].end();

expand_whole_tokensequence(
expanded_args[arg], begin_it, end_it,
expand_operator_defined, expand_operator_has_include);
impl::remove_placeholders(expanded_args[arg]);
has_expanded_args[arg] = true;
}
}

template <typename ContextT>
template <typename ContainerT>
inline void
macromap<ContextT>::expand_replacement_list(
typename macro_definition_type::const_definition_iterator_t cit,
typename macro_definition_type::const_definition_iterator_t cend,
std::vector<ContainerT> &arguments, bool expand_operator_defined,
bool expand_operator_has_include,
ContainerT &expanded)
{
using namespace boost::wave;
typedef typename macro_definition_type::const_definition_iterator_t
macro_definition_iter_t;

std::vector<ContainerT> expanded_args(arguments.size());
std::vector<bool> has_expanded_args(arguments.size());
bool seen_concat = false;
bool adjacent_concat = false;
bool adjacent_stringize = false;

for (;cit != cend; ++cit)
{
bool use_replaced_arg = true;
token_id base_id = BASE_TOKEN(token_id(*cit));

if (T_POUND_POUND == base_id) {
adjacent_concat = true;
seen_concat = true;
}
else if (T_POUND == base_id) {
adjacent_stringize = true;
}
else {
if (adjacent_stringize || adjacent_concat ||
T_POUND_POUND == impl::next_token<macro_definition_iter_t>
::peek(cit, cend))
{
use_replaced_arg = false;
}
if (adjacent_concat)    
adjacent_concat = IS_CATEGORY(*cit, WhiteSpaceTokenType);
}

if (IS_CATEGORY((*cit), ParameterTokenType)) {
typename ContainerT::size_type i;
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
bool is_ellipsis = false;
#if BOOST_WAVE_SUPPORT_VA_OPT != 0
bool is_va_opt = false;
#endif

if (IS_EXTCATEGORY((*cit), ExtParameterTokenType)) {
BOOST_ASSERT(boost::wave::need_variadics(ctx.get_language()));
i = token_id(*cit) - T_EXTPARAMETERBASE;
is_ellipsis = true;
}
else
#if BOOST_WAVE_SUPPORT_VA_OPT != 0

if (IS_EXTCATEGORY((*cit), OptParameterTokenType)) {
BOOST_ASSERT(boost::wave::need_va_opt(ctx.get_language()));
i = token_id(*cit) - T_OPTPARAMETERBASE;
is_va_opt = true;
}
else
#endif
#endif
{
i = token_id(*cit) - T_PARAMETERBASE;
}

BOOST_ASSERT(i <= arguments.size());
if (use_replaced_arg) {

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (is_ellipsis) {
position_type const& pos = (*cit).get_position();

BOOST_ASSERT(boost::wave::need_variadics(ctx.get_language()));

for (typename vector<ContainerT>::size_type arg = i;
arg < expanded_args.size(); ++arg)
{
expand_argument(
arg, arguments, expanded_args,
expand_operator_defined, expand_operator_has_include,
has_expanded_args);
}
impl::replace_ellipsis(expanded_args, i, expanded, pos);
}
else

#if BOOST_WAVE_SUPPORT_VA_OPT != 0
if (is_va_opt) {
position_type const &pos = (*cit).get_position();

BOOST_ASSERT(boost::wave::need_va_opt(ctx.get_language()));

for (typename vector<ContainerT>::size_type arg = i;
arg < expanded_args.size(); ++arg)
{
expand_argument(
arg, arguments, expanded_args,
expand_operator_defined, expand_operator_has_include,
has_expanded_args);
}

typename macro_definition_type::const_definition_iterator_t cstart = cit;
if (!impl::find_va_opt_args(cit, cend)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
improperly_terminated_macro, "missing '(' or ')' in __VA_OPT__",
pos);
}

typename macro_definition_type::const_definition_iterator_t arg_start = cstart;
++arg_start;  
++arg_start;  

token_type macroname(T_IDENTIFIER, "__VA_OPT__", position_type("<built-in>"));
parameter_container_type macroparameters;
macroparameters.push_back(token_type(T_ELLIPSIS, "...", position_type("<built-in>")));
definition_container_type macrodefinition;

bool suppress_expand = false;
typename std::vector<ContainerT> va_opt_args(1, ContainerT(arg_start, cit));
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().expanding_function_like_macro(
macroname, macroparameters, macrodefinitions,
*cstart, va_opt_args);
#else
suppress_expand = ctx.get_hooks().expanding_function_like_macro(
ctx.derived(),
macroname, macroparameters, macrodefinition,
*cstart, va_opt_args,
cstart, cit);
#endif

if (suppress_expand) {
std::copy(cstart, cit, std::back_inserter(expanded));
expanded.push_back(*cit);  
} else {
ContainerT va_expanded;
if ((i == arguments.size()) ||                 
impl::is_whitespace_only(arguments[i])) {  
va_expanded.push_back(
typename ContainerT::value_type(T_PLACEMARKER, "\xA7", pos));
} else if (!impl::is_blank_only(arguments[i])) {
expand_replacement_list(arg_start, cit, arguments,
expand_operator_defined,
expand_operator_has_include,
va_expanded);
}
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().expanded_macro(va_expanded);
#else
ctx.get_hooks().expanded_macro(ctx.derived(), va_expanded);
#endif

expanded.splice(expanded.end(), va_expanded);
}
}
else

#endif
#endif
{
BOOST_ASSERT(i < arguments.size());
expand_argument(
i, arguments, expanded_args,
expand_operator_defined, expand_operator_has_include,
has_expanded_args);

BOOST_ASSERT(i < expanded_args.size());
ContainerT const& arg = expanded_args[i];

std::copy(arg.begin(), arg.end(),
std::inserter(expanded, expanded.end()));
}
}
else if (adjacent_stringize &&
!IS_CATEGORY(*cit, WhiteSpaceTokenType))
{
BOOST_ASSERT(!arguments[i].empty());

position_type pos((*arguments[i].begin()).get_position());

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (is_ellipsis && boost::wave::need_variadics(ctx.get_language())) {
impl::trim_sequence_left(arguments[i]);
impl::trim_sequence_right(arguments.back());
expanded.push_back(token_type(T_STRINGLIT,
impl::as_stringlit(arguments, i, pos), pos));
}
else
#endif
{
impl::trim_sequence(arguments[i]);
expanded.push_back(token_type(T_STRINGLIT,
impl::as_stringlit(arguments[i], pos), pos));
}
adjacent_stringize = false;
}
else {
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (is_ellipsis) {
position_type const& pos = (*cit).get_position();
#if BOOST_WAVE_SUPPORT_CPP2A != 0
if (i < arguments.size())
#endif
{

impl::trim_sequence_left(arguments[i]);
impl::trim_sequence_right(arguments.back());
BOOST_ASSERT(boost::wave::need_variadics(ctx.get_language()));
impl::replace_ellipsis(arguments, i, expanded, pos);
}
#if BOOST_WAVE_SUPPORT_CPP2A != 0
else if (boost::wave::need_cpp2a(ctx.get_language())) {
BOOST_ASSERT(i == arguments.size());
expanded.push_back(
typename ContainerT::value_type(T_PLACEMARKER, "\xA7", pos));
}
#endif
}
else
#endif
{
ContainerT& arg = arguments[i];

impl::trim_sequence(arg);
std::copy(arg.begin(), arg.end(),
std::inserter(expanded, expanded.end()));
}
}
}
else if (!adjacent_stringize || T_POUND != base_id) {
expanded.push_back(*cit);
}
}

if (adjacent_stringize) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, ill_formed_operator,
"stringize ('#')", main_pos);
return;
}

if (seen_concat)
concat_tokensequence(expanded);
}

template <typename ContextT>
template <typename IteratorT, typename ContainerT>
inline void
macromap<ContextT>::rescan_replacement_list(token_type const &curr_token,
macro_definition_type &macro_def, ContainerT &replacement_list,
ContainerT &expanded,
bool expand_operator_defined,
bool expand_operator_has_include,
IteratorT &nfirst, IteratorT const &nlast)
{
if (!replacement_list.empty()) {
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_variadics(ctx.get_language())) {
typename ContainerT::iterator end = replacement_list.end();
typename ContainerT::iterator it = replacement_list.begin();

while (it != end) {
using namespace boost::wave;
if (T_PLACEMARKER == token_id(*it)) {
typename ContainerT::iterator placemarker = it;

++it;
replacement_list.erase(placemarker);
}
else {
++it;
}
}
}
#endif

on_exit::reset<bool> on_exit(macro_def.is_available_for_replacement, false);
typename ContainerT::iterator begin_it = replacement_list.begin();
typename ContainerT::iterator end_it = replacement_list.end();

expand_whole_tokensequence(
expanded, begin_it, end_it,
expand_operator_defined, expand_operator_has_include);

impl::trim_replacement_list(expanded);
}

if (expanded.empty()) {
expanded.push_back(token_type(T_PLACEHOLDER, "_", curr_token.get_position()));
}
}

template <typename ContextT>
template <typename IteratorT, typename ContainerT>
inline bool
macromap<ContextT>::expand_macro(ContainerT &expanded,
token_type const &curr_token, typename defined_macros_type::iterator it,
IteratorT &first, IteratorT const &last,
bool& seen_newline, bool expand_operator_defined,
bool expand_operator_has_include,
boost::optional<position_type> expanding_pos,
defined_macros_type *scope, ContainerT *queue_symbol)
{
using namespace boost::wave;

if (0 == scope) scope = current_macros;

BOOST_ASSERT(T_IDENTIFIER == token_id(curr_token) ||
IS_CATEGORY(token_id(curr_token), KeywordTokenType) ||
IS_EXTCATEGORY(token_id(curr_token), OperatorTokenType|AltExtTokenType) ||
IS_CATEGORY(token_id(curr_token), BoolLiteralTokenType));

if (it == scope->end()) {
++first;    

if (expand_predefined_macro(curr_token, expanded))
return false;

if (0 != queue_symbol) {
expanded.splice(expanded.end(), *queue_symbol);
}
else {
expanded.push_back(curr_token);
}
return false;
}

macro_definition_type& macro_def = *(*it).second.get();

macro_def.replace_parameters(ctx);

if (!macro_def.is_available_for_replacement) {
if (0 != queue_symbol) {
queue_symbol->push_back(token_type(T_NONREPLACABLE_IDENTIFIER,
curr_token.get_value(), curr_token.get_position()));
expanded.splice(expanded.end(), *queue_symbol);
}
else {
expanded.push_back(token_type(T_NONREPLACABLE_IDENTIFIER,
curr_token.get_value(), curr_token.get_position()));
}
++first;
return false;
}

ContainerT replacement_list;

if (T_LEFTPAREN == impl::next_token<IteratorT>::peek(first, last)) {
impl::skip_to_token(ctx, first, last, T_LEFTPAREN, seen_newline);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS == 0
IteratorT seqstart = first;
IteratorT seqend = first;
#endif

if (macro_def.is_functionlike) {

std::vector<ContainerT> arguments;
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
typename std::vector<ContainerT>::size_type count_args =
collect_arguments(curr_token, arguments, first, last,
macro_def.macroparameters.size(), seen_newline);
#else
typename std::vector<ContainerT>::size_type count_args =
collect_arguments(curr_token, arguments, first, seqend, last,
macro_def.macroparameters.size(), seen_newline);
#endif

std::size_t parm_count_required = macro_def.macroparameters.size();
#if BOOST_WAVE_SUPPORT_CPP2A
if (boost::wave::need_cpp2a(ctx.get_language())) {
if ((parm_count_required > 0) &&
(T_ELLIPSIS == token_id(macro_def.macroparameters.back()))) {
--parm_count_required;
}
}
#endif

if (count_args < parm_count_required ||
arguments.size() < parm_count_required)
{
if (count_args != arguments.size()) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
empty_macroarguments, curr_token.get_value().c_str(),
main_pos);
}
else {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
too_few_macroarguments, curr_token.get_value().c_str(),
main_pos);
}
return false;
}

if (count_args > macro_def.macroparameters.size() ||
arguments.size() > macro_def.macroparameters.size())
{
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (!macro_def.has_ellipsis)
#endif
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
too_many_macroarguments,
curr_token.get_value().c_str(), main_pos);
return false;
}
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().expanding_function_like_macro(
macro_def.macroname, macro_def.macroparameters,
macro_def.macrodefinition, curr_token, arguments);
#else
if (ctx.get_hooks().expanding_function_like_macro(ctx.derived(),
macro_def.macroname, macro_def.macroparameters,
macro_def.macrodefinition, curr_token, arguments,
seqstart, seqend))
{
expanded.push_back(curr_token);
expanded.push_back(*seqstart);
first = ++seqstart;
return false;           
}
#endif

expand_replacement_list(macro_def.macrodefinition.begin(),
macro_def.macrodefinition.end(),
arguments, expand_operator_defined,
expand_operator_has_include,
replacement_list);

if (!expanding_pos)
expanding_pos = curr_token.get_expand_position();
set_expand_positions(replacement_list, *expanding_pos);
}
else {
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().expanding_object_like_macro(
macro_def.macroname, macro_def.macrodefinition, curr_token);
#else
if (ctx.get_hooks().expanding_object_like_macro(ctx.derived(),
macro_def.macroname, macro_def.macrodefinition, curr_token))
{
expanded.push_back(curr_token);
return false;           
}
#endif

bool found = false;
impl::find_concat_operator concat_tag(found);

std::remove_copy_if(macro_def.macrodefinition.begin(),
macro_def.macrodefinition.end(),
std::inserter(replacement_list, replacement_list.end()),
concat_tag);

if (found && !concat_tokensequence(replacement_list))
return false;
}
}
else {
if ((*it).second->is_functionlike) {
if (0 != queue_symbol) {
queue_symbol->push_back(curr_token);
expanded.splice(expanded.end(), *queue_symbol);
}
else {
expanded.push_back(curr_token);
}
++first;                
return false;           
}
else {
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().expanding_object_like_macro(
macro_def.macroname, macro_def.macrodefinition, curr_token);
#else
if (ctx.get_hooks().expanding_object_like_macro(ctx.derived(),
macro_def.macroname, macro_def.macrodefinition, curr_token))
{
expanded.push_back(curr_token);
++first;                
return false;           
}
#endif

bool found = false;
impl::find_concat_operator concat_tag(found);

std::remove_copy_if(macro_def.macrodefinition.begin(),
macro_def.macrodefinition.end(),
std::inserter(replacement_list, replacement_list.end()),
concat_tag);

if (found && !concat_tokensequence(replacement_list))
return false;

++first;                
}
}

ContainerT expanded_list;

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().expanded_macro(replacement_list);
#else
ctx.get_hooks().expanded_macro(ctx.derived(), replacement_list);
#endif

rescan_replacement_list(
curr_token, macro_def, replacement_list,
expanded_list, expand_operator_defined,
expand_operator_has_include, first, last);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().rescanned_macro(expanded_list);
#else
ctx.get_hooks().rescanned_macro(ctx.derived(), expanded_list);
#endif

if (!expanding_pos)
expanding_pos = curr_token.get_expand_position();

set_expand_positions(expanded_list, *expanding_pos);

expanded.splice(expanded.end(), expanded_list);
return true;        
}

template <typename ContextT>
template <typename ContainerT>
inline bool
macromap<ContextT>::expand_predefined_macro(token_type const &curr_token,
ContainerT &expanded)
{
using namespace boost::wave;

string_type const& value = curr_token.get_value();

if ((value != "__LINE__") && (value != "__FILE__") && (value != "__INCLUDE_LEVEL__"))
return false;

token_type deftoken(T_IDENTIFIER, value, position_type("<built-in>"));

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().expanding_object_like_macro(
deftoken, Container(), curr_token);
#else
if (ctx.get_hooks().expanding_object_like_macro(ctx.derived(),
deftoken, ContainerT(), curr_token))
{
expanded.push_back(curr_token);
return false;           
}
#endif

token_type replacement;

if (value == "__LINE__") {
std::string buffer = lexical_cast<std::string>(curr_token.get_expand_position().get_line());

replacement = token_type(T_INTLIT, buffer.c_str(), curr_token.get_position());
}
else if (value == "__FILE__") {
namespace fs = boost::filesystem;

std::string file("\"");
fs::path filename(
wave::util::create_path(curr_token.get_expand_position().get_file().c_str()));

using boost::wave::util::impl::escape_lit;
file += escape_lit(wave::util::native_file_string(filename)) + "\"";
replacement = token_type(T_STRINGLIT, file.c_str(),
curr_token.get_position());
}
else if (value == "__INCLUDE_LEVEL__") {
char buffer[22]; 

using namespace std;    
sprintf(buffer, "%d", (int)ctx.get_iteration_depth());
replacement = token_type(T_INTLIT, buffer, curr_token.get_position());
}

ContainerT replacement_list;
replacement_list.push_back(replacement);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().expanded_macro(replacement_list);
#else
ctx.get_hooks().expanded_macro(ctx.derived(), replacement_list);
#endif

expanded.push_back(replacement);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().rescanned_macro(expanded);
#else
ctx.get_hooks().rescanned_macro(ctx.derived(), expanded);
#endif

return true;

}

template <typename ContextT>
template <typename IteratorT, typename ContainerT>
inline typename ContextT::token_type const &
macromap<ContextT>::resolve_defined(IteratorT &first,
IteratorT const &last, ContainerT &pending)
{
using namespace boost::wave;
using namespace boost::wave::grammars;

ContainerT result;
IteratorT start = first;
boost::spirit::classic::parse_info<IteratorT> hit =
defined_grammar_gen<typename ContextT::lexer_type>::
parse_operator_defined(start, last, result);

if (!hit.hit) {
string_type msg ("defined(): ");
msg = msg + util::impl::as_string<string_type>(first, last);
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, ill_formed_expression,
msg.c_str(), main_pos);

pending.push_back(token_type(T_INTLIT, "0", main_pos));
}
else {
impl::assign_iterator<IteratorT>::do_(first, hit.stop);

pending.push_back(token_type(T_INTLIT,
is_defined(result.begin(), result.end()) ? "1" : "0",
main_pos));
}

on_exit::pop_front<definition_container_type> pop_front_token(pending);

return act_token = pending.front();
}

#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
template <typename ContextT>
template <typename IteratorT, typename ContainerT>
inline typename ContextT::token_type const &
macromap<ContextT>::resolve_has_include(IteratorT &first,
IteratorT const &last, ContainerT &pending)
{
using namespace boost::wave;
using namespace boost::wave::grammars;

ContainerT result;
bool is_quoted_filename;
bool is_system;

IteratorT end_find_it = first;
++end_find_it;
IteratorT rparen_it = first;
while (end_find_it != last) {
++end_find_it;
++rparen_it;
}

boost::spirit::classic::parse_info<IteratorT> hit(first);
if ((rparen_it != first) && (T_RIGHTPAREN == *rparen_it)) {
IteratorT start = first;
hit = has_include_grammar_gen<typename ContextT::lexer_type>::
parse_operator_has_include(start, rparen_it, result, is_quoted_filename, is_system);
}

if (!hit.hit) {
string_type msg ("__has_include(): ");
msg = msg + util::impl::as_string<string_type>(first, last);
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, ill_formed_expression,
msg.c_str(), main_pos);

pending.push_back(token_type(T_INTLIT, "0", main_pos));
}
else {
impl::assign_iterator<IteratorT>::do_(first, last);

pending.push_back(
token_type(T_INTLIT,
has_include(result.begin(), result.end(),
is_quoted_filename, is_system) ? "1" : "0",
main_pos));
}

on_exit::pop_front<definition_container_type> pop_front_token(pending);

return act_token = pending.front();
}
#endif

template <typename ContextT>
template <typename IteratorT, typename ContainerT>
inline bool
macromap<ContextT>::resolve_operator_pragma(IteratorT &first,
IteratorT const &last, ContainerT &pending, bool& seen_newline)
{
token_type pragma_token = *first;

if (!impl::skip_to_token(ctx, first, last, T_LEFTPAREN, seen_newline)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, ill_formed_expression,
"operator _Pragma()", pragma_token.get_position());
return false;
}

std::vector<ContainerT> arguments;
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
typename std::vector<ContainerT>::size_type count_args =
collect_arguments (pragma_token, arguments, first, last, 1, seen_newline);
#else
IteratorT endparen = first;
typename std::vector<ContainerT>::size_type count_args =
collect_arguments (pragma_token, arguments, first, endparen, last, 1,
seen_newline);
#endif

if (pragma_token.get_position().get_file().empty())
pragma_token.set_position(act_token.get_position());

if (count_args < 1 || arguments.size() < 1) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, too_few_macroarguments,
pragma_token.get_value().c_str(), pragma_token.get_position());
return false;
}
if (count_args > 1 || arguments.size() > 1) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, too_many_macroarguments,
pragma_token.get_value().c_str(), pragma_token.get_position());
return false;
}

typedef typename std::vector<ContainerT>::value_type::iterator
argument_iterator_type;

ContainerT expanded;
argument_iterator_type begin_it = arguments[0].begin();
argument_iterator_type end_it = arguments[0].end();
expand_whole_tokensequence(expanded, begin_it, end_it, false, false);

typedef typename token_type::string_type string_type;

string_type pragma_cmd;
typename ContainerT::const_iterator end_exp = expanded.end();
for (typename ContainerT::const_iterator it_exp = expanded.begin();
it_exp != end_exp; ++it_exp)
{
if (T_EOF == token_id(*it_exp))
break;
if (IS_CATEGORY(*it_exp, WhiteSpaceTokenType))
continue;

if (T_STRINGLIT != token_id(*it_exp)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
ill_formed_pragma_option, "_Pragma",
pragma_token.get_position());
return false;
}
if (pragma_cmd.size() > 0) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
ill_formed_pragma_option, "_Pragma",
pragma_token.get_position());
return false;
}

string_type token_str = (*it_exp).get_value();
pragma_cmd += token_str.substr(1, token_str.size() - 2);
}
string_type pragma_cmd_unesc = impl::unescape_lit(pragma_cmd);

typedef typename ContextT::lexer_type lexer_type;

ContainerT pragma;
std::string pragma_cmd_str(pragma_cmd_unesc.c_str());
lexer_type it = lexer_type(pragma_cmd_str.begin(), pragma_cmd_str.end(),
pragma_token.get_position(), ctx.get_language());
lexer_type end = lexer_type();
for (; it != end; ++it)
pragma.push_back(*it);

if (interpret_pragma(ctx, pragma_token, pragma.begin(), pragma.end(),
pending))
{
return true;    
}

pending.push_front(token_type(T_SPACE, " ", pragma_token.get_position()));
pending.push_front(token_type(T_RIGHTPAREN, ")", pragma_token.get_position()));
pending.push_front(token_type(T_STRINGLIT, string_type("\"") + pragma_cmd + "\"",
pragma_token.get_position()));
pending.push_front(token_type(T_LEFTPAREN, "(", pragma_token.get_position()));
pending.push_front(pragma_token);
return false;
}

template <typename ContextT>
template <typename ContainerT>
inline bool
macromap<ContextT>::is_valid_concat(string_type new_value,
position_type const &pos, ContainerT &rescanned)
{
typedef typename ContextT::lexer_type lexer_type;

std::string value_to_test(new_value.c_str());

boost::wave::language_support lang =
boost::wave::enable_prefer_pp_numbers(ctx.get_language());
lang = boost::wave::enable_single_line(lang);

lexer_type it = lexer_type(value_to_test.begin(), value_to_test.end(), pos,
lang);
lexer_type end = lexer_type();
for (; it != end && T_EOF != token_id(*it); ++it)
{
if (!is_pp_token(*it))
return false;
rescanned.push_back(*it);
}

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_variadics(ctx.get_language()))
return true;       
#endif

return 1 == rescanned.size();
}

template <typename ContextT>
template <typename ContainerT>
void macromap<ContextT>::set_expand_positions(ContainerT &tokens, position_type pos)
{
typename ContainerT::iterator ex_end = tokens.end();
for (typename ContainerT::iterator it = tokens.begin();
it != ex_end; ++it) {
if (token_id(*it) == T_IDENTIFIER)
it->set_expand_position(pos);
}
}

template <typename Context>
inline void report_invalid_concatenation(Context& ctx,
typename Context::token_type const& prev,
typename Context::token_type const& next,
typename Context::position_type const& main_pos)
{
typename Context::string_type error_string("\"");

error_string += prev.get_value();
error_string += "\" and \"";
error_string += next.get_value();
error_string += "\"";
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, invalid_concat,
error_string.c_str(), main_pos);
}

template <typename ContextT>
template <typename ContainerT>
inline bool
macromap<ContextT>::concat_tokensequence(ContainerT &expanded)
{
using namespace boost::wave;
typedef typename ContainerT::iterator iterator_type;

iterator_type end = expanded.end();
iterator_type prev = end;
for (iterator_type it = expanded.begin(); it != end; )
{
if (T_POUND_POUND == BASE_TOKEN(token_id(*it))) {
iterator_type next = it;

++next;
if (prev == end || next == end) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
ill_formed_operator, "concat ('##')", main_pos);
return false;
}

while (IS_CATEGORY(*next, WhiteSpaceTokenType)) {
++next;
if (next == end) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
ill_formed_operator, "concat ('##')", main_pos);
return false;
}
}

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_variadics(ctx.get_language())) {
if (T_PLACEMARKER == token_id(*next)) {
iterator_type first_to_delete = prev;

expanded.erase(++first_to_delete, ++next);
it = next;
continue;
}
else if (T_PLACEMARKER == token_id(*prev)) {
iterator_type first_to_delete = prev;

*prev = *next;
expanded.erase(++first_to_delete, ++next);
it = next;
continue;
}
}
#endif 

string_type concat_result;
ContainerT rescanned;

concat_result = ((*prev).get_value() + (*next).get_value());

if (!is_valid_concat(concat_result, (*prev).get_position(),
rescanned) &&
!IS_CATEGORY(*prev, WhiteSpaceTokenType) &&
!IS_CATEGORY(*next, WhiteSpaceTokenType))
{
report_invalid_concatenation(ctx, *prev, *next, main_pos);
return false;
}

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_variadics(ctx.get_language())) {
expanded.erase(prev, ++next); 

if (expanded.empty())
end = next = expanded.end();

expanded.splice(next, rescanned);

prev = next;
if (next != expanded.end())
--prev;
}
else
#endif 
{
(*prev).set_value(concat_result);
if (T_NONREPLACABLE_IDENTIFIER == token_id(*prev))
(*prev).set_token_id(T_IDENTIFIER);

iterator_type first_to_delete = prev;

expanded.erase(++first_to_delete, ++next);
}
it = next;
continue;
}

if (!IS_CATEGORY(*it, WhiteSpaceTokenType))
prev = it;

++it;           
}
return true;
}

template <typename ContextT>
inline void
macromap<ContextT>::predefine_macro(defined_macros_type *scope,
string_type const &name, token_type const &t)
{
definition_container_type macrodefinition;
std::vector<token_type> param;

macrodefinition.push_back(t);
add_macro(token_type(T_IDENTIFIER, name, t.get_position()),
false, param, macrodefinition, true, scope);
}

template <typename ContextT>
inline void
macromap<ContextT>::init_predefined_macros(char const *fname,
defined_macros_type *scope, bool at_global_scope)
{
defined_macros_type* current_scope = scope ? scope : current_macros;

position_type pos("<built-in>");

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_c99(ctx.get_language())) {
for (int i = 0; 0 != predef.static_data_c99(i).name; ++i) {
predefined_macros::static_macros const& m = predef.static_data_c99(i);
predefine_macro(current_scope, m.name,
token_type(m.token_id, m.value, pos));
}
}
else
#endif
{
#if BOOST_WAVE_SUPPORT_CPP0X != 0
if (boost::wave::need_cpp0x(ctx.get_language())) {
for (int i = 0; 0 != predef.static_data_cpp0x(i).name; ++i) {
predefined_macros::static_macros const& m = predef.static_data_cpp0x(i);
predefine_macro(current_scope, m.name,
token_type(m.token_id, m.value, pos));
}
}
else
#endif
#if BOOST_WAVE_SUPPORT_CPP2A != 0
if (boost::wave::need_cpp2a(ctx.get_language())) {
for (int i = 0; 0 != predef.static_data_cpp2a(i).name; ++i) {
predefined_macros::static_macros const& m = predef.static_data_cpp2a(i);
predefine_macro(current_scope, m.name,
token_type(m.token_id, m.value, pos));
}
}
else
#endif
{
for (int i = 0; 0 != predef.static_data_cpp(i).name; ++i) {
predefined_macros::static_macros const& m = predef.static_data_cpp(i);
predefine_macro(current_scope, m.name,
token_type(m.token_id, m.value, pos));
}

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_variadics(ctx.get_language())) {
predefine_macro(current_scope, "__WAVE_HAS_VARIADICS__",
token_type(T_INTLIT, "1", pos));
}
#endif
}
}

namespace fs = boost::filesystem;
if (string_type(fname) != "<Unknown>") {
fs::path filename(create_path(fname));

using boost::wave::util::impl::escape_lit;
predefine_macro(current_scope, "__BASE_FILE__",
token_type(T_STRINGLIT, string_type("\"") +
escape_lit(native_file_string(filename)).c_str() + "\"", pos));
base_name = fname;
}
else if (!base_name.empty()) {
fs::path filename(create_path(base_name.c_str()));

using boost::wave::util::impl::escape_lit;
predefine_macro(current_scope, "__BASE_FILE__",
token_type(T_STRINGLIT, string_type("\"") +
escape_lit(native_file_string(filename)).c_str() + "\"", pos));
}

for (int j = 0; 0 != predef.dynamic_data(j).name; ++j) {
predefined_macros::dynamic_macros const& m = predef.dynamic_data(j);
predefine_macro(current_scope, m.name,
token_type(m.token_id, (predef.* m.generator)(), pos));
}
}

template <typename ContextT>
inline void
macromap<ContextT>::reset_macromap()
{
current_macros->clear();
predef.reset();
act_token = token_type();
}

}}}   

#if BOOST_WAVE_SERIALIZATION != 0
namespace boost { namespace serialization {

template<typename ContextT>
struct version<boost::wave::util::macromap<ContextT> >
{
typedef boost::wave::util::macromap<ContextT> target_type;
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
