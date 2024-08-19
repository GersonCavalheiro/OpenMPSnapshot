

#if !defined(BOOST_CPP_ITERATOR_HPP_175CA88F_7273_43FA_9039_BCF7459E1F29_INCLUDED)
#define BOOST_CPP_ITERATOR_HPP_175CA88F_7273_43FA_9039_BCF7459E1F29_INCLUDED

#include <string>
#include <vector>
#include <list>
#include <cstdlib>
#include <cctype>

#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/classic_multi_pass.hpp>
#include <boost/spirit/include/classic_parse_tree_utils.hpp>

#include <boost/wave/wave_config.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <boost/wave/util/insert_whitespace_detection.hpp>
#include <boost/wave/util/macro_helpers.hpp>
#include <boost/wave/util/cpp_macromap_utils.hpp>
#include <boost/wave/util/interpret_pragma.hpp>
#include <boost/wave/util/transform_iterator.hpp>
#include <boost/wave/util/functor_input.hpp>
#include <boost/wave/util/filesystem_compatibility.hpp>

#include <boost/wave/grammars/cpp_grammar_gen.hpp>
#include <boost/wave/grammars/cpp_expression_grammar_gen.hpp>
#if BOOST_WAVE_ENABLE_COMMANDLINE_MACROS != 0
#include <boost/wave/grammars/cpp_predef_macros_gen.hpp>
#endif

#include <boost/wave/whitespace_handling.hpp>
#include <boost/wave/cpp_iteration_context.hpp>
#include <boost/wave/cpp_exceptions.hpp>
#include <boost/wave/language_support.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace wave {
namespace util {

template <
typename ContextT, typename ParseNodeT, typename TokenT,
typename PositionT
>
inline bool
retrieve_macroname(ContextT& ctx, ParseNodeT const &node,
boost::spirit::classic::parser_id id, TokenT &macroname, PositionT& act_pos,
bool update_position)
{
ParseNodeT const* name_node = 0;

using boost::spirit::classic::find_node;
if (!find_node(node, id, &name_node))
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_define_statement,
"bad parse tree (unexpected)", act_pos);
return false;
}

typename ParseNodeT::children_t const& children = name_node->children;

if (0 == children.size() ||
children.front().value.begin() == children.front().value.end())
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_define_statement,
"bad parse tree (unexpected)", act_pos);
return false;
}

macroname = *children.front().value.begin();
if (update_position) {
macroname.set_position(act_pos);
act_pos.set_column(act_pos.get_column() + macroname.get_value().size());
}
return true;
}

template <typename ParseNodeT, typename ContainerT, typename PositionT>
inline bool
retrieve_macrodefinition(
ParseNodeT const &node, boost::spirit::classic::parser_id id,
ContainerT &macrodefinition, PositionT& act_pos, bool update_position)
{
using namespace boost::wave;
typedef typename ParseNodeT::const_tree_iterator const_tree_iterator;

std::pair<const_tree_iterator, const_tree_iterator> nodes;

using boost::spirit::classic::get_node_range;
if (get_node_range(node, id, nodes)) {
typename ContainerT::iterator last_nonwhite = macrodefinition.end();
const_tree_iterator end = nodes.second;

for (const_tree_iterator cit = nodes.first; cit != end; ++cit) {
if ((*cit).value.begin() != (*cit).value.end()) {
typename ContainerT::iterator inserted = macrodefinition.insert(
macrodefinition.end(), *(*cit).value.begin());

if (!IS_CATEGORY(macrodefinition.back(), WhiteSpaceTokenType) &&
T_NEWLINE != token_id(macrodefinition.back()) &&
T_EOF != token_id(macrodefinition.back()))
{
last_nonwhite = inserted;
}

if (update_position) {
(*inserted).set_position(act_pos);
act_pos.set_column(
act_pos.get_column() + (*inserted).get_value().size());
}
}
}

if (last_nonwhite != macrodefinition.end()) {
if (update_position) {
act_pos.set_column((*last_nonwhite).get_position().get_column() +
(*last_nonwhite).get_value().size());
}
macrodefinition.erase(++last_nonwhite, macrodefinition.end());
}
return true;
}
return false;
}

#if BOOST_WAVE_ENABLE_COMMANDLINE_MACROS != 0
template <typename ContextT>
bool add_macro_definition(ContextT &ctx, std::string macrostring,
bool is_predefined, boost::wave::language_support language)
{
typedef typename ContextT::token_type token_type;
typedef typename ContextT::lexer_type lexer_type;
typedef typename token_type::position_type position_type;
typedef boost::wave::grammars::predefined_macros_grammar_gen<lexer_type>
predef_macros_type;

using namespace boost::wave;
using namespace std;    

std::string::iterator begin = macrostring.begin();
std::string::iterator end = macrostring.end();

while(begin != end && isspace(*begin))
++begin;

position_type act_pos("<command line>");
boost::spirit::classic::tree_parse_info<lexer_type> hit =
predef_macros_type::parse_predefined_macro(
lexer_type(begin, end, position_type(), language), lexer_type());

if (!hit.match || (!hit.full && T_EOF != token_id(*hit.stop))) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_macro_definition,
macrostring.c_str(), act_pos);
return false;
}

token_type macroname;
std::vector<token_type> macroparameters;
typename ContextT::token_sequence_type macrodefinition;
bool has_parameters = false;

if (!boost::wave::util::retrieve_macroname(ctx, *hit.trees.begin(),
BOOST_WAVE_PLAIN_DEFINE_ID, macroname, act_pos, true))
return false;
has_parameters = boost::wave::util::retrieve_macrodefinition(*hit.trees.begin(),
BOOST_WAVE_MACRO_PARAMETERS_ID, macroparameters, act_pos, true);
boost::wave::util::retrieve_macrodefinition(*hit.trees.begin(),
BOOST_WAVE_MACRO_DEFINITION_ID, macrodefinition, act_pos, true);

if (!macrodefinition.empty() && token_id(macrodefinition.back()) == T_EOF)
macrodefinition.pop_back();

if (macrodefinition.empty() && '=' != macrostring[macrostring.size()-1])
macrodefinition.push_back(token_type(T_INTLIT, "1", act_pos));

return ctx.add_macro_definition(macroname, has_parameters, macroparameters,
macrodefinition, is_predefined);
}
#endif 

}   

template <typename ContextT> class pp_iterator;

namespace impl {

template <typename ContextT>
class pp_iterator_functor {

public:
typedef typename ContextT::token_type               result_type;

static result_type const eof;

private:
typedef typename ContextT::token_sequence_type      token_sequence_type;

typedef typename ContextT::lexer_type               lexer_type;
typedef typename result_type::string_type           string_type;
typedef typename result_type::position_type         position_type;
typedef boost::wave::grammars::cpp_grammar_gen<lexer_type, token_sequence_type>
cpp_grammar_type;

typedef base_iteration_context<ContextT, lexer_type>
base_iteration_context_type;
typedef iteration_context<ContextT, lexer_type> iteration_context_type;

typedef typename cpp_grammar_type::node_factory_type node_factory_type;
typedef boost::spirit::classic::tree_parse_info<lexer_type, node_factory_type>
tree_parse_info_type;
typedef boost::spirit::classic::tree_match<lexer_type, node_factory_type>
parse_tree_match_type;
typedef typename parse_tree_match_type::node_t       parse_node_type;       
typedef typename parse_tree_match_type::parse_node_t parse_node_value_type; 
typedef typename parse_tree_match_type::container_t  parse_tree_type;       

public:
template <typename IteratorT>
pp_iterator_functor(ContextT &ctx_, IteratorT const &first_,
IteratorT const &last_, typename ContextT::position_type const &pos_)
:   ctx(ctx_),
iter_ctx(new base_iteration_context_type(ctx,
lexer_type(first_, last_, pos_,
boost::wave::enable_prefer_pp_numbers(ctx.get_language())),
lexer_type(),
pos_.get_file().c_str()
)),
seen_newline(true), skipped_newline(false),
must_emit_line_directive(false), act_pos(ctx_.get_main_pos()),
whitespace(boost::wave::need_insert_whitespace(ctx.get_language()))
{
act_pos.set_file(pos_.get_file());
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
ctx_.set_current_filename(pos_.get_file().c_str());
#endif
iter_ctx->emitted_lines = (unsigned int)(-1);   
}

result_type const &operator()();

result_type const &current_token() const { return act_token; }

protected:
friend class pp_iterator<ContextT>;
bool on_include_helper(char const *t, char const *s, bool is_system,
bool include_next);

protected:
result_type const &get_next_token();
result_type const &pp_token();

template <typename IteratorT>
bool extract_identifier(IteratorT &it);
template <typename IteratorT>
bool ensure_is_last_on_line(IteratorT& it, bool call_hook = true);
template <typename IteratorT>
bool skip_to_eol_with_check(IteratorT &it, bool call_hook = true);

bool pp_directive();
template <typename IteratorT>
bool handle_pp_directive(IteratorT &it);
bool dispatch_directive(tree_parse_info_type const &hit,
result_type const& found_directive,
token_sequence_type const& found_eoltokens);
void replace_undefined_identifiers(token_sequence_type &expanded);

void on_include(string_type const &s, bool is_system, bool include_next);
void on_include(typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end, bool include_next);

void on_define(parse_node_type const &node);
void on_undefine(lexer_type const &it);

void on_ifdef(result_type const& found_directive, lexer_type const &it);
void on_ifndef(result_type const& found_directive, lexer_type const& it);
void on_else();
void on_endif();
void on_illformed(typename result_type::string_type s);

void on_line(typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end);
void on_if(result_type const& found_directive,
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end);
void on_elif(result_type const& found_directive,
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end);
void on_error(typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end);
#if BOOST_WAVE_SUPPORT_WARNING_DIRECTIVE != 0
void on_warning(typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end);
#endif
bool on_pragma(typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end);

bool emit_line_directive();
bool returned_from_include();

bool interpret_pragma(token_sequence_type const &pragma_body,
token_sequence_type &result);

private:
ContextT &ctx;              
boost::shared_ptr<base_iteration_context_type> iter_ctx;

bool seen_newline;              
bool skipped_newline;           
bool must_emit_line_directive;  
result_type act_token;          
typename result_type::position_type &act_pos;   

token_sequence_type unput_queue;     
token_sequence_type pending_queue;   

boost::wave::util::insert_whitespace_detection whitespace;
};

template <typename ContextT>
typename pp_iterator_functor<ContextT>::result_type const
pp_iterator_functor<ContextT>::eof;

template <typename ContextT>
inline bool
pp_iterator_functor<ContextT>::returned_from_include()
{
if (iter_ctx->first == iter_ctx->last && ctx.get_iteration_depth() > 0) {
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().returning_from_include_file();
#else
ctx.get_hooks().returning_from_include_file(ctx.derived());
#endif

BOOST_WAVE_STRINGTYPE oldfile = iter_ctx->real_filename;
position_type old_pos (act_pos);

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
if (need_include_guard_detection(ctx.get_language())) {
std::string guard_name;
if (iter_ctx->first.has_include_guards(guard_name))
ctx.add_pragma_once_header(ctx.get_current_filename(), guard_name);
}
#endif
iter_ctx = ctx.pop_iteration_context();

must_emit_line_directive = true;
iter_ctx->emitted_lines = (unsigned int)(-1);   
seen_newline = true;

act_pos.set_file(iter_ctx->filename);
act_pos.set_line(iter_ctx->line);
act_pos.set_column(0);

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
namespace fs = boost::filesystem;
fs::path rfp(wave::util::create_path(iter_ctx->real_filename.c_str()));
std::string real_filename(rfp.string());
ctx.set_current_filename(real_filename.c_str());
#endif
ctx.set_current_directory(iter_ctx->real_filename.c_str());
ctx.set_current_relative_filename(iter_ctx->real_relative_filename.c_str());

if (iter_ctx->if_block_depth != ctx.get_if_block_depth()) {
using boost::wave::util::impl::escape_lit;
BOOST_WAVE_STRINGTYPE msg(escape_lit(oldfile));
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, unbalanced_if_endif,
msg.c_str(), old_pos);
}
return true;
}
return false;
}

namespace impl {

template <typename ContextT>
bool consider_emitting_line_directive(ContextT const& ctx, token_id id)
{
if (need_preserve_comments(ctx.get_language()))
{
if (!IS_CATEGORY(id, EOLTokenType) && !IS_CATEGORY(id, EOFTokenType))
{
return true;
}
}
if (!IS_CATEGORY(id, WhiteSpaceTokenType) &&
!IS_CATEGORY(id, EOLTokenType) && !IS_CATEGORY(id, EOFTokenType))
{
return true;
}
return false;
}
}

template <typename ContextT>
inline typename pp_iterator_functor<ContextT>::result_type const &
pp_iterator_functor<ContextT>::operator()()
{
using namespace boost::wave;

ctx.init_context();

bool was_seen_newline = seen_newline;
bool was_skipped_newline = skipped_newline;
token_id id = T_UNKNOWN;

try {   
do {
if (skipped_newline) {
was_skipped_newline = true;
skipped_newline = false;
}

get_next_token();

id = token_id(act_token);
if (!need_preserve_comments(ctx.get_language()) &&
(T_CPPCOMMENT == id || context_policies::util::ccomment_has_newline(act_token)))
{
act_token.set_token_id(id = T_NEWLINE);
act_token.set_value("\n");
}

if (IS_CATEGORY(id, EOLTokenType))
seen_newline = true;

} while (ctx.get_hooks().may_skip_whitespace(ctx.derived(), act_token, skipped_newline));
}
catch (boost::wave::cpplexer::lexing_exception const& e) {
ctx.get_hooks().throw_exception(ctx.derived(), e);
return act_token;
}

if (was_skipped_newline)
skipped_newline = true;

if ((must_emit_line_directive || (was_seen_newline && skipped_newline)) &&
impl::consider_emitting_line_directive(ctx, id))
{
if (need_emit_line_directives(ctx.get_language()) && emit_line_directive())
{
skipped_newline = false;
ctx.get_hooks().may_skip_whitespace(ctx.derived(), act_token, skipped_newline);     
id = token_id(act_token);
}
}

seen_newline = false;
switch (id) {
case T_NONREPLACABLE_IDENTIFIER:
act_token.set_token_id(id = T_IDENTIFIER);
break;

case T_GENERATEDNEWLINE:  
act_token.set_token_id(id = T_NEWLINE);
++iter_ctx->emitted_lines;
seen_newline = true;
break;

case T_NEWLINE:
case T_CPPCOMMENT:
seen_newline = true;
++iter_ctx->emitted_lines;
break;

#if BOOST_WAVE_SUPPORT_CPP0X != 0
case T_RAWSTRINGLIT:
iter_ctx->emitted_lines +=
context_policies::util::rawstring_count_newlines(act_token);
break;
#endif

case T_CCOMMENT:          
iter_ctx->emitted_lines +=
context_policies::util::ccomment_count_newlines(act_token);
break;

case T_PP_NUMBER:        
{
token_sequence_type rescanned;

std::string pp_number(
util::to_string<std::string>(act_token.get_value()));

lexer_type it = lexer_type(pp_number.begin(),
pp_number.end(), act_token.get_position(),
ctx.get_language());
lexer_type end = lexer_type();

for (; it != end && T_EOF != token_id(*it); ++it)
rescanned.push_back(*it);

pending_queue.splice(pending_queue.begin(), rescanned);
act_token = pending_queue.front();
id = token_id(act_token);
pending_queue.pop_front();
}
break;

case T_EOF:
seen_newline = true;
break;

default:    
if (IS_CATEGORY(id, WhiteSpaceTokenType))
seen_newline = was_seen_newline;
break;
}

if (token_is_valid(act_token) && whitespace.must_insert(id, act_token.get_value())) {
whitespace.shift_tokens(T_SPACE);
pending_queue.push_front(act_token);        
return act_token = result_type(T_SPACE,
typename result_type::string_type(" "),
act_token.get_position());
}
whitespace.shift_tokens(id);
return ctx.get_hooks().generated_token(ctx.derived(), act_token);
}

template <typename ContextT>
inline typename pp_iterator_functor<ContextT>::result_type const &
pp_iterator_functor<ContextT>::get_next_token()
{
using namespace boost::wave;

if (!pending_queue.empty() || !unput_queue.empty())
return pp_token();      

bool returned_from_include_file = returned_from_include();

if (iter_ctx->first != iter_ctx->last) {
do {
if (!pending_queue.empty()) {
util::on_exit::pop_front<token_sequence_type>
pop_front_token(pending_queue);

return act_token = pending_queue.front();
}

bool was_seen_newline = seen_newline || returned_from_include_file;

act_token = *iter_ctx->first;
act_pos = act_token.get_position();

token_id id = token_id(act_token);

if (T_EOF == id) {
whitespace.shift_tokens(T_EOF);
++iter_ctx->first;

if ((!seen_newline || act_pos.get_column() > 1) &&
!need_single_line(ctx.get_language()))
{
if (need_no_newline_at_end_of_file(ctx.get_language()))
{
seen_newline = true;
pending_queue.push_back(
result_type(T_NEWLINE, "\n", act_pos)
);
}
else
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
last_line_not_terminated, "", act_pos);
}
}
continue;   
}
else if (T_NEWLINE == id || T_CPPCOMMENT == id) {
seen_newline = true;
++iter_ctx->first;

if (!ctx.get_if_block_status()) {
whitespace.shift_tokens(id);  
util::impl::call_skipped_token_hook(ctx, act_token);
continue;
}
return act_token;
}
seen_newline = false;

if (was_seen_newline && pp_directive()) {
if (iter_ctx->first == iter_ctx->last)
{
seen_newline = true;
act_token = result_type(T_NEWLINE, "\n", act_pos);
}

}
else if (ctx.get_if_block_status()) {
return pp_token();
}
else {
if (T_NEWLINE == token_id(act_token)) {
seen_newline = true;
must_emit_line_directive = true;
}

util::impl::call_skipped_token_hook(ctx, act_token);
++iter_ctx->first;
}

} while ((iter_ctx->first != iter_ctx->last) ||
(returned_from_include_file = returned_from_include()));

if (ctx.get_if_block_depth() > 0 && !need_single_line(ctx.get_language()))
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
missing_matching_endif, "", act_pos);
}
}
else {
act_token = eof;            
}

return act_token;                   
}

template <typename ContextT>
inline bool
pp_iterator_functor<ContextT>::emit_line_directive()
{
using namespace boost::wave;

typename ContextT::position_type pos = act_token.get_position();


if (must_emit_line_directive ||
iter_ctx->emitted_lines+1 != act_pos.get_line())
{
pending_queue.push_front(act_token);
pos.set_line(act_pos.get_line());

if (iter_ctx->emitted_lines+2 == act_pos.get_line() && act_pos.get_line() != 1) {
act_token = result_type(T_NEWLINE, "\n", pos);
}
else {
act_pos.set_line(act_pos.get_line()-1);
iter_ctx->emitted_lines = act_pos.get_line()-1;

token_sequence_type pending;

if (!ctx.get_hooks().emit_line_directive(ctx, pending, act_token))
{
unsigned int column = 6;

pos.set_column(1);
pending.push_back(result_type(T_PP_LINE, "#line", pos));

pos.set_column(column);      
pending.push_back(result_type(T_SPACE, " ", pos));


std::string buffer = lexical_cast<std::string>(pos.get_line());

pos.set_column(++column);                 
pending.push_back(result_type(T_INTLIT, buffer.c_str(), pos));
pos.set_column(column += (unsigned int)buffer.size()); 
pending.push_back(result_type(T_SPACE, " ", pos));
pos.set_column(++column);                 

std::string file("\"");
boost::filesystem::path filename(
wave::util::create_path(act_pos.get_file().c_str()));

using wave::util::impl::escape_lit;
file += escape_lit(wave::util::native_file_string(filename)) + "\"";

pending.push_back(result_type(T_STRINGLIT, file.c_str(), pos));
pos.set_column(column += (unsigned int)file.size());    
pending.push_back(result_type(T_GENERATEDNEWLINE, "\n", pos));
}

if (!pending.empty()) {
pending_queue.splice(pending_queue.begin(), pending);
act_token = pending_queue.front();
pending_queue.pop_front();
}
}

must_emit_line_directive = false;     
return true;
}

must_emit_line_directive = false;         
return false;
}

template <typename ContextT>
inline typename pp_iterator_functor<ContextT>::result_type const &
pp_iterator_functor<ContextT>::pp_token()
{
using namespace boost::wave;

token_id id = token_id(*iter_ctx->first);

do {
if (!pending_queue.empty()) {
act_token = pending_queue.front();
pending_queue.pop_front();
act_pos = act_token.get_position();
}
else if (!unput_queue.empty()
|| T_IDENTIFIER == id
|| IS_CATEGORY(id, KeywordTokenType)
|| IS_EXTCATEGORY(id, OperatorTokenType|AltExtTokenType)
|| IS_CATEGORY(id, BoolLiteralTokenType))
{
act_token = ctx.expand_tokensequence(iter_ctx->first,
iter_ctx->last, pending_queue, unput_queue, skipped_newline);
}
else {
act_token = *iter_ctx->first;
++iter_ctx->first;
}
id = token_id(act_token);

} while (T_PLACEHOLDER == id);
return act_token;
}

namespace impl {

template <typename ContextT>
bool call_found_directive_hook(ContextT& ctx,
typename ContextT::token_type const& found_directive)
{
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().found_directive(found_directive);
#else
if (ctx.get_hooks().found_directive(ctx.derived(), found_directive))
return true;    
#endif
return false;
}


template <typename ContextT, typename IteratorT>
bool next_token_is_pp_directive(ContextT &ctx, IteratorT &it, IteratorT const &end)
{
using namespace boost::wave;

token_id id = T_UNKNOWN;
for (; it != end; ++it) {
id = token_id(*it);
if (!IS_CATEGORY(id, WhiteSpaceTokenType))
break;          
if (IS_CATEGORY(id, EOLTokenType) || IS_CATEGORY(id, EOFTokenType))
break;          
if (T_CPPCOMMENT == id ||
context_policies::util::ccomment_has_newline(*it))
{
break;
}

util::impl::call_skipped_token_hook(ctx, *it);
}
BOOST_ASSERT(it == end || id != T_UNKNOWN);
return it != end && IS_CATEGORY(id, PPTokenType);
}

template <typename ContextT, typename IteratorT>
bool pp_is_last_on_line(ContextT &ctx, IteratorT &it, IteratorT const &end,
bool call_hook = true)
{
using namespace boost::wave;

if (call_hook)
util::impl::call_skipped_token_hook(ctx, *it);

for (++it; it != end; ++it) {
token_id id = token_id(*it);

if (T_CPPCOMMENT == id || T_NEWLINE == id ||
context_policies::util::ccomment_has_newline(*it))
{
if (call_hook)
util::impl::call_skipped_token_hook(ctx, *it);
++it;           
return true;    
}

if (!IS_CATEGORY(id, WhiteSpaceTokenType))
break;

if (call_hook)
util::impl::call_skipped_token_hook(ctx, *it);
}
return need_no_newline_at_end_of_file(ctx.get_language());
}

template <typename ContextT, typename IteratorT>
bool skip_to_eol(ContextT &ctx, IteratorT &it, IteratorT const &end,
bool call_hook = true)
{
using namespace boost::wave;

for (; it != end; ++it) {
token_id id = token_id(*it);

if (T_CPPCOMMENT == id || T_NEWLINE == id ||
context_policies::util::ccomment_has_newline(*it))
{
util::impl::call_skipped_token_hook(ctx, *it);
++it;           
return true;    
}

if (call_hook)
util::impl::call_skipped_token_hook(ctx, *it);
}
return false;
}

template <typename ContextT, typename ContainerT>
inline void
remove_leading_whitespace(ContextT &ctx, ContainerT& c, bool call_hook = true)
{
typename ContainerT::iterator it = c.begin();
while (IS_CATEGORY(*it, WhiteSpaceTokenType)) {
typename ContainerT::iterator save = it++;
if (call_hook)
util::impl::call_skipped_token_hook(ctx, *save);
c.erase(save);
}
}
}

template <typename ContextT>
template <typename IteratorT>
inline bool
pp_iterator_functor<ContextT>::extract_identifier(IteratorT &it)
{
token_id id = util::impl::skip_whitespace(it, iter_ctx->last);
if (T_IDENTIFIER == id || IS_CATEGORY(id, KeywordTokenType) ||
IS_EXTCATEGORY(id, OperatorTokenType|AltExtTokenType) ||
IS_CATEGORY(id, BoolLiteralTokenType))
{
IteratorT save = it;
if (impl::pp_is_last_on_line(ctx, save, iter_ctx->last, false))
return true;
}

impl::skip_to_eol(ctx, it, iter_ctx->last);

string_type str(util::impl::as_string<string_type>(iter_ctx->first, it));

seen_newline = true;
iter_ctx->first = it;
on_illformed(str);
return false;
}

template <typename ContextT>
template <typename IteratorT>
inline bool
pp_iterator_functor<ContextT>::ensure_is_last_on_line(IteratorT& it, bool call_hook)
{
if (!impl::pp_is_last_on_line(ctx, it, iter_ctx->last, call_hook))
{
impl::skip_to_eol(ctx, it, iter_ctx->last);

string_type str(util::impl::as_string<string_type>(
iter_ctx->first, it));

seen_newline = true;
iter_ctx->first = it;

on_illformed(str);
return false;
}

if (it == iter_ctx->last && !need_single_line(ctx.get_language()))
{
seen_newline = true;    
iter_ctx->first = it;

if (!need_no_newline_at_end_of_file(ctx.get_language()))
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
last_line_not_terminated, "", act_pos);
}

return false;
}
return true;
}

template <typename ContextT>
template <typename IteratorT>
inline bool
pp_iterator_functor<ContextT>::skip_to_eol_with_check(IteratorT &it, bool call_hook)
{
typename ContextT::string_type value ((*it).get_value());
if (!impl::skip_to_eol(ctx, it, iter_ctx->last, call_hook) &&
!need_single_line(ctx.get_language()))
{
seen_newline = true;    
iter_ctx->first = it;

if (!need_no_newline_at_end_of_file(ctx.get_language()))
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
last_line_not_terminated, "", act_pos);
}
return false;
}

seen_newline = true;
iter_ctx->first = it;
return true;
}

template <typename ContextT>
template <typename IteratorT>
inline bool
pp_iterator_functor<ContextT>::handle_pp_directive(IteratorT &it)
{
token_id id = token_id(*it);
bool can_exit = true;
bool call_hook_in_skip = true;
if (!ctx.get_if_block_status()) {
if (IS_EXTCATEGORY(*it, PPConditionalTokenType)) {
switch (id) {
case T_PP_IFDEF:        
case T_PP_IFNDEF:       
case T_PP_IF:           
ctx.enter_if_block(false);
break;

case T_PP_ELIF:         
if (!ctx.get_enclosing_if_block_status()) {
if (!ctx.enter_elif_block(false)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
missing_matching_if, "#elif", act_pos);
return true;  
}
}
else {
can_exit = false;   
}
break;

case T_PP_ELSE:         
case T_PP_ENDIF:        
{
if (T_PP_ELSE == token_id(*it))
on_else();
else
on_endif();

ensure_is_last_on_line(it);

seen_newline = true;
iter_ctx->first = it;
}
return true;

default:                
on_illformed((*it).get_value());
break;
}
}
else {
util::impl::call_skipped_token_hook(ctx, *it);
++it;
}
}
else {
result_type directive = *it;
bool include_next = false;
switch (id) {
case T_PP_QHEADER:        
#if BOOST_WAVE_SUPPORT_INCLUDE_NEXT != 0
case T_PP_QHEADER_NEXT:
#endif
include_next = (T_PP_QHEADER_NEXT == id) ? true : false;
if (!impl::call_found_directive_hook(ctx, *it))
{
string_type dir((*it).get_value());

if (ensure_is_last_on_line(it))
{
seen_newline = true;
iter_ctx->first = it;
on_include (dir, false, include_next);
}
return true;
}
break;

case T_PP_HHEADER:        
#if BOOST_WAVE_SUPPORT_INCLUDE_NEXT != 0
case T_PP_HHEADER_NEXT:
#endif
include_next = (T_PP_HHEADER_NEXT == id) ? true : false;
if (!impl::call_found_directive_hook(ctx, *it))
{
string_type dir((*it).get_value());

if (ensure_is_last_on_line(it))
{
seen_newline = true;
iter_ctx->first = it;
on_include (dir, true, include_next);
}
return true;
}
break;

case T_PP_ELSE:         
case T_PP_ENDIF:        
if (!impl::call_found_directive_hook(ctx, *it))
{
if (T_PP_ELSE == token_id(*it))
on_else();
else
on_endif();

ensure_is_last_on_line(it);

seen_newline = true;
iter_ctx->first = it;
return true;
}
break;


case T_PP_UNDEF:                
if (!impl::call_found_directive_hook(ctx, *it) &&
extract_identifier(it))
{
on_undefine(it);
}
call_hook_in_skip = false;
break;

case T_PP_IFDEF:                
if (!impl::call_found_directive_hook(ctx, *it) &&
extract_identifier(it))
{
on_ifdef(directive, it);
}
call_hook_in_skip = false;
break;

case T_PP_IFNDEF:               
if (!impl::call_found_directive_hook(ctx, *it) &&
extract_identifier(it))
{
on_ifndef(directive, it);
}
call_hook_in_skip = false;
break;

#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
#endif

default:
can_exit = false;
break;
}
}

if (can_exit) {
skip_to_eol_with_check(it, call_hook_in_skip);
return true;    
}
return false;   
}

template <typename ContextT>
inline bool
pp_iterator_functor<ContextT>::pp_directive()
{
using namespace cpplexer;

lexer_type it = iter_ctx->first;

if (!impl::next_token_is_pp_directive(ctx, it, iter_ctx->last)) {
if (it != iter_ctx->last && T_POUND == BASE_TOKEN(token_id(*it))) {
if (impl::pp_is_last_on_line(ctx, it, iter_ctx->last)) {
seen_newline = true;
iter_ctx->first = it;
return true;
}
else if (ctx.get_if_block_status()) {
impl::skip_to_eol(ctx, it, iter_ctx->last);
seen_newline = true;

string_type str(boost::wave::util::impl::as_string<string_type>(
iter_ctx->first, it));

token_sequence_type faulty_line;

for (; iter_ctx->first != it; ++iter_ctx->first)
faulty_line.push_back(*iter_ctx->first);

token_sequence_type pending;
if (ctx.get_hooks().found_unknown_directive(ctx, faulty_line, pending))
{
if (!pending.empty())
pending_queue.splice(pending_queue.begin(), pending);
return true;
}

on_illformed(str);
}
}

return false;
}

if (it == iter_ctx->last)
return false;

if (handle_pp_directive(it)) {
return true;    
}

bool found_eof = false;
result_type found_directive;
token_sequence_type found_eoltokens;

tree_parse_info_type hit = cpp_grammar_type::parse_cpp_grammar(
it, iter_ctx->last, act_pos, found_eof, found_directive, found_eoltokens);

if (hit.match) {
iter_ctx->first = hit.stop;
seen_newline = true;
must_emit_line_directive = true;

bool result = dispatch_directive(hit, found_directive, found_eoltokens);

if (found_eof && !need_single_line(ctx.get_language()) &&
!need_no_newline_at_end_of_file(ctx.get_language()))
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
last_line_not_terminated, "", act_pos);
}
return result;
}
else if (token_id(found_directive) != T_EOF) {
impl::skip_to_eol(ctx, it, iter_ctx->last);
seen_newline = true;

string_type str(boost::wave::util::impl::as_string<string_type>(
iter_ctx->first, it));
iter_ctx->first = it;

on_illformed(str);
}
return false;
}

template <typename ContextT>
inline bool
pp_iterator_functor<ContextT>::dispatch_directive(
tree_parse_info_type const &hit, result_type const& found_directive,
token_sequence_type const& found_eoltokens)
{
using namespace cpplexer;

typedef typename parse_tree_type::const_iterator const_child_iterator_t;

const_child_iterator_t begin = hit.trees.begin();

parse_tree_type const& root = (*begin).children;
parse_node_value_type const& nodeval = get_first_leaf(*root.begin()).value;

const_child_iterator_t begin_child_it = (*root.begin()).children.begin();
const_child_iterator_t end_child_it = (*root.begin()).children.end();

token_id id = token_id(found_directive);

if (impl::call_found_directive_hook(ctx, found_directive))
return true;    

switch (id) {


case T_PP_INCLUDE:      
#if BOOST_WAVE_SUPPORT_INCLUDE_NEXT != 0
case T_PP_INCLUDE_NEXT: 
#endif
on_include (begin_child_it, end_child_it, T_PP_INCLUDE_NEXT == id);
break;

case T_PP_DEFINE:       
on_define (*begin);
break;


case T_PP_IF:           
on_if(found_directive, begin_child_it, end_child_it);
break;

case T_PP_ELIF:         
on_elif(found_directive, begin_child_it, end_child_it);
break;



case T_PP_LINE:         
on_line(begin_child_it, end_child_it);
break;

case T_PP_ERROR:        
on_error(begin_child_it, end_child_it);
break;

#if BOOST_WAVE_SUPPORT_WARNING_DIRECTIVE != 0
case T_PP_WARNING:      
on_warning(begin_child_it, end_child_it);
break;
#endif

case T_PP_PRAGMA:       
return on_pragma(begin_child_it, end_child_it);

#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
case T_MSEXT_PP_REGION:
case T_MSEXT_PP_ENDREGION:
break;              
#endif

default:                
on_illformed((*nodeval.begin()).get_value());

{
token_sequence_type expanded;
get_token_value<result_type, parse_node_type> get_value;

std::copy(make_ref_transform_iterator(begin_child_it, get_value),
make_ref_transform_iterator(end_child_it, get_value),
std::inserter(expanded, expanded.end()));
pending_queue.splice(pending_queue.begin(), expanded);
}
break;
}

typename token_sequence_type::const_iterator eol = found_eoltokens.begin();
impl::skip_to_eol(ctx, eol, found_eoltokens.end());
return true;    
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_include (string_type const &s,
bool is_system, bool include_next)
{
BOOST_ASSERT(ctx.get_if_block_status());

typename string_type::size_type pos_end = s.find_last_of(is_system ? '>' : '\"');

if (string_type::npos == pos_end) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_include_statement,
s.c_str(), act_pos);
return;
}

typename string_type::size_type pos_begin =
s.find_last_of(is_system ? '<' : '\"', pos_end-1);

if (string_type::npos == pos_begin) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_include_statement,
s.c_str(), act_pos);
return;
}

std::string file_token(s.substr(pos_begin, pos_end - pos_begin + 1).c_str());
std::string file_path(s.substr(pos_begin + 1, pos_end - pos_begin - 1).c_str());

on_include_helper(file_token.c_str(), file_path.c_str(), is_system,
include_next);
}

template <typename ContextT>
inline bool
pp_iterator_functor<ContextT>::on_include_helper(char const* f, char const* s,
bool is_system, bool include_next)
{
namespace fs = boost::filesystem;

std::string file_path(s);
std::string dir_path;
#if BOOST_WAVE_SUPPORT_INCLUDE_NEXT != 0
char const* current_name = include_next ? iter_ctx->real_filename.c_str() : 0;
#else
char const* current_name = 0; 
#endif

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().found_include_directive(f, include_next);
#else
if (ctx.get_hooks().found_include_directive(ctx.derived(), f, include_next))
return true;    
#endif

file_path = util::impl::unescape_lit(file_path);
std::string native_path_str;

if (!ctx.get_hooks().locate_include_file(ctx, file_path, is_system,
current_name, dir_path, native_path_str))
{
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_include_file,
file_path.c_str(), act_pos);
return false;
}

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
if (!ctx.has_pragma_once(native_path_str))
#endif
{
ctx.set_current_directory(native_path_str.c_str());

boost::shared_ptr<base_iteration_context_type> new_iter_ctx(
new iteration_context_type(ctx, native_path_str.c_str(), act_pos,
boost::wave::enable_prefer_pp_numbers(ctx.get_language()),
is_system ? base_iteration_context_type::system_header :
base_iteration_context_type::user_header));

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().opened_include_file(dir_path, file_path,
ctx.get_iteration_depth(), is_system);
#else
ctx.get_hooks().opened_include_file(ctx.derived(), dir_path, file_path,
is_system);
#endif

iter_ctx->real_relative_filename = ctx.get_current_relative_filename().c_str();
iter_ctx->filename = act_pos.get_file();
iter_ctx->line = act_pos.get_line();
iter_ctx->if_block_depth = ctx.get_if_block_depth();
iter_ctx->emitted_lines = (unsigned int)(-1);   

ctx.push_iteration_context(act_pos, iter_ctx);
iter_ctx = new_iter_ctx;
seen_newline = true;        
must_emit_line_directive = true;

act_pos.set_file(iter_ctx->filename);  
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
fs::path rfp(wave::util::create_path(iter_ctx->real_filename.c_str()));
std::string real_filename(rfp.string());
ctx.set_current_filename(real_filename.c_str());
#endif

ctx.set_current_relative_filename(dir_path.c_str());
iter_ctx->real_relative_filename = dir_path.c_str();

act_pos.set_line(iter_ctx->line);
act_pos.set_column(0);
}
return true;
}


template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_include(
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end, bool include_next)
{
BOOST_ASSERT(ctx.get_if_block_status());

get_token_value<result_type, parse_node_type> get_value;
token_sequence_type expanded;
token_sequence_type toexpand;

std::copy(make_ref_transform_iterator(begin, get_value),
make_ref_transform_iterator(end, get_value),
std::inserter(toexpand, toexpand.end()));

typename token_sequence_type::iterator begin2 = toexpand.begin();
ctx.expand_whole_tokensequence(begin2, toexpand.end(), expanded,
false, false);

using namespace boost::wave::util::impl;
string_type s (trim_whitespace(as_string(expanded)));
bool is_system = '<' == s[0] && '>' == s[s.size()-1];

if (!is_system && !('\"' == s[0] && '\"' == s[s.size()-1])) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_include_statement,
s.c_str(), act_pos);
return;
}
on_include(s, is_system, include_next);
}


template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_define (parse_node_type const &node)
{
BOOST_ASSERT(ctx.get_if_block_status());

result_type macroname;
std::vector<result_type> macroparameters;
token_sequence_type macrodefinition;
bool has_parameters = false;
position_type pos(act_token.get_position());

if (!boost::wave::util::retrieve_macroname(ctx, node,
BOOST_WAVE_PLAIN_DEFINE_ID, macroname, pos, false))
return;
has_parameters = boost::wave::util::retrieve_macrodefinition(node,
BOOST_WAVE_MACRO_PARAMETERS_ID, macroparameters, pos, false);
boost::wave::util::retrieve_macrodefinition(node,
BOOST_WAVE_MACRO_DEFINITION_ID, macrodefinition, pos, false);

if (has_parameters) {
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (boost::wave::need_variadics(ctx.get_language())) {
using namespace cpplexer;
typedef typename std::vector<result_type>::iterator
parameter_iterator_t;

bool seen_ellipses = false;
parameter_iterator_t end = macroparameters.end();
for (parameter_iterator_t pit = macroparameters.begin();
pit != end; ++pit)
{
if (seen_ellipses) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
bad_define_statement, macroname.get_value().c_str(),
(*pit).get_position());
return;
}
if (T_ELLIPSIS == token_id(*pit))
seen_ellipses = true;

if ("__VA_ARGS__" == (*pit).get_value()) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
bad_define_statement_va_args,
macroname.get_value().c_str(), (*pit).get_position());
return;
}

#if BOOST_WAVE_SUPPORT_VA_OPT != 0
if (boost::wave::need_va_opt(ctx.get_language()) &&
("__VA_OPT__" == (*pit).get_value())) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
bad_define_statement_va_opt,
macroname.get_value().c_str(), (*pit).get_position());
return;
}
#endif
}

if (!seen_ellipses) {
typedef typename token_sequence_type::iterator definition_iterator_t;

bool seen_va_args = false;
#if BOOST_WAVE_SUPPORT_VA_OPT != 0
bool seen_va_opt = false;
#endif
definition_iterator_t pend = macrodefinition.end();
for (definition_iterator_t dit = macrodefinition.begin();
dit != pend; ++dit)
{
if (T_IDENTIFIER == token_id(*dit) &&
"__VA_ARGS__" == (*dit).get_value())
{
seen_va_args = true;
}
#if BOOST_WAVE_SUPPORT_VA_OPT != 0
if (T_IDENTIFIER == token_id(*dit) &&
"__VA_OPT__" == (*dit).get_value())
{
seen_va_opt = true;
}
#endif
}
if (seen_va_args) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
bad_define_statement_va_args,
macroname.get_value().c_str(), act_token.get_position());
return;
}
#if BOOST_WAVE_SUPPORT_VA_OPT != 0
if (seen_va_opt) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
bad_define_statement_va_opt,
macroname.get_value().c_str(), act_token.get_position());
return;
}
#endif
}
}
else
#endif 
{
using namespace cpplexer;
typedef typename std::vector<result_type>::iterator
parameter_iterator_t;

parameter_iterator_t end = macroparameters.end();
for (parameter_iterator_t pit = macroparameters.begin();
pit != end; ++pit)
{
if (T_ELLIPSIS == token_id(*pit)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
bad_define_statement, macroname.get_value().c_str(),
(*pit).get_position());
return;
}
}
}
}

ctx.add_macro_definition(macroname, has_parameters, macroparameters,
macrodefinition);
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_undefine (lexer_type const &it)
{
BOOST_ASSERT(ctx.get_if_block_status());

ctx.remove_macro_definition((*it).get_value()); 
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_ifdef(
result_type const& found_directive, lexer_type const &it)
{

bool is_defined = false;
token_sequence_type directive;

directive.insert(directive.end(), *it);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
is_defined = ctx.is_defined_macro((*it).get_value()); 
ctx.get_hooks().evaluated_conditional_expression(directive, is_defined);
#else
do {
is_defined = ctx.is_defined_macro((*it).get_value()); 
} while (ctx.get_hooks().evaluated_conditional_expression(ctx.derived(),
found_directive, directive, is_defined));
#endif
ctx.enter_if_block(is_defined);
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_ifndef(
result_type const& found_directive, lexer_type const &it)
{

bool is_defined = false;
token_sequence_type directive;

directive.insert(directive.end(), *it);

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
is_defined = ctx.is_defined_macro((*it).get_value()); 
ctx.get_hooks().evaluated_conditional_expression(directive, is_defined);
#else
do {
is_defined = ctx.is_defined_macro((*it).get_value()); 
} while (ctx.get_hooks().evaluated_conditional_expression(ctx.derived(),
found_directive, directive, is_defined));
#endif
ctx.enter_if_block(!is_defined);
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_else()
{
if (!ctx.enter_else_block()) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, missing_matching_if,
"#else", act_pos);
}
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_endif()
{
if (!ctx.exit_if_block()) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, missing_matching_if,
"#endif", act_pos);
}
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::replace_undefined_identifiers(
token_sequence_type &expanded)
{
typename token_sequence_type::iterator exp_end = expanded.end();
for (typename token_sequence_type::iterator exp_it = expanded.begin();
exp_it != exp_end; ++exp_it)
{
using namespace boost::wave;

token_id id = token_id(*exp_it);
if (IS_CATEGORY(id, IdentifierTokenType) ||
IS_CATEGORY(id, KeywordTokenType))
{
(*exp_it).set_token_id(T_INTLIT);
(*exp_it).set_value("0");
}
}
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_if(
result_type const& found_directive,
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end)
{
get_token_value<result_type, parse_node_type> get_value;
token_sequence_type toexpand;

std::copy(make_ref_transform_iterator(begin, get_value),
make_ref_transform_iterator(end, get_value),
std::inserter(toexpand, toexpand.end()));

impl::remove_leading_whitespace(ctx, toexpand);

bool if_status = false;
grammars::value_error status = grammars::error_noerror;
token_sequence_type expanded;

do {
expanded.clear();

typename token_sequence_type::iterator begin2 = toexpand.begin();
ctx.expand_whole_tokensequence(begin2, toexpand.end(), expanded);

replace_undefined_identifiers(expanded);

#if BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS != 0
{
string_type outstr(boost::wave::util::impl::as_string(toexpand));
outstr += "(" + boost::wave::util::impl::as_string(expanded) + ")";
BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS_OUT << "#if " << outstr
<< std::endl;
}
#endif
try {
if_status = grammars::expression_grammar_gen<result_type>::
evaluate(expanded.begin(), expanded.end(), act_pos,
ctx.get_if_block_status(), status);
}
catch (boost::wave::preprocess_exception const& e) {
ctx.get_hooks().throw_exception(ctx.derived(), e);
break;
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().evaluated_conditional_expression(toexpand, if_status);
} while (false);
#else
} while (ctx.get_hooks().evaluated_conditional_expression(ctx.derived(),
found_directive, toexpand, if_status)
&& status == grammars::error_noerror);
#endif

ctx.enter_if_block(if_status);
if (grammars::error_noerror != status) {
string_type expression = util::impl::as_string(expanded);
if (0 == expression.size())
expression = "<empty expression>";

if (grammars::error_division_by_zero & status) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, division_by_zero,
expression.c_str(), act_pos);
}
else if (grammars::error_integer_overflow & status) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, integer_overflow,
expression.c_str(), act_pos);
}
else if (grammars::error_character_overflow & status) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
character_literal_out_of_range, expression.c_str(), act_pos);
}
}
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_elif(
result_type const& found_directive,
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end)
{
get_token_value<result_type, parse_node_type> get_value;
token_sequence_type toexpand;

std::copy(make_ref_transform_iterator(begin, get_value),
make_ref_transform_iterator(end, get_value),
std::inserter(toexpand, toexpand.end()));

impl::remove_leading_whitespace(ctx, toexpand);

if (ctx.get_if_block_some_part_status()) {
if (!ctx.enter_elif_block(false)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
missing_matching_if, "#elif", act_pos);
}

typename token_sequence_type::iterator begin2 = toexpand.begin();

impl::skip_to_eol(ctx, begin2, toexpand.end());
return;     
}

bool if_status = false;
grammars::value_error status = grammars::error_noerror;
token_sequence_type expanded;

do {
expanded.clear();

typename token_sequence_type::iterator begin2 = toexpand.begin();
ctx.expand_whole_tokensequence(begin2, toexpand.end(), expanded);

replace_undefined_identifiers(expanded);

#if BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS != 0
{
string_type outstr(boost::wave::util::impl::as_string(toexpand));
outstr += "(" + boost::wave::util::impl::as_string(expanded) + ")";
BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS_OUT << "#elif " << outstr << std::endl;
}
#endif

try {
if_status = grammars::expression_grammar_gen<result_type>::
evaluate(expanded.begin(), expanded.end(), act_pos,
ctx.get_if_block_status(), status);
}
catch (boost::wave::preprocess_exception const& e) {
ctx.get_hooks().throw_exception(ctx.derived(), e);
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
ctx.get_hooks().evaluated_conditional_expression(toexpand, if_status);
} while (false);
#else
} while (ctx.get_hooks().evaluated_conditional_expression(ctx.derived(),
found_directive, toexpand, if_status)
&& status == grammars::error_noerror);
#endif

if (!ctx.enter_elif_block(if_status)) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, missing_matching_if,
"#elif", act_pos);
return;
}

if (grammars::error_noerror != status) {
string_type expression = util::impl::as_string(expanded);
if (0 == expression.size())
expression = "<empty expression>";

if (grammars::error_division_by_zero & status) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, division_by_zero,
expression.c_str(), act_pos);
}
else if (grammars::error_integer_overflow & status) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
integer_overflow, expression.c_str(), act_pos);
}
else if (grammars::error_character_overflow & status) {
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception,
character_literal_out_of_range, expression.c_str(), act_pos);
}
}
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_illformed(
typename result_type::string_type s)
{
BOOST_ASSERT(ctx.get_if_block_status());

typename string_type::size_type p = s.find_last_not_of('\n');
if (string_type::npos != p)
s = s.substr(0, p+1);

BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, ill_formed_directive,
s.c_str(), act_pos);
}


namespace impl {

template <typename IteratorT, typename StringT>
bool retrieve_line_info (IteratorT first, IteratorT const &last,
unsigned int &line, StringT &file,
boost::wave::preprocess_exception::error_code& error)
{
using namespace boost::wave;
token_id id = token_id(*first);
if (T_PP_NUMBER == id || T_INTLIT == id) {
using namespace std;    
line = (unsigned int)atoi((*first).get_value().c_str());
if (0 == line)
error = preprocess_exception::bad_line_number;

using namespace boost::spirit::classic;
if (!parse((*first).get_value().c_str(), int_p).full)
error = preprocess_exception::bad_line_number;

while (++first != last && IS_CATEGORY(*first, WhiteSpaceTokenType))
;   

if (first != last) {
if (T_STRINGLIT != token_id(*first)) {
error = preprocess_exception::bad_line_filename;
return false;
}

StringT const& file_lit = (*first).get_value();

if ('L' == file_lit[0]) {
error = preprocess_exception::bad_line_filename;
return false;       
}

file = file_lit.substr(1, file_lit.size()-2);

while (++first != last && IS_CATEGORY(*first, WhiteSpaceTokenType))
;   
}
return first == last;
}
error = preprocess_exception::bad_line_statement;
return false;
}
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_line(
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end)
{
BOOST_ASSERT(ctx.get_if_block_status());

token_sequence_type expanded;
get_token_value<result_type, parse_node_type> get_value;

typedef typename ref_transform_iterator_generator<
get_token_value<result_type, parse_node_type>,
typename parse_tree_type::const_iterator
>::type const_tree_iterator_t;

const_tree_iterator_t first = make_ref_transform_iterator(begin, get_value);
const_tree_iterator_t last = make_ref_transform_iterator(end, get_value);

unsigned int line = 0;
preprocess_exception::error_code error = preprocess_exception::no_error;
string_type file_name;
token_sequence_type toexpand;

std::copy(first, last, std::inserter(toexpand, toexpand.end()));
if (!impl::retrieve_line_info(first, last, line, file_name, error)) {
typename token_sequence_type::iterator begin2 = toexpand.begin();
ctx.expand_whole_tokensequence(begin2, toexpand.end(),
expanded, false, false);

error = preprocess_exception::no_error;
if (!impl::retrieve_line_info(expanded.begin(), expanded.end(),
line, file_name, error))
{
typename ContextT::string_type msg(
boost::wave::util::impl::as_string(expanded));
BOOST_WAVE_THROW_VAR_CTX(ctx, preprocess_exception, error,
msg.c_str(), act_pos);
return;
}

ctx.get_hooks().found_line_directive(ctx.derived(), expanded, line,
file_name.c_str());
}
else {
ctx.get_hooks().found_line_directive(ctx.derived(), toexpand, line,
file_name.c_str());
}

BOOST_ASSERT(unput_queue.empty());
BOOST_ASSERT(pending_queue.empty());

must_emit_line_directive = true;

if (error != preprocess_exception::no_error) {
typename ContextT::string_type msg(
boost::wave::util::impl::as_string(expanded));
BOOST_WAVE_THROW_VAR_CTX(ctx, preprocess_exception, error,
msg.c_str(), act_pos);
return;
}

if (!file_name.empty()) {    
using boost::wave::util::impl::unescape_lit;
act_pos.set_file(unescape_lit(file_name).c_str());
}
act_pos.set_line(line);
if (iter_ctx->first != iter_ctx->last)
{
iter_ctx->first.set_position(act_pos);
}
}

template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_error(
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end)
{
BOOST_ASSERT(ctx.get_if_block_status());

token_sequence_type expanded;
get_token_value<result_type, parse_node_type> get_value;

typename ref_transform_iterator_generator<
get_token_value<result_type, parse_node_type>,
typename parse_tree_type::const_iterator
>::type first = make_ref_transform_iterator(begin, get_value);

#if BOOST_WAVE_PREPROCESS_ERROR_MESSAGE_BODY != 0
token_sequence_type toexpand;

std::copy(first, make_ref_transform_iterator(end, get_value),
std::inserter(toexpand, toexpand.end()));

typename token_sequence_type::iterator begin2 = toexpand.begin();
ctx.expand_whole_tokensequence(begin2, toexpand.end(), expanded,
false, false);
if (!ctx.get_hooks().found_error_directive(ctx.derived(), toexpand))
#else
std::copy(first, make_ref_transform_iterator(end, get_value),
std::inserter(expanded, expanded.end()));
if (!ctx.get_hooks().found_error_directive(ctx.derived(), expanded))
#endif
{
BOOST_WAVE_STRINGTYPE msg(boost::wave::util::impl::as_string(expanded));
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, error_directive,
msg.c_str(), act_pos);
}
}

#if BOOST_WAVE_SUPPORT_WARNING_DIRECTIVE != 0
template <typename ContextT>
inline void
pp_iterator_functor<ContextT>::on_warning(
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end)
{
BOOST_ASSERT(ctx.get_if_block_status());

token_sequence_type expanded;
get_token_value<result_type, parse_node_type> get_value;

typename ref_transform_iterator_generator<
get_token_value<result_type, parse_node_type>,
typename parse_tree_type::const_iterator
>::type first = make_ref_transform_iterator(begin, get_value);

#if BOOST_WAVE_PREPROCESS_ERROR_MESSAGE_BODY != 0
token_sequence_type toexpand;

std::copy(first, make_ref_transform_iterator(end, get_value),
std::inserter(toexpand, toexpand.end()));

typename token_sequence_type::iterator begin2 = toexpand.begin();
ctx.expand_whole_tokensequence(begin2, toexpand.end(), expanded,
false, false);
if (!ctx.get_hooks().found_warning_directive(ctx.derived(), toexpand))
#else
std::copy(first, make_ref_transform_iterator(end, get_value),
std::inserter(expanded, expanded.end()));
if (!ctx.get_hooks().found_warning_directive(ctx.derived(), expanded))
#endif
{
BOOST_WAVE_STRINGTYPE msg(boost::wave::util::impl::as_string(expanded));
BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, warning_directive,
msg.c_str(), act_pos);
}
}
#endif 

template <typename ContextT>
inline bool
pp_iterator_functor<ContextT>::on_pragma(
typename parse_tree_type::const_iterator const &begin,
typename parse_tree_type::const_iterator const &end)
{
using namespace boost::wave;

BOOST_ASSERT(ctx.get_if_block_status());

token_sequence_type expanded;
get_token_value<result_type, parse_node_type> get_value;

typedef typename ref_transform_iterator_generator<
get_token_value<result_type, parse_node_type>,
typename parse_tree_type::const_iterator
>::type const_tree_iterator_t;

const_tree_iterator_t first = make_ref_transform_iterator(begin, get_value);
const_tree_iterator_t last = make_ref_transform_iterator(end, get_value);

expanded.push_back(result_type(T_PP_PRAGMA, "#pragma", act_token.get_position()));
expanded.push_back(result_type(T_SPACE, " ", act_token.get_position()));

while (++first != last && IS_CATEGORY(*first, WhiteSpaceTokenType))
expanded.push_back(*first);   

if (first != last) {
if (T_IDENTIFIER == token_id(*first) &&
boost::wave::need_c99(ctx.get_language()) &&
(*first).get_value() == "STDC")
{
std::copy(first, last, std::inserter(expanded, expanded.end()));
}
else {
#if BOOST_WAVE_PREPROCESS_PRAGMA_BODY != 0
token_sequence_type toexpand;

std::copy(first, last, std::inserter(toexpand, toexpand.end()));

typename token_sequence_type::iterator begin2 = toexpand.begin();
ctx.expand_whole_tokensequence(begin2, toexpand.end(),
expanded, false, false);
#else
std::copy(first, last, std::inserter(expanded, expanded.end()));
#endif
}
}
expanded.push_back(result_type(T_NEWLINE, "\n", act_token.get_position()));

BOOST_ASSERT(unput_queue.empty());
BOOST_ASSERT(pending_queue.empty());

token_sequence_type pending;
if (interpret_pragma(expanded, pending)) {
if (!pending.empty())
pending_queue.splice(pending_queue.begin(), pending);
return true;        
}

#if BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES != 0
if (boost::wave::need_emit_pragma_directives(ctx.get_language())) {
pending_queue.splice(pending_queue.begin(), expanded);
return false;       
}
#endif
return true;            
}

template <typename ContextT>
inline bool
pp_iterator_functor<ContextT>::interpret_pragma(
token_sequence_type const &pragma_body, token_sequence_type &result)
{
using namespace cpplexer;

typename token_sequence_type::const_iterator end = pragma_body.end();
typename token_sequence_type::const_iterator it = pragma_body.begin();
for (++it; it != end && IS_CATEGORY(*it, WhiteSpaceTokenType); ++it)
;   

if (it == end)      
return false;

return boost::wave::util::interpret_pragma(
ctx.derived(), act_token, it, end, result);
}

}   


template <typename ContextT>
class pp_iterator
:   public boost::spirit::classic::multi_pass<
boost::wave::impl::pp_iterator_functor<ContextT>,
boost::wave::util::functor_input
>
{
public:
typedef boost::wave::impl::pp_iterator_functor<ContextT> input_policy_type;

private:
typedef
boost::spirit::classic::multi_pass<input_policy_type, boost::wave::util::functor_input>
base_type;
typedef pp_iterator<ContextT> self_type;
typedef boost::wave::util::functor_input functor_input_type;

public:
pp_iterator()
{}

template <typename IteratorT>
pp_iterator(ContextT &ctx, IteratorT const &first, IteratorT const &last,
typename ContextT::position_type const &pos)
:   base_type(input_policy_type(ctx, first, last, pos))
{}

bool force_include(char const *path_, bool is_last)
{
bool result = this->get_functor().on_include_helper(path_, path_,
false, false);
if (is_last) {
this->functor_input_type::
template inner<input_policy_type>::advance_input();
}
return result;
}
};

}   
}   

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif 
