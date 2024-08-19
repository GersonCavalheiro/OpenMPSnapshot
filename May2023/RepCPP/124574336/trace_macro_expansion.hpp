

#if !defined(BOOST_TRACE_MACRO_EXPANSION_HPP_D8469318_8407_4B9D_A19F_13CA60C1661F_INCLUDED)
#define BOOST_TRACE_MACRO_EXPANSION_HPP_D8469318_8407_4B9D_A19F_13CA60C1661F_INCLUDED

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <ostream>
#include <string>
#include <stack>
#include <set>

#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include <boost/wave/token_ids.hpp>
#include <boost/wave/util/macro_helpers.hpp>
#include <boost/wave/util/filesystem_compatibility.hpp>
#include <boost/wave/preprocessing_hooks.hpp>
#include <boost/wave/whitespace_handling.hpp>
#include <boost/wave/language_support.hpp>
#include <boost/wave/cpp_exceptions.hpp>

#include "stop_watch.hpp"

#ifdef BOOST_NO_STRINGSTREAM
#include <strstream>
#define BOOST_WAVE_OSSTREAM std::ostrstream
std::string BOOST_WAVE_GETSTRING(std::ostrstream& ss)
{
ss << std::ends;
std::string rval = ss.str();
ss.freeze(false);
return rval;
}
#else
#include <sstream>
#define BOOST_WAVE_GETSTRING(ss) ss.str()
#define BOOST_WAVE_OSSTREAM std::ostringstream
#endif

enum trace_flags {
trace_nothing = 0,      
trace_macros = 1,       
trace_macro_counts = 2, 
trace_includes = 4,     
trace_guards = 8        
};

class bad_pragma_exception :
public boost::wave::preprocess_exception
{
public:
enum error_code {
pragma_system_not_enabled =
boost::wave::preprocess_exception::last_error_number + 1,
pragma_mismatched_push_pop,
};

bad_pragma_exception(char const *what_, error_code code, std::size_t line_,
std::size_t column_, char const *filename_) throw()
:   boost::wave::preprocess_exception(what_,
(boost::wave::preprocess_exception::error_code)code, line_,
column_, filename_)
{
}
~bad_pragma_exception() throw() {}

char const *what() const throw() BOOST_OVERRIDE
{
return "boost::wave::bad_pragma_exception";
}
virtual bool is_recoverable() const throw() BOOST_OVERRIDE
{
return true;
}
virtual int get_severity() const throw() BOOST_OVERRIDE
{
return boost::wave::util::severity_remark;
}

static char const *error_text(int code)
{
switch(code) {
case pragma_system_not_enabled:
return "the directive '#pragma wave system()' was not enabled, use the "
"-x command line argument to enable the execution of";

case pragma_mismatched_push_pop:
return "unbalanced #pragma push/pop in input file(s) for option";
}
return "Unknown exception";
}
static boost::wave::util::severity severity_level(int code)
{
switch(code) {
case pragma_system_not_enabled:
return boost::wave::util::severity_remark;

case pragma_mismatched_push_pop:
return boost::wave::util::severity_error;
}
return boost::wave::util::severity_fatal;
}
static char const *severity_text(int code)
{
return boost::wave::util::get_severity(boost::wave::util::severity_remark);
}
};

template <typename TokenT>
class trace_macro_expansion
:   public boost::wave::context_policies::eat_whitespace<TokenT>
{
typedef boost::wave::context_policies::eat_whitespace<TokenT> base_type;

public:
trace_macro_expansion(
bool preserve_whitespace_, bool preserve_bol_whitespace_,
std::ofstream &output_, std::ostream &tracestrm_,
std::ostream &includestrm_, std::ostream &guardstrm_,
trace_flags flags_, bool enable_system_command_,
bool& generate_output_, std::string const& default_outfile_)
:   outputstrm(output_), tracestrm(tracestrm_),
includestrm(includestrm_), guardstrm(guardstrm_),
level(0), flags(flags_), logging_flags(trace_nothing),
enable_system_command(enable_system_command_),
preserve_whitespace(preserve_whitespace_),
preserve_bol_whitespace(preserve_bol_whitespace_),
generate_output(generate_output_),
default_outfile(default_outfile_),
emit_relative_filenames(false)
{
}
~trace_macro_expansion()
{
}

void enable_macro_counting()
{
logging_flags = trace_flags(logging_flags | trace_macro_counts);
}
std::map<std::string, std::size_t> const& get_macro_counts() const
{
return counts;
}

void enable_relative_names_in_line_directives(bool flag)
{
emit_relative_filenames = flag;
}
bool enable_relative_names_in_line_directives() const
{
return emit_relative_filenames;
}

void add_noexpandmacro(std::string const& name)
{
noexpandmacros.insert(name);
}

void set_license_info(std::string const& info)
{
license_info = info;
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename ContainerT>
void expanding_function_like_macro(
TokenT const &macrodef, std::vector<TokenT> const &formal_args,
ContainerT const &definition,
TokenT const &macrocall, std::vector<ContainerT> const &arguments)
{
if (enabled_macro_counting())
count_invocation(macrodef.get_value().c_str());

if (!enabled_macro_tracing())
return;
#else
template <typename ContextT, typename ContainerT, typename IteratorT>
bool
expanding_function_like_macro(ContextT const& ctx,
TokenT const &macrodef, std::vector<TokenT> const &formal_args,
ContainerT const &definition,
TokenT const &macrocall, std::vector<ContainerT> const &arguments,
IteratorT const& seqstart, IteratorT const& seqend)
{
if (enabled_macro_counting() || !noexpandmacros.empty()) {
std::string name (macrodef.get_value().c_str());

if (noexpandmacros.find(name.c_str()) != noexpandmacros.end())
return true;    

if (enabled_macro_counting())
count_invocation(name.c_str());
}

if (!enabled_macro_tracing())
return false;
#endif
if (0 == get_level()) {
BOOST_WAVE_OSSTREAM stream;

stream
<< macrocall.get_position() << ": "
<< macrocall.get_value() << "(";

for (typename ContainerT::size_type i = 0; i < arguments.size(); ++i) {
stream << boost::wave::util::impl::as_string(arguments[i]);
if (i < arguments.size()-1)
stream << ", ";
}
stream << ")" << std::endl;
output(BOOST_WAVE_GETSTRING(stream));
increment_level();
}

{
BOOST_WAVE_OSSTREAM stream;

stream
<< macrodef.get_position() << ": see macro definition: "
<< macrodef.get_value() << "(";

for (typename std::vector<TokenT>::size_type i = 0;
i < formal_args.size(); ++i)
{
stream << formal_args[i].get_value();
if (i < formal_args.size()-1)
stream << ", ";
}
stream << ")" << std::endl;
output(BOOST_WAVE_GETSTRING(stream));
}

if (formal_args.size() > 0) {
open_trace_body("invoked with\n");
for (typename std::vector<TokenT>::size_type j = 0;
j < formal_args.size(); ++j)
{
using namespace boost::wave;

BOOST_WAVE_OSSTREAM stream;
stream << formal_args[j].get_value() << " = ";
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (T_ELLIPSIS == token_id(formal_args[j])) {
for (typename ContainerT::size_type k = j;
k < arguments.size(); ++k)
{
stream << boost::wave::util::impl::as_string(arguments[k]);
if (k < arguments.size()-1)
stream << ", ";
}
}
else
#endif
{
stream << boost::wave::util::impl::as_string(arguments[j]);
}
stream << std::endl;
output(BOOST_WAVE_GETSTRING(stream));
}
close_trace_body();
}
open_trace_body();

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS == 0
return false;
#endif
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename ContainerT>
void expanding_object_like_macro(TokenT const &macrodef,
ContainerT const &definition, TokenT const &macrocall)
{
if (enabled_macro_counting())
count_invocation(macrodef.get_value().c_str());

if (!enabled_macro_tracing())
return;
#else
template <typename ContextT, typename ContainerT>
bool
expanding_object_like_macro(ContextT const& ctx,
TokenT const &macrodef, ContainerT const &definition,
TokenT const &macrocall)
{
if (enabled_macro_counting() || !noexpandmacros.empty()) {
std::string name (macrodef.get_value().c_str());

if (noexpandmacros.find(name.c_str()) != noexpandmacros.end())
return true;    

if (enabled_macro_counting())
count_invocation(name.c_str());
}

if (!enabled_macro_tracing())
return false;
#endif
if (0 == get_level()) {
BOOST_WAVE_OSSTREAM stream;

stream
<< macrocall.get_position() << ": "
<< macrocall.get_value() << std::endl;
output(BOOST_WAVE_GETSTRING(stream));
increment_level();
}

{
BOOST_WAVE_OSSTREAM stream;

stream
<< macrodef.get_position() << ": see macro definition: "
<< macrodef.get_value() << std::endl;
output(BOOST_WAVE_GETSTRING(stream));
}
open_trace_body();

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS == 0
return false;
#endif
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename ContainerT>
void expanded_macro(ContainerT const &result)
#else
template <typename ContextT, typename ContainerT>
void expanded_macro(ContextT const& ctx,ContainerT const &result)
#endif
{
if (!enabled_macro_tracing()) return;

BOOST_WAVE_OSSTREAM stream;
stream << boost::wave::util::impl::as_string(result) << std::endl;
output(BOOST_WAVE_GETSTRING(stream));

open_trace_body("rescanning\n");
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
template <typename ContainerT>
void rescanned_macro(ContainerT const &result)
#else
template <typename ContextT, typename ContainerT>
void rescanned_macro(ContextT const& ctx,ContainerT const &result)
#endif
{
if (!enabled_macro_tracing() || get_level() == 0)
return;

BOOST_WAVE_OSSTREAM stream;
stream << boost::wave::util::impl::as_string(result) << std::endl;
output(BOOST_WAVE_GETSTRING(stream));
close_trace_body();
close_trace_body();

if (1 == get_level())
decrement_level();
}

template <typename ContextT, typename ContainerT>
bool
interpret_pragma(ContextT &ctx, ContainerT &pending,
typename ContextT::token_type const &option, ContainerT const &valuetokens,
typename ContextT::token_type const &act_token)
{
typedef typename ContextT::token_type token_type;

ContainerT values(valuetokens);
boost::wave::util::impl::trim_sequence(values);    

if (option.get_value() == "timer") {
if (0 == values.size()) {
using namespace boost::wave;
timer(token_type(T_INTLIT, "1", act_token.get_position()));
}
else {
timer(values.front());
}
return true;
}
if (option.get_value() == "trace") {
return interpret_pragma_trace(ctx, values, act_token);
}
if (option.get_value() == "system") {
if (!enable_system_command) {
typename ContextT::string_type msg(
boost::wave::util::impl::as_string(values));
BOOST_WAVE_THROW_CTX(ctx, bad_pragma_exception,
pragma_system_not_enabled,
msg.c_str(), act_token.get_position());
return false;
}

return interpret_pragma_system(ctx, pending, values, act_token);
}
if (option.get_value() == "stop") {
typename ContextT::string_type msg(
boost::wave::util::impl::as_string(values));
BOOST_WAVE_THROW_CTX(ctx, boost::wave::preprocess_exception,
error_directive, msg.c_str(), act_token.get_position());
return false;
}
if (option.get_value() == "option") {
return interpret_pragma_option(ctx, values, act_token);
}
return false;
}

template <typename ContextT, typename ContainerT>
bool
emit_line_directive(ContextT const& ctx, ContainerT &pending,
typename ContextT::token_type const& act_token)
{
if (!need_emit_line_directives(ctx.get_language()) ||
!enable_relative_names_in_line_directives())
{
return false;
}

typename ContextT::position_type pos = act_token.get_position();
unsigned int column = 6;

typedef typename ContextT::token_type result_type;
using namespace boost::wave;

pos.set_column(1);
pending.push_back(result_type(T_PP_LINE, "#line", pos));

pos.set_column(column);      
pending.push_back(result_type(T_SPACE, " ", pos));

char buffer[22];

using namespace std;    
sprintf (buffer, "%zd", pos.get_line());

pos.set_column(++column);                 
pending.push_back(result_type(T_INTLIT, buffer, pos));
pos.set_column(column += (unsigned int)strlen(buffer)); 
pending.push_back(result_type(T_SPACE, " ", pos));
pos.set_column(++column);                 

std::string file("\"");
boost::filesystem::path filename(
boost::wave::util::create_path(ctx.get_current_relative_filename().c_str()));

using boost::wave::util::impl::escape_lit;
file += escape_lit(boost::wave::util::native_file_string(filename)) + "\"";

pending.push_back(result_type(T_STRINGLIT, file.c_str(), pos));
pos.set_column(column += (unsigned int)file.size());    
pending.push_back(result_type(T_GENERATEDNEWLINE, "\n", pos));

return true;
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
void
opened_include_file(std::string const &relname, std::string const &absname,
std::size_t include_depth, bool is_system_include)
{
#else
template <typename ContextT>
void
opened_include_file(ContextT const& ctx, std::string const &relname,
std::string const &absname, bool is_system_include)
{
std::size_t include_depth = ctx.get_iteration_depth();
#endif
if (enabled_include_tracing()) {
for (std::size_t i = 0; i < include_depth; ++i)
includestrm << " ";

if (is_system_include)
includestrm << "<" << relname << "> (" << absname << ")";
else
includestrm << "\"" << relname << "\" (" << absname << ")";

includestrm << std::endl;
}
}

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
template <typename ContextT>
void
detected_include_guard(ContextT const& ctx, std::string const& filename,
std::string const& include_guard)
{
if (enabled_guard_tracing()) {
guardstrm << include_guard << ":" << std::endl
<< "  " << filename << std::endl;
}
}
#endif

template <typename ContextT>
bool may_skip_whitespace(ContextT const &ctx, TokenT &token,
bool &skipped_newline)
{
return this->base_type::may_skip_whitespace(
ctx, token, need_preserve_comments(ctx.get_language()),
preserve_bol_whitespace, skipped_newline) ?
!preserve_whitespace : false;
}

template <typename ContextT>
void
throw_exception(ContextT const& ctx, boost::wave::preprocess_exception const& e)
{
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
if (!is_import_directive_error(e))
boost::throw_exception(e);
#else
boost::throw_exception(e);
#endif
}
using base_type::throw_exception;

protected:
#if BOOST_WAVE_SUPPORT_MS_EXTENSIONS != 0
bool is_import_directive_error(boost::wave::preprocess_exception const& e)
{
using namespace boost::wave;
if (e.get_errorcode() != preprocess_exception::ill_formed_directive)
return false;

std::string error(e.description());
std::string::size_type p = error.find_last_of(":");
return p != std::string::npos && error.substr(p+2) == "import";
}
#endif

template <typename ContextT, typename ContainerT>
bool
interpret_pragma_trace(ContextT& ctx, ContainerT const &values,
typename ContextT::token_type const &act_token)
{
typedef typename ContextT::token_type token_type;
typedef typename token_type::string_type string_type;

bool valid_option = false;

if (1 == values.size()) {
token_type const& value = values.front();

if (value.get_value() == "enable" ||
value.get_value() == "on" ||
value.get_value() == "1")
{
enable_tracing(static_cast<trace_flags>(
tracing_enabled() | trace_macros));
valid_option = true;
}
else if (value.get_value() == "disable" ||
value.get_value() == "off" ||
value.get_value() == "0")
{
enable_tracing(static_cast<trace_flags>(
tracing_enabled() & ~trace_macros));
valid_option = true;
}
}
if (!valid_option) {
string_type option_str("trace");

if (values.size() > 0) {
option_str += "(";
option_str += boost::wave::util::impl::as_string(values);
option_str += ")";
}
BOOST_WAVE_THROW_CTX(ctx, boost::wave::preprocess_exception,
ill_formed_pragma_option, option_str.c_str(),
act_token.get_position());
return false;
}
return true;
}

template <typename ContextT>
static bool
interpret_pragma_option_preserve_set(int mode, bool &preserve_whitespace,
bool& preserve_bol_whitespace, ContextT &ctx)
{
switch(mode) {
case 0:
preserve_whitespace = false;
preserve_bol_whitespace = false;
ctx.set_language(
enable_preserve_comments(ctx.get_language(), false),
false);
break;

case 1:
preserve_whitespace = false;
preserve_bol_whitespace = true;
ctx.set_language(
enable_preserve_comments(ctx.get_language(), false),
false);
break;

case 2:
preserve_whitespace = false;
preserve_bol_whitespace = true;
ctx.set_language(
enable_preserve_comments(ctx.get_language()),
false);
break;

case 3:
preserve_whitespace = true;
preserve_bol_whitespace = true;
ctx.set_language(
enable_preserve_comments(ctx.get_language()),
false);
break;

default:
return false;
}
return true;
}

template <typename ContextT, typename IteratorT>
bool
interpret_pragma_option_preserve(ContextT &ctx, IteratorT &it,
IteratorT end, typename ContextT::token_type const &act_token)
{
using namespace boost::wave;

token_id id = util::impl::skip_whitespace(it, end);
if (T_COLON == id)
id = util::impl::skip_whitespace(it, end);

if (T_IDENTIFIER == id) {
if ((*it).get_value() == "push") {
if (need_preserve_comments(ctx.get_language())) {
if (preserve_whitespace)
preserve_options.push(3);
else
preserve_options.push(2);
}
else if (preserve_bol_whitespace) {
preserve_options.push(1);
}
else {
preserve_options.push(0);
}
return true;
}
else if ((*it).get_value() == "pop") {
if (preserve_options.empty()) {
BOOST_WAVE_THROW_CTX(ctx, bad_pragma_exception,
pragma_mismatched_push_pop, "preserve",
act_token.get_position());
}

bool result = interpret_pragma_option_preserve_set(
preserve_options.top(), preserve_whitespace,
preserve_bol_whitespace, ctx);
preserve_options.pop();
return result;
}
return false;
}

if (T_PP_NUMBER != id)
return false;

using namespace std;    
return interpret_pragma_option_preserve_set(
atoi((*it).get_value().c_str()), preserve_whitespace,
preserve_bol_whitespace, ctx);
}

template <typename ContextT, typename IteratorT>
bool
interpret_pragma_option_line(ContextT &ctx, IteratorT &it,
IteratorT end, typename ContextT::token_type const &act_token)
{
using namespace boost::wave;

token_id id = util::impl::skip_whitespace(it, end);
if (T_COLON == id)
id = util::impl::skip_whitespace(it, end);

if (T_IDENTIFIER == id) {
if ((*it).get_value() == "push") {
int mode = 0;
if (need_emit_line_directives(ctx.get_language())) {
mode = 1;
if (enable_relative_names_in_line_directives())
mode = 2;
}
line_options.push(mode);
return true;
}
else if ((*it).get_value() == "pop") {
if (line_options.empty()) {
BOOST_WAVE_THROW_CTX(ctx, bad_pragma_exception,
pragma_mismatched_push_pop, "line",
act_token.get_position());
}

ctx.set_language(
enable_emit_line_directives(ctx.get_language(), 0 != line_options.top()),
false);
enable_relative_names_in_line_directives(2 == line_options.top());
line_options.pop();
return true;
}
return false;
}

if (T_PP_NUMBER != id)
return false;

using namespace std;    
int emit_lines = atoi((*it).get_value().c_str());
if (0 == emit_lines || 1 == emit_lines || 2 == emit_lines) {
ctx.set_language(
enable_emit_line_directives(ctx.get_language(), emit_lines),
false);
return true;
}
return false;
}

template <typename ContextT>
bool
interpret_pragma_option_output_open(boost::filesystem::path &fpath,
ContextT& ctx, typename ContextT::token_type const &act_token)
{
namespace fs = boost::filesystem;

boost::wave::util::create_directories(
boost::wave::util::branch_path(fpath));

std::ios::openmode mode = std::ios::out;
if (fs::exists(fpath) && written_by_us.find(fpath) != written_by_us.end())
mode = (std::ios::openmode)(std::ios::out | std::ios::app);

written_by_us.insert(fpath);

if (outputstrm.is_open())
outputstrm.close();

outputstrm.open(fpath.string().c_str(), mode);
if (!outputstrm.is_open()) {
BOOST_WAVE_THROW_CTX(ctx, boost::wave::preprocess_exception,
could_not_open_output_file,
fpath.string().c_str(), act_token.get_position());
return false;
}

if (mode == std::ios::out && !license_info.empty())
outputstrm << license_info;

generate_output = true;
current_outfile = fpath;
return true;
}

bool interpret_pragma_option_output_close(bool generate)
{
if (outputstrm.is_open())
outputstrm.close();
current_outfile = boost::filesystem::path();
generate_output = generate;
return true;
}

template <typename ContextT, typename IteratorT>
bool
interpret_pragma_option_output(ContextT &ctx, IteratorT &it,
IteratorT end, typename ContextT::token_type const &act_token)
{
using namespace boost::wave;
namespace fs = boost::filesystem;

typedef typename ContextT::token_type token_type;
typedef typename token_type::string_type string_type;

token_id id = util::impl::skip_whitespace(it, end);
if (T_COLON == id)
id = util::impl::skip_whitespace(it, end);

bool result = false;
if (T_STRINGLIT == id) {
namespace fs = boost::filesystem;

string_type fname ((*it).get_value());
fs::path fpath (boost::wave::util::create_path(
util::impl::unescape_lit(fname.substr(1, fname.size()-2)).c_str()));
fpath = boost::wave::util::complete_path(fpath, ctx.get_current_directory());
result = interpret_pragma_option_output_open(fpath, ctx, act_token);
}
else if (T_IDENTIFIER == id) {
if ((*it).get_value() == "null") {
result = interpret_pragma_option_output_close(false);
}
else if ((*it).get_value() == "push") {
if (output_options.empty() && current_outfile.empty() &&
!default_outfile.empty() && default_outfile != "-")
{
current_outfile = boost::wave::util::complete_path(
default_outfile, ctx.get_current_directory());
}

output_options.push(
output_option_type(generate_output, current_outfile));
result = true;
}
else if ((*it).get_value() == "pop") {
if (output_options.empty()) {
BOOST_WAVE_THROW_CTX(ctx, bad_pragma_exception,
pragma_mismatched_push_pop, "output",
act_token.get_position());
return false;
}

output_option_type const& opts = output_options.top();
generate_output = opts.first;
current_outfile = opts.second;
if (!current_outfile.empty()) {
result = interpret_pragma_option_output_open(current_outfile,
ctx, act_token);
}
else {
result = interpret_pragma_option_output_close(generate_output);
}
output_options.pop();
}
}
else if (T_DEFAULT == id) {
if (!default_outfile.empty()) {
if (default_outfile == "-") {
result = interpret_pragma_option_output_close(false);
}
else {
fs::path fpath(boost::wave::util::create_path(default_outfile));
result = interpret_pragma_option_output_open(fpath, ctx,
act_token);
}
}
else {
result = interpret_pragma_option_output_close(true);
}
}
return result;
}

template <typename StringT>
StringT unlit(StringT const& str)
{
return str.substr(1, str.size()-2);
}

template <typename StringT>
StringT merge_string_lits(StringT const& lhs, StringT const& rhs)
{
StringT result ("\"");

result += unlit(lhs);
result += unlit(rhs);
result += "\"";
return result;
}

template <typename ContextT, typename ContainerT>
void join_adjacent_string_tokens(ContextT &ctx, ContainerT const& values,
ContainerT& joined_values)
{
using namespace boost::wave;

typedef typename ContextT::token_type token_type;
typedef typename ContainerT::const_iterator const_iterator;

token_type* current = 0;

const_iterator end = values.end();
for (const_iterator it = values.begin(); it != end; ++it) {
token_id id(*it);

if (id == T_STRINGLIT) {
if (!current) {
joined_values.push_back(*it);
current = &joined_values.back();
}
else {
current->set_value(merge_string_lits(
current->get_value(), (*it).get_value()));
}
}
else if (current) {
typedef util::impl::next_token<const_iterator> next_token_type;
token_id next_id (next_token_type::peek(it, end, true));

if (next_id != T_STRINGLIT) {
current = 0;
joined_values.push_back(*it);
}
}
else {
joined_values.push_back(*it);
}
}
}

template <typename ContextT, typename ContainerT>
bool
interpret_pragma_option(ContextT &ctx, ContainerT const &cvalues,
typename ContextT::token_type const &act_token)
{
using namespace boost::wave;

typedef typename ContextT::token_type token_type;
typedef typename token_type::string_type string_type;
typedef typename ContainerT::const_iterator const_iterator;

ContainerT values;
join_adjacent_string_tokens(ctx, cvalues, values);

const_iterator end = values.end();
for (const_iterator it = values.begin(); it != end; ) {
bool valid_option = false;

token_type const &value = *it;
if (value.get_value() == "preserve") {
valid_option = interpret_pragma_option_preserve(ctx, it, end,
act_token);
}
else if (value.get_value() == "line") {
valid_option = interpret_pragma_option_line(ctx, it, end,
act_token);
}
else if (value.get_value() == "output") {
valid_option = interpret_pragma_option_output(ctx, it, end,
act_token);
}

if (!valid_option) {
string_type option_str("option");

if (values.size() > 0) {
option_str += "(";
option_str += util::impl::as_string(values);
option_str += ")";
}
BOOST_WAVE_THROW_CTX(ctx, boost::wave::preprocess_exception,
ill_formed_pragma_option,
option_str.c_str(), act_token.get_position());
return false;
}

token_id id = util::impl::skip_whitespace(it, end);
if (id == T_COMMA)
util::impl::skip_whitespace(it, end);
}
return true;
}

template <typename ContextT, typename ContainerT>
bool
interpret_pragma_system(ContextT& ctx, ContainerT &pending,
ContainerT const &values,
typename ContextT::token_type const &act_token)
{
typedef typename ContextT::token_type token_type;
typedef typename token_type::string_type string_type;

if (0 == values.size()) return false;   

string_type stdout_file(std::tmpnam(0));
string_type stderr_file(std::tmpnam(0));
string_type system_str(boost::wave::util::impl::as_string(values));
string_type native_cmd(system_str);

system_str += " >" + stdout_file + " 2>" + stderr_file;
if (0 != std::system(system_str.c_str())) {
string_type error_str("unable to spawn command: ");

error_str += native_cmd;
BOOST_WAVE_THROW_CTX(ctx, boost::wave::preprocess_exception,
ill_formed_pragma_option,
error_str.c_str(), act_token.get_position());
return false;
}

typedef typename ContextT::lexer_type lexer_type;
typedef typename ContextT::input_policy_type input_policy_type;
typedef boost::wave::iteration_context<
ContextT, lexer_type, input_policy_type>
iteration_context_type;

iteration_context_type iter_ctx(ctx, stdout_file.c_str(),
act_token.get_position(), ctx.get_language());
ContainerT pragma;

for (; iter_ctx.first != iter_ctx.last; ++iter_ctx.first)
pragma.push_back(*iter_ctx.first);

pending.splice(pending.begin(), pragma);

std::remove(stdout_file.c_str());
std::remove(stderr_file.c_str());
return true;
}

void enable_tracing(trace_flags flags)
{ logging_flags = flags; }

trace_flags tracing_enabled()
{ return logging_flags; }

void open_trace_body(char const *label = 0)
{
if (label)
output(label);
output("[\n");
increment_level();
}
void close_trace_body()
{
if (get_level() > 0) {
decrement_level();
output("]\n");
tracestrm << std::flush;      
}
}

template <typename StringT>
void output(StringT const &outstr) const
{
indent(get_level());
tracestrm << outstr;          
}

void indent(int level) const
{
for (int i = 0; i < level; ++i)
tracestrm << "  ";        
}

int increment_level() { return ++level; }
int decrement_level() { BOOST_ASSERT(level > 0); return --level; }
int get_level() const { return level; }

bool enabled_macro_tracing() const
{
return (flags & trace_macros) && (logging_flags & trace_macros);
}
bool enabled_include_tracing() const
{
return (flags & trace_includes);
}
bool enabled_guard_tracing() const
{
return (flags & trace_guards);
}
bool enabled_macro_counting() const
{
return logging_flags & trace_macro_counts;
}

void count_invocation(std::string const& name)
{
typedef std::map<std::string, std::size_t>::iterator iterator;
typedef std::map<std::string, std::size_t>::value_type value_type;

iterator it = counts.find(name);
if (it == counts.end())
{
std::pair<iterator, bool> p = counts.insert(value_type(name, 0));
if (p.second)
it = p.first;
}

if (it != counts.end())
++(*it).second;
}

void timer(TokenT const &value)
{
if (value.get_value() == "0" || value.get_value() == "restart") {
elapsed_time.start();
}
else if (value.get_value() == "1") {
std::cerr
<< value.get_position() << ": "
<< elapsed_time.format_elapsed_time()
<< std::endl;
}
else if (value.get_value() == "suspend") {
elapsed_time.stop();
}
else if (value.get_value() == "resume") {
elapsed_time.resume();
}
}

private:
std::ofstream &outputstrm;      
std::ostream &tracestrm;        
std::ostream &includestrm;      
std::ostream &guardstrm;        
int level;                      
trace_flags flags;              
trace_flags logging_flags;      
bool enable_system_command;     
bool preserve_whitespace;       
bool preserve_bol_whitespace;   
bool& generate_output;          
std::string const& default_outfile;         
boost::filesystem::path current_outfile;    

stop_watch elapsed_time;        
std::set<boost::filesystem::path> written_by_us;    

typedef std::pair<bool, boost::filesystem::path> output_option_type;
std::stack<output_option_type> output_options;  
std::stack<int> line_options;       
std::stack<int> preserve_options;   

std::map<std::string, std::size_t> counts;    
bool emit_relative_filenames;   

std::set<std::string> noexpandmacros;   

std::string license_info;       
};

#undef BOOST_WAVE_GETSTRING
#undef BOOST_WAVE_OSSTREAM

#endif 
