

#define BOOST_WAVE_SERIALIZATION        0             
#define BOOST_WAVE_BINARY_SERIALIZATION 0             
#define BOOST_WAVE_XML_SERIALIZATION    1             

#include "cpp.hpp"                                    

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/timer/timer.hpp>
#include <boost/any.hpp>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/range/algorithm/find.hpp>
#include <boost/range/end.hpp>
#include <boost/foreach.hpp>

#include <boost/wave.hpp>

#include <boost/wave/cpplexer/cpp_lex_token.hpp>      
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>   

#include <iostream>

#if BOOST_WAVE_SERIALIZATION != 0
#include <boost/serialization/serialization.hpp>
#if BOOST_WAVE_BINARY_SERIALIZATION != 0
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
typedef boost::archive::binary_iarchive iarchive;
typedef boost::archive::binary_oarchive oarchive;
#elif BOOST_WAVE_XML_SERIALIZATION != 0
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
typedef boost::archive::xml_iarchive iarchive;
typedef boost::archive::xml_oarchive oarchive;
#else
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
typedef boost::archive::text_iarchive iarchive;
typedef boost::archive::text_oarchive oarchive;
#endif
#endif

#include "trace_macro_expansion.hpp"

#if BOOST_WAVE_SEPARATE_LEXER_INSTANTIATION == 0
#include <boost/wave/cpplexer/re2clex/cpp_re2c_lexer.hpp>
#endif

#if BOOST_WAVE_SEPARATE_GRAMMAR_INSTANTIATION == 0
#include <boost/wave/grammars/cpp_intlit_grammar.hpp>
#include <boost/wave/grammars/cpp_chlit_grammar.hpp>
#include <boost/wave/grammars/cpp_grammar.hpp>
#include <boost/wave/grammars/cpp_expression_grammar.hpp>
#include <boost/wave/grammars/cpp_predef_macros_grammar.hpp>
#include <boost/wave/grammars/cpp_defined_grammar.hpp>
#include <boost/wave/grammars/cpp_has_include_grammar.hpp>
#endif

using namespace boost::spirit::classic;

using std::pair;
using std::vector;
using std::getline;
using boost::filesystem::ofstream;
using boost::filesystem::ifstream;
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::istreambuf_iterator;

typedef boost::wave::cpplexer::lex_token<> token_type;
typedef boost::wave::cpplexer::lex_iterator<token_type>
lex_iterator_type;

typedef boost::wave::context<
std::string::iterator, lex_iterator_type,
boost::wave::iteration_context_policies::load_file_to_string,
trace_macro_expansion<token_type> >
context_type;

std::string get_version()
{
std::string version (context_type::get_version_string());
version = version.substr(1, version.size()-2);      
version += std::string(" (" CPP_VERSION_DATE_STR ")");   
return version;
}

int print_interactive_version()
{
cout << "Wave: A Standard conformant C++ preprocessor based on the Boost.Wave library" << endl;
cout << "Version: " << get_version() << endl;
return 0;
}

int print_copyright()
{
char const *copyright[] = {
"",
"Wave: A Standard conformant C++ preprocessor based on the Boost.Wave library",
"http:
"",
"Copyright (c) 2001-2012 Hartmut Kaiser, Distributed under the Boost",
"Software License, Version 1.0. (See accompanying file",
"LICENSE_1_0.txt or copy at http:
0
};

for (int i = 0; 0 != copyright[i]; ++i)
cout << copyright[i] << endl;

return 0;                       
}

namespace cmd_line_utils
{
class include_paths;
}

namespace boost { namespace program_options {

void validate(boost::any &v, std::vector<std::string> const &s,
cmd_line_utils::include_paths *, long);

}} 

#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace cmd_line_utils {
inline pair<std::string, std::string>
at_option_parser(std::string const&s)
{
if ('@' == s[0])
return std::make_pair(std::string("config-file"), s.substr(1));
else
return pair<std::string, std::string>();
}

class include_paths {
public:
include_paths() : seen_separator(false) {}

vector<std::string> paths;       
vector<std::string> syspaths;    
bool seen_separator;        

static void
validate(boost::any &v, vector<std::string> const &tokens)
{
if (v.empty())
v = boost::any(include_paths());

include_paths *p = boost::any_cast<include_paths>(&v);

BOOST_ASSERT(p);
std::string const& t = po::validators::get_single_string(tokens);
if (t == "-") {
p->seen_separator = true;
}
else if (p->seen_separator) {
p->syspaths.push_back(t);
}
else {
p->paths.push_back(t);
}
}
};

bool read_config_file_options(std::string const &filename,
po::options_description const &desc, po::variables_map &vm,
bool may_fail = false)
{
ifstream ifs(filename.c_str());

if (!ifs.is_open()) {
if (!may_fail) {
cerr << filename
<< ": command line warning: config file not found"
<< endl;
}
return false;
}

vector<std::string> options;
std::string line;

while (std::getline(ifs, line)) {
std::string::size_type pos = line.find_first_not_of(" \t");
if (pos == std::string::npos)
continue;

if ('#' != line[pos]) {
std::string::size_type endpos = line.find_last_not_of(" \t");
BOOST_ASSERT(endpos != std::string::npos);
options.push_back(line.substr(pos, endpos-pos+1));
}
}

if (options.size() > 0) {
using namespace boost::program_options::command_line_style;
po::store(po::command_line_parser(options)
.options(desc).style(unix_style).run(), vm);
po::notify(vm);
}
return true;
}

struct is_argument {
bool operator()(po::option const &opt)
{
return (opt.position_key == -1) ? true : false;
}
};

std::string trim_quotes(std::string const& file)
{
if (('"' == file[0] || '\'' == file[0]) && file[0] == file[file.size()-1])
{
return file.substr(1, file.size()-2);
}
return file;
}

}

namespace boost { namespace program_options {

void validate(boost::any &v, std::vector<std::string> const &s,
cmd_line_utils::include_paths *, long)
{
cmd_line_utils::include_paths::validate(v, s);
}

}}  

namespace {

class auto_stop_watch : public stop_watch
{
public:
auto_stop_watch(std::ostream &outstrm_)
:   print_time(false), outstrm(outstrm_)
{
}

~auto_stop_watch()
{
if (print_time) {
outstrm << "Elapsed time: "
<< this->format_elapsed_time()
<< std::endl;
}
}

void set_print_time(bool print_time_)
{
print_time = print_time_;
}

private:
bool print_time;
std::ostream &outstrm;
};

inline std::string
report_iostate_error(std::ios::iostate state)
{
BOOST_ASSERT(state & (std::ios::badbit | std::ios::failbit | std::ios::eofbit));
std::string result;
if (state & std::ios::badbit) {
result += "      the reported problem was: "
"loss of integrity of the stream buffer\n";
}
if (state & std::ios::failbit) {
result += "      the reported problem was: "
"an operation was not processed correctly\n";
}
if (state & std::ios::eofbit) {
result += "      the reported problem was: "
"end-of-file while writing to the stream\n";
}
return result;
}

template <typename Context>
inline bool
get_macro_position(Context &ctx,
typename Context::token_type::string_type const& name,
typename Context::position_type &pos)
{
bool has_parameters = false;
bool is_predefined = false;
std::vector<typename Context::token_type> parameters;
typename Context::token_sequence_type definition;

return ctx.get_macro_definition(name, has_parameters, is_predefined,
pos, parameters, definition);
}

template <typename Exception>
inline int
report_error_message(Exception const &e, bool treat_warnings_as_error)
{
cerr
<< e.file_name() << ":" << e.line_no() << ":" << e.column_no()
<< ": " << e.description() << endl;

return (treat_warnings_as_error ||
e.get_severity() == boost::wave::util::severity_error ||
e.get_severity() == boost::wave::util::severity_fatal) ? 1 : 0;
}

template <typename Context>
inline int
report_error_message(Context &ctx, boost::wave::cpp_exception const &e,
bool treat_warnings_as_error)
{
int result = report_error_message(e, treat_warnings_as_error);

using boost::wave::preprocess_exception;
switch(e.get_errorcode()) {
case preprocess_exception::macro_redefinition:
{
typename Context::position_type pos;
if (get_macro_position(ctx, e.get_related_name(), pos)) {
cerr
<< pos << ": "
<< preprocess_exception::severity_text(e.get_severity())
<< ": this is the location of the previous definition."
<< endl;
}
else {
cerr
<< e.file_name() << ":" << e.line_no() << ":"
<< e.column_no() << ": "
<< preprocess_exception::severity_text(e.get_severity())
<< ": not able to retrieve the location of the previous "
<< "definition." << endl;
}
}
break;

default:
break;
}

return result;
}

inline bool
read_a_line (std::istream &instream, std::string &instring)
{
bool eol = true;
do {
std::string line;
std::getline(instream, line);
if (instream.rdstate() & std::ios::failbit)
return false;       

eol = true;
if (line.find_last_of('\\') == line.size()-1)
eol = false;

instring += line + '\n';
} while (!eol);
return true;
}

template <typename Context>
inline void
load_state(po::variables_map const &vm, Context &ctx)
{
#if BOOST_WAVE_SERIALIZATION != 0
try {
if (vm.count("state") > 0) {
fs::path state_file (
boost::wave::util::create_path(vm["state"].as<std::string>()));
if (state_file == "-")
state_file = boost::wave::util::create_path("wave.state");

std::ios::openmode mode = std::ios::in;

#if BOOST_WAVE_BINARY_SERIALIZATION != 0
mode = (std::ios::openmode)(mode | std::ios::binary);
#endif
ifstream ifs (state_file.string().c_str(), mode);
if (ifs.is_open()) {
using namespace boost::serialization;
iarchive ia(ifs);
std::string version;

ia >> make_nvp("version", version);  
if (version == CPP_VERSION_FULL_STR)
ia >> make_nvp("state", ctx);    
else {
cerr << "wave: detected version mismatch while loading state, state was not loaded." << endl;
cerr << "      loaded version:   " << version << endl;
cerr << "      expected version: " << CPP_VERSION_FULL_STR << endl;
}
}
}
}
catch (boost::archive::archive_exception const& e) {
cerr << "wave: error while loading state: "
<< e.what() << endl;
}
catch (boost::wave::preprocess_exception const& e) {
cerr << "wave: error while loading state: "
<< e.description() << endl;
}
#endif
}

template <typename Context>
inline void
save_state(po::variables_map const &vm, Context const &ctx)
{
#if BOOST_WAVE_SERIALIZATION != 0
try {
if (vm.count("state") > 0) {
fs::path state_file (boost::wave::util::create_path(
vm["state"].as<std::string>()));
if (state_file == "-")
state_file = boost::wave::util::create_path("wave.state");

std::ios::openmode mode = std::ios::out;

#if BOOST_WAVE_BINARY_SERIALIZATION != 0
mode = (std::ios::openmode)(mode | std::ios::binary);
#endif
ofstream ofs(state_file.string().c_str(), mode);
if (!ofs.is_open()) {
cerr << "wave: could not open state file for writing: "
<< state_file.string() << endl;
}
else {
using namespace boost::serialization;
oarchive oa(ofs);
std::string version(CPP_VERSION_FULL_STR);
oa << make_nvp("version", version);  
oa << make_nvp("state", ctx);        
}
}
}
catch (boost::archive::archive_exception const& e) {
cerr << "wave: error while writing state: "
<< e.what() << endl;
}
#endif
}

bool list_macro_names(context_type const& ctx, std::string filename)
{
ofstream macronames_out;
fs::path macronames_file (boost::wave::util::create_path(filename));

if (macronames_file != "-") {
macronames_file = boost::wave::util::complete_path(macronames_file);
boost::wave::util::create_directories(
boost::wave::util::branch_path(macronames_file));
macronames_out.open(macronames_file.string().c_str());
if (!macronames_out.is_open()) {
cerr << "wave: could not open file for macro name listing: "
<< macronames_file.string() << endl;
return false;
}
}
else {
macronames_out.copyfmt(cout);
macronames_out.clear(cout.rdstate());
static_cast<std::basic_ios<char> &>(macronames_out).rdbuf(cout.rdbuf());
}

typedef context_type::const_name_iterator name_iterator;
name_iterator end = ctx.macro_names_end();
for (name_iterator it = ctx.macro_names_begin(); it != end; ++it)
{
typedef std::vector<context_type::token_type> parameters_type;

bool has_pars = false;
bool predef = false;
context_type::position_type pos;
parameters_type pars;
context_type::token_sequence_type def;

if (ctx.get_macro_definition(*it, has_pars, predef, pos, pars, def))
{
macronames_out << (predef ? "-P" : "-D") << *it;
if (has_pars) {
macronames_out << "(";
parameters_type::const_iterator pend = pars.end();
for (parameters_type::const_iterator pit = pars.begin();
pit != pend; )
{
macronames_out << (*pit).get_value();
if (++pit != pend)
macronames_out << ", ";
}
macronames_out << ")";
}
macronames_out << "=";

context_type::token_sequence_type::const_iterator dend = def.end();
for (context_type::token_sequence_type::const_iterator dit = def.begin();
dit != dend; ++dit)
{
macronames_out << (*dit).get_value();
}

macronames_out << std::endl;
}
}
return true;
}

bool list_macro_counts(context_type const& ctx, std::string filename)
{
ofstream macrocounts_out;
fs::path macrocounts_file (boost::wave::util::create_path(filename));

if (macrocounts_file != "-") {
macrocounts_file = boost::wave::util::complete_path(macrocounts_file);
boost::wave::util::create_directories(
boost::wave::util::branch_path(macrocounts_file));
macrocounts_out.open(macrocounts_file.string().c_str());
if (!macrocounts_out.is_open()) {
cerr << "wave: could not open file for macro invocation count listing: "
<< macrocounts_file.string() << endl;
return false;
}
}
else {
macrocounts_out.copyfmt(cout);
macrocounts_out.clear(cout.rdstate());
static_cast<std::basic_ios<char> &>(macrocounts_out).rdbuf(cout.rdbuf());
}

std::map<std::string, std::size_t> const& counts =
ctx.get_hooks().get_macro_counts();

typedef std::map<std::string, std::size_t>::const_iterator iterator;
iterator end = counts.end();
for (iterator it = counts.begin(); it != end; ++it)
macrocounts_out << (*it).first << "," << (*it).second << std::endl;

return true;
}

std::string read_entire_file(std::istream& instream)
{
std::string content;

instream.unsetf(std::ios::skipws);

#if defined(BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS)
copy (std::istream_iterator<char>(instream),
std::istream_iterator<char>(),
std::inserter(content, content.end()));
#else
content = std::string(std::istreambuf_iterator<char>(instream.rdbuf()),
std::istreambuf_iterator<char>());
#endif
return content;
}
}   

int
do_actual_work (std::string file_name, std::istream &instream,
po::variables_map const &vm, bool input_is_stdin)
{
boost::wave::util::file_position_type current_position;
auto_stop_watch elapsed_time(cerr);
int error_count = 0;
const bool treat_warnings_as_error = vm.count("warning") &&
boost::algorithm::any_of_equal(
vm["warning"].as<std::vector<std::string> >(), "error");

try {
std::string instring;

instream.unsetf(std::ios::skipws);
if (!input_is_stdin)
instring = read_entire_file(instream);

ofstream output;
ofstream traceout;
ofstream includelistout;
ofstream listguardsout;

trace_flags enable_trace = trace_nothing;

if (vm.count("traceto")) {
fs::path trace_file(boost::wave::util::create_path(
vm["traceto"].as<std::string>()));

if (trace_file != "-") {
boost::wave::util::create_directories(
boost::wave::util::branch_path(trace_file));
traceout.open(trace_file.string().c_str());
if (!traceout.is_open()) {
cerr << "wave: could not open trace file: " << trace_file
<< endl;
return -1;
}
}
enable_trace = trace_macros;
}
if ((enable_trace & trace_macros) && !traceout.is_open()) {
traceout.copyfmt(cerr);
traceout.clear(cerr.rdstate());
static_cast<std::basic_ios<char> &>(traceout).rdbuf(cerr.rdbuf());
}

if (vm.count("listincludes")) {
fs::path includes_file(boost::wave::util::create_path(
vm["listincludes"].as<std::string>()));

if (includes_file != "-") {
boost::wave::util::create_directories(
boost::wave::util::branch_path(includes_file));
includelistout.open(includes_file.string().c_str());
if (!includelistout.is_open()) {
cerr << "wave: could not open include list file: "
<< includes_file.string() << endl;
return -1;
}
}
enable_trace = trace_flags(enable_trace | trace_includes);
}
if ((enable_trace & trace_includes) && !includelistout.is_open()) {
includelistout.copyfmt(cout);
includelistout.clear(cout.rdstate());
static_cast<std::basic_ios<char> &>(includelistout).
rdbuf(cout.rdbuf());
}

if (vm.count("listguards")) {
fs::path listguards_file(boost::wave::util::create_path(
vm["listguards"].as<std::string>()));

if (listguards_file != "-") {
boost::wave::util::create_directories(
boost::wave::util::branch_path(listguards_file));
listguardsout.open(listguards_file.string().c_str());
if (!listguardsout.is_open()) {
cerr << "wave: could not open include guard list file: "
<< listguards_file.string() << endl;
return -1;
}
}
enable_trace = trace_flags(enable_trace | trace_guards);
}
if ((enable_trace & trace_guards) && !listguardsout.is_open()) {
listguardsout.copyfmt(cout);
listguardsout.clear(cout.rdstate());
static_cast<std::basic_ios<char> &>(listguardsout).
rdbuf(cout.rdbuf());
}

bool preserve_comments = false;
bool preserve_whitespace = false;
bool preserve_bol_whitespace = false;

if (vm.count("preserve")) {
int preserve = vm["preserve"].as<int>();

switch(preserve) {
case 0:   break;                
case 3:                         
preserve_whitespace = true;
preserve_comments = true;
preserve_bol_whitespace = true;
break;

case 2:                         
preserve_comments = true;
preserve_bol_whitespace = true;
break;

case 1:                         
preserve_bol_whitespace = true;
break;

default:
cerr << "wave: bogus preserve whitespace option value: "
<< preserve << ", should be 0, 1, 2, or 3" << endl;
return -1;
}
}

bool enable_system_command = false;

if (vm.count("extended"))
enable_system_command = true;

bool allow_output = true;    
std::string default_outfile; 
trace_macro_expansion<token_type> hooks(preserve_whitespace,
preserve_bol_whitespace, output, traceout, includelistout,
listguardsout, enable_trace, enable_system_command, allow_output,
default_outfile);

if (vm.count("macrocounts"))
hooks.enable_macro_counting();

std::string license;

if (vm.count("license")) {
std::string license_file(vm["license"].as<std::string>());
ifstream license_stream(license_file.c_str());

if (!license_stream.is_open()) {
cerr << "wave: could not open specified license file: "
<< license_file << endl;
return -1;
}
license = read_entire_file(license_stream);
hooks.set_license_info(license);
}

context_type ctx(instring.begin(), instring.end(), file_name.c_str(), hooks);

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
if (vm.count("c99")) {
#if BOOST_WAVE_SUPPORT_CPP0X != 0
if (vm.count("c++11")) {
cerr << "wave: multiple language options specified: --c99 "
"and --c++11" << endl;
return -1;
}
#endif
ctx.set_language(
boost::wave::language_support(
boost::wave::support_c99
|  boost::wave::support_option_convert_trigraphs
|  boost::wave::support_option_emit_line_directives
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
|  boost::wave::support_option_include_guard_detection
#endif
#if BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES != 0
|  boost::wave::support_option_emit_pragma_directives
#endif
|  boost::wave::support_option_insert_whitespace
));
}
else if (vm.count("variadics")) {
ctx.set_language(boost::wave::enable_variadics(ctx.get_language()));
}
#endif 
#if BOOST_WAVE_SUPPORT_CPP0X != 0
if (vm.count("c++11")) {
if (vm.count("c99")) {
cerr << "wave: multiple language options specified: --c99 "
"and --c++11" << endl;
return -1;
}
ctx.set_language(
boost::wave::language_support(
boost::wave::support_cpp0x
|  boost::wave::support_option_convert_trigraphs
|  boost::wave::support_option_long_long
|  boost::wave::support_option_emit_line_directives
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
|  boost::wave::support_option_include_guard_detection
#endif
#if BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES != 0
|  boost::wave::support_option_emit_pragma_directives
#endif
|  boost::wave::support_option_insert_whitespace
));
}
#endif 

#if BOOST_WAVE_SUPPORT_CPP1Z != 0
if (vm.count("c++17")) {
ctx.set_language(
boost::wave::language_support(
boost::wave::support_cpp1z
#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
|  boost::wave::support_option_has_include
#endif
|  boost::wave::support_option_convert_trigraphs
|  boost::wave::support_option_long_long
|  boost::wave::support_option_emit_line_directives
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
|  boost::wave::support_option_include_guard_detection
#endif
#if BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES != 0
|  boost::wave::support_option_emit_pragma_directives
#endif
|  boost::wave::support_option_insert_whitespace
));
}
#endif 

#if BOOST_WAVE_SUPPORT_CPP2A != 0
if (vm.count("c++20")) {
ctx.set_language(
boost::wave::language_support(
boost::wave::support_cpp2a
#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
|  boost::wave::support_option_has_include
#endif
#if BOOST_WAVE_SUPPORT_VA_OPT != 0
|  boost::wave::support_option_va_opt
#endif
|  boost::wave::support_option_convert_trigraphs
|  boost::wave::support_option_long_long
|  boost::wave::support_option_emit_line_directives
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
|  boost::wave::support_option_include_guard_detection
#endif
#if BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES != 0
|  boost::wave::support_option_emit_pragma_directives
#endif
|  boost::wave::support_option_insert_whitespace
));
}
#endif 

if (vm.count("long_long")) {
ctx.set_language(
boost::wave::enable_long_long(ctx.get_language()));
}

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
if (vm.count("noguard")) {
ctx.set_language(
boost::wave::enable_include_guard_detection(
ctx.get_language(), false));
}
#endif

if (preserve_comments) {
ctx.set_language(
boost::wave::enable_preserve_comments(ctx.get_language()));
}

if (vm.count("line")) {
int lineopt = vm["line"].as<int>();
if (0 != lineopt && 1 != lineopt && 2 != lineopt) {
cerr << "wave: bogus value for --line command line option: "
<< lineopt << endl;
return -1;
}
ctx.set_language(
boost::wave::enable_emit_line_directives(ctx.get_language(),
lineopt != 0));

if (2 == lineopt)
ctx.get_hooks().enable_relative_names_in_line_directives(true);
}

if (vm.count("disambiguate")) {
int disambiguateopt = vm["disambiguate"].as<int>();
if (0 != disambiguateopt && 1 != disambiguateopt) {
cerr << "wave: bogus value for --disambiguate command line option: "
<< disambiguateopt << endl;
return -1;
}
ctx.set_language(
boost::wave::enable_insert_whitespace(ctx.get_language(),
disambiguateopt != 0));
}

if (vm.count("sysinclude")) {
vector<std::string> syspaths = vm["sysinclude"].as<vector<std::string> >();

vector<std::string>::const_iterator end = syspaths.end();
for (vector<std::string>::const_iterator cit = syspaths.begin();
cit != end; ++cit)
{
ctx.add_sysinclude_path(cmd_line_utils::trim_quotes(*cit).c_str());
}
}

if (vm.count("include")) {
cmd_line_utils::include_paths const &ip =
vm["include"].as<cmd_line_utils::include_paths>();
vector<std::string>::const_iterator end = ip.paths.end();

for (vector<std::string>::const_iterator cit = ip.paths.begin();
cit != end; ++cit)
{
ctx.add_include_path(cmd_line_utils::trim_quotes(*cit).c_str());
}

if (ip.seen_separator)
ctx.set_sysinclude_delimiter();

vector<std::string>::const_iterator sysend = ip.syspaths.end();
for (vector<std::string>::const_iterator syscit = ip.syspaths.begin();
syscit != sysend; ++syscit)
{
ctx.add_sysinclude_path(cmd_line_utils::trim_quotes(*syscit).c_str());
}
}

if (vm.count("define")) {
vector<std::string> const &macros = vm["define"].as<vector<std::string> >();
vector<std::string>::const_iterator end = macros.end();
for (vector<std::string>::const_iterator cit = macros.begin();
cit != end; ++cit)
{
ctx.add_macro_definition(*cit);
}
}

if (vm.count("predefine")) {
vector<std::string> const &predefmacros =
vm["predefine"].as<vector<std::string> >();
vector<std::string>::const_iterator end = predefmacros.end();
for (vector<std::string>::const_iterator cit = predefmacros.begin();
cit != end; ++cit)
{
ctx.add_macro_definition(*cit, true);
}
}

if (vm.count("undefine")) {
vector<std::string> const &undefmacros =
vm["undefine"].as<vector<std::string> >();
vector<std::string>::const_iterator end = undefmacros.end();
for (vector<std::string>::const_iterator cit = undefmacros.begin();
cit != end; ++cit)
{
ctx.remove_macro_definition(*cit, true);
}
}

#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS == 0
if (vm.count("noexpand")) {
vector<std::string> const &noexpandmacros =
vm["noexpand"].as<vector<std::string> >();
vector<std::string>::const_iterator end = noexpandmacros.end();
for (vector<std::string>::const_iterator cit = noexpandmacros.begin();
cit != end; ++cit)
{
ctx.get_hooks().add_noexpandmacro(*cit);
}
}
#endif

if (vm.count("nesting")) {
int max_depth = vm["nesting"].as<int>();
if (max_depth < 1 || max_depth > 100000) {
cerr << "wave: bogus maximal include nesting depth: "
<< max_depth << endl;
return -1;
}
ctx.set_max_include_nesting_depth(max_depth);
}

if (vm.count("output")) {
fs::path out_file(boost::wave::util::create_path(
vm["output"].as<std::string>()));

if (out_file == "-") {
allow_output = false;     
default_outfile = "-";
}
else {
out_file = boost::wave::util::complete_path(out_file);
boost::wave::util::create_directories(
boost::wave::util::branch_path(out_file));
output.open(out_file.string().c_str());
if (!output.is_open()) {
cerr << "wave: could not open output file: "
<< out_file.string() << endl;
return -1;
}
if (!license.empty())
output << license;
default_outfile = out_file.string();
}
}
else if (!input_is_stdin && vm.count("autooutput")) {
fs::path out_file(boost::wave::util::create_path(file_name));
std::string basename(boost::wave::util::leaf(out_file));
std::string::size_type pos = basename.find_last_of(".");

if (std::string::npos != pos)
basename = basename.substr(0, pos);
out_file = boost::wave::util::branch_path(out_file) / (basename + ".i");

boost::wave::util::create_directories(
boost::wave::util::branch_path(out_file));
output.open(out_file.string().c_str());
if (!output.is_open()) {
cerr << "wave: could not open output file: "
<< out_file.string() << endl;
return -1;
}
if (!license.empty())
output << license;
default_outfile = out_file.string();
}

bool is_interactive = input_is_stdin && !output.is_open() && allow_output;

if (is_interactive) {
ctx.set_language(
boost::wave::enable_single_line(ctx.get_language()), false);
}

context_type::iterator_type first = ctx.begin();
context_type::iterator_type last = ctx.end();

if (vm.count("forceinclude")) {
std::vector<std::string> const& force =
vm["forceinclude"].as<std::vector<std::string> >();
std::vector<std::string>::const_reverse_iterator rend = force.rend();
for (std::vector<std::string>::const_reverse_iterator cit = force.rbegin();
cit != rend; )
{
std::string filename(*cit);
first.force_include(filename.c_str(), ++cit == rend);
}
}

elapsed_time.set_print_time(!input_is_stdin && vm.count("timer") > 0);
if (is_interactive) {
print_interactive_version();  
load_state(vm, ctx);          
}
else if (vm.count("state")) {
cerr << "wave: ignoring the command line option 'state', "
<< "use it in interactive mode only." << endl;
}

do {
bool finished = false;

if (input_is_stdin) {
if (is_interactive)
cout << ">>> ";     

instring.clear();
if (!read_a_line(instream, instring))
break;        
first = ctx.begin(instring.begin(), instring.end());
}

bool need_to_advanve = false;

do {
try {
if (need_to_advanve) {
++first;
need_to_advanve = false;
}

while (first != last) {
current_position = (*first).get_position();

if (allow_output) {
if (!output.good()) {
cerr << "wave: problem writing to the current "
<< "output file" << endl;
cerr << report_iostate_error(output.rdstate());
break;
}
if (output.is_open())
output << (*first).get_value();
else
cout << (*first).get_value();
}

++first;
}
finished = true;
}
catch (boost::wave::cpp_exception const &e) {
if (is_interactive || boost::wave::is_recoverable(e)) {
error_count += report_error_message(ctx, e,
treat_warnings_as_error);
need_to_advanve = true;   
}
else {
throw;      
}
}
catch (boost::wave::cpplexer::lexing_exception const &e) {
if (is_interactive ||
boost::wave::cpplexer::is_recoverable(e))
{
error_count +=
report_error_message(e, treat_warnings_as_error);
need_to_advanve = true;   
}
else {
throw;      
}
}
} while (!finished);
} while (input_is_stdin);

if (is_interactive)
save_state(vm, ctx);    

if (vm.count("macronames")) {
if (!list_macro_names(ctx, vm["macronames"].as<std::string>()))
return -1;
}
if (vm.count("macrocounts")) {
if (!list_macro_counts(ctx, vm["macrocounts"].as<std::string>()))
return -1;
}
}
catch (boost::wave::cpp_exception const &e) {
report_error_message(e, treat_warnings_as_error);
return 1;
}
catch (boost::wave::cpplexer::lexing_exception const &e) {
report_error_message(e, treat_warnings_as_error);
return 2;
}
catch (std::exception const &e) {
cerr
<< current_position << ": "
<< "exception caught: " << e.what()
<< endl;
return 3;
}
catch (...) {
cerr
<< current_position << ": "
<< "unexpected exception caught." << endl;
return 4;
}
return -error_count;  
}

int
main (int argc, char *argv[])
{
const std::string accepted_w_args[] = {"error"};

if (!BOOST_WAVE_TEST_CONFIGURATION()) {
cout << "wave: warning: the library this application was linked against was compiled "
<< endl
<< "               using a different configuration (see wave_config.hpp)."
<< endl;
}

try {
po::options_description desc_cmdline ("Options allowed on the command line only");

desc_cmdline.add_options()
("help,h", "print out program usage (this message)")
("version,v", "print the version number")
("copyright", "print out the copyright statement")
("config-file", po::value<vector<std::string> >()->composing(),
"specify a config file (alternatively: @filepath)")
;

const std::string w_arg_desc = "Warning settings. Currently supported: -W" +
boost::algorithm::join(accepted_w_args, ", -W");

po::options_description desc_generic ("Options allowed additionally in a config file");

desc_generic.add_options()
("output,o", po::value<std::string>(),
"specify a file [arg] to use for output instead of stdout or "
"disable output [-]")
("autooutput,E",
"output goes into a file named <input_basename>.i")
("license", po::value<std::string>(),
"prepend the content of the specified file to each created file")
("include,I", po::value<cmd_line_utils::include_paths>()->composing(),
"specify an additional include directory")
("sysinclude,S", po::value<vector<std::string> >()->composing(),
"specify an additional system include directory")
("forceinclude,F", po::value<std::vector<std::string> >()->composing(),
"force inclusion of the given file")
("define,D", po::value<std::vector<std::string> >()->composing(),
"specify a macro to define (as macro[=[value]])")
("predefine,P", po::value<std::vector<std::string> >()->composing(),
"specify a macro to predefine (as macro[=[value]])")
("undefine,U", po::value<std::vector<std::string> >()->composing(),
"specify a macro to undefine")
#if BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS == 0
("noexpand,N", po::value<std::vector<std::string> >()->composing(),
"specify a macro name, which should not be expanded")
#endif
("nesting,n", po::value<int>(),
"specify a new maximal include nesting depth")
("warning,W", po::value<std::vector<std::string> >()->composing(),
w_arg_desc.c_str())
;

po::options_description desc_ext ("Extended options (allowed everywhere)");

desc_ext.add_options()
("traceto,t", po::value<std::string>(),
"output macro expansion tracing information to a file [arg] "
"or to stderr [-]")
("timer", "output overall elapsed computing time to stderr")
("long_long", "enable long long support in C++ mode")
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
("variadics", "enable certain C99 extensions in C++ mode")
("c99", "enable C99 mode (implies --variadics)")
#endif
#if BOOST_WAVE_SUPPORT_CPP0X != 0
("c++11", "enable C++11 mode (implies --variadics and --long_long)")
#endif
#if BOOST_WAVE_SUPPORT_CPP1Z != 0
("c++17", "enable C++17 mode (implies --variadics and --long_long, adds __has_include)")
#endif
#if BOOST_WAVE_SUPPORT_CPP2A != 0
("c++20", "enable C++20 mode (implies --variadics and --long_long, adds __VA_OPT__)")
#endif
("listincludes,l", po::value<std::string>(),
"list names of included files to a file [arg] or to stdout [-]")
("macronames,m", po::value<std::string>(),
"list all defined macros to a file [arg] or to stdout [-]")
("macrocounts,c", po::value<std::string>(),
"list macro invocation counts to a file [arg] or to stdout [-]")
("preserve,p", po::value<int>()->default_value(0),
"preserve whitespace\n"
"0: no whitespace is preserved (default),\n"
"1: begin of line whitespace is preserved,\n"
"2: comments and begin of line whitespace is preserved,\n"
"3: all whitespace is preserved")
("line,L", po::value<int>()->default_value(1),
"control the generation of #line directives\n"
"0: no #line directives are generated,\n"
"1: #line directives will be emitted (default),\n"
"2: #line directives will be emitted using relative\n"
"   filenames")
("disambiguate", po::value<int>()->default_value(1),
"control whitespace insertion to disambiguate\n"
"consecutive tokens\n"
"0: no additional whitespace is generated,\n"
"1: whitespace is used to disambiguate output (default)")
("extended,x", "enable the #pragma wave system() directive")
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
("noguard,G", "disable include guard detection")
("listguards,g", po::value<std::string>(),
"list names of files flagged as 'include once' to a file [arg] "
"or to stdout [-]")
#endif
#if BOOST_WAVE_SERIALIZATION != 0
("state,s", po::value<std::string>(),
"load and save state information from/to the given file [arg] "
"or 'wave.state' [-] (interactive mode only)")
#endif
;

po::options_description desc_overall_cmdline;
po::options_description desc_overall_cfgfile;

desc_overall_cmdline.add(desc_cmdline).add(desc_generic).add(desc_ext);
desc_overall_cfgfile.add(desc_generic).add(desc_ext);

using namespace boost::program_options::command_line_style;

po::parsed_options opts(po::parse_command_line(argc, argv,
desc_overall_cmdline, unix_style, cmd_line_utils::at_option_parser));
po::variables_map vm;

po::store(opts, vm);
po::notify(vm);


vector<po::option> arguments;

std::remove_copy_if(opts.options.begin(), opts.options.end(),
back_inserter(arguments), cmd_line_utils::is_argument());

if (arguments.size() > 0 && arguments[0].value[0] != "-") {
fs::path input_dir(boost::wave::util::complete_path(
boost::wave::util::create_path(arguments[0].value[0])));

input_dir = boost::wave::util::branch_path(
boost::wave::util::normalize(input_dir));

while (!input_dir.empty()) {
fs::path filename = input_dir / "wave.cfg";
if (cmd_line_utils::read_config_file_options(filename.string(),
desc_overall_cfgfile, vm, true))
{
break;    
}
input_dir = boost::wave::util::branch_path(input_dir);
}
}

if (vm.count("config-file")) {
vector<std::string> const &cfg_files =
vm["config-file"].as<vector<std::string> >();
vector<std::string>::const_iterator end = cfg_files.end();
for (vector<std::string>::const_iterator cit = cfg_files.begin();
cit != end; ++cit)
{
cmd_line_utils::read_config_file_options(*cit,
desc_overall_cfgfile, vm);
}
}

if (vm.count("warning"))
{
BOOST_FOREACH(const std::string& arg,
vm["warning"].as<std::vector<std::string> >())
{
if (boost::range::find(accepted_w_args, arg) ==
boost::end(accepted_w_args))
{
cerr << "wave: Invalid warning setting: " << arg << endl;
return -1;
}
}
}

if (vm.count("help")) {
po::options_description desc_help(
"Usage: wave [options] [@config-file(s)] [file]");

desc_help.add(desc_cmdline).add(desc_generic).add(desc_ext);
cout << desc_help << endl;
return 1;
}

if (vm.count("version")) {
cout << get_version() << endl;
return 0;
}

if (vm.count("copyright")) {
return print_copyright();
}

if (0 == arguments.size() || 0 == arguments[0].value.size() ||
arguments[0].value[0] == "-")
{
return do_actual_work("<stdin>", std::cin, vm, true);
}
else {
if (arguments.size() > 1) {
cerr << "wave: more than one input file specified, "
<< "ignoring all but the first!" << endl;
}

std::string file_name(arguments[0].value[0]);
ifstream instream(file_name.c_str());

if (!instream.is_open()) {
cerr << "wave: could not open input file: " << file_name << endl;
return -1;
}
return do_actual_work(file_name, instream, vm, false);
}
}
catch (std::exception const &e) {
cout << "wave: exception caught: " << e.what() << endl;
return 6;
}
catch (...) {
cerr << "wave: unexpected exception caught." << endl;
return 7;
}
}
