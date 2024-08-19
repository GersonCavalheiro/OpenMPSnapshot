

#ifndef BOOST_PARSERS_VP_2003_05_19
#define BOOST_PARSERS_VP_2003_05_19

#include <boost/program_options/config.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/detail/cmdline.hpp>

#include <boost/function/function1.hpp>

#include <iosfwd>
#include <vector>
#include <utility>

#if defined(BOOST_MSVC)
#   pragma warning (push)
#   pragma warning (disable:4251) 
#endif

namespace boost { namespace program_options {

class options_description;
class positional_options_description;



template<class charT>
class basic_parsed_options {
public:
explicit basic_parsed_options(const options_description* xdescription, int options_prefix = 0)
: description(xdescription), m_options_prefix(options_prefix) {}

std::vector< basic_option<charT> > options;

const options_description* description;


int m_options_prefix;
};


template<>
class BOOST_PROGRAM_OPTIONS_DECL basic_parsed_options<wchar_t> {
public:

explicit basic_parsed_options(const basic_parsed_options<char>& po);

std::vector< basic_option<wchar_t> > options;
const options_description* description;


basic_parsed_options<char> utf8_encoded_options;


int m_options_prefix;
};

typedef basic_parsed_options<char> parsed_options;
typedef basic_parsed_options<wchar_t> wparsed_options;




typedef function1<std::pair<std::string, std::string>, const std::string&> ext_parser;


template<class charT>
class basic_command_line_parser : private detail::cmdline {
public:

basic_command_line_parser(const std::vector<
std::basic_string<charT> >& args);

basic_command_line_parser(int argc, const charT* const argv[]);


basic_command_line_parser& options(const options_description& desc);

basic_command_line_parser& positional(
const positional_options_description& desc);


basic_command_line_parser& style(int);

basic_command_line_parser& extra_parser(ext_parser);


basic_parsed_options<charT> run();


basic_command_line_parser& allow_unregistered();

using detail::cmdline::style_parser;

basic_command_line_parser& extra_style_parser(style_parser s);

private:
const options_description* m_desc;
};

typedef basic_command_line_parser<char> command_line_parser;
typedef basic_command_line_parser<wchar_t> wcommand_line_parser;


template<class charT>
basic_parsed_options<charT>
parse_command_line(int argc, const charT* const argv[],
const options_description&,
int style = 0,
function1<std::pair<std::string, std::string>,
const std::string&> ext
= ext_parser());


template<class charT>
#if ! BOOST_WORKAROUND(__ICL, BOOST_TESTED_AT(700))
BOOST_PROGRAM_OPTIONS_DECL
#endif
basic_parsed_options<charT>
parse_config_file(std::basic_istream<charT>&, const options_description&,
bool allow_unregistered = false);


#ifdef BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
template<class charT>
#else
template<class charT = char>
#endif
#if ! BOOST_WORKAROUND(__ICL, BOOST_TESTED_AT(700))
BOOST_PROGRAM_OPTIONS_DECL
#endif
basic_parsed_options<charT>
parse_config_file(const char* filename, const options_description&,
bool allow_unregistered = false);


enum collect_unrecognized_mode
{ include_positional, exclude_positional };


template<class charT>
std::vector< std::basic_string<charT> >
collect_unrecognized(const std::vector< basic_option<charT> >& options,
enum collect_unrecognized_mode mode);


BOOST_PROGRAM_OPTIONS_DECL parsed_options
parse_environment(const options_description&,
const function1<std::string, std::string>& name_mapper);


BOOST_PROGRAM_OPTIONS_DECL parsed_options
parse_environment(const options_description&, const std::string& prefix);


BOOST_PROGRAM_OPTIONS_DECL parsed_options
parse_environment(const options_description&, const char* prefix);


BOOST_PROGRAM_OPTIONS_DECL std::vector<std::string>
split_unix(const std::string& cmdline, const std::string& seperator = " \t",
const std::string& quote = "'\"", const std::string& escape = "\\");

#ifndef BOOST_NO_STD_WSTRING

BOOST_PROGRAM_OPTIONS_DECL std::vector<std::wstring>
split_unix(const std::wstring& cmdline, const std::wstring& seperator = L" \t",
const std::wstring& quote = L"'\"", const std::wstring& escape = L"\\");
#endif

#ifdef _WIN32

BOOST_PROGRAM_OPTIONS_DECL std::vector<std::string>
split_winmain(const std::string& cmdline);

#ifndef BOOST_NO_STD_WSTRING

BOOST_PROGRAM_OPTIONS_DECL std::vector<std::wstring>
split_winmain(const std::wstring& cmdline);
#endif
#endif


}}

#if defined(BOOST_MSVC)
#   pragma warning (pop)
#endif

#undef DECL

#include "boost/program_options/detail/parsers.hpp"

#endif
