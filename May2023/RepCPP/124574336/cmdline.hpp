

#ifndef BOOST_CMDLINE_VP_2003_05_19
#define BOOST_CMDLINE_VP_2003_05_19

#include <boost/program_options/config.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>


#include <boost/detail/workaround.hpp>

#include <boost/function.hpp>

#include <string>
#include <vector>

#if defined(BOOST_MSVC)
#   pragma warning (push)
#   pragma warning (disable:4251) 
#endif

namespace boost { namespace program_options { namespace detail {


class BOOST_PROGRAM_OPTIONS_DECL cmdline {
public:

typedef ::boost::program_options::command_line_style::style_t style_t;

typedef function1<std::pair<std::string, std::string>, 
const std::string&> 
additional_parser;

typedef function1<std::vector<option>, std::vector<std::string>&>
style_parser;


cmdline(const std::vector<std::string>& args);


cmdline(int argc, const char*const * argv);

void style(int style);


int         get_canonical_option_prefix();

void allow_unregistered();

void set_options_description(const options_description& desc);
void set_positional_options(
const positional_options_description& m_positional);

std::vector<option> run();

std::vector<option> parse_long_option(std::vector<std::string>& args);
std::vector<option> parse_short_option(std::vector<std::string>& args);
std::vector<option> parse_dos_option(std::vector<std::string>& args);
std::vector<option> parse_disguised_long_option(
std::vector<std::string>& args);
std::vector<option> parse_terminator(
std::vector<std::string>& args);
std::vector<option> handle_additional_parser(
std::vector<std::string>& args);



void set_additional_parser(additional_parser p);

void extra_style_parser(style_parser s);

void check_style(int style) const;

bool is_style_active(style_t style) const;

void init(const std::vector<std::string>& args);

void
finish_option(option& opt,
std::vector<std::string>& other_tokens,
const std::vector<style_parser>& style_parsers);

std::vector<std::string> m_args;
style_t m_style;
bool m_allow_unregistered;

const options_description* m_desc;
const positional_options_description* m_positional;

additional_parser m_additional_parser;
style_parser m_style_parser;
};

void test_cmdline_detail();

}}}

#if defined(BOOST_MSVC)
#   pragma warning (pop)
#endif

#endif

