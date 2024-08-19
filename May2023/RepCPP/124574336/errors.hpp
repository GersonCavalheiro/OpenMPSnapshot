

#ifndef BOOST_ERRORS_VP_2003_01_02
#define BOOST_ERRORS_VP_2003_01_02

#include <boost/program_options/config.hpp>

#include <string>
#include <stdexcept>
#include <vector>
#include <map>


#if defined(BOOST_MSVC)
#   pragma warning (push)
#   pragma warning (disable:4275) 
#   pragma warning (disable:4251) 
#endif

namespace boost { namespace program_options {

inline std::string strip_prefixes(const std::string& text)
{
std::string::size_type i = text.find_first_not_of("-/");
if (i == std::string::npos) {
return text;
} else {
return text.substr(i);
}
}


class BOOST_PROGRAM_OPTIONS_DECL error : public std::logic_error {
public:
error(const std::string& xwhat) : std::logic_error(xwhat) {}
};



class BOOST_PROGRAM_OPTIONS_DECL too_many_positional_options_error : public error {
public:
too_many_positional_options_error() 
: error("too many positional options have been specified on the command line") 
{}
};


class BOOST_PROGRAM_OPTIONS_DECL invalid_command_line_style : public error {
public:
invalid_command_line_style(const std::string& msg)
: error(msg)
{}
};


class BOOST_PROGRAM_OPTIONS_DECL reading_file : public error {
public:
reading_file(const char* filename)
: error(std::string("can not read options configuration file '").append(filename).append("'"))
{}
};



class BOOST_PROGRAM_OPTIONS_DECL error_with_option_name : public error {

protected:

int m_option_style;



std::map<std::string, std::string> m_substitutions;
typedef std::pair<std::string, std::string> string_pair;
std::map<std::string, string_pair > m_substitution_defaults;

public:

std::string m_error_template;

error_with_option_name(const std::string& template_,
const std::string& option_name = "",
const std::string& original_token = "",
int option_style = 0);


~error_with_option_name() throw() {}




void set_substitute(const std::string& parameter_name,  const std::string& value)
{           m_substitutions[parameter_name] = value;    }


void set_substitute_default(const std::string& parameter_name, 
const std::string& from,  
const std::string& to)
{           
m_substitution_defaults[parameter_name] = std::make_pair(from, to); 
}



void add_context(const std::string& option_name,
const std::string& original_token,
int option_style)
{
set_option_name(option_name);
set_original_token(original_token);
set_prefix(option_style);
}

void set_prefix(int option_style)
{           m_option_style = option_style;}


virtual void set_option_name(const std::string& option_name)
{           set_substitute("option", option_name);}

std::string get_option_name() const
{           return get_canonical_option_name();         }

void set_original_token(const std::string& original_token)
{           set_substitute("original_token", original_token);}



virtual const char* what() const throw();

protected:

mutable std::string m_message;  


virtual void substitute_placeholders(const std::string& error_template) const;

void replace_token(const std::string& from, const std::string& to) const;


std::string get_canonical_option_name() const;
std::string get_canonical_option_prefix() const;
};



class BOOST_PROGRAM_OPTIONS_DECL multiple_values : public error_with_option_name {
public:
multiple_values() 
: error_with_option_name("option '%canonical_option%' only takes a single argument"){}

~multiple_values() throw() {}
};


class BOOST_PROGRAM_OPTIONS_DECL multiple_occurrences : public error_with_option_name {
public:
multiple_occurrences() 
: error_with_option_name("option '%canonical_option%' cannot be specified more than once"){}

~multiple_occurrences() throw() {}

};


class BOOST_PROGRAM_OPTIONS_DECL required_option : public error_with_option_name {
public:
required_option(const std::string& option_name)
: error_with_option_name("the option '%canonical_option%' is required but missing", "", option_name)
{
}

~required_option() throw() {}
};


class BOOST_PROGRAM_OPTIONS_DECL error_with_no_option_name : public error_with_option_name {
public:
error_with_no_option_name(const std::string& template_,
const std::string& original_token = "")
: error_with_option_name(template_, "", original_token)
{
}


virtual void set_option_name(const std::string&) {}

~error_with_no_option_name() throw() {}
};



class BOOST_PROGRAM_OPTIONS_DECL unknown_option : public error_with_no_option_name {
public:
unknown_option(const std::string& original_token = "")
: error_with_no_option_name("unrecognised option '%canonical_option%'", original_token)
{
}

~unknown_option() throw() {}
};




class BOOST_PROGRAM_OPTIONS_DECL ambiguous_option : public error_with_no_option_name {
public:
ambiguous_option(const std::vector<std::string>& xalternatives)
: error_with_no_option_name("option '%canonical_option%' is ambiguous"),
m_alternatives(xalternatives)
{}

~ambiguous_option() throw() {}

const std::vector<std::string>& alternatives() const throw() {return m_alternatives;}

protected:

virtual void substitute_placeholders(const std::string& error_template) const;
private:
std::vector<std::string> m_alternatives;
};



class BOOST_PROGRAM_OPTIONS_DECL invalid_syntax : public error_with_option_name {
public:
enum kind_t {
long_not_allowed = 30,
long_adjacent_not_allowed,
short_adjacent_not_allowed,
empty_adjacent_parameter,
missing_parameter,
extra_parameter,
unrecognized_line
};

invalid_syntax(kind_t kind, 
const std::string& option_name = "",
const std::string& original_token = "",
int option_style              = 0):
error_with_option_name(get_template(kind), option_name, original_token, option_style),
m_kind(kind)
{
}

~invalid_syntax() throw() {}

kind_t kind() const {return m_kind;}


virtual std::string tokens() const {return get_option_name();   }
protected:

std::string get_template(kind_t kind);
kind_t m_kind;
};

class BOOST_PROGRAM_OPTIONS_DECL invalid_config_file_syntax : public invalid_syntax {
public:
invalid_config_file_syntax(const std::string& invalid_line, kind_t kind):
invalid_syntax(kind)
{
m_substitutions["invalid_line"] = invalid_line;
}

~invalid_config_file_syntax() throw() {}


virtual std::string tokens() const {return m_substitutions.find("invalid_line")->second;    }
};



class BOOST_PROGRAM_OPTIONS_DECL invalid_command_line_syntax : public invalid_syntax {
public:
invalid_command_line_syntax(kind_t kind,
const std::string& option_name = "",
const std::string& original_token = "",
int option_style              = 0):
invalid_syntax(kind, option_name, original_token, option_style) {}
~invalid_command_line_syntax() throw() {}
};



class BOOST_PROGRAM_OPTIONS_DECL validation_error : public error_with_option_name {
public:
enum kind_t {
multiple_values_not_allowed = 30,
at_least_one_value_required, 
invalid_bool_value,
invalid_option_value,
invalid_option
};

public:
validation_error(kind_t kind, 
const std::string& option_name = "",
const std::string& original_token = "",
int option_style              = 0):
error_with_option_name(get_template(kind), option_name, original_token, option_style),
m_kind(kind)
{
}

~validation_error() throw() {}

kind_t kind() const { return m_kind; }

protected:

std::string get_template(kind_t kind);
kind_t m_kind;
};


class BOOST_PROGRAM_OPTIONS_DECL invalid_option_value 
: public validation_error
{
public:
invalid_option_value(const std::string& value);
#ifndef BOOST_NO_STD_WSTRING
invalid_option_value(const std::wstring& value);
#endif
};


class BOOST_PROGRAM_OPTIONS_DECL invalid_bool_value 
: public validation_error
{
public:
invalid_bool_value(const std::string& value);
};







}}

#if defined(BOOST_MSVC)
#   pragma warning (pop)
#endif

#endif
