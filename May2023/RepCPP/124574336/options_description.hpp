

#ifndef BOOST_OPTION_DESCRIPTION_VP_2003_05_19
#define BOOST_OPTION_DESCRIPTION_VP_2003_05_19

#include <boost/program_options/config.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/value_semantic.hpp>

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/any.hpp>

#include <string>
#include <vector>
#include <set>
#include <map>
#include <stdexcept>
#include <utility>

#include <iosfwd>

#if defined(BOOST_MSVC)
#   pragma warning (push)
#   pragma warning (disable:4251) 
#endif



namespace boost { 

namespace program_options {


class BOOST_PROGRAM_OPTIONS_DECL option_description {
public:

option_description();


option_description(const char* name,
const value_semantic* s);


option_description(const char* name,
const value_semantic* s,
const char* description);

virtual ~option_description();

enum match_result { no_match, full_match, approximate_match };


match_result match(const std::string& option, bool approx,
bool long_ignore_case, bool short_ignore_case) const;


const std::string& key(const std::string& option) const;



std::string canonical_display_name(int canonical_option_style = 0) const;

const std::string& long_name() const;

const std::pair<const std::string*, std::size_t> long_names() const;

const std::string& description() const;

shared_ptr<const value_semantic> semantic() const;

std::string format_name() const;


std::string format_parameter() const;

private:

option_description& set_names(const char* name);


std::string m_short_name;


std::vector<std::string> m_long_names;

std::string m_description;

shared_ptr<const value_semantic> m_value_semantic;
};

class options_description;


class BOOST_PROGRAM_OPTIONS_DECL options_description_easy_init {
public:
options_description_easy_init(options_description* owner);

options_description_easy_init&
operator()(const char* name,
const char* description);

options_description_easy_init&
operator()(const char* name,
const value_semantic* s);

options_description_easy_init&
operator()(const char* name,
const value_semantic* s,
const char* description);

private:
options_description* owner;
};



class BOOST_PROGRAM_OPTIONS_DECL options_description {
public:
static const unsigned m_default_line_length;


options_description(unsigned line_length = m_default_line_length,
unsigned min_description_length = m_default_line_length / 2);

options_description(const std::string& caption,
unsigned line_length = m_default_line_length,
unsigned min_description_length = m_default_line_length / 2);

void add(shared_ptr<option_description> desc);

options_description& add(const options_description& desc);


unsigned get_option_column_width() const;

public:

options_description_easy_init add_options();

const option_description& find(const std::string& name, 
bool approx, 
bool long_ignore_case = false,
bool short_ignore_case = false) const;

const option_description* find_nothrow(const std::string& name, 
bool approx,
bool long_ignore_case = false,
bool short_ignore_case = false) const;


const std::vector< shared_ptr<option_description> >& options() const;


friend BOOST_PROGRAM_OPTIONS_DECL std::ostream& operator<<(std::ostream& os, 
const options_description& desc);


void print(std::ostream& os, unsigned width = 0) const;

private:
#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1800))
options_description& operator=(const options_description&);
#endif

typedef std::map<std::string, int>::const_iterator name2index_iterator;
typedef std::pair<name2index_iterator, name2index_iterator> 
approximation_range;


std::string m_caption;
const unsigned m_line_length;
const unsigned m_min_description_length;

std::vector< shared_ptr<option_description> > m_options;

#if BOOST_WORKAROUND(BOOST_DINKUMWARE_STDLIB, BOOST_TESTED_AT(313))
std::vector<char> belong_to_group;
#else
std::vector<bool> belong_to_group;
#endif

std::vector< shared_ptr<options_description> > groups;

};


class BOOST_PROGRAM_OPTIONS_DECL duplicate_option_error : public error {
public:
duplicate_option_error(const std::string& xwhat) : error(xwhat) {}
};
}}

#if defined(BOOST_MSVC)
#   pragma warning (pop)
#endif

#endif
