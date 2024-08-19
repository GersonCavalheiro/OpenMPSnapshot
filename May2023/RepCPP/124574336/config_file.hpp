

#ifndef BOOST_CONFIG_FILE_VP_2003_01_02
#define BOOST_CONFIG_FILE_VP_2003_01_02

#include <iosfwd>
#include <string>
#include <set>

#include <boost/noncopyable.hpp>
#include <boost/program_options/config.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/eof_iterator.hpp>

#include <boost/detail/workaround.hpp>
#include <boost/program_options/detail/convert.hpp>

#if BOOST_WORKAROUND(__DECCXX_VER, BOOST_TESTED_AT(60590042))
#include <istream> 
#endif

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/shared_ptr.hpp>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4251) 
#endif



namespace boost { namespace program_options { namespace detail {


class BOOST_PROGRAM_OPTIONS_DECL common_config_file_iterator
: public eof_iterator<common_config_file_iterator, option>
{
public:
common_config_file_iterator() { found_eof(); }
common_config_file_iterator(
const std::set<std::string>& allowed_options,
bool allow_unregistered = false);

virtual ~common_config_file_iterator() {}

public: 

void get();

#if BOOST_WORKAROUND(_MSC_VER, <= 1900)
void decrement() {}
void advance(difference_type) {}
#endif

protected: 

virtual bool getline(std::string&) { return false; }

private:

void add_option(const char* name);

bool allowed_option(const std::string& s) const; 

std::set<std::string> allowed_options;
std::set<std::string> allowed_prefixes;
std::string m_prefix;
bool m_allow_unregistered;
};

template<class charT>
class basic_config_file_iterator : public common_config_file_iterator {
public:
basic_config_file_iterator()
{
found_eof();
}


basic_config_file_iterator(std::basic_istream<charT>& is, 
const std::set<std::string>& allowed_options,
bool allow_unregistered = false); 

private: 

bool getline(std::string&);

private: 
shared_ptr<std::basic_istream<charT> > is;
};

typedef basic_config_file_iterator<char> config_file_iterator;
typedef basic_config_file_iterator<wchar_t> wconfig_file_iterator;


struct null_deleter
{
void operator()(void const *) const {}
};


template<class charT>
basic_config_file_iterator<charT>::
basic_config_file_iterator(std::basic_istream<charT>& is, 
const std::set<std::string>& allowed_options,
bool allow_unregistered)
: common_config_file_iterator(allowed_options, allow_unregistered)
{
this->is.reset(&is, null_deleter());                 
get();
}

template<class charT>
bool
basic_config_file_iterator<charT>::getline(std::string& s)
{
std::basic_string<charT> in;
if (std::getline(*is, in)) {
s = to_internal(in);
return true;
} else {
return false;
}
}

#if BOOST_WORKAROUND(__COMO_VERSION__, BOOST_TESTED_AT(4303)) || \
(defined(__sgi) && BOOST_WORKAROUND(_COMPILER_VERSION, BOOST_TESTED_AT(741)))
template<>
bool
basic_config_file_iterator<wchar_t>::getline(std::string& s);
#endif



}}}

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#endif
