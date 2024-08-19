

#ifndef BOOST_VARIABLES_MAP_VP_2003_05_19
#define BOOST_VARIABLES_MAP_VP_2003_05_19

#include <boost/program_options/config.hpp>

#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>

#include <string>
#include <map>
#include <set>

#if defined(BOOST_MSVC)
#   pragma warning (push)
#   pragma warning (disable:4251) 
#endif

namespace boost { namespace program_options {

template<class charT>
class basic_parsed_options;

class value_semantic;
class variables_map;



BOOST_PROGRAM_OPTIONS_DECL
void store(const basic_parsed_options<char>& options, variables_map& m,
bool utf8 = false);


BOOST_PROGRAM_OPTIONS_DECL
void store(const basic_parsed_options<wchar_t>& options,
variables_map& m);



BOOST_PROGRAM_OPTIONS_DECL void notify(variables_map& m);


class BOOST_PROGRAM_OPTIONS_DECL variable_value {
public:
variable_value() : m_defaulted(false) {}
variable_value(const boost::any& xv, bool xdefaulted)
: v(xv), m_defaulted(xdefaulted)
{}


template<class T>
const T& as() const {
return boost::any_cast<const T&>(v);
}

template<class T>
T& as() {
return boost::any_cast<T&>(v);
}

bool empty() const;

bool defaulted() const;

const boost::any& value() const;


boost::any& value();
private:
boost::any v;
bool m_defaulted;
shared_ptr<const value_semantic> m_value_semantic;

friend BOOST_PROGRAM_OPTIONS_DECL
void store(const basic_parsed_options<char>& options,
variables_map& m, bool);

friend class BOOST_PROGRAM_OPTIONS_DECL variables_map;
};


class BOOST_PROGRAM_OPTIONS_DECL abstract_variables_map {
public:
abstract_variables_map();
abstract_variables_map(const abstract_variables_map* next);

virtual ~abstract_variables_map() {}


const variable_value& operator[](const std::string& name) const;


void next(abstract_variables_map* next);

private:

virtual const variable_value& get(const std::string& name) const = 0;

const abstract_variables_map* m_next;
};


class BOOST_PROGRAM_OPTIONS_DECL variables_map : public abstract_variables_map,
public std::map<std::string, variable_value>
{
public:
variables_map();
variables_map(const abstract_variables_map* next);

const variable_value& operator[](const std::string& name) const
{ return abstract_variables_map::operator[](name); }

void clear();

void notify();

private:

const variable_value& get(const std::string& name) const;


std::set<std::string> m_final;

friend BOOST_PROGRAM_OPTIONS_DECL
void store(const basic_parsed_options<char>& options,
variables_map& xm,
bool utf8);


std::map<std::string, std::string> m_required;
};




inline bool
variable_value::empty() const
{
return v.empty();
}

inline bool
variable_value::defaulted() const
{
return m_defaulted;
}

inline
const boost::any&
variable_value::value() const
{
return v;
}

inline
boost::any&
variable_value::value()
{
return v;
}

}}

#if defined(BOOST_MSVC)
#   pragma warning (pop)
#endif

#endif
