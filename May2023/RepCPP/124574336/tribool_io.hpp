
#ifndef BOOST_LOGIC_TRIBOOL_IO_HPP
#define BOOST_LOGIC_TRIBOOL_IO_HPP

#include <boost/logic/tribool.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/noncopyable.hpp>

#if defined(_MSC_VER)
#  pragma once
#endif

#ifndef BOOST_NO_STD_LOCALE
#  include <locale>
#endif

#include <string>
#include <iostream>

namespace boost { namespace logic {

#ifdef BOOST_NO_STD_LOCALE


template<typename T> std::basic_string<T> default_false_name();


template<>
inline std::basic_string<char> default_false_name<char>()
{ return "false"; }

#  if !defined(BOOST_NO_CWCHAR)

template<>
inline std::basic_string<wchar_t> default_false_name<wchar_t>()
{ return L"false"; }
#  endif


template<typename T> std::basic_string<T> default_true_name();


template<>
inline std::basic_string<char> default_true_name<char>()
{ return "true"; }

#  if !defined(BOOST_NO_CWCHAR)

template<>
inline std::basic_string<wchar_t> default_true_name<wchar_t>()
{ return L"true"; }
#  endif
#endif


template<typename T> std::basic_string<T> get_default_indeterminate_name();

template<>
inline std::basic_string<char> get_default_indeterminate_name<char>()
{ return "indeterminate"; }

#if !defined(BOOST_NO_CWCHAR)
template<>
inline std::basic_string<wchar_t> get_default_indeterminate_name<wchar_t>()
{ return L"indeterminate"; }
#endif


#ifndef BOOST_NO_STD_LOCALE

template<typename CharT>
class indeterminate_name : public std::locale::facet, private boost::noncopyable
{
public:
typedef CharT char_type;
typedef std::basic_string<CharT> string_type;

indeterminate_name() : name_(get_default_indeterminate_name<CharT>()) {}

explicit indeterminate_name(const string_type& initial_name)
: name_(initial_name) {}

string_type name() const { return name_; }

static std::locale::id id;

private:
string_type name_;
};

template<typename CharT> std::locale::id indeterminate_name<CharT>::id;
#endif


template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, tribool x)
{
if (!indeterminate(x)) {
out << static_cast<bool>(x);
} else {
typename std::basic_ostream<CharT, Traits>::sentry cerberus(out);
if (cerberus) {
if (out.flags() & std::ios_base::boolalpha) {
#ifndef BOOST_NO_STD_LOCALE
if (BOOST_HAS_FACET(indeterminate_name<CharT>, out.getloc())) {
const indeterminate_name<CharT>& facet =
BOOST_USE_FACET(indeterminate_name<CharT>, out.getloc());
out << facet.name();
} else {
out << get_default_indeterminate_name<CharT>();
}
#else
out << get_default_indeterminate_name<CharT>();
#endif
}
else
out << 2;
}
}
return out;
}


template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, 
bool (*)(tribool, detail::indeterminate_t))
{ return out << tribool(indeterminate); } 


template<typename CharT, typename Traits>
inline std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& in, tribool& x)
{
if (in.flags() & std::ios_base::boolalpha) {
typename std::basic_istream<CharT, Traits>::sentry cerberus(in);
if (cerberus) {
typedef std::basic_string<CharT> string_type;

#ifndef BOOST_NO_STD_LOCALE
const std::numpunct<CharT>& numpunct_facet =
BOOST_USE_FACET(std::numpunct<CharT>, in.getloc());

string_type falsename = numpunct_facet.falsename();
string_type truename = numpunct_facet.truename();

string_type othername;
if (BOOST_HAS_FACET(indeterminate_name<CharT>, in.getloc())) {
othername =
BOOST_USE_FACET(indeterminate_name<CharT>, in.getloc()).name();
} else {
othername = get_default_indeterminate_name<CharT>();
}
#else
string_type falsename = default_false_name<CharT>();
string_type truename = default_true_name<CharT>();
string_type othername = get_default_indeterminate_name<CharT>();
#endif

typename string_type::size_type pos = 0;
bool falsename_ok = true, truename_ok = true, othername_ok = true;

while ((falsename_ok && pos < falsename.size())
|| (truename_ok && pos < truename.size())
|| (othername_ok && pos < othername.size())) {
typename Traits::int_type c = in.get();
if (c == Traits::eof())
return in;

bool matched = false;
if (falsename_ok && pos < falsename.size()) {
if (Traits::eq(Traits::to_char_type(c), falsename[pos]))
matched = true;
else
falsename_ok = false;
}

if (truename_ok && pos < truename.size()) {
if (Traits::eq(Traits::to_char_type(c), truename[pos]))
matched = true;
else
truename_ok = false;
}

if (othername_ok && pos < othername.size()) {
if (Traits::eq(Traits::to_char_type(c), othername[pos]))
matched = true;
else
othername_ok = false;
}

if (matched) { ++pos; }
if (pos > falsename.size()) falsename_ok = false;
if (pos > truename.size())  truename_ok = false;
if (pos > othername.size()) othername_ok = false;
}

if (pos == 0)
in.setstate(std::ios_base::failbit);
else {
if (falsename_ok)      x = false;
else if (truename_ok)  x = true;
else if (othername_ok) x = indeterminate;
else in.setstate(std::ios_base::failbit);
}
}
} else {
long value;
if (in >> value) {
switch (value) {
case 0: x = false; break;
case 1: x = true; break;
case 2: x = indeterminate; break;
default: in.setstate(std::ios_base::failbit); break;
}
}
}

return in;
}

} } 

#endif 
