#ifndef BOOST_ARCHIVE_BINARY_IPRIMITIVE_HPP
#define BOOST_ARCHIVE_BINARY_IPRIMITIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#if defined(_MSC_VER)
#pragma warning( disable : 4800 )
#endif





#include <iosfwd>
#include <boost/assert.hpp>
#include <locale>
#include <cstring> 
#include <cstddef> 
#include <streambuf> 
#include <string>

#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::memcpy;
using ::size_t;
} 
#endif

#include <boost/cstdint.hpp>
#include <boost/serialization/throw_exception.hpp>
#include <boost/integer.hpp>
#include <boost/integer_traits.hpp>

#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/serialization/array_wrapper.hpp>

#include <boost/archive/basic_streambuf_locale_saver.hpp>
#include <boost/archive/codecvt_null.hpp>
#include <boost/archive/archive_exception.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace archive {

template<class Archive, class Elem, class Tr>
class BOOST_SYMBOL_VISIBLE basic_binary_iprimitive {
#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
friend class load_access;
protected:
#else
public:
#endif
std::basic_streambuf<Elem, Tr> & m_sb;
Archive * This(){
return static_cast<Archive *>(this);
}

#ifndef BOOST_NO_STD_LOCALE
boost::archive::codecvt_null<Elem> codecvt_null_facet;
basic_streambuf_locale_saver<Elem, Tr> locale_saver;
std::locale archive_locale;
#endif

template<class T>
void load(T & t){
load_binary(& t, sizeof(T));
}


void load(bool & t){
load_binary(& t, sizeof(t));
int i = t;
BOOST_ASSERT(0 == i || 1 == i);
(void)i; 
}
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load(std::string &s);
#ifndef BOOST_NO_STD_WSTRING
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load(std::wstring &ws);
#endif
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load(char * t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load(wchar_t * t);

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
init();
BOOST_ARCHIVE_OR_WARCHIVE_DECL
basic_binary_iprimitive(
std::basic_streambuf<Elem, Tr> & sb,
bool no_codecvt
);
BOOST_ARCHIVE_OR_WARCHIVE_DECL
~basic_binary_iprimitive();
public:
struct use_array_optimization {
template <class T>
#if defined(BOOST_NO_DEPENDENT_NESTED_DERIVATIONS)
struct apply {
typedef typename boost::serialization::is_bitwise_serializable< T >::type type;
};
#else
struct apply : public boost::serialization::is_bitwise_serializable< T > {};
#endif
};

template <class ValueType>
void load_array(serialization::array_wrapper<ValueType>& a, unsigned int)
{
load_binary(a.address(),a.count()*sizeof(ValueType));
}

void
load_binary(void *address, std::size_t count);
};

template<class Archive, class Elem, class Tr>
inline void
basic_binary_iprimitive<Archive, Elem, Tr>::load_binary(
void *address,
std::size_t count
){
BOOST_ASSERT(
static_cast<std::streamsize>(count / sizeof(Elem))
<= boost::integer_traits<std::streamsize>::const_max
);
std::streamsize s = static_cast<std::streamsize>(count / sizeof(Elem));
std::streamsize scount = m_sb.sgetn(
static_cast<Elem *>(address),
s
);
if(scount != s)
boost::serialization::throw_exception(
archive_exception(archive_exception::input_stream_error)
);
BOOST_ASSERT(count % sizeof(Elem) <= boost::integer_traits<std::streamsize>::const_max);
s = static_cast<std::streamsize>(count % sizeof(Elem));
if(0 < s){
Elem t;
scount = m_sb.sgetn(& t, 1);
if(scount != 1)
boost::serialization::throw_exception(
archive_exception(archive_exception::input_stream_error)
);
std::memcpy(static_cast<char*>(address) + (count - s), &t, static_cast<std::size_t>(s));
}
}

} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
