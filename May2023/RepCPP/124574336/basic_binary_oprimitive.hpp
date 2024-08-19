#ifndef BOOST_ARCHIVE_BASIC_BINARY_OPRIMITIVE_HPP
#define BOOST_ARCHIVE_BASIC_BINARY_OPRIMITIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif






#include <iosfwd>
#include <boost/assert.hpp>
#include <locale>
#include <streambuf> 
#include <string>
#include <cstddef> 

#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#include <boost/cstdint.hpp>
#include <boost/integer.hpp>
#include <boost/integer_traits.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/throw_exception.hpp>

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
class BOOST_SYMBOL_VISIBLE basic_binary_oprimitive {
#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
friend class save_access;
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
void save(const T & t)
{
save_binary(& t, sizeof(T));
}


void save(const bool t){
BOOST_ASSERT(0 == static_cast<int>(t) || 1 == static_cast<int>(t));
save_binary(& t, sizeof(t));
}
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save(const std::string &s);
#ifndef BOOST_NO_STD_WSTRING
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save(const std::wstring &ws);
#endif
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save(const char * t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save(const wchar_t * t);

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
init();

BOOST_ARCHIVE_OR_WARCHIVE_DECL
basic_binary_oprimitive(
std::basic_streambuf<Elem, Tr> & sb,
bool no_codecvt
);
BOOST_ARCHIVE_OR_WARCHIVE_DECL
~basic_binary_oprimitive();
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
void save_array(boost::serialization::array_wrapper<ValueType> const& a, unsigned int)
{
save_binary(a.address(),a.count()*sizeof(ValueType));
}

void save_binary(const void *address, std::size_t count);
};

template<class Archive, class Elem, class Tr>
inline void
basic_binary_oprimitive<Archive, Elem, Tr>::save_binary(
const void *address,
std::size_t count
){
count = ( count + sizeof(Elem) - 1) / sizeof(Elem);
std::streamsize scount = m_sb.sputn(
static_cast<const Elem *>(address),
static_cast<std::streamsize>(count)
);
if(count != static_cast<std::size_t>(scount))
boost::serialization::throw_exception(
archive_exception(archive_exception::output_stream_error)
);
}

} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
