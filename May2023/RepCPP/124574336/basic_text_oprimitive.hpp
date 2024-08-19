#ifndef BOOST_ARCHIVE_BASIC_TEXT_OPRIMITIVE_HPP
#define BOOST_ARCHIVE_BASIC_TEXT_OPRIMITIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <iomanip>
#include <locale>
#include <cstddef> 

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/io/ios_state.hpp>

#include <boost/detail/workaround.hpp>
#if BOOST_WORKAROUND(BOOST_DINKUMWARE_STDLIB, == 1)
#include <boost/archive/dinkumware.hpp>
#endif

#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
#if ! defined(BOOST_DINKUMWARE_STDLIB) && ! defined(__SGI_STL_PORT)
using ::locale;
#endif
} 
#endif

#include <boost/type_traits/is_floating_point.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/limits.hpp>
#include <boost/integer.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/serialization/throw_exception.hpp>
#include <boost/archive/basic_streambuf_locale_saver.hpp>
#include <boost/archive/codecvt_null.hpp>
#include <boost/archive/archive_exception.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace archive {

template<class OStream>
class BOOST_SYMBOL_VISIBLE basic_text_oprimitive
{
protected:
OStream &os;
io::ios_flags_saver flags_saver;
io::ios_precision_saver precision_saver;

#ifndef BOOST_NO_STD_LOCALE
boost::archive::codecvt_null<typename OStream::char_type> codecvt_null_facet;
std::locale archive_locale;
basic_ostream_locale_saver<
typename OStream::char_type,
typename OStream::traits_type
> locale_saver;
#endif

void save(const bool t){
BOOST_ASSERT(0 == static_cast<int>(t) || 1 == static_cast<int>(t));
if(os.fail())
boost::serialization::throw_exception(
archive_exception(archive_exception::output_stream_error)
);
os << t;
}
void save(const signed char t)
{
save(static_cast<short int>(t));
}
void save(const unsigned char t)
{
save(static_cast<short unsigned int>(t));
}
void save(const char t)
{
save(static_cast<short int>(t));
}
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
void save(const wchar_t t)
{
BOOST_STATIC_ASSERT(sizeof(wchar_t) <= sizeof(int));
save(static_cast<int>(t));
}
#endif


template<class T>
void save_impl(const T &t, boost::mpl::bool_<false> &){
if(os.fail())
boost::serialization::throw_exception(
archive_exception(archive_exception::output_stream_error)
);
os << t;
}


template<class T>
struct is_float {
typedef typename mpl::bool_<
boost::is_floating_point<T>::value
|| (std::numeric_limits<T>::is_specialized
&& !std::numeric_limits<T>::is_integer
&& !std::numeric_limits<T>::is_exact
&& std::numeric_limits<T>::max_exponent)
>::type type;
};

template<class T>
void save_impl(const T &t, boost::mpl::bool_<true> &){
if(os.fail()){
boost::serialization::throw_exception(
archive_exception(archive_exception::output_stream_error)
);
}
#ifndef BOOST_NO_CXX11_NUMERIC_LIMITS
const unsigned int digits = std::numeric_limits<T>::max_digits10;
#else
const unsigned int digits = std::numeric_limits<T>::digits10 + 2;
#endif
os << std::setprecision(digits) << std::scientific << t;
}

template<class T>
void save(const T & t){
typename is_float<T>::type tf;
save_impl(t, tf);
}

BOOST_ARCHIVE_OR_WARCHIVE_DECL
basic_text_oprimitive(OStream & os, bool no_codecvt);
BOOST_ARCHIVE_OR_WARCHIVE_DECL
~basic_text_oprimitive();
public:
void put(typename OStream::char_type c){
if(os.fail())
boost::serialization::throw_exception(
archive_exception(archive_exception::output_stream_error)
);
os.put(c);
}
void put(const char * s){
while('\0' != *s)
os.put(*s++);
}
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_binary(const void *address, std::size_t count);
};

} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
