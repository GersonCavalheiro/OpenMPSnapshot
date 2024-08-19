#ifndef BOOST_ARCHIVE_BASIC_TEXT_IPRIMITIVE_HPP
#define BOOST_ARCHIVE_BASIC_TEXT_IPRIMITIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <locale>
#include <cstddef> 

#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
#if ! defined(BOOST_DINKUMWARE_STDLIB) && ! defined(__SGI_STL_PORT)
using ::locale;
#endif
} 
#endif

#include <boost/io/ios_state.hpp>
#include <boost/static_assert.hpp>

#include <boost/detail/workaround.hpp>
#if BOOST_WORKAROUND(BOOST_DINKUMWARE_STDLIB, == 1)
#include <boost/archive/dinkumware.hpp>
#endif
#include <boost/serialization/throw_exception.hpp>
#include <boost/archive/codecvt_null.hpp>
#include <boost/archive/archive_exception.hpp>
#include <boost/archive/basic_streambuf_locale_saver.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace archive {

#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4244 4267 )
#endif

template<class IStream>
class BOOST_SYMBOL_VISIBLE basic_text_iprimitive {
protected:
IStream &is;
io::ios_flags_saver flags_saver;
io::ios_precision_saver precision_saver;

#ifndef BOOST_NO_STD_LOCALE
boost::archive::codecvt_null<typename IStream::char_type> codecvt_null_facet;
std::locale archive_locale;
basic_istream_locale_saver<
typename IStream::char_type,
typename IStream::traits_type
> locale_saver;
#endif

template<class T>
void load(T & t)
{
if(is >> t)
return;
boost::serialization::throw_exception(
archive_exception(archive_exception::input_stream_error)
);
}

void load(char & t)
{
short int i;
load(i);
t = i;
}
void load(signed char & t)
{
short int i;
load(i);
t = i;
}
void load(unsigned char & t)
{
unsigned short int i;
load(i);
t = i;
}

#ifndef BOOST_NO_INTRINSIC_WCHAR_T
void load(wchar_t & t)
{
BOOST_STATIC_ASSERT(sizeof(wchar_t) <= sizeof(int));
int i;
load(i);
t = i;
}
#endif
BOOST_ARCHIVE_OR_WARCHIVE_DECL
basic_text_iprimitive(IStream  &is, bool no_codecvt);
BOOST_ARCHIVE_OR_WARCHIVE_DECL
~basic_text_iprimitive();
public:
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_binary(void *address, std::size_t count);
};

#if defined(_MSC_VER)
#pragma warning( pop )
#endif

} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
