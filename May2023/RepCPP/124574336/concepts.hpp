

#ifndef BOOST_IOSTREAMS_CONCEPTS_HPP_INCLUDED
#define BOOST_IOSTREAMS_CONCEPTS_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>  
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/detail/default_arg.hpp>
#include <boost/iostreams/detail/ios.hpp>  
#include <boost/iostreams/positioning.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_convertible.hpp>

namespace boost { namespace iostreams {


template<typename Mode, typename Ch = char>
struct device {
typedef Ch char_type;
struct category
: Mode,
device_tag,
closable_tag,
localizable_tag
{ };

void close()
{
using namespace detail;
BOOST_STATIC_ASSERT((!is_convertible<Mode, two_sequence>::value));
}

void close(BOOST_IOS::openmode)
{
using namespace detail;
BOOST_STATIC_ASSERT((is_convertible<Mode, two_sequence>::value));
}

template<typename Locale>
void imbue(const Locale&) { }
};

template<typename Mode, typename Ch = wchar_t>
struct wdevice : device<Mode, Ch> { };

typedef device<input>    source;
typedef wdevice<input>   wsource;
typedef device<output>   sink;
typedef wdevice<output>  wsink;


template<typename Mode, typename Ch = char>
struct filter {
typedef Ch char_type;
struct category
: Mode,
filter_tag,
closable_tag,
localizable_tag
{ };

template<typename Device>
void close(Device&)
{
using namespace detail;
BOOST_STATIC_ASSERT((!is_convertible<Mode, two_sequence>::value));
BOOST_STATIC_ASSERT((!is_convertible<Mode, dual_use>::value));
}

template<typename Device>
void close(Device&, BOOST_IOS::openmode)
{
using namespace detail;
BOOST_STATIC_ASSERT(
(is_convertible<Mode, two_sequence>::value) ||
(is_convertible<Mode, dual_use>::value)
);
}

template<typename Locale>
void imbue(const Locale&) { }
};

template<typename Mode, typename Ch = wchar_t>
struct wfilter : filter<Mode, Ch> { };

typedef filter<input>      input_filter;
typedef wfilter<input>     input_wfilter;
typedef filter<output>     output_filter;
typedef wfilter<output>    output_wfilter;
typedef filter<seekable>   seekable_filter;
typedef wfilter<seekable>  seekable_wfilter;
typedef filter<dual_use>   dual_use_filter;
typedef wfilter<dual_use>  dual_use_wfilter;


template<typename Mode, typename Ch = char>
struct multichar_filter : filter<Mode, Ch> {
struct category : filter<Mode, Ch>::category, multichar_tag { };
};

template<typename Mode, typename Ch = wchar_t>
struct multichar_wfilter : multichar_filter<Mode, Ch> { };

typedef multichar_filter<input>      multichar_input_filter;
typedef multichar_wfilter<input>     multichar_input_wfilter;
typedef multichar_filter<output>     multichar_output_filter;
typedef multichar_wfilter<output>    multichar_output_wfilter;
typedef multichar_filter<dual_use>   multichar_dual_use_filter;
typedef multichar_wfilter<dual_use>  multichar_dual_use_wfilter;


} } 

#endif 
