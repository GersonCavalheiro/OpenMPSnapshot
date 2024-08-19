


#ifndef BOOST_IOSTREAMS_COMBINE_HPP_INCLUDED
#define BOOST_IOSTREAMS_COMBINE_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/config.hpp> 
#ifndef BOOST_NO_STD_LOCALE
# include <locale>
#endif
#include <boost/iostreams/detail/ios.hpp>   
#include <boost/iostreams/detail/wrap_unwrap.hpp>       
#include <boost/iostreams/traits.hpp>         
#include <boost/iostreams/operations.hpp>        
#include <boost/mpl/if.hpp>    
#include <boost/static_assert.hpp>  
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_same.hpp> 

#include <boost/iostreams/detail/config/disable_warnings.hpp>

namespace boost { namespace iostreams {

namespace detail {

template<typename Source, typename Sink>
class combined_device {
private:
typedef typename category_of<Source>::type  in_category;
typedef typename category_of<Sink>::type    out_category;
typedef typename char_type_of<Sink>::type   sink_char_type;
public:
typedef typename char_type_of<Source>::type char_type;
struct category
: bidirectional, 
device_tag, 
closable_tag, 
localizable_tag
{ };
BOOST_STATIC_ASSERT(is_device<Source>::value);
BOOST_STATIC_ASSERT(is_device<Sink>::value);
BOOST_STATIC_ASSERT((is_convertible<in_category, input>::value));
BOOST_STATIC_ASSERT((is_convertible<out_category, output>::value));
BOOST_STATIC_ASSERT((is_same<char_type, sink_char_type>::value));
combined_device(const Source& src, const Sink& snk);
std::streamsize read(char_type* s, std::streamsize n);
std::streamsize write(const char_type* s, std::streamsize n);
void close(BOOST_IOS::openmode);
#ifndef BOOST_NO_STD_LOCALE
void imbue(const std::locale& loc);
#endif
private:
Source  src_;
Sink    sink_;
};

template<typename InputFilter, typename OutputFilter>
class combined_filter {
private:
typedef typename category_of<InputFilter>::type    in_category;
typedef typename category_of<OutputFilter>::type   out_category;
typedef typename char_type_of<OutputFilter>::type  output_char_type;
public:
typedef typename char_type_of<InputFilter>::type   char_type;
struct category 
: multichar_bidirectional_filter_tag,
closable_tag, 
localizable_tag
{ };
BOOST_STATIC_ASSERT(is_filter<InputFilter>::value);
BOOST_STATIC_ASSERT(is_filter<OutputFilter>::value);
BOOST_STATIC_ASSERT((is_convertible<in_category, input>::value));
BOOST_STATIC_ASSERT((is_convertible<out_category, output>::value));
BOOST_STATIC_ASSERT((is_same<char_type, output_char_type>::value));
combined_filter(const InputFilter& in, const OutputFilter& out);

template<typename Source>
std::streamsize read(Source& src, char_type* s, std::streamsize n)
{ return boost::iostreams::read(in_, src, s, n); }

template<typename Sink>
std::streamsize write(Sink& snk, const char_type* s, std::streamsize n)
{ return boost::iostreams::write(out_, snk, s, n); }

template<typename Sink>
void close(Sink& snk, BOOST_IOS::openmode which)
{
if (which == BOOST_IOS::in) {
if (is_convertible<in_category, dual_use>::value) {
iostreams::close(in_, snk, BOOST_IOS::in);
} else {
detail::close_all(in_, snk);
}
}
if (which == BOOST_IOS::out) {
if (is_convertible<out_category, dual_use>::value) {
iostreams::close(out_, snk, BOOST_IOS::out);
} else {
detail::close_all(out_, snk);
}
}
}
#ifndef BOOST_NO_STD_LOCALE
void imbue(const std::locale& loc);
#endif
private:
InputFilter   in_;
OutputFilter  out_;
};

template<typename In, typename Out>
struct combination_traits 
: mpl::if_<
is_device<In>,
combined_device<
typename wrapped_type<In>::type,
typename wrapped_type<Out>::type
>,
combined_filter<
typename wrapped_type<In>::type,
typename wrapped_type<Out>::type
>
>
{ };

} 

template<typename In, typename Out>
struct combination : detail::combination_traits<In, Out>::type {
typedef typename detail::combination_traits<In, Out>::type  base_type;
typedef typename detail::wrapped_type<In>::type          in_type;
typedef typename detail::wrapped_type<Out>::type         out_type;
combination(const in_type& in, const out_type& out)
: base_type(in, out) { }
};

namespace detail {

template<typename In, typename Out>
struct combine_traits {
typedef combination<
BOOST_DEDUCED_TYPENAME detail::unwrapped_type<In>::type, 
BOOST_DEDUCED_TYPENAME detail::unwrapped_type<Out>::type
> type;
};

} 

template<typename In, typename Out>
typename detail::combine_traits<In, Out>::type
combine(const In& in, const Out& out) 
{ 
typedef typename detail::combine_traits<In, Out>::type return_type;
return return_type(in, out); 
}


namespace detail {


template<typename Source, typename Sink>
inline combined_device<Source, Sink>::combined_device
(const Source& src, const Sink& snk)
: src_(src), sink_(snk) { }

template<typename Source, typename Sink>
inline std::streamsize
combined_device<Source, Sink>::read(char_type* s, std::streamsize n)
{ return iostreams::read(src_, s, n); }

template<typename Source, typename Sink>
inline std::streamsize
combined_device<Source, Sink>::write(const char_type* s, std::streamsize n)
{ return iostreams::write(sink_, s, n); }

template<typename Source, typename Sink>
inline void
combined_device<Source, Sink>::close(BOOST_IOS::openmode which)
{ 
if (which == BOOST_IOS::in)
detail::close_all(src_); 
if (which == BOOST_IOS::out)
detail::close_all(sink_); 
}

#ifndef BOOST_NO_STD_LOCALE
template<typename Source, typename Sink>
void combined_device<Source, Sink>::imbue(const std::locale& loc)
{
iostreams::imbue(src_, loc);
iostreams::imbue(sink_, loc);
}
#endif


template<typename InputFilter, typename OutputFilter>
inline combined_filter<InputFilter, OutputFilter>::combined_filter
(const InputFilter& in, const OutputFilter& out) : in_(in), out_(out)
{ }

#ifndef BOOST_NO_STD_LOCALE
template<typename InputFilter, typename OutputFilter>
void combined_filter<InputFilter, OutputFilter>::imbue
(const std::locale& loc)
{
iostreams::imbue(in_, loc);
iostreams::imbue(out_, loc);
}
#endif


} 

} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp>

#endif 
