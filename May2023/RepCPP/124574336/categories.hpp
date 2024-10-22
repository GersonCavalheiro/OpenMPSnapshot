


#ifndef BOOST_IOSTREAMS_CATEGORIES_HPP_INCLUDED
#define BOOST_IOSTREAMS_CATEGORIES_HPP_INCLUDED 

#if defined(_MSC_VER)
# pragma once
#endif

namespace boost { namespace iostreams {


struct any_tag { };
namespace detail { struct two_sequence : virtual any_tag { }; }
namespace detail { struct random_access : virtual any_tag { }; }
namespace detail { struct one_head : virtual any_tag { }; }
namespace detail { struct two_head : virtual any_tag { }; }
struct input : virtual any_tag { };
struct output : virtual any_tag { };
struct bidirectional : virtual input, virtual output, detail::two_sequence { };
struct dual_use : virtual input, virtual output { }; 
struct input_seekable : virtual input, virtual detail::random_access { };
struct output_seekable : virtual output, virtual detail::random_access { };
struct seekable
: virtual input_seekable, 
virtual output_seekable,
detail::one_head
{ };
struct dual_seekable
: virtual input_seekable,
virtual output_seekable,
detail::two_head
{ };  
struct bidirectional_seekable
: input_seekable, output_seekable,
bidirectional, detail::two_head
{ };


struct device_tag : virtual any_tag { };
struct filter_tag : virtual any_tag { };


struct peekable_tag : virtual any_tag { };        
struct closable_tag : virtual any_tag { };
struct flushable_tag : virtual any_tag { };
struct localizable_tag : virtual any_tag { };
struct optimally_buffered_tag : virtual any_tag { };
struct direct_tag : virtual any_tag { };          
struct multichar_tag : virtual any_tag { };       

struct source_tag : device_tag, input { };
struct sink_tag : device_tag, output { };
struct bidirectional_device_tag : device_tag, bidirectional { };
struct seekable_device_tag : virtual device_tag, seekable { };

struct input_filter_tag : filter_tag, input { };
struct output_filter_tag : filter_tag, output { };
struct bidirectional_filter_tag : filter_tag, bidirectional { };
struct seekable_filter_tag : filter_tag, seekable { };
struct dual_use_filter_tag : filter_tag, dual_use { };

struct multichar_input_filter_tag
: multichar_tag,
input_filter_tag
{ };
struct multichar_output_filter_tag
: multichar_tag,
output_filter_tag
{ };
struct multichar_bidirectional_filter_tag
: multichar_tag,
bidirectional_filter_tag
{ };
struct multichar_seekable_filter_tag
: multichar_tag,
seekable_filter_tag
{ };
struct multichar_dual_use_filter_tag 
: multichar_tag, 
dual_use_filter_tag
{ };


struct std_io_tag : virtual localizable_tag { };
struct istream_tag
: virtual device_tag,
virtual peekable_tag,
virtual std_io_tag
{ };
struct ostream_tag
: virtual device_tag,
virtual std_io_tag
{ };
struct iostream_tag
: istream_tag,
ostream_tag
{ };
struct streambuf_tag
: device_tag,
peekable_tag,
std_io_tag
{ };
struct ifstream_tag
: input_seekable,
closable_tag,
istream_tag
{ };
struct ofstream_tag
: output_seekable,
closable_tag,
ostream_tag
{ };
struct fstream_tag
: seekable,
closable_tag,
iostream_tag
{ };
struct filebuf_tag
: seekable,
closable_tag,
streambuf_tag
{ };
struct istringstream_tag
: input_seekable,
istream_tag
{ };
struct ostringstream_tag
: output_seekable,
ostream_tag
{ };
struct stringstream_tag
: dual_seekable,
iostream_tag
{ };
struct stringbuf_tag
: dual_seekable,
streambuf_tag
{ };
struct generic_istream_tag 
: input_seekable,
istream_tag
{ };
struct generic_ostream_tag 
: output_seekable,
ostream_tag
{ };
struct generic_iostream_tag 
: seekable,
iostream_tag
{ };
struct generic_streambuf_tag 
: seekable,
streambuf_tag
{ };

} } 

#endif 
