


#ifndef BOOST_IOSTREAMS_GZIP_HPP_INCLUDED
#define BOOST_IOSTREAMS_GZIP_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp> 
#include <algorithm>                      
#include <boost/assert.hpp>
#include <cstdio>                         
#include <cstddef>                        
#include <ctime>                          
#include <memory>                         
#include <boost/config.hpp>               
#include <boost/detail/workaround.hpp>
#include <boost/cstdint.hpp>              
#include <boost/iostreams/checked_operations.hpp>
#include <boost/iostreams/constants.hpp>  
#include <boost/iostreams/detail/adapter/non_blocking_adapter.hpp>
#include <boost/iostreams/detail/adapter/range_adapter.hpp>
#include <boost/iostreams/detail/char_traits.hpp>
#include <boost/iostreams/detail/ios.hpp> 
#include <boost/iostreams/detail/error.hpp>
#include <boost/iostreams/operations.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/pipeline.hpp>     
#include <boost/iostreams/putback.hpp>
#include <boost/throw_exception.hpp>

#if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable:4244)    
# pragma warning(disable:4251)    
# pragma warning(disable:4309)    
#endif

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; }
#endif

namespace boost { namespace iostreams {


namespace gzip {

using namespace boost::iostreams::zlib;


const int zlib_error        = 1;
const int bad_crc           = 2; 
const int bad_length        = 3; 
const int bad_header        = 4; 
const int bad_footer        = 5; 
const int bad_method        = 6; 

namespace magic {


const int id1               = 0x1f;
const int id2               = 0x8b;

} 

namespace method {


const int deflate           = 8;

} 

namespace flags {


const int text              = 1;
const int header_crc        = 2;
const int extra             = 4;
const int name              = 8;
const int comment           = 16;

} 

namespace extra_flags {


const int best_compression  = 2;
const int best_speed        = 4;

} 


const int os_fat            = 0;
const int os_amiga          = 1;
const int os_vms            = 2;
const int os_unix           = 3;
const int os_vm_cms         = 4;
const int os_atari          = 5;
const int os_hpfs           = 6;
const int os_macintosh      = 7;
const int os_z_system       = 8;
const int os_cp_m           = 9;
const int os_tops_20        = 10;
const int os_ntfs           = 11;
const int os_qdos           = 12;
const int os_acorn          = 13;
const int os_unknown        = 255;

} 


struct gzip_params : zlib_params {

gzip_params( int level              = gzip::default_compression,
int method             = gzip::deflated,
int window_bits        = gzip::default_window_bits,
int mem_level          = gzip::default_mem_level,
int strategy           = gzip::default_strategy,
std::string file_name_  = "",
std::string comment_    = "",
std::time_t mtime_      = 0 )
: zlib_params(level, method, window_bits, mem_level, strategy),
file_name(file_name_), comment(comment_), mtime(mtime_)
{ }
std::string  file_name;
std::string  comment;
std::time_t  mtime;
};


class gzip_error : public BOOST_IOSTREAMS_FAILURE {
public:
explicit gzip_error(int error)
: BOOST_IOSTREAMS_FAILURE("gzip error"),
error_(error), zlib_error_code_(zlib::okay) { }
explicit gzip_error(const zlib_error& e)
: BOOST_IOSTREAMS_FAILURE("gzip error"),
error_(gzip::zlib_error), zlib_error_code_(e.error())
{ }
int error() const { return error_; }
int zlib_error_code() const { return zlib_error_code_; }
private:
int error_;
int zlib_error_code_;
};


template<typename Alloc = std::allocator<char> >
class basic_gzip_compressor : basic_zlib_compressor<Alloc> {
private:
typedef basic_zlib_compressor<Alloc>  base_type;
public:
typedef char char_type;
struct category
: dual_use,
filter_tag,
multichar_tag,
closable_tag
{ };
basic_gzip_compressor( const gzip_params& = gzip::default_compression,
std::streamsize buffer_size = default_device_buffer_size );

template<typename Source>
std::streamsize read(Source& src, char_type* s, std::streamsize n)
{
std::streamsize result = 0;

if (!(flags_ & f_header_done))
result += read_string(s, n, header_);

if (!(flags_ & f_body_done)) {

std::streamsize amt = base_type::read(src, s + result, n - result);
if (amt != -1) {
result += amt;
if (amt < n - result) { 
amt = base_type::read(src, s + result, n - result);
if (amt != -1)
result += amt;
}
}
if (amt == -1)
prepare_footer();
}

if ((flags_ & f_body_done) != 0 && result < n)
result += read_string(s + result, n - result, footer_);

return result != 0 ? result : -1;
}

template<typename Sink>
std::streamsize write(Sink& snk, const char_type* s, std::streamsize n)
{
if (!(flags_ & f_header_done)) {
std::streamsize amt = 
static_cast<std::streamsize>(header_.size() - offset_);
offset_ += boost::iostreams::write_if(snk, header_.data() + offset_, amt);
if (offset_ == header_.size())
flags_ |= f_header_done;
else
return 0;
}
return base_type::write(snk, s, n);
}

template<typename Sink>
void close(Sink& snk, BOOST_IOS::openmode m)
{
try {
if (m == BOOST_IOS::out && !(flags_ & f_header_done))
this->write(snk, 0, 0);

base_type::close(snk, m);

if (m == BOOST_IOS::out) {
if (flags_ & f_header_done) {

write_long(this->crc(), snk);
write_long(this->total_in(), snk);
}
}
} catch(...) {
close_impl();
throw;
}
close_impl();
}
private:
static gzip_params normalize_params(gzip_params p);
void prepare_footer();
std::streamsize read_string(char* s, std::streamsize n, std::string& str);

template<typename Sink>
static void write_long(long n, Sink& next, boost::mpl::true_)
{
boost::iostreams::put(next, static_cast<char>(0xFF & n));
boost::iostreams::put(next, static_cast<char>(0xFF & (n >> 8)));
boost::iostreams::put(next, static_cast<char>(0xFF & (n >> 16)));
boost::iostreams::put(next, static_cast<char>(0xFF & (n >> 24)));
}
template<typename Sink>
static void write_long(long, Sink&, boost::mpl::false_)
{
}
template<typename Sink>
static void write_long(long n, Sink& next)
{
typedef typename category_of<Sink>::type category;
typedef is_convertible<category, output> can_write;
write_long(n, next, can_write());
}

void close_impl()
{
footer_.clear();
offset_ = 0;
flags_ = 0;
}

enum state_type {
f_header_done = 1,
f_body_done = f_header_done << 1,
f_footer_done = f_body_done << 1
};
std::string  header_;
std::string  footer_;
std::size_t  offset_;
int          flags_;
};
BOOST_IOSTREAMS_PIPABLE(basic_gzip_compressor, 1)

typedef basic_gzip_compressor<> gzip_compressor;


namespace detail {

class BOOST_IOSTREAMS_DECL gzip_header {
public:
gzip_header() { reset(); }

void process(char c);
bool done() const { return state_ == s_done; }
void reset();

std::string file_name() const { return file_name_; }
std::string comment() const { return comment_; }
bool text() const { return (flags_ & gzip::flags::text) != 0; }
int os() const { return os_; }
std::time_t mtime() const { return mtime_; }
private:
enum state_type {
s_id1       = 1,
s_id2       = s_id1 + 1,
s_cm        = s_id2 + 1,
s_flg       = s_cm + 1,
s_mtime     = s_flg + 1,
s_xfl       = s_mtime + 1,
s_os        = s_xfl + 1,
s_xlen      = s_os + 1,
s_extra     = s_xlen + 1,
s_name      = s_extra + 1,
s_comment   = s_name + 1,
s_hcrc      = s_comment + 1,
s_done      = s_hcrc + 1
};
std::string  file_name_;
std::string  comment_;
int          os_;
std::time_t  mtime_;
int          flags_;
int          state_;
int          offset_;  
int          xlen_;    
};

class BOOST_IOSTREAMS_DECL gzip_footer {
public:
gzip_footer() { reset(); }

void process(char c);
bool done() const { return state_ == s_done; }
void reset();

zlib::ulong crc() const { return crc_; }
zlib::ulong uncompressed_size() const { return isize_; }
private:
enum state_type {
s_crc     = 1,
s_isize   = s_crc + 1,
s_done    = s_isize + 1
};
zlib::ulong  crc_;
zlib::ulong  isize_;
int          state_;
int          offset_; 
};

} 


template<typename Alloc = std::allocator<char> >
class basic_gzip_decompressor : basic_zlib_decompressor<Alloc> {
private:
typedef basic_zlib_decompressor<Alloc>   base_type;
typedef typename base_type::string_type  string_type;
public:
typedef char char_type;
struct category
: dual_use,
filter_tag,
multichar_tag,
closable_tag
{ };
basic_gzip_decompressor( int window_bits = gzip::default_window_bits,
std::streamsize buffer_size = default_device_buffer_size );

template<typename Sink>
std::streamsize write(Sink& snk, const char_type* s, std::streamsize n)
{
std::streamsize result = 0;
while(result < n) {
if(state_ == s_start) {
state_ = s_header;
header_.reset();
footer_.reset();
}
if (state_ == s_header) {
int c = s[result++];
header_.process(c);
if (header_.done())
state_ = s_body;
} else if (state_ == s_body) {
try {
std::streamsize amt = 
base_type::write(snk, s + result, n - result);
result += amt;
if (!this->eof()) {
break;
} else {
state_ = s_footer;
}
} catch (const zlib_error& e) {
boost::throw_exception(gzip_error(e));
}
} else { 
if (footer_.done()) {
if (footer_.crc() != this->crc())
boost::throw_exception(gzip_error(gzip::bad_crc));

base_type::close(snk, BOOST_IOS::out);
state_ = s_start;
} else {
int c = s[result++];
footer_.process(c);
}
}
}
return result;
}

template<typename Source>
std::streamsize read(Source& src, char_type* s, std::streamsize n)
{
typedef char_traits<char>  traits_type;
std::streamsize            result = 0;
peekable_source<Source>    peek(src, putback_);
while (result < n && state_ != s_done) {
if (state_ == s_start) {
state_ = s_header;
header_.reset();
footer_.reset();
}
if (state_ == s_header) {
int c = boost::iostreams::get(peek);
if (traits_type::is_eof(c)) {
boost::throw_exception(gzip_error(gzip::bad_header));
} else if (traits_type::would_block(c)) {
break;
}
header_.process(c);
if (header_.done())
state_ = s_body;
} else if (state_ == s_body) {
try {
std::streamsize amt = 
base_type::read(peek, s + result, n - result);
if (amt != -1) {
result += amt;
if (amt < n - result)
break;
} else {
peek.putback(this->unconsumed_input());
state_ = s_footer;
}
} catch (const zlib_error& e) {
boost::throw_exception(gzip_error(e));
}
} else { 
int c = boost::iostreams::get(peek);
if (traits_type::is_eof(c)) {
boost::throw_exception(gzip_error(gzip::bad_footer));
} else if (traits_type::would_block(c)) {
break;
}
footer_.process(c);
if (footer_.done()) {
if (footer_.crc() != this->crc())
boost::throw_exception(gzip_error(gzip::bad_crc));
c = boost::iostreams::get(peek);
if (traits_type::is_eof(c)) {
state_ = s_done;
} else {
peek.putback(c);
base_type::close(peek, BOOST_IOS::in);
state_ = s_start;
header_.reset();
footer_.reset();
}
}
}
}
if (peek.has_unconsumed_input()) {
putback_ = peek.unconsumed_input();
} else {
putback_.clear();
}
return result != 0 || state_ != s_done ?
result :
-1;
}

template<typename Source>
void close(Source& src, BOOST_IOS::openmode m)
{
try {
base_type::close(src, m);
} catch (const zlib_error& e) {
state_ = s_start;
boost::throw_exception(gzip_error(e));
}
if (m == BOOST_IOS::out) {
if (state_ == s_start || state_ == s_header)
boost::throw_exception(gzip_error(gzip::bad_header));
else if (state_ == s_body)
boost::throw_exception(gzip_error(gzip::bad_footer));
else if (state_ == s_footer) {
if (!footer_.done())
boost::throw_exception(gzip_error(gzip::bad_footer));
else if(footer_.crc() != this->crc())
boost::throw_exception(gzip_error(gzip::bad_crc));
} else {
BOOST_ASSERT(!"Bad state");
}
}
state_ = s_start;
}

std::string file_name() const { return header_.file_name(); }
std::string comment() const { return header_.comment(); }
bool text() const { return header_.text(); }
int os() const { return header_.os(); }
std::time_t mtime() const { return header_.mtime(); }
private:
static gzip_params make_params(int window_bits);

template<typename Source>
struct peekable_source {
typedef char char_type;
struct category : source_tag, peekable_tag { };
explicit peekable_source(Source& src, const string_type& putback = "") 
: src_(src), putback_(putback), offset_(0)
{ }
std::streamsize read(char* s, std::streamsize n)
{
std::streamsize result = 0;

std::streamsize pbsize = 
static_cast<std::streamsize>(putback_.size());
if (offset_ < pbsize) {
result = (std::min)(n, pbsize - offset_);
BOOST_IOSTREAMS_CHAR_TRAITS(char)::copy(
s, putback_.data() + offset_, result);
offset_ += result;
if (result == n)
return result;
}

std::streamsize amt = 
boost::iostreams::read(src_, s + result, n - result);
return amt != -1 ? 
result + amt : 
result ? result : -1;
}
bool putback(char c)
{
if (offset_) {
putback_[--offset_] = c;
} else {
boost::throw_exception(
boost::iostreams::detail::bad_putback());
}
return true;
}
void putback(const string_type& s)
{
putback_.replace(0, offset_, s);
offset_ = 0;
}

bool has_unconsumed_input() const 
{
return offset_ < static_cast<std::streamsize>(putback_.size());
}

string_type unconsumed_input() const
{
return string_type(putback_, offset_, putback_.size() - offset_);
}
Source&          src_;
string_type      putback_;
std::streamsize  offset_;
};

enum state_type {
s_start   = 1,
s_header  = s_start + 1,
s_body    = s_header + 1,
s_footer  = s_body + 1,
s_done    = s_footer + 1
};
detail::gzip_header  header_;
detail::gzip_footer  footer_;
string_type          putback_;
int                  state_;
};
BOOST_IOSTREAMS_PIPABLE(basic_gzip_decompressor, 1)

typedef basic_gzip_decompressor<> gzip_decompressor;


template<typename Alloc>
basic_gzip_compressor<Alloc>::basic_gzip_compressor
(const gzip_params& p, std::streamsize buffer_size)
: base_type(normalize_params(p), buffer_size),
offset_(0), flags_(0)
{
bool has_name = !p.file_name.empty();
bool has_comment = !p.comment.empty();

std::string::size_type length =
10 +
(has_name ? p.file_name.size() + 1 : 0) +
(has_comment ? p.comment.size() + 1 : 0);
int flags =
(has_name ? gzip::flags::name : 0) +
(has_comment ? gzip::flags::comment : 0);
int extra_flags =
( p.level == zlib::best_compression ?
gzip::extra_flags::best_compression :
0 ) +
( p.level == zlib::best_speed ?
gzip::extra_flags::best_speed :
0 );
header_.reserve(length);
header_ += gzip::magic::id1;                         
header_ += static_cast<char>(gzip::magic::id2);      
header_ += gzip::method::deflate;                    
header_ += static_cast<char>(flags);                 
header_ += static_cast<char>(0xFF & p.mtime);        
header_ += static_cast<char>(0xFF & (p.mtime >> 8));
header_ += static_cast<char>(0xFF & (p.mtime >> 16));
header_ += static_cast<char>(0xFF & (p.mtime >> 24));
header_ += static_cast<char>(extra_flags);           
header_ += static_cast<char>(gzip::os_unknown);      
if (has_name) {
header_ += p.file_name;
header_ += '\0';
}
if (has_comment) {
header_ += p.comment;
header_ += '\0';
}
}

template<typename Alloc>
gzip_params basic_gzip_compressor<Alloc>::normalize_params(gzip_params p)
{
p.noheader = true;
p.calculate_crc = true;
return p;
}

template<typename Alloc>
void basic_gzip_compressor<Alloc>::prepare_footer()
{
boost::iostreams::back_insert_device<std::string> out(footer_);
write_long(this->crc(), out);
write_long(this->total_in(), out);
flags_ |= f_body_done;
offset_ = 0;
}

template<typename Alloc>
std::streamsize basic_gzip_compressor<Alloc>::read_string
(char* s, std::streamsize n, std::string& str)
{
std::streamsize avail =
static_cast<std::streamsize>(str.size() - offset_);
std::streamsize amt = (std::min)(avail, n);
std::copy( str.data() + offset_,
str.data() + offset_ + amt,
s );
offset_ += amt;
if ( !(flags_ & f_header_done) &&
offset_ == static_cast<std::size_t>(str.size()) )
{
flags_ |= f_header_done;
}
return amt;
}


template<typename Alloc>
basic_gzip_decompressor<Alloc>::basic_gzip_decompressor
(int window_bits, std::streamsize buffer_size)
: base_type(make_params(window_bits), buffer_size),
state_(s_start)
{ }

template<typename Alloc>
gzip_params basic_gzip_decompressor<Alloc>::make_params(int window_bits)
{
gzip_params p;
p.window_bits = window_bits;
p.noheader = true;
p.calculate_crc = true;
return p;
}


} } 

#if defined(BOOST_MSVC)
# pragma warning(pop)
#endif

#endif 
