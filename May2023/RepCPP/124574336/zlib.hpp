


#ifndef BOOST_IOSTREAMS_ZLIB_HPP_INCLUDED
#define BOOST_IOSTREAMS_ZLIB_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <cassert>                            
#include <iosfwd>            
#include <memory>            
#include <new>          
#include <boost/config.hpp>  
#include <boost/cstdint.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/constants.hpp>   
#include <boost/iostreams/detail/config/auto_link.hpp>
#include <boost/iostreams/detail/config/dyn_link.hpp>
#include <boost/iostreams/detail/config/wide_streams.hpp>
#include <boost/iostreams/detail/config/zlib.hpp>
#include <boost/iostreams/detail/ios.hpp>  
#include <boost/iostreams/filter/symmetric.hpp>                
#include <boost/iostreams/pipeline.hpp>                
#include <boost/type_traits/is_same.hpp>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable:4251 4275 4231 4660)         
#endif
#include <boost/config/abi_prefix.hpp>           

namespace boost { namespace iostreams {

namespace zlib {

typedef uint32_t uint;
typedef uint8_t byte;
typedef uint32_t ulong;

typedef void* (*xalloc_func)(void*, zlib::uint, zlib::uint);
typedef void (*xfree_func)(void*, void*);


BOOST_IOSTREAMS_DECL extern const int no_compression;
BOOST_IOSTREAMS_DECL extern const int best_speed;
BOOST_IOSTREAMS_DECL extern const int best_compression;
BOOST_IOSTREAMS_DECL extern const int default_compression;


BOOST_IOSTREAMS_DECL extern const int deflated;


BOOST_IOSTREAMS_DECL extern const int default_strategy;
BOOST_IOSTREAMS_DECL extern const int filtered;
BOOST_IOSTREAMS_DECL extern const int huffman_only;


BOOST_IOSTREAMS_DECL extern const int okay;
BOOST_IOSTREAMS_DECL extern const int stream_end;
BOOST_IOSTREAMS_DECL extern const int stream_error;
BOOST_IOSTREAMS_DECL extern const int version_error;
BOOST_IOSTREAMS_DECL extern const int data_error;
BOOST_IOSTREAMS_DECL extern const int mem_error;
BOOST_IOSTREAMS_DECL extern const int buf_error;


BOOST_IOSTREAMS_DECL extern const int finish;
BOOST_IOSTREAMS_DECL extern const int no_flush;
BOOST_IOSTREAMS_DECL extern const int sync_flush;




const int null                               = 0;


const int default_window_bits                = 15;
const int default_mem_level                  = 8;
const bool default_crc                       = false;
const bool default_noheader                  = false;

} 

struct zlib_params {

zlib_params( int level_          = zlib::default_compression,
int method_         = zlib::deflated,
int window_bits_    = zlib::default_window_bits, 
int mem_level_      = zlib::default_mem_level, 
int strategy_       = zlib::default_strategy,
bool noheader_      = zlib::default_noheader,
bool calculate_crc_ = zlib::default_crc )
: level(level_), method(method_), window_bits(window_bits_),
mem_level(mem_level_), strategy(strategy_),  
noheader(noheader_), calculate_crc(calculate_crc_)
{ }
int level;
int method;
int window_bits;
int mem_level;
int strategy;
bool noheader;
bool calculate_crc;
};

class BOOST_IOSTREAMS_DECL zlib_error : public BOOST_IOSTREAMS_FAILURE {
public:
explicit zlib_error(int error);
int error() const { return error_; }
static void check BOOST_PREVENT_MACRO_SUBSTITUTION(int error);
private:
int error_;
};

namespace detail {

template<typename Alloc>
struct zlib_allocator_traits {
#ifndef BOOST_NO_STD_ALLOCATOR
#if defined(BOOST_NO_CXX11_ALLOCATOR)
typedef typename Alloc::template rebind<char>::other type;
#else
typedef typename std::allocator_traits<Alloc>::template rebind_alloc<char> type;
#endif
#else
typedef std::allocator<char> type;
#endif
};

template< typename Alloc,
typename Base = 
BOOST_DEDUCED_TYPENAME zlib_allocator_traits<Alloc>::type >
struct zlib_allocator : private Base {
private:
#if defined(BOOST_NO_CXX11_ALLOCATOR) || defined(BOOST_NO_STD_ALLOCATOR)
typedef typename Base::size_type size_type;
#else
typedef typename std::allocator_traits<Base>::size_type size_type;
#endif
public:
BOOST_STATIC_CONSTANT(bool, custom = 
(!is_same<std::allocator<char>, Base>::value));
typedef typename zlib_allocator_traits<Alloc>::type allocator_type;
static void* allocate(void* self, zlib::uint items, zlib::uint size);
static void deallocate(void* self, void* address);
};

class BOOST_IOSTREAMS_DECL zlib_base { 
public:
typedef char char_type;
protected:
zlib_base();
~zlib_base();
void* stream() { return stream_; }
template<typename Alloc> 
void init( const zlib_params& p, 
bool compress,
zlib_allocator<Alloc>& zalloc )
{
bool custom = zlib_allocator<Alloc>::custom;
do_init( p, compress,
custom ? zlib_allocator<Alloc>::allocate : 0,
custom ? zlib_allocator<Alloc>::deallocate : 0,
&zalloc );
}
void before( const char*& src_begin, const char* src_end,
char*& dest_begin, char* dest_end );
void after( const char*& src_begin, char*& dest_begin, 
bool compress );
int xdeflate(int flush);  
int xinflate(int flush);  
void reset(bool compress, bool realloc);
public:
zlib::ulong crc() const { return crc_; }
int total_in() const { return total_in_; }
int total_out() const { return total_out_; }
private:
void do_init( const zlib_params& p, bool compress, 
zlib::xalloc_func,
zlib::xfree_func, 
void* derived );
void*        stream_;         
bool         calculate_crc_;
zlib::ulong  crc_;
zlib::ulong  crc_imp_;
int          total_in_;
int          total_out_;
};

template<typename Alloc = std::allocator<char> >
class zlib_compressor_impl : public zlib_base, public zlib_allocator<Alloc> { 
public: 
zlib_compressor_impl(const zlib_params& = zlib::default_compression);
~zlib_compressor_impl();
bool filter( const char*& src_begin, const char* src_end,
char*& dest_begin, char* dest_end, bool flush );
void close();
};

template<typename Alloc = std::allocator<char> >
class zlib_decompressor_impl : public zlib_base, public zlib_allocator<Alloc> {
public:
zlib_decompressor_impl(const zlib_params&);
zlib_decompressor_impl(int window_bits = zlib::default_window_bits);
~zlib_decompressor_impl();
bool filter( const char*& begin_in, const char* end_in,
char*& begin_out, char* end_out, bool flush );
void close();
bool eof() const { return eof_; }
private:
bool eof_;
};

} 

template<typename Alloc = std::allocator<char> >
struct basic_zlib_compressor 
: symmetric_filter<detail::zlib_compressor_impl<Alloc>, Alloc> 
{
private:
typedef detail::zlib_compressor_impl<Alloc>         impl_type;
typedef symmetric_filter<impl_type, Alloc>  base_type;
public:
typedef typename base_type::char_type               char_type;
typedef typename base_type::category                category;
basic_zlib_compressor( const zlib_params& = zlib::default_compression, 
std::streamsize buffer_size = default_device_buffer_size );
zlib::ulong crc() { return this->filter().crc(); }
int total_in() {  return this->filter().total_in(); }
};
BOOST_IOSTREAMS_PIPABLE(basic_zlib_compressor, 1)

typedef basic_zlib_compressor<> zlib_compressor;

template<typename Alloc = std::allocator<char> >
struct basic_zlib_decompressor 
: symmetric_filter<detail::zlib_decompressor_impl<Alloc>, Alloc> 
{
private:
typedef detail::zlib_decompressor_impl<Alloc>       impl_type;
typedef symmetric_filter<impl_type, Alloc>  base_type;
public:
typedef typename base_type::char_type               char_type;
typedef typename base_type::category                category;
basic_zlib_decompressor( int window_bits = zlib::default_window_bits,
std::streamsize buffer_size = default_device_buffer_size );
basic_zlib_decompressor( const zlib_params& p,
std::streamsize buffer_size = default_device_buffer_size );
zlib::ulong crc() { return this->filter().crc(); }
int total_out() {  return this->filter().total_out(); }
bool eof() { return this->filter().eof(); }
};
BOOST_IOSTREAMS_PIPABLE(basic_zlib_decompressor, 1)

typedef basic_zlib_decompressor<> zlib_decompressor;



namespace detail {

template<typename Alloc, typename Base>
void* zlib_allocator<Alloc, Base>::allocate
(void* self, zlib::uint items, zlib::uint size)
{ 
size_type len = items * size;
char* ptr = 
static_cast<allocator_type*>(self)->allocate
(len + sizeof(size_type)
#if BOOST_WORKAROUND(BOOST_DINKUMWARE_STDLIB, == 1)
, (char*)0
#endif
);
*reinterpret_cast<size_type*>(ptr) = len;
return ptr + sizeof(size_type);
}

template<typename Alloc, typename Base>
void zlib_allocator<Alloc, Base>::deallocate(void* self, void* address)
{ 
char* ptr = reinterpret_cast<char*>(address) - sizeof(size_type);
size_type len = *reinterpret_cast<size_type*>(ptr) + sizeof(size_type);
static_cast<allocator_type*>(self)->deallocate(ptr, len); 
}


template<typename Alloc>
zlib_compressor_impl<Alloc>::zlib_compressor_impl(const zlib_params& p)
{ init(p, true, static_cast<zlib_allocator<Alloc>&>(*this)); }

template<typename Alloc>
zlib_compressor_impl<Alloc>::~zlib_compressor_impl()
{ reset(true, false); }

template<typename Alloc>
bool zlib_compressor_impl<Alloc>::filter
( const char*& src_begin, const char* src_end,
char*& dest_begin, char* dest_end, bool flush )
{
before(src_begin, src_end, dest_begin, dest_end);
int result = xdeflate(flush ? zlib::finish : zlib::no_flush);
after(src_begin, dest_begin, true);
zlib_error::check BOOST_PREVENT_MACRO_SUBSTITUTION(result);
return result != zlib::stream_end;
}

template<typename Alloc>
void zlib_compressor_impl<Alloc>::close() { reset(true, true); }


template<typename Alloc>
zlib_decompressor_impl<Alloc>::zlib_decompressor_impl(const zlib_params& p)
: eof_(false)
{ init(p, false, static_cast<zlib_allocator<Alloc>&>(*this)); }

template<typename Alloc>
zlib_decompressor_impl<Alloc>::~zlib_decompressor_impl()
{ reset(false, false); }

template<typename Alloc>
zlib_decompressor_impl<Alloc>::zlib_decompressor_impl(int window_bits)
{ 
zlib_params p;
p.window_bits = window_bits;
init(p, false, static_cast<zlib_allocator<Alloc>&>(*this)); 
}

template<typename Alloc>
bool zlib_decompressor_impl<Alloc>::filter
( const char*& src_begin, const char* src_end,
char*& dest_begin, char* dest_end, bool  )
{
before(src_begin, src_end, dest_begin, dest_end);
int result = xinflate(zlib::sync_flush);
after(src_begin, dest_begin, false);
zlib_error::check BOOST_PREVENT_MACRO_SUBSTITUTION(result);
return !(eof_ = result == zlib::stream_end);
}

template<typename Alloc>
void zlib_decompressor_impl<Alloc>::close() {
eof_ = false;
reset(false, true);
}

} 


template<typename Alloc>
basic_zlib_compressor<Alloc>::basic_zlib_compressor
(const zlib_params& p, std::streamsize buffer_size) 
: base_type(buffer_size, p) { }


template<typename Alloc>
basic_zlib_decompressor<Alloc>::basic_zlib_decompressor
(int window_bits, std::streamsize buffer_size) 
: base_type(buffer_size, window_bits) { }

template<typename Alloc>
basic_zlib_decompressor<Alloc>::basic_zlib_decompressor
(const zlib_params& p, std::streamsize buffer_size) 
: base_type(buffer_size, p) { }


} } 

#include <boost/config/abi_suffix.hpp> 
#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#endif 
