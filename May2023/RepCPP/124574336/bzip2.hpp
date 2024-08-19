


#ifndef BOOST_IOSTREAMS_BZIP2_HPP_INCLUDED
#define BOOST_IOSTREAMS_BZIP2_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <cassert>                            
#include <memory>            
#include <new>               
#include <boost/config.hpp>  
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/constants.hpp>   
#include <boost/iostreams/detail/config/auto_link.hpp>
#include <boost/iostreams/detail/config/bzip2.hpp>
#include <boost/iostreams/detail/config/dyn_link.hpp>
#include <boost/iostreams/detail/config/wide_streams.hpp>
#include <boost/iostreams/detail/ios.hpp>  
#include <boost/iostreams/filter/symmetric.hpp>               
#include <boost/iostreams/pipeline.hpp>       
#include <boost/type_traits/is_same.hpp>     

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable:4251 4231 4660)
#endif
#include <boost/config/abi_prefix.hpp>           

#undef small

namespace boost { namespace iostreams {

namespace bzip2 {


typedef void* (*alloc_func)(void*, int, int);
typedef void (*free_func)(void*, void*);


BOOST_IOSTREAMS_DECL extern const int ok;
BOOST_IOSTREAMS_DECL extern const int run_ok;
BOOST_IOSTREAMS_DECL extern const int flush_ok;
BOOST_IOSTREAMS_DECL extern const int finish_ok;
BOOST_IOSTREAMS_DECL extern const int stream_end;    
BOOST_IOSTREAMS_DECL extern const int sequence_error;
BOOST_IOSTREAMS_DECL extern const int param_error;
BOOST_IOSTREAMS_DECL extern const int mem_error;
BOOST_IOSTREAMS_DECL extern const int data_error;
BOOST_IOSTREAMS_DECL extern const int data_error_magic;
BOOST_IOSTREAMS_DECL extern const int io_error;
BOOST_IOSTREAMS_DECL extern const int unexpected_eof;
BOOST_IOSTREAMS_DECL extern const int outbuff_full;
BOOST_IOSTREAMS_DECL extern const int config_error;


BOOST_IOSTREAMS_DECL extern const int finish;
BOOST_IOSTREAMS_DECL extern const int run;


const int default_block_size   = 9;
const int default_work_factor  = 30;
const bool default_small       = false;

} 

struct bzip2_params {

bzip2_params( int block_size_  = bzip2::default_block_size,
int work_factor_ = bzip2::default_work_factor )
: block_size(block_size_), work_factor(work_factor_)
{ }

bzip2_params(bool small)
: small(small), work_factor(0)
{ }

union {
int   block_size;    
bool  small;         
};
int       work_factor;
};

class BOOST_IOSTREAMS_DECL bzip2_error : public BOOST_IOSTREAMS_FAILURE {
public:
explicit bzip2_error(int error);
int error() const { return error_; }
static void check BOOST_PREVENT_MACRO_SUBSTITUTION(int error);
private:
int error_;
};

namespace detail {

template<typename Alloc>
struct bzip2_allocator_traits {
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
BOOST_DEDUCED_TYPENAME bzip2_allocator_traits<Alloc>::type >
struct bzip2_allocator : private Base {
private:
#if defined(BOOST_NO_CXX11_ALLOCATOR) || defined(BOOST_NO_STD_ALLOCATOR)
typedef typename Base::size_type size_type;
#else
typedef typename std::allocator_traits<Base>::size_type size_type;
#endif
public:
BOOST_STATIC_CONSTANT(bool, custom = 
(!is_same<std::allocator<char>, Base>::value));
typedef typename bzip2_allocator_traits<Alloc>::type allocator_type;
static void* allocate(void* self, int items, int size);
static void deallocate(void* self, void* address);
};

class BOOST_IOSTREAMS_DECL bzip2_base  { 
public:
typedef char char_type;
protected:
bzip2_base(const bzip2_params& params);
~bzip2_base();
bzip2_params& params() { return params_; }
bool& ready() { return ready_; }
template<typename Alloc> 
void init( bool compress,
bzip2_allocator<Alloc>& alloc )
{
bool custom = bzip2_allocator<Alloc>::custom;
do_init( compress,
custom ? bzip2_allocator<Alloc>::allocate : 0,
custom ? bzip2_allocator<Alloc>::deallocate : 0,
custom ? &alloc : 0 );
}
void before( const char*& src_begin, const char* src_end,
char*& dest_begin, char* dest_end );
void after(const char*& src_begin, char*& dest_begin);
int check_end(const char* src_begin, const char* dest_begin);
int compress(int action);
int decompress();
int end(bool compress, std::nothrow_t);
void end(bool compress);
private:
void do_init( bool compress, 
bzip2::alloc_func,
bzip2::free_func, 
void* derived );
bzip2_params  params_;
void*         stream_; 
bool          ready_;
};

template<typename Alloc = std::allocator<char> >
class bzip2_compressor_impl 
: public bzip2_base, 
#if BOOST_WORKAROUND(BOOST_BORLANDC, < 0x600)
public
#endif
bzip2_allocator<Alloc> 
{
public: 
bzip2_compressor_impl(const bzip2_params&);
~bzip2_compressor_impl();
bool filter( const char*& src_begin, const char* src_end,
char*& dest_begin, char* dest_end, bool flush );
void close();
private:
void init();
bool eof_; 
};

template<typename Alloc = std::allocator<char> >
class bzip2_decompressor_impl 
: public bzip2_base, 
#if BOOST_WORKAROUND(BOOST_BORLANDC, < 0x600)
public
#endif
bzip2_allocator<Alloc> 
{ 
public:
bzip2_decompressor_impl(bool small = bzip2::default_small);
~bzip2_decompressor_impl();
bool filter( const char*& begin_in, const char* end_in,
char*& begin_out, char* end_out, bool flush );
void close();
private:
void init();
bool eof_; 
};

} 

template<typename Alloc = std::allocator<char> >
struct basic_bzip2_compressor 
: symmetric_filter<detail::bzip2_compressor_impl<Alloc>, Alloc> 
{
private:
typedef detail::bzip2_compressor_impl<Alloc>        impl_type;
typedef symmetric_filter<impl_type, Alloc>  base_type;
public:
typedef typename base_type::char_type               char_type;
typedef typename base_type::category                category;
basic_bzip2_compressor( const bzip2_params& = bzip2::default_block_size, 
std::streamsize buffer_size =  default_device_buffer_size );
};
BOOST_IOSTREAMS_PIPABLE(basic_bzip2_compressor, 1)

typedef basic_bzip2_compressor<> bzip2_compressor;

template<typename Alloc = std::allocator<char> >
struct basic_bzip2_decompressor 
: symmetric_filter<detail::bzip2_decompressor_impl<Alloc>, Alloc> 
{
private:
typedef detail::bzip2_decompressor_impl<Alloc>      impl_type;
typedef symmetric_filter<impl_type, Alloc>  base_type;
public:
typedef typename base_type::char_type               char_type;
typedef typename base_type::category                category;
basic_bzip2_decompressor( bool small = bzip2::default_small,
std::streamsize buffer_size = default_device_buffer_size );
};
BOOST_IOSTREAMS_PIPABLE(basic_bzip2_decompressor, 1)

typedef basic_bzip2_decompressor<> bzip2_decompressor;



namespace detail {

template<typename Alloc, typename Base>
void* bzip2_allocator<Alloc, Base>::allocate(void* self, int items, int size)
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
void bzip2_allocator<Alloc, Base>::deallocate(void* self, void* address)
{ 
char* ptr = reinterpret_cast<char*>(address) - sizeof(size_type);
size_type len = *reinterpret_cast<size_type*>(ptr) + sizeof(size_type);
static_cast<allocator_type*>(self)->deallocate(ptr, len); 
}


template<typename Alloc>
bzip2_compressor_impl<Alloc>::bzip2_compressor_impl(const bzip2_params& p)
: bzip2_base(p), eof_(false) { }

template<typename Alloc>
bzip2_compressor_impl<Alloc>::~bzip2_compressor_impl()
{ (void) bzip2_base::end(true, std::nothrow); }

template<typename Alloc>
bool bzip2_compressor_impl<Alloc>::filter
( const char*& src_begin, const char* src_end,
char*& dest_begin, char* dest_end, bool flush )
{
if (!ready()) init();
if (eof_) return false;
before(src_begin, src_end, dest_begin, dest_end);
int result = compress(flush ? bzip2::finish : bzip2::run);
after(src_begin, dest_begin);
bzip2_error::check BOOST_PREVENT_MACRO_SUBSTITUTION(result);
return !(eof_ = result == bzip2::stream_end);
}

template<typename Alloc>
void bzip2_compressor_impl<Alloc>::close() 
{ 
try {
end(true);
} catch (...) { 
eof_ = false; 
throw;
}
eof_ = false;
}

template<typename Alloc>
inline void bzip2_compressor_impl<Alloc>::init() 
{ bzip2_base::init(true, static_cast<bzip2_allocator<Alloc>&>(*this)); }


template<typename Alloc>
bzip2_decompressor_impl<Alloc>::bzip2_decompressor_impl(bool small)
: bzip2_base(bzip2_params(small)), eof_(false) { }

template<typename Alloc>
bzip2_decompressor_impl<Alloc>::~bzip2_decompressor_impl()
{ (void) bzip2_base::end(false, std::nothrow); }

template<typename Alloc>
bool bzip2_decompressor_impl<Alloc>::filter
( const char*& src_begin, const char* src_end,
char*& dest_begin, char* dest_end, bool flush )
{
do {
if (eof_) {
if(src_begin == src_end)
return false;
else
close();
}
if (!ready()) 
init();
before(src_begin, src_end, dest_begin, dest_end);
int result = decompress();
if(result == bzip2::ok && flush)
result = check_end(src_begin, dest_begin);
after(src_begin, dest_begin);
bzip2_error::check BOOST_PREVENT_MACRO_SUBSTITUTION(result);
eof_ = result == bzip2::stream_end;
} while (eof_ && src_begin != src_end && dest_begin != dest_end);
return true; 
}

template<typename Alloc>
void bzip2_decompressor_impl<Alloc>::close() 
{ 
try {
end(false);
} catch (...) { 
eof_ = false; 
throw;
}
eof_ = false;
}

template<typename Alloc>
inline void bzip2_decompressor_impl<Alloc>::init()
{ bzip2_base::init(false, static_cast<bzip2_allocator<Alloc>&>(*this)); }
} 


template<typename Alloc>
basic_bzip2_compressor<Alloc>::basic_bzip2_compressor
(const bzip2_params& p, std::streamsize buffer_size) 
: base_type(buffer_size, p) 
{ }


template<typename Alloc>
basic_bzip2_decompressor<Alloc>::basic_bzip2_decompressor
(bool small, std::streamsize buffer_size) 
: base_type(buffer_size, small)
{ }


} } 

#include <boost/config/abi_suffix.hpp> 
#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#endif 
