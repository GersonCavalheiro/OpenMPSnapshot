



#ifndef BOOST_IOSTREAMS_SYMMETRIC_FILTER_HPP_INCLUDED
#define BOOST_IOSTREAMS_SYMMETRIC_FILTER_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <memory>                               
#include <boost/config.hpp>                     
#include <boost/iostreams/char_traits.hpp>
#include <boost/iostreams/constants.hpp>        
#include <boost/iostreams/detail/buffer.hpp>
#include <boost/iostreams/detail/char_traits.hpp>
#include <boost/iostreams/detail/config/limits.hpp>
#include <boost/iostreams/detail/ios.hpp>  
#include <boost/iostreams/detail/template_params.hpp>
#include <boost/iostreams/traits.hpp>
#include <boost/iostreams/operations.hpp>       
#include <boost/iostreams/pipeline.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/iostreams/detail/config/disable_warnings.hpp>  

namespace boost { namespace iostreams {

template< typename SymmetricFilter,
typename Alloc =
std::allocator<
BOOST_DEDUCED_TYPENAME char_type_of<SymmetricFilter>::type
> >
class symmetric_filter {
public:
typedef typename char_type_of<SymmetricFilter>::type      char_type;
typedef BOOST_IOSTREAMS_CHAR_TRAITS(char_type)            traits_type;
typedef std::basic_string<char_type, traits_type, Alloc>  string_type;
struct category
: dual_use,
filter_tag,
multichar_tag,
closable_tag
{ };

#define BOOST_PP_LOCAL_MACRO(n) \
BOOST_IOSTREAMS_TEMPLATE_PARAMS(n, T) \
explicit symmetric_filter( \
std::streamsize buffer_size BOOST_PP_COMMA_IF(n) \
BOOST_PP_ENUM_BINARY_PARAMS(n, const T, &t) ) \
: pimpl_(new impl(buffer_size BOOST_PP_COMMA_IF(n) \
BOOST_PP_ENUM_PARAMS(n, t))) \
{ BOOST_ASSERT(buffer_size > 0); } \

#define BOOST_PP_LOCAL_LIMITS (0, BOOST_IOSTREAMS_MAX_FORWARDING_ARITY)
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_LOCAL_MACRO

template<typename Source>
std::streamsize read(Source& src, char_type* s, std::streamsize n)
{
using namespace std;
if (!(state() & f_read))
begin_read();

buffer_type&  buf = pimpl_->buf_;
int           status = (state() & f_eof) != 0 ? f_eof : f_good;
char_type    *next_s = s,
*end_s = s + n;
while (true)
{
bool flush = status == f_eof;
if (buf.ptr() != buf.eptr() || flush) {
const char_type* next = buf.ptr();
bool done =
!filter().filter(next, buf.eptr(), next_s, end_s, flush);
buf.ptr() = buf.data() + (next - buf.data());
if (done)
return detail::check_eof(
static_cast<std::streamsize>(next_s - s)
);
}

if ( (status == f_would_block && buf.ptr() == buf.eptr()) ||
next_s == end_s )
{
return static_cast<std::streamsize>(next_s - s);
}

if (status == f_good)
status = fill(src);
}
}

template<typename Sink>
std::streamsize write(Sink& snk, const char_type* s, std::streamsize n)
{
if (!(state() & f_write))
begin_write();

buffer_type&     buf = pimpl_->buf_;
const char_type *next_s, *end_s;
for (next_s = s, end_s = s + n; next_s != end_s; ) {
if (buf.ptr() == buf.eptr() && !flush(snk))
break;
if(!filter().filter(next_s, end_s, buf.ptr(), buf.eptr(), false)) {
flush(snk);
break;
}
}
return static_cast<std::streamsize>(next_s - s);
}

template<typename Sink>
void close(Sink& snk, BOOST_IOS::openmode mode)
{
if (mode == BOOST_IOS::out) {

if (!(state() & f_write))
begin_write();

try {
buffer_type&     buf = pimpl_->buf_;
char_type        dummy;
const char_type* end = &dummy;
bool             again = true;
while (again) {
if (buf.ptr() != buf.eptr())
again = filter().filter( end, end, buf.ptr(),
buf.eptr(), true );
flush(snk);
}
} catch (...) {
try { close_impl(); } catch (...) { }
throw;
}
close_impl();
} else {
close_impl();
}
}
SymmetricFilter& filter() { return *pimpl_; }
string_type unconsumed_input() const;

#if !BOOST_WORKAROUND(__DECCXX_VER, BOOST_TESTED_AT(60590042)) 
private:
#endif
typedef detail::buffer<char_type, Alloc> buffer_type;
private:
buffer_type& buf() { return pimpl_->buf_; }
const buffer_type& buf() const { return pimpl_->buf_; }
int& state() { return pimpl_->state_; }
void begin_read();
void begin_write();

template<typename Source>
int fill(Source& src)
{
std::streamsize amt = iostreams::read(src, buf().data(), buf().size());
if (amt == -1) {
state() |= f_eof;
return f_eof;
}
buf().set(0, amt);
return amt != 0 ? f_good : f_would_block;
}

template<typename Sink>
bool flush(Sink& snk)
{
typedef typename iostreams::category_of<Sink>::type  category;
typedef is_convertible<category, output>             can_write;
return flush(snk, can_write());
}

template<typename Sink>
bool flush(Sink& snk, mpl::true_)
{
std::streamsize amt =
static_cast<std::streamsize>(buf().ptr() - buf().data());
std::streamsize result =
boost::iostreams::write(snk, buf().data(), amt);
if (result < amt && result > 0)
traits_type::move(buf().data(), buf().data() + result, amt - result);
buf().set(amt - result, buf().size());
return result != 0;
}

template<typename Sink>
bool flush(Sink&, mpl::false_) { return true;}

void close_impl();

enum flag_type {
f_read   = 1,
f_write  = f_read << 1,
f_eof    = f_write << 1,
f_good,
f_would_block
};

struct impl : SymmetricFilter {

#define BOOST_PP_LOCAL_MACRO(n) \
BOOST_IOSTREAMS_TEMPLATE_PARAMS(n, T) \
impl( std::streamsize buffer_size BOOST_PP_COMMA_IF(n) \
BOOST_PP_ENUM_BINARY_PARAMS(n, const T, &t) ) \
: SymmetricFilter(BOOST_PP_ENUM_PARAMS(n, t)), \
buf_(buffer_size), state_(0) \
{ } \

#define BOOST_PP_LOCAL_LIMITS (0, BOOST_IOSTREAMS_MAX_FORWARDING_ARITY)
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_LOCAL_MACRO

buffer_type  buf_;
int          state_;
};

shared_ptr<impl> pimpl_;
};
BOOST_IOSTREAMS_PIPABLE(symmetric_filter, 2)


template<typename SymmetricFilter, typename Alloc>
void symmetric_filter<SymmetricFilter, Alloc>::begin_read()
{
BOOST_ASSERT(!(state() & f_write));
state() |= f_read;
buf().set(0, 0);
}

template<typename SymmetricFilter, typename Alloc>
void symmetric_filter<SymmetricFilter, Alloc>::begin_write()
{
BOOST_ASSERT(!(state() & f_read));
state() |= f_write;
buf().set(0, buf().size());
}

template<typename SymmetricFilter, typename Alloc>
void symmetric_filter<SymmetricFilter, Alloc>::close_impl()
{
state() = 0;
buf().set(0, 0);
filter().close();
}

template<typename SymmetricFilter, typename Alloc>
typename symmetric_filter<SymmetricFilter, Alloc>::string_type
symmetric_filter<SymmetricFilter, Alloc>::unconsumed_input() const
{ return string_type(buf().ptr(), buf().eptr()); }


} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp>  

#endif 
