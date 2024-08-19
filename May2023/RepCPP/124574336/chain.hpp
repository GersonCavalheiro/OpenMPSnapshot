

#ifndef BOOST_IOSTREAMS_DETAIL_CHAIN_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_CHAIN_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <exception>
#include <iterator>                             
#include <list>
#include <memory>                               
#include <stdexcept>                            
#include <boost/checked_delete.hpp>
#include <boost/config.hpp>                     
#include <boost/detail/workaround.hpp>          
#include <boost/core/typeinfo.hpp>
#include <boost/iostreams/constants.hpp>
#include <boost/iostreams/detail/access_control.hpp>
#include <boost/iostreams/detail/char_traits.hpp>
#include <boost/iostreams/detail/push.hpp>
#include <boost/iostreams/detail/streambuf.hpp> 
#include <boost/iostreams/detail/wrap_unwrap.hpp>
#include <boost/iostreams/device/null.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/iostreams/traits.hpp>           
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/next_prior.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type.hpp>
#include <boost/iostreams/detail/execute.hpp>

#if defined(__GNUC__) || \
defined(_AIX) || \
(defined(__sgi) && defined(__host_mips)) || \
(defined(linux) && defined(__INTEL_COMPILER) && defined(__ICC)) \

# include <cstring>
# define BOOST_IOSTREAMS_COMPARE_TYPE_ID(X,Y) \
(std::strcmp((X).name(),(Y).name()) == 0)
#else
# define BOOST_IOSTREAMS_COMPARE_TYPE_ID(X,Y) ((X)==(Y))
#endif

#define BOOST_IOSTREAMS_COMPONENT_TYPE(chain, index) \
chain.component_type( index ) \


#define BOOST_IOSTREAMS_COMPONENT(chain, index, target) \
chain.component< target >( index ) \


namespace boost { namespace iostreams {


namespace detail {

template<typename Chain> class chain_client;


template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
class chain_base {
public:
typedef Ch                                     char_type;
BOOST_IOSTREAMS_STREAMBUF_TYPEDEFS(Tr)
typedef Alloc                                  allocator_type;
typedef Mode                                   mode;
struct category
: Mode,
device_tag
{ };
typedef chain_client<Self>                     client_type;
friend class chain_client<Self>;
private:
typedef linked_streambuf<Ch>                   streambuf_type;
typedef std::list<streambuf_type*>             list_type;
typedef chain_base<Self, Ch, Tr, Alloc, Mode>  my_type;
protected:
chain_base() : pimpl_(new chain_impl) { }
chain_base(const chain_base& rhs): pimpl_(rhs.pimpl_) { }
public:

BOOST_STATIC_ASSERT((!is_convertible<mode, dual_use>::value));


void set_device_buffer_size(std::streamsize n) 
{ pimpl_->device_buffer_size_ = n; }

void set_filter_buffer_size(std::streamsize n) 
{ pimpl_->filter_buffer_size_ = n; }

void set_pback_size(std::streamsize n) 
{ pimpl_->pback_size_ = n; }


std::streamsize read(char_type* s, std::streamsize n);
std::streamsize write(const char_type* s, std::streamsize n);
std::streampos seek(stream_offset off, BOOST_IOS::seekdir way);


const boost::core::typeinfo& component_type(int n) const
{
if (static_cast<size_type>(n) >= size())
boost::throw_exception(std::out_of_range("bad chain offset"));
return (*boost::next(list().begin(), n))->component_type();
}

template<int N>
const boost::core::typeinfo& component_type() const { return component_type(N); }

template<typename T>
T* component(int n) const { return component(n, boost::type<T>()); }

template<int N, typename T> 
T* component() const { return component<T>(N); }

#if !BOOST_WORKAROUND(BOOST_MSVC, == 1310)
private:
#endif
template<typename T>
T* component(int n, boost::type<T>) const
{
if (static_cast<size_type>(n) >= size())
boost::throw_exception(std::out_of_range("bad chain offset"));
streambuf_type* link = *boost::next(list().begin(), n);
if (BOOST_IOSTREAMS_COMPARE_TYPE_ID(link->component_type(), BOOST_CORE_TYPEID(T)))
return static_cast<T*>(link->component_impl());
else
return 0;
}
public:


typedef typename list_type::size_type size_type;
streambuf_type& front() { return *list().front(); }
BOOST_IOSTREAMS_DEFINE_PUSH(push, mode, char_type, push_impl)
void pop();
bool empty() const { return list().empty(); }
size_type size() const { return list().size(); }
void reset();


bool is_complete() const;
bool auto_close() const;
void set_auto_close(bool close);
bool sync() { return front().BOOST_IOSTREAMS_PUBSYNC() != -1; }
bool strict_sync();
private:
template<typename T>
void push_impl(const T& t, std::streamsize buffer_size = -1, 
std::streamsize pback_size = -1)
{
typedef typename iostreams::category_of<T>::type  category;
typedef typename unwrap_ios<T>::type              component_type;
typedef stream_buffer<
component_type,
BOOST_IOSTREAMS_CHAR_TRAITS(char_type),
Alloc, Mode
>                                         streambuf_t;
typedef typename list_type::iterator              iterator;
BOOST_STATIC_ASSERT((is_convertible<category, Mode>::value));
if (is_complete())
boost::throw_exception(std::logic_error("chain complete"));
streambuf_type* prev = !empty() ? list().back() : 0;
buffer_size =
buffer_size != -1 ?
buffer_size :
iostreams::optimal_buffer_size(t);
pback_size =
pback_size != -1 ?
pback_size :
pimpl_->pback_size_;

#if defined(BOOST_NO_CXX11_SMART_PTR)

std::auto_ptr<streambuf_t>
buf(new streambuf_t(t, buffer_size, pback_size));

#else

std::unique_ptr<streambuf_t>
buf(new streambuf_t(t, buffer_size, pback_size));

#endif

list().push_back(buf.get());
buf.release();
if (is_device<component_type>::value) {
pimpl_->flags_ |= f_complete | f_open;
for ( iterator first = list().begin(),
last = list().end();
first != last;
++first )
{
(*first)->set_needs_close();
}
}
if (prev) prev->set_next(list().back());
notify();
}

list_type& list() { return pimpl_->links_; }
const list_type& list() const { return pimpl_->links_; }
void register_client(client_type* client) { pimpl_->client_ = client; }
void notify() { if (pimpl_->client_) pimpl_->client_->notify(); }


static void close(streambuf_type* b, BOOST_IOS::openmode m)
{
if (m == BOOST_IOS::out && is_convertible<Mode, output>::value)
b->BOOST_IOSTREAMS_PUBSYNC();
b->close(m);
}

static void set_next(streambuf_type* b, streambuf_type* next)
{ b->set_next(next); }

static void set_auto_close(streambuf_type* b, bool close)
{ b->set_auto_close(close); }

struct closer {
typedef streambuf_type* argument_type;
typedef void result_type;
closer(BOOST_IOS::openmode m) : mode_(m) { }
void operator() (streambuf_type* b)
{
close(b, mode_);
}
BOOST_IOS::openmode mode_;
};
friend struct closer;

enum flags {
f_complete = 1,
f_open = 2,
f_auto_close = 4
};

struct chain_impl {
chain_impl()
: client_(0), device_buffer_size_(default_device_buffer_size),
filter_buffer_size_(default_filter_buffer_size),
pback_size_(default_pback_buffer_size),
flags_(f_auto_close)
{ }
~chain_impl()
{
try { close(); } catch (...) { }
try { reset(); } catch (...) { }
}
void close()
{
if ((flags_ & f_open) != 0) {
flags_ &= ~f_open;
stream_buffer< basic_null_device<Ch, Mode> > null;
if ((flags_ & f_complete) == 0) {
null.open(basic_null_device<Ch, Mode>());
set_next(links_.back(), &null);
}
links_.front()->BOOST_IOSTREAMS_PUBSYNC();
try {
boost::iostreams::detail::execute_foreach(
links_.rbegin(), links_.rend(), 
closer(BOOST_IOS::in)
);
} catch (...) {
try {
boost::iostreams::detail::execute_foreach(
links_.begin(), links_.end(), 
closer(BOOST_IOS::out)
);
} catch (...) { }
throw;
}
boost::iostreams::detail::execute_foreach(
links_.begin(), links_.end(), 
closer(BOOST_IOS::out)
);
}
}
void reset()
{
typedef typename list_type::iterator iterator;
for ( iterator first = links_.begin(),
last = links_.end();
first != last;
++first )
{
if ( (flags_ & f_complete) == 0 ||
(flags_ & f_auto_close) == 0 )
{
set_auto_close(*first, false);
}
streambuf_type* buf = 0;
std::swap(buf, *first);
delete buf;
}
links_.clear();
flags_ &= ~f_complete;
flags_ &= ~f_open;
}
list_type        links_;
client_type*     client_;
std::streamsize  device_buffer_size_,
filter_buffer_size_,
pback_size_;
int              flags_;
};
friend struct chain_impl;


private:
shared_ptr<chain_impl> pimpl_;
};

} 

#define BOOST_IOSTREAMS_DECL_CHAIN(name_, default_char_) \
template< typename Mode, typename Ch = default_char_, \
typename Tr = BOOST_IOSTREAMS_CHAR_TRAITS(Ch), \
typename Alloc = std::allocator<Ch> > \
class name_ : public boost::iostreams::detail::chain_base< \
name_<Mode, Ch, Tr, Alloc>, \
Ch, Tr, Alloc, Mode \
> \
{ \
public: \
struct category : device_tag, Mode { }; \
typedef Mode                                   mode; \
private: \
typedef boost::iostreams::detail::chain_base< \
name_<Mode, Ch, Tr, Alloc>, \
Ch, Tr, Alloc, Mode \
>                                      base_type; \
public: \
typedef Ch                                     char_type; \
typedef Tr                                     traits_type; \
typedef typename traits_type::int_type         int_type; \
typedef typename traits_type::off_type         off_type; \
name_() { } \
name_(const name_& rhs) : base_type(rhs) { } \
name_& operator=(const name_& rhs) \
{ base_type::operator=(rhs); return *this; } \
}; \

BOOST_IOSTREAMS_DECL_CHAIN(chain, char)
BOOST_IOSTREAMS_DECL_CHAIN(wchain, wchar_t)
#undef BOOST_IOSTREAMS_DECL_CHAIN


namespace detail {

template<typename Chain>
class chain_client {
public:
typedef Chain                             chain_type;
typedef typename chain_type::char_type    char_type;
typedef typename chain_type::traits_type  traits_type;
typedef typename chain_type::size_type    size_type;
typedef typename chain_type::mode         mode;

chain_client(chain_type* chn = 0) : chain_(chn ) { }
chain_client(chain_client* client) : chain_(client->chain_) { }
virtual ~chain_client() { }

const boost::core::typeinfo& component_type(int n) const
{ return chain_->component_type(n); }

template<int N>
const boost::core::typeinfo& component_type() const
{ return chain_->BOOST_NESTED_TEMPLATE component_type<N>(); }

template<typename T>
T* component(int n) const
{ return chain_->BOOST_NESTED_TEMPLATE component<T>(n); }

template<int N, typename T>
T* component() const
{ return chain_->BOOST_NESTED_TEMPLATE component<N, T>(); }

bool is_complete() const { return chain_->is_complete(); }
bool auto_close() const { return chain_->auto_close(); }
void set_auto_close(bool close) { chain_->set_auto_close(close); }
bool strict_sync() { return chain_->strict_sync(); }
void set_device_buffer_size(std::streamsize n)
{ chain_->set_device_buffer_size(n); }
void set_filter_buffer_size(std::streamsize n)
{ chain_->set_filter_buffer_size(n); }
void set_pback_size(std::streamsize n) { chain_->set_pback_size(n); }
BOOST_IOSTREAMS_DEFINE_PUSH(push, mode, char_type, push_impl)
void pop() { chain_->pop(); }
bool empty() const { return chain_->empty(); }
size_type size() const { return chain_->size(); }
void reset() { chain_->reset(); }

chain_type filters() { return *chain_; }
chain_type filters() const { return *chain_; }
protected:
template<typename T>
void push_impl(const T& t BOOST_IOSTREAMS_PUSH_PARAMS())
{ chain_->push(t BOOST_IOSTREAMS_PUSH_ARGS()); }
chain_type& ref() { return *chain_; }
void set_chain(chain_type* c)
{ chain_ = c; chain_->register_client(this); }
#if !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS) && \
(!BOOST_WORKAROUND(BOOST_BORLANDC, < 0x600))
template<typename S, typename C, typename T, typename A, typename M>
friend class chain_base;
#else
public:
#endif
virtual void notify() { }
private:
chain_type* chain_;
};


template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
inline std::streamsize chain_base<Self, Ch, Tr, Alloc, Mode>::read
(char_type* s, std::streamsize n)
{ return iostreams::read(*list().front(), s, n); }

template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
inline std::streamsize chain_base<Self, Ch, Tr, Alloc, Mode>::write
(const char_type* s, std::streamsize n)
{ return iostreams::write(*list().front(), s, n); }

template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
inline std::streampos chain_base<Self, Ch, Tr, Alloc, Mode>::seek
(stream_offset off, BOOST_IOS::seekdir way)
{ return iostreams::seek(*list().front(), off, way); }

template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
void chain_base<Self, Ch, Tr, Alloc, Mode>::reset()
{
using namespace std;
pimpl_->close();
pimpl_->reset();
}

template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
bool chain_base<Self, Ch, Tr, Alloc, Mode>::is_complete() const
{
return (pimpl_->flags_ & f_complete) != 0;
}

template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
bool chain_base<Self, Ch, Tr, Alloc, Mode>::auto_close() const
{
return (pimpl_->flags_ & f_auto_close) != 0;
}

template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
void chain_base<Self, Ch, Tr, Alloc, Mode>::set_auto_close(bool close)
{
pimpl_->flags_ =
(pimpl_->flags_ & ~f_auto_close) |
(close ? f_auto_close : 0);
}

template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
bool chain_base<Self, Ch, Tr, Alloc, Mode>::strict_sync()
{
typedef typename list_type::iterator iterator;
bool result = true;
for ( iterator first = list().begin(),
last = list().end();
first != last;
++first )
{
bool s = (*first)->strict_sync();
result = result && s;
}
return result;
}

template<typename Self, typename Ch, typename Tr, typename Alloc, typename Mode>
void chain_base<Self, Ch, Tr, Alloc, Mode>::pop()
{
BOOST_ASSERT(!empty());
if (auto_close())
pimpl_->close();
streambuf_type* buf = 0;
std::swap(buf, list().back());
buf->set_auto_close(false);
buf->set_next(0);
delete buf;
list().pop_back();
pimpl_->flags_ &= ~f_complete;
if (auto_close() || list().empty())
pimpl_->flags_ &= ~f_open;
}

} 

} } 

#endif 
