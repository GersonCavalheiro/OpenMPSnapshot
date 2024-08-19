
#ifndef BOOST_LOCKFREE_SPSC_QUEUE_HPP_INCLUDED
#define BOOST_LOCKFREE_SPSC_QUEUE_HPP_INCLUDED

#include <algorithm>
#include <memory>

#include <boost/aligned_storage.hpp>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/core/allocator_access.hpp>
#include <boost/utility.hpp>
#include <boost/next_prior.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/config.hpp> 

#include <boost/type_traits/has_trivial_destructor.hpp>
#include <boost/type_traits/is_convertible.hpp>

#include <boost/lockfree/detail/atomic.hpp>
#include <boost/lockfree/detail/copy_payload.hpp>
#include <boost/lockfree/detail/parameter.hpp>
#include <boost/lockfree/detail/prefix.hpp>

#include <boost/lockfree/lockfree_forward.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost    {
namespace lockfree {
namespace detail   {

typedef parameter::parameters<boost::parameter::optional<tag::capacity>,
boost::parameter::optional<tag::allocator>
> ringbuffer_signature;

template <typename T>
class ringbuffer_base
{
#ifndef BOOST_DOXYGEN_INVOKED
protected:
typedef std::size_t size_t;
static const int padding_size = BOOST_LOCKFREE_CACHELINE_BYTES - sizeof(size_t);
atomic<size_t> write_index_;
char padding1[padding_size]; 
atomic<size_t> read_index_;

BOOST_DELETED_FUNCTION(ringbuffer_base(ringbuffer_base const&))
BOOST_DELETED_FUNCTION(ringbuffer_base& operator= (ringbuffer_base const&))

protected:
ringbuffer_base(void):
write_index_(0), read_index_(0)
{}

static size_t next_index(size_t arg, size_t max_size)
{
size_t ret = arg + 1;
while (BOOST_UNLIKELY(ret >= max_size))
ret -= max_size;
return ret;
}

static size_t read_available(size_t write_index, size_t read_index, size_t max_size)
{
if (write_index >= read_index)
return write_index - read_index;

const size_t ret = write_index + max_size - read_index;
return ret;
}

static size_t write_available(size_t write_index, size_t read_index, size_t max_size)
{
size_t ret = read_index - write_index - 1;
if (write_index >= read_index)
ret += max_size;
return ret;
}

size_t read_available(size_t max_size) const
{
size_t write_index = write_index_.load(memory_order_acquire);
const size_t read_index  = read_index_.load(memory_order_relaxed);
return read_available(write_index, read_index, max_size);
}

size_t write_available(size_t max_size) const
{
size_t write_index = write_index_.load(memory_order_relaxed);
const size_t read_index  = read_index_.load(memory_order_acquire);
return write_available(write_index, read_index, max_size);
}

bool push(T const & t, T * buffer, size_t max_size)
{
const size_t write_index = write_index_.load(memory_order_relaxed);  
const size_t next = next_index(write_index, max_size);

if (next == read_index_.load(memory_order_acquire))
return false; 

new (buffer + write_index) T(t); 

write_index_.store(next, memory_order_release);

return true;
}

size_t push(const T * input_buffer, size_t input_count, T * internal_buffer, size_t max_size)
{
return push(input_buffer, input_buffer + input_count, internal_buffer, max_size) - input_buffer;
}

template <typename ConstIterator>
ConstIterator push(ConstIterator begin, ConstIterator end, T * internal_buffer, size_t max_size)
{

const size_t write_index = write_index_.load(memory_order_relaxed);  
const size_t read_index  = read_index_.load(memory_order_acquire);
const size_t avail = write_available(write_index, read_index, max_size);

if (avail == 0)
return begin;

size_t input_count = std::distance(begin, end);
input_count = (std::min)(input_count, avail);

size_t new_write_index = write_index + input_count;

const ConstIterator last = boost::next(begin, input_count);

if (write_index + input_count > max_size) {

const size_t count0 = max_size - write_index;
const ConstIterator midpoint = boost::next(begin, count0);

std::uninitialized_copy(begin, midpoint, internal_buffer + write_index);
std::uninitialized_copy(midpoint, last, internal_buffer);
new_write_index -= max_size;
} else {
std::uninitialized_copy(begin, last, internal_buffer + write_index);

if (new_write_index == max_size)
new_write_index = 0;
}

write_index_.store(new_write_index, memory_order_release);
return last;
}

template <typename Functor>
bool consume_one(Functor & functor, T * buffer, size_t max_size)
{
const size_t write_index = write_index_.load(memory_order_acquire);
const size_t read_index  = read_index_.load(memory_order_relaxed); 
if ( empty(write_index, read_index) )
return false;

T & object_to_consume = buffer[read_index];
functor( object_to_consume );
object_to_consume.~T();

size_t next = next_index(read_index, max_size);
read_index_.store(next, memory_order_release);
return true;
}

template <typename Functor>
bool consume_one(Functor const & functor, T * buffer, size_t max_size)
{
const size_t write_index = write_index_.load(memory_order_acquire);
const size_t read_index  = read_index_.load(memory_order_relaxed); 
if ( empty(write_index, read_index) )
return false;

T & object_to_consume = buffer[read_index];
functor( object_to_consume );
object_to_consume.~T();

size_t next = next_index(read_index, max_size);
read_index_.store(next, memory_order_release);
return true;
}

template <typename Functor>
size_t consume_all (Functor const & functor, T * internal_buffer, size_t max_size)
{
const size_t write_index = write_index_.load(memory_order_acquire);
const size_t read_index = read_index_.load(memory_order_relaxed); 

const size_t avail = read_available(write_index, read_index, max_size);

if (avail == 0)
return 0;

const size_t output_count = avail;

size_t new_read_index = read_index + output_count;

if (read_index + output_count > max_size) {

const size_t count0 = max_size - read_index;
const size_t count1 = output_count - count0;

run_functor_and_delete(internal_buffer + read_index, internal_buffer + max_size, functor);
run_functor_and_delete(internal_buffer, internal_buffer + count1, functor);

new_read_index -= max_size;
} else {
run_functor_and_delete(internal_buffer + read_index, internal_buffer + read_index + output_count, functor);

if (new_read_index == max_size)
new_read_index = 0;
}

read_index_.store(new_read_index, memory_order_release);
return output_count;
}

template <typename Functor>
size_t consume_all (Functor & functor, T * internal_buffer, size_t max_size)
{
const size_t write_index = write_index_.load(memory_order_acquire);
const size_t read_index = read_index_.load(memory_order_relaxed); 

const size_t avail = read_available(write_index, read_index, max_size);

if (avail == 0)
return 0;

const size_t output_count = avail;

size_t new_read_index = read_index + output_count;

if (read_index + output_count > max_size) {

const size_t count0 = max_size - read_index;
const size_t count1 = output_count - count0;

run_functor_and_delete(internal_buffer + read_index, internal_buffer + max_size, functor);
run_functor_and_delete(internal_buffer, internal_buffer + count1, functor);

new_read_index -= max_size;
} else {
run_functor_and_delete(internal_buffer + read_index, internal_buffer + read_index + output_count, functor);

if (new_read_index == max_size)
new_read_index = 0;
}

read_index_.store(new_read_index, memory_order_release);
return output_count;
}

size_t pop (T * output_buffer, size_t output_count, T * internal_buffer, size_t max_size)
{
const size_t write_index = write_index_.load(memory_order_acquire);
const size_t read_index = read_index_.load(memory_order_relaxed); 

const size_t avail = read_available(write_index, read_index, max_size);

if (avail == 0)
return 0;

output_count = (std::min)(output_count, avail);

size_t new_read_index = read_index + output_count;

if (read_index + output_count > max_size) {

const size_t count0 = max_size - read_index;
const size_t count1 = output_count - count0;

copy_and_delete(internal_buffer + read_index, internal_buffer + max_size, output_buffer);
copy_and_delete(internal_buffer, internal_buffer + count1, output_buffer + count0);

new_read_index -= max_size;
} else {
copy_and_delete(internal_buffer + read_index, internal_buffer + read_index + output_count, output_buffer);
if (new_read_index == max_size)
new_read_index = 0;
}

read_index_.store(new_read_index, memory_order_release);
return output_count;
}

template <typename OutputIterator>
size_t pop_to_output_iterator (OutputIterator it, T * internal_buffer, size_t max_size)
{
const size_t write_index = write_index_.load(memory_order_acquire);
const size_t read_index = read_index_.load(memory_order_relaxed); 

const size_t avail = read_available(write_index, read_index, max_size);
if (avail == 0)
return 0;

size_t new_read_index = read_index + avail;

if (read_index + avail > max_size) {

const size_t count0 = max_size - read_index;
const size_t count1 = avail - count0;

it = copy_and_delete(internal_buffer + read_index, internal_buffer + max_size, it);
copy_and_delete(internal_buffer, internal_buffer + count1, it);

new_read_index -= max_size;
} else {
copy_and_delete(internal_buffer + read_index, internal_buffer + read_index + avail, it);
if (new_read_index == max_size)
new_read_index = 0;
}

read_index_.store(new_read_index, memory_order_release);
return avail;
}

const T& front(const T * internal_buffer) const
{
const size_t read_index = read_index_.load(memory_order_relaxed); 
return *(internal_buffer + read_index);
}

T& front(T * internal_buffer)
{
const size_t read_index = read_index_.load(memory_order_relaxed); 
return *(internal_buffer + read_index);
}
#endif


public:

void reset(void)
{
if ( !boost::has_trivial_destructor<T>::value ) {

T dummy_element;
while (pop(dummy_element))
{}
} else {
write_index_.store(0, memory_order_relaxed);
read_index_.store(0, memory_order_release);
}
}


bool empty(void)
{
return empty(write_index_.load(memory_order_relaxed), read_index_.load(memory_order_relaxed));
}


bool is_lock_free(void) const
{
return write_index_.is_lock_free() && read_index_.is_lock_free();
}

private:
bool empty(size_t write_index, size_t read_index)
{
return write_index == read_index;
}

template< class OutputIterator >
OutputIterator copy_and_delete( T * first, T * last, OutputIterator out )
{
if (boost::has_trivial_destructor<T>::value) {
return std::copy(first, last, out); 
} else {
for (; first != last; ++first, ++out) {
*out = *first;
first->~T();
}
return out;
}
}

template< class Functor >
void run_functor_and_delete( T * first, T * last, Functor & functor )
{
for (; first != last; ++first) {
functor(*first);
first->~T();
}
}

template< class Functor >
void run_functor_and_delete( T * first, T * last, Functor const & functor )
{
for (; first != last; ++first) {
functor(*first);
first->~T();
}
}
};

template <typename T, std::size_t MaxSize>
class compile_time_sized_ringbuffer:
public ringbuffer_base<T>
{
typedef std::size_t size_type;
static const std::size_t max_size = MaxSize + 1;

typedef typename boost::aligned_storage<max_size * sizeof(T),
boost::alignment_of<T>::value
>::type storage_type;

storage_type storage_;

T * data()
{
return static_cast<T*>(storage_.address());
}

const T * data() const
{
return static_cast<const T*>(storage_.address());
}

protected:
size_type max_number_of_elements() const
{
return max_size;
}

public:
bool push(T const & t)
{
return ringbuffer_base<T>::push(t, data(), max_size);
}

template <typename Functor>
bool consume_one(Functor & f)
{
return ringbuffer_base<T>::consume_one(f, data(), max_size);
}

template <typename Functor>
bool consume_one(Functor const & f)
{
return ringbuffer_base<T>::consume_one(f, data(), max_size);
}

template <typename Functor>
size_type consume_all(Functor & f)
{
return ringbuffer_base<T>::consume_all(f, data(), max_size);
}

template <typename Functor>
size_type consume_all(Functor const & f)
{
return ringbuffer_base<T>::consume_all(f, data(), max_size);
}

size_type push(T const * t, size_type size)
{
return ringbuffer_base<T>::push(t, size, data(), max_size);
}

template <size_type size>
size_type push(T const (&t)[size])
{
return push(t, size);
}

template <typename ConstIterator>
ConstIterator push(ConstIterator begin, ConstIterator end)
{
return ringbuffer_base<T>::push(begin, end, data(), max_size);
}

size_type pop(T * ret, size_type size)
{
return ringbuffer_base<T>::pop(ret, size, data(), max_size);
}

template <typename OutputIterator>
size_type pop_to_output_iterator(OutputIterator it)
{
return ringbuffer_base<T>::pop_to_output_iterator(it, data(), max_size);
}

const T& front(void) const
{
return ringbuffer_base<T>::front(data());
}

T& front(void)
{
return ringbuffer_base<T>::front(data());
}
};

template <typename T, typename Alloc>
class runtime_sized_ringbuffer:
public ringbuffer_base<T>,
private Alloc
{
typedef std::size_t size_type;
size_type max_elements_;
#ifdef BOOST_NO_CXX11_ALLOCATOR
typedef typename Alloc::pointer pointer;
#else
typedef std::allocator_traits<Alloc> allocator_traits;
typedef typename allocator_traits::pointer pointer;
#endif
pointer array_;

protected:
size_type max_number_of_elements() const
{
return max_elements_;
}

public:
explicit runtime_sized_ringbuffer(size_type max_elements):
max_elements_(max_elements + 1)
{
#ifdef BOOST_NO_CXX11_ALLOCATOR
array_ = Alloc::allocate(max_elements_);
#else
Alloc& alloc = *this;
array_ = allocator_traits::allocate(alloc, max_elements_);
#endif
}

template <typename U>
runtime_sized_ringbuffer(typename boost::allocator_rebind<Alloc, U>::type const & alloc, size_type max_elements):
Alloc(alloc), max_elements_(max_elements + 1)
{
#ifdef BOOST_NO_CXX11_ALLOCATOR
array_ = Alloc::allocate(max_elements_);
#else
Alloc& allocator = *this;
array_ = allocator_traits::allocate(allocator, max_elements_);
#endif
}

runtime_sized_ringbuffer(Alloc const & alloc, size_type max_elements):
Alloc(alloc), max_elements_(max_elements + 1)
{
#ifdef BOOST_NO_CXX11_ALLOCATOR
array_ = Alloc::allocate(max_elements_);
#else
Alloc& allocator = *this;
array_ = allocator_traits::allocate(allocator, max_elements_);
#endif
}

~runtime_sized_ringbuffer(void)
{
T out;
while (pop(&out, 1)) {}

#ifdef BOOST_NO_CXX11_ALLOCATOR
Alloc::deallocate(array_, max_elements_);
#else
Alloc& allocator = *this;
allocator_traits::deallocate(allocator, array_, max_elements_);
#endif
}

bool push(T const & t)
{
return ringbuffer_base<T>::push(t, &*array_, max_elements_);
}

template <typename Functor>
bool consume_one(Functor & f)
{
return ringbuffer_base<T>::consume_one(f, &*array_, max_elements_);
}

template <typename Functor>
bool consume_one(Functor const & f)
{
return ringbuffer_base<T>::consume_one(f, &*array_, max_elements_);
}

template <typename Functor>
size_type consume_all(Functor & f)
{
return ringbuffer_base<T>::consume_all(f, &*array_, max_elements_);
}

template <typename Functor>
size_type consume_all(Functor const & f)
{
return ringbuffer_base<T>::consume_all(f, &*array_, max_elements_);
}

size_type push(T const * t, size_type size)
{
return ringbuffer_base<T>::push(t, size, &*array_, max_elements_);
}

template <size_type size>
size_type push(T const (&t)[size])
{
return push(t, size);
}

template <typename ConstIterator>
ConstIterator push(ConstIterator begin, ConstIterator end)
{
return ringbuffer_base<T>::push(begin, end, &*array_, max_elements_);
}

size_type pop(T * ret, size_type size)
{
return ringbuffer_base<T>::pop(ret, size, &*array_, max_elements_);
}

template <typename OutputIterator>
size_type pop_to_output_iterator(OutputIterator it)
{
return ringbuffer_base<T>::pop_to_output_iterator(it, &*array_, max_elements_);
}

const T& front(void) const
{
return ringbuffer_base<T>::front(&*array_);
}

T& front(void)
{
return ringbuffer_base<T>::front(&*array_);
}
};

#ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <typename T, typename A0, typename A1>
#else
template <typename T, typename ...Options>
#endif
struct make_ringbuffer
{
#ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATES
typedef typename ringbuffer_signature::bind<A0, A1>::type bound_args;
#else
typedef typename ringbuffer_signature::bind<Options...>::type bound_args;
#endif

typedef extract_capacity<bound_args> extract_capacity_t;

static const bool runtime_sized = !extract_capacity_t::has_capacity;
static const size_t capacity    =  extract_capacity_t::capacity;

typedef extract_allocator<bound_args, T> extract_allocator_t;
typedef typename extract_allocator_t::type allocator;

BOOST_STATIC_ASSERT((mpl::if_<mpl::bool_<!runtime_sized>,
mpl::bool_<!extract_allocator_t::has_allocator>,
mpl::true_
>::type::value));

typedef typename mpl::if_c<runtime_sized,
runtime_sized_ringbuffer<T, allocator>,
compile_time_sized_ringbuffer<T, capacity>
>::type ringbuffer_type;
};


} 



#ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template <typename T, class A0, class A1>
#else
template <typename T, typename ...Options>
#endif
class spsc_queue:
#ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATES
public detail::make_ringbuffer<T, A0, A1>::ringbuffer_type
#else
public detail::make_ringbuffer<T, Options...>::ringbuffer_type
#endif
{
private:

#ifndef BOOST_DOXYGEN_INVOKED

#ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATES
typedef typename detail::make_ringbuffer<T, A0, A1>::ringbuffer_type base_type;
static const bool runtime_sized = detail::make_ringbuffer<T, A0, A1>::runtime_sized;
typedef typename detail::make_ringbuffer<T, A0, A1>::allocator allocator_arg;
#else
typedef typename detail::make_ringbuffer<T, Options...>::ringbuffer_type base_type;
static const bool runtime_sized = detail::make_ringbuffer<T, Options...>::runtime_sized;
typedef typename detail::make_ringbuffer<T, Options...>::allocator allocator_arg;
#endif


struct implementation_defined
{
typedef allocator_arg allocator;
typedef std::size_t size_type;
};
#endif

public:
typedef T value_type;
typedef typename implementation_defined::allocator allocator;
typedef typename implementation_defined::size_type size_type;


spsc_queue(void)
{
BOOST_ASSERT(!runtime_sized);
}


template <typename U>
explicit spsc_queue(typename boost::allocator_rebind<allocator, U>::type const &)
{
BOOST_STATIC_ASSERT(!runtime_sized);
}


explicit spsc_queue(allocator const &)
{
BOOST_ASSERT(!runtime_sized);
}


explicit spsc_queue(size_type element_count):
base_type(element_count)
{
BOOST_ASSERT(runtime_sized);
}


template <typename U>
spsc_queue(size_type element_count, typename boost::allocator_rebind<allocator, U>::type const & alloc):
base_type(alloc, element_count)
{
BOOST_STATIC_ASSERT(runtime_sized);
}


spsc_queue(size_type element_count, allocator_arg const & alloc):
base_type(alloc, element_count)
{
BOOST_ASSERT(runtime_sized);
}


bool push(T const & t)
{
return base_type::push(t);
}


bool pop ()
{
detail::consume_noop consume_functor;
return consume_one( consume_functor );
}


template <typename U>
typename boost::enable_if<typename is_convertible<T, U>::type, bool>::type
pop (U & ret)
{
detail::consume_via_copy<U> consume_functor(ret);
return consume_one( consume_functor );
}


size_type push(T const * t, size_type size)
{
return base_type::push(t, size);
}


template <size_type size>
size_type push(T const (&t)[size])
{
return push(t, size);
}


template <typename ConstIterator>
ConstIterator push(ConstIterator begin, ConstIterator end)
{
return base_type::push(begin, end);
}


size_type pop(T * ret, size_type size)
{
return base_type::pop(ret, size);
}


template <size_type size>
size_type pop(T (&ret)[size])
{
return pop(ret, size);
}


template <typename OutputIterator>
typename boost::disable_if<typename is_convertible<T, OutputIterator>::type, size_type>::type
pop(OutputIterator it)
{
return base_type::pop_to_output_iterator(it);
}


template <typename Functor>
bool consume_one(Functor & f)
{
return base_type::consume_one(f);
}

template <typename Functor>
bool consume_one(Functor const & f)
{
return base_type::consume_one(f);
}


template <typename Functor>
size_type consume_all(Functor & f)
{
return base_type::consume_all(f);
}

template <typename Functor>
size_type consume_all(Functor const & f)
{
return base_type::consume_all(f);
}


size_type read_available() const
{
return base_type::read_available(base_type::max_number_of_elements());
}


size_type write_available() const
{
return base_type::write_available(base_type::max_number_of_elements());
}


const T& front() const
{
BOOST_ASSERT(read_available() > 0);
return base_type::front();
}

T& front()
{
BOOST_ASSERT(read_available() > 0);
return base_type::front();
}


void reset(void)
{
if ( !boost::has_trivial_destructor<T>::value ) {

T dummy_element;
while (pop(dummy_element))
{}
} else {
base_type::write_index_.store(0, memory_order_relaxed);
base_type::read_index_.store(0, memory_order_release);
}
}
};

} 
} 


#endif 
