
#ifndef BOOST_STACKTRACE_STACKTRACE_HPP
#define BOOST_STACKTRACE_STACKTRACE_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <boost/core/explicit_operator_bool.hpp>
#include <boost/core/no_exceptions_support.hpp>
#include <boost/container_hash/hash_fwd.hpp>

#include <iosfwd>
#include <string>
#include <vector>

#ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#endif

#include <boost/stacktrace/stacktrace_fwd.hpp>
#include <boost/stacktrace/safe_dump_to.hpp>
#include <boost/stacktrace/detail/frame_decl.hpp>

#ifdef BOOST_INTEL
#   pragma warning(push)
#   pragma warning(disable:2196) 
#endif

namespace boost { namespace stacktrace {

template <class Allocator>
class basic_stacktrace {
std::vector<boost::stacktrace::frame, Allocator> impl_;
typedef boost::stacktrace::detail::native_frame_ptr_t native_frame_ptr_t;

void fill(native_frame_ptr_t* begin, std::size_t size) {
if (!size) {
return;
}

impl_.reserve(static_cast<std::size_t>(size));
for (std::size_t i = 0; i < size; ++i) {
if (!begin[i]) {
return;
}
impl_.push_back(
frame(begin[i])
);
}
}

static std::size_t frames_count_from_buffer_size(std::size_t buffer_size) BOOST_NOEXCEPT {
const std::size_t ret = (buffer_size > sizeof(native_frame_ptr_t) ? buffer_size / sizeof(native_frame_ptr_t) : 0);
return (ret > 1024 ? 1024 : ret); 
}

BOOST_NOINLINE void init(std::size_t frames_to_skip, std::size_t max_depth) {
BOOST_CONSTEXPR_OR_CONST std::size_t buffer_size = 128;
if (!max_depth) {
return;
}

BOOST_TRY {
{   
native_frame_ptr_t buffer[buffer_size];
const std::size_t frames_count = boost::stacktrace::detail::this_thread_frames::collect(buffer, buffer_size < max_depth ? buffer_size : max_depth, frames_to_skip + 1);
if (buffer_size > frames_count || frames_count == max_depth) {
fill(buffer, frames_count);
return;
}
}

#ifdef BOOST_NO_CXX11_ALLOCATOR
typedef typename Allocator::template rebind<native_frame_ptr_t>::other allocator_void_t;
#else
typedef typename std::allocator_traits<Allocator>::template rebind_alloc<native_frame_ptr_t> allocator_void_t;
#endif
std::vector<native_frame_ptr_t, allocator_void_t> buf(buffer_size * 2, 0, impl_.get_allocator());
do {
const std::size_t frames_count = boost::stacktrace::detail::this_thread_frames::collect(&buf[0], buf.size() < max_depth ? buf.size() : max_depth, frames_to_skip + 1);
if (buf.size() > frames_count || frames_count == max_depth) {
fill(&buf[0], frames_count);
return;
}

buf.resize(buf.size() * 2);
} while (buf.size() < buf.max_size()); 
} BOOST_CATCH (...) {
}
BOOST_CATCH_END
}

public:
typedef typename std::vector<boost::stacktrace::frame, Allocator>::value_type             value_type;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::allocator_type         allocator_type;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::const_pointer          pointer;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::const_pointer          const_pointer;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::const_reference        reference;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::const_reference        const_reference;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::size_type              size_type;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::difference_type        difference_type;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::const_iterator         iterator;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::const_iterator         const_iterator;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::const_reverse_iterator reverse_iterator;
typedef typename std::vector<boost::stacktrace::frame, Allocator>::const_reverse_iterator const_reverse_iterator;

BOOST_FORCEINLINE basic_stacktrace() BOOST_NOEXCEPT
: impl_()
{
init(0 , static_cast<std::size_t>(-1));
}

BOOST_FORCEINLINE explicit basic_stacktrace(const allocator_type& a) BOOST_NOEXCEPT
: impl_(a)
{
init(0 , static_cast<std::size_t>(-1));
}

BOOST_FORCEINLINE basic_stacktrace(std::size_t skip, std::size_t max_depth, const allocator_type& a = allocator_type()) BOOST_NOEXCEPT
: impl_(a)
{
init(skip , max_depth);
}

basic_stacktrace(const basic_stacktrace& st)
: impl_(st.impl_)
{}

basic_stacktrace& operator=(const basic_stacktrace& st) {
impl_ = st.impl_;
return *this;
}

#ifdef BOOST_STACKTRACE_DOXYGEN_INVOKED
~basic_stacktrace() BOOST_NOEXCEPT = default;
#endif

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
basic_stacktrace(basic_stacktrace&& st) BOOST_NOEXCEPT
: impl_(std::move(st.impl_))
{}

basic_stacktrace& operator=(basic_stacktrace&& st)
#ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
BOOST_NOEXCEPT_IF(( std::is_nothrow_move_assignable< std::vector<boost::stacktrace::frame, Allocator> >::value ))
#else
BOOST_NOEXCEPT
#endif
{
impl_ = std::move(st.impl_);
return *this;
}
#endif

size_type size() const BOOST_NOEXCEPT {
return impl_.size();
}

const_reference operator[](std::size_t frame_no) const BOOST_NOEXCEPT {
return impl_[frame_no];
}

const_iterator begin() const BOOST_NOEXCEPT { return impl_.begin(); }
const_iterator cbegin() const BOOST_NOEXCEPT { return impl_.begin(); }
const_iterator end() const BOOST_NOEXCEPT { return impl_.end(); }
const_iterator cend() const BOOST_NOEXCEPT { return impl_.end(); }

const_reverse_iterator rbegin() const BOOST_NOEXCEPT { return impl_.rbegin(); }
const_reverse_iterator crbegin() const BOOST_NOEXCEPT { return impl_.rbegin(); }
const_reverse_iterator rend() const BOOST_NOEXCEPT { return impl_.rend(); }
const_reverse_iterator crend() const BOOST_NOEXCEPT { return impl_.rend(); }


BOOST_EXPLICIT_OPERATOR_BOOL_NOEXCEPT()

bool empty() const BOOST_NOEXCEPT { return !size(); }

bool operator!() const BOOST_NOEXCEPT { return !size(); }

const std::vector<boost::stacktrace::frame, Allocator>& as_vector() const BOOST_NOEXCEPT {
return impl_;
}

template <class Char, class Trait>
static basic_stacktrace from_dump(std::basic_istream<Char, Trait>& in, const allocator_type& a = allocator_type()) {
typedef typename std::basic_istream<Char, Trait>::pos_type pos_type;
basic_stacktrace ret(0, 0, a);

const pos_type pos = in.tellg();
in.seekg(0, in.end);
const std::size_t frames_count = frames_count_from_buffer_size(static_cast<std::size_t>(in.tellg()));
in.seekg(pos);

if (!frames_count) {
return ret;
}

native_frame_ptr_t ptr = 0;
ret.impl_.reserve(frames_count);
while (in.read(reinterpret_cast<Char*>(&ptr), sizeof(ptr))) {
if (!ptr) {
break;
}

ret.impl_.push_back(frame(ptr));
}

return ret;
}

static basic_stacktrace from_dump(const void* begin, std::size_t buffer_size_in_bytes, const allocator_type& a = allocator_type()) {
basic_stacktrace ret(0, 0, a);
const native_frame_ptr_t* first = static_cast<const native_frame_ptr_t*>(begin);
const std::size_t frames_count = frames_count_from_buffer_size(buffer_size_in_bytes);
if (!frames_count) {
return ret;
}

const native_frame_ptr_t* const last = first + frames_count;
ret.impl_.reserve(frames_count);
for (; first != last; ++first) {
if (!*first) {
break;
}

ret.impl_.push_back(frame(*first));
}

return ret;
}
};

template <class Allocator1, class Allocator2>
bool operator< (const basic_stacktrace<Allocator1>& lhs, const basic_stacktrace<Allocator2>& rhs) BOOST_NOEXCEPT {
return lhs.size() < rhs.size() || (lhs.size() == rhs.size() && lhs.as_vector() < rhs.as_vector());
}

template <class Allocator1, class Allocator2>
bool operator==(const basic_stacktrace<Allocator1>& lhs, const basic_stacktrace<Allocator2>& rhs) BOOST_NOEXCEPT {
return lhs.as_vector() == rhs.as_vector();
}


template <class Allocator1, class Allocator2>
bool operator> (const basic_stacktrace<Allocator1>& lhs, const basic_stacktrace<Allocator2>& rhs) BOOST_NOEXCEPT {
return rhs < lhs;
}

template <class Allocator1, class Allocator2>
bool operator<=(const basic_stacktrace<Allocator1>& lhs, const basic_stacktrace<Allocator2>& rhs) BOOST_NOEXCEPT {
return !(lhs > rhs);
}

template <class Allocator1, class Allocator2>
bool operator>=(const basic_stacktrace<Allocator1>& lhs, const basic_stacktrace<Allocator2>& rhs) BOOST_NOEXCEPT {
return !(lhs < rhs);
}

template <class Allocator1, class Allocator2>
bool operator!=(const basic_stacktrace<Allocator1>& lhs, const basic_stacktrace<Allocator2>& rhs) BOOST_NOEXCEPT {
return !(lhs == rhs);
}

template <class Allocator>
std::size_t hash_value(const basic_stacktrace<Allocator>& st) BOOST_NOEXCEPT {
return boost::hash_range(st.as_vector().begin(), st.as_vector().end());
}

template <class Allocator>
std::string to_string(const basic_stacktrace<Allocator>& bt) {
if (!bt) {
return std::string();
}

return boost::stacktrace::detail::to_string(&bt.as_vector()[0], bt.size());
}

template <class CharT, class TraitsT, class Allocator>
std::basic_ostream<CharT, TraitsT>& operator<<(std::basic_ostream<CharT, TraitsT>& os, const basic_stacktrace<Allocator>& bt) {
return os << boost::stacktrace::to_string(bt);
}

typedef basic_stacktrace<> stacktrace;

}} 

#ifdef BOOST_INTEL
#   pragma warning(pop)
#endif

#endif 
