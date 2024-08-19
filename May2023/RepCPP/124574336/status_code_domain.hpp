

#ifndef BOOST_OUTCOME_SYSTEM_ERROR2_STATUS_CODE_DOMAIN_HPP
#define BOOST_OUTCOME_SYSTEM_ERROR2_STATUS_CODE_DOMAIN_HPP

#include "config.hpp"

#include <cstring>  

BOOST_OUTCOME_SYSTEM_ERROR2_NAMESPACE_BEGIN


template <class DomainType> class status_code;
class _generic_code_domain;
using generic_code = status_code<_generic_code_domain>;

namespace detail
{
template <class StatusCode> class indirecting_domain;
template <class T> struct status_code_sizer
{
void *a;
T b;
};
template <class To, class From> struct type_erasure_is_safe
{
static constexpr bool value = traits::is_move_bitcopying<From>::value  
&& (sizeof(status_code_sizer<From>) <= sizeof(status_code_sizer<To>));
};

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdiv-by-zero"
#endif
#if defined(__cpp_exceptions) || (defined(_MSC_VER) && !defined(__clang__))
#define BOOST_OUTCOME_SYSTEM_ERROR2_FAIL_CONSTEXPR(msg) throw msg
#else
#define BOOST_OUTCOME_SYSTEM_ERROR2_FAIL_CONSTEXPR(msg) ((void) msg, 1 / 0)
#endif
constexpr inline unsigned long long parse_hex_byte(char c) { return ('0' <= c && c <= '9') ? (c - '0') : ('a' <= c && c <= 'f') ? (10 + c - 'a') : ('A' <= c && c <= 'F') ? (10 + c - 'A') : BOOST_OUTCOME_SYSTEM_ERROR2_FAIL_CONSTEXPR("Invalid character in UUID"); }
constexpr inline unsigned long long parse_uuid2(const char *s)
{
return ((parse_hex_byte(s[0]) << 0) | (parse_hex_byte(s[1]) << 4) | (parse_hex_byte(s[2]) << 8) | (parse_hex_byte(s[3]) << 12) | (parse_hex_byte(s[4]) << 16) | (parse_hex_byte(s[5]) << 20) | (parse_hex_byte(s[6]) << 24) | (parse_hex_byte(s[7]) << 28) | (parse_hex_byte(s[9]) << 32) | (parse_hex_byte(s[10]) << 36) |
(parse_hex_byte(s[11]) << 40) | (parse_hex_byte(s[12]) << 44) | (parse_hex_byte(s[14]) << 48) | (parse_hex_byte(s[15]) << 52) | (parse_hex_byte(s[16]) << 56) | (parse_hex_byte(s[17]) << 60))  
^                                                                                                                                                                                                
((parse_hex_byte(s[19]) << 0) | (parse_hex_byte(s[20]) << 4) | (parse_hex_byte(s[21]) << 8) | (parse_hex_byte(s[22]) << 12) | (parse_hex_byte(s[24]) << 16) | (parse_hex_byte(s[25]) << 20) | (parse_hex_byte(s[26]) << 24) | (parse_hex_byte(s[27]) << 28) | (parse_hex_byte(s[28]) << 32) |
(parse_hex_byte(s[29]) << 36) | (parse_hex_byte(s[30]) << 40) | (parse_hex_byte(s[31]) << 44) | (parse_hex_byte(s[32]) << 48) | (parse_hex_byte(s[33]) << 52) | (parse_hex_byte(s[34]) << 56) | (parse_hex_byte(s[35]) << 60));
}
template <size_t N> constexpr inline unsigned long long parse_uuid_from_array(const char (&uuid)[N]) { return (N == 37) ? parse_uuid2(uuid) : ((N == 39) ? parse_uuid2(uuid + 1) : BOOST_OUTCOME_SYSTEM_ERROR2_FAIL_CONSTEXPR("UUID does not have correct length")); }
template <size_t N> constexpr inline unsigned long long parse_uuid_from_pointer(const char *uuid) { return (N == 36) ? parse_uuid2(uuid) : ((N == 38) ? parse_uuid2(uuid + 1) : BOOST_OUTCOME_SYSTEM_ERROR2_FAIL_CONSTEXPR("UUID does not have correct length")); }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
static constexpr unsigned long long test_uuid_parse = parse_uuid_from_array("430f1201-94fc-06c7-430f-120194fc06c7");
}  


class status_code_domain
{
template <class DomainType> friend class status_code;
template <class StatusCode> friend class indirecting_domain;

public:
using unique_id_type = unsigned long long;

class string_ref
{
public:
using value_type = const char;
using size_type = size_t;
using pointer = const char *;
using const_pointer = const char *;
using iterator = const char *;
using const_iterator = const char *;

protected:
enum class _thunk_op
{
copy,
move,
destruct
};
using _thunk_spec = void (*)(string_ref *dest, const string_ref *src, _thunk_op op);
#ifndef NDEBUG
private:
static void _checking_string_thunk(string_ref *dest, const string_ref *src, _thunk_op ) noexcept
{
(void) dest;
(void) src;
assert(dest->_thunk == _checking_string_thunk);                   
assert(src == nullptr || src->_thunk == _checking_string_thunk);  
}

protected:
#endif
pointer _begin{}, _end{};
void *_state[3]{};  
const _thunk_spec _thunk{nullptr};

constexpr explicit string_ref(_thunk_spec thunk) noexcept
: _thunk(thunk)
{
}

public:
BOOST_OUTCOME_SYSTEM_ERROR2_CONSTEXPR14 explicit string_ref(const char *str, size_type len = static_cast<size_type>(-1), void *state0 = nullptr, void *state1 = nullptr, void *state2 = nullptr,
#ifndef NDEBUG
_thunk_spec thunk = _checking_string_thunk
#else
_thunk_spec thunk = nullptr
#endif
) noexcept
: _begin(str)
, _end((len == static_cast<size_type>(-1)) ? (str + detail::cstrlen(str)) : (str + len))
,  
_state{state0, state1, state2}
, _thunk(thunk)
{
}
string_ref(const string_ref &o)
: _begin(o._begin)
, _end(o._end)
, _state{o._state[0], o._state[1], o._state[2]}
, _thunk(o._thunk)
{
if(_thunk != nullptr)
{
_thunk(this, &o, _thunk_op::copy);
}
}
string_ref(string_ref &&o) noexcept
: _begin(o._begin)
, _end(o._end)
, _state{o._state[0], o._state[1], o._state[2]}
, _thunk(o._thunk)
{
if(_thunk != nullptr)
{
_thunk(this, &o, _thunk_op::move);
}
}
string_ref &operator=(const string_ref &o)
{
if(this != &o)
{
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
string_ref temp(static_cast<string_ref &&>(*this));
this->~string_ref();
try
{
new(this) string_ref(o);  
}
catch(...)
{
new(this) string_ref(static_cast<string_ref &&>(temp));
throw;
}
#else
this->~string_ref();
new(this) string_ref(o);
#endif
}
return *this;
}
string_ref &operator=(string_ref &&o) noexcept
{
if(this != &o)
{
this->~string_ref();
new(this) string_ref(static_cast<string_ref &&>(o));
}
return *this;
}
~string_ref()
{
if(_thunk != nullptr)
{
_thunk(this, nullptr, _thunk_op::destruct);
}
_begin = _end = nullptr;
}

BOOST_OUTCOME_SYSTEM_ERROR2_NODISCARD bool empty() const noexcept { return _begin == _end; }
size_type size() const noexcept { return _end - _begin; }
const_pointer c_str() const noexcept { return _begin; }
const_pointer data() const noexcept { return _begin; }
iterator begin() noexcept { return _begin; }
const_iterator begin() const noexcept { return _begin; }
const_iterator cbegin() const noexcept { return _begin; }
iterator end() noexcept { return _end; }
const_iterator end() const noexcept { return _end; }
const_iterator cend() const noexcept { return _end; }
};


class atomic_refcounted_string_ref : public string_ref
{
struct _allocated_msg
{
mutable std::atomic<unsigned> count{1};
};
_allocated_msg *&_msg() noexcept { return reinterpret_cast<_allocated_msg *&>(this->_state[0]); }                  
const _allocated_msg *_msg() const noexcept { return reinterpret_cast<const _allocated_msg *>(this->_state[0]); }  

static void _refcounted_string_thunk(string_ref *_dest, const string_ref *_src, _thunk_op op) noexcept
{
auto dest = static_cast<atomic_refcounted_string_ref *>(_dest);      
auto src = static_cast<const atomic_refcounted_string_ref *>(_src);  
(void) src;
assert(dest->_thunk == _refcounted_string_thunk);                   
assert(src == nullptr || src->_thunk == _refcounted_string_thunk);  
switch(op)
{
case _thunk_op::copy:
{
if(dest->_msg() != nullptr)
{
auto count = dest->_msg()->count.fetch_add(1, std::memory_order_relaxed);
(void) count;
assert(count != 0);  
}
return;
}
case _thunk_op::move:
{
assert(src);                                                  
auto msrc = const_cast<atomic_refcounted_string_ref *>(src);  
msrc->_begin = msrc->_end = nullptr;
msrc->_state[0] = msrc->_state[1] = msrc->_state[2] = nullptr;
return;
}
case _thunk_op::destruct:
{
if(dest->_msg() != nullptr)
{
auto count = dest->_msg()->count.fetch_sub(1, std::memory_order_release);
if(count == 1)
{
std::atomic_thread_fence(std::memory_order_acquire);
free((void *) dest->_begin);  
delete dest->_msg();          
}
}
}
}
}

public:
explicit atomic_refcounted_string_ref(const char *str, size_type len = static_cast<size_type>(-1), void *state1 = nullptr, void *state2 = nullptr) noexcept
: string_ref(str, len, new(std::nothrow) _allocated_msg, state1, state2, _refcounted_string_thunk)
{
if(_msg() == nullptr)
{
free((void *) this->_begin);  
_msg() = nullptr;             
this->_begin = "failed to get message from system";
this->_end = strchr(this->_begin, 0);
return;
}
}
};

private:
unique_id_type _id;

protected:

constexpr explicit status_code_domain(unique_id_type id) noexcept
: _id(id)
{
}

template <size_t N>
constexpr explicit status_code_domain(const char (&uuid)[N]) noexcept
: _id(detail::parse_uuid_from_array<N>(uuid))
{
}
template <size_t N> struct _uuid_size
{
};
template <size_t N>
constexpr explicit status_code_domain(const char *uuid, _uuid_size<N> ) noexcept
: _id(detail::parse_uuid_from_pointer<N>(uuid))
{
}
status_code_domain(const status_code_domain &) = default;
status_code_domain(status_code_domain &&) = default;
status_code_domain &operator=(const status_code_domain &) = default;
status_code_domain &operator=(status_code_domain &&) = default;
~status_code_domain() = default;

public:
constexpr bool operator==(const status_code_domain &o) const noexcept { return _id == o._id; }
constexpr bool operator!=(const status_code_domain &o) const noexcept { return _id != o._id; }
constexpr bool operator<(const status_code_domain &o) const noexcept { return _id < o._id; }

constexpr unique_id_type id() const noexcept { return _id; }
virtual string_ref name() const noexcept = 0;

protected:
virtual bool _do_failure(const status_code<void> &code) const noexcept = 0;
virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept = 0;
virtual generic_code _generic_code(const status_code<void> &code) const noexcept = 0;
virtual string_ref _do_message(const status_code<void> &code) const noexcept = 0;
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || defined(BOOST_OUTCOME_STANDARDESE_IS_IN_THE_HOUSE)
BOOST_OUTCOME_SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const = 0;
#else
BOOST_OUTCOME_SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> & ) const { abort(); }
#endif
virtual void _do_erased_copy(status_code<void> &dst, const status_code<void> &src, size_t bytes) const { memcpy(&dst, &src, bytes); }  
virtual void _do_erased_destroy(status_code<void> &code, size_t bytes) const noexcept  
{
(void) code;
(void) bytes;
}
};

BOOST_OUTCOME_SYSTEM_ERROR2_NAMESPACE_END

#endif
