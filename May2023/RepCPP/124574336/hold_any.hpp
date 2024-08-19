
#if !defined(BOOST_SPIRIT_HOLD_ANY_MAY_02_2007_0857AM)
#define BOOST_SPIRIT_HOLD_ANY_MAY_02_2007_0857AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/throw_exception.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/assert.hpp>
#include <boost/core/typeinfo.hpp>

#include <algorithm>
#include <iosfwd>
#include <stdexcept>
#include <typeinfo>

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(push)
# pragma warning(disable: 4100)   
# pragma warning(disable: 4127)   
#endif

namespace boost { namespace spirit
{
struct bad_any_cast
: std::bad_cast
{
bad_any_cast(boost::core::typeinfo const& src, boost::core::typeinfo const& dest)
: from(src.name()), to(dest.name())
{}

const char* what() const BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE
{ 
return "bad any cast";
}

const char* from;
const char* to;
};

namespace detail
{
template <typename Char>
struct fxn_ptr_table
{
boost::core::typeinfo const& (*get_type)();
void (*static_delete)(void**);
void (*destruct)(void**);
void (*clone)(void* const*, void**);
void (*move)(void* const*, void**);
std::basic_istream<Char>& (*stream_in)(std::basic_istream<Char>&, void**);
std::basic_ostream<Char>& (*stream_out)(std::basic_ostream<Char>&, void* const*);
};

template <typename Small>
struct fxns;

template <>
struct fxns<mpl::true_>
{
template<typename T, typename Char>
struct type
{
static boost::core::typeinfo const& get_type()
{
return BOOST_CORE_TYPEID(T);
}
static void static_delete(void** x)
{
reinterpret_cast<T*>(x)->~T();
}
static void destruct(void** x)
{
reinterpret_cast<T*>(x)->~T();
}
static void clone(void* const* src, void** dest)
{
new (dest) T(*reinterpret_cast<T const*>(src));
}
static void move(void* const* src, void** dest)
{
*reinterpret_cast<T*>(dest) =
*reinterpret_cast<T const*>(src);
}
static std::basic_istream<Char>&
stream_in (std::basic_istream<Char>& i, void** obj)
{
i >> *reinterpret_cast<T*>(obj);
return i;
}
static std::basic_ostream<Char>&
stream_out(std::basic_ostream<Char>& o, void* const* obj)
{
o << *reinterpret_cast<T const*>(obj);
return o;
}
};
};

template <>
struct fxns<mpl::false_>
{
template<typename T, typename Char>
struct type
{
static boost::core::typeinfo const& get_type()
{
return BOOST_CORE_TYPEID(T);
}
static void static_delete(void** x)
{
delete (*reinterpret_cast<T**>(x));
}
static void destruct(void** x)
{
(*reinterpret_cast<T**>(x))->~T();
}
static void clone(void* const* src, void** dest)
{
*dest = new T(**reinterpret_cast<T* const*>(src));
}
static void move(void* const* src, void** dest)
{
**reinterpret_cast<T**>(dest) =
**reinterpret_cast<T* const*>(src);
}
static std::basic_istream<Char>&
stream_in(std::basic_istream<Char>& i, void** obj)
{
i >> **reinterpret_cast<T**>(obj);
return i;
}
static std::basic_ostream<Char>&
stream_out(std::basic_ostream<Char>& o, void* const* obj)
{
o << **reinterpret_cast<T* const*>(obj);
return o;
}
};
};

template <typename T>
struct get_table
{
typedef mpl::bool_<(sizeof(T) <= sizeof(void*))> is_small;

template <typename Char>
static fxn_ptr_table<Char>* get()
{
static fxn_ptr_table<Char> static_table =
{
fxns<is_small>::template type<T, Char>::get_type,
fxns<is_small>::template type<T, Char>::static_delete,
fxns<is_small>::template type<T, Char>::destruct,
fxns<is_small>::template type<T, Char>::clone,
fxns<is_small>::template type<T, Char>::move,
fxns<is_small>::template type<T, Char>::stream_in,
fxns<is_small>::template type<T, Char>::stream_out
};
return &static_table;
}
};

struct empty {};

template <typename Char>
inline std::basic_istream<Char>&
operator>> (std::basic_istream<Char>& i, empty&)
{
BOOST_ASSERT(false &&
"Tried to insert from a std istream into an empty "
"hold_any instance");
return i;
}

template <typename Char>
inline std::basic_ostream<Char>&
operator<< (std::basic_ostream<Char>& o, empty const&)
{
return o;
}
}

template <typename Char>
class basic_hold_any
{
public:
template <typename T>
explicit basic_hold_any(T const& x)
: table(spirit::detail::get_table<T>::template get<Char>()), object(0)
{
new_object(object, x,
typename spirit::detail::get_table<T>::is_small());
}

basic_hold_any()
: table(spirit::detail::get_table<spirit::detail::empty>::template get<Char>()),
object(0)
{
}

basic_hold_any(basic_hold_any const& x)
: table(spirit::detail::get_table<spirit::detail::empty>::template get<Char>()),
object(0)
{
assign(x);
}

~basic_hold_any()
{
table->static_delete(&object);
}

basic_hold_any& assign(basic_hold_any const& x)
{
if (&x != this) {
if (table == x.table) {
table->move(&x.object, &object);
}
else {
reset();
x.table->clone(&x.object, &object);
table = x.table;
}
}
return *this;
}

template <typename T>
basic_hold_any& assign(T const& x)
{
spirit::detail::fxn_ptr_table<Char>* x_table =
spirit::detail::get_table<T>::template get<Char>();
if (table == x_table) {
table->destruct(&object);    
if (spirit::detail::get_table<T>::is_small::value) {
new (&object) T(x);
}
else {
new (object) T(x);
}
}
else {
if (spirit::detail::get_table<T>::is_small::value) {
table->destruct(&object); 
new (&object) T(x);
}
else {
reset();                  
object = new T(x);
}
table = x_table;      
}
return *this;
}

template <typename T>
static void new_object(void*& object, T const& x, mpl::true_)
{
new (&object) T(x);
}

template <typename T>
static void new_object(void*& object, T const& x, mpl::false_)
{
object = new T(x);
}

#ifdef BOOST_HAS_RVALUE_REFS
template <typename T>
basic_hold_any& operator=(T&& x)
{
return assign(std::forward<T>(x));
}
#else
template <typename T>
basic_hold_any& operator=(T& x)
{
return assign(x);
}

template <typename T>
basic_hold_any& operator=(T const& x)
{
return assign(x);
}
#endif
basic_hold_any& operator=(basic_hold_any const& x)
{
return assign(x);
}

basic_hold_any& swap(basic_hold_any& x)
{
std::swap(table, x.table);
std::swap(object, x.object);
return *this;
}

boost::core::typeinfo const& type() const
{
return table->get_type();
}

template <typename T>
T const& cast() const
{
if (type() != BOOST_CORE_TYPEID(T))
throw bad_any_cast(type(), BOOST_CORE_TYPEID(T));

return spirit::detail::get_table<T>::is_small::value ?
*reinterpret_cast<T const*>(&object) :
*reinterpret_cast<T const*>(object);
}

#ifdef BOOST_SPIRIT_ANY_IMPLICIT_CASTING
template <typename T>
operator T const& () const { return cast<T>(); }
#endif 

bool empty() const
{
return table == spirit::detail::get_table<spirit::detail::empty>::template get<Char>();
}

void reset()
{
if (!empty())
{
table->static_delete(&object);
table = spirit::detail::get_table<spirit::detail::empty>::template get<Char>();
object = 0;
}
}

template <typename Char_>
friend inline std::basic_istream<Char_>&
operator>> (std::basic_istream<Char_>& i, basic_hold_any<Char_>& obj)
{
return obj.table->stream_in(i, &obj.object);
}

template <typename Char_>
friend inline std::basic_ostream<Char_>&
operator<< (std::basic_ostream<Char_>& o, basic_hold_any<Char_> const& obj)
{
return obj.table->stream_out(o, &obj.object);
}

#ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
private: 
template <typename T, typename Char_>
friend T* any_cast(basic_hold_any<Char_> *);
#else
public: 
#endif
spirit::detail::fxn_ptr_table<Char>* table;
void* object;
};

template <typename T, typename Char>
inline T* any_cast (basic_hold_any<Char>* operand)
{
if (operand && operand->type() == BOOST_CORE_TYPEID(T)) {
return spirit::detail::get_table<T>::is_small::value ?
reinterpret_cast<T*>(&operand->object) :
reinterpret_cast<T*>(operand->object);
}
return 0;
}

template <typename T, typename Char>
inline T const* any_cast(basic_hold_any<Char> const* operand)
{
return any_cast<T>(const_cast<basic_hold_any<Char>*>(operand));
}

template <typename T, typename Char>
T any_cast(basic_hold_any<Char>& operand)
{
typedef BOOST_DEDUCED_TYPENAME remove_reference<T>::type nonref;


nonref* result = any_cast<nonref>(&operand);
if(!result)
boost::throw_exception(bad_any_cast(operand.type(), BOOST_CORE_TYPEID(T)));
return *result;
}

template <typename T, typename Char>
T const& any_cast(basic_hold_any<Char> const& operand)
{
typedef BOOST_DEDUCED_TYPENAME remove_reference<T>::type nonref;


return any_cast<nonref const&>(const_cast<basic_hold_any<Char> &>(operand));
}

typedef basic_hold_any<char> hold_any;
typedef basic_hold_any<wchar_t> whold_any;

namespace traits
{
template <typename T>
struct is_hold_any : mpl::false_ {};

template <typename Char>
struct is_hold_any<basic_hold_any<Char> > : mpl::true_ {};
}

}}    

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
# pragma warning(pop)
#endif

#endif
