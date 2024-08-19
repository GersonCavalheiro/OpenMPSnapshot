#ifndef BOOST_SERIALIZATION_VOID_CAST_HPP
#define BOOST_SERIALIZATION_VOID_CAST_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstddef> 
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>

#include <boost/serialization/smart_cast.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/force_include.hpp>
#include <boost/serialization/type_info_implementation.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>
#include <boost/type_traits/aligned_storage.hpp>
#include <boost/serialization/void_cast_fwd.hpp>

#include <boost/serialization/config.hpp>
#include <boost/config/abi_prefix.hpp> 

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4251 4231 4660 4275)
#endif

namespace boost {
namespace serialization {

class extended_type_info;


BOOST_SERIALIZATION_DECL void const *
void_upcast(
extended_type_info const & derived,
extended_type_info const & base,
void const * const t
);

inline void *
void_upcast(
extended_type_info const & derived,
extended_type_info const & base,
void * const t
){
return const_cast<void*>(void_upcast(
derived,
base,
const_cast<void const *>(t)
));
}

BOOST_SERIALIZATION_DECL void const *
void_downcast(
extended_type_info const & derived,
extended_type_info const & base,
void const * const t
);

inline void *
void_downcast(
extended_type_info const & derived,
extended_type_info const & base,
void * const t
){
return const_cast<void*>(void_downcast(
derived,
base,
const_cast<void const *>(t)
));
}

namespace void_cast_detail {

class BOOST_SYMBOL_VISIBLE void_caster :
private boost::noncopyable
{
friend
BOOST_SERIALIZATION_DECL void const *
boost::serialization::void_upcast(
extended_type_info const & derived,
extended_type_info const & base,
void const * const
);
friend
BOOST_SERIALIZATION_DECL void const *
boost::serialization::void_downcast(
extended_type_info const & derived,
extended_type_info const & base,
void const * const
);
protected:
BOOST_SERIALIZATION_DECL void recursive_register(bool includes_virtual_base = false) const;
BOOST_SERIALIZATION_DECL void recursive_unregister() const;
virtual bool has_virtual_base() const = 0;
public:
const extended_type_info * m_derived;
const extended_type_info * m_base;
std::ptrdiff_t m_difference;
void_caster const * const m_parent;

bool operator<(const void_caster & rhs) const;

const void_caster & operator*(){
return *this;
}
virtual void const * upcast(void const * const t) const = 0;
virtual void const * downcast(void const * const t) const = 0;
void_caster(
extended_type_info const * derived,
extended_type_info const * base,
std::ptrdiff_t difference = 0,
void_caster const * const parent = 0
) :
m_derived(derived),
m_base(base),
m_difference(difference),
m_parent(parent)
{}
virtual ~void_caster(){}
};

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4251 4231 4660 4275 4511 4512)
#endif

template <class Derived, class Base>
class BOOST_SYMBOL_VISIBLE void_caster_primitive :
public void_caster
{
void const * downcast(void const * const t) const BOOST_OVERRIDE {
const Derived * d =
boost::serialization::smart_cast<const Derived *, const Base *>(
static_cast<const Base *>(t)
);
return d;
}
void const * upcast(void const * const t) const BOOST_OVERRIDE {
const Base * b =
boost::serialization::smart_cast<const Base *, const Derived *>(
static_cast<const Derived *>(t)
);
return b;
}
bool has_virtual_base() const BOOST_OVERRIDE {
return false;
}
public:
void_caster_primitive();
~void_caster_primitive() BOOST_OVERRIDE;

private:
static std::ptrdiff_t base_offset() {
typename boost::aligned_storage<sizeof(Derived)>::type data;
return reinterpret_cast<char*>(&data)
- reinterpret_cast<char*>(
static_cast<Base*>(
reinterpret_cast<Derived*>(&data)));
}
};

template <class Derived, class Base>
void_caster_primitive<Derived, Base>::void_caster_primitive() :
void_caster(
& type_info_implementation<Derived>::type::get_const_instance(),
& type_info_implementation<Base>::type::get_const_instance(),
base_offset()
)
{
recursive_register();
}

template <class Derived, class Base>
void_caster_primitive<Derived, Base>::~void_caster_primitive(){
recursive_unregister();
}

template <class Derived, class Base>
class BOOST_SYMBOL_VISIBLE void_caster_virtual_base :
public void_caster
{
bool has_virtual_base() const BOOST_OVERRIDE {
return true;
}
public:
void const * downcast(void const * const t) const BOOST_OVERRIDE {
const Derived * d =
dynamic_cast<const Derived *>(
static_cast<const Base *>(t)
);
return d;
}
void const * upcast(void const * const t) const BOOST_OVERRIDE {
const Base * b =
dynamic_cast<const Base *>(
static_cast<const Derived *>(t)
);
return b;
}
void_caster_virtual_base();
~void_caster_virtual_base() BOOST_OVERRIDE;
};

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template <class Derived, class Base>
void_caster_virtual_base<Derived,Base>::void_caster_virtual_base() :
void_caster(
& (type_info_implementation<Derived>::type::get_const_instance()),
& (type_info_implementation<Base>::type::get_const_instance())
)
{
recursive_register(true);
}

template <class Derived, class Base>
void_caster_virtual_base<Derived,Base>::~void_caster_virtual_base(){
recursive_unregister();
}

template <class Derived, class Base>
struct BOOST_SYMBOL_VISIBLE void_caster_base :
public void_caster
{
typedef
typename mpl::eval_if<boost::is_virtual_base_of<Base,Derived>,
mpl::identity<
void_cast_detail::void_caster_virtual_base<Derived, Base>
>
,
mpl::identity<
void_cast_detail::void_caster_primitive<Derived, Base>
>
>::type type;
};

} 

template<class Derived, class Base>
BOOST_DLLEXPORT
inline const void_cast_detail::void_caster & void_cast_register(
Derived const * ,
Base const * 
){
typedef
typename mpl::eval_if<boost::is_virtual_base_of<Base,Derived>,
mpl::identity<
void_cast_detail::void_caster_virtual_base<Derived, Base>
>
,
mpl::identity<
void_cast_detail::void_caster_primitive<Derived, Base>
>
>::type typex;
return singleton<typex>::get_const_instance();
}

template<class Derived, class Base>
class BOOST_SYMBOL_VISIBLE void_caster :
public void_cast_detail::void_caster_base<Derived, Base>::type
{
};

} 
} 

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

#include <boost/config/abi_suffix.hpp> 

#endif 
