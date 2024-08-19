
#ifndef BOOST_TYPE_ERASURE_ANY_HPP_INCLUDED
#define BOOST_TYPE_ERASURE_ANY_HPP_INCLUDED

#include <algorithm>
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   include <utility>  
#endif
#include <boost/config.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/addressof.hpp>
#include <boost/utility/declval.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/reverse_fold.hpp>
#include <boost/type_traits/decay.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#include <boost/type_erasure/detail/access.hpp>
#include <boost/type_erasure/detail/any_base.hpp>
#include <boost/type_erasure/detail/normalize.hpp>
#include <boost/type_erasure/detail/storage.hpp>
#include <boost/type_erasure/detail/instantiate.hpp>
#include <boost/type_erasure/config.hpp>
#include <boost/type_erasure/binding.hpp>
#include <boost/type_erasure/static_binding.hpp>
#include <boost/type_erasure/concept_interface.hpp>
#include <boost/type_erasure/call.hpp>
#include <boost/type_erasure/relaxed.hpp>
#include <boost/type_erasure/param.hpp>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4355)
#pragma warning(disable:4521)
#pragma warning(disable:4522) 
#endif

namespace boost {
namespace type_erasure {

#ifndef BOOST_TYPE_ERASURE_DOXYGEN

template<class Sig>
struct constructible;

template<class T>
struct destructible;

template<class T, class U>
struct assignable;

#endif

namespace detail {

#if defined(BOOST_NO_CXX11_DECLTYPE) || defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)

template<class Concept, class Base, class ID>
struct choose_concept_interface
{
typedef ::boost::type_erasure::concept_interface<Concept, Base, ID> type;
};

#else

struct default_concept_interface
{
template<class Concept, class Base, class ID>
using apply = ::boost::type_erasure::concept_interface<Concept, Base, ID>;
};

default_concept_interface boost_type_erasure_find_interface(...);

template<class Concept, class Base, class ID>
struct choose_concept_interface
{
typedef decltype(boost_type_erasure_find_interface(::boost::declval<Concept>())) finder;
typedef typename finder::template apply<Concept, Base, ID> type;
};

#endif

#ifndef BOOST_TYPE_ERASURE_USE_MP11

template<class Derived, class Concept, class T>
struct compute_bases
{
typedef typename ::boost::mpl::reverse_fold<
typename ::boost::type_erasure::detail::collect_concepts<
Concept
>::type,
::boost::type_erasure::any_base<Derived>,
::boost::type_erasure::detail::choose_concept_interface<
::boost::mpl::_2,
::boost::mpl::_1,
T
>
>::type type;
};

#else

template<class ID>
struct compute_bases_f
{
template<class Concept, class Base>
using apply = typename ::boost::type_erasure::detail::choose_concept_interface<Concept, Base, ID>::type;
};

template<class Derived, class Concept, class T>
using compute_bases_t =
::boost::mp11::mp_reverse_fold<
typename ::boost::type_erasure::detail::collect_concepts_t<
Concept
>,
::boost::type_erasure::any_base<Derived>,
::boost::type_erasure::detail::compute_bases_f<T>::template apply
>;

template<class Derived, class Concept, class T>
using compute_bases = ::boost::mpl::identity< ::boost::type_erasure::detail::compute_bases_t<Derived, Concept, T> >;

#endif

template<class T>
T make(T*) { return T(); }


typedef char no;
struct yes { no dummy[2]; };

template<class Op>
yes check_overload(const Op*);
no check_overload(const void*);

struct fallback {};

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

template<class T>
fallback make_fallback(T&&, boost::mpl::false_)
{
return fallback();
}

template<class T>
T&& make_fallback(T&& arg, boost::mpl::true_)
{
return std::forward<T>(arg);
}

#else

template<class T>
fallback make_fallback(const T&, boost::mpl::false_)
{
return fallback();
}

template<class T>
const T& make_fallback(const T& arg, boost::mpl::true_)
{
return arg;
}

#endif

template<class T>
struct is_any : ::boost::mpl::false_ {};

template<class Concept, class T>
struct is_any<any<Concept, T> > : ::boost::mpl::true_ {};

#ifdef BOOST_TYPE_ERASURE_SFINAE_FRIENDLY_CONSTRUCTORS

template<class Any, class... U>
struct has_constructor :
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<Any&>().
_boost_type_erasure_deduce_constructor(::boost::declval<U>()...)
)
) == sizeof(::boost::type_erasure::detail::yes)
>
{};

template<class Any>
using has_copy_constructor =
::boost::type_erasure::is_subconcept<
::boost::type_erasure::constructible<
typename ::boost::type_erasure::placeholder_of<Any>::type(typename ::boost::type_erasure::placeholder_of<Any>::type const&)
>,
typename ::boost::type_erasure::concept_of<Any>::type
>;

template<class Any>
using has_move_constructor =
::boost::type_erasure::is_subconcept<
::boost::type_erasure::constructible<
typename ::boost::type_erasure::placeholder_of<Any>::type(typename ::boost::type_erasure::placeholder_of<Any>::type &&)
>,
typename ::boost::type_erasure::concept_of<Any>::type
>;

template<class Any>
using has_mutable_copy_constructor =
::boost::type_erasure::is_subconcept<
::boost::type_erasure::constructible<
typename ::boost::type_erasure::placeholder_of<Any>::type(typename ::boost::type_erasure::placeholder_of<Any>::type &)
>,
typename ::boost::type_erasure::concept_of<Any>::type
>;

struct empty {};

template<class T>
struct is_binding_arg : ::boost::mpl::false_ {};
template<class T>
struct is_binding_arg<binding<T> > : ::boost::mpl::true_ {};
template<class T>
struct is_binding_arg<binding<T>&&> : ::boost::mpl::true_ {};
template<class T>
struct is_binding_arg<binding<T>&> : ::boost::mpl::true_ {};
template<class T>
struct is_binding_arg<binding<T> const&> : ::boost::mpl::true_ {};

template<class T>
struct is_static_binding_arg : ::boost::mpl::false_ {};
template<class T>
struct is_static_binding_arg<static_binding<T> > : ::boost::mpl::true_ {};
template<class T>
struct is_static_binding_arg<static_binding<T>&&> : ::boost::mpl::true_ {};
template<class T>
struct is_static_binding_arg<static_binding<T>&> : ::boost::mpl::true_ {};
template<class T>
struct is_static_binding_arg<static_binding<T> const&> : ::boost::mpl::true_ {};

template<class T>
struct is_any_arg : ::boost::mpl::false_ {};
template<class Concept, class T>
struct is_any_arg<any<Concept, T> > : ::boost::mpl::true_ {};
template<class Concept, class T>
struct is_any_arg<any<Concept, T>&&> : ::boost::mpl::true_ {};
template<class Concept, class T>
struct is_any_arg<any<Concept, T>&> : ::boost::mpl::true_ {};
template<class Concept, class T>
struct is_any_arg<any<Concept, T> const&> : ::boost::mpl::true_ {};

template<class T>
struct safe_concept_of;
template<class Concept, class T>
struct safe_concept_of<any<Concept, T> > { typedef Concept type; };
template<class Concept, class T>
struct safe_concept_of<any<Concept, T>&&> { typedef Concept type; };
template<class Concept, class T>
struct safe_concept_of<any<Concept, T>&> { typedef Concept type; };
template<class Concept, class T>
struct safe_concept_of<any<Concept, T> const&> { typedef Concept type; };

template<class T>
struct safe_placeholder_of;
template<class Concept, class T>
struct safe_placeholder_of<any<Concept, T> > { typedef T type; };
template<class Concept, class T>
struct safe_placeholder_of<any<Concept, T>&&> { typedef T type; };
template<class Concept, class T>
struct safe_placeholder_of<any<Concept, T>&> { typedef T type; };
template<class Concept, class T>
struct safe_placeholder_of<any<Concept, T> const&> { typedef T type; };

template<class T>
using safe_placeholder_t = ::boost::remove_cv_t< ::boost::remove_reference_t<typename safe_placeholder_of<T>::type> >;

}

template<class Base, class Enable = void>
struct any_constructor_control : Base
{
using Base::Base;
};

template<class Base>
struct any_constructor_control<
Base,
typename boost::enable_if_c<
!::boost::type_erasure::detail::has_copy_constructor<Base>::value &&
::boost::type_erasure::detail::has_move_constructor<Base>::value &&
::boost::type_erasure::detail::has_mutable_copy_constructor<Base>::value
>::type
> : Base
{
using Base::Base;
any_constructor_control() = default;
any_constructor_control(any_constructor_control&) = default;
any_constructor_control(any_constructor_control&&) = default;
any_constructor_control& operator=(any_constructor_control const& other) { static_cast<Base&>(*this) = static_cast<Base const&>(other); return *this; }
any_constructor_control& operator=(any_constructor_control & other) { static_cast<Base&>(*this) = static_cast<Base&>(other); return *this; }
any_constructor_control& operator=(any_constructor_control &&) = default;
};

template<class Base>
struct any_constructor_control<
Base,
typename boost::enable_if_c<
!::boost::type_erasure::detail::has_copy_constructor<Base>::value &&
!::boost::type_erasure::detail::has_move_constructor<Base>::value &&
::boost::type_erasure::detail::has_mutable_copy_constructor<Base>::value
>::type
> : Base
{
using Base::Base;
any_constructor_control() = default;
any_constructor_control(any_constructor_control&) = default;
any_constructor_control(any_constructor_control&&) = delete;
any_constructor_control& operator=(any_constructor_control const& other) { static_cast<Base&>(*this) = static_cast<Base const&>(other); return *this; }
any_constructor_control& operator=(any_constructor_control & other) { static_cast<Base&>(*this) = static_cast<Base&>(other); return *this; }
any_constructor_control& operator=(any_constructor_control &&) = default;
};

template<class Base>
struct any_constructor_control<
Base,
typename boost::enable_if_c<
!::boost::type_erasure::detail::has_copy_constructor<Base>::value &&
::boost::type_erasure::detail::has_move_constructor<Base>::value &&
!::boost::type_erasure::detail::has_mutable_copy_constructor<Base>::value
>::type
> : Base
{
using Base::Base;
any_constructor_control() = default;
any_constructor_control(any_constructor_control const&) = delete;
any_constructor_control(any_constructor_control&&) = default;
any_constructor_control& operator=(any_constructor_control const& other) { static_cast<Base&>(*this) = static_cast<Base const&>(other); return *this; }
any_constructor_control& operator=(any_constructor_control & other) { static_cast<Base&>(*this) = static_cast<Base&>(other); return *this; }
any_constructor_control& operator=(any_constructor_control &&) = default;
};

template<class Base>
struct any_constructor_control<
Base,
typename boost::enable_if_c<
!::boost::type_erasure::detail::has_copy_constructor<Base>::value &&
!::boost::type_erasure::detail::has_move_constructor<Base>::value &&
!::boost::type_erasure::detail::has_mutable_copy_constructor<Base>::value
>::type
> : Base
{
using Base::Base;
any_constructor_control() = default;
any_constructor_control(any_constructor_control const&) = delete;
any_constructor_control(any_constructor_control&&) = delete;
any_constructor_control& operator=(any_constructor_control const& other) { static_cast<Base&>(*this) = static_cast<Base const&>(other); return *this; }
any_constructor_control& operator=(any_constructor_control & other) { static_cast<Base&>(*this) = static_cast<Base&>(other); return *this; }
any_constructor_control& operator=(any_constructor_control &&) = default;
};

template<class Concept, class T>
struct any_constructor_impl :
::boost::type_erasure::detail::compute_bases<
::boost::type_erasure::any<Concept, T>,
Concept,
T
>::type
{
typedef typename ::boost::type_erasure::detail::compute_bases<
::boost::type_erasure::any<Concept, T>,
Concept,
T
>::type _boost_type_erasure_base;
typedef ::boost::type_erasure::binding<Concept> _boost_type_erasure_table_type;
any_constructor_impl(const ::boost::type_erasure::detail::storage& data_arg, const _boost_type_erasure_table_type& table_arg)
: _boost_type_erasure_table(table_arg),
_boost_type_erasure_data(data_arg)
{}
any_constructor_impl(::boost::type_erasure::detail::storage&& data_arg, const _boost_type_erasure_table_type& table_arg)
: _boost_type_erasure_table(table_arg),
_boost_type_erasure_data(data_arg)
{}
any_constructor_impl()
{
BOOST_MPL_ASSERT((::boost::type_erasure::is_relaxed<Concept>));
_boost_type_erasure_data.data = 0;
}
template<class U,
typename ::boost::enable_if_c<
!::boost::type_erasure::detail::is_any_arg<U>::value &&
!::boost::type_erasure::detail::is_binding_arg<U>::value &&
!::boost::type_erasure::detail::is_static_binding_arg<U>::value
>::type* = nullptr
>
any_constructor_impl(U&& data_arg)
: _boost_type_erasure_table((
BOOST_TYPE_ERASURE_INSTANTIATE1(Concept, T, ::boost::decay_t<U>),
::boost::type_erasure::make_binding<
::boost::mpl::map1< ::boost::mpl::pair<T, ::boost::decay_t<U> > >
>()
)),
_boost_type_erasure_data(std::forward<U>(data_arg))
{}
template<class U, class Map,
typename ::boost::enable_if_c<
!::boost::type_erasure::detail::is_any_arg<U>::value &&
!::boost::type_erasure::detail::is_binding_arg<U>::value &&
!::boost::type_erasure::detail::is_static_binding_arg<U>::value
>::type* = nullptr
>
any_constructor_impl(U&& data_arg, const static_binding<Map>& b)
: _boost_type_erasure_table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
b
)),
_boost_type_erasure_data(std::forward<U>(data_arg))
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, ::boost::decay_t<U> >));
}
template<class U,
typename ::boost::enable_if_c<
::boost::type_erasure::is_subconcept<
Concept, typename ::boost::type_erasure::detail::safe_concept_of<U>::type,
typename ::boost::mpl::if_c< ::boost::is_same<T, ::boost::type_erasure::detail::safe_placeholder_t<U> >::value,
void,
::boost::mpl::map1<
::boost::mpl::pair<T, ::boost::type_erasure::detail::safe_placeholder_t<U> >
>
>::type
>::value
>::type* = nullptr
>
any_constructor_impl(U&& other)
: _boost_type_erasure_table(
::boost::type_erasure::detail::access::table(other),
typename ::boost::mpl::if_c< ::boost::is_same<T, ::boost::type_erasure::detail::safe_placeholder_t<U> >::value,
#ifndef BOOST_TYPE_ERASURE_USE_MP11
::boost::type_erasure::detail::substitution_map< ::boost::mpl::map0<> >,
#else
::boost::type_erasure::detail::make_identity_placeholder_map<Concept>,
#endif
::boost::mpl::map1<
::boost::mpl::pair<
T,
::boost::type_erasure::detail::safe_placeholder_t<U>
>
>
>::type()
),
_boost_type_erasure_data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? other._boost_type_erasure_deduce_constructor(std::forward<U>(other)) : 0
), std::forward<U>(other))
)
{}
template<class U,
typename ::boost::enable_if_c<
::boost::type_erasure::detail::is_any_arg<U>::value
>::type* = nullptr
>
any_constructor_impl(U&& other, const binding<Concept>& binding_arg)
: _boost_type_erasure_table(binding_arg),
_boost_type_erasure_data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? other._boost_type_erasure_deduce_constructor(std::forward<U>(other)) : 0
), std::forward<U>(other))
)
{}
template<class U, class Map,
typename ::boost::enable_if_c<
::boost::type_erasure::is_subconcept<
Concept, typename ::boost::type_erasure::detail::safe_concept_of<U>::type,
Map
>::value
>::type* = nullptr
>
any_constructor_impl(U&& other, const static_binding<Map>& binding_arg)
: _boost_type_erasure_table(::boost::type_erasure::detail::access::table(other), binding_arg),
_boost_type_erasure_data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? other._boost_type_erasure_deduce_constructor(std::forward<U>(other)) : 0
), std::forward<U>(other))
)
{}
any_constructor_impl(const any_constructor_impl& other)
: _boost_type_erasure_table(
::boost::type_erasure::detail::access::table(other)
),
_boost_type_erasure_data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? other._boost_type_erasure_deduce_constructor(
static_cast<typename _boost_type_erasure_base::_boost_type_erasure_derived_type const&>(other)) : 0
), other)
)
{}
any_constructor_impl(any_constructor_impl& other)
: _boost_type_erasure_table(
::boost::type_erasure::detail::access::table(other)
),
_boost_type_erasure_data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? other._boost_type_erasure_deduce_constructor(
static_cast<typename _boost_type_erasure_base::_boost_type_erasure_derived_type &>(other)) : 0
), other)
)
{}
any_constructor_impl(any_constructor_impl&& other)
: _boost_type_erasure_table(
::boost::type_erasure::detail::access::table(other)
),
_boost_type_erasure_data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? other._boost_type_erasure_deduce_constructor(
static_cast<typename _boost_type_erasure_base::_boost_type_erasure_derived_type &&>(other)) : 0
), std::move(other))
)
{}

template<class R, class... A, class... U>
const _boost_type_erasure_table_type& _boost_type_erasure_extract_table(
::boost::type_erasure::constructible<R(A...)>*,
U&&... u)
{
return *::boost::type_erasure::detail::extract_table(static_cast<void(*)(A...)>(0), u...);
}
template<class... U,
typename ::boost::enable_if_c<
::boost::type_erasure::detail::has_constructor<any_constructor_impl, U...>::value
>::type* = nullptr
>
explicit any_constructor_impl(U&&... u)
: _boost_type_erasure_table(
_boost_type_erasure_extract_table(
false? this->_boost_type_erasure_deduce_constructor(std::forward<U>(u)...) : 0,
std::forward<U>(u)...
)
),
_boost_type_erasure_data(
::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(std::forward<U>(u)...) : 0
),
std::forward<U>(u)...
)
)
{}
template<class... U,
typename ::boost::enable_if_c<
::boost::type_erasure::detail::has_constructor<any_constructor_impl, U...>::value
>::type* = nullptr
>
explicit any_constructor_impl(const binding<Concept>& binding_arg, U&&... u)
: _boost_type_erasure_table(binding_arg),
_boost_type_erasure_data(
::boost::type_erasure::call(
binding_arg,
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(std::forward<U>(u)...) : 0
),
std::forward<U>(u)...
)
)
{}


any_constructor_impl& operator=(const any_constructor_impl& other)
{
static_cast<typename _boost_type_erasure_base::_boost_type_erasure_derived_type*>(this)->_boost_type_erasure_resolve_assign(
static_cast<const typename _boost_type_erasure_base::_boost_type_erasure_derived_type&>(other));
return *this;
}

any_constructor_impl& operator=(any_constructor_impl& other)
{
static_cast<typename _boost_type_erasure_base::_boost_type_erasure_derived_type*>(this)->_boost_type_erasure_resolve_assign(
static_cast<typename _boost_type_erasure_base::_boost_type_erasure_derived_type&>(other));
return *this;
}

any_constructor_impl& operator=(any_constructor_impl&& other)
{
static_cast<typename _boost_type_erasure_base::_boost_type_erasure_derived_type*>(this)->_boost_type_erasure_resolve_assign(
static_cast<typename _boost_type_erasure_base::_boost_type_erasure_derived_type&&>(other));
return *this;
}

~any_constructor_impl()
{
_boost_type_erasure_table.template find<
::boost::type_erasure::destructible<T>
>()(_boost_type_erasure_data);
}

protected:
friend struct ::boost::type_erasure::detail::access;

_boost_type_erasure_table_type _boost_type_erasure_table;
::boost::type_erasure::detail::storage _boost_type_erasure_data;
};

namespace detail {

#endif

template<class T>
struct is_rvalue_for_any : 
::boost::mpl::not_<
::boost::is_lvalue_reference<T>
>
{};

template<class C, class P>
struct is_rvalue_for_any<any<C, P> > :
::boost::mpl::not_<
::boost::is_lvalue_reference<P>
>
{};

}


template<class Concept, class T = _self>
class any :
#ifdef BOOST_TYPE_ERASURE_SFINAE_FRIENDLY_CONSTRUCTORS
public ::boost::type_erasure::any_constructor_control<
::boost::type_erasure::any_constructor_impl<

Concept,
T
>
>
#else
public ::boost::type_erasure::detail::compute_bases<
::boost::type_erasure::any<Concept, T>,
Concept,
T
>::type
#endif
{
typedef ::boost::type_erasure::binding<Concept> table_type;
public:

typedef Concept _boost_type_erasure_concept_type;

#if defined(BOOST_TYPE_ERASURE_SFINAE_FRIENDLY_CONSTRUCTORS)
using _boost_type_erasure_base = ::boost::type_erasure::any_constructor_control<
::boost::type_erasure::any_constructor_impl<
Concept,
T
>
>;
using _boost_type_erasure_base::_boost_type_erasure_base;
#else


any(const ::boost::type_erasure::detail::storage& data_arg, const table_type& table_arg)
: table(table_arg),
data(data_arg)
{}
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

any(::boost::type_erasure::detail::storage&& data_arg, const table_type& table_arg)
: table(table_arg),
data(data_arg)
{}
#endif

any()
{
BOOST_MPL_ASSERT((::boost::type_erasure::is_relaxed<Concept>));
data.data = 0;
}

#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES)

template<class U>
any(const U& data_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE1(Concept, T, U),
::boost::type_erasure::make_binding<
::boost::mpl::map1< ::boost::mpl::pair<T, U> >
>()
)),
data(data_arg)
{}
template<class U, class Map>
any(const U& data_arg, const static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U>));
}

#else


template<class U>
any(U&& data_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE1(Concept, T, typename ::boost::remove_cv<typename ::boost::remove_reference<U>::type>::type),
::boost::type_erasure::make_binding<
::boost::mpl::map1< ::boost::mpl::pair<T, typename ::boost::remove_cv<typename ::boost::remove_reference<U>::type>::type> >
>()
)),
data(std::forward<U>(data_arg))
{}

template<class U, class Map>
any(U&& data_arg, const static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(std::forward<U>(data_arg))
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, typename ::boost::remove_cv<typename ::boost::remove_reference<U>::type>::type>));
}

#endif


template<class U>
any(U* data_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE1(Concept, T, U*),
::boost::type_erasure::make_binding<
::boost::mpl::map1< ::boost::mpl::pair<T, U*> >
>()
)),
data(data_arg)
{}

template<class U, class Map>
any(U* data_arg, const static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U*>));
}

any(const any& other)
: table(other.table),
data(::boost::type_erasure::call(constructible<T(const T&)>(), other))
{}

template<class Concept2, class Tag2>
any(const any<Concept2, Tag2>& other)
: table(
::boost::type_erasure::detail::access::table(other),
::boost::mpl::map1<
::boost::mpl::pair<
T,
typename ::boost::remove_const<
typename ::boost::remove_reference<Tag2>::type
>::type
>
>()
),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}

template<class Concept2, class Tag2, class Map>
any(const any<Concept2, Tag2>& other, const static_binding<Map>& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}

template<class Concept2, class Tag2>
any(const any<Concept2, Tag2>& other, const binding<Concept>& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}

#ifdef BOOST_TYPE_ERASURE_DOXYGEN


template<class... U>
explicit any(U&&... arg);


template<class... U>
explicit any(const binding<Concept>& binding_arg, U&&... arg)
: table(binding_arg),
data(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(arg...) : 0
)(arg...)
)
{}

#else
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
any(any&& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(std::move(other)) : 0
), std::move(other))
)
{}
any(any& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
template<class Concept2, class Tag2>
any(any<Concept2, Tag2>& other)
: table(
::boost::type_erasure::detail::access::table(other),
::boost::mpl::map1<
::boost::mpl::pair<
T,
typename ::boost::remove_const<
typename ::boost::remove_reference<Tag2>::type
>::type
>
>()
),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? other._boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
template<class Concept2, class Tag2>
any(any<Concept2, Tag2>&& other)
: table(
::boost::type_erasure::detail::access::table(other),
::boost::mpl::map1<
::boost::mpl::pair<
T,
typename ::boost::remove_const<
typename ::boost::remove_reference<Tag2>::type
>::type
>
>()
),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? other._boost_type_erasure_deduce_constructor(std::move(other)) : 0
), std::move(other))
)
{}
#endif
any(const any<Concept, T&>& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
any(any<Concept, T&>& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
any(any<Concept, T&>&& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
#endif
any(const any<Concept, const T&>& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
any(any<Concept, const T&>& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
any(any<Concept, const T&>&& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
#endif

template<class U, class Map>
any(U* data_arg, static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U*>));
}
#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
template<class U, class Map>
any(U& data_arg, static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U>));
}
template<class U, class Map>
any(const U& data_arg, static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U>));
}
template<class U, class Map>
any(U& data_arg, const static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U>));
}
#endif
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template<class U, class Map>
any(U* data_arg, static_binding<Map>&& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U*>));
}
template<class U, class Map>
any(U&& data_arg, static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, typename ::boost::remove_cv<typename ::boost::remove_reference<U>::type>::type>));
}
template<class U, class Map>
any(U&& data_arg, static_binding<Map>&& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
)),
data(data_arg)
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, typename ::boost::remove_cv<typename ::boost::remove_reference<U>::type>::type>));
}
#endif
template<class Concept2, class Tag2, class Map>
any(any<Concept2, Tag2>& other, static_binding<Map>& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2, class Map>
any(any<Concept2, Tag2>& other, const static_binding<Map>& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2, class Map>
any(const any<Concept2, Tag2>& other, static_binding<Map>& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2>
any(any<Concept2, Tag2>& other, binding<Concept>& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2>
any(any<Concept2, Tag2>& other, const binding<Concept>& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2>
any(const any<Concept2, Tag2>& other, binding<Concept>& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template<class Concept2, class Tag2, class Map>
any(any<Concept2, Tag2>& other, static_binding<Map>&& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2, class Map>
any(const any<Concept2, Tag2>& other, static_binding<Map>&& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2, class Map>
any(any<Concept2, Tag2>&& other, static_binding<Map>&& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), std::move(other))
)
{}
template<class Concept2, class Tag2, class Map>
any(any<Concept2, Tag2>&& other, static_binding<Map>& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), std::move(other))
)
{}
template<class Concept2, class Tag2, class Map>
any(any<Concept2, Tag2>&& other, const static_binding<Map>& binding_arg)
: table(::boost::type_erasure::detail::access::table(other), binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), std::move(other))
)
{}
template<class Concept2, class Tag2>
any(any<Concept2, Tag2>& other, binding<Concept>&& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2>
any(const any<Concept2, Tag2>& other, binding<Concept>&& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), other)
)
{}
template<class Concept2, class Tag2>
any(any<Concept2, Tag2>&& other, binding<Concept>&& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), std::move(other))
)
{}
template<class Concept2, class Tag2>
any(any<Concept2, Tag2>&& other, binding<Concept>& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), std::move(other))
)
{}
template<class Concept2, class Tag2>
any(any<Concept2, Tag2>&& other, const binding<Concept>& binding_arg)
: table(binding_arg),
data(::boost::type_erasure::call(
constructible<
typename ::boost::remove_const<
typename boost::remove_reference<Tag2>::type
>::type(const typename boost::remove_reference<Tag2>::type&)
>(), std::move(other))
)
{}
#endif

template<class Tag2>
explicit any(const any<Concept, Tag2>& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}
template<class Tag2>
explicit any(any<Concept, Tag2>& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(other) : 0
), other)
)
{}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template<class Tag2>
explicit any(any<Concept, Tag2>&& other)
: table(::boost::type_erasure::detail::access::table(other)),
data(::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(std::move(other)) : 0
), std::move(other))
)
{}
#endif

explicit any(const binding<Concept>& binding_arg)
: table(binding_arg),
data(
::boost::type_erasure::call(
binding_arg,
::boost::type_erasure::constructible<T()>()
)
)
{}
explicit any(binding<Concept>& binding_arg)
: table(binding_arg),
data(
::boost::type_erasure::call(
binding_arg,
::boost::type_erasure::constructible<T()>()
)
)
{}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

explicit any(binding<Concept>&& binding_arg)
: table(binding_arg),
data(
::boost::type_erasure::call(
binding_arg,
::boost::type_erasure::constructible<T()>()
)
)
{}

#endif

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)

template<class R, class... A, class... U>
const table_type& _boost_type_erasure_extract_table(
::boost::type_erasure::constructible<R(A...)>*,
U&&... u)
{
return *::boost::type_erasure::detail::extract_table(static_cast<void(*)(A...)>(0), u...);
}

template<class U0, class U1, class... U>
any(U0&& u0, U1&& u1, U&&... u)
: table(
_boost_type_erasure_extract_table(
false? this->_boost_type_erasure_deduce_constructor(std::forward<U0>(u0), std::forward<U1>(u1), std::forward<U>(u)...) : 0,
std::forward<U0>(u0), std::forward<U1>(u1), std::forward<U>(u)...
)
),
data(
::boost::type_erasure::call(
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(std::forward<U0>(u0), std::forward<U1>(u1), std::forward<U>(u)...) : 0
),
std::forward<U0>(u0), std::forward<U1>(u1), std::forward<U>(u)...
)
)
{}

template<class U0, class... U>
any(const binding<Concept>& binding_arg, U0&& u0, U&&... u)
: table(binding_arg),
data(
::boost::type_erasure::call(
binding_arg,
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(std::forward<U0>(u0), std::forward<U>(u)...) : 0
),
std::forward<U0>(u0), std::forward<U>(u)...
)
)
{}

template<class U0, class... U>
any(binding<Concept>& binding_arg, U0&& u0, U&&... u)
: table(binding_arg),
data(
::boost::type_erasure::call(
binding_arg,
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(std::forward<U0>(u0), std::forward<U>(u)...) : 0
),
std::forward<U0>(u0), std::forward<U>(u)...
)
)
{}
template<class U0, class... U>
any(binding<Concept>&& binding_arg, U0&& u0, U&&... u)
: table(binding_arg),
data(
::boost::type_erasure::call(
binding_arg,
::boost::type_erasure::detail::make(
false? this->_boost_type_erasure_deduce_constructor(std::forward<U0>(u0), std::forward<U>(u)...) : 0
),
std::forward<U0>(u0), std::forward<U>(u)...
)
)
{}

#else

#include <boost/type_erasure/detail/construct.hpp>

#endif

#endif


any& operator=(const any& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}

any& operator=(any& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}

#endif 

#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
template<class U>
any& operator=(U& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}
template<class U>
any& operator=(const U& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}
#else
#ifndef BOOST_TYPE_ERASURE_SFINAE_FRIENDLY_CONSTRUCTORS

any& operator=(any&& other)
{
_boost_type_erasure_resolve_assign(std::move(other));
return *this;
}
#endif

template<class U>
any& operator=(U&& other)
{
_boost_type_erasure_resolve_assign(std::forward<U>(other));
return *this;
}
#endif

#ifndef BOOST_TYPE_ERASURE_SFINAE_FRIENDLY_CONSTRUCTORS

~any()
{
::boost::type_erasure::detail::access::table(*this).template find<
::boost::type_erasure::destructible<T>
>()(::boost::type_erasure::detail::access::data(*this));
}
#endif

#ifndef BOOST_NO_CXX11_REF_QUALIFIERS

operator param<Concept, T&>() &
{
return param<Concept, T&>(
boost::type_erasure::detail::access::data(*this),
boost::type_erasure::detail::access::table(*this));
}

operator param<Concept, T&&>() && {
return param<Concept, T&&>(
boost::type_erasure::detail::access::data(*this),
boost::type_erasure::detail::access::table(*this));
}
#endif
private:
#ifndef BOOST_TYPE_ERASURE_SFINAE_FRIENDLY_CONSTRUCTORS

void _boost_type_erasure_swap(any& other)
{
::std::swap(data, other.data);
::std::swap(table, other.table);
}
#else
void _boost_type_erasure_swap(any& other)
{
::std::swap(this->_boost_type_erasure_data, other._boost_type_erasure_data);
::std::swap(this->_boost_type_erasure_table, other._boost_type_erasure_table);
}
#endif
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

template<class Other>
void _boost_type_erasure_resolve_assign(Other&& other)
{
_boost_type_erasure_assign_impl(
std::forward<Other>(other),
false? this->_boost_type_erasure_deduce_assign(
::boost::type_erasure::detail::make_fallback(
std::forward<Other>(other),
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_assign(std::forward<Other>(other))
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
::boost::type_erasure::is_relaxed<Concept>()
);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
::boost::mpl::false_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, std::forward<Other>(other));
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
::boost::mpl::true_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, std::forward<Other>(other));
}

template<class Other>
void _boost_type_erasure_assign_impl(
Other&& other,
const void*,
::boost::mpl::true_)
{
any temp(std::forward<Other>(other));
_boost_type_erasure_swap(temp);
}
#else

template<class Other>
void _boost_type_erasure_resolve_assign(Other& other)
{
_boost_type_erasure_assign_impl(
other,
false? this->_boost_type_erasure_deduce_assign(
::boost::type_erasure::detail::make_fallback(
other,
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_assign(other)
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
::boost::type_erasure::is_relaxed<Concept>()
);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other& other,
const assignable<T, U>*,
::boost::mpl::false_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, other);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other& other,
const assignable<T, U>*,
::boost::mpl::true_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, other);
}

template<class Other>
void _boost_type_erasure_assign_impl(
Other& other,
const void*,
::boost::mpl::true_)
{
any temp(other);
_boost_type_erasure_swap(temp);
}
#endif

template<class Concept2, class Tag2>
void _boost_type_erasure_resolve_assign(const any<Concept2, Tag2>& other)
{
_boost_type_erasure_resolve_assign_any(other);
}

template<class Concept2, class Tag2>
void _boost_type_erasure_resolve_assign(any<Concept2, Tag2>& other)
{
_boost_type_erasure_resolve_assign_any(other);
}
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

template<class Concept2, class Tag2>
void _boost_type_erasure_resolve_assign(any<Concept2, Tag2>&& other)
{
_boost_type_erasure_resolve_assign_any(std::move(other));
}

template<class Other>
void _boost_type_erasure_resolve_assign_any(Other&& other)
{
_boost_type_erasure_assign_impl(
std::forward<Other>(other),
false? this->_boost_type_erasure_deduce_assign(
::boost::type_erasure::detail::make_fallback(
std::forward<Other>(other),
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_assign(std::forward<Other>(other))
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
false? this->_boost_type_erasure_deduce_constructor(
::boost::type_erasure::detail::make_fallback(
std::forward<Other>(other),
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_constructor(std::forward<Other>(other))
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
::boost::type_erasure::is_relaxed<Concept>()
);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
const void*,
::boost::mpl::false_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, std::forward<Other>(other));
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
const void*,
::boost::mpl::true_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, std::forward<Other>(other));
}

template<class Other, class Sig>
void _boost_type_erasure_assign_impl(
Other&& other,
const void*,
const constructible<Sig>*,
::boost::mpl::true_)
{
any temp(std::forward<Other>(other));
_boost_type_erasure_swap(temp);
}

template<class Other, class U, class Sig>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
const constructible<Sig>*,
::boost::mpl::true_)
{
if(::boost::type_erasure::check_match(assignable<T, U>(), *this, other))  
{
::boost::type_erasure::unchecked_call(assignable<T, U>(), *this, std::forward<Other>(other));
}
else
{
any temp(std::forward<Other>(other));
_boost_type_erasure_swap(temp);
}
}
#else

template<class Other>
void _boost_type_erasure_resolve_assign_any(Other& other)
{
_boost_type_erasure_assign_impl(
other,
false? this->_boost_type_erasure_deduce_assign(
::boost::type_erasure::detail::make_fallback(
other,
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_assign(other)
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
false? this->_boost_type_erasure_deduce_constructor(
::boost::type_erasure::detail::make_fallback(
other,
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_constructor(other)
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
::boost::type_erasure::is_relaxed<Concept>()
);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other& other,
const assignable<T, U>*,
const void*,
::boost::mpl::false_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, other);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other& other,
const assignable<T, U>*,
const void*,
::boost::mpl::true_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, other);
}

template<class Other, class Sig>
void _boost_type_erasure_assign_impl(
Other& other,
const void*,
const constructible<Sig>*,
::boost::mpl::true_)
{
any temp(other);
_boost_type_erasure_swap(temp);
}

template<class Other, class U, class Sig>
void _boost_type_erasure_assign_impl(
Other& other,
const assignable<T, U>*,
const constructible<Sig>*,
::boost::mpl::true_)
{
if(::boost::type_erasure::check_match(assignable<T, U>(), *this, other))
{
::boost::type_erasure::unchecked_call(assignable<T, U>(), *this, other);
}
else
{
any temp(other);
_boost_type_erasure_swap(temp);
}
}
#endif
friend struct ::boost::type_erasure::detail::access;
#ifndef BOOST_TYPE_ERASURE_SFINAE_FRIENDLY_CONSTRUCTORS
table_type table;
::boost::type_erasure::detail::storage data;
#else
template<class Concept2, class T2>
friend struct ::boost::type_erasure::any_constructor_impl;
#endif
};

template<class Concept, class T>
class any<Concept, T&> :
public ::boost::type_erasure::detail::compute_bases<
::boost::type_erasure::any<Concept, T&>,
Concept,
T
>::type
{
typedef ::boost::type_erasure::binding<Concept> table_type;
public:

typedef Concept _boost_type_erasure_concept_type;

any(const ::boost::type_erasure::detail::storage& data_arg,
const table_type& table_arg)
: data(data_arg),
table(table_arg)
{}

template<class U>
any(U& arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
,  typename ::boost::disable_if<
::boost::mpl::or_<
::boost::is_const<U>,
::boost::type_erasure::detail::is_any<U>
>
>::type* = 0
#endif
)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE1(Concept, T, U),
::boost::type_erasure::make_binding<
::boost::mpl::map1< ::boost::mpl::pair<T, U> >
>()
))
{
data.data = ::boost::addressof(arg);
}

template<class U, class Map>
any(U& arg, const static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
))
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U>));
data.data = ::boost::addressof(arg);
}

any(const any& other)
: data(other.data),
table(other.table)
{}
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
any(any& other)
: data(other.data),
table(other.table)
{}
#endif

any(any<Concept, T>& other)
: data(::boost::type_erasure::detail::access::data(other)),
table(::boost::type_erasure::detail::access::table(other))
{}

template<class Concept2, class Tag2>
any(const any<Concept2, Tag2&>& other
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::mpl::or_<
::boost::is_same<Concept, Concept2>,
::boost::is_const<Tag2>
>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(
::boost::type_erasure::detail::access::table(other),
::boost::mpl::map1<
::boost::mpl::pair<
T,
Tag2
>
>())
{}

template<class Concept2, class Tag2>
any(any<Concept2, Tag2>& other
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::mpl::or_<
::boost::is_same<Concept, Concept2>,
::boost::is_const<typename ::boost::remove_reference<Tag2>::type>
>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(
::boost::type_erasure::detail::access::table(other),
::boost::mpl::map1<
::boost::mpl::pair<
T,
typename ::boost::remove_reference<Tag2>::type
>
>())
{}

template<class Concept2, class Tag2, class Map>
any(const any<Concept2, Tag2&>& other, const static_binding<Map>& binding_arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if< ::boost::is_const<Tag2> >::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(::boost::type_erasure::detail::access::table(other), binding_arg)
{}

template<class Concept2, class Tag2, class Map>
any(any<Concept2, Tag2>& other, const static_binding<Map>& binding_arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::is_const<typename ::boost::remove_reference<Tag2>::type>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(::boost::type_erasure::detail::access::table(other), binding_arg)
{}

template<class Concept2, class Tag2>
any(const any<Concept2, Tag2&>& other, const binding<Concept>& binding_arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::is_const<Tag2>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(binding_arg)
{}

template<class Concept2, class Tag2>
any(any<Concept2, Tag2>& other, const binding<Concept>& binding_arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::is_const<typename ::boost::remove_reference<Tag2>::type>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(binding_arg)
{}


any& operator=(const any& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}


any& operator=(any& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

any& operator=(any&& other)
{
_boost_type_erasure_resolve_assign(std::move(other));
return *this;
}

template<class U>
any& operator=(U&& other)
{
_boost_type_erasure_resolve_assign(std::forward<U>(other));
return *this;
}
#else
template<class U>
any& operator=(U& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}

template<class U>
any& operator=(const U& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}
#endif

#ifndef BOOST_NO_CXX11_REF_QUALIFIERS

operator param<Concept, T&>() const { return param<Concept, T&>(data, table); }
#endif
private:


void _boost_type_erasure_swap(any& other)
{
::std::swap(data, other.data);
::std::swap(table, other.table);
}
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

template<class Other>
void _boost_type_erasure_resolve_assign(Other&& other)
{
_boost_type_erasure_assign_impl(
std::forward<Other>(other),
false? this->_boost_type_erasure_deduce_assign(
::boost::type_erasure::detail::make_fallback(
std::forward<Other>(other),
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_assign(std::forward<Other>(other))
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
::boost::mpl::and_<
::boost::type_erasure::is_relaxed<Concept>,
::boost::is_convertible<Other, any>
#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1900))
, ::boost::mpl::not_<
::boost::type_erasure::detail::is_rvalue_for_any<Other>
>
#endif
>()
);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
::boost::mpl::false_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, std::forward<Other>(other));
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
::boost::mpl::true_)
{
if(::boost::type_erasure::check_match(assignable<T, U>(), *this, other)) {
::boost::type_erasure::unchecked_call(assignable<T, U>(), *this, std::forward<Other>(other));
} else {
any temp(std::forward<Other>(other));
_boost_type_erasure_swap(temp);
}
}

template<class Other>
void _boost_type_erasure_assign_impl(
Other&& other,
const void*,
::boost::mpl::true_)
{
any temp(std::forward<Other>(other));
_boost_type_erasure_swap(temp);
}
#else

template<class Other>
void _boost_type_erasure_resolve_assign(Other& other)
{
_boost_type_erasure_assign_impl(
other,
false? this->_boost_type_erasure_deduce_assign(
::boost::type_erasure::detail::make_fallback(
other,
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_assign(other)
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
::boost::mpl::and_<
::boost::type_erasure::is_relaxed<Concept>,
::boost::is_convertible<Other&, any>
>()
);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other& other,
const assignable<T, U>*,
::boost::mpl::false_)
{
::boost::type_erasure::call(assignable<T, U>(), *this, other);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other& other,
const assignable<T, U>*,
::boost::mpl::true_)
{
if(::boost::type_erasure::check_match(assignable<T, U>(), *this, other)) {
::boost::type_erasure::unchecked_call(assignable<T, U>(), *this, other);
} else {
any temp(other);
_boost_type_erasure_swap(temp);
}
}

template<class Other>
void _boost_type_erasure_assign_impl(
Other& other,
const void*,
::boost::mpl::true_)
{
any temp(other);
_boost_type_erasure_swap(temp);
}
#endif

friend struct ::boost::type_erasure::detail::access;
::boost::type_erasure::detail::storage data;
table_type table;
};

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template<class Concept, class T>
class any<Concept, const T&> :
public ::boost::type_erasure::detail::compute_bases<
::boost::type_erasure::any<Concept, const T&>,
Concept,
T
>::type
{
typedef ::boost::type_erasure::binding<Concept> table_type;
public:

typedef Concept _boost_type_erasure_concept_type;

any(const ::boost::type_erasure::detail::storage& data_arg,
const table_type& table_arg)
: data(data_arg),
table(table_arg)
{}

template<class U>
any(const U& arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE1(Concept, T, U),
::boost::type_erasure::make_binding<
::boost::mpl::map1< ::boost::mpl::pair<T, U> >
>()
))
{
data.data = const_cast<void*>(static_cast<const void*>(::boost::addressof(arg)));
}

template<class U, class Map>
any(const U& arg, const static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
))
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U>));
data.data = const_cast<void*>(static_cast<const void*>(::boost::addressof(arg)));
}

any(const any& other)
: data(other.data),
table(other.table)
{}

any(const any<Concept, T&>& other)
: data(::boost::type_erasure::detail::access::data(other)),
table(::boost::type_erasure::detail::access::table(other))
{}

any(const any<Concept, T>& other)
: data(::boost::type_erasure::detail::access::data(other)),
table(::boost::type_erasure::detail::access::table(other))
{}
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

any(const any<Concept, T&&>& other)
: data(::boost::type_erasure::detail::access::data(other)),
table(::boost::type_erasure::detail::access::table(other))
{}
#endif

template<class Concept2, class Tag2>
any(const any<Concept2, Tag2>& other
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if< ::boost::is_same<Concept, Concept2> >::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(
::boost::type_erasure::detail::access::table(other),
::boost::mpl::map1<
::boost::mpl::pair<
T,
typename ::boost::remove_const<
typename ::boost::remove_reference<Tag2>::type
>::type
>
>())
{}

template<class Concept2, class Tag2, class Map>
any(const any<Concept2, Tag2>& other, const static_binding<Map>& binding_arg)
: data(::boost::type_erasure::detail::access::data(other)),
table(::boost::type_erasure::detail::access::table(other), binding_arg)
{}

template<class Concept2, class Tag2>
any(const any<Concept2, Tag2>& other, const binding<Concept>& binding_arg)
: data(::boost::type_erasure::detail::access::data(other)),
table(binding_arg)
{}



any& operator=(const any& other)
{
BOOST_MPL_ASSERT((::boost::type_erasure::is_relaxed<Concept>));
any temp(other);
_boost_type_erasure_swap(temp);
return *this;
}

template<class U>
any& operator=(const U& other)
{
BOOST_MPL_ASSERT((::boost::type_erasure::is_relaxed<Concept>));
any temp(other);
_boost_type_erasure_swap(temp);
return *this;
}

#ifndef BOOST_NO_CXX11_REF_QUALIFIERS

operator param<Concept, const T&>() const { return param<Concept, const T&>(data, table); }
#endif
private:

void _boost_type_erasure_swap(any& other)
{
::std::swap(data, other.data);
::std::swap(table, other.table);
}
friend struct ::boost::type_erasure::detail::access;
::boost::type_erasure::detail::storage data;
table_type table;
};

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

template<class Concept, class T>
class any<Concept, T&&> :
public ::boost::type_erasure::detail::compute_bases<
::boost::type_erasure::any<Concept, T&&>,
Concept,
T
>::type
{
typedef ::boost::type_erasure::binding<Concept> table_type;
public:

typedef Concept _boost_type_erasure_concept_type;

any(const ::boost::type_erasure::detail::storage& data_arg,
const table_type& table_arg)
: data(data_arg),
table(table_arg)
{}

template<class U>
any(U&& arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
,  typename ::boost::disable_if<
::boost::mpl::or_<
::boost::is_reference<U>,
::boost::is_const<U>,
::boost::type_erasure::detail::is_any<U>
>
>::type* = 0
#endif
)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE1(Concept, T, U),
::boost::type_erasure::make_binding<
::boost::mpl::map1< ::boost::mpl::pair<T, U> >
>()
))
{
data.data = ::boost::addressof(arg);
}

template<class U, class Map>
any(U&& arg, const static_binding<Map>& binding_arg)
: table((
BOOST_TYPE_ERASURE_INSTANTIATE(Concept, Map),
binding_arg
))
{
BOOST_MPL_ASSERT((::boost::is_same<
typename ::boost::mpl::at<Map, T>::type, U>));
data.data = ::boost::addressof(arg);
}

#ifndef BOOST_TYPE_ERASURE_DOXYGEN
any(any&& other)
: data(other.data),
table(std::move(other.table))
{}
any(const any& other)
: data(other.data),
table(other.table)
{}
#endif

any(any<Concept, T>&& other)
: data(::boost::type_erasure::detail::access::data(other)),
table(std::move(::boost::type_erasure::detail::access::table(other)))
{}

template<class Concept2, class Tag2>
any(any<Concept2, Tag2&&>&& other
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::mpl::or_<
::boost::is_reference<Tag2>,
::boost::is_same<Concept, Concept2>,
::boost::is_const<Tag2>
>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(
std::move(::boost::type_erasure::detail::access::table(other)),
::boost::mpl::map1<
::boost::mpl::pair<
T,
Tag2
>
>())
{}

template<class Concept2, class Tag2>
any(any<Concept2, Tag2>&& other
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::mpl::or_<
::boost::is_same<Concept, Concept2>,
::boost::is_const<typename ::boost::remove_reference<Tag2>::type>
>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(
std::move(::boost::type_erasure::detail::access::table(other)),
::boost::mpl::map1<
::boost::mpl::pair<
T,
typename ::boost::remove_reference<Tag2>::type
>
>())
{}

template<class Concept2, class Tag2, class Map>
any(const any<Concept2, Tag2&&>& other, const static_binding<Map>& binding_arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if< ::boost::is_const<Tag2> >::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(std::move(::boost::type_erasure::detail::access::table(other)), binding_arg)
{}

template<class Concept2, class Tag2, class Map>
any(any<Concept2, Tag2>&& other, const static_binding<Map>& binding_arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::is_const<typename ::boost::remove_reference<Tag2>::type>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(::boost::type_erasure::detail::access::table(other), binding_arg)
{}

template<class Concept2, class Tag2>
any(const any<Concept2, Tag2&&>& other, const binding<Concept>& binding_arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::is_const<Tag2>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(binding_arg)
{}

template<class Concept2, class Tag2>
any(any<Concept2, Tag2>&& other, const binding<Concept>& binding_arg
#ifndef BOOST_TYPE_ERASURE_DOXYGEN
, typename ::boost::disable_if<
::boost::is_const<typename ::boost::remove_reference<Tag2>::type>
>::type* = 0
#endif
)
: data(::boost::type_erasure::detail::access::data(other)),
table(binding_arg)
{}


any& operator=(const any& other)
{
_boost_type_erasure_resolve_assign(other);
return *this;
}


template<class U>
any& operator=(U&& other)
{
_boost_type_erasure_resolve_assign(std::forward<U>(other));
return *this;
}

#ifndef BOOST_NO_CXX11_REF_QUALIFIERS

operator param<Concept, T&&>() const { return param<Concept, T&&>(data, table); }
#endif
private:


void _boost_type_erasure_swap(any& other)
{
::std::swap(data, other.data);
::std::swap(table, other.table);
}

template<class Other>
void _boost_type_erasure_resolve_assign(Other&& other)
{
_boost_type_erasure_assign_impl(
std::forward<Other>(other),
false? this->_boost_type_erasure_deduce_assign(
::boost::type_erasure::detail::make_fallback(
std::forward<Other>(other),
::boost::mpl::bool_<
sizeof(
::boost::type_erasure::detail::check_overload(
::boost::declval<any&>().
_boost_type_erasure_deduce_assign(std::forward<Other>(other))
)
) == sizeof(::boost::type_erasure::detail::yes)
>()
)
) : 0,
::boost::mpl::and_<
::boost::type_erasure::is_relaxed<Concept>,
::boost::is_convertible<Other, any>
>()
);
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
::boost::mpl::false_)
{
::boost::type_erasure::call(
assignable<T, U>(),
::boost::type_erasure::param<Concept, T&>(data, table),
std::forward<Other>(other));
}

template<class Other, class U>
void _boost_type_erasure_assign_impl(
Other&& other,
const assignable<T, U>*,
::boost::mpl::true_)
{
if(::boost::type_erasure::check_match(assignable<T, U>(), *this, other)) {
::boost::type_erasure::unchecked_call(
assignable<T, U>(),
::boost::type_erasure::param<Concept, T&>(data, table),
std::forward<Other>(other));
} else {
any temp(std::forward<Other>(other));
_boost_type_erasure_swap(temp);
}
}

template<class Other>
void _boost_type_erasure_assign_impl(
Other&& other,
const void*,
::boost::mpl::true_)
{
any temp(std::forward<Other>(other));
_boost_type_erasure_swap(temp);
}

friend struct ::boost::type_erasure::detail::access;
::boost::type_erasure::detail::storage data;
table_type table;
};

#endif

#ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
template<class Concept, class T>
using any_ref = any<Concept, T&>;
template<class Concept, class T>
using any_cref = any<Concept, const T&>;
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template<class Concept, class T>
using any_rvref = any<Concept, T&&>;
#endif
#endif

}
}

#endif
