
#ifndef BOOST_VARIANT_GET_HPP
#define BOOST_VARIANT_GET_HPP

#include <exception>

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/static_assert.hpp>
#include <boost/throw_exception.hpp>
#include <boost/utility/addressof.hpp>
#include <boost/variant/variant_fwd.hpp>
#include <boost/variant/detail/element_index.hpp>
#include <boost/variant/detail/move.hpp>

#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/type_traits/is_lvalue_reference.hpp>

namespace boost {

#if defined(BOOST_CLANG)
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wweak-vtables"
#endif
class BOOST_SYMBOL_VISIBLE bad_get
: public std::exception
{
public: 

const char * what() const BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE
{
return "boost::bad_get: "
"failed value get using boost::get";
}

};
#if defined(BOOST_CLANG)
#   pragma clang diagnostic pop
#endif



namespace detail { namespace variant {

template <typename T>
struct get_visitor
{
private: 

typedef typename add_pointer<T>::type pointer;
typedef typename add_reference<T>::type reference;

public: 

typedef pointer result_type;

public: 

pointer operator()(reference operand) const BOOST_NOEXCEPT
{
return boost::addressof(operand);
}

template <typename U>
pointer operator()(const U&) const BOOST_NOEXCEPT
{
return static_cast<pointer>(0);
}
};

}} 

#ifndef BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE
#   if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x0551))
#       define BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(t)
#   else
#       if defined(BOOST_NO_NULLPTR)
#           define BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(t)  \
, t* = 0
#       else
#           define BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(t)  \
, t* = nullptr
#       endif
#   endif
#endif

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_pointer<U>::type
relaxed_get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >* operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
) BOOST_NOEXCEPT
{
typedef typename add_pointer<U>::type U_ptr;
if (!operand) return static_cast<U_ptr>(0);

detail::variant::get_visitor<U> v;
return operand->apply_visitor(v);
}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_pointer<const U>::type
relaxed_get(
const boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >* operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
) BOOST_NOEXCEPT
{
typedef typename add_pointer<const U>::type U_ptr;
if (!operand) return static_cast<U_ptr>(0);

detail::variant::get_visitor<const U> v;
return operand->apply_visitor(v);
}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_reference<U>::type
relaxed_get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
typedef typename add_pointer<U>::type U_ptr;
U_ptr result = relaxed_get<U>(boost::addressof(operand));

if (!result)
boost::throw_exception(bad_get());
return *result;
}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_reference<const U>::type
relaxed_get(
const boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
typedef typename add_pointer<const U>::type U_ptr;
U_ptr result = relaxed_get<const U>(boost::addressof(operand));

if (!result)
boost::throw_exception(bad_get());
return *result;
}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES

#if defined(BOOST_MSVC) && (_MSC_VER < 1900) 
#   pragma warning(push)
#   pragma warning(disable: 4172) 
#endif

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
U&&
relaxed_get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >&& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
typedef typename add_pointer<U>::type U_ptr;
U_ptr result = relaxed_get<U>(boost::addressof(operand));

if (!result)
boost::throw_exception(bad_get());
return static_cast<U&&>(*result);
}

#if defined(BOOST_MSVC) && (_MSC_VER < 1900)
#   pragma warning(pop)
#endif

#endif

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_pointer<U>::type
strict_get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >* operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
) BOOST_NOEXCEPT
{
BOOST_STATIC_ASSERT_MSG(
(boost::detail::variant::holds_element<boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >, U >::value),
"boost::variant does not contain specified type U, "
"call to boost::get<U>(boost::variant<T...>*) will always return NULL"
);

return relaxed_get<U>(operand);
}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_pointer<const U>::type
strict_get(
const boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >* operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
) BOOST_NOEXCEPT
{
BOOST_STATIC_ASSERT_MSG(
(boost::detail::variant::holds_element<boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >, const U >::value),
"boost::variant does not contain specified type U, "
"call to boost::get<U>(const boost::variant<T...>*) will always return NULL"
);

return relaxed_get<U>(operand);
}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_reference<U>::type
strict_get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
BOOST_STATIC_ASSERT_MSG(
(boost::detail::variant::holds_element<boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >, U >::value),
"boost::variant does not contain specified type U, "
"call to boost::get<U>(boost::variant<T...>&) will always throw boost::bad_get exception"
);

return relaxed_get<U>(operand);
}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_reference<const U>::type
strict_get(
const boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
BOOST_STATIC_ASSERT_MSG(
(boost::detail::variant::holds_element<boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >, const U >::value),
"boost::variant does not contain specified type U, "
"call to boost::get<U>(const boost::variant<T...>&) will always throw boost::bad_get exception"
);

return relaxed_get<U>(operand);
}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
U&&
strict_get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >&& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
BOOST_STATIC_ASSERT_MSG(
(!boost::is_lvalue_reference<U>::value),
"remove ampersand '&' from template type U in boost::get<U>(boost::variant<T...>&&) "
);

BOOST_STATIC_ASSERT_MSG(
(boost::detail::variant::holds_element<boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >, U >::value),
"boost::variant does not contain specified type U, "
"call to boost::get<U>(const boost::variant<T...>&) will always throw boost::bad_get exception"
);

return relaxed_get<U>(detail::variant::move(operand));
}
#endif


template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_pointer<U>::type
get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >* operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
) BOOST_NOEXCEPT
{
#ifdef BOOST_VARIANT_USE_RELAXED_GET_BY_DEFAULT
return relaxed_get<U>(operand);
#else
return strict_get<U>(operand);
#endif

}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_pointer<const U>::type
get(
const boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >* operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
) BOOST_NOEXCEPT
{
#ifdef BOOST_VARIANT_USE_RELAXED_GET_BY_DEFAULT
return relaxed_get<U>(operand);
#else
return strict_get<U>(operand);
#endif
}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_reference<U>::type
get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
#ifdef BOOST_VARIANT_USE_RELAXED_GET_BY_DEFAULT
return relaxed_get<U>(operand);
#else
return strict_get<U>(operand);
#endif
}

template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
typename add_reference<const U>::type
get(
const boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
#ifdef BOOST_VARIANT_USE_RELAXED_GET_BY_DEFAULT
return relaxed_get<U>(operand);
#else
return strict_get<U>(operand);
#endif
}

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename U, BOOST_VARIANT_ENUM_PARAMS(typename T) >
inline
U&&
get(
boost::variant< BOOST_VARIANT_ENUM_PARAMS(T) >&& operand
BOOST_VARIANT_AUX_GET_EXPLICIT_TEMPLATE_TYPE(U)
)
{
#ifdef BOOST_VARIANT_USE_RELAXED_GET_BY_DEFAULT
return relaxed_get<U>(detail::variant::move(operand));
#else
return strict_get<U>(detail::variant::move(operand));
#endif
}
#endif

} 

#endif 
