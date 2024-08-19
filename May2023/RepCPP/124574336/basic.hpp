#ifndef BOOST_GIL_CONCEPTS_BASIC_HPP
#define BOOST_GIL_CONCEPTS_BASIC_HPP

#include <boost/config.hpp>

#if defined(BOOST_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunused-local-typedefs"
#pragma clang diagnostic ignored "-Wuninitialized"
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

#include <boost/gil/concepts/concept_check.hpp>

#include <type_traits>
#include <utility> 

namespace boost { namespace gil {

template <typename T>
struct DefaultConstructible
{
void constraints()
{
function_requires<boost::DefaultConstructibleConcept<T>>();
}
};

template <typename T>
struct CopyConstructible
{
void constraints()
{
function_requires<boost::CopyConstructibleConcept<T>>();
}
};

template <typename T>
struct Assignable
{
void constraints()
{
function_requires<boost::AssignableConcept<T>>();
}
};

template <typename T>
struct EqualityComparable
{
void constraints()
{
function_requires<boost::EqualityComparableConcept<T>>();
}
};

template <typename T>
struct Swappable
{
void constraints()
{
using std::swap;
swap(x,y);
}
T x,y;
};

template <typename T>
struct Regular
{
void constraints()
{
gil_function_requires< boost::DefaultConstructibleConcept<T>>();
gil_function_requires< boost::CopyConstructibleConcept<T>>();
gil_function_requires< boost::EqualityComparableConcept<T>>(); 
gil_function_requires< boost::AssignableConcept<T>>();
gil_function_requires< Swappable<T>>();
}
};

template <typename T>
struct Metafunction
{
void constraints()
{
using type = typename T::type;
}
};

template <typename T, typename U>
struct SameType
{
void constraints()
{
static_assert(std::is_same<T, U>::value, "");
}
};

}} 

#if defined(BOOST_CLANG)
#pragma clang diagnostic pop
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic pop
#endif

#endif
