#ifndef BOOST_GIL_CONCEPTS_DYNAMIC_STEP_HPP
#define BOOST_GIL_CONCEPTS_DYNAMIC_STEP_HPP

#include <boost/gil/concepts/fwd.hpp>
#include <boost/gil/concepts/concept_check.hpp>

#if defined(BOOST_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunused-local-typedefs"
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

namespace boost { namespace gil {

template <typename T>
struct HasDynamicXStepTypeConcept
{
void constraints()
{
using type = typename dynamic_x_step_type<T>::type;
ignore_unused_variable_warning(type{});
}
};

template <typename T>
struct HasDynamicYStepTypeConcept
{
void constraints()
{
using type = typename dynamic_y_step_type<T>::type;
ignore_unused_variable_warning(type{});
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
