#ifndef BOOST_GIL_CONCEPTS_PIXEL_DEREFERENCE_HPP
#define BOOST_GIL_CONCEPTS_PIXEL_DEREFERENCE_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>
#include <boost/gil/concepts/pixel.hpp>
#include <boost/gil/concepts/detail/type_traits.hpp>

#include <boost/concept_check.hpp>

#include <cstddef>
#include <type_traits>

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

template <typename D>
struct PixelDereferenceAdaptorConcept
{
void constraints()
{
gil_function_requires
<
boost::UnaryFunctionConcept
<
D,
typename detail::remove_const_and_reference<typename D::result_type>::type,
typename D::argument_type
>
>();
gil_function_requires<boost::DefaultConstructibleConcept<D>>();
gil_function_requires<boost::CopyConstructibleConcept<D>>();
gil_function_requires<boost::AssignableConcept<D>>();

gil_function_requires<PixelConcept
<
typename detail::remove_const_and_reference<typename D::result_type>::type
>>();

using const_t = typename D::const_t;
gil_function_requires<PixelDereferenceAdaptorConcept<const_t>>();

using value_type = typename D::value_type;
gil_function_requires<PixelValueConcept<value_type>>();

using reference = typename D::reference; 
using const_reference = typename D::const_reference; 

bool const is_mutable = D::is_mutable;
ignore_unused_variable_warning(is_mutable);
}
D d;
};

template <typename P>
struct PixelDereferenceAdaptorArchetype
{
using argument_type = P;
using result_type = P;
using const_t = PixelDereferenceAdaptorArchetype;
using value_type = typename std::remove_reference<P>::type;
using reference = typename std::add_lvalue_reference<P>::type;
using const_reference = reference;

static const bool is_mutable = false;
P operator()(P) const { throw; }
};

}} 

#if defined(BOOST_CLANG)
#pragma clang diagnostic pop
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic pop
#endif

#endif
