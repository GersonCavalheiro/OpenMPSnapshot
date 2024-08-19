#ifndef BOOST_GIL_CONCEPTS_PIXEL_ITERATOR_HPP
#define BOOST_GIL_CONCEPTS_PIXEL_ITERATOR_HPP

#include <boost/gil/concepts/channel.hpp>
#include <boost/gil/concepts/color.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>
#include <boost/gil/concepts/pixel.hpp>
#include <boost/gil/concepts/pixel_based.hpp>

#include <boost/iterator/iterator_concepts.hpp>

#include <cstddef>
#include <iterator>
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

template <typename It> struct const_iterator_type;
template <typename It> struct iterator_is_mutable;
template <typename It> struct is_iterator_adaptor;
template <typename It, typename NewBaseIt> struct iterator_adaptor_rebind;
template <typename It> struct iterator_adaptor_get_base;

namespace detail {

template <class TT>
struct ForwardIteratorIsMutableConcept
{
void constraints()
{
auto const tmp = *i;
*i++ = tmp; 
}
TT i;
};

template <class TT>
struct BidirectionalIteratorIsMutableConcept
{
void constraints()
{
gil_function_requires< ForwardIteratorIsMutableConcept<TT>>();
auto const tmp = *i;
*i-- = tmp; 
}
TT i;
};

template <class TT>
struct RandomAccessIteratorIsMutableConcept
{
void constraints()
{
gil_function_requires<BidirectionalIteratorIsMutableConcept<TT>>();

typename std::iterator_traits<TT>::difference_type n = 0;
ignore_unused_variable_warning(n);
i[n] = *i; 
}
TT i;
};

template <typename Iterator>
struct RandomAccessIteratorIsMemoryBasedConcept
{
void constraints()
{
std::ptrdiff_t bs = memunit_step(it);
ignore_unused_variable_warning(bs);

it = memunit_advanced(it, 3);
std::ptrdiff_t bd = memunit_distance(it, it);
ignore_unused_variable_warning(bd);

memunit_advance(it,3);
}
Iterator it;
};

template <typename Iterator>
struct PixelIteratorIsMutableConcept
{
void constraints()
{
gil_function_requires<detail::RandomAccessIteratorIsMutableConcept<Iterator>>();

using ref_t = typename std::remove_reference
<
typename std::iterator_traits<Iterator>::reference
>::type;
using channel_t = typename element_type<ref_t>::type;
gil_function_requires<detail::ChannelIsMutableConcept<channel_t>>();
}
};

} 

template <typename T>
struct HasTransposedTypeConcept
{
void constraints()
{
using type = typename transposed_type<T>::type;
ignore_unused_variable_warning(type{});
}
};


template <typename Iterator>
struct PixelIteratorConcept
{
void constraints()
{
gil_function_requires<boost_concepts::RandomAccessTraversalConcept<Iterator>>();
gil_function_requires<PixelBasedConcept<Iterator>>();

using value_type = typename std::iterator_traits<Iterator>::value_type;
gil_function_requires<PixelValueConcept<value_type>>();

using const_t = typename const_iterator_type<Iterator>::type;
static bool const is_mutable = iterator_is_mutable<Iterator>::value;
ignore_unused_variable_warning(is_mutable);

const_t const_it(it);
ignore_unused_variable_warning(const_it);

check_base(typename is_iterator_adaptor<Iterator>::type());
}

void check_base(std::false_type) {}

void check_base(std::true_type)
{
using base_t = typename iterator_adaptor_get_base<Iterator>::type;
gil_function_requires<PixelIteratorConcept<base_t>>();
}

Iterator it;
};

template <typename Iterator>
struct MutablePixelIteratorConcept
{
void constraints()
{
gil_function_requires<PixelIteratorConcept<Iterator>>();
gil_function_requires<detail::PixelIteratorIsMutableConcept<Iterator>>();
}
};


template <typename Iterator>
struct MemoryBasedIteratorConcept
{
void constraints()
{
gil_function_requires<boost_concepts::RandomAccessTraversalConcept<Iterator>>();
gil_function_requires<detail::RandomAccessIteratorIsMemoryBasedConcept<Iterator>>();
}
};

template <typename Iterator>
struct StepIteratorConcept
{
void constraints()
{
gil_function_requires<boost_concepts::ForwardTraversalConcept<Iterator>>();
it.set_step(0);
}
Iterator it;
};


template <typename Iterator>
struct MutableStepIteratorConcept
{
void constraints()
{
gil_function_requires<StepIteratorConcept<Iterator>>();
gil_function_requires<detail::ForwardIteratorIsMutableConcept<Iterator>>();
}
};


template <typename Iterator>
struct IteratorAdaptorConcept
{
void constraints()
{
gil_function_requires<boost_concepts::ForwardTraversalConcept<Iterator>>();

using base_t = typename iterator_adaptor_get_base<Iterator>::type;
gil_function_requires<boost_concepts::ForwardTraversalConcept<base_t>>();

static_assert(is_iterator_adaptor<Iterator>::value, "");
using rebind_t = typename iterator_adaptor_rebind<Iterator, void*>::type;

base_t base = it.base();
ignore_unused_variable_warning(base);
}
Iterator it;
};

template <typename Iterator>
struct MutableIteratorAdaptorConcept
{
void constraints()
{
gil_function_requires<IteratorAdaptorConcept<Iterator>>();
gil_function_requires<detail::ForwardIteratorIsMutableConcept<Iterator>>();
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
