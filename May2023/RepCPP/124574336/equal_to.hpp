
#if !defined(FUSION_EQUAL_TO_05052005_0431)
#define FUSION_EQUAL_TO_05052005_0431

#include <boost/fusion/support/config.hpp>
#include <boost/fusion/sequence/intrinsic/begin.hpp>
#include <boost/fusion/sequence/intrinsic/end.hpp>
#include <boost/fusion/sequence/intrinsic/size.hpp>
#include <boost/fusion/sequence/comparison/detail/equal_to.hpp>
#include <boost/fusion/sequence/comparison/enable_comparison.hpp>
#include <boost/config.hpp>

#if defined (BOOST_MSVC)
#  pragma warning(push)
#  pragma warning (disable: 4100) 
#endif

namespace boost { namespace fusion
{
template <typename Seq1, typename Seq2>
BOOST_CONSTEXPR BOOST_FUSION_GPU_ENABLED
inline bool
equal_to(Seq1 const& a, Seq2 const& b)
{
return result_of::size<Seq1>::value == result_of::size<Seq2>::value
&& detail::sequence_equal_to<
Seq1 const, Seq2 const
, result_of::size<Seq1>::value == result_of::size<Seq2>::value>::
call(fusion::begin(a), fusion::begin(b));
}

namespace operators
{
template <typename Seq1, typename Seq2>
BOOST_CONSTEXPR BOOST_FUSION_GPU_ENABLED
inline typename
boost::enable_if<
traits::enable_equality<Seq1, Seq2>
, bool
>::type
operator==(Seq1 const& a, Seq2 const& b)
{
return fusion::equal_to(a, b);
}
}
using operators::operator==;
}}

#if defined (BOOST_MSVC)
#  pragma warning(pop)
#endif

#endif
