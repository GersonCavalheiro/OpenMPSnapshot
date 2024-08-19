


#ifndef BOOST_STRING_CASE_CONV_DETAIL_HPP
#define BOOST_STRING_CASE_CONV_DETAIL_HPP

#include <boost/algorithm/string/config.hpp>
#include <locale>
#include <functional>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/type_traits/make_unsigned.hpp>

namespace boost {
namespace algorithm {
namespace detail {


#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(push)
#pragma warning(disable:4512) 
#endif

template<typename CharT>
struct to_lowerF
{
typedef CharT argument_type;
typedef CharT result_type;
to_lowerF( const std::locale& Loc ) : m_Loc( &Loc ) {}

CharT operator ()( CharT Ch ) const
{
#if defined(BOOST_BORLANDC) && (BOOST_BORLANDC >= 0x560) && (BOOST_BORLANDC <= 0x564) && !defined(_USE_OLD_RW_STL)
return std::tolower( static_cast<typename boost::make_unsigned <CharT>::type> ( Ch ));
#else
return std::tolower<CharT>( Ch, *m_Loc );
#endif
}
private:
const std::locale* m_Loc;
};

template<typename CharT>
struct to_upperF
{
typedef CharT argument_type;
typedef CharT result_type;
to_upperF( const std::locale& Loc ) : m_Loc( &Loc ) {}

CharT operator ()( CharT Ch ) const
{
#if defined(BOOST_BORLANDC) && (BOOST_BORLANDC >= 0x560) && (BOOST_BORLANDC <= 0x564) && !defined(_USE_OLD_RW_STL)
return std::toupper( static_cast<typename boost::make_unsigned <CharT>::type> ( Ch ));
#else
return std::toupper<CharT>( Ch, *m_Loc );
#endif
}
private:
const std::locale* m_Loc;
};

#if BOOST_WORKAROUND(BOOST_MSVC, >= 1400)
#pragma warning(pop)
#endif


template<typename OutputIteratorT, typename RangeT, typename FunctorT>
OutputIteratorT transform_range_copy(
OutputIteratorT Output,
const RangeT& Input,
FunctorT Functor)
{
return std::transform( 
::boost::begin(Input), 
::boost::end(Input), 
Output,
Functor);
}

template<typename RangeT, typename FunctorT>
void transform_range(
const RangeT& Input,
FunctorT Functor)
{
std::transform( 
::boost::begin(Input), 
::boost::end(Input), 
::boost::begin(Input),
Functor);
}

template<typename SequenceT, typename RangeT, typename FunctorT>
inline SequenceT transform_range_copy( 
const RangeT& Input, 
FunctorT Functor)
{
return SequenceT(
::boost::make_transform_iterator(
::boost::begin(Input),
Functor),
::boost::make_transform_iterator(
::boost::end(Input), 
Functor));
}

} 
} 
} 


#endif  
