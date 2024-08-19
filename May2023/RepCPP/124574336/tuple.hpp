



#ifndef BOOST_TUPLE_HPP
#define BOOST_TUPLE_HPP

#if defined(__sgi) && defined(_COMPILER_VERSION) && _COMPILER_VERSION <= 730
namespace boost { namespace python { class tuple; }}
#endif

#include <boost/config.hpp>
#include <boost/static_assert.hpp>

#include <boost/ref.hpp>
#include <boost/tuple/detail/tuple_basic.hpp>


namespace boost {

using tuples::tuple;
using tuples::make_tuple;
using tuples::tie;
#if !defined(BOOST_NO_USING_TEMPLATE)
using tuples::get;
#else
template<int N, class HT, class TT>
inline typename tuples::access_traits<
typename tuples::element<N, tuples::cons<HT, TT> >::type
>::non_const_type
get(tuples::cons<HT, TT>& c) {
return tuples::get<N,HT,TT>(c);
}
template<int N, class HT, class TT>
inline typename tuples::access_traits<
typename tuples::element<N, tuples::cons<HT, TT> >::type
>::const_type
get(const tuples::cons<HT, TT>& c) {
return tuples::get<N,HT,TT>(c);
}

#endif 

} 

#if !defined(BOOST_NO_CXX11_HDR_TUPLE)

#include <tuple>
#include <cstddef>

namespace std
{

#if defined(BOOST_CLANG)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wmismatched-tags"
#endif


template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
class tuple_size< boost::tuples::tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> >:
public boost::tuples::length< boost::tuples::tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> >
{
};

template<class H, class T> class tuple_size< boost::tuples::cons<H, T> >:
public boost::tuples::length< boost::tuples::cons<H, T> >
{
};

template<> class tuple_size< boost::tuples::null_type >:
public boost::tuples::length< boost::tuples::null_type >
{
};


template<std::size_t I, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
class tuple_element< I, boost::tuples::tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> >:
public boost::tuples::element< I, boost::tuples::tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> >
{
};

template<std::size_t I, class H, class T> class tuple_element< I, boost::tuples::cons<H, T> >:
public boost::tuples::element< I, boost::tuples::cons<H, T> >
{
};

#if defined(BOOST_CLANG)
# pragma clang diagnostic pop
#endif

} 

#endif 

#endif 
