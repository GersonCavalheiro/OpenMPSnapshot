

#ifndef BOOST_POLY_COLLECTION_DETAIL_IS_FINAL_HPP
#define BOOST_POLY_COLLECTION_DETAIL_IS_FINAL_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/type_traits/is_final.hpp>
#include <type_traits>



namespace boost{
namespace poly_collection{
namespace detail{
namespace is_final_fallback{

template<typename T> using is_final=boost::is_final<T>;

struct hook{};

}}}}

namespace std{

template<>
struct is_void< ::boost::poly_collection::detail::is_final_fallback::hook>:
std::false_type
{      
template<typename T>
static constexpr bool is_final_f()
{
using namespace ::boost::poly_collection::detail::is_final_fallback;
return is_final<T>::value;
}
};

} 

namespace boost{

namespace poly_collection{

namespace detail{

template<typename T>
struct is_final:std::integral_constant<
bool,
std::is_void<is_final_fallback::hook>::template is_final_f<T>()
>{};

} 

} 

} 

#endif
