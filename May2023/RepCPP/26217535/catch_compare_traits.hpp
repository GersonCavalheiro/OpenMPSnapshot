

#ifndef CATCH_COMPARE_TRAITS_HPP_INCLUDED
#define CATCH_COMPARE_TRAITS_HPP_INCLUDED

#include <catch2/internal/catch_void_type.hpp>

#include <type_traits>

namespace Catch {
namespace Detail {

#if defined( __GNUC__ ) && !defined( __clang__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wbool-compare"
#    pragma GCC diagnostic ignored "-Wextra"
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#if defined( __clang__ )
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wfloat-equal"
#endif

#define CATCH_DEFINE_COMPARABLE_TRAIT( id, op )                               \
template <typename, typename, typename = void>                            \
struct is_##id##_comparable : std::false_type {};                         \
template <typename T, typename U>                                         \
struct is_##id##_comparable<                                              \
T,                                                                    \
U,                                                                    \
void_t<decltype( std::declval<T>() op std::declval<U>() )>>           \
: std::true_type {};                                                  \
template <typename, typename = void>                                      \
struct is_##id##_0_comparable : std::false_type {};                       \
template <typename T>                                                     \
struct is_##id##_0_comparable<T,                                          \
void_t<decltype( std::declval<T>() op 0 )>> \
: std::true_type {};

CATCH_DEFINE_COMPARABLE_TRAIT( lt, < )
CATCH_DEFINE_COMPARABLE_TRAIT( le, <= )
CATCH_DEFINE_COMPARABLE_TRAIT( gt, > )
CATCH_DEFINE_COMPARABLE_TRAIT( ge, >= )
CATCH_DEFINE_COMPARABLE_TRAIT( eq, == )
CATCH_DEFINE_COMPARABLE_TRAIT( ne, != )

#undef CATCH_DEFINE_COMPARABLE_TRAIT

#if defined( __GNUC__ ) && !defined( __clang__ )
#    pragma GCC diagnostic pop
#endif
#if defined( __clang__ )
#    pragma clang diagnostic pop
#endif


} 
} 

#endif 
