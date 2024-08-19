


#ifndef BOOST_CHRONO_DETAIL_NO_WARNING_SIGNED_UNSIGNED_CMP_HPP
#define BOOST_CHRONO_DETAIL_NO_WARNING_SIGNED_UNSIGNED_CMP_HPP


#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#elif defined __SUNPRO_CC
#pragma disable_warn
#elif defined _MSC_VER
#pragma warning(push, 1)
#endif

namespace boost {
namespace chrono {
namespace detail {

template <class T, class U>
bool lt(T t, U u)
{
return t < u;
}

template <class T, class U>
bool gt(T t, U u)
{
return t > u;
}

} 
} 
} 

#if defined __SUNPRO_CC
#pragma enable_warn
#elif defined _MSC_VER
#pragma warning(pop)
#endif

#endif 
