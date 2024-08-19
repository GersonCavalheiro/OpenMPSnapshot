

#pragma once

#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC) || (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_CLANG)
#include <stdint.h>
#endif

namespace hydra_thrust
{
namespace detail
{

#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC)

#if (_MSC_VER < 1300)
typedef signed   char     int8_t;
typedef signed   short    int16_t;
typedef signed   int      int32_t;
typedef unsigned char     uint8_t;
typedef unsigned short    uint16_t;
typedef unsigned int      uint32_t;
#else
typedef signed   __int8   int8_t;
typedef signed   __int16  int16_t;
typedef signed   __int32  int32_t;
typedef unsigned __int8   uint8_t;
typedef unsigned __int16  uint16_t;
typedef unsigned __int32  uint32_t;
#endif
typedef signed   __int64     int64_t;
typedef unsigned __int64     uint64_t;

#else

typedef ::int8_t   int8_t;
typedef ::int16_t  int16_t;
typedef ::int32_t  int32_t;
typedef ::int64_t  int64_t;
typedef ::uint8_t  uint8_t;
typedef ::uint16_t uint16_t;
typedef ::uint32_t uint32_t;
typedef ::uint64_t uint64_t;

#endif


template<int word_size = sizeof(void*)> struct divine_intptr_t;
template<int word_size = sizeof(void*)> struct divine_uintptr_t;

template<>  struct divine_intptr_t<4>  {  typedef hydra_thrust::detail::int32_t  type; };
template<>  struct divine_uintptr_t<4> {  typedef hydra_thrust::detail::uint32_t type; };

template<>  struct divine_intptr_t<8>  { typedef hydra_thrust::detail::int64_t  type; };
template<>  struct divine_uintptr_t<8> { typedef hydra_thrust::detail::uint64_t type; };

typedef divine_intptr_t<>::type   intptr_t;
typedef divine_uintptr_t<>::type  uintptr_t;

} 
} 

