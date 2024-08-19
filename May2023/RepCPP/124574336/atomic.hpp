
#ifndef BOOST_INTERPROCESS_DETAIL_ATOMIC_HPP
#define BOOST_INTERPROCESS_DETAIL_ATOMIC_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/cstdint.hpp>

namespace boost{
namespace interprocess{
namespace ipcdetail{

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem);

inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem);

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val);

inline boost::uint32_t atomic_cas32
(volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp);

}  
}  
}  

#if defined (BOOST_INTERPROCESS_WINDOWS)

#include <boost/interprocess/detail/win32_api.hpp>

#if defined( _MSC_VER )
extern "C" void _ReadWriteBarrier(void);
#pragma intrinsic(_ReadWriteBarrier)
#define BOOST_INTERPROCESS_READ_WRITE_BARRIER _ReadWriteBarrier()
#elif defined(__GNUC__)
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100
#define BOOST_INTERPROCESS_READ_WRITE_BARRIER __sync_synchronize()
#else
#define BOOST_INTERPROCESS_READ_WRITE_BARRIER __asm__ __volatile__("" : : : "memory")
#endif
#endif

namespace boost{
namespace interprocess{
namespace ipcdetail{

inline boost::uint32_t atomic_dec32(volatile boost::uint32_t *mem)
{  return winapi::interlocked_decrement(reinterpret_cast<volatile long*>(mem)) + 1;  }

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem)
{  return winapi::interlocked_increment(reinterpret_cast<volatile long*>(mem))-1;  }

inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem)
{
const boost::uint32_t val = *mem;
BOOST_INTERPROCESS_READ_WRITE_BARRIER;
return val;
}

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val)
{  winapi::interlocked_exchange(reinterpret_cast<volatile long*>(mem), val);  }

inline boost::uint32_t atomic_cas32
(volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp)
{  return winapi::interlocked_compare_exchange(reinterpret_cast<volatile long*>(mem), with, cmp);  }

}  
}  
}  

#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__)) && !defined(_CRAYC)

namespace boost {
namespace interprocess {
namespace ipcdetail{

inline boost::uint32_t atomic_cas32
(volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp)
{
boost::uint32_t prev = cmp;
__asm__ __volatile__ ( "lock\n\t"
"cmpxchg %2,%0"
: "+m"(*mem), "+a"(prev)
: "r"(with)
: "cc");

return prev;
}

inline boost::uint32_t atomic_add32
(volatile boost::uint32_t *mem, boost::uint32_t val)
{
int r;

asm volatile
(
"lock\n\t"
"xadd %1, %0":
"+m"( *mem ), "=r"( r ): 
"1"( val ): 
"memory", "cc" 
);

return r;
}

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem)
{  return atomic_add32(mem, 1);  }

inline boost::uint32_t atomic_dec32(volatile boost::uint32_t *mem)
{  return atomic_add32(mem, (boost::uint32_t)-1);  }

inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem)
{
const boost::uint32_t val = *mem;
__asm__ __volatile__ ( "" ::: "memory" );
return val;
}

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val)
{
__asm__ __volatile__
(
"xchgl %0, %1"
: "+r" (val), "+m" (*mem)
:: "memory"
);
}

}  
}  
}  

#elif defined(__GNUC__) && (defined(__PPC__) || defined(__ppc__))

namespace boost {
namespace interprocess {
namespace ipcdetail{

inline boost::uint32_t atomic_add32(volatile boost::uint32_t *mem, boost::uint32_t val)
{
boost::uint32_t prev, temp;

asm volatile ("1:\n\t"
"lwarx  %0,0,%2\n\t"
"add    %1,%0,%3\n\t"
"stwcx. %1,0,%2\n\t"
"bne-   1b"
: "=&r" (prev), "=&r" (temp)
: "b" (mem), "r" (val)
: "cc", "memory");
return prev;
}

inline boost::uint32_t atomic_cas32
(volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp)
{
boost::uint32_t prev;

asm volatile ("1:\n\t"
"lwarx  %0,0,%1\n\t"
"cmpw   %0,%3\n\t"
"bne-   2f\n\t"
"stwcx. %2,0,%1\n\t"
"bne-   1b\n\t"
"2:"
: "=&r"(prev)
: "b" (mem), "r" (with), "r" (cmp)
: "cc", "memory");
return prev;
}

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem)
{  return atomic_add32(mem, 1);  }

inline boost::uint32_t atomic_dec32(volatile boost::uint32_t *mem)
{  return atomic_add32(mem, boost::uint32_t(-1u));  }

inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem)
{
const boost::uint32_t val = *mem;
__asm__ __volatile__ ( "" ::: "memory" );
return val;
}

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val)
{  *mem = val; }

}  
}  
}  

#elif (defined(sun) || defined(__sun))

#include <atomic.h>

namespace boost{
namespace interprocess{
namespace ipcdetail{

inline boost::uint32_t atomic_add32(volatile boost::uint32_t *mem, boost::uint32_t val)
{   return atomic_add_32_nv(reinterpret_cast<volatile ::uint32_t*>(mem), (int32_t)val) - val;   }

inline boost::uint32_t atomic_cas32
(volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp)
{  return atomic_cas_32(reinterpret_cast<volatile ::uint32_t*>(mem), cmp, with);  }

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem)
{  return atomic_add_32_nv(reinterpret_cast<volatile ::uint32_t*>(mem), 1) - 1; }

inline boost::uint32_t atomic_dec32(volatile boost::uint32_t *mem)
{  return atomic_add_32_nv(reinterpret_cast<volatile ::uint32_t*>(mem), (boost::uint32_t)-1) + 1; }

inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem)
{  return *mem;   }

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val)
{  *mem = val; }

}  
}  
}  

#elif defined(__osf__) && defined(__DECCXX)

#include <machine/builtins.h>
#include <c_asm.h>

namespace boost{
namespace interprocess{
namespace ipcdetail{

inline boost::uint32_t atomic_dec32(volatile boost::uint32_t *mem)
{  boost::uint32_t old_val = __ATOMIC_DECREMENT_LONG(mem); __MB(); return old_val; }

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem)
{  __MB(); return __ATOMIC_INCREMENT_LONG(mem); }


inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem)
{  boost::uint32_t old_val = *mem; __MB(); return old_val;  }

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val)
{  __MB(); *mem = val; }

inline boost::uint32_t atomic_cas32(
volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp)
{

return asm(
"10: ldl_l %v0,(%a0) ;"    
"    cmpeq %v0,%a2,%t0 ;"  
"    beq %t0,20f ;"        
"    mb ;"                 
"    mov %a1,%t0 ;"        
"    stl_c %t0,(%a0) ;"    
"    beq %t0,10b ;"        
"20: ",
mem, with, cmp);
}

}  
}  
}  

#elif defined(__IBMCPP__) && (__IBMCPP__ >= 800) && defined(_AIX)

#include <builtins.h>

namespace boost {
namespace interprocess {
namespace ipcdetail{


inline boost::uint32_t lwarxu(volatile boost::uint32_t *mem)
{
return static_cast<boost::uint32_t>(__lwarx(reinterpret_cast<volatile int*>(mem)));
}

inline bool stwcxu(volatile boost::uint32_t* mem, boost::uint32_t val)
{
return (__stwcx(reinterpret_cast<volatile int*>(mem), static_cast<int>(val)) != 0);
}

inline boost::uint32_t atomic_add32
(volatile boost::uint32_t *mem, boost::uint32_t val)
{
boost::uint32_t oldValue;
do
{
oldValue = lwarxu(mem);
}while (!stwcxu(mem, oldValue+val));
return oldValue;
}

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem)
{  return atomic_add32(mem, 1);  }

inline boost::uint32_t atomic_dec32(volatile boost::uint32_t *mem)
{  return atomic_add32(mem, (boost::uint32_t)-1);   }

inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem)
{  return *mem;   }

inline boost::uint32_t atomic_cas32
(volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp)
{
boost::uint32_t oldValue;
boost::uint32_t valueToStore;
do
{
oldValue = lwarxu(mem);
} while (!stwcxu(mem, (oldValue == with) ? cmp : oldValue));

return oldValue;
}

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val)
{  *mem = val; }

}  
}  
}  

#elif defined(__GNUC__) && ( __GNUC__ * 100 + __GNUC_MINOR__ >= 401 )

namespace boost {
namespace interprocess {
namespace ipcdetail{

inline boost::uint32_t atomic_add32
(volatile boost::uint32_t *mem, boost::uint32_t val)
{  return __sync_fetch_and_add(const_cast<boost::uint32_t *>(mem), val);   }

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem)
{  return atomic_add32(mem, 1);  }

inline boost::uint32_t atomic_dec32(volatile boost::uint32_t *mem)
{  return atomic_add32(mem, (boost::uint32_t)-1);   }

inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem)
{  boost::uint32_t old_val = *mem; __sync_synchronize(); return old_val;  }

inline boost::uint32_t atomic_cas32
(volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp)
{  return __sync_val_compare_and_swap(const_cast<boost::uint32_t *>(mem), cmp, with);   }

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val)
{  __sync_synchronize(); *mem = val;  }

}  
}  
}  
#elif defined(__VXWORKS__)

#include <vxAtomicLib.h>
#define vx_atomic_cast(_i)   (reinterpret_cast< ::atomic32_t *>( const_cast<boost::uint32_t *>(_i)))

namespace boost {
namespace interprocess {
namespace ipcdetail{

inline boost::uint32_t atomic_add32
(volatile boost::uint32_t *mem, boost::uint32_t val)
{  return ::vxAtomic32Add( vx_atomic_cast(mem), val);   }

inline boost::uint32_t atomic_inc32(volatile boost::uint32_t *mem)
{  return ::vxAtomic32Inc( vx_atomic_cast(mem) );  }

inline boost::uint32_t atomic_dec32(volatile boost::uint32_t *mem)
{  return ::vxAtomic32Dec( vx_atomic_cast(mem) );   }

inline boost::uint32_t atomic_read32(volatile boost::uint32_t *mem)
{  return ::vxAtomic32Get( vx_atomic_cast(mem) );  }

inline boost::uint32_t atomic_cas32
(volatile boost::uint32_t *mem, boost::uint32_t with, boost::uint32_t cmp)
{  return ::vxAtomic32Cas( vx_atomic_cast(mem), cmp, with);  }

inline void atomic_write32(volatile boost::uint32_t *mem, boost::uint32_t val)
{  ::vxAtomic32Set( vx_atomic_cast(mem), val);  }


}  
}  
}  

#else

#error No atomic operations implemented for this platform, sorry!

#endif

namespace boost{
namespace interprocess{
namespace ipcdetail{

inline bool atomic_add_unless32
(volatile boost::uint32_t *mem, boost::uint32_t value, boost::uint32_t unless_this)
{
boost::uint32_t old, c(atomic_read32(mem));
while(c != unless_this && (old = atomic_cas32(mem, c + value, c)) != c){
c = old;
}
return c != unless_this;
}

}  
}  
}  


#include <boost/interprocess/detail/config_end.hpp>

#endif   
