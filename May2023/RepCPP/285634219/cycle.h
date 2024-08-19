












#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif

#define INLINE_ELAPSED(INL) static INL double elapsed(ticks t1, ticks t0) \
{									  \
return (double)t1 - (double)t0;					  \
}



#if defined(HAVE_GETHRTIME) && defined(HAVE_HRTIME_T) && !defined(HAVE_TICK_COUNTER)
typedef hrtime_t ticks;

#define getticks gethrtime

INLINE_ELAPSED(inline)

#define HAVE_TICK_COUNTER
#endif



#if defined(HAVE_READ_REAL_TIME) && defined(HAVE_TIME_BASE_TO_TIME) && !defined(HAVE_TICK_COUNTER)
typedef timebasestruct_t ticks;

static __inline ticks getticks(void)
{
ticks t;
read_real_time(&t, TIMEBASE_SZ);
return t;
}

static __inline double elapsed(ticks t1, ticks t0) 
{
time_base_to_time(&t1, TIMEBASE_SZ);
time_base_to_time(&t0, TIMEBASE_SZ);
return (((double)t1.tb_high - (double)t0.tb_high) * 1.0e9 + 
((double)t1.tb_low - (double)t0.tb_low));
}

#define HAVE_TICK_COUNTER
#endif



#if ((((defined(__GNUC__) && (defined(__powerpc__) || defined(__ppc__))) || (defined(__MWERKS__) && defined(macintosh)))) || (defined(__IBM_GCC_ASM) && (defined(__powerpc__) || defined(__ppc__))))  && !defined(HAVE_TICK_COUNTER)
typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
unsigned int tbl, tbu0, tbu1;

do {
__asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
__asm__ __volatile__ ("mftb %0" : "=r"(tbl));
__asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
} while (tbu0 != tbu1);

return (((unsigned long long)tbu0) << 32) | tbl;
}

INLINE_ELAPSED(__inline__)

#define HAVE_TICK_COUNTER
#endif


#if defined(HAVE_MACH_ABSOLUTE_TIME) && defined(HAVE_MACH_MACH_TIME_H) && !defined(HAVE_TICK_COUNTER)
#include <mach/mach_time.h>
typedef uint64_t ticks;
#define getticks mach_absolute_time
INLINE_ELAPSED(__inline__)
#define HAVE_TICK_COUNTER
#endif



#if (defined(__GNUC__) || defined(__ICC)) && defined(__i386__)  && !defined(HAVE_TICK_COUNTER)
typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
ticks ret;

__asm__ __volatile__("rdtsc": "=A" (ret));

return ret;
}

INLINE_ELAPSED(__inline__)

#define HAVE_TICK_COUNTER
#define TIME_MIN 5000.0   
#endif


#if _MSC_VER >= 1200 && _M_IX86 >= 500 && !defined(HAVE_TICK_COUNTER)
#include <windows.h>
typedef LARGE_INTEGER ticks;
#define RDTSC __asm __emit 0fh __asm __emit 031h 

static __inline ticks getticks(void)
{
ticks retval;

__asm {
RDTSC
mov retval.HighPart, edx
mov retval.LowPart, eax
}
return retval;
}

static __inline double elapsed(ticks t1, ticks t0)
{  
return (double)t1.QuadPart - (double)t0.QuadPart;
}  

#define HAVE_TICK_COUNTER
#define TIME_MIN 5000.0   
#endif



#if (defined(__GNUC__) || defined(__ICC) || defined(__SUNPRO_C)) && defined(__x86_64__)  && !defined(HAVE_TICK_COUNTER)
typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
unsigned a, d; 
asm volatile("rdtsc" : "=a" (a), "=d" (d)); 
return ((ticks)a) | (((ticks)d) << 32); 
}

INLINE_ELAPSED(__inline__)

#define HAVE_TICK_COUNTER
#endif


#if defined(__PGI) && defined(__x86_64__) && !defined(HAVE_TICK_COUNTER) 
typedef unsigned long long ticks;
static ticks getticks(void)
{
asm(" rdtsc; shl    $0x20,%rdx; mov    %eax,%eax; or     %rdx,%rax;    ");
}
INLINE_ELAPSED(__inline__)
#define HAVE_TICK_COUNTER
#endif


#if _MSC_VER >= 1400 && (defined(_M_AMD64) || defined(_M_X64)) && !defined(HAVE_TICK_COUNTER)

#include <intrin.h>
#pragma intrinsic(__rdtsc)
typedef unsigned __int64 ticks;
#define getticks __rdtsc
INLINE_ELAPSED(__inline)

#define HAVE_TICK_COUNTER
#endif





#if (defined(__EDG_VERSION) || defined(__ECC)) && defined(__ia64__) && !defined(HAVE_TICK_COUNTER)
typedef unsigned long ticks;
#include <ia64intrin.h>

static __inline__ ticks getticks(void)
{
return __getReg(_IA64_REG_AR_ITC);
}

INLINE_ELAPSED(__inline__)

#define HAVE_TICK_COUNTER
#endif


#if defined(__GNUC__) && defined(__ia64__) && !defined(HAVE_TICK_COUNTER)
typedef unsigned long ticks;

static __inline__ ticks getticks(void)
{
ticks ret;

__asm__ __volatile__ ("mov %0=ar.itc" : "=r"(ret));
return ret;
}

INLINE_ELAPSED(__inline__)

#define HAVE_TICK_COUNTER
#endif


#if defined(__hpux) && defined(__ia64) && !defined(HAVE_TICK_COUNTER)
#include <machine/sys/inline.h>
typedef unsigned long ticks;

static inline ticks getticks(void)
{
ticks ret;

ret = _Asm_mov_from_ar (_AREG_ITC);
return ret;
}

INLINE_ELAPSED(inline)

#define HAVE_TICK_COUNTER
#endif


#if defined(_MSC_VER) && defined(_M_IA64) && !defined(HAVE_TICK_COUNTER)
typedef unsigned __int64 ticks;

#  ifdef __cplusplus
extern "C"
#  endif
ticks __getReg(int whichReg);
#pragma intrinsic(__getReg)

static __inline ticks getticks(void)
{
volatile ticks temp;
temp = __getReg(3116);
return temp;
}

INLINE_ELAPSED(inline)

#define HAVE_TICK_COUNTER
#endif



#if defined(__hppa__) || defined(__hppa) && !defined(HAVE_TICK_COUNTER)
typedef unsigned long ticks;

#  ifdef __GNUC__
static __inline__ ticks getticks(void)
{
ticks ret;

__asm__ __volatile__("mfctl 16, %0": "=r" (ret));

return ret;
}
#  else
#  include <machine/inline.h>
static inline unsigned long getticks(void)
{
register ticks ret;
_MFCTL(16, ret);
return ret;
}
#  endif

INLINE_ELAPSED(inline)

#define HAVE_TICK_COUNTER
#endif



#if defined(__GNUC__) && defined(__s390__) && !defined(HAVE_TICK_COUNTER)
typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
ticks cycles;
__asm__("stck 0(%0)" : : "a" (&(cycles)) : "memory", "cc");
return cycles;
}

INLINE_ELAPSED(__inline__)

#define HAVE_TICK_COUNTER
#endif

#if defined(__GNUC__) && defined(__alpha__) && !defined(HAVE_TICK_COUNTER)

typedef unsigned int ticks;

static __inline__ ticks getticks(void)
{
unsigned long cc;
__asm__ __volatile__ ("rpcc %0" : "=r"(cc));
return (cc & 0xFFFFFFFF);
}

INLINE_ELAPSED(__inline__)

#define HAVE_TICK_COUNTER
#endif


#if defined(__GNUC__) && defined(__sparc_v9__) && !defined(HAVE_TICK_COUNTER)
typedef unsigned long ticks;

static __inline__ ticks getticks(void)
{
ticks ret;
__asm__ __volatile__("rd %%tick, %0" : "=r" (ret));
return ret;
}

INLINE_ELAPSED(__inline__)

#define HAVE_TICK_COUNTER
#endif


#if (defined(__DECC) || defined(__DECCXX)) && defined(__alpha) && defined(HAVE_C_ASM_H) && !defined(HAVE_TICK_COUNTER)
#  include <c_asm.h>
typedef unsigned int ticks;

static __inline ticks getticks(void)
{
unsigned long cc;
cc = asm("rpcc %v0");
return (cc & 0xFFFFFFFF);
}

INLINE_ELAPSED(__inline)

#define HAVE_TICK_COUNTER
#endif


#if defined(HAVE_CLOCK_GETTIME) && defined(CLOCK_SGI_CYCLE) && !defined(HAVE_TICK_COUNTER)
typedef struct timespec ticks;

static inline ticks getticks(void)
{
struct timespec t;
clock_gettime(CLOCK_SGI_CYCLE, &t);
return t;
}

static inline double elapsed(ticks t1, ticks t0)
{
return ((double)t1.tv_sec - (double)t0.tv_sec) * 1.0E9 +
((double)t1.tv_nsec - (double)t0.tv_nsec);
}
#define HAVE_TICK_COUNTER
#endif



#if defined(HAVE__RTC) && !defined(HAVE_TICK_COUNTER)
#ifdef HAVE_INTRINSICS_H
#  include <intrinsics.h>
#endif

typedef long long ticks;

#define getticks _rtc

INLINE_ELAPSED(inline)

#define HAVE_TICK_COUNTER
#endif



#if HAVE_MIPS_ZBUS_TIMER
#if defined(__mips__) && !defined(HAVE_TICK_COUNTER)
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

typedef uint64_t ticks;

static inline ticks getticks(void)
{
static uint64_t* addr = 0;

if (addr == 0)
{
uint32_t rq_addr = 0x10030000;
int fd;
int pgsize;

pgsize = getpagesize();
fd = open ("/dev/mem", O_RDONLY | O_SYNC, 0);
if (fd < 0) {
perror("open");
return NULL;
}
addr = mmap(0, pgsize, PROT_READ, MAP_SHARED, fd, rq_addr);
close(fd);
if (addr == (uint64_t *)-1) {
perror("mmap");
return NULL;
}
}

return *addr;
}

INLINE_ELAPSED(inline)

#define HAVE_TICK_COUNTER
#endif
#endif 

