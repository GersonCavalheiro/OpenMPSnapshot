#ifndef HEADER_CURL_SETUP_ONCE_H
#define HEADER_CURL_SETUP_ONCE_H





#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef NEED_MALLOC_H
#include <malloc.h>
#endif

#ifdef NEED_MEMORY_H
#include <memory.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#ifdef TIME_WITH_SYS_TIME
#include <time.h>
#endif
#else
#ifdef HAVE_TIME_H
#include <time.h>
#endif
#endif

#ifdef WIN32
#include <io.h>
#include <fcntl.h>
#endif

#if defined(HAVE_STDBOOL_H) && defined(HAVE_BOOL_T)
#include <stdbool.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef __hpux
#  if !defined(_XOPEN_SOURCE_EXTENDED) || defined(_KERNEL)
#    ifdef _APP32_64BIT_OFF_T
#      define OLD_APP32_64BIT_OFF_T _APP32_64BIT_OFF_T
#      undef _APP32_64BIT_OFF_T
#    else
#      undef OLD_APP32_64BIT_OFF_T
#    endif
#  endif
#endif

#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif

#ifdef __hpux
#  if !defined(_XOPEN_SOURCE_EXTENDED) || defined(_KERNEL)
#    ifdef OLD_APP32_64BIT_OFF_T
#      define _APP32_64BIT_OFF_T OLD_APP32_64BIT_OFF_T
#      undef OLD_APP32_64BIT_OFF_T
#    endif
#  endif
#endif



#ifndef HAVE_STRUCT_TIMEVAL
struct timeval {
long tv_sec;
long tv_usec;
};
#endif




#ifdef HAVE_MSG_NOSIGNAL
#define SEND_4TH_ARG MSG_NOSIGNAL
#else
#define SEND_4TH_ARG 0
#endif


#if defined(__minix)

#define sread(x,y,z) (ssize_t)read((RECV_TYPE_ARG1)(x), \
(RECV_TYPE_ARG2)(y), \
(RECV_TYPE_ARG3)(z))

#elif defined(HAVE_RECV)


#if !defined(RECV_TYPE_ARG1) || \
!defined(RECV_TYPE_ARG2) || \
!defined(RECV_TYPE_ARG3) || \
!defined(RECV_TYPE_ARG4) || \
!defined(RECV_TYPE_RETV)

Error Missing_definition_of_return_and_arguments_types_of_recv

#else
#define sread(x,y,z) (ssize_t)recv((RECV_TYPE_ARG1)(x), \
(RECV_TYPE_ARG2)(y), \
(RECV_TYPE_ARG3)(z), \
(RECV_TYPE_ARG4)(0))
#endif
#else 
#ifndef sread

Error Missing_definition_of_macro_sread

#endif
#endif 


#if defined(__minix)

#define swrite(x,y,z) (ssize_t)write((SEND_TYPE_ARG1)(x), \
(SEND_TYPE_ARG2)(y), \
(SEND_TYPE_ARG3)(z))

#elif defined(HAVE_SEND)
#if !defined(SEND_TYPE_ARG1) || \
!defined(SEND_QUAL_ARG2) || \
!defined(SEND_TYPE_ARG2) || \
!defined(SEND_TYPE_ARG3) || \
!defined(SEND_TYPE_ARG4) || \
!defined(SEND_TYPE_RETV)

Error Missing_definition_of_return_and_arguments_types_of_send

#else
#define swrite(x,y,z) (ssize_t)send((SEND_TYPE_ARG1)(x), \
(SEND_QUAL_ARG2 SEND_TYPE_ARG2)(y), \
(SEND_TYPE_ARG3)(z), \
(SEND_TYPE_ARG4)(SEND_4TH_ARG))
#endif
#else 
#ifndef swrite

Error Missing_definition_of_macro_swrite

#endif
#endif 


#if 0
#if defined(HAVE_RECVFROM)

#if !defined(RECVFROM_TYPE_ARG1) || \
!defined(RECVFROM_TYPE_ARG2) || \
!defined(RECVFROM_TYPE_ARG3) || \
!defined(RECVFROM_TYPE_ARG4) || \
!defined(RECVFROM_TYPE_ARG5) || \
!defined(RECVFROM_TYPE_ARG6) || \
!defined(RECVFROM_TYPE_RETV)

Error Missing_definition_of_return_and_arguments_types_of_recvfrom

#else
#define sreadfrom(s,b,bl,f,fl) (ssize_t)recvfrom((RECVFROM_TYPE_ARG1)  (s),  \
(RECVFROM_TYPE_ARG2 *)(b),  \
(RECVFROM_TYPE_ARG3)  (bl), \
(RECVFROM_TYPE_ARG4)  (0),  \
(RECVFROM_TYPE_ARG5 *)(f),  \
(RECVFROM_TYPE_ARG6 *)(fl))
#endif
#else 
#ifndef sreadfrom

Error Missing_definition_of_macro_sreadfrom

#endif
#endif 


#ifdef RECVFROM_TYPE_ARG6_IS_VOID
#  define RECVFROM_ARG6_T int
#else
#  define RECVFROM_ARG6_T RECVFROM_TYPE_ARG6
#endif
#endif 




#if defined(HAVE_CLOSESOCKET)
#  define sclose(x)  closesocket((x))
#elif defined(HAVE_CLOSESOCKET_CAMEL)
#  define sclose(x)  CloseSocket((x))
#elif defined(HAVE_CLOSE_S)
#  define sclose(x)  close_s((x))
#elif defined(USE_LWIPSOCK)
#  define sclose(x)  lwip_close((x))
#else
#  define sclose(x)  close((x))
#endif


#if defined(USE_LWIPSOCK)
#  define sfcntl  lwip_fcntl
#else
#  define sfcntl  fcntl
#endif

#define TOLOWER(x)  (tolower((int)  ((unsigned char)x)))




#if defined(__hpux) && !defined(HAVE_BOOL_T)
typedef int bool;
#  define false 0
#  define true 1
#  define HAVE_BOOL_T
#endif




#ifndef HAVE_BOOL_T
typedef enum {
bool_false = 0,
bool_true  = 1
} bool;


#  define false bool_false
#  define true  bool_true
#  define HAVE_BOOL_T
#endif




#ifndef TRUE
#define TRUE true
#endif
#ifndef FALSE
#define FALSE false
#endif

#include "curl_ctype.h"



#define WHILE_FALSE  while(0)

#if defined(_MSC_VER) && !defined(__POCC__)
#  undef WHILE_FALSE
#  if (_MSC_VER < 1500)
#    define WHILE_FALSE  while(1, 0)
#  else
#    define WHILE_FALSE \
__pragma(warning(push)) \
__pragma(warning(disable:4127)) \
while(0) \
__pragma(warning(pop))
#  endif
#endif




#ifndef HAVE_SIG_ATOMIC_T
typedef int sig_atomic_t;
#define HAVE_SIG_ATOMIC_T
#endif




#ifdef HAVE_SIG_ATOMIC_T_VOLATILE
#define SIG_ATOMIC_T static sig_atomic_t
#else
#define SIG_ATOMIC_T static volatile sig_atomic_t
#endif




#ifndef RETSIGTYPE
#define RETSIGTYPE void
#endif




#ifdef DEBUGBUILD
#define DEBUGF(x) x
#else
#define DEBUGF(x) do { } WHILE_FALSE
#endif




#if defined(DEBUGBUILD) && defined(HAVE_ASSERT_H)
#define DEBUGASSERT(x) assert(x)
#else
#define DEBUGASSERT(x) do { } WHILE_FALSE
#endif




#ifdef USE_WINSOCK
#define SOCKERRNO         ((int)WSAGetLastError())
#define SET_SOCKERRNO(x)  (WSASetLastError((int)(x)))
#else
#define SOCKERRNO         (errno)
#define SET_SOCKERRNO(x)  (errno = (x))
#endif




#ifdef USE_WINSOCK
#undef  EBADF            
#define EBADF            WSAEBADF
#undef  EINTR            
#define EINTR            WSAEINTR
#undef  EINVAL           
#define EINVAL           WSAEINVAL
#undef  EWOULDBLOCK      
#define EWOULDBLOCK      WSAEWOULDBLOCK
#undef  EINPROGRESS      
#define EINPROGRESS      WSAEINPROGRESS
#undef  EALREADY         
#define EALREADY         WSAEALREADY
#undef  ENOTSOCK         
#define ENOTSOCK         WSAENOTSOCK
#undef  EDESTADDRREQ     
#define EDESTADDRREQ     WSAEDESTADDRREQ
#undef  EMSGSIZE         
#define EMSGSIZE         WSAEMSGSIZE
#undef  EPROTOTYPE       
#define EPROTOTYPE       WSAEPROTOTYPE
#undef  ENOPROTOOPT      
#define ENOPROTOOPT      WSAENOPROTOOPT
#undef  EPROTONOSUPPORT  
#define EPROTONOSUPPORT  WSAEPROTONOSUPPORT
#define ESOCKTNOSUPPORT  WSAESOCKTNOSUPPORT
#undef  EOPNOTSUPP       
#define EOPNOTSUPP       WSAEOPNOTSUPP
#define EPFNOSUPPORT     WSAEPFNOSUPPORT
#undef  EAFNOSUPPORT     
#define EAFNOSUPPORT     WSAEAFNOSUPPORT
#undef  EADDRINUSE       
#define EADDRINUSE       WSAEADDRINUSE
#undef  EADDRNOTAVAIL    
#define EADDRNOTAVAIL    WSAEADDRNOTAVAIL
#undef  ENETDOWN         
#define ENETDOWN         WSAENETDOWN
#undef  ENETUNREACH      
#define ENETUNREACH      WSAENETUNREACH
#undef  ENETRESET        
#define ENETRESET        WSAENETRESET
#undef  ECONNABORTED     
#define ECONNABORTED     WSAECONNABORTED
#undef  ECONNRESET       
#define ECONNRESET       WSAECONNRESET
#undef  ENOBUFS          
#define ENOBUFS          WSAENOBUFS
#undef  EISCONN          
#define EISCONN          WSAEISCONN
#undef  ENOTCONN         
#define ENOTCONN         WSAENOTCONN
#define ESHUTDOWN        WSAESHUTDOWN
#define ETOOMANYREFS     WSAETOOMANYREFS
#undef  ETIMEDOUT        
#define ETIMEDOUT        WSAETIMEDOUT
#undef  ECONNREFUSED     
#define ECONNREFUSED     WSAECONNREFUSED
#undef  ELOOP            
#define ELOOP            WSAELOOP
#ifndef ENAMETOOLONG     
#define ENAMETOOLONG     WSAENAMETOOLONG
#endif
#define EHOSTDOWN        WSAEHOSTDOWN
#undef  EHOSTUNREACH     
#define EHOSTUNREACH     WSAEHOSTUNREACH
#ifndef ENOTEMPTY        
#define ENOTEMPTY        WSAENOTEMPTY
#endif
#define EPROCLIM         WSAEPROCLIM
#define EUSERS           WSAEUSERS
#define EDQUOT           WSAEDQUOT
#define ESTALE           WSAESTALE
#define EREMOTE          WSAEREMOTE
#endif



#ifdef __VMS
#define argv_item_t  __char_ptr32
#else
#define argv_item_t  char *
#endif




#define ZERO_NULL 0


#endif 

