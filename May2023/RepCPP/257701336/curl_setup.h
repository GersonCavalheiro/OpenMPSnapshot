#ifndef HEADER_CURL_SETUP_H
#define HEADER_CURL_SETUP_H


#if defined(BUILDING_LIBCURL) && !defined(CURL_NO_OLDIES)
#define CURL_NO_OLDIES
#endif



#if (defined(_WIN32) || defined(__WIN32__)) && !defined(WIN32) && \
!defined(__SYMBIAN32__)
#define WIN32
#endif

#ifdef WIN32

#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#endif



#ifdef HAVE_CONFIG_H

#include "curl_config.h"

#else 

#ifdef _WIN32_WCE
#  include "config-win32ce.h"
#else
#  ifdef WIN32
#    include "config-win32.h"
#  endif
#endif

#if defined(macintosh) && defined(__MRC__)
#  include "config-mac.h"
#endif

#ifdef __riscos__
#  include "config-riscos.h"
#endif

#ifdef __AMIGA__
#  include "config-amigaos.h"
#endif

#ifdef __SYMBIAN32__
#  include "config-symbian.h"
#endif

#ifdef __OS400__
#  include "config-os400.h"
#endif

#ifdef TPF
#  include "config-tpf.h"
#endif

#ifdef __VXWORKS__
#  include "config-vxworks.h"
#endif

#endif 












#ifdef NEED_THREAD_SAFE
#  ifndef _THREAD_SAFE
#    define _THREAD_SAFE
#  endif
#endif



#ifdef NEED_REENTRANT
#  ifndef _REENTRANT
#    define _REENTRANT
#  endif
#endif


#if defined(sun) || defined(__sun)
#  ifndef _POSIX_PTHREAD_SEMANTICS
#    define _POSIX_PTHREAD_SEMANTICS 1
#  endif
#endif






#include <curl/curl.h>

#define CURL_SIZEOF_CURL_OFF_T SIZEOF_CURL_OFF_T



#ifdef HTTP_ONLY
#  ifndef CURL_DISABLE_TFTP
#    define CURL_DISABLE_TFTP
#  endif
#  ifndef CURL_DISABLE_FTP
#    define CURL_DISABLE_FTP
#  endif
#  ifndef CURL_DISABLE_LDAP
#    define CURL_DISABLE_LDAP
#  endif
#  ifndef CURL_DISABLE_TELNET
#    define CURL_DISABLE_TELNET
#  endif
#  ifndef CURL_DISABLE_DICT
#    define CURL_DISABLE_DICT
#  endif
#  ifndef CURL_DISABLE_FILE
#    define CURL_DISABLE_FILE
#  endif
#  ifndef CURL_DISABLE_RTSP
#    define CURL_DISABLE_RTSP
#  endif
#  ifndef CURL_DISABLE_POP3
#    define CURL_DISABLE_POP3
#  endif
#  ifndef CURL_DISABLE_IMAP
#    define CURL_DISABLE_IMAP
#  endif
#  ifndef CURL_DISABLE_SMTP
#    define CURL_DISABLE_SMTP
#  endif
#  ifndef CURL_DISABLE_GOPHER
#    define CURL_DISABLE_GOPHER
#  endif
#  ifndef CURL_DISABLE_SMB
#    define CURL_DISABLE_SMB
#  endif
#endif



#if defined(CURL_DISABLE_HTTP) && !defined(CURL_DISABLE_RTSP)
#  define CURL_DISABLE_RTSP
#endif








#ifdef __OS400__
#  include "setup-os400.h"
#endif



#ifdef __VMS
#  include "setup-vms.h"
#endif


#ifdef  __APPLE__
#define USE_RESOLVE_ON_IPS 1
#endif



#ifdef HAVE_WINDOWS_H
#  if defined(UNICODE) && !defined(_UNICODE)
#    define _UNICODE
#  endif
#  if defined(_UNICODE) && !defined(UNICODE)
#    define UNICODE
#  endif
#  include <windows.h>
#  ifdef HAVE_WINSOCK2_H
#    include <winsock2.h>
#    ifdef HAVE_WS2TCPIP_H
#      include <ws2tcpip.h>
#    endif
#  else
#    ifdef HAVE_WINSOCK_H
#      include <winsock.h>
#    endif
#  endif
#  include <tchar.h>
#  ifdef UNICODE
typedef wchar_t *(*curl_wcsdup_callback)(const wchar_t *str);
#  endif
#endif



#undef USE_WINSOCK

#ifdef HAVE_WINSOCK2_H
#  define USE_WINSOCK 2
#else
#  ifdef HAVE_WINSOCK_H
#    define USE_WINSOCK 1
#  endif
#endif

#ifdef USE_LWIPSOCK
#  include <lwip/init.h>
#  include <lwip/sockets.h>
#  include <lwip/netdb.h>
#endif

#ifdef HAVE_EXTRA_STRICMP_H
#  include <extra/stricmp.h>
#endif

#ifdef HAVE_EXTRA_STRDUP_H
#  include <extra/strdup.h>
#endif

#ifdef TPF
#  include <strings.h>    
#  include <string.h>     
#  include <stdlib.h>     
#  include <sys/socket.h> 
#  include <netdb.h>      
#  include <tpf/sysapi.h> 

#  define select(a,b,c,d,e) tpf_select_libcurl(a,b,c,d,e)
#endif

#ifdef __VXWORKS__
#  include <sockLib.h>    
#  include <ioLib.h>      
#endif

#ifdef __AMIGA__
#  ifndef __ixemul__
#    include <exec/types.h>
#    include <exec/execbase.h>
#    include <proto/exec.h>
#    include <proto/dos.h>
#    define select(a,b,c,d,e) WaitSelect(a,b,c,d,e,0)
#  endif
#endif

#include <stdio.h>
#ifdef HAVE_ASSERT_H
#include <assert.h>
#endif

#ifdef __TANDEM 
#include <floss.h>
#endif

#ifndef STDC_HEADERS 
#include <curl/stdcheaders.h>
#endif

#ifdef __POCC__
#  include <sys/types.h>
#  include <unistd.h>
#  define sys_nerr EILSEQ
#endif


#ifdef __SALFORDC__
#pragma suppress 353             
#pragma suppress 593             
#pragma suppress 61              
#pragma suppress 106             
#include <clib.h>
#endif



#ifdef USE_WIN32_LARGE_FILES
#  include <io.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  undef  lseek
#  define lseek(fdes,offset,whence)  _lseeki64(fdes, offset, whence)
#  undef  fstat
#  define fstat(fdes,stp)            _fstati64(fdes, stp)
#  undef  stat
#  define stat(fname,stp)            _stati64(fname, stp)
#  define struct_stat                struct _stati64
#  define LSEEK_ERROR                (__int64)-1
#endif



#ifdef USE_WIN32_SMALL_FILES
#  include <io.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  ifndef _WIN32_WCE
#    undef  lseek
#    define lseek(fdes,offset,whence)  _lseek(fdes, (long)offset, whence)
#    define fstat(fdes,stp)            _fstat(fdes, stp)
#    define stat(fname,stp)            _stat(fname, stp)
#    define struct_stat                struct _stat
#  endif
#  define LSEEK_ERROR                (long)-1
#endif

#ifndef struct_stat
#  define struct_stat struct stat
#endif

#ifndef LSEEK_ERROR
#  define LSEEK_ERROR (off_t)-1
#endif

#ifndef SIZEOF_TIME_T

#define SIZEOF_TIME_T 4
#endif



#ifndef SIZEOF_OFF_T
#  if defined(__VMS) && !defined(__VAX)
#    if defined(_LARGEFILE)
#      define SIZEOF_OFF_T 8
#    endif
#  elif defined(__OS400__) && defined(__ILEC400__)
#    if defined(_LARGE_FILES)
#      define SIZEOF_OFF_T 8
#    endif
#  elif defined(__MVS__) && defined(__IBMC__)
#    if defined(_LP64) || defined(_LARGE_FILES)
#      define SIZEOF_OFF_T 8
#    endif
#  elif defined(__370__) && defined(__IBMC__)
#    if defined(_LP64) || defined(_LARGE_FILES)
#      define SIZEOF_OFF_T 8
#    endif
#  endif
#  ifndef SIZEOF_OFF_T
#    define SIZEOF_OFF_T 4
#  endif
#endif

#if (SIZEOF_CURL_OFF_T == 4)
#  define CURL_OFF_T_MAX CURL_OFF_T_C(0x7FFFFFFF)
#else

#  define CURL_OFF_T_MAX CURL_OFF_T_C(0x7FFFFFFFFFFFFFFF)
#endif
#define CURL_OFF_T_MIN (-CURL_OFF_T_MAX - CURL_OFF_T_C(1))

#if (SIZEOF_TIME_T == 4)
#  ifdef HAVE_TIME_T_UNSIGNED
#  define TIME_T_MAX UINT_MAX
#  define TIME_T_MIN 0
#  else
#  define TIME_T_MAX INT_MAX
#  define TIME_T_MIN INT_MIN
#  endif
#else
#  ifdef HAVE_TIME_T_UNSIGNED
#  define TIME_T_MAX 0xFFFFFFFFFFFFFFFF
#  define TIME_T_MIN 0
#  else
#  define TIME_T_MAX 0x7FFFFFFFFFFFFFFF
#  define TIME_T_MIN (-TIME_T_MAX - 1)
#  endif
#endif

#ifndef SIZE_T_MAX

#if defined(SIZEOF_SIZE_T) && (SIZEOF_SIZE_T > 4)
#define SIZE_T_MAX 18446744073709551615U
#else
#define SIZE_T_MAX 4294967295U
#endif
#endif



#ifndef GETHOSTNAME_TYPE_ARG2
#  ifdef USE_WINSOCK
#    define GETHOSTNAME_TYPE_ARG2 int
#  else
#    define GETHOSTNAME_TYPE_ARG2 size_t
#  endif
#endif



#ifdef WIN32

#  define DIR_CHAR      "\\"
#  define DOT_CHAR      "_"

#else 

#  ifdef MSDOS  

#    include <sys/ioctl.h>
#    define select(n,r,w,x,t) select_s(n,r,w,x,t)
#    define ioctl(x,y,z) ioctlsocket(x,y,(char *)(z))
#    include <tcp.h>
#    ifdef word
#      undef word
#    endif
#    ifdef byte
#      undef byte
#    endif

#  endif 

#  ifdef __minix

extern char *strtok_r(char *s, const char *delim, char **last);
extern struct tm *gmtime_r(const time_t * const timep, struct tm *tmp);
#  endif

#  define DIR_CHAR      "/"
#  ifndef DOT_CHAR
#    define DOT_CHAR      "."
#  endif

#  ifdef MSDOS
#    undef DOT_CHAR
#    define DOT_CHAR      "_"
#  endif

#  ifndef fileno 
int fileno(FILE *stream);
#  endif

#endif 



#if defined(_MSC_VER) && !defined(__POCC__) && !defined(USE_LWIPSOCK)
#  if !defined(HAVE_WS2TCPIP_H) || \
((_MSC_VER < 1300) && !defined(INET6_ADDRSTRLEN))
#    undef HAVE_GETADDRINFO_THREADSAFE
#    undef HAVE_FREEADDRINFO
#    undef HAVE_GETADDRINFO
#    undef HAVE_GETNAMEINFO
#    undef ENABLE_IPV6
#  endif
#endif








#if defined(__LCC__) && defined(WIN32)
#  undef USE_THREADS_POSIX
#  undef USE_THREADS_WIN32
#endif



#if defined(_MSC_VER) && !defined(__POCC__) && !defined(_MT)
#  undef USE_THREADS_POSIX
#  undef USE_THREADS_WIN32
#endif



#ifdef USE_ARES
#  define CURLRES_ASYNCH
#  define CURLRES_ARES

#  undef HAVE_GETADDRINFO
#  undef HAVE_FREEADDRINFO
#  undef HAVE_GETHOSTBYNAME
#elif defined(USE_THREADS_POSIX) || defined(USE_THREADS_WIN32)
#  define CURLRES_ASYNCH
#  define CURLRES_THREADED
#else
#  define CURLRES_SYNCH
#endif

#ifdef ENABLE_IPV6
#  define CURLRES_IPV6
#else
#  define CURLRES_IPV4
#endif





#if defined(USE_WINSOCK) && (USE_WINSOCK != 2)
#  define CURL_DISABLE_TELNET 1
#endif



#if defined(_MSC_VER) && !defined(__POCC__)
#  if !defined(HAVE_WINSOCK2_H) || ((_MSC_VER < 1300) && !defined(IPPROTO_ESP))
#    undef HAVE_STRUCT_SOCKADDR_STORAGE
#  endif
#endif



#if defined(_MSC_VER) && !defined(__POCC__)
#  if !defined(HAVE_WINDOWS_H) || ((_MSC_VER < 1300) && !defined(_FILETIME_))
#    if !defined(ALLOW_MSVC6_WITHOUT_PSDK)
#      error MSVC 6.0 requires "February 2003 Platform SDK" a.k.a. \
"Windows Server 2003 PSDK"
#    else
#      define CURL_DISABLE_LDAP 1
#    endif
#  endif
#endif

#ifdef NETWARE
int netware_init(void);
#ifndef __NOVELL_LIBC__
#include <sys/bsdskt.h>
#include <sys/timeval.h>
#endif
#endif

#if defined(HAVE_LIBIDN2) && defined(HAVE_IDN2_H) && !defined(USE_WIN32_IDN)

#define USE_LIBIDN2
#endif

#if defined(USE_LIBIDN2) && defined(USE_WIN32_IDN)
#error "Both libidn2 and WinIDN are enabled, choose one."
#endif

#define LIBIDN_REQUIRED_VERSION "0.4.1"

#if defined(USE_GNUTLS) || defined(USE_OPENSSL) || defined(USE_NSS) || \
defined(USE_POLARSSL) || defined(USE_AXTLS) || defined(USE_MBEDTLS) || \
defined(USE_CYASSL) || defined(USE_SCHANNEL) || \
defined(USE_DARWINSSL) || defined(USE_GSKIT)
#define USE_SSL    
#endif


#if !defined(CURL_DISABLE_CRYPTO_AUTH) && \
(defined(HAVE_GSSAPI) || defined(USE_WINDOWS_SSPI))
#define USE_SPNEGO
#endif


#if !defined(CURL_DISABLE_CRYPTO_AUTH) && \
(defined(HAVE_GSSAPI) || defined(USE_WINDOWS_SSPI))
#define USE_KERBEROS5
#endif


#if !defined(CURL_DISABLE_NTLM) && !defined(CURL_DISABLE_CRYPTO_AUTH)
#if defined(USE_OPENSSL) || defined(USE_WINDOWS_SSPI) || \
defined(USE_GNUTLS) || defined(USE_NSS) || defined(USE_DARWINSSL) || \
defined(USE_OS400CRYPTO) || defined(USE_WIN32_CRYPTO) || \
defined(USE_MBEDTLS)

#define USE_NTLM

#  if defined(USE_MBEDTLS)

#  include <mbedtls/md4.h>
#  endif

#endif
#endif

#ifdef CURL_WANTS_CA_BUNDLE_ENV
#error "No longer supported. Set CURLOPT_CAINFO at runtime instead."
#endif



#if defined(__GNUC__) && ((__GNUC__ >= 3) || \
((__GNUC__ == 2) && defined(__GNUC_MINOR__) && (__GNUC_MINOR__ >= 7)))
#  define UNUSED_PARAM __attribute__((__unused__))
#  define WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#else
#  define UNUSED_PARAM 
#  define WARN_UNUSED_RESULT
#endif



#ifndef HEADER_CURL_SETUP_ONCE_H
#include "curl_setup_once.h"
#endif



#ifndef Curl_nop_stmt
#  define Curl_nop_stmt do { } WHILE_FALSE
#endif



#if defined(__LWIP_OPT_H__) || defined(LWIP_HDR_OPT_H)
#  if defined(SOCKET) || \
defined(USE_WINSOCK) || \
defined(HAVE_WINSOCK_H) || \
defined(HAVE_WINSOCK2_H) || \
defined(HAVE_WS2TCPIP_H)
#    error "Winsock and lwIP TCP/IP stack definitions shall not coexist!"
#  endif
#endif



#ifdef USE_WINSOCK
#  define SHUT_RD   0x00
#  define SHUT_WR   0x01
#  define SHUT_RDWR 0x02
#endif


#if !defined(S_ISREG) && defined(S_IFMT) && defined(S_IFREG)
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#endif


#if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif


#if defined(WIN32) || defined(MSDOS)
#define FOPEN_READTEXT "rt"
#define FOPEN_WRITETEXT "wt"
#define FOPEN_APPENDTEXT "at"
#elif defined(__CYGWIN__)

#define FOPEN_READTEXT "rt"
#define FOPEN_WRITETEXT "w"
#define FOPEN_APPENDTEXT "a"
#else
#define FOPEN_READTEXT "r"
#define FOPEN_WRITETEXT "w"
#define FOPEN_APPENDTEXT "a"
#endif


#if !defined(DONT_USE_RECV_BEFORE_SEND_WORKAROUND)
#  if defined(WIN32) || defined(__CYGWIN__)
#    define USE_RECV_BEFORE_SEND_WORKAROUND
#  endif
#else  
#  ifdef USE_RECV_BEFORE_SEND_WORKAROUND
#    undef USE_RECV_BEFORE_SEND_WORKAROUND
#  endif
#endif 


# if (defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0602)) || \
defined(WINAPI_FAMILY)
#  include <winapifamily.h>
#  if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP) &&  \
!WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#    define CURL_WINDOWS_APP
#  endif
# endif


#ifndef CURL_SA_FAMILY_T
#define CURL_SA_FAMILY_T unsigned short
#endif

#endif 
