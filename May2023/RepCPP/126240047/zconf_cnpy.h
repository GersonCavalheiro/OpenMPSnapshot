



#ifndef ZCONF_H
#define ZCONF_H







#if defined(__MSDOS__) && !defined(MSDOS)
#  define MSDOS
#endif
#if (defined(OS_2) || defined(__OS2__)) && !defined(OS2)
#  define OS2
#endif
#if defined(_WINDOWS) && !defined(WINDOWS)
#  define WINDOWS
#endif
#if defined(_WIN32) || defined(_WIN32_WCE) || defined(__WIN32__)
#  ifndef WIN32
#    define WIN32
#  endif
#endif
#if (defined(MSDOS) || defined(OS2) || defined(WINDOWS)) && !defined(WIN32)
#  if !defined(__GNUC__) && !defined(__FLAT__) && !defined(__386__)
#    ifndef SYS16BIT
#      define SYS16BIT
#    endif
#  endif
#endif


#ifdef SYS16BIT
#  define MAXSEG_64K
#endif
#ifdef MSDOS
#  define UNALIGNED_OK
#endif

#ifdef __STDC_VERSION__
#  ifndef STDC
#    define STDC
#  endif
#  if __STDC_VERSION__ >= 199901L
#    ifndef STDC99
#      define STDC99
#    endif
#  endif
#endif
#if !defined(STDC) && (defined(__STDC__) || defined(__cplusplus))
#  define STDC
#endif
#if !defined(STDC) && (defined(__GNUC__) || defined(__BORLANDC__))
#  define STDC
#endif
#if !defined(STDC) && (defined(MSDOS) || defined(WINDOWS) || defined(WIN32))
#  define STDC
#endif
#if !defined(STDC) && (defined(OS2) || defined(__HOS_AIX__))
#  define STDC
#endif

#if defined(__OS400__) && !defined(STDC)    
#  define STDC
#endif

#ifndef STDC
#  ifndef const 
#    define const       
#  endif
#endif

#if defined(ZLIB_CONST) && !defined(z_const)
#  define z_const const
#else
#  define z_const
#endif


#if defined(__MWERKS__)||defined(applec)||defined(THINK_C)||defined(__SC__)
#  define NO_DUMMY_DECL
#endif


#ifndef MAX_MEM_LEVEL
#  ifdef MAXSEG_64K
#    define MAX_MEM_LEVEL 8
#  else
#    define MAX_MEM_LEVEL 9
#  endif
#endif


#ifndef MAX_WBITS
#  define MAX_WBITS   15 
#endif





#ifndef OF 
#  ifdef STDC
#    define OF(args)  args
#  else
#    define OF(args)  ()
#  endif
#endif

#ifndef Z_ARG 
#  if defined(STDC) || defined(Z_HAVE_STDARG_H)
#    define Z_ARG(args)  args
#  else
#    define Z_ARG(args)  ()
#  endif
#endif


#ifdef SYS16BIT
#  if defined(M_I86SM) || defined(M_I86MM)

#    define SMALL_MEDIUM
#    ifdef _MSC_VER
#      define FAR _far
#    else
#      define FAR far
#    endif
#  endif
#  if (defined(__SMALL__) || defined(__MEDIUM__))

#    define SMALL_MEDIUM
#    ifdef __BORLANDC__
#      define FAR _far
#    else
#      define FAR far
#    endif
#  endif
#endif

#if defined(WINDOWS) || defined(WIN32)

#  ifdef ZLIB_DLL
#    if defined(WIN32) && (!defined(__BORLANDC__) || (__BORLANDC__ >= 0x500))
#      ifdef ZLIB_INTERNAL
#        define ZEXTERN extern __declspec(dllexport)
#      else
#        define ZEXTERN extern __declspec(dllimport)
#      endif
#    endif
#  endif  

#  ifdef ZLIB_WINAPI
#    ifdef FAR
#      undef FAR
#    endif
#    include <windows.h>


#    define ZEXPORT WINAPI
#    ifdef WIN32
#      define ZEXPORTVA WINAPIV
#    else
#      define ZEXPORTVA FAR CDECL
#    endif
#  endif
#endif

#if defined (__BEOS__)
#  ifdef ZLIB_DLL
#    ifdef ZLIB_INTERNAL
#      define ZEXPORT   __declspec(dllexport)
#      define ZEXPORTVA __declspec(dllexport)
#    else
#      define ZEXPORT   __declspec(dllimport)
#      define ZEXPORTVA __declspec(dllimport)
#    endif
#  endif
#endif

#ifndef ZEXTERN
#  define ZEXTERN extern
#endif
#ifndef ZEXPORT
#  define ZEXPORT
#endif
#ifndef ZEXPORTVA
#  define ZEXPORTVA
#endif

#ifndef FAR
#  define FAR
#endif

#if !defined(__MACTYPES__)
typedef unsigned char  Byte;  
#endif
typedef unsigned int   uInt;  
typedef unsigned long  uLong; 

#ifdef SMALL_MEDIUM

#  define Bytef Byte FAR
#else
typedef Byte  FAR Bytef;
#endif
typedef char  FAR charf;
typedef int   FAR intf;
typedef uInt  FAR uIntf;
typedef uLong FAR uLongf;

#ifdef STDC
typedef void const *voidpc;
typedef void FAR   *voidpf;
typedef void       *voidp;
#else
typedef Byte const *voidpc;
typedef Byte FAR   *voidpf;
typedef Byte       *voidp;
#endif

#if !defined(Z_U4) && !defined(Z_SOLO) && defined(STDC)
#  include <limits.h>
#  if (UINT_MAX == 0xffffffffUL)
#    define Z_U4 unsigned
#  elif (ULONG_MAX == 0xffffffffUL)
#    define Z_U4 unsigned long
#  elif (USHRT_MAX == 0xffffffffUL)
#    define Z_U4 unsigned short
#  endif
#endif

#ifdef Z_U4
typedef Z_U4 z_crc_t;
#else
typedef unsigned long z_crc_t;
#endif

#ifdef HAVE_UNISTD_H    
#  define Z_HAVE_UNISTD_H
#endif

#ifdef HAVE_STDARG_H    
#  define Z_HAVE_STDARG_H
#endif

#ifdef STDC
#  ifndef Z_SOLO
#    include <sys/types.h>      
#  endif
#endif

#if defined(STDC) || defined(Z_HAVE_STDARG_H)
#  ifndef Z_SOLO
#    include <stdarg.h>         
#  endif
#endif

#ifdef _WIN32
#  ifndef Z_SOLO
#    include <stddef.h>         
#  endif
#endif


#if defined(_LARGEFILE64_SOURCE) && -_LARGEFILE64_SOURCE - -1 == 1
#  undef _LARGEFILE64_SOURCE
#endif

#if defined(__WATCOMC__) && !defined(Z_HAVE_UNISTD_H)
#  define Z_HAVE_UNISTD_H
#endif
#ifndef Z_SOLO
#  if defined(Z_HAVE_UNISTD_H) || defined(_LARGEFILE64_SOURCE)
#    include <unistd.h>         
#    ifdef VMS
#      include <unixio.h>       
#    endif
#    ifndef z_off_t
#      define z_off_t off_t
#    endif
#  endif
#endif

#if defined(_LFS64_LARGEFILE) && _LFS64_LARGEFILE-0
#  define Z_LFS64
#endif

#if defined(_LARGEFILE64_SOURCE) && defined(Z_LFS64)
#  define Z_LARGE64
#endif

#if defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS-0 == 64 && defined(Z_LFS64)
#  define Z_WANT64
#endif

#if !defined(SEEK_SET) && !defined(Z_SOLO)
#  define SEEK_SET        0       
#  define SEEK_CUR        1       
#  define SEEK_END        2       
#endif

#ifndef z_off_t
#  define z_off_t long
#endif

#if !defined(_WIN32) && defined(Z_LARGE64)
#  define z_off64_t off64_t
#else
#  if defined(_WIN32) && !defined(__GNUC__) && !defined(Z_SOLO)
#    define z_off64_t __int64
#  else
#    define z_off64_t z_off_t
#  endif
#endif


#if defined(__MVS__)
#pragma map(deflateInit_,"DEIN")
#pragma map(deflateInit2_,"DEIN2")
#pragma map(deflateEnd,"DEEND")
#pragma map(deflateBound,"DEBND")
#pragma map(inflateInit_,"ININ")
#pragma map(inflateInit2_,"ININ2")
#pragma map(inflateEnd,"INEND")
#pragma map(inflateSync,"INSY")
#pragma map(inflateSetDictionary,"INSEDI")
#pragma map(compressBound,"CMBND")
#pragma map(inflate_table,"INTABL")
#pragma map(inflate_fast,"INFA")
#pragma map(inflate_copyright,"INCOPY")
#endif

#endif 
