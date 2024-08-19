#ifndef _JTR_JUMBO_H
#define _JTR_JUMBO_H
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include "arch.h"
#include <stdio.h>
#include <errno.h>
#if !AC_BUILT || HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#if (!AC_BUILT || HAVE_UNISTD_H) && !_MSC_VER
#include <unistd.h>
#endif
#if !AC_BUILT || HAVE_STRING_H
#include <string.h>
#endif
#if !AC_BUILT && (_MSC_VER || __MINGW32__ || __MINGW64__)
#define HAVE__ATOI64 1
#endif
#include <stdint.h>
#if (!AC_BUILT || HAVE_INTTYPES_H) && ! defined(_MSC_VER)
#include <inttypes.h>
#else
#ifndef PRIx64
#define PRIx64    "llx"
#endif
#ifndef PRIu64
#define PRIu64    "llu"
#endif
#ifndef PRId64
#define PRId64    "lld"
#endif
#endif
#if SIZEOF_LONG == 8
#define jtr_fseek64 fseek
#elif HAVE_FSEEK64 
#define jtr_fseek64 fseek64
#elif HAVE_FSEEKO64 
#define jtr_fseek64 fseeko64
#elif defined (HAVE__FSEEKI64) || defined (_MSC_VER) 
#define jtr_fseek64 _fseeki64
#elif SIZEOF_OFF_T == 8 && HAVE_FSEEKO 
#define jtr_fseek64 fseeko
#elif HAVE_LSEEK64 
#define jtr_fseek64(s,o,w) lseek64(fileno(s),o,w);
#elif SIZEOF_OFF_T == 8 && HAVE_LSEEK 
#define jtr_fseek64(s,o,w) lseek(fileno(s),o,w)
#else
#if defined (__CYGWIN32__) && !defined (__CYGWIN64__)
extern  int fseeko64 (FILE* stream, int64_t offset, int whence);
#define jtr_fseek64 fseeko64
#elif defined (__CYGWIN64__)
extern  int fseeko (FILE* stream, int64_t offset, int whence);
#define jtr_fseek64 fseeko
#else
#if defined(__GNUC__) && defined (AC_BUILT)
#warning Using 32-bit fseek(). Files larger than 2GB will be handled unreliably
#endif
#define jtr_fseek64 fseek
#endif
#endif 
#if SIZEOF_LONG == 8
#define jtr_ftell64 ftell
#elif HAVE_FTELL64 
#define jtr_ftell64 ftell64
#elif HAVE_FTELLO64 
#define jtr_ftell64 ftello64
#elif defined (HAVE__FTELLI64) || defined (_MSC_VER) 
#define jtr_ftell64 _ftelli64
#elif SIZEOF_OFF_T == 8 && HAVE_FTELLO 
#define jtr_ftell64 ftello
#else
#if defined (__CYGWIN32__) && !defined (__CYGWIN64__)
extern  int64_t ftello64 (FILE* stream);
#define jtr_ftell64 ftello64
#elif defined (__CYGWIN64__)
extern  int64_t ftello (FILE* stream);
#define jtr_ftell64 ftello
#else
#if defined(__GNUC__) && defined (AC_BUILT)
#warning Using 32-bit ftell(). Files larger than 2GB will be handled unreliably
#endif
#define jtr_ftell64 ftell
#endif
#endif 
#if SIZEOF_LONG == 8
#define jtr_fopen fopen
#elif HAVE_FOPEN64
#define jtr_fopen fopen64
#elif HAVE__FOPEN64
#define jtr_fopen _fopen64
#else
#define jtr_fopen fopen
#endif
#if __CYGWIN32__ || _MSC_VER
extern  FILE *_fopen64 (const char *Fname, const char *type);
#endif
#ifndef O_LARGEFILE
#define O_LARGEFILE 0
#endif
extern char *jtr_basename(const char *name);
extern char *jtr_basename_r(const char *name, char *buf);
#undef basename
#define basename(a) jtr_basename(a)
extern char *strip_suffixes(const char *src, const char *suffixes[], int count);
#if !HAVE_MEMMEM
#undef memmem
#define memmem	jtr_memmem
extern void *memmem(const void *haystack, size_t haystack_len,
const void *needle, size_t needle_len);
#endif 
#if (AC_BUILT && !HAVE_SLEEP) || (!AC_BUILT && (_MSC_VER || __MINGW32__ || __MINGW64__))
extern unsigned int sleep(unsigned int i);
#endif
#if !AC_BUILT
#if _MSC_VER
#define strcasecmp _stricmp
#endif
#else
#if !HAVE_STRCASECMP
#if HAVE__STRICMP
#define strcasecmp _stricmp
#elif HAVE__STRCMPI
#define strcasecmp _strcmpi
#elif HAVE_STRICMP
#define strcasecmp stricmp
#elif HAVE_STRCMPI
#define strcasecmp strcmpi
#else
#define NEED_STRCASECMP_NATIVE 1
extern int strcasecmp(const char *dst, const char *src);
#endif
#endif
#endif
#if !AC_BUILT
#if _MSC_VER
#define strncasecmp _strnicmp
#endif
#else
#if !HAVE_STRNCASECMP
#if HAVE__STRNICMP
#define strncasecmp _strnicmp
#elif HAVE__STRNCMPI
#define strncasecmp _strncmpi
#elif HAVE_STRNICMP
#define strncasecmp strnicmp
#elif HAVE_STRNCMPI
#define strncasecmp strncmpi
#else
#define NEED_STRNCASECMP_NATIVE 1
extern int strncasecmp(const char *dst, const char *src, size_t count);
#endif
#endif
#endif
#if (AC_BUILT && HAVE__STRUPR && HAVE_STRUPR) || (!AC_BUILT && _MSC_VER)
#define strupr _strupr
#endif
#if (AC_BUILT && HAVE__STRLWR && HAVE_STRLWR) || (!AC_BUILT && _MSC_VER)
#define strlwr _strlwr
#endif
#if (AC_BUILT && !HAVE_STRLWR) || (!AC_BUILT && !_MSC_VER)
extern char *strlwr(char *s);
#endif
#if (AC_BUILT && !HAVE_STRUPR) || (!AC_BUILT && !_MSC_VER)
extern char *strupr(char *s);
#endif
#if !HAVE_ATOLL
#if HAVE__ATOI64
#define atoll _atoi64
#else
#define NEED_ATOLL_NATIVE 1
#undef atoll
#define atoll jtr_atoll
extern long long jtr_atoll(const char *);
#endif
#endif
void memcpylwr(char *, const char *, size_t);
#if (__MINGW32__ || __MINGW64__) && __STRICT_ANSI__
extern char *strdup(const char *);
extern char *strlwr(char *);
extern char *strupr(char *);
extern int _strncmp(const char*, const char *);
extern FILE *fopen64(const char *, const char *);
extern FILE *fdopen(int, const char *);
extern long long ftello64(FILE *);
extern int fseeko64(FILE *, long long, int);
extern int fileno(FILE *);
#define off64_t long long
#undef __STRICT_ANSI__
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#define __STRICT_ANSI__ 1
#endif
#ifdef _MSC_VER
#undef inline
#define inline _inline
#define strupr _strupr
#define strlwr _strlwr
#define open _open
#define fdopen _fdopen
#pragma warning(disable: 4244) 
#pragma warning(disable: 4334) 
#pragma warning(disable: 4133) 
#pragma warning(disable: 4146) 
#pragma warning(disable: 4715) 
#endif
#if (AC_BUILT && !HAVE_SNPRINTF && HAVE_SPRINTF_S) || (!AC_BUILT && _MSC_VER)
#undef  snprintf
#define snprintf sprintf_s
#endif
#if _MSC_VER
#undef snprintf
#if _MSC_VER < 1900
#define snprintf(str, size, ...) vc_fixed_snprintf((str), (size), __VA_ARGS__)
extern int vc_fixed_snprintf(char *Dest, size_t max_cnt, const char *Fmt, ...);
#endif
#undef alloca
#define alloca _alloca
#undef unlink
#define unlink _unlink
#undef fileno
#define fileno _fileno
#pragma warning (disable : 4018 297 )
#endif
#if (AC_BUILT && !HAVE_SETENV && HAVE_PUTENV) || \
(!AC_BUILT && (_MSC_VER || __MINGW32__ || __MINGW64__))
extern int setenv(const char *name, const char *val, int overwrite);
#endif
#if (__MINGW32__ && !__MINGW64__) || _MSC_VER
#define LLu "%I64u"
#define LLd "%I64d"
#define LLx "%I64x"
#define Zu  "%u"
#define Zd  "%d"
#else
#define LLu "%llu"
#define LLd "%lld"
#define LLx "%llx"
#define Zu  "%zu"
#define Zd  "%zd"
#endif
#if (AC_BUILT && !HAVE_STRREV) ||(!AC_BUILT && !_MSC_VER)
char *strrev(char *str);
#endif
#if AC_BUILT && !HAVE_STRNLEN
#undef strnlen
#define strnlen jtr_strnlen
extern size_t strnlen(const char *s, size_t max);
#endif
#if AC_BUILT && !HAVE_STRCASESTR || !AC_BUILT && defined(__MINGW__)
char *strcasestr(const char *haystack, const char *needle);
#endif
extern int check_pkcs_pad(const unsigned char* data, size_t len, int blocksize);
extern char *replace(char *string, char c, char n);
#endif 
