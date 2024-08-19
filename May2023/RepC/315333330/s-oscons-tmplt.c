#if defined (__linux__) || defined (__ANDROID__)
# if !defined (_XOPEN_SOURCE)
#  define _XOPEN_SOURCE 500
# endif
# define _BSD_SOURCE
#endif 
#include "gsocket.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>
#include <time.h>
#if ! (defined (__vxworks) || defined (__MINGW32__))
# define HAVE_TERMIOS
#endif
#if defined (__vxworks)
# include <vxWorks.h>
#endif
#include "adaint.h"
#ifdef DUMMY
# if defined (TARGET)
#   error TARGET may not be defined when generating the dummy version
# else
#   define TARGET "batch runtime compilation (dummy values)"
# endif
# if !(defined (HAVE_SOCKETS) && defined (HAVE_TERMIOS))
#   error Features missing on platform
# endif
# define NATIVE
#endif 
#ifndef TARGET
# error Please define TARGET
#endif
#ifndef HAVE_SOCKETS
# include <errno.h>
#endif
#ifdef HAVE_TERMIOS
# include <termios.h>
#endif
#ifdef __APPLE__
# include <_types.h>
#endif
#if defined (__linux__) || defined (__ANDROID__) || defined (__QNX__) \
|| defined (__rtems__)
# include <pthread.h>
# include <signal.h>
#endif
#if defined(__MINGW32__) || defined(__CYGWIN__)
# include <windef.h>
# include <winbase.h>
#endif
#ifdef NATIVE
#include <stdio.h>
#ifdef DUMMY
int counter = 0;
# define _VAL(x) counter++
#else
# define _VAL(x) x
#endif
#define CND(name,comment) \
printf ("\n->CND:$%d:" #name ":$%d:" comment, __LINE__, ((int) _VAL (name)));
#define CNU(name,comment) \
printf ("\n->CNU:$%d:" #name ":$%u:" comment, __LINE__, ((unsigned int) _VAL (name)));
#define CNS(name,comment) \
printf ("\n->CNS:$%d:" #name ":" name ":" comment, __LINE__);
#define C(sname,type,value,comment)\
printf ("\n->C:$%d:" sname ":" #type ":" value ":" comment, __LINE__);
#define SUB(sname)\
printf ("\n->SUB:$%d:" #sname ":" sname, __LINE__);
#define TXT(text) \
printf ("\n->TXT:$%d:" text, __LINE__);
#else
#define CND(name, comment) \
asm volatile("\n->CND:%0:" #name ":%1:" comment \
: : "i" (__LINE__), "i" ((int) name));
#define CNU(name, comment) \
asm volatile("\n->CNU:%0:" #name ":%1:" comment \
: : "i" (__LINE__), "i" ((int) name));
#define CNS(name, comment) \
asm volatile("\n->CNS:%0:" #name ":" name ":" comment \
: : "i" (__LINE__));
#define C(sname, type, value, comment) \
asm volatile("\n->C:%0:" sname ":" #type ":" value ":" comment \
: : "i" (__LINE__));
#define SUB(sname) \
asm volatile("\n->SUB:%0:" #sname ":" sname \
: : "i" (__LINE__));
#define TXT(text) \
asm volatile("\n->TXT:%0:" text \
: : "i" (__LINE__));
#endif 
#define CST(name,comment) C(#name,String,name,comment)
#define STR(x) STR1(x)
#define STR1(x) #x
#ifdef __MINGW32__
unsigned int _CRT_fmode = _O_BINARY;
#endif
int
main (void) {
TXT("--  This is the version for " TARGET)
TXT("")
TXT("with Interfaces.C;")
#if defined (__MINGW32__)
# define TARGET_OS "Windows"
#else
# define TARGET_OS "Other_OS"
#endif
C("Target_OS", OS_Type, TARGET_OS, "")
#define Target_Name TARGET
CST(Target_Name, "")
#define SIZEOF_unsigned_int sizeof (unsigned int)
CND(SIZEOF_unsigned_int, "Size of unsigned int")
#ifndef IOV_MAX
# define IOV_MAX INT_MAX
#endif
CND(IOV_MAX, "Maximum writev iovcnt")
#ifndef NAME_MAX
# ifdef MAXNAMELEN
#  define NAME_MAX MAXNAMELEN
# elif defined(PATH_MAX)
#  define NAME_MAX PATH_MAX
# elif defined(FILENAME_MAX)
#  define NAME_MAX FILENAME_MAX
# else
#  define NAME_MAX 1024
# endif
#endif
CND(NAME_MAX, "Maximum file name length")
#ifndef O_RDWR
# define O_RDWR -1
#endif
CND(O_RDWR, "Read/write")
#ifndef O_NOCTTY
# define O_NOCTTY -1
#endif
CND(O_NOCTTY, "Don't change ctrl tty")
#ifndef O_NDELAY
# define O_NDELAY -1
#endif
CND(O_NDELAY, "Nonblocking")
#ifndef F_GETFL
# define F_GETFL -1
#endif
CND(F_GETFL, "Get flags")
#ifndef F_SETFL
# define F_SETFL -1
#endif
CND(F_SETFL, "Set flags")
#ifndef FNDELAY
# define FNDELAY -1
#endif
CND(FNDELAY, "Nonblocking")
#if defined (__FreeBSD__) || defined (__DragonFly__)
# define CNI CNU
# define IOCTL_Req_T "unsigned"
#else
# define CNI CND
# define IOCTL_Req_T "int"
#endif
SUB(IOCTL_Req_T)
#ifndef FIONBIO
# define FIONBIO -1
#endif
CNI(FIONBIO, "Set/clear non-blocking io")
#ifndef FIONREAD
# define FIONREAD -1
#endif
CNI(FIONREAD, "How many bytes to read")
#ifndef EAGAIN
# define EAGAIN -1
#endif
CND(EAGAIN, "Try again")
#ifndef ENOENT
# define ENOENT -1
#endif
CND(ENOENT, "File not found")
#ifndef ENOMEM
# define ENOMEM -1
#endif
CND(ENOMEM, "Out of memory")
#ifdef __MINGW32__
#endif
#ifndef EACCES
# define EACCES -1
#endif
CND(EACCES, "Permission denied")
#ifndef EADDRINUSE
# define EADDRINUSE -1
#endif
CND(EADDRINUSE, "Address already in use")
#ifndef EADDRNOTAVAIL
# define EADDRNOTAVAIL -1
#endif
CND(EADDRNOTAVAIL, "Cannot assign address")
#ifndef EAFNOSUPPORT
# define EAFNOSUPPORT -1
#endif
CND(EAFNOSUPPORT, "Addr family not supported")
#ifndef EALREADY
# define EALREADY -1
#endif
CND(EALREADY, "Operation in progress")
#ifndef EBADF
# define EBADF -1
#endif
CND(EBADF, "Bad file descriptor")
#ifndef ECONNABORTED
# define ECONNABORTED -1
#endif
CND(ECONNABORTED, "Connection aborted")
#ifndef ECONNREFUSED
# define ECONNREFUSED -1
#endif
CND(ECONNREFUSED, "Connection refused")
#ifndef ECONNRESET
# define ECONNRESET -1
#endif
CND(ECONNRESET, "Connection reset by peer")
#ifndef EDESTADDRREQ
# define EDESTADDRREQ -1
#endif
CND(EDESTADDRREQ, "Destination addr required")
#ifndef EFAULT
# define EFAULT -1
#endif
CND(EFAULT, "Bad address")
#ifndef EHOSTDOWN
# define EHOSTDOWN -1
#endif
CND(EHOSTDOWN, "Host is down")
#ifndef EHOSTUNREACH
# define EHOSTUNREACH -1
#endif
CND(EHOSTUNREACH, "No route to host")
#ifndef EINPROGRESS
# define EINPROGRESS -1
#endif
CND(EINPROGRESS, "Operation now in progress")
#ifndef EINTR
# define EINTR -1
#endif
CND(EINTR, "Interrupted system call")
#ifndef EINVAL
# define EINVAL -1
#endif
CND(EINVAL, "Invalid argument")
#ifndef EIO
# define EIO -1
#endif
CND(EIO, "Input output error")
#ifndef EISCONN
# define EISCONN -1
#endif
CND(EISCONN, "Socket already connected")
#ifndef ELOOP
# define ELOOP -1
#endif
CND(ELOOP, "Too many symbolic links")
#ifndef EMFILE
# define EMFILE -1
#endif
CND(EMFILE, "Too many open files")
#ifndef EMSGSIZE
# define EMSGSIZE -1
#endif
CND(EMSGSIZE, "Message too long")
#ifndef ENAMETOOLONG
# define ENAMETOOLONG -1
#endif
CND(ENAMETOOLONG, "Name too long")
#ifndef ENETDOWN
# define ENETDOWN -1
#endif
CND(ENETDOWN, "Network is down")
#ifndef ENETRESET
# define ENETRESET -1
#endif
CND(ENETRESET, "Disconn. on network reset")
#ifndef ENETUNREACH
# define ENETUNREACH -1
#endif
CND(ENETUNREACH, "Network is unreachable")
#ifndef ENOBUFS
# define ENOBUFS -1
#endif
CND(ENOBUFS, "No buffer space available")
#ifndef ENOPROTOOPT
# define ENOPROTOOPT -1
#endif
CND(ENOPROTOOPT, "Protocol not available")
#ifndef ENOTCONN
# define ENOTCONN -1
#endif
CND(ENOTCONN, "Socket not connected")
#ifndef ENOTSOCK
# define ENOTSOCK -1
#endif
CND(ENOTSOCK, "Operation on non socket")
#ifndef EOPNOTSUPP
# define EOPNOTSUPP -1
#endif
CND(EOPNOTSUPP, "Operation not supported")
#ifndef EPIPE
# define EPIPE -1
#endif
CND(EPIPE, "Broken pipe")
#ifndef EPFNOSUPPORT
# define EPFNOSUPPORT -1
#endif
CND(EPFNOSUPPORT, "Unknown protocol family")
#ifndef EPROTONOSUPPORT
# define EPROTONOSUPPORT -1
#endif
CND(EPROTONOSUPPORT, "Unknown protocol")
#ifndef EPROTOTYPE
# define EPROTOTYPE -1
#endif
CND(EPROTOTYPE, "Unknown protocol type")
#ifndef ERANGE
# define ERANGE -1
#endif
CND(ERANGE, "Result too large")
#ifndef ESHUTDOWN
# define ESHUTDOWN -1
#endif
CND(ESHUTDOWN, "Cannot send once shutdown")
#ifndef ESOCKTNOSUPPORT
# define ESOCKTNOSUPPORT -1
#endif
CND(ESOCKTNOSUPPORT, "Socket type not supported")
#ifndef ETIMEDOUT
# define ETIMEDOUT -1
#endif
CND(ETIMEDOUT, "Connection timed out")
#ifndef ETOOMANYREFS
# define ETOOMANYREFS -1
#endif
CND(ETOOMANYREFS, "Too many references")
#ifndef EWOULDBLOCK
# define EWOULDBLOCK -1
#endif
CND(EWOULDBLOCK, "Operation would block")
#ifndef E2BIG
# define E2BIG -1
#endif
CND(E2BIG, "Argument list too long")
#ifndef EILSEQ
# define EILSEQ -1
#endif
CND(EILSEQ, "Illegal byte sequence")
#if defined(HAVE_TERMIOS) || defined(__MINGW32__)
#endif
#ifdef HAVE_TERMIOS
#ifndef TCSANOW
# define TCSANOW -1
#endif
CND(TCSANOW, "Immediate")
#ifndef TCIFLUSH
# define TCIFLUSH -1
#endif
CND(TCIFLUSH, "Flush input")
#ifndef IXON
# define IXON -1
#endif
CNU(IXON, "Output sw flow control")
#ifndef CLOCAL
# define CLOCAL -1
#endif
CNU(CLOCAL, "Local")
#ifndef CRTSCTS
# define CRTSCTS -1
#endif
CNU(CRTSCTS, "Output hw flow control")
#ifndef CREAD
# define CREAD -1
#endif
CNU(CREAD, "Read")
#ifndef CS5
# define CS5 -1
#endif
CNU(CS5, "5 data bits")
#ifndef CS6
# define CS6 -1
#endif
CNU(CS6, "6 data bits")
#ifndef CS7
# define CS7 -1
#endif
CNU(CS7, "7 data bits")
#ifndef CS8
# define CS8 -1
#endif
CNU(CS8, "8 data bits")
#ifndef CSTOPB
# define CSTOPB -1
#endif
CNU(CSTOPB, "2 stop bits")
#ifndef PARENB
# define PARENB -1
#endif
CNU(PARENB, "Parity enable")
#ifndef PARODD
# define PARODD -1
#endif
CNU(PARODD, "Parity odd")
#ifndef B0
# define B0 -1
#endif
CNU(B0, "0 bps")
#ifndef B50
# define B50 -1
#endif
CNU(B50, "50 bps")
#ifndef B75
# define B75 -1
#endif
CNU(B75, "75 bps")
#ifndef B110
# define B110 -1
#endif
CNU(B110, "110 bps")
#ifndef B134
# define B134 -1
#endif
CNU(B134, "134 bps")
#ifndef B150
# define B150 -1
#endif
CNU(B150, "150 bps")
#ifndef B200
# define B200 -1
#endif
CNU(B200, "200 bps")
#ifndef B300
# define B300 -1
#endif
CNU(B300, "300 bps")
#ifndef B600
# define B600 -1
#endif
CNU(B600, "600 bps")
#ifndef B1200
# define B1200 -1
#endif
CNU(B1200, "1200 bps")
#ifndef B1800
# define B1800 -1
#endif
CNU(B1800, "1800 bps")
#ifndef B2400
# define B2400 -1
#endif
CNU(B2400, "2400 bps")
#ifndef B4800
# define B4800 -1
#endif
CNU(B4800, "4800 bps")
#ifndef B9600
# define B9600 -1
#endif
CNU(B9600, "9600 bps")
#ifndef B19200
# define B19200 -1
#endif
CNU(B19200, "19200 bps")
#ifndef B38400
# define B38400 -1
#endif
CNU(B38400, "38400 bps")
#ifndef B57600
# define B57600 -1
#endif
CNU(B57600, "57600 bps")
#ifndef B115200
# define B115200 -1
#endif
CNU(B115200, "115200 bps")
#ifndef B230400
# define B230400 -1
#endif
CNU(B230400, "230400 bps")
#ifndef B460800
# define B460800 -1
#endif
CNU(B460800, "460800 bps")
#ifndef B500000
# define B500000 -1
#endif
CNU(B500000, "500000 bps")
#ifndef B576000
# define B576000 -1
#endif
CNU(B576000, "576000 bps")
#ifndef B921600
# define B921600 -1
#endif
CNU(B921600, "921600 bps")
#ifndef B1000000
# define B1000000 -1
#endif
CNU(B1000000, "1000000 bps")
#ifndef B1152000
# define B1152000 -1
#endif
CNU(B1152000, "1152000 bps")
#ifndef B1500000
# define B1500000 -1
#endif
CNU(B1500000, "1500000 bps")
#ifndef B2000000
# define B2000000 -1
#endif
CNU(B2000000, "2000000 bps")
#ifndef B2500000
# define B2500000 -1
#endif
CNU(B2500000, "2500000 bps")
#ifndef B3000000
# define B3000000 -1
#endif
CNU(B3000000, "3000000 bps")
#ifndef B3500000
# define B3500000 -1
#endif
CNU(B3500000, "3500000 bps")
#ifndef B4000000
# define B4000000 -1
#endif
CNU(B4000000, "4000000 bps")
#ifndef VINTR
# define VINTR -1
#endif
CND(VINTR, "Interrupt")
#ifndef VQUIT
# define VQUIT -1
#endif
CND(VQUIT, "Quit")
#ifndef VERASE
# define VERASE -1
#endif
CND(VERASE, "Erase")
#ifndef VKILL
# define VKILL -1
#endif
CND(VKILL, "Kill")
#ifndef VEOF
# define VEOF -1
#endif
CND(VEOF, "EOF")
#ifndef VTIME
# define VTIME -1
#endif
CND(VTIME, "Read timeout")
#ifndef VMIN
# define VMIN -1
#endif
CND(VMIN, "Read min chars")
#ifndef VSWTC
# define VSWTC -1
#endif
CND(VSWTC, "Switch")
#ifndef VSTART
# define VSTART -1
#endif
CND(VSTART, "Flow control start")
#ifndef VSTOP
# define VSTOP -1
#endif
CND(VSTOP, "Flow control stop")
#ifndef VSUSP
# define VSUSP -1
#endif
CND(VSUSP, "Suspend")
#ifndef VEOL
# define VEOL -1
#endif
CND(VEOL, "EOL")
#ifndef VREPRINT
# define VREPRINT -1
#endif
CND(VREPRINT, "Reprint unread")
#ifndef VDISCARD
# define VDISCARD -1
#endif
CND(VDISCARD, "Discard pending")
#ifndef VWERASE
# define VWERASE -1
#endif
CND(VWERASE, "Word erase")
#ifndef VLNEXT
# define VLNEXT -1
#endif
CND(VLNEXT, "Literal next")
#ifndef VEOL2
# define VEOL2 -1
#endif
CND(VEOL2, "Alternative EOL")
#endif 
#if defined(__MINGW32__) || defined(__CYGWIN__)
CNU(DTR_CONTROL_ENABLE, "Enable DTR flow ctrl")
CNU(RTS_CONTROL_ENABLE, "Enable RTS flow ctrl")
#endif
#if defined (__FreeBSD__) || defined (__linux__) || defined (__DragonFly__)
# define PTY_Library "-lutil"
#else
# define PTY_Library ""
#endif
CST(PTY_Library, "for g-exptty")
#ifdef HAVE_SOCKETS
#ifndef AF_INET
# define AF_INET -1
#endif
CND(AF_INET, "IPv4 address family")
#if defined(__rtems__)
# undef AF_INET6
#endif
#ifndef AF_INET6
# define AF_INET6 -1
#else
# define HAVE_AF_INET6 1
#endif
CND(AF_INET6, "IPv6 address family")
#ifndef SOCK_STREAM
# define SOCK_STREAM -1
#endif
CND(SOCK_STREAM, "Stream socket")
#ifndef SOCK_DGRAM
# define SOCK_DGRAM -1
#endif
CND(SOCK_DGRAM, "Datagram socket")
#ifndef HOST_NOT_FOUND
# define HOST_NOT_FOUND -1
#endif
CND(HOST_NOT_FOUND, "Unknown host")
#ifndef TRY_AGAIN
# define TRY_AGAIN -1
#endif
CND(TRY_AGAIN, "Host name lookup failure")
#ifndef NO_DATA
# define NO_DATA -1
#endif
CND(NO_DATA, "No data record for name")
#ifndef NO_RECOVERY
# define NO_RECOVERY -1
#endif
CND(NO_RECOVERY, "Non recoverable errors")
#ifndef SHUT_RD
# define SHUT_RD -1
#endif
CND(SHUT_RD, "No more recv")
#ifndef SHUT_WR
# define SHUT_WR -1
#endif
CND(SHUT_WR, "No more send")
#ifndef SHUT_RDWR
# define SHUT_RDWR -1
#endif
CND(SHUT_RDWR, "No more recv/send")
#ifndef SOL_SOCKET
# define SOL_SOCKET -1
#endif
CND(SOL_SOCKET, "Options for socket level")
#ifndef IPPROTO_IP
# define IPPROTO_IP -1
#endif
CND(IPPROTO_IP, "Dummy protocol for IP")
#ifndef IPPROTO_UDP
# define IPPROTO_UDP -1
#endif
CND(IPPROTO_UDP, "UDP")
#ifndef IPPROTO_TCP
# define IPPROTO_TCP -1
#endif
CND(IPPROTO_TCP, "TCP")
#ifndef MSG_OOB
# define MSG_OOB -1
#endif
CND(MSG_OOB, "Process out-of-band data")
#ifndef MSG_PEEK
# define MSG_PEEK -1
#endif
CND(MSG_PEEK, "Peek at incoming data")
#ifndef MSG_EOR
# define MSG_EOR -1
#endif
CND(MSG_EOR, "Send end of record")
#ifndef MSG_WAITALL
#ifdef __MINWGW32__
# define MSG_WAITALL (1 << 3)
#else
# define MSG_WAITALL -1
#endif
#endif
CND(MSG_WAITALL, "Wait for full reception")
#ifndef MSG_NOSIGNAL
# define MSG_NOSIGNAL -1
#endif
CND(MSG_NOSIGNAL, "No SIGPIPE on send")
#if defined (__linux__) || defined (__ANDROID__) || defined (__QNX__)
# define MSG_Forced_Flags "MSG_NOSIGNAL"
#else
# define MSG_Forced_Flags "0"
#endif
CNS(MSG_Forced_Flags, "")
#ifndef TCP_NODELAY
# define TCP_NODELAY -1
#endif
CND(TCP_NODELAY, "Do not coalesce packets")
#ifndef SO_REUSEADDR
# define SO_REUSEADDR -1
#endif
CND(SO_REUSEADDR, "Bind reuse local address")
#ifndef SO_REUSEPORT
# define SO_REUSEPORT -1
#endif
CND(SO_REUSEPORT, "Bind reuse port number")
#ifndef SO_KEEPALIVE
# define SO_KEEPALIVE -1
#endif
CND(SO_KEEPALIVE, "Enable keep-alive msgs")
#ifndef SO_LINGER
# define SO_LINGER -1
#endif
CND(SO_LINGER, "Defer close to flush data")
#ifndef SO_BROADCAST
# define SO_BROADCAST -1
#endif
CND(SO_BROADCAST, "Can send broadcast msgs")
#ifndef SO_SNDBUF
# define SO_SNDBUF -1
#endif
CND(SO_SNDBUF, "Set/get send buffer size")
#ifndef SO_RCVBUF
# define SO_RCVBUF -1
#endif
CND(SO_RCVBUF, "Set/get recv buffer size")
#ifndef SO_SNDTIMEO
# define SO_SNDTIMEO -1
#endif
CND(SO_SNDTIMEO, "Emission timeout")
#ifndef SO_RCVTIMEO
# define SO_RCVTIMEO -1
#endif
CND(SO_RCVTIMEO, "Reception timeout")
#ifndef SO_ERROR
# define SO_ERROR -1
#endif
CND(SO_ERROR, "Get/clear error status")
#ifndef SO_BUSY_POLL
# define SO_BUSY_POLL -1
#endif
CND(SO_BUSY_POLL, "Busy polling")
#ifndef IP_MULTICAST_IF
# define IP_MULTICAST_IF -1
#endif
CND(IP_MULTICAST_IF, "Set/get mcast interface")
#ifndef IP_MULTICAST_TTL
# define IP_MULTICAST_TTL -1
#endif
CND(IP_MULTICAST_TTL, "Set/get multicast TTL")
#ifndef IP_MULTICAST_LOOP
# define IP_MULTICAST_LOOP -1
#endif
CND(IP_MULTICAST_LOOP, "Set/get mcast loopback")
#ifndef IP_ADD_MEMBERSHIP
# define IP_ADD_MEMBERSHIP -1
#endif
CND(IP_ADD_MEMBERSHIP, "Join a multicast group")
#ifndef IP_DROP_MEMBERSHIP
# define IP_DROP_MEMBERSHIP -1
#endif
CND(IP_DROP_MEMBERSHIP, "Leave a multicast group")
#ifndef IP_PKTINFO
# define IP_PKTINFO -1
#endif
CND(IP_PKTINFO, "Get datagram info")
{
struct timeval tv;
#define SIZEOF_tv_sec (sizeof tv.tv_sec)
CND(SIZEOF_tv_sec, "tv_sec")
#define SIZEOF_tv_usec (sizeof tv.tv_usec)
CND(SIZEOF_tv_usec, "tv_usec")
#if defined (__sun__)
# define MAX_tv_sec "100_000_000"
#elif defined (__hpux__)
# define MAX_tv_sec "16#7fffffff#"
#else
# define MAX_tv_sec "2 ** (SIZEOF_tv_sec * 8 - 1) - 1"
#endif
CNS(MAX_tv_sec, "")
}
#define SIZEOF_sockaddr_in (sizeof (struct sockaddr_in))
CND(SIZEOF_sockaddr_in, "struct sockaddr_in")
#ifdef HAVE_AF_INET6
# define SIZEOF_sockaddr_in6 (sizeof (struct sockaddr_in6))
#else
# define SIZEOF_sockaddr_in6 0
#endif
CND(SIZEOF_sockaddr_in6, "struct sockaddr_in6")
#define SIZEOF_fd_set (sizeof (fd_set))
CND(SIZEOF_fd_set, "fd_set")
CND(FD_SETSIZE, "Max fd value")
#define SIZEOF_struct_hostent (sizeof (struct hostent))
CND(SIZEOF_struct_hostent, "struct hostent")
#define SIZEOF_struct_servent (sizeof (struct servent))
CND(SIZEOF_struct_servent, "struct servent")
#if defined (__linux__) || defined (__ANDROID__) || defined (__QNX__)
#define SIZEOF_sigset (sizeof (sigset_t))
CND(SIZEOF_sigset, "sigset")
#endif
#if defined (__sun__) || defined (__hpux__)
# define Msg_Iovlen_T "int"
#else
# define Msg_Iovlen_T "size_t"
#endif
SUB(Msg_Iovlen_T)
CND(Need_Netdb_Buffer, "Need buffer for Netdb ops")
CND(Need_Netdb_Lock,   "Need lock for Netdb ops")
CND(Has_Sockaddr_Len,  "Sockaddr has sa_len field")
C("Thread_Blocking_IO", Boolean, "True", "")
#ifdef HAVE_INET_PTON
# define Inet_Pton_Linkname "inet_pton"
#else
# define Inet_Pton_Linkname "__gnat_inet_pton"
#endif
CST(Inet_Pton_Linkname, "")
#endif 
#if !(defined(CLOCK_REALTIME) || defined (__hpux__))
# define CLOCK_REALTIME (-1)
#endif
CND(CLOCK_REALTIME, "System realtime clock")
#ifdef CLOCK_MONOTONIC
CND(CLOCK_MONOTONIC, "System monotonic clock")
#endif
#ifdef CLOCK_FASTEST
CND(CLOCK_FASTEST, "Fastest clock")
#endif
#ifndef CLOCK_THREAD_CPUTIME_ID
# define CLOCK_THREAD_CPUTIME_ID -1
#endif
CND(CLOCK_THREAD_CPUTIME_ID, "Thread CPU clock")
#if defined(__linux__) || defined(__FreeBSD__) \
|| (defined(_AIX) && defined(_AIXVERSION_530)) \
|| defined(__DragonFly__) || defined(__QNX__)
# define CLOCK_RT_Ada "CLOCK_MONOTONIC"
#else
# define CLOCK_RT_Ada "CLOCK_REALTIME"
#endif
#ifdef CLOCK_RT_Ada
CNS(CLOCK_RT_Ada, "")
#endif
#if defined (__APPLE__) || defined (__linux__) || defined (__ANDROID__) \
|| defined (__QNX__) || defined (__rtems__) || defined (DUMMY)
#if defined (__APPLE__) || defined (DUMMY)
#define PTHREAD_SIZE            __PTHREAD_SIZE__
#define PTHREAD_ATTR_SIZE       __PTHREAD_ATTR_SIZE__
#define PTHREAD_MUTEXATTR_SIZE  __PTHREAD_MUTEXATTR_SIZE__
#define PTHREAD_MUTEX_SIZE      __PTHREAD_MUTEX_SIZE__
#define PTHREAD_CONDATTR_SIZE   __PTHREAD_CONDATTR_SIZE__
#define PTHREAD_COND_SIZE       __PTHREAD_COND_SIZE__
#define PTHREAD_RWLOCKATTR_SIZE __PTHREAD_RWLOCKATTR_SIZE__
#define PTHREAD_RWLOCK_SIZE     __PTHREAD_RWLOCK_SIZE__
#define PTHREAD_ONCE_SIZE       __PTHREAD_ONCE_SIZE__
#else
#define PTHREAD_SIZE            (sizeof (pthread_t))
#define PTHREAD_ATTR_SIZE       (sizeof (pthread_attr_t))
#define PTHREAD_MUTEXATTR_SIZE  (sizeof (pthread_mutexattr_t))
#define PTHREAD_MUTEX_SIZE      (sizeof (pthread_mutex_t))
#define PTHREAD_CONDATTR_SIZE   (sizeof (pthread_condattr_t))
#define PTHREAD_COND_SIZE       (sizeof (pthread_cond_t))
#define PTHREAD_RWLOCKATTR_SIZE (sizeof (pthread_rwlockattr_t))
#define PTHREAD_RWLOCK_SIZE     (sizeof (pthread_rwlock_t))
#define PTHREAD_ONCE_SIZE       (sizeof (pthread_once_t))
#endif
CND(PTHREAD_SIZE,            "pthread_t")
CND(PTHREAD_ATTR_SIZE,       "pthread_attr_t")
CND(PTHREAD_MUTEXATTR_SIZE,  "pthread_mutexattr_t")
CND(PTHREAD_MUTEX_SIZE,      "pthread_mutex_t")
CND(PTHREAD_CONDATTR_SIZE,   "pthread_condattr_t")
CND(PTHREAD_COND_SIZE,       "pthread_cond_t")
CND(PTHREAD_RWLOCKATTR_SIZE, "pthread_rwlockattr_t")
CND(PTHREAD_RWLOCK_SIZE,     "pthread_rwlock_t")
CND(PTHREAD_ONCE_SIZE,       "pthread_once_t")
#endif 
#define SIZEOF_struct_file_attributes (sizeof (struct file_attributes))
CND(SIZEOF_struct_file_attributes, "struct file_attributes")
{
struct dirent dent;
#define SIZEOF_struct_dirent_alloc \
((char*) &dent.d_name - (char*) &dent) + NAME_MAX + 1
CND(SIZEOF_struct_dirent_alloc, "struct dirent allocation")
}
#if defined (__vxworks) || defined (DUMMY)
CND(OK,    "VxWorks generic success")
CND(ERROR, "VxWorks generic error")
#endif 
#if defined (__MINGW32__) || defined (DUMMY)
CND(WSASYSNOTREADY,     "System not ready")
CND(WSAVERNOTSUPPORTED, "Version not supported")
CND(WSANOTINITIALISED,  "Winsock not initialized")
CND(WSAEDISCON,         "Disconnected")
#endif 
#ifdef NATIVE
putchar ('\n');
#endif
}
