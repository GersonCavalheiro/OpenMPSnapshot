typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;
typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;
typedef __int8_t __int_least8_t;
typedef __uint8_t __uint_least8_t;
typedef __int16_t __int_least16_t;
typedef __uint16_t __uint_least16_t;
typedef __int32_t __int_least32_t;
typedef __uint32_t __uint_least32_t;
typedef __int64_t __int_least64_t;
typedef __uint64_t __uint_least64_t;
typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
typedef long int __intmax_t;
typedef unsigned long int __uintmax_t;
typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
struct named_avilib_c_317
{
int __val[2];
};
typedef struct named_avilib_c_317 __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;
typedef int __daddr_t;
typedef int __key_t;
typedef int __clockid_t;
typedef void * __timer_t;
typedef long int __blksize_t;
typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;
typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;
typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;
typedef long int __fsword_t;
typedef long int __ssize_t;
typedef long int __syscall_slong_t;
typedef unsigned long int __syscall_ulong_t;
typedef __off64_t __loff_t;
typedef char * __caddr_t;
typedef long int __intptr_t;
typedef unsigned int __socklen_t;
typedef int __sig_atomic_t;
typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;
typedef __loff_t loff_t;
typedef __ino_t ino_t;
typedef __dev_t dev_t;
typedef __gid_t gid_t;
typedef __mode_t mode_t;
typedef __nlink_t nlink_t;
typedef __uid_t uid_t;
typedef __off_t off_t;
typedef __pid_t pid_t;
typedef __id_t id_t;
typedef __ssize_t ssize_t;
typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;
typedef __key_t key_t;
typedef __clock_t clock_t;
typedef __clockid_t clockid_t;
typedef __time_t time_t;
typedef __timer_t timer_t;
typedef long unsigned int size_t;
typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
typedef unsigned int u_int8_t;
typedef unsigned int u_int16_t;
typedef unsigned int u_int32_t;
typedef unsigned int u_int64_t;
typedef int register_t;
static inline __uint16_t __bswap_16(__uint16_t __bsx)
{
__uint16_t _ret_val_0;
_ret_val_0=__builtin_bswap16(__bsx);
return _ret_val_0;
}
static inline __uint32_t __bswap_32(__uint32_t __bsx)
{
__uint32_t _ret_val_0;
_ret_val_0=__builtin_bswap32(__bsx);
return _ret_val_0;
}
static inline __uint64_t __bswap_64(__uint64_t __bsx)
{
__uint64_t _ret_val_0;
_ret_val_0=__builtin_bswap64(__bsx);
return _ret_val_0;
}
static inline __uint16_t __uint16_identity(__uint16_t __x)
{
return __x;
}
static inline __uint32_t __uint32_identity(__uint32_t __x)
{
return __x;
}
static inline __uint64_t __uint64_identity(__uint64_t __x)
{
return __x;
}
struct named_avilib_c_1066
{
unsigned long int __val[(1024/(8*sizeof (unsigned long int)))];
};
typedef struct named_avilib_c_1066 __sigset_t;
typedef __sigset_t sigset_t;
struct timeval
{
__time_t tv_sec;
__suseconds_t tv_usec;
};
struct timespec
{
__time_t tv_sec;
__syscall_slong_t tv_nsec;
};
typedef __suseconds_t suseconds_t;
typedef long int __fd_mask;
struct named_avilib_c_1161
{
__fd_mask __fds_bits[(1024/(8*((int)sizeof (__fd_mask))))];
};
typedef struct named_avilib_c_1161 fd_set;
typedef __fd_mask fd_mask;
extern int select(int __nfds, fd_set * __readfds, fd_set * __writefds, fd_set * __exceptfds, struct timeval * __timeout);
extern int pselect(int __nfds, fd_set * __readfds, fd_set * __writefds, fd_set * __exceptfds, const struct timespec * __timeout, const __sigset_t * __sigmask);
typedef __blksize_t blksize_t;
typedef __blkcnt_t blkcnt_t;
typedef __fsblkcnt_t fsblkcnt_t;
typedef __fsfilcnt_t fsfilcnt_t;
struct __pthread_rwlock_arch_t
{
unsigned int __readers;
unsigned int __writers;
unsigned int __wrphase_futex;
unsigned int __writers_futex;
unsigned int __pad3;
unsigned int __pad4;
int __cur_writer;
int __shared;
signed char __rwelision;
unsigned char __pad1[7];
unsigned long int __pad2;
unsigned int __flags;
};
struct __pthread_internal_list
{
struct __pthread_internal_list * __prev;
struct __pthread_internal_list * __next;
};
typedef struct __pthread_internal_list __pthread_list_t;
struct __pthread_mutex_s
{
int __lock;
unsigned int __count;
int __owner;
unsigned int __nusers;
int __kind;
short __spins;
short __elision;
__pthread_list_t __list;
};
struct named_avilib_c_1514
{
unsigned int __low;
unsigned int __high;
};
union named_avilib_c_1499
{
unsigned long long int __wseq;
struct named_avilib_c_1514 __wseq32;
};
struct named_avilib_c_1553
{
unsigned int __low;
unsigned int __high;
};
union named_avilib_c_1538
{
unsigned long long int __g1_start;
struct named_avilib_c_1553 __g1_start32;
};
struct __pthread_cond_s
{
union named_avilib_c_1499 ;
union named_avilib_c_1538 ;
unsigned int __g_refs[2];
unsigned int __g_size[2];
unsigned int __g1_orig_size;
unsigned int __wrefs;
unsigned int __g_signals[2];
};
typedef unsigned long int pthread_t;
union named_avilib_c_1635
{
char __size[4];
int __align;
};
typedef union named_avilib_c_1635 pthread_mutexattr_t;
union named_avilib_c_1657
{
char __size[4];
int __align;
};
typedef union named_avilib_c_1657 pthread_condattr_t;
typedef unsigned int pthread_key_t;
typedef int pthread_once_t;
union pthread_attr_t
{
char __size[56];
long int __align;
};
typedef union pthread_attr_t pthread_attr_t;
union named_avilib_c_1726
{
struct __pthread_mutex_s __data;
char __size[40];
long int __align;
};
typedef union named_avilib_c_1726 pthread_mutex_t;
union named_avilib_c_1757
{
struct __pthread_cond_s __data;
char __size[48];
long long int __align;
};
typedef union named_avilib_c_1757 pthread_cond_t;
union named_avilib_c_1791
{
struct __pthread_rwlock_arch_t __data;
char __size[56];
long int __align;
};
typedef union named_avilib_c_1791 pthread_rwlock_t;
union named_avilib_c_1822
{
char __size[8];
long int __align;
};
typedef union named_avilib_c_1822 pthread_rwlockattr_t;
typedef volatile int pthread_spinlock_t;
union named_avilib_c_1855
{
char __size[32];
long int __align;
};
typedef union named_avilib_c_1855 pthread_barrier_t;
union named_avilib_c_1879
{
char __size[4];
int __align;
};
typedef union named_avilib_c_1879 pthread_barrierattr_t;
struct stat
{
__dev_t st_dev;
__ino_t st_ino;
__nlink_t st_nlink;
__mode_t st_mode;
__uid_t st_uid;
__gid_t st_gid;
int __pad0;
__dev_t st_rdev;
__off_t st_size;
__blksize_t st_blksize;
__blkcnt_t st_blocks;
struct timespec st_atim;
struct timespec st_mtim;
struct timespec st_ctim;
__syscall_slong_t __glibc_reserved[3];
};
struct timespec;
extern int stat(const char * __file, struct stat * __buf);
extern int fstat(int __fd, struct stat * __buf);
extern int fstatat(int __fd, const char * __file, struct stat * __buf, int __flag);
extern int lstat(const char * __file, struct stat * __buf);
extern int chmod(const char * __file, __mode_t __mode);
extern int lchmod(const char * __file, __mode_t __mode);
extern int fchmod(int __fd, __mode_t __mode);
extern int fchmodat(int __fd, const char * __file, __mode_t __mode, int __flag);
extern __mode_t umask(__mode_t __mask);
extern int mkdir(const char * __path, __mode_t __mode);
extern int mkdirat(int __fd, const char * __path, __mode_t __mode);
extern int mknod(const char * __path, __mode_t __mode, __dev_t __dev);
extern int mknodat(int __fd, const char * __path, __mode_t __mode, __dev_t __dev);
extern int mkfifo(const char * __path, __mode_t __mode);
extern int mkfifoat(int __fd, const char * __path, __mode_t __mode);
extern int utimensat(int __fd, const char * __path, const struct timespec __times[2], int __flags);
extern int futimens(int __fd, const struct timespec __times[2]);
extern int __fxstat(int __ver, int __fildes, struct stat * __stat_buf);
extern int __xstat(int __ver, const char * __filename, struct stat * __stat_buf);
extern int __lxstat(int __ver, const char * __filename, struct stat * __stat_buf);
extern int __fxstatat(int __ver, int __fildes, const char * __filename, struct stat * __stat_buf, int __flag);
extern int __xmknod(int __ver, const char * __path, __mode_t __mode, __dev_t * __dev);
extern int __xmknodat(int __ver, int __fd, const char * __path, __mode_t __mode, __dev_t * __dev);
typedef __builtin_va_list __gnuc_va_list;
union named_avilib_c_3115
{
unsigned int __wch;
char __wchb[4];
};
struct named_avilib_c_3107
{
int __count;
union named_avilib_c_3115 __value;
};
typedef struct named_avilib_c_3107 __mbstate_t;
struct _G_fpos_t
{
__off_t __pos;
__mbstate_t __state;
};
typedef struct _G_fpos_t __fpos_t;
struct _G_fpos64_t
{
__off64_t __pos;
__mbstate_t __state;
};
typedef struct _G_fpos64_t __fpos64_t;
struct _IO_FILE;
typedef struct _IO_FILE __FILE;
struct _IO_FILE;
typedef struct _IO_FILE FILE;
struct _IO_FILE;
struct _IO_marker;
struct _IO_codecvt;
struct _IO_wide_data;
typedef void _IO_lock_t;
struct _IO_FILE
{
int _flags;
char * _IO_read_ptr;
char * _IO_read_end;
char * _IO_read_base;
char * _IO_write_base;
char * _IO_write_ptr;
char * _IO_write_end;
char * _IO_buf_base;
char * _IO_buf_end;
char * _IO_save_base;
char * _IO_backup_base;
char * _IO_save_end;
struct _IO_marker * _markers;
struct _IO_FILE * _chain;
int _fileno;
int _flags2;
__off_t _old_offset;
unsigned short _cur_column;
signed char _vtable_offset;
char _shortbuf[1];
_IO_lock_t * _lock;
__off64_t _offset;
struct _IO_codecvt * _codecvt;
struct _IO_wide_data * _wide_data;
struct _IO_FILE * _freeres_list;
void * _freeres_buf;
size_t __pad5;
int _mode;
char _unused2[(((15*sizeof (int))-(4*sizeof (void * )))-sizeof (size_t))];
};
struct _IO_FILE;
typedef __gnuc_va_list va_list;
typedef __fpos_t fpos_t;
extern FILE * stdin;
extern FILE * stdout;
extern FILE * stderr;
extern int remove(const char * __filename);
extern int rename(const char * __old, const char * __new);
extern int renameat(int __oldfd, const char * __old, int __newfd, const char * __new);
extern FILE *tmpfile(void );
extern char *tmpnam(char * __s);
extern char *tmpnam_r(char * __s);
extern char *tempnam(const char * __dir, const char * __pfx);
extern int fclose(FILE * __stream);
extern int fflush(FILE * __stream);
extern int fflush_unlocked(FILE * __stream);
extern FILE *fopen(const char * __filename, const char * __modes);
extern FILE *freopen(const char * __filename, const char * __modes, FILE * __stream);
extern FILE *fdopen(int __fd, const char * __modes);
extern FILE *fmemopen(void * __s, size_t __len, const char * __modes);
extern FILE *open_memstream(char * * __bufloc, size_t * __sizeloc);
extern void setbuf(FILE * __stream, char * __buf);
extern int setvbuf(FILE * __stream, char * __buf, int __modes, size_t __n);
extern void setbuffer(FILE * __stream, char * __buf, size_t __size);
extern void setlinebuf(FILE * __stream);
extern int fprintf(FILE * __stream, const char * __format,  ...);
extern int printf(const char * __format,  ...);
extern int sprintf(char * __s, const char * __format,  ...);
extern int vfprintf(FILE * __s, const char * __format, __gnuc_va_list __arg);
extern int vprintf(const char * __format, __gnuc_va_list __arg);
extern int vsprintf(char * __s, const char * __format, __gnuc_va_list __arg);
extern int snprintf(char * __s, size_t __maxlen, const char * __format,  ...);
extern int vsnprintf(char * __s, size_t __maxlen, const char * __format, __gnuc_va_list __arg);
extern int vdprintf(int __fd, const char * __fmt, __gnuc_va_list __arg);
extern int dprintf(int __fd, const char * __fmt,  ...);
extern int fscanf(FILE * __stream, const char * __format,  ...);
extern int scanf(const char * __format,  ...);
extern int sscanf(const char * __s, const char * __format,  ...);
extern int fscanf(FILE * __stream, const char * __format,  ...);
extern int scanf(const char * __format,  ...);
extern int sscanf(const char * __s, const char * __format,  ...);
extern int vfscanf(FILE * __s, const char * __format, __gnuc_va_list __arg);
extern int vscanf(const char * __format, __gnuc_va_list __arg);
extern int vsscanf(const char * __s, const char * __format, __gnuc_va_list __arg);
extern int vfscanf(FILE * __s, const char * __format, __gnuc_va_list __arg);
extern int vscanf(const char * __format, __gnuc_va_list __arg);
extern int vsscanf(const char * __s, const char * __format, __gnuc_va_list __arg);
extern int fgetc(FILE * __stream);
extern int getc(FILE * __stream);
extern int getchar(void );
extern int getc_unlocked(FILE * __stream);
extern int getchar_unlocked(void );
extern int fgetc_unlocked(FILE * __stream);
extern int fputc(int __c, FILE * __stream);
extern int putc(int __c, FILE * __stream);
extern int putchar(int __c);
extern int fputc_unlocked(int __c, FILE * __stream);
extern int putc_unlocked(int __c, FILE * __stream);
extern int putchar_unlocked(int __c);
extern int getw(FILE * __stream);
extern int putw(int __w, FILE * __stream);
extern char *fgets(char * __s, int __n, FILE * __stream);
extern __ssize_t __getdelim(char * * __lineptr, size_t * __n, int __delimiter, FILE * __stream);
extern __ssize_t getdelim(char * * __lineptr, size_t * __n, int __delimiter, FILE * __stream);
extern __ssize_t getline(char * * __lineptr, size_t * __n, FILE * __stream);
extern int fputs(const char * __s, FILE * __stream);
extern int puts(const char * __s);
extern int ungetc(int __c, FILE * __stream);
extern size_t fread(void * __ptr, size_t __size, size_t __n, FILE * __stream);
extern size_t fwrite(const void * __ptr, size_t __size, size_t __n, FILE * __s);
extern size_t fread_unlocked(void * __ptr, size_t __size, size_t __n, FILE * __stream);
extern size_t fwrite_unlocked(const void * __ptr, size_t __size, size_t __n, FILE * __stream);
extern int fseek(FILE * __stream, long int __off, int __whence);
extern long int ftell(FILE * __stream);
extern void rewind(FILE * __stream);
extern int fseeko(FILE * __stream, __off_t __off, int __whence);
extern __off_t ftello(FILE * __stream);
extern int fgetpos(FILE * __stream, fpos_t * __pos);
extern int fsetpos(FILE * __stream, const fpos_t * __pos);
extern void clearerr(FILE * __stream);
extern int feof(FILE * __stream);
extern int ferror(FILE * __stream);
extern void clearerr_unlocked(FILE * __stream);
extern int feof_unlocked(FILE * __stream);
extern int ferror_unlocked(FILE * __stream);
extern void perror(const char * __s);
extern int sys_nerr;
extern const char * const sys_errlist[];
extern int fileno(FILE * __stream);
extern int fileno_unlocked(FILE * __stream);
extern FILE *popen(const char * __command, const char * __modes);
extern int pclose(FILE * __stream);
extern char *ctermid(char * __s);
extern void flockfile(FILE * __stream);
extern int ftrylockfile(FILE * __stream);
extern void funlockfile(FILE * __stream);
extern int __uflow(FILE * );
extern int __overflow(FILE * , int );
struct flock
{
short int l_type;
short int l_whence;
__off_t l_start;
__off_t l_len;
__pid_t l_pid;
};
extern int fcntl(int __fd, int __cmd,  ...);
extern int open(const char * __file, int __oflag,  ...);
extern int openat(int __fd, const char * __file, int __oflag,  ...);
extern int creat(const char * __file, mode_t __mode);
extern int lockf(int __fd, int __cmd, off_t __len);
extern int posix_fadvise(int __fd, off_t __offset, off_t __len, int __advise);
extern int posix_fallocate(int __fd, off_t __offset, off_t __len);
typedef __useconds_t useconds_t;
typedef __intptr_t intptr_t;
typedef __socklen_t socklen_t;
extern int access(const char * __name, int __type);
extern int faccessat(int __fd, const char * __file, int __type, int __flag);
extern __off_t lseek(int __fd, __off_t __offset, int __whence);
extern int close(int __fd);
extern ssize_t read(int __fd, void * __buf, size_t __nbytes);
extern ssize_t write(int __fd, const void * __buf, size_t __n);
extern ssize_t pread(int __fd, void * __buf, size_t __nbytes, __off_t __offset);
extern ssize_t pwrite(int __fd, const void * __buf, size_t __n, __off_t __offset);
extern int pipe(int __pipedes[2]);
extern unsigned int alarm(unsigned int __seconds);
extern unsigned int sleep(unsigned int __seconds);
extern __useconds_t ualarm(__useconds_t __value, __useconds_t __interval);
extern int usleep(__useconds_t __useconds);
extern int pause(void );
extern int chown(const char * __file, __uid_t __owner, __gid_t __group);
extern int fchown(int __fd, __uid_t __owner, __gid_t __group);
extern int lchown(const char * __file, __uid_t __owner, __gid_t __group);
extern int fchownat(int __fd, const char * __file, __uid_t __owner, __gid_t __group, int __flag);
extern int chdir(const char * __path);
extern int fchdir(int __fd);
extern char *getcwd(char * __buf, size_t __size);
extern char *getwd(char * __buf);
extern int dup(int __fd);
extern int dup2(int __fd, int __fd2);
extern char * * __environ;
extern int execve(const char * __path, char * const __argv[], char * const __envp[]);
extern int fexecve(int __fd, char * const __argv[], char * const __envp[]);
extern int execv(const char * __path, char * const __argv[]);
extern int execle(const char * __path, const char * __arg,  ...);
extern int execl(const char * __path, const char * __arg,  ...);
extern int execvp(const char * __file, char * const __argv[]);
extern int execlp(const char * __file, const char * __arg,  ...);
extern int nice(int __inc);
extern void _exit(int __status);
enum avilib_c_7418 { _PC_LINK_MAX, _PC_MAX_CANON, _PC_MAX_INPUT, _PC_NAME_MAX, _PC_PATH_MAX, _PC_PIPE_BUF, _PC_CHOWN_RESTRICTED, _PC_NO_TRUNC, _PC_VDISABLE, _PC_SYNC_IO, _PC_ASYNC_IO, _PC_PRIO_IO, _PC_SOCK_MAXBUF, _PC_FILESIZEBITS, _PC_REC_INCR_XFER_SIZE, _PC_REC_MAX_XFER_SIZE, _PC_REC_MIN_XFER_SIZE, _PC_REC_XFER_ALIGN, _PC_ALLOC_SIZE_MIN, _PC_SYMLINK_MAX, _PC_2_SYMLINKS };
enum avilib_c_7485 { _SC_ARG_MAX, _SC_CHILD_MAX, _SC_CLK_TCK, _SC_NGROUPS_MAX, _SC_OPEN_MAX, _SC_STREAM_MAX, _SC_TZNAME_MAX, _SC_JOB_CONTROL, _SC_SAVED_IDS, _SC_REALTIME_SIGNALS, _SC_PRIORITY_SCHEDULING, _SC_TIMERS, _SC_ASYNCHRONOUS_IO, _SC_PRIORITIZED_IO, _SC_SYNCHRONIZED_IO, _SC_FSYNC, _SC_MAPPED_FILES, _SC_MEMLOCK, _SC_MEMLOCK_RANGE, _SC_MEMORY_PROTECTION, _SC_MESSAGE_PASSING, _SC_SEMAPHORES, _SC_SHARED_MEMORY_OBJECTS, _SC_AIO_LISTIO_MAX, _SC_AIO_MAX, _SC_AIO_PRIO_DELTA_MAX, _SC_DELAYTIMER_MAX, _SC_MQ_OPEN_MAX, _SC_MQ_PRIO_MAX, _SC_VERSION, _SC_PAGESIZE, _SC_RTSIG_MAX, _SC_SEM_NSEMS_MAX, _SC_SEM_VALUE_MAX, _SC_SIGQUEUE_MAX, _SC_TIMER_MAX, _SC_BC_BASE_MAX, _SC_BC_DIM_MAX, _SC_BC_SCALE_MAX, _SC_BC_STRING_MAX, _SC_COLL_WEIGHTS_MAX, _SC_EQUIV_CLASS_MAX, _SC_EXPR_NEST_MAX, _SC_LINE_MAX, _SC_RE_DUP_MAX, _SC_CHARCLASS_NAME_MAX, _SC_2_VERSION, _SC_2_C_BIND, _SC_2_C_DEV, _SC_2_FORT_DEV, _SC_2_FORT_RUN, _SC_2_SW_DEV, _SC_2_LOCALEDEF, _SC_PII, _SC_PII_XTI, _SC_PII_SOCKET, _SC_PII_INTERNET, _SC_PII_OSI, _SC_POLL, _SC_SELECT, _SC_UIO_MAXIOV, _SC_IOV_MAX = _SC_UIO_MAXIOV, _SC_PII_INTERNET_STREAM, _SC_PII_INTERNET_DGRAM, _SC_PII_OSI_COTS, _SC_PII_OSI_CLTS, _SC_PII_OSI_M, _SC_T_IOV_MAX, _SC_THREADS, _SC_THREAD_SAFE_FUNCTIONS, _SC_GETGR_R_SIZE_MAX, _SC_GETPW_R_SIZE_MAX, _SC_LOGIN_NAME_MAX, _SC_TTY_NAME_MAX, _SC_THREAD_DESTRUCTOR_ITERATIONS, _SC_THREAD_KEYS_MAX, _SC_THREAD_STACK_MIN, _SC_THREAD_THREADS_MAX, _SC_THREAD_ATTR_STACKADDR, _SC_THREAD_ATTR_STACKSIZE, _SC_THREAD_PRIORITY_SCHEDULING, _SC_THREAD_PRIO_INHERIT, _SC_THREAD_PRIO_PROTECT, _SC_THREAD_PROCESS_SHARED, _SC_NPROCESSORS_CONF, _SC_NPROCESSORS_ONLN, _SC_PHYS_PAGES, _SC_AVPHYS_PAGES, _SC_ATEXIT_MAX, _SC_PASS_MAX, _SC_XOPEN_VERSION, _SC_XOPEN_XCU_VERSION, _SC_XOPEN_UNIX, _SC_XOPEN_CRYPT, _SC_XOPEN_ENH_I18N, _SC_XOPEN_SHM, _SC_2_CHAR_TERM, _SC_2_C_VERSION, _SC_2_UPE, _SC_XOPEN_XPG2, _SC_XOPEN_XPG3, _SC_XOPEN_XPG4, _SC_CHAR_BIT, _SC_CHAR_MAX, _SC_CHAR_MIN, _SC_INT_MAX, _SC_INT_MIN, _SC_LONG_BIT, _SC_WORD_BIT, _SC_MB_LEN_MAX, _SC_NZERO, _SC_SSIZE_MAX, _SC_SCHAR_MAX, _SC_SCHAR_MIN, _SC_SHRT_MAX, _SC_SHRT_MIN, _SC_UCHAR_MAX, _SC_UINT_MAX, _SC_ULONG_MAX, _SC_USHRT_MAX, _SC_NL_ARGMAX, _SC_NL_LANGMAX, _SC_NL_MSGMAX, _SC_NL_NMAX, _SC_NL_SETMAX, _SC_NL_TEXTMAX, _SC_XBS5_ILP32_OFF32, _SC_XBS5_ILP32_OFFBIG, _SC_XBS5_LP64_OFF64, _SC_XBS5_LPBIG_OFFBIG, _SC_XOPEN_LEGACY, _SC_XOPEN_REALTIME, _SC_XOPEN_REALTIME_THREADS, _SC_ADVISORY_INFO, _SC_BARRIERS, _SC_BASE, _SC_C_LANG_SUPPORT, _SC_C_LANG_SUPPORT_R, _SC_CLOCK_SELECTION, _SC_CPUTIME, _SC_THREAD_CPUTIME, _SC_DEVICE_IO, _SC_DEVICE_SPECIFIC, _SC_DEVICE_SPECIFIC_R, _SC_FD_MGMT, _SC_FIFO, _SC_PIPE, _SC_FILE_ATTRIBUTES, _SC_FILE_LOCKING, _SC_FILE_SYSTEM, _SC_MONOTONIC_CLOCK, _SC_MULTI_PROCESS, _SC_SINGLE_PROCESS, _SC_NETWORKING, _SC_READER_WRITER_LOCKS, _SC_SPIN_LOCKS, _SC_REGEXP, _SC_REGEX_VERSION, _SC_SHELL, _SC_SIGNALS, _SC_SPAWN, _SC_SPORADIC_SERVER, _SC_THREAD_SPORADIC_SERVER, _SC_SYSTEM_DATABASE, _SC_SYSTEM_DATABASE_R, _SC_TIMEOUTS, _SC_TYPED_MEMORY_OBJECTS, _SC_USER_GROUPS, _SC_USER_GROUPS_R, _SC_2_PBS, _SC_2_PBS_ACCOUNTING, _SC_2_PBS_LOCATE, _SC_2_PBS_MESSAGE, _SC_2_PBS_TRACK, _SC_SYMLOOP_MAX, _SC_STREAMS, _SC_2_PBS_CHECKPOINT, _SC_V6_ILP32_OFF32, _SC_V6_ILP32_OFFBIG, _SC_V6_LP64_OFF64, _SC_V6_LPBIG_OFFBIG, _SC_HOST_NAME_MAX, _SC_TRACE, _SC_TRACE_EVENT_FILTER, _SC_TRACE_INHERIT, _SC_TRACE_LOG, _SC_LEVEL1_ICACHE_SIZE, _SC_LEVEL1_ICACHE_ASSOC, _SC_LEVEL1_ICACHE_LINESIZE, _SC_LEVEL1_DCACHE_SIZE, _SC_LEVEL1_DCACHE_ASSOC, _SC_LEVEL1_DCACHE_LINESIZE, _SC_LEVEL2_CACHE_SIZE, _SC_LEVEL2_CACHE_ASSOC, _SC_LEVEL2_CACHE_LINESIZE, _SC_LEVEL3_CACHE_SIZE, _SC_LEVEL3_CACHE_ASSOC, _SC_LEVEL3_CACHE_LINESIZE, _SC_LEVEL4_CACHE_SIZE, _SC_LEVEL4_CACHE_ASSOC, _SC_LEVEL4_CACHE_LINESIZE, _SC_IPV6 = (_SC_LEVEL1_ICACHE_SIZE+50), _SC_RAW_SOCKETS, _SC_V7_ILP32_OFF32, _SC_V7_ILP32_OFFBIG, _SC_V7_LP64_OFF64, _SC_V7_LPBIG_OFFBIG, _SC_SS_REPL_MAX, _SC_TRACE_EVENT_NAME_MAX, _SC_TRACE_NAME_MAX, _SC_TRACE_SYS_MAX, _SC_TRACE_USER_EVENT_MAX, _SC_XOPEN_STREAMS, _SC_THREAD_ROBUST_PRIO_INHERIT, _SC_THREAD_ROBUST_PRIO_PROTECT };
enum avilib_c_8142 { _CS_PATH, _CS_V6_WIDTH_RESTRICTED_ENVS, _CS_GNU_LIBC_VERSION, _CS_GNU_LIBPTHREAD_VERSION, _CS_V5_WIDTH_RESTRICTED_ENVS, _CS_V7_WIDTH_RESTRICTED_ENVS, _CS_LFS_CFLAGS = 1000, _CS_LFS_LDFLAGS, _CS_LFS_LIBS, _CS_LFS_LINTFLAGS, _CS_LFS64_CFLAGS, _CS_LFS64_LDFLAGS, _CS_LFS64_LIBS, _CS_LFS64_LINTFLAGS, _CS_XBS5_ILP32_OFF32_CFLAGS = 1100, _CS_XBS5_ILP32_OFF32_LDFLAGS, _CS_XBS5_ILP32_OFF32_LIBS, _CS_XBS5_ILP32_OFF32_LINTFLAGS, _CS_XBS5_ILP32_OFFBIG_CFLAGS, _CS_XBS5_ILP32_OFFBIG_LDFLAGS, _CS_XBS5_ILP32_OFFBIG_LIBS, _CS_XBS5_ILP32_OFFBIG_LINTFLAGS, _CS_XBS5_LP64_OFF64_CFLAGS, _CS_XBS5_LP64_OFF64_LDFLAGS, _CS_XBS5_LP64_OFF64_LIBS, _CS_XBS5_LP64_OFF64_LINTFLAGS, _CS_XBS5_LPBIG_OFFBIG_CFLAGS, _CS_XBS5_LPBIG_OFFBIG_LDFLAGS, _CS_XBS5_LPBIG_OFFBIG_LIBS, _CS_XBS5_LPBIG_OFFBIG_LINTFLAGS, _CS_POSIX_V6_ILP32_OFF32_CFLAGS, _CS_POSIX_V6_ILP32_OFF32_LDFLAGS, _CS_POSIX_V6_ILP32_OFF32_LIBS, _CS_POSIX_V6_ILP32_OFF32_LINTFLAGS, _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS, _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS, _CS_POSIX_V6_ILP32_OFFBIG_LIBS, _CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS, _CS_POSIX_V6_LP64_OFF64_CFLAGS, _CS_POSIX_V6_LP64_OFF64_LDFLAGS, _CS_POSIX_V6_LP64_OFF64_LIBS, _CS_POSIX_V6_LP64_OFF64_LINTFLAGS, _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS, _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS, _CS_POSIX_V6_LPBIG_OFFBIG_LIBS, _CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS, _CS_POSIX_V7_ILP32_OFF32_CFLAGS, _CS_POSIX_V7_ILP32_OFF32_LDFLAGS, _CS_POSIX_V7_ILP32_OFF32_LIBS, _CS_POSIX_V7_ILP32_OFF32_LINTFLAGS, _CS_POSIX_V7_ILP32_OFFBIG_CFLAGS, _CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS, _CS_POSIX_V7_ILP32_OFFBIG_LIBS, _CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS, _CS_POSIX_V7_LP64_OFF64_CFLAGS, _CS_POSIX_V7_LP64_OFF64_LDFLAGS, _CS_POSIX_V7_LP64_OFF64_LIBS, _CS_POSIX_V7_LP64_OFF64_LINTFLAGS, _CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS, _CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS, _CS_POSIX_V7_LPBIG_OFFBIG_LIBS, _CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS, _CS_V6_ENV, _CS_V7_ENV };
extern long int pathconf(const char * __path, int __name);
extern long int fpathconf(int __fd, int __name);
extern long int sysconf(int __name);
extern size_t confstr(int __name, char * __buf, size_t __len);
extern __pid_t getpid(void );
extern __pid_t getppid(void );
extern __pid_t getpgrp(void );
extern __pid_t __getpgid(__pid_t __pid);
extern __pid_t getpgid(__pid_t __pid);
extern int setpgid(__pid_t __pid, __pid_t __pgid);
extern int setpgrp(void );
extern __pid_t setsid(void );
extern __pid_t getsid(__pid_t __pid);
extern __uid_t getuid(void );
extern __uid_t geteuid(void );
extern __gid_t getgid(void );
extern __gid_t getegid(void );
extern int getgroups(int __size, __gid_t __list[]);
extern int setuid(__uid_t __uid);
extern int setreuid(__uid_t __ruid, __uid_t __euid);
extern int seteuid(__uid_t __uid);
extern int setgid(__gid_t __gid);
extern int setregid(__gid_t __rgid, __gid_t __egid);
extern int setegid(__gid_t __gid);
extern __pid_t fork(void );
extern __pid_t vfork(void );
extern char *ttyname(int __fd);
extern int ttyname_r(int __fd, char * __buf, size_t __buflen);
extern int isatty(int __fd);
extern int ttyslot(void );
extern int link(const char * __from, const char * __to);
extern int linkat(int __fromfd, const char * __from, int __tofd, const char * __to, int __flags);
extern int symlink(const char * __from, const char * __to);
extern ssize_t readlink(const char * __path, char * __buf, size_t __len);
extern int symlinkat(const char * __from, int __tofd, const char * __to);
extern ssize_t readlinkat(int __fd, const char * __path, char * __buf, size_t __len);
extern int unlink(const char * __name);
extern int unlinkat(int __fd, const char * __name, int __flag);
extern int rmdir(const char * __path);
extern __pid_t tcgetpgrp(int __fd);
extern int tcsetpgrp(int __fd, __pid_t __pgrp_id);
extern char *getlogin(void );
extern int getlogin_r(char * __name, size_t __name_len);
extern int setlogin(const char * __name);
extern char * optarg;
extern int optind;
extern int opterr;
extern int optopt;
extern int getopt(int ___argc, char * const * ___argv, const char * __shortopts);
extern int gethostname(char * __name, size_t __len);
extern int sethostname(const char * __name, size_t __len);
extern int sethostid(long int __id);
extern int getdomainname(char * __name, size_t __len);
extern int setdomainname(const char * __name, size_t __len);
extern int vhangup(void );
extern int revoke(const char * __file);
extern int profil(unsigned short int * __sample_buffer, size_t __size, size_t __offset, unsigned int __scale);
extern int acct(const char * __name);
extern char *getusershell(void );
extern void endusershell(void );
extern void setusershell(void );
extern int daemon(int __nochdir, int __noclose);
extern int chroot(const char * __path);
extern char *getpass(const char * __prompt);
extern int fsync(int __fd);
extern long int gethostid(void );
extern void sync(void );
extern int getpagesize(void );
extern int getdtablesize(void );
extern int truncate(const char * __file, __off_t __length);
extern int ftruncate(int __fd, __off_t __length);
extern int brk(void * __addr);
extern void *sbrk(intptr_t __delta);
extern long int syscall(long int __sysno,  ...);
extern int fdatasync(int __fildes);
extern char *crypt(const char * __key, const char * __salt);
int getentropy(void * __buffer, size_t __length);
typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;
typedef __int_least8_t int_least8_t;
typedef __int_least16_t int_least16_t;
typedef __int_least32_t int_least32_t;
typedef __int_least64_t int_least64_t;
typedef __uint_least8_t uint_least8_t;
typedef __uint_least16_t uint_least16_t;
typedef __uint_least32_t uint_least32_t;
typedef __uint_least64_t uint_least64_t;
typedef signed char int_fast8_t;
typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
typedef unsigned long int uintptr_t;
typedef __intmax_t intmax_t;
typedef __uintmax_t uintmax_t;
typedef int __gwchar_t;
struct named_avilib_c_10811
{
long int quot;
long int rem;
};
typedef struct named_avilib_c_10811 imaxdiv_t;
extern intmax_t imaxabs(intmax_t __n);
extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom);
extern intmax_t strtoimax(const char * __nptr, char * * __endptr, int __base);
extern uintmax_t strtoumax(const char * __nptr, char * * __endptr, int __base);
extern intmax_t wcstoimax(const __gwchar_t * __nptr, __gwchar_t * * __endptr, int __base);
extern uintmax_t wcstoumax(const __gwchar_t * __nptr, __gwchar_t * * __endptr, int __base);
typedef int wchar_t;
enum avilib_c_11074 { P_ALL, P_PID, P_PGID };
typedef enum avilib_c_11074 idtype_t;
struct named_avilib_c_11091
{
int quot;
int rem;
};
typedef struct named_avilib_c_11091 div_t;
struct named_avilib_c_11110
{
long int quot;
long int rem;
};
typedef struct named_avilib_c_11110 ldiv_t;
struct named_avilib_c_11134
{
long long int quot;
long long int rem;
};
typedef struct named_avilib_c_11134 lldiv_t;
extern size_t __ctype_get_mb_cur_max(void );
extern double atof(const char * __nptr);
extern int atoi(const char * __nptr);
extern long int atol(const char * __nptr);
extern long long int atoll(const char * __nptr);
extern double strtod(const char * __nptr, char * * __endptr);
extern float strtof(const char * __nptr, char * * __endptr);
extern long double strtold(const char * __nptr, char * * __endptr);
extern long int strtol(const char * __nptr, char * * __endptr, int __base);
extern unsigned long int strtoul(const char * __nptr, char * * __endptr, int __base);
extern long long int strtoq(const char * __nptr, char * * __endptr, int __base);
extern unsigned long long int strtouq(const char * __nptr, char * * __endptr, int __base);
extern long long int strtoll(const char * __nptr, char * * __endptr, int __base);
extern unsigned long long int strtoull(const char * __nptr, char * * __endptr, int __base);
extern char *l64a(long int __n);
extern long int a64l(const char * __s);
extern long int random(void );
extern void srandom(unsigned int __seed);
extern char *initstate(unsigned int __seed, char * __statebuf, size_t __statelen);
extern char *setstate(char * __statebuf);
struct random_data
{
int32_t * fptr;
int32_t * rptr;
int32_t * state;
int rand_type;
int rand_deg;
int rand_sep;
int32_t * end_ptr;
};
extern int random_r(struct random_data * __buf, int32_t * __result);
extern int srandom_r(unsigned int __seed, struct random_data * __buf);
extern int initstate_r(unsigned int __seed, char * __statebuf, size_t __statelen, struct random_data * __buf);
extern int setstate_r(char * __statebuf, struct random_data * __buf);
extern int rand(void );
extern void srand(unsigned int __seed);
extern int rand_r(unsigned int * __seed);
extern double drand48(void );
extern double erand48(unsigned short int __xsubi[3]);
extern long int lrand48(void );
extern long int nrand48(unsigned short int __xsubi[3]);
extern long int mrand48(void );
extern long int jrand48(unsigned short int __xsubi[3]);
extern void srand48(long int __seedval);
extern unsigned short int *seed48(unsigned short int __seed16v[3]);
extern void lcong48(unsigned short int __param[7]);
struct drand48_data
{
unsigned short int __x[3];
unsigned short int __old_x[3];
unsigned short int __c;
unsigned short int __init;
unsigned long long int __a;
};
extern int drand48_r(struct drand48_data * __buffer, double * __result);
extern int erand48_r(unsigned short int __xsubi[3], struct drand48_data * __buffer, double * __result);
extern int lrand48_r(struct drand48_data * __buffer, long int * __result);
extern int nrand48_r(unsigned short int __xsubi[3], struct drand48_data * __buffer, long int * __result);
extern int mrand48_r(struct drand48_data * __buffer, long int * __result);
extern int jrand48_r(unsigned short int __xsubi[3], struct drand48_data * __buffer, long int * __result);
extern int srand48_r(long int __seedval, struct drand48_data * __buffer);
extern int seed48_r(unsigned short int __seed16v[3], struct drand48_data * __buffer);
extern int lcong48_r(unsigned short int __param[7], struct drand48_data * __buffer);
extern void *malloc(size_t __size);
extern void *calloc(size_t __nmemb, size_t __size);
extern void *realloc(void * __ptr, size_t __size);
extern void free(void * __ptr);
extern void *alloca(size_t __size);
extern void *valloc(size_t __size);
extern int posix_memalign(void * * __memptr, size_t __alignment, size_t __size);
extern void *aligned_alloc(size_t __alignment, size_t __size);
extern void abort(void );
extern int atexit(void (* __func)(void ));
extern int at_quick_exit(void (* __func)(void ));
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg);
extern void exit(int __status);
extern void quick_exit(int __status);
extern void _Exit(int __status);
extern char *getenv(const char * __name);
extern int putenv(char * __string);
extern int setenv(const char * __name, const char * __value, int __replace);
extern int unsetenv(const char * __name);
extern int clearenv(void );
extern char *mktemp(char * __template);
extern int mkstemp(char * __template);
extern int mkstemps(char * __template, int __suffixlen);
extern char *mkdtemp(char * __template);
extern int system(const char * __command);
extern char *realpath(const char * __name, char * __resolved);
typedef int (* __compar_fn_t)(const void * , const void * );
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar);
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar);
extern int abs(int __x);
extern long int labs(long int __x);
extern long long int llabs(long long int __x);
extern div_t div(int __numer, int __denom);
extern ldiv_t ldiv(long int __numer, long int __denom);
extern lldiv_t lldiv(long long int __numer, long long int __denom);
extern char *ecvt(double __value, int __ndigit, int * __decpt, int * __sign);
extern char *fcvt(double __value, int __ndigit, int * __decpt, int * __sign);
extern char *gcvt(double __value, int __ndigit, char * __buf);
extern char *qecvt(long double __value, int __ndigit, int * __decpt, int * __sign);
extern char *qfcvt(long double __value, int __ndigit, int * __decpt, int * __sign);
extern char *qgcvt(long double __value, int __ndigit, char * __buf);
extern int ecvt_r(double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, size_t __len);
extern int fcvt_r(double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, size_t __len);
extern int qecvt_r(long double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, size_t __len);
extern int qfcvt_r(long double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, size_t __len);
extern int mblen(const char * __s, size_t __n);
extern int mbtowc(wchar_t * __pwc, const char * __s, size_t __n);
extern int wctomb(char * __s, wchar_t __wchar);
extern size_t mbstowcs(wchar_t * __pwcs, const char * __s, size_t __n);
extern size_t wcstombs(char * __s, const wchar_t * __pwcs, size_t __n);
extern int rpmatch(const char * __response);
extern int getsubopt(char * * __optionp, char * const * __tokens, char * * __valuep);
extern int getloadavg(double __loadavg[], int __nelem);
extern void *memcpy(void * __dest, const void * __src, size_t __n);
extern void *memmove(void * __dest, const void * __src, size_t __n);
extern void *memccpy(void * __dest, const void * __src, int __c, size_t __n);
extern void *memset(void * __s, int __c, size_t __n);
extern int memcmp(const void * __s1, const void * __s2, size_t __n);
extern void *memchr(const void * __s, int __c, size_t __n);
extern char *strcpy(char * __dest, const char * __src);
extern char *strncpy(char * __dest, const char * __src, size_t __n);
extern char *strcat(char * __dest, const char * __src);
extern char *strncat(char * __dest, const char * __src, size_t __n);
extern int strcmp(const char * __s1, const char * __s2);
extern int strncmp(const char * __s1, const char * __s2, size_t __n);
extern int strcoll(const char * __s1, const char * __s2);
extern size_t strxfrm(char * __dest, const char * __src, size_t __n);
struct __locale_struct
{
struct __locale_data * __locales[13];
const unsigned short int * __ctype_b;
const int * __ctype_tolower;
const int * __ctype_toupper;
const char * __names[13];
};
struct __locale_data;
typedef struct __locale_struct * __locale_t;
typedef __locale_t locale_t;
extern int strcoll_l(const char * __s1, const char * __s2, locale_t __l);
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, locale_t __l);
extern char *strdup(const char * __s);
extern char *strndup(const char * __string, size_t __n);
extern char *strchr(const char * __s, int __c);
extern char *strrchr(const char * __s, int __c);
extern size_t strcspn(const char * __s, const char * __reject);
extern size_t strspn(const char * __s, const char * __accept);
extern char *strpbrk(const char * __s, const char * __accept);
extern char *strstr(const char * __haystack, const char * __needle);
extern char *strtok(char * __s, const char * __delim);
extern char *__strtok_r(char * __s, const char * __delim, char * * __save_ptr);
extern char *strtok_r(char * __s, const char * __delim, char * * __save_ptr);
extern size_t strlen(const char * __s);
extern size_t strnlen(const char * __string, size_t __maxlen);
extern char *strerror(int __errnum);
extern int strerror_r(int __errnum, char * __buf, size_t __buflen);
extern char *strerror_l(int __errnum, locale_t __l);
extern int bcmp(const void * __s1, const void * __s2, size_t __n);
extern void bcopy(const void * __src, void * __dest, size_t __n);
extern void bzero(void * __s, size_t __n);
extern char *index(const char * __s, int __c);
extern char *rindex(const char * __s, int __c);
extern int ffs(int __i);
extern int ffsl(long int __l);
extern int ffsll(long long int __ll);
extern int strcasecmp(const char * __s1, const char * __s2);
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n);
extern int strcasecmp_l(const char * __s1, const char * __s2, locale_t __loc);
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, locale_t __loc);
extern void explicit_bzero(void * __s, size_t __n);
extern char *strsep(char * * __stringp, const char * __delim);
extern char *strsignal(int __sig);
extern char *__stpcpy(char * __dest, const char * __src);
extern char *stpcpy(char * __dest, const char * __src);
extern char *__stpncpy(char * __dest, const char * __src, size_t __n);
extern char *stpncpy(char * __dest, const char * __src, size_t __n);
extern int *__errno_location(void );
struct named_avilib_c_18166
{
unsigned long key;
unsigned long pos;
unsigned long len;
};
typedef struct named_avilib_c_18166 video_index_entry;
struct named_avilib_c_18196
{
unsigned long pos;
unsigned long len;
unsigned long tot;
};
typedef struct named_avilib_c_18196 audio_index_entry;
struct track_s
{
long a_fmt;
long a_chans;
long a_rate;
long a_bits;
long mp3rate;
long audio_strn;
long audio_bytes;
long audio_chunks;
char audio_tag[4];
long audio_posc;
long audio_posb;
long a_codech_off;
long a_codecf_off;
audio_index_entry * audio_index;
};
typedef struct track_s track_t;
struct named_avilib_c_18311
{
long fdes;
long mode;
long width;
long height;
double fps;
char compressor[8];
char compressor2[8];
long video_strn;
long video_frames;
char video_tag[4];
long video_pos;
unsigned long max_len;
track_t track[8];
unsigned long pos;
long n_idx;
long max_idx;
long v_codech_off;
long v_codecf_off;
unsigned char (* idx)[16];
video_index_entry * video_index;
unsigned long last_pos;
unsigned long last_len;
int must_use_index;
unsigned long movi_start;
int anum;
int aptr;
};
typedef struct named_avilib_c_18311 avi_t;
avi_t *AVI_open_output_file(char * filename);
void AVI_set_video(avi_t * AVI, int width, int height, double fps, char * compressor);
void AVI_set_audio(avi_t * AVI, int channels, long rate, int bits, int format, long mp3rate);
int AVI_write_frame(avi_t * AVI, char * data, long bytes, int keyframe);
int AVI_dup_frame(avi_t * AVI);
int AVI_write_audio(avi_t * AVI, char * data, long bytes);
int AVI_append_audio(avi_t * AVI, char * data, long bytes);
long AVI_bytes_remain(avi_t * AVI);
int AVI_close(avi_t * AVI);
long AVI_bytes_written(avi_t * AVI);
avi_t *AVI_open_input_file(char * filename, int getIndex);
avi_t *AVI_open_fd(int fd, int getIndex);
int avi_parse_input_file(avi_t * AVI, int getIndex);
long AVI_audio_mp3rate(avi_t * AVI);
long AVI_video_frames(avi_t * AVI);
int AVI_video_width(avi_t * AVI);
int AVI_video_height(avi_t * AVI);
double AVI_frame_rate(avi_t * AVI);
char *AVI_video_compressor(avi_t * AVI);
int AVI_audio_channels(avi_t * AVI);
int AVI_audio_bits(avi_t * AVI);
int AVI_audio_format(avi_t * AVI);
long AVI_audio_rate(avi_t * AVI);
long AVI_audio_bytes(avi_t * AVI);
long AVI_audio_chunks(avi_t * AVI);
long AVI_max_video_chunk(avi_t * AVI);
long AVI_frame_size(avi_t * AVI, long frame);
long AVI_audio_size(avi_t * AVI, long frame);
int AVI_seek_start(avi_t * AVI);
int AVI_set_video_position(avi_t * AVI, long frame);
long AVI_get_video_position(avi_t * AVI, long frame);
long AVI_read_frame(avi_t * AVI, char * vidbuf, int * keyframe);
int AVI_set_audio_position(avi_t * AVI, long byte);
int AVI_set_audio_bitrate(avi_t * AVI, long bitrate);
long AVI_read_audio(avi_t * AVI, char * audbuf, long bytes);
long AVI_audio_codech_offset(avi_t * AVI);
long AVI_audio_codecf_offset(avi_t * AVI);
long AVI_video_codech_offset(avi_t * AVI);
long AVI_video_codecf_offset(avi_t * AVI);
int AVI_read_data(avi_t * AVI, char * vidbuf, long max_vidbuf, char * audbuf, long max_audbuf, long * len);
void AVI_print_error(char * str);
char *AVI_strerror();
char *AVI_syserror();
int AVI_scan(char * name);
int AVI_dump(char * name, int mode);
char *AVI_codec2str(short cc);
int AVI_file_check(char * import_file);
void AVI_info(avi_t * avifile);
uint64_t AVI_max_size();
int avi_update_header(avi_t * AVI);
int AVI_set_audio_track(avi_t * AVI, int track);
int AVI_get_audio_track(avi_t * AVI);
int AVI_audio_tracks(avi_t * AVI);
struct riff_struct
{
unsigned char id[4];
unsigned long len;
unsigned char wave_id[4];
};
struct chunk_struct
{
unsigned char id[4];
unsigned long len;
};
struct common_struct
{
unsigned short wFormatTag;
unsigned short wChannels;
unsigned long dwSamplesPerSec;
unsigned long dwAvgBytesPerSec;
unsigned short wBlockAlign;
unsigned short wBitsPerSample;
};
struct wave_header
{
struct riff_struct riff;
struct chunk_struct format;
struct common_struct common;
struct chunk_struct data;
};
struct chunk_struct;
struct AVIStreamHeader
{
long fccType;
long fccHandler;
long dwFlags;
long dwPriority;
long dwInitialFrames;
long dwScale;
long dwRate;
long dwStart;
long dwLength;
long dwSuggestedBufferSize;
long dwQuality;
long dwSampleSize;
};
long AVI_errno;
static char id_str[64];
static size_t avi_read(int fd, char * buf, size_t len)
{
size_t n = 0;
size_t r = 0;
while (r<len)
{
n=read(fd, buf+r, len-r);
if (n<=0)
{
return r;
}
r+=n;
}
return r;
}
static size_t avi_write(int fd, char * buf, size_t len)
{
size_t n = 0;
size_t r = 0;
while (r<len)
{
n=write(fd, buf+r, len-r);
if (n<0)
{
return n;
}
r+=n;
}
return r;
}
static void long2str(unsigned char * dst, int n)
{
dst[0]=(n&255);
dst[1]=((n>>8)&255);
dst[2]=((n>>16)&255);
dst[3]=((n>>24)&255);
return ;
}
static unsigned long str2ulong(unsigned char * str)
{
unsigned long _ret_val_0;
_ret_val_0=(((str[0]|(str[1]<<8))|(str[2]<<16))|(str[3]<<24));
return _ret_val_0;
}
static unsigned long str2ushort(unsigned char * str)
{
unsigned long _ret_val_0;
_ret_val_0=(str[0]|(str[1]<<8));
return _ret_val_0;
}
static int avi_sampsize(avi_t * AVI, int j)
{
int s;
s=(((AVI->track[j].a_bits+7)/8)*AVI->track[j].a_chans);
if (s<4)
{
s=4;
}
return s;
}
static int avi_add_chunk(avi_t * AVI, unsigned char * tag, unsigned char * data, int length)
{
unsigned char c[8];
int _ret_val_0;
memcpy(c, tag, 4);
long2str(c+4, length);
length=((length+1)&( ~ 1));
if ((avi_write(AVI->fdes, (char * )c, 8)!=8)||(avi_write(AVI->fdes, (char * )data, length)!=length))
{
lseek(AVI->fdes, AVI->pos, 0);
AVI_errno=4;
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->pos+=(8+length);
_ret_val_0=0;
return _ret_val_0;
}
static int avi_add_index_entry(avi_t * AVI, unsigned char * tag, long flags, unsigned long pos, unsigned long len)
{
void * ptr;
int _ret_val_0;
if (AVI->n_idx>=AVI->max_idx)
{
ptr=realloc((void * )AVI->idx, (AVI->max_idx+4096)*16);
if (ptr==0)
{
AVI_errno=8;
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->max_idx+=4096;
AVI->idx=((unsigned char ((* )[16]))ptr);
}
memcpy(AVI->idx[AVI->n_idx], tag, 4);
long2str(AVI->idx[AVI->n_idx]+4, flags);
long2str(AVI->idx[AVI->n_idx]+8, pos);
long2str(AVI->idx[AVI->n_idx]+12, len);
AVI->n_idx ++ ;
if (len>AVI->max_len)
{
AVI->max_len=len;
}
_ret_val_0=0;
return _ret_val_0;
}
avi_t *AVI_open_output_file(char * filename)
{
avi_t * AVI;
int i;
int mask = 0;
unsigned char AVI_header[2048];
avi_t * _ret_val_0;
AVI=((avi_t * )malloc(sizeof (avi_t)));
if (AVI==0)
{
AVI_errno=8;
_ret_val_0=0;
return _ret_val_0;
}
memset((void * )AVI, 0, sizeof (avi_t));
AVI->fdes=open(filename, (2|64)|0, 420&( ~ mask));
if (AVI->fdes<0)
{
AVI_errno=2;
free(AVI);
_ret_val_0=0;
return _ret_val_0;
}
#pragma loop name AVI_open_output_file#0 
#pragma cetus parallel 
#pragma omp parallel for
for (i=0; i<2048; i ++ )
{
AVI_header[i]=0;
}
i=avi_write(AVI->fdes, (char * )AVI_header, 2048);
if (i!=2048)
{
close(AVI->fdes);
AVI_errno=4;
free(AVI);
_ret_val_0=0;
return _ret_val_0;
}
AVI->pos=2048;
AVI->mode=0;
AVI->anum=0;
AVI->aptr=0;
return AVI;
}
void AVI_set_video(avi_t * AVI, int width, int height, double fps, char * compressor)
{
if (AVI->mode==1)
{
return ;
}
AVI->width=width;
AVI->height=height;
AVI->fps=fps;
if (strncmp(compressor, "RGB", 3)==0)
{
memset(AVI->compressor, 0, 4);
}
else
{
memcpy(AVI->compressor, compressor, 4);
}
AVI->compressor[4]=0;
avi_update_header(AVI);
return ;
}
void AVI_set_audio(avi_t * AVI, int channels, long rate, int bits, int format, long mp3rate)
{
if (AVI->mode==1)
{
return ;
}
AVI->aptr=AVI->anum;
++ AVI->anum;
if (AVI->anum>8)
{
fprintf(stderr, "error - only %d audio tracks supported\n", 8);
exit(1);
}
AVI->track[AVI->aptr].a_chans=channels;
AVI->track[AVI->aptr].a_rate=rate;
AVI->track[AVI->aptr].a_bits=bits;
AVI->track[AVI->aptr].a_fmt=format;
AVI->track[AVI->aptr].mp3rate=mp3rate;
avi_update_header(AVI);
return ;
}
int avi_update_header(avi_t * AVI)
{
int njunk, sampsize, hasIndex, ms_per_frame, frate, flag;
int movi_len, hdrl_start, strl_start, j;
unsigned char AVI_header[2048];
long nhb;
int _ret_val_0;
movi_len=((((((2147483647*2)+1)-((1<<20)*16))-2048)-2048)+4);
hasIndex=1;
if (AVI->fps<0.001)
{
frate=0;
ms_per_frame=0;
}
else
{
frate=((int)((1000000*AVI->fps)+0.5));
ms_per_frame=((int)((1000000/AVI->fps)+0.5));
}
nhb=0;
memcpy(AVI_header+nhb, "RIFF", 4);
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, movi_len);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "AVI ", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
hdrl_start=nhb;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "hdrl", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "avih", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 56);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, ms_per_frame);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
flag=256;
if (hasIndex)
{
flag|=16;
}
if (hasIndex&&AVI->must_use_index)
{
flag|=32;
}
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, flag);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->anum+1);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->width);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->height);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
strl_start=nhb;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strl", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strh", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 56);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "vids", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, AVI->compressor, 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 1000000);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, frate);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb,  - 1);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strf", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 40);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 40);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->width);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->height);
}
nhb+=4;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(1&255);
AVI_header[nhb+1]=((1>>8)&255);
}
nhb+=2;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(24&255);
AVI_header[nhb+1]=((24>>8)&255);
}
nhb+=2;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, AVI->compressor, 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, (AVI->width*AVI->height)*3);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
long2str((AVI_header+strl_start)-4, nhb-strl_start);
#pragma loop name avi_update_header#0 
for (j=0; j<AVI->anum;  ++ j)
{
sampsize=avi_sampsize(AVI, j);
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
strl_start=nhb;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strl", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strh", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 56);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "auds", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, sampsize/4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, (1000*AVI->track[j].mp3rate)/8);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, (4*AVI->track[j].audio_bytes)/sampsize);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb,  - 1);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, sampsize/4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strf", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 16);
}
nhb+=4;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(AVI->track[j].a_fmt&255);
AVI_header[nhb+1]=((AVI->track[j].a_fmt>>8)&255);
}
nhb+=2;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(AVI->track[j].a_chans&255);
AVI_header[nhb+1]=((AVI->track[j].a_chans>>8)&255);
}
nhb+=2;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->track[j].a_rate);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, (1000*AVI->track[j].mp3rate)/8);
}
nhb+=4;
if (nhb<=(2048-2))
{
AVI_header[nhb]=((sampsize/4)&255);
AVI_header[nhb+1]=(((sampsize/4)>>8)&255);
}
nhb+=2;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(AVI->track[j].a_bits&255);
AVI_header[nhb+1]=((AVI->track[j].a_bits>>8)&255);
}
nhb+=2;
long2str((AVI_header+strl_start)-4, nhb-strl_start);
}
long2str((AVI_header+hdrl_start)-4, nhb-hdrl_start);
njunk=(((2048-nhb)-8)-12);
if (njunk<=0)
{
fprintf(stderr, "AVI_close_output_file: # of header bytes too small\n");
exit(1);
}
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "JUNK", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, njunk);
}
nhb+=4;
memset(AVI_header+nhb, 0, njunk);
if (njunk>(strlen(id_str)+8))
{
sprintf(id_str, "%s-%s", "my", "0.00");
memcpy(AVI_header+nhb, id_str, strlen(id_str));
}
nhb+=njunk;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, movi_len);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "movi", 4);
}
nhb+=4;
if (((lseek(AVI->fdes, 0, 0)<0)||(avi_write(AVI->fdes, (char * )AVI_header, 2048)!=2048))||(lseek(AVI->fdes, AVI->pos, 0)<0))
{
AVI_errno=6;
_ret_val_0=( - 1);
return _ret_val_0;
}
_ret_val_0=0;
return _ret_val_0;
}
static int avi_close_output_file(avi_t * AVI)
{
int ret, njunk, sampsize, hasIndex, ms_per_frame, frate, idxerror, flag;
unsigned long movi_len;
int hdrl_start, strl_start, j;
unsigned char AVI_header[2048];
long nhb;
long info_len;
int _ret_val_0;
movi_len=((AVI->pos-2048)+4);
idxerror=0;
ret=avi_add_chunk(AVI, (unsigned char * )"idx1", (unsigned char * )((void * )AVI->idx), AVI->n_idx*16);
hasIndex=(ret==0);
if (ret)
{
idxerror=1;
AVI_errno=5;
}
if (AVI->fps<0.001)
{
frate=0;
ms_per_frame=0;
}
else
{
frate=((int)((1000000*AVI->fps)+0.5));
ms_per_frame=((int)((1000000/AVI->fps)+0.5));
}
nhb=0;
memcpy(AVI_header+nhb, "RIFF", 4);
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->pos-8);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "AVI ", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
hdrl_start=nhb;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "hdrl", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "avih", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 56);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, ms_per_frame);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
flag=256;
if (hasIndex)
{
flag|=16;
}
if (hasIndex&&AVI->must_use_index)
{
flag|=32;
}
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, flag);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->video_frames);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->anum+1);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->width);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->height);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
strl_start=nhb;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strl", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strh", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 56);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "vids", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, AVI->compressor, 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 1000000);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, frate);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->video_frames);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb,  - 1);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strf", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 40);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 40);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->width);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->height);
}
nhb+=4;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(1&255);
AVI_header[nhb+1]=((1>>8)&255);
}
nhb+=2;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(24&255);
AVI_header[nhb+1]=((24>>8)&255);
}
nhb+=2;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, AVI->compressor, 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, (AVI->width*AVI->height)*3);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
long2str((AVI_header+strl_start)-4, nhb-strl_start);
#pragma loop name avi_close_output_file#0 
for (j=0; j<AVI->anum;  ++ j)
{
sampsize=avi_sampsize(AVI, j);
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
strl_start=nhb;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strl", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strh", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 56);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "auds", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, sampsize/4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, (1000*AVI->track[j].mp3rate)/8);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, (4*AVI->track[j].audio_bytes)/sampsize);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb,  - 1);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, sampsize/4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 0);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "strf", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 16);
}
nhb+=4;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(AVI->track[j].a_fmt&255);
AVI_header[nhb+1]=((AVI->track[j].a_fmt>>8)&255);
}
nhb+=2;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(AVI->track[j].a_chans&255);
AVI_header[nhb+1]=((AVI->track[j].a_chans>>8)&255);
}
nhb+=2;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, AVI->track[j].a_rate);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, (1000*AVI->track[j].mp3rate)/8);
}
nhb+=4;
if (nhb<=(2048-2))
{
AVI_header[nhb]=((sampsize/4)&255);
AVI_header[nhb+1]=(((sampsize/4)>>8)&255);
}
nhb+=2;
if (nhb<=(2048-2))
{
AVI_header[nhb]=(AVI->track[j].a_bits&255);
AVI_header[nhb+1]=((AVI->track[j].a_bits>>8)&255);
}
nhb+=2;
long2str((AVI_header+strl_start)-4, nhb-strl_start);
}
long2str((AVI_header+hdrl_start)-4, nhb-hdrl_start);
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
info_len=(64+12);
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, info_len);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "INFO", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "ISFT", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, 64);
}
nhb+=4;
sprintf(id_str, "%s-%s", "my", "0.00");
memset(AVI_header+nhb, 0, 64);
memcpy(AVI_header+nhb, id_str, strlen(id_str));
nhb+=64;
njunk=(((2048-nhb)-8)-12);
if (njunk<=0)
{
fprintf(stderr, "AVI_close_output_file: # of header bytes too small\n");
exit(1);
}
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "JUNK", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, njunk);
}
nhb+=4;
memset(AVI_header+nhb, 0, njunk);
nhb+=njunk;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "LIST", 4);
}
nhb+=4;
if (nhb<=(2048-4))
{
long2str(AVI_header+nhb, movi_len);
}
nhb+=4;
if (nhb<=(2048-4))
{
memcpy(AVI_header+nhb, "movi", 4);
}
nhb+=4;
if ((lseek(AVI->fdes, 0, 0)<0)||(avi_write(AVI->fdes, (char * )AVI_header, 2048)!=2048))
{
AVI_errno=6;
_ret_val_0=( - 1);
return _ret_val_0;
}
if (idxerror)
{
_ret_val_0=( - 1);
return _ret_val_0;
}
_ret_val_0=0;
return _ret_val_0;
}
static int avi_write_data(avi_t * AVI, char * data, unsigned long length, int audio, int keyframe)
{
int n;
unsigned char astr[5];
int _ret_val_0;
if (((((AVI->pos+8)+length)+8)+((AVI->n_idx+1)*16))>((((2147483647*2)+1)-((1<<20)*16))-2048))
{
AVI_errno=1;
_ret_val_0=( - 1);
return _ret_val_0;
}
sprintf((char * )astr, "0%1dwb", AVI->aptr+1);
if (audio)
{
n=avi_add_index_entry(AVI, astr, 0, AVI->pos, length);
}
else
{
n=avi_add_index_entry(AVI, (unsigned char * )"00db", (keyframe ? 16 : 0), AVI->pos, length);
}
if (n)
{
_ret_val_0=( - 1);
return _ret_val_0;
}
if (audio)
{
n=avi_add_chunk(AVI, (unsigned char * )astr, (unsigned char * )data, length);
}
else
{
n=avi_add_chunk(AVI, (unsigned char * )"00db", (unsigned char * )data, length);
}
if (n)
{
_ret_val_0=( - 1);
return _ret_val_0;
}
_ret_val_0=0;
return _ret_val_0;
}
int AVI_write_frame(avi_t * AVI, char * data, long bytes, int keyframe)
{
unsigned long pos;
int _ret_val_0;
if (AVI->mode==1)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
pos=AVI->pos;
if (avi_write_data(AVI, data, bytes, 0, keyframe))
{
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->last_pos=pos;
AVI->last_len=bytes;
AVI->video_frames ++ ;
_ret_val_0=0;
return _ret_val_0;
}
int AVI_dup_frame(avi_t * AVI)
{
int _ret_val_0;
if (AVI->mode==1)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if (AVI->last_pos==0)
{
_ret_val_0=0;
return _ret_val_0;
}
if (avi_add_index_entry(AVI, (unsigned char * )"00db", 16, AVI->last_pos, AVI->last_len))
{
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->video_frames ++ ;
AVI->must_use_index=1;
_ret_val_0=0;
return _ret_val_0;
}
int AVI_write_audio(avi_t * AVI, char * data, long bytes)
{
int _ret_val_0;
if (AVI->mode==1)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if (avi_write_data(AVI, data, bytes, 1, 0))
{
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->track[AVI->aptr].audio_bytes+=bytes;
_ret_val_0=0;
return _ret_val_0;
}
int AVI_append_audio(avi_t * AVI, char * data, long bytes)
{
long i, length, pos;
unsigned char c[4];
int _ret_val_0;
if (AVI->mode==1)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
-- AVI->n_idx;
length=str2ulong(AVI->idx[AVI->n_idx]+12);
pos=str2ulong(AVI->idx[AVI->n_idx]+8);
long2str(AVI->idx[AVI->n_idx]+12, length+bytes);
++ AVI->n_idx;
AVI->track[AVI->aptr].audio_bytes+=bytes;
lseek(AVI->fdes, pos+4, 0);
long2str(c, length+bytes);
avi_write(AVI->fdes, (char * )c, 4);
lseek(AVI->fdes, (pos+8)+length, 0);
i=(((length+bytes)+1)&( ~ 1));
bytes=(i-length);
avi_write(AVI->fdes, data, bytes);
AVI->pos=((pos+8)+i);
_ret_val_0=0;
return _ret_val_0;
}
long AVI_bytes_remain(avi_t * AVI)
{
long _ret_val_0;
if (AVI->mode==1)
{
_ret_val_0=0;
return _ret_val_0;
}
_ret_val_0=(((((2147483647*2)+1)-((1<<20)*16))-2048)-((AVI->pos+8)+(16*AVI->n_idx)));
return _ret_val_0;
}
long AVI_bytes_written(avi_t * AVI)
{
long _ret_val_0;
if (AVI->mode==1)
{
_ret_val_0=0;
return _ret_val_0;
}
_ret_val_0=((AVI->pos+8)+(16*AVI->n_idx));
return _ret_val_0;
}
int AVI_set_audio_track(avi_t * AVI, int track)
{
int _ret_val_0;
if ((track<0)||((track+1)>AVI->anum))
{
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->aptr=track;
_ret_val_0=0;
return _ret_val_0;
}
int AVI_get_audio_track(avi_t * AVI)
{
int _ret_val_0;
_ret_val_0=AVI->aptr;
return _ret_val_0;
}
int AVI_close(avi_t * AVI)
{
int ret;
if (AVI->mode==0)
{
ret=avi_close_output_file(AVI);
}
else
{
ret=0;
}
close(AVI->fdes);
if (AVI->idx)
{
free(AVI->idx);
}
if (AVI->video_index)
{
free(AVI->video_index);
}
free(AVI);
return ret;
}
avi_t *AVI_open_input_file(char * filename, int getIndex)
{
avi_t * AVI = (void * )0;
avi_t * _ret_val_0;
AVI=((avi_t * )malloc(sizeof (avi_t)));
if (AVI==((void * )0))
{
AVI_errno=8;
_ret_val_0=0;
return _ret_val_0;
}
memset((void * )AVI, 0, sizeof (avi_t));
AVI->mode=1;
AVI->fdes=open(filename, 0|0);
if (AVI->fdes<0)
{
AVI_errno=2;
free(AVI);
_ret_val_0=0;
return _ret_val_0;
}
avi_parse_input_file(AVI, getIndex);
AVI->aptr=0;
return AVI;
}
avi_t *AVI_open_fd(int fd, int getIndex)
{
avi_t * AVI = (void * )0;
avi_t * _ret_val_0;
AVI=((avi_t * )malloc(sizeof (avi_t)));
if (AVI==((void * )0))
{
AVI_errno=8;
_ret_val_0=0;
return _ret_val_0;
}
memset((void * )AVI, 0, sizeof (avi_t));
AVI->mode=1;
AVI->fdes=fd;
avi_parse_input_file(AVI, getIndex);
AVI->aptr=0;
return AVI;
}
int avi_parse_input_file(avi_t * AVI, int getIndex)
{
long i, n, rate, scale, idx_type;
unsigned char * hdrl_data;
long header_offset = 0, hdrl_len = 0;
long nvi, nai[8], ioff;
long tot[8];
int j;
int lasttag = 0;
int vids_strh_seen = 0;
int vids_strf_seen = 0;
int auds_strh_seen = 0;
int num_stream = 0;
char data[256];
int _ret_val_0;
if (avi_read(AVI->fdes, data, 12)!=12)
{
AVI_close(AVI);
AVI_errno=3;
_ret_val_0=0;
return _ret_val_0;
}
if ((strncasecmp(data, "RIFF", 4)!=0)||(strncasecmp(data+8, "AVI ", 4)!=0))
{
AVI_close(AVI);
AVI_errno=9;
_ret_val_0=0;
return _ret_val_0;
}
hdrl_data=0;
while (1)
{
if (avi_read(AVI->fdes, data, 8)!=8)
{
break;
}
n=str2ulong(((unsigned char * )data)+4);
n=((n+1)&( ~ 1));
if (strncasecmp(data, "LIST", 4)==0)
{
if (avi_read(AVI->fdes, data, 4)!=4)
{
AVI_close(AVI);
AVI_errno=3;
_ret_val_0=0;
return _ret_val_0;
}
n-=4;
if (strncasecmp(data, "hdrl", 4)==0)
{
hdrl_len=n;
hdrl_data=((unsigned char * )malloc(n));
if (hdrl_data==0)
{
AVI_close(AVI);
AVI_errno=8;
_ret_val_0=0;
return _ret_val_0;
}
;
header_offset=lseek(AVI->fdes, 0, 1);
if (avi_read(AVI->fdes, (char * )hdrl_data, n)!=n)
{
AVI_close(AVI);
AVI_errno=3;
_ret_val_0=0;
return _ret_val_0;
}
}
else
{
if (strncasecmp(data, "movi", 4)==0)
{
AVI->movi_start=lseek(AVI->fdes, 0, 1);
lseek(AVI->fdes, n, 1);
}
else
{
lseek(AVI->fdes, n, 1);
}
}
}
else
{
if (strncasecmp(data, "idx1", 4)==0)
{
AVI->n_idx=(AVI->max_idx=(n/16));
AVI->idx=((unsigned char ((* )[16]))malloc(n));
if (AVI->idx==0)
{
AVI_close(AVI);
AVI_errno=8;
_ret_val_0=0;
return _ret_val_0;
}
if (avi_read(AVI->fdes, (char * )AVI->idx, n)!=n)
{
AVI_close(AVI);
AVI_errno=3;
_ret_val_0=0;
return _ret_val_0;
}
}
else
{
lseek(AVI->fdes, n, 1);
}
}
}
if ( ! hdrl_data)
{
AVI_close(AVI);
AVI_errno=10;
_ret_val_0=0;
return _ret_val_0;
}
if ( ! AVI->movi_start)
{
AVI_close(AVI);
AVI_errno=11;
_ret_val_0=0;
return _ret_val_0;
}
#pragma loop name avi_parse_input_file#0 
for (i=0; i<hdrl_len; )
{
if (strncasecmp(((char * )hdrl_data)+i, "LIST", 4)==0)
{
i+=12;
continue;
}
n=str2ulong((hdrl_data+i)+4);
n=((n+1)&( ~ 1));
if (strncasecmp(((char * )hdrl_data)+i, "strh", 4)==0)
{
i+=8;
if ((strncasecmp(((char * )hdrl_data)+i, "vids", 4)==0)&&( ! vids_strh_seen))
{
memcpy(AVI->compressor, (hdrl_data+i)+4, 4);
AVI->compressor[4]=0;
AVI->v_codech_off=((header_offset+i)+4);
scale=str2ulong((((unsigned char * )hdrl_data)+i)+20);
rate=str2ulong((hdrl_data+i)+24);
if (scale!=0)
{
AVI->fps=(((double)rate)/((double)scale));
}
AVI->video_frames=str2ulong((hdrl_data+i)+32);
AVI->video_strn=num_stream;
AVI->max_len=0;
vids_strh_seen=1;
lasttag=1;
}
else
{
if ((strncasecmp(((char * )hdrl_data)+i, "auds", 4)==0)&&( ! auds_strh_seen))
{
AVI->aptr=AVI->anum;
++ AVI->anum;
if (AVI->anum>8)
{
fprintf(stderr, "error - only %d audio tracks supported\n", 8);
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->track[AVI->aptr].audio_bytes=(str2ulong((hdrl_data+i)+32)*avi_sampsize(AVI, 0));
AVI->track[AVI->aptr].audio_strn=num_stream;
lasttag=2;
AVI->track[AVI->aptr].a_codech_off=(header_offset+i);
}
else
{
lasttag=0;
}
}
num_stream ++ ;
}
else
{
if (strncasecmp(((char * )hdrl_data)+i, "strf", 4)==0)
{
i+=8;
if (lasttag==1)
{
AVI->width=str2ulong((hdrl_data+i)+4);
AVI->height=str2ulong((hdrl_data+i)+8);
vids_strf_seen=1;
AVI->v_codecf_off=((header_offset+i)+16);
memcpy(AVI->compressor2, (hdrl_data+i)+16, 4);
AVI->compressor2[4]=0;
}
else
{
if (lasttag==2)
{
AVI->track[AVI->aptr].a_fmt=str2ushort(hdrl_data+i);
AVI->track[AVI->aptr].a_codecf_off=(header_offset+i);
AVI->track[AVI->aptr].a_chans=str2ushort((hdrl_data+i)+2);
AVI->track[AVI->aptr].a_rate=str2ulong((hdrl_data+i)+4);
AVI->track[AVI->aptr].mp3rate=((8*str2ulong((hdrl_data+i)+8))/1000);
AVI->track[AVI->aptr].a_bits=str2ushort((hdrl_data+i)+14);
}
}
lasttag=0;
}
else
{
i+=8;
lasttag=0;
}
}
i+=n;
}
free(hdrl_data);
if (( ! vids_strh_seen)||( ! vids_strf_seen))
{
AVI_close(AVI);
AVI_errno=12;
_ret_val_0=0;
return _ret_val_0;
}
AVI->video_tag[0]=((AVI->video_strn/10)+'0');
AVI->video_tag[1]=((AVI->video_strn%10)+'0');
AVI->video_tag[2]='d';
AVI->video_tag[3]='b';
if ( ! AVI->track[0].a_chans)
{
AVI->track[0].audio_strn=99;
}
#pragma loop name avi_parse_input_file#1 
for (j=0; j<AVI->anum;  ++ j)
{
AVI->track[j].audio_tag[0]=(((j+1)/10)+'0');
AVI->track[j].audio_tag[1]=(((j+1)%10)+'0');
AVI->track[j].audio_tag[2]='w';
AVI->track[j].audio_tag[3]='b';
}
lseek(AVI->fdes, AVI->movi_start, 0);
if ( ! getIndex)
{
_ret_val_0=0;
return _ret_val_0;
}
idx_type=0;
if (AVI->idx)
{
long pos, len;
#pragma loop name avi_parse_input_file#2 
for (i=0; i<AVI->n_idx; i ++ )
{
if (strncasecmp((char * )AVI->idx[i], (char * )AVI->video_tag, 3)==0)
{
break;
}
}
if (i>=AVI->n_idx)
{
AVI_close(AVI);
AVI_errno=12;
_ret_val_0=0;
return _ret_val_0;
}
pos=str2ulong(AVI->idx[i]+8);
len=str2ulong(AVI->idx[i]+12);
lseek(AVI->fdes, pos, 0);
if (avi_read(AVI->fdes, data, 8)!=8)
{
AVI_close(AVI);
AVI_errno=3;
_ret_val_0=0;
return _ret_val_0;
}
if ((strncasecmp((char * )data, (char * )AVI->idx[i], 4)==0)&&(str2ulong(((unsigned char * )data)+4)==len))
{
idx_type=1;
}
else
{
lseek(AVI->fdes, (pos+AVI->movi_start)-4, 0);
if (avi_read(AVI->fdes, data, 8)!=8)
{
AVI_close(AVI);
AVI_errno=3;
_ret_val_0=0;
return _ret_val_0;
}
if ((strncasecmp((char * )data, (char * )AVI->idx[i], 4)==0)&&(str2ulong(((unsigned char * )data)+4)==len))
{
idx_type=2;
}
}
}
if (idx_type==0)
{
lseek(AVI->fdes, AVI->movi_start, 0);
AVI->n_idx=0;
while (1)
{
if (avi_read(AVI->fdes, data, 8)!=8)
{
break;
}
n=str2ulong(((unsigned char * )data)+4);
if (strncasecmp(data, "LIST", 4)==0)
{
lseek(AVI->fdes, 4, 1);
continue;
}
if ((((data[2]=='d')||(data[2]=='D'))&&((((data[3]=='b')||(data[3]=='B'))||(data[3]=='c'))||(data[3]=='C')))||(((data[2]=='w')||(data[2]=='W'))&&((data[3]=='b')||(data[3]=='B'))))
{
avi_add_index_entry(AVI, (unsigned char * )data, 0, lseek(AVI->fdes, 0, 1)-8, n);
}
lseek(AVI->fdes, (n+1)&( ~ 1), 1);
}
idx_type=1;
}
nvi=0;
#pragma loop name avi_parse_input_file#3 
#pragma cetus parallel 
#pragma omp parallel for
for (j=0; j<AVI->anum;  ++ j)
{
nai[j]=0;
}
#pragma loop name avi_parse_input_file#4 
#pragma cetus reduction(+: nai[j], nvi) 
for (i=0; i<AVI->n_idx; i ++ )
{
if (strncasecmp((char * )AVI->idx[i], (char * )AVI->video_tag, 3)==0)
{
nvi ++ ;
}
#pragma loop name avi_parse_input_file#4#0 
for (j=0; j<AVI->anum;  ++ j)
{
if (strncasecmp((char * )AVI->idx[i], AVI->track[j].audio_tag, 4)==0)
{
nai[j] ++ ;
}
}
}
AVI->video_frames=nvi;
#pragma loop name avi_parse_input_file#5 
for (j=0; j<AVI->anum;  ++ j)
{
AVI->track[j].audio_chunks=nai[j];
}
if (AVI->video_frames==0)
{
AVI_close(AVI);
AVI_errno=12;
_ret_val_0=0;
return _ret_val_0;
}
;
AVI->video_index=((video_index_entry * )malloc(nvi*sizeof (video_index_entry)));
if (AVI->video_index==0)
{
AVI_close(AVI);
AVI_errno=8;
_ret_val_0=0;
return _ret_val_0;
}
;
#pragma loop name avi_parse_input_file#6 
for (j=0; j<AVI->anum;  ++ j)
{
if (AVI->track[j].audio_chunks)
{
AVI->track[j].audio_index=((audio_index_entry * )malloc(nai[j]*sizeof (audio_index_entry)));
if (AVI->track[j].audio_index==0)
{
AVI_close(AVI);
AVI_errno=8;
_ret_val_0=0;
return _ret_val_0;
}
;
}
}
nvi=0;
#pragma loop name avi_parse_input_file#7 
#pragma cetus parallel 
#pragma omp parallel for
for (j=0; j<AVI->anum;  ++ j)
{
nai[j]=(tot[j]=0);
}
ioff=((idx_type==1) ? 8 : (AVI->movi_start+4));
#pragma loop name avi_parse_input_file#8 
for (i=0; i<AVI->n_idx; i ++ )
{
if (strncasecmp((char * )AVI->idx[i], (char * )AVI->video_tag, 3)==0)
{
AVI->video_index[nvi].key=str2ulong(AVI->idx[i]+4);
AVI->video_index[nvi].pos=(str2ulong(AVI->idx[i]+8)+ioff);
AVI->video_index[nvi].len=str2ulong(AVI->idx[i]+12);
nvi ++ ;
}
#pragma loop name avi_parse_input_file#8#0 
for (j=0; j<AVI->anum;  ++ j)
{
if (strncasecmp((char * )AVI->idx[i], AVI->track[j].audio_tag, 4)==0)
{
AVI->track[j].audio_index[nai[j]].pos=(str2ulong(AVI->idx[i]+8)+ioff);
AVI->track[j].audio_index[nai[j]].len=str2ulong(AVI->idx[i]+12);
AVI->track[j].audio_index[nai[j]].tot=tot[j];
tot[j]+=AVI->track[j].audio_index[nai[j]].len;
nai[j] ++ ;
}
}
}
#pragma loop name avi_parse_input_file#9 
for (j=0; j<AVI->anum;  ++ j)
{
AVI->track[j].audio_bytes=tot[j];
}
lseek(AVI->fdes, AVI->movi_start, 0);
AVI->video_pos=0;
_ret_val_0=0;
return _ret_val_0;
}
long AVI_video_frames(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->video_frames;
return _ret_val_0;
}
int AVI_video_width(avi_t * AVI)
{
int _ret_val_0;
_ret_val_0=AVI->width;
return _ret_val_0;
}
int AVI_video_height(avi_t * AVI)
{
int _ret_val_0;
_ret_val_0=AVI->height;
return _ret_val_0;
}
double AVI_frame_rate(avi_t * AVI)
{
double _ret_val_0;
_ret_val_0=AVI->fps;
return _ret_val_0;
}
char *AVI_video_compressor(avi_t * AVI)
{
char * _ret_val_0;
_ret_val_0=AVI->compressor2;
return _ret_val_0;
}
long AVI_max_video_chunk(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->max_len;
return _ret_val_0;
}
int AVI_audio_tracks(avi_t * AVI)
{
int _ret_val_0;
_ret_val_0=AVI->anum;
return _ret_val_0;
}
int AVI_audio_channels(avi_t * AVI)
{
int _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].a_chans;
return _ret_val_0;
}
long AVI_audio_mp3rate(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].mp3rate;
return _ret_val_0;
}
int AVI_audio_bits(avi_t * AVI)
{
int _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].a_bits;
return _ret_val_0;
}
int AVI_audio_format(avi_t * AVI)
{
int _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].a_fmt;
return _ret_val_0;
}
long AVI_audio_rate(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].a_rate;
return _ret_val_0;
}
long AVI_audio_bytes(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].audio_bytes;
return _ret_val_0;
}
long AVI_audio_chunks(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].audio_chunks;
return _ret_val_0;
}
long AVI_audio_codech_offset(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].a_codech_off;
return _ret_val_0;
}
long AVI_audio_codecf_offset(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->track[AVI->aptr].a_codecf_off;
return _ret_val_0;
}
long AVI_video_codech_offset(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->v_codech_off;
return _ret_val_0;
}
long AVI_video_codecf_offset(avi_t * AVI)
{
long _ret_val_0;
_ret_val_0=AVI->v_codecf_off;
return _ret_val_0;
}
long AVI_frame_size(avi_t * AVI, long frame)
{
long _ret_val_0;
if (AVI->mode==0)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ( ! AVI->video_index)
{
AVI_errno=13;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ((frame<0)||(frame>=AVI->video_frames))
{
_ret_val_0=0;
return _ret_val_0;
}
_ret_val_0=AVI->video_index[frame].len;
return _ret_val_0;
}
long AVI_audio_size(avi_t * AVI, long frame)
{
long _ret_val_0;
if (AVI->mode==0)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ( ! AVI->track[AVI->aptr].audio_index)
{
AVI_errno=13;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ((frame<0)||(frame>=AVI->track[AVI->aptr].audio_chunks))
{
_ret_val_0=0;
return _ret_val_0;
}
_ret_val_0=AVI->track[AVI->aptr].audio_index[frame].len;
return _ret_val_0;
}
long AVI_get_video_position(avi_t * AVI, long frame)
{
long _ret_val_0;
if (AVI->mode==0)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ( ! AVI->video_index)
{
AVI_errno=13;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ((frame<0)||(frame>=AVI->video_frames))
{
_ret_val_0=0;
return _ret_val_0;
}
_ret_val_0=AVI->video_index[frame].pos;
return _ret_val_0;
}
int AVI_seek_start(avi_t * AVI)
{
int _ret_val_0;
if (AVI->mode==0)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
lseek(AVI->fdes, AVI->movi_start, 0);
AVI->video_pos=0;
_ret_val_0=0;
return _ret_val_0;
}
int AVI_set_video_position(avi_t * AVI, long frame)
{
int _ret_val_0;
if (AVI->mode==0)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ( ! AVI->video_index)
{
AVI_errno=13;
_ret_val_0=( - 1);
return _ret_val_0;
}
if (frame<0)
{
frame=0;
}
AVI->video_pos=frame;
_ret_val_0=0;
return _ret_val_0;
}
int AVI_set_audio_bitrate(avi_t * AVI, long bitrate)
{
int _ret_val_0;
if (AVI->mode==1)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->track[AVI->aptr].mp3rate=bitrate;
_ret_val_0=0;
return _ret_val_0;
}
long AVI_read_frame(avi_t * AVI, char * vidbuf, int * keyframe)
{
long n;
long _ret_val_0;
if (AVI->mode==0)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ( ! AVI->video_index)
{
AVI_errno=13;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ((AVI->video_pos<0)||(AVI->video_pos>=AVI->video_frames))
{
_ret_val_0=( - 1);
return _ret_val_0;
}
n=AVI->video_index[AVI->video_pos].len;
( * keyframe)=((AVI->video_index[AVI->video_pos].key==16) ? 1 : 0);
lseek(AVI->fdes, AVI->video_index[AVI->video_pos].pos, 0);
if (avi_read(AVI->fdes, vidbuf, n)!=n)
{
AVI_errno=3;
_ret_val_0=( - 1);
return _ret_val_0;
}
AVI->video_pos ++ ;
return n;
}
int AVI_set_audio_position(avi_t * AVI, long byte)
{
long n0, n1, n;
int _ret_val_0;
if (AVI->mode==0)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ( ! AVI->track[AVI->aptr].audio_index)
{
AVI_errno=13;
_ret_val_0=( - 1);
return _ret_val_0;
}
if (byte<0)
{
byte=0;
}
n0=0;
n1=AVI->track[AVI->aptr].audio_chunks;
while (n0<(n1-1))
{
n=((n0+n1)/2);
if (AVI->track[AVI->aptr].audio_index[n].tot>byte)
{
n1=n;
}
else
{
n0=n;
}
}
AVI->track[AVI->aptr].audio_posc=n0;
AVI->track[AVI->aptr].audio_posb=(byte-AVI->track[AVI->aptr].audio_index[n0].tot);
_ret_val_0=0;
return _ret_val_0;
}
long AVI_read_audio(avi_t * AVI, char * audbuf, long bytes)
{
long nr, pos, left, todo;
long _ret_val_0;
if (AVI->mode==0)
{
AVI_errno=7;
_ret_val_0=( - 1);
return _ret_val_0;
}
if ( ! AVI->track[AVI->aptr].audio_index)
{
AVI_errno=13;
_ret_val_0=( - 1);
return _ret_val_0;
}
nr=0;
while (bytes>0)
{
left=(AVI->track[AVI->aptr].audio_index[AVI->track[AVI->aptr].audio_posc].len-AVI->track[AVI->aptr].audio_posb);
if (left==0)
{
if (AVI->track[AVI->aptr].audio_posc>=(AVI->track[AVI->aptr].audio_chunks-1))
{
return nr;
}
AVI->track[AVI->aptr].audio_posc ++ ;
AVI->track[AVI->aptr].audio_posb=0;
continue;
}
if (bytes<left)
{
todo=bytes;
}
else
{
todo=left;
}
pos=(AVI->track[AVI->aptr].audio_index[AVI->track[AVI->aptr].audio_posc].pos+AVI->track[AVI->aptr].audio_posb);
lseek(AVI->fdes, pos, 0);
if (avi_read(AVI->fdes, audbuf+nr, todo)!=todo)
{
AVI_errno=3;
_ret_val_0=( - 1);
return _ret_val_0;
}
bytes-=todo;
nr+=todo;
AVI->track[AVI->aptr].audio_posb+=todo;
}
return nr;
}
int AVI_read_data(avi_t * AVI, char * vidbuf, long max_vidbuf, char * audbuf, long max_audbuf, long * len)
{
int n;
char data[8];
int _ret_val_0;
if (AVI->mode==0)
{
_ret_val_0=0;
return _ret_val_0;
}
while (1)
{
if (avi_read(AVI->fdes, data, 8)!=8)
{
_ret_val_0=0;
return _ret_val_0;
}
if (strncasecmp(data, "LIST", 4)==0)
{
lseek(AVI->fdes, 4, 1);
continue;
}
n=((str2ulong(((unsigned char * )data)+4)+1)&( ~ 1));
if (strncasecmp(data, AVI->video_tag, 3)==0)
{
( * len)=n;
AVI->video_pos ++ ;
if (n>max_vidbuf)
{
lseek(AVI->fdes, n, 1);
_ret_val_0=( - 1);
return _ret_val_0;
}
if (avi_read(AVI->fdes, vidbuf, n)!=n)
{
_ret_val_0=0;
return _ret_val_0;
}
_ret_val_0=1;
return _ret_val_0;
}
else
{
if (strncasecmp(data, AVI->track[AVI->aptr].audio_tag, 4)==0)
{
( * len)=n;
if (n>max_audbuf)
{
lseek(AVI->fdes, n, 1);
_ret_val_0=( - 2);
return _ret_val_0;
}
if (avi_read(AVI->fdes, audbuf, n)!=n)
{
_ret_val_0=0;
return _ret_val_0;
}
_ret_val_0=2;
return _ret_val_0;
break;
}
else
{
if (lseek(AVI->fdes, n, 1)<0)
{
_ret_val_0=0;
return _ret_val_0;
}
}
}
}
return _ret_val_0;
}
char *(avi_errors[]) = {(char * )"avilib - No Error", (char * )"avilib - AVI file size limit reached", (char * )"avilib - Error opening AVI file", (char * )"avilib - Error reading from AVI file", (char * )"avilib - Error writing to AVI file", (char * )"avilib - Error writing index (file may still be useable)", (char * )"avilib - Error closing AVI file", (char * )"avilib - Operation (read/write) not permitted", (char * )"avilib - Out of memory (malloc failed)", (char * )"avilib - Not an AVI file", (char * )"avilib - AVI file has no header list (corrupted?)", (char * )"avilib - AVI file has no MOVI list (corrupted?)", (char * )"avilib - AVI file has no video data", (char * )"avilib - operation needs an index", (char * )"avilib - Unkown Error"};
static int num_avi_errors = sizeof avi_errors/sizeof (char * );
static char error_string[4096];
void AVI_print_error(char * str)
{
int aerrno;
aerrno=(((AVI_errno>=0)&&(AVI_errno<num_avi_errors)) ? AVI_errno : (num_avi_errors-1));
fprintf(stderr, "%s: %s\n", str, avi_errors[aerrno]);
if (((((AVI_errno==2)||(AVI_errno==3))||(AVI_errno==4))||(AVI_errno==5))||(AVI_errno==6))
{
perror("REASON");
}
return ;
}
char *AVI_strerror()
{
int aerrno;
char * _ret_val_0;
aerrno=(((AVI_errno>=0)&&(AVI_errno<num_avi_errors)) ? AVI_errno : (num_avi_errors-1));
if (((((AVI_errno==2)||(AVI_errno==3))||(AVI_errno==4))||(AVI_errno==5))||(AVI_errno==6))
{
sprintf(error_string, "%s - %s", avi_errors[aerrno], strerror( * __errno_location()));
return error_string;
}
else
{
_ret_val_0=avi_errors[aerrno];
return _ret_val_0;
}
return _ret_val_0;
}
uint64_t AVI_max_size()
{
uint64_t _ret_val_0;
_ret_val_0=((uint64_t)((((2147483647*2)+1)-((1<<20)*16))-2048));
return _ret_val_0;
}
