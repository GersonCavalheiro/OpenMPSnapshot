#include "private/autogen/config.h"
#include "hwloc.h"
#include "hwloc/linux.h"
#include "private/misc.h"
#include "private/private.h"
#include "private/misc.h"
#include "private/debug.h"
#include <limits.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <assert.h>
#ifdef HAVE_DIRENT_H
#include <dirent.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HWLOC_HAVE_LIBUDEV
#include <libudev.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <mntent.h>
struct hwloc_linux_backend_data_s {
char *root_path; 
int root_fd; 
int is_real_fsroot; 
#ifdef HWLOC_HAVE_LIBUDEV
struct udev *udev; 
#endif
char *dumped_hwdata_dirname;
enum {
HWLOC_LINUX_ARCH_X86, 
HWLOC_LINUX_ARCH_IA64,
HWLOC_LINUX_ARCH_ARM,
HWLOC_LINUX_ARCH_POWER,
HWLOC_LINUX_ARCH_S390,
HWLOC_LINUX_ARCH_UNKNOWN
} arch;
int is_knl;
int is_amd_with_CU;
int use_dt;
int use_numa_distances;
int use_numa_distances_for_cpuless;
int use_numa_initiators;
struct utsname utsname; 
int fallback_nbprocessors; 
unsigned pagesize;
};
#if !(defined HWLOC_HAVE_SCHED_SETAFFINITY) && (defined HWLOC_HAVE_SYSCALL)
#    ifndef __NR_sched_setaffinity
#       ifdef __i386__
#         define __NR_sched_setaffinity 241
#       elif defined(__x86_64__)
#         define __NR_sched_setaffinity 203
#       elif defined(__ia64__)
#         define __NR_sched_setaffinity 1231
#       elif defined(__hppa__)
#         define __NR_sched_setaffinity 211
#       elif defined(__alpha__)
#         define __NR_sched_setaffinity 395
#       elif defined(__s390__)
#         define __NR_sched_setaffinity 239
#       elif defined(__sparc__)
#         define __NR_sched_setaffinity 261
#       elif defined(__m68k__)
#         define __NR_sched_setaffinity 311
#       elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#         define __NR_sched_setaffinity 222
#       elif defined(__aarch64__)
#         define __NR_sched_setaffinity 122
#       elif defined(__arm__)
#         define __NR_sched_setaffinity 241
#       elif defined(__cris__)
#         define __NR_sched_setaffinity 241
#       else
#         warning "don't know the syscall number for sched_setaffinity on this architecture, will not support binding"
#         define sched_setaffinity(pid, lg, mask) (errno = ENOSYS, -1)
#       endif
#    endif
#    ifndef sched_setaffinity
#      define sched_setaffinity(pid, lg, mask) syscall(__NR_sched_setaffinity, pid, lg, mask)
#    endif
#    ifndef __NR_sched_getaffinity
#       ifdef __i386__
#         define __NR_sched_getaffinity 242
#       elif defined(__x86_64__)
#         define __NR_sched_getaffinity 204
#       elif defined(__ia64__)
#         define __NR_sched_getaffinity 1232
#       elif defined(__hppa__)
#         define __NR_sched_getaffinity 212
#       elif defined(__alpha__)
#         define __NR_sched_getaffinity 396
#       elif defined(__s390__)
#         define __NR_sched_getaffinity 240
#       elif defined(__sparc__)
#         define __NR_sched_getaffinity 260
#       elif defined(__m68k__)
#         define __NR_sched_getaffinity 312
#       elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#         define __NR_sched_getaffinity 223
#       elif defined(__aarch64__)
#         define __NR_sched_getaffinity 123
#       elif defined(__arm__)
#         define __NR_sched_getaffinity 242
#       elif defined(__cris__)
#         define __NR_sched_getaffinity 242
#       else
#         warning "don't know the syscall number for sched_getaffinity on this architecture, will not support getting binding"
#         define sched_getaffinity(pid, lg, mask) (errno = ENOSYS, -1)
#       endif
#    endif
#    ifndef sched_getaffinity
#      define sched_getaffinity(pid, lg, mask) (syscall(__NR_sched_getaffinity, pid, lg, mask) < 0 ? -1 : 0)
#    endif
#endif
#ifndef MPOL_DEFAULT
# define MPOL_DEFAULT 0
#endif
#ifndef MPOL_PREFERRED
# define MPOL_PREFERRED 1
#endif
#ifndef MPOL_BIND
# define MPOL_BIND 2
#endif
#ifndef MPOL_INTERLEAVE
# define MPOL_INTERLEAVE 3
#endif
#ifndef MPOL_LOCAL
# define MPOL_LOCAL 4
#endif
#ifndef MPOL_F_ADDR
# define  MPOL_F_ADDR (1<<1)
#endif
#ifndef MPOL_MF_STRICT
# define MPOL_MF_STRICT (1<<0)
#endif
#ifndef MPOL_MF_MOVE
# define MPOL_MF_MOVE (1<<1)
#endif
#ifndef __NR_mbind
# ifdef __i386__
#  define __NR_mbind 274
# elif defined(__x86_64__)
#  define __NR_mbind 237
# elif defined(__ia64__)
#  define __NR_mbind 1259
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_mbind 259
# elif defined(__sparc__)
#  define __NR_mbind 353
# elif defined(__aarch64__)
#  define __NR_mbind 235
# elif defined(__arm__)
#  define __NR_mbind 319
# endif
#endif
static __hwloc_inline long hwloc_mbind(void *addr __hwloc_attribute_unused,
unsigned long len __hwloc_attribute_unused,
int mode __hwloc_attribute_unused,
const unsigned long *nodemask __hwloc_attribute_unused,
unsigned long maxnode __hwloc_attribute_unused,
unsigned flags __hwloc_attribute_unused)
{
#if (defined __NR_mbind) && (defined HWLOC_HAVE_SYSCALL)
return syscall(__NR_mbind, (long) addr, len, mode, (long)nodemask, maxnode, flags);
#else
#warning Couldn't find __NR_mbind syscall number, memory binding won't be supported
errno = ENOSYS;
return -1;
#endif
}
#ifndef __NR_set_mempolicy
# ifdef __i386__
#  define __NR_set_mempolicy 276
# elif defined(__x86_64__)
#  define __NR_set_mempolicy 239
# elif defined(__ia64__)
#  define __NR_set_mempolicy 1261
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_set_mempolicy 261
# elif defined(__sparc__)
#  define __NR_set_mempolicy 305
# elif defined(__aarch64__)
#  define __NR_set_mempolicy 237
# elif defined(__arm__)
#  define __NR_set_mempolicy 321
# endif
#endif
static __hwloc_inline long hwloc_set_mempolicy(int mode __hwloc_attribute_unused,
const unsigned long *nodemask __hwloc_attribute_unused,
unsigned long maxnode __hwloc_attribute_unused)
{
#if (defined __NR_set_mempolicy) && (defined HWLOC_HAVE_SYSCALL)
return syscall(__NR_set_mempolicy, mode, nodemask, maxnode);
#else
#warning Couldn't find __NR_set_mempolicy syscall number, memory binding won't be supported
errno = ENOSYS;
return -1;
#endif
}
#ifndef __NR_get_mempolicy
# ifdef __i386__
#  define __NR_get_mempolicy 275
# elif defined(__x86_64__)
#  define __NR_get_mempolicy 238
# elif defined(__ia64__)
#  define __NR_get_mempolicy 1260
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_get_mempolicy 260
# elif defined(__sparc__)
#  define __NR_get_mempolicy 304
# elif defined(__aarch64__)
#  define __NR_get_mempolicy 236
# elif defined(__arm__)
#  define __NR_get_mempolicy 320
# endif
#endif
static __hwloc_inline long hwloc_get_mempolicy(int *mode __hwloc_attribute_unused,
const unsigned long *nodemask __hwloc_attribute_unused,
unsigned long maxnode __hwloc_attribute_unused,
void *addr __hwloc_attribute_unused,
int flags __hwloc_attribute_unused)
{
#if (defined __NR_get_mempolicy) && (defined HWLOC_HAVE_SYSCALL)
return syscall(__NR_get_mempolicy, mode, nodemask, maxnode, addr, flags);
#else
#warning Couldn't find __NR_get_mempolicy syscall number, memory binding won't be supported
errno = ENOSYS;
return -1;
#endif
}
#ifndef __NR_migrate_pages
# ifdef __i386__
#  define __NR_migrate_pages 204
# elif defined(__x86_64__)
#  define __NR_migrate_pages 256
# elif defined(__ia64__)
#  define __NR_migrate_pages 1280
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_migrate_pages 258
# elif defined(__sparc__)
#  define __NR_migrate_pages 302
# elif defined(__aarch64__)
#  define __NR_migrate_pages 238
# elif defined(__arm__)
#  define __NR_migrate_pages 400
# endif
#endif
static __hwloc_inline long hwloc_migrate_pages(int pid __hwloc_attribute_unused,
unsigned long maxnode __hwloc_attribute_unused,
const unsigned long *oldnodes __hwloc_attribute_unused,
const unsigned long *newnodes __hwloc_attribute_unused)
{
#if (defined __NR_migrate_pages) && (defined HWLOC_HAVE_SYSCALL)
return syscall(__NR_migrate_pages, pid, maxnode, oldnodes, newnodes);
#else
#warning Couldn't find __NR_migrate_pages syscall number, memory migration won't be supported
errno = ENOSYS;
return -1;
#endif
}
#ifndef __NR_move_pages
# ifdef __i386__
#  define __NR_move_pages 317
# elif defined(__x86_64__)
#  define __NR_move_pages 279
# elif defined(__ia64__)
#  define __NR_move_pages 1276
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_move_pages 301
# elif defined(__sparc__)
#  define __NR_move_pages 307
# elif defined(__aarch64__)
#  define __NR_move_pages 239
# elif defined(__arm__)
#  define __NR_move_pages 344
# endif
#endif
static __hwloc_inline long hwloc_move_pages(int pid __hwloc_attribute_unused,
unsigned long count __hwloc_attribute_unused,
void **pages __hwloc_attribute_unused,
const int *nodes __hwloc_attribute_unused,
int *status __hwloc_attribute_unused,
int flags __hwloc_attribute_unused)
{
#if (defined __NR_move_pages) && (defined HWLOC_HAVE_SYSCALL)
return syscall(__NR_move_pages, pid, count, pages, nodes, status, flags);
#else
#warning Couldn't find __NR_move_pages syscall number, getting memory location won't be supported
errno = ENOSYS;
return -1;
#endif
}
#include <arpa/inet.h>
#ifdef HAVE_OPENAT
static const char *
hwloc_checkat(const char *path, int fsroot_fd)
{
const char *relative_path = path;
if (fsroot_fd >= 0)
for (; *relative_path == '/'; relative_path++);
return relative_path;
}
static int
hwloc_openat(const char *path, int fsroot_fd)
{
const char *relative_path;
relative_path = hwloc_checkat(path, fsroot_fd);
if (!relative_path)
return -1;
return openat (fsroot_fd, relative_path, O_RDONLY);
}
static FILE *
hwloc_fopenat(const char *path, const char *mode, int fsroot_fd)
{
int fd;
if (strcmp(mode, "r")) {
errno = ENOTSUP;
return NULL;
}
fd = hwloc_openat (path, fsroot_fd);
if (fd == -1)
return NULL;
return fdopen(fd, mode);
}
static int
hwloc_accessat(const char *path, int mode, int fsroot_fd)
{
const char *relative_path;
relative_path = hwloc_checkat(path, fsroot_fd);
if (!relative_path)
return -1;
return faccessat(fsroot_fd, relative_path, mode, 0);
}
static int
hwloc_fstatat(const char *path, struct stat *st, int flags, int fsroot_fd)
{
const char *relative_path;
relative_path = hwloc_checkat(path, fsroot_fd);
if (!relative_path)
return -1;
return fstatat(fsroot_fd, relative_path, st, flags);
}
static DIR*
hwloc_opendirat(const char *path, int fsroot_fd)
{
int dir_fd;
const char *relative_path;
relative_path = hwloc_checkat(path, fsroot_fd);
if (!relative_path)
return NULL;
dir_fd = openat(fsroot_fd, relative_path, O_RDONLY | O_DIRECTORY);
if (dir_fd < 0)
return NULL;
return fdopendir(dir_fd);
}
static int
hwloc_readlinkat(const char *path, char *buf, size_t buflen, int fsroot_fd)
{
const char *relative_path;
relative_path = hwloc_checkat(path, fsroot_fd);
if (!relative_path)
return -1;
return readlinkat(fsroot_fd, relative_path, buf, buflen);
}
#endif 
static __hwloc_inline int
hwloc_open(const char *p, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
return hwloc_openat(p, d);
#else
return open(p, O_RDONLY);
#endif
}
static __hwloc_inline FILE *
hwloc_fopen(const char *p, const char *m, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
return hwloc_fopenat(p, m, d);
#else
return fopen(p, m);
#endif
}
static __hwloc_inline int
hwloc_access(const char *p, int m, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
return hwloc_accessat(p, m, d);
#else
return access(p, m);
#endif
}
static __hwloc_inline int
hwloc_stat(const char *p, struct stat *st, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
return hwloc_fstatat(p, st, 0, d);
#else
return stat(p, st);
#endif
}
static __hwloc_inline int
hwloc_lstat(const char *p, struct stat *st, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
return hwloc_fstatat(p, st, AT_SYMLINK_NOFOLLOW, d);
#else
return lstat(p, st);
#endif
}
static __hwloc_inline DIR *
hwloc_opendir(const char *p, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
return hwloc_opendirat(p, d);
#else
return opendir(p);
#endif
}
static __hwloc_inline int
hwloc_readlink(const char *p, char *l, size_t ll, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
return hwloc_readlinkat(p, l, ll, d);
#else
return readlink(p, l, ll);
#endif
}
static __hwloc_inline int
hwloc_read_path_by_length(const char *path, char *string, size_t length, int fsroot_fd)
{
int fd, ret;
fd = hwloc_open(path, fsroot_fd);
if (fd < 0)
return -1;
ret = read(fd, string, length-1); 
close(fd);
if (ret <= 0)
return -1;
string[ret] = 0;
return 0;
}
static __hwloc_inline int
hwloc_read_path_as_int(const char *path, int *value, int fsroot_fd)
{
char string[11];
if (hwloc_read_path_by_length(path, string, sizeof(string), fsroot_fd) < 0)
return -1;
*value = atoi(string);
return 0;
}
static __hwloc_inline int
hwloc_read_path_as_uint(const char *path, unsigned *value, int fsroot_fd)
{
char string[11];
if (hwloc_read_path_by_length(path, string, sizeof(string), fsroot_fd) < 0)
return -1;
*value = (unsigned) strtoul(string, NULL, 10);
return 0;
}
static __hwloc_inline int
hwloc_read_path_as_uint64(const char *path, uint64_t *value, int fsroot_fd)
{
char string[22];
if (hwloc_read_path_by_length(path, string, sizeof(string), fsroot_fd) < 0)
return -1;
*value = (uint64_t) strtoull(string, NULL, 10);
return 0;
}
static __hwloc_inline int
hwloc__read_fd(int fd, char **bufferp, size_t *sizep)
{
char *buffer;
size_t toread, filesize, totalread;
ssize_t ret;
toread = filesize = *sizep;
buffer = malloc(filesize+1);
if (!buffer)
return -1;
ret = read(fd, buffer, toread+1);
if (ret < 0) {
free(buffer);
return -1;
}
totalread = (size_t) ret;
if (totalread < toread + 1)
goto done;
do {
char *tmp;
toread = filesize;
filesize *= 2;
tmp = realloc(buffer, filesize+1);
if (!tmp) {
free(buffer);
return -1;
}
buffer = tmp;
ret = read(fd, buffer+toread+1, toread);
if (ret < 0) {
free(buffer);
return -1;
}
totalread += ret;
} while ((size_t) ret == toread);
done:
buffer[totalread] = '\0';
*bufferp = buffer;
*sizep = filesize;
return 0;
}
#define KERNEL_CPU_MASK_BITS 32
#define KERNEL_CPU_MAP_LEN (KERNEL_CPU_MASK_BITS/4+2)
static __hwloc_inline int
hwloc__read_fd_as_cpumask(int fd, hwloc_bitmap_t set)
{
static size_t _filesize = 0; 
size_t filesize;
unsigned long *maps;
unsigned long map;
int nr_maps = 0;
static int _nr_maps_allocated = 8; 
int nr_maps_allocated = _nr_maps_allocated;
char *buffer, *tmpbuf;
int i;
filesize = _filesize;
if (!filesize)
filesize = hwloc_getpagesize();
if (hwloc__read_fd(fd, &buffer, &filesize) < 0)
return -1;
_filesize = filesize;
maps = malloc(nr_maps_allocated * sizeof(*maps));
if (!maps) {
free(buffer);
return -1;
}
hwloc_bitmap_zero(set);
tmpbuf = buffer;
while (sscanf(tmpbuf, "%lx", &map) == 1) {
if (nr_maps == nr_maps_allocated) {
unsigned long *tmp = realloc(maps, 2*nr_maps_allocated * sizeof(*maps));
if (!tmp) {
free(buffer);
free(maps);
return -1;
}
maps = tmp;
nr_maps_allocated *= 2;
}
tmpbuf = strchr(tmpbuf, ',');
if (!tmpbuf) {
maps[nr_maps++] = map;
break;
} else
tmpbuf++;
if (!map && !nr_maps)
continue;
maps[nr_maps++] = map;
}
free(buffer);
#if KERNEL_CPU_MASK_BITS == HWLOC_BITS_PER_LONG
for(i=0; i<nr_maps; i++)
hwloc_bitmap_set_ith_ulong(set, i, maps[nr_maps-1-i]);
#else
for(i=0; i<(nr_maps+1)/2; i++) {
unsigned long mask;
mask = maps[nr_maps-2*i-1];
if (2*i+1<nr_maps)
mask |= maps[nr_maps-2*i-2] << KERNEL_CPU_MASK_BITS;
hwloc_bitmap_set_ith_ulong(set, i, mask);
}
#endif
free(maps);
if (nr_maps_allocated > _nr_maps_allocated)
_nr_maps_allocated = nr_maps_allocated;
return 0;
}
static __hwloc_inline int
hwloc__read_path_as_cpumask(const char *maskpath, hwloc_bitmap_t set, int fsroot_fd)
{
int fd, err;
fd = hwloc_open(maskpath, fsroot_fd);
if (fd < 0)
return -1;
err = hwloc__read_fd_as_cpumask(fd, set);
close(fd);
return err;
}
static __hwloc_inline hwloc_bitmap_t
hwloc__alloc_read_path_as_cpumask(const char *maskpath, int fsroot_fd)
{
hwloc_bitmap_t set;
int err;
set = hwloc_bitmap_alloc();
if (!set)
return NULL;
err = hwloc__read_path_as_cpumask(maskpath, set, fsroot_fd);
if (err < 0) {
hwloc_bitmap_free(set);
return NULL;
} else
return set;
}
int
hwloc_linux_read_path_as_cpumask(const char *maskpath, hwloc_bitmap_t set)
{
int fd, err;
fd = open(maskpath, O_RDONLY);
if (fd < 0)
return -1;
err = hwloc__read_fd_as_cpumask(fd, set);
close(fd);
return err;
}
static __hwloc_inline int
hwloc__read_fd_as_cpulist(int fd, hwloc_bitmap_t set)
{
size_t filesize = hwloc_getpagesize();
char *buffer, *current, *comma, *tmp;
int prevlast, nextfirst, nextlast; 
if (hwloc__read_fd(fd, &buffer, &filesize) < 0)
return -1;
hwloc_bitmap_fill(set);
current = buffer;
prevlast = -1;
while (1) {
comma = strchr(current, ',');
if (comma)
*comma = '\0';
nextfirst = strtoul(current, &tmp, 0);
if (*tmp == '-')
nextlast = strtoul(tmp+1, NULL, 0);
else
nextlast = nextfirst;
if (prevlast+1 <= nextfirst-1)
hwloc_bitmap_clr_range(set, prevlast+1, nextfirst-1);
prevlast = nextlast;
if (!comma)
break;
current = comma+1;
}
hwloc_bitmap_clr_range(set, prevlast+1, -1);
free(buffer);
return 0;
}
static __hwloc_inline int
hwloc__read_path_as_cpulist(const char *maskpath, hwloc_bitmap_t set, int fsroot_fd)
{
int fd, err;
fd = hwloc_open(maskpath, fsroot_fd);
if (fd < 0)
return -1;
err = hwloc__read_fd_as_cpulist(fd, set);
close(fd);
return err;
}
static __hwloc_inline hwloc_bitmap_t
hwloc__alloc_read_path_as_cpulist(const char *maskpath, int fsroot_fd)
{
hwloc_bitmap_t set;
int err;
set = hwloc_bitmap_alloc_full();
if (!set)
return NULL;
err = hwloc__read_path_as_cpulist(maskpath, set, fsroot_fd);
if (err < 0) {
hwloc_bitmap_free(set);
return NULL;
} else
return set;
}
int
hwloc_linux_set_tid_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, pid_t tid __hwloc_attribute_unused, hwloc_const_bitmap_t hwloc_set __hwloc_attribute_unused)
{
#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
cpu_set_t *plinux_set;
unsigned cpu;
int last;
size_t setsize;
int err;
last = hwloc_bitmap_last(hwloc_set);
if (last == -1) {
errno = EINVAL;
return -1;
}
setsize = CPU_ALLOC_SIZE(last+1);
plinux_set = CPU_ALLOC(last+1);
CPU_ZERO_S(setsize, plinux_set);
hwloc_bitmap_foreach_begin(cpu, hwloc_set)
CPU_SET_S(cpu, setsize, plinux_set);
hwloc_bitmap_foreach_end();
err = sched_setaffinity(tid, setsize, plinux_set);
CPU_FREE(plinux_set);
return err;
#elif defined(HWLOC_HAVE_CPU_SET)
cpu_set_t linux_set;
unsigned cpu;
CPU_ZERO(&linux_set);
hwloc_bitmap_foreach_begin(cpu, hwloc_set)
CPU_SET(cpu, &linux_set);
hwloc_bitmap_foreach_end();
#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
return sched_setaffinity(tid, &linux_set);
#else 
return sched_setaffinity(tid, sizeof(linux_set), &linux_set);
#endif 
#elif defined(HWLOC_HAVE_SYSCALL)
unsigned long mask = hwloc_bitmap_to_ulong(hwloc_set);
#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
return sched_setaffinity(tid, (void*) &mask);
#else 
return sched_setaffinity(tid, sizeof(mask), (void*) &mask);
#endif 
#else 
errno = ENOSYS;
return -1;
#endif 
}
#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
static int
hwloc_linux_find_kernel_nr_cpus(hwloc_topology_t topology)
{
static int _nr_cpus = -1;
int nr_cpus = _nr_cpus;
int fd;
if (nr_cpus != -1)
return nr_cpus;
if (topology->levels[0][0]->complete_cpuset)
nr_cpus = hwloc_bitmap_last(topology->levels[0][0]->complete_cpuset) + 1;
if (nr_cpus <= 0)
nr_cpus = 1;
fd = open("/sys/devices/system/cpu/possible", O_RDONLY); 
if (fd >= 0) {
hwloc_bitmap_t possible_bitmap = hwloc_bitmap_alloc();
if (hwloc__read_fd_as_cpulist(fd, possible_bitmap) == 0) {
int max_possible = hwloc_bitmap_last(possible_bitmap);
hwloc_debug_bitmap("possible CPUs are %s\n", possible_bitmap);
if (nr_cpus < max_possible + 1)
nr_cpus = max_possible + 1;
}
close(fd);
hwloc_bitmap_free(possible_bitmap);
}
while (1) {
cpu_set_t *set = CPU_ALLOC(nr_cpus);
size_t setsize = CPU_ALLOC_SIZE(nr_cpus);
int err = sched_getaffinity(0, setsize, set); 
CPU_FREE(set);
nr_cpus = setsize * 8; 
if (!err)
return _nr_cpus = nr_cpus;
nr_cpus *= 2;
}
}
#endif
int
hwloc_linux_get_tid_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, pid_t tid __hwloc_attribute_unused, hwloc_bitmap_t hwloc_set __hwloc_attribute_unused)
{
int err __hwloc_attribute_unused;
#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
cpu_set_t *plinux_set;
unsigned cpu;
int last;
size_t setsize;
int kernel_nr_cpus;
kernel_nr_cpus = hwloc_linux_find_kernel_nr_cpus(topology);
setsize = CPU_ALLOC_SIZE(kernel_nr_cpus);
plinux_set = CPU_ALLOC(kernel_nr_cpus);
err = sched_getaffinity(tid, setsize, plinux_set);
if (err < 0) {
CPU_FREE(plinux_set);
return -1;
}
last = -1;
if (topology->levels[0][0]->complete_cpuset)
last = hwloc_bitmap_last(topology->levels[0][0]->complete_cpuset);
if (last == -1)
last = kernel_nr_cpus-1;
hwloc_bitmap_zero(hwloc_set);
for(cpu=0; cpu<=(unsigned) last; cpu++)
if (CPU_ISSET_S(cpu, setsize, plinux_set))
hwloc_bitmap_set(hwloc_set, cpu);
CPU_FREE(plinux_set);
#elif defined(HWLOC_HAVE_CPU_SET)
cpu_set_t linux_set;
unsigned cpu;
#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
err = sched_getaffinity(tid, &linux_set);
#else 
err = sched_getaffinity(tid, sizeof(linux_set), &linux_set);
#endif 
if (err < 0)
return -1;
hwloc_bitmap_zero(hwloc_set);
for(cpu=0; cpu<CPU_SETSIZE; cpu++)
if (CPU_ISSET(cpu, &linux_set))
hwloc_bitmap_set(hwloc_set, cpu);
#elif defined(HWLOC_HAVE_SYSCALL)
unsigned long mask;
#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
err = sched_getaffinity(tid, (void*) &mask);
#else 
err = sched_getaffinity(tid, sizeof(mask), (void*) &mask);
#endif 
if (err < 0)
return -1;
hwloc_bitmap_from_ulong(hwloc_set, mask);
#else 
errno = ENOSYS;
return -1;
#endif 
return 0;
}
static int
hwloc_linux_get_proc_tids(DIR *taskdir, unsigned *nr_tidsp, pid_t ** tidsp)
{
struct dirent *dirent;
unsigned nr_tids = 0;
unsigned max_tids = 32;
pid_t *tids;
struct stat sb;
if (fstat(dirfd(taskdir), &sb) == 0)
max_tids = sb.st_nlink;
tids = malloc(max_tids*sizeof(pid_t));
if (!tids) {
errno = ENOMEM;
return -1;
}
rewinddir(taskdir);
while ((dirent = readdir(taskdir)) != NULL) {
if (nr_tids == max_tids) {
pid_t *newtids;
max_tids += 8;
newtids = realloc(tids, max_tids*sizeof(pid_t));
if (!newtids) {
free(tids);
errno = ENOMEM;
return -1;
}
tids = newtids;
}
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
tids[nr_tids++] = atoi(dirent->d_name);
}
*nr_tidsp = nr_tids;
*tidsp = tids;
return 0;
}
typedef int (*hwloc_linux_foreach_proc_tid_cb_t)(hwloc_topology_t topology, pid_t tid, void *data, int idx);
static int
hwloc_linux_foreach_proc_tid(hwloc_topology_t topology,
pid_t pid, hwloc_linux_foreach_proc_tid_cb_t cb,
void *data)
{
char taskdir_path[128];
DIR *taskdir;
pid_t *tids, *newtids;
unsigned i, nr, newnr, failed = 0, failed_errno = 0;
unsigned retrynr = 0;
int err;
if (pid)
snprintf(taskdir_path, sizeof(taskdir_path), "/proc/%u/task", (unsigned) pid);
else
snprintf(taskdir_path, sizeof(taskdir_path), "/proc/self/task");
taskdir = opendir(taskdir_path);
if (!taskdir) {
if (errno == ENOENT)
errno = EINVAL;
err = -1;
goto out;
}
err = hwloc_linux_get_proc_tids(taskdir, &nr, &tids);
if (err < 0)
goto out_with_dir;
retry:
failed=0;
for(i=0; i<nr; i++) {
err = cb(topology, tids[i], data, i);
if (err < 0) {
failed++;
failed_errno = errno;
}
}
err = hwloc_linux_get_proc_tids(taskdir, &newnr, &newtids);
if (err < 0)
goto out_with_tids;
if (newnr != nr || memcmp(newtids, tids, nr*sizeof(pid_t)) || (failed && failed != nr)) {
free(tids);
tids = newtids;
nr = newnr;
if (++retrynr > 10) {
errno = EAGAIN;
err = -1;
goto out_with_tids;
}
goto retry;
} else {
free(newtids);
}
if (failed) {
err = -1;
errno = failed_errno;
goto out_with_tids;
}
err = 0;
out_with_tids:
free(tids);
out_with_dir:
closedir(taskdir);
out:
return err;
}
static int
hwloc_linux_foreach_proc_tid_set_cpubind_cb(hwloc_topology_t topology, pid_t tid, void *data, int idx __hwloc_attribute_unused)
{
return hwloc_linux_set_tid_cpubind(topology, tid, (hwloc_bitmap_t) data);
}
static int
hwloc_linux_set_pid_cpubind(hwloc_topology_t topology, pid_t pid, hwloc_const_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
return hwloc_linux_foreach_proc_tid(topology, pid,
hwloc_linux_foreach_proc_tid_set_cpubind_cb,
(void*) hwloc_set);
}
struct hwloc_linux_foreach_proc_tid_get_cpubind_cb_data_s {
hwloc_bitmap_t cpuset;
hwloc_bitmap_t tidset;
int flags;
};
static int
hwloc_linux_foreach_proc_tid_get_cpubind_cb(hwloc_topology_t topology, pid_t tid, void *_data, int idx)
{
struct hwloc_linux_foreach_proc_tid_get_cpubind_cb_data_s *data = _data;
hwloc_bitmap_t cpuset = data->cpuset;
hwloc_bitmap_t tidset = data->tidset;
int flags = data->flags;
if (hwloc_linux_get_tid_cpubind(topology, tid, tidset))
return -1;
if (!idx)
hwloc_bitmap_zero(cpuset);
if (flags & HWLOC_CPUBIND_STRICT) {
if (!idx) {
hwloc_bitmap_copy(cpuset, tidset);
} else if (!hwloc_bitmap_isequal(cpuset, tidset)) {
errno = EXDEV;
return -1;
}
} else {
hwloc_bitmap_or(cpuset, cpuset, tidset);
}
return 0;
}
static int
hwloc_linux_get_pid_cpubind(hwloc_topology_t topology, pid_t pid, hwloc_bitmap_t hwloc_set, int flags)
{
struct hwloc_linux_foreach_proc_tid_get_cpubind_cb_data_s data;
hwloc_bitmap_t tidset = hwloc_bitmap_alloc();
int ret;
data.cpuset = hwloc_set;
data.tidset = tidset;
data.flags = flags;
ret = hwloc_linux_foreach_proc_tid(topology, pid,
hwloc_linux_foreach_proc_tid_get_cpubind_cb,
(void*) &data);
hwloc_bitmap_free(tidset);
return ret;
}
static int
hwloc_linux_set_proc_cpubind(hwloc_topology_t topology, pid_t pid, hwloc_const_bitmap_t hwloc_set, int flags)
{
if (pid == 0)
pid = topology->pid;
if (flags & HWLOC_CPUBIND_THREAD)
return hwloc_linux_set_tid_cpubind(topology, pid, hwloc_set);
else
return hwloc_linux_set_pid_cpubind(topology, pid, hwloc_set, flags);
}
static int
hwloc_linux_get_proc_cpubind(hwloc_topology_t topology, pid_t pid, hwloc_bitmap_t hwloc_set, int flags)
{
if (pid == 0)
pid = topology->pid;
if (flags & HWLOC_CPUBIND_THREAD)
return hwloc_linux_get_tid_cpubind(topology, pid, hwloc_set);
else
return hwloc_linux_get_pid_cpubind(topology, pid, hwloc_set, flags);
}
static int
hwloc_linux_set_thisproc_cpubind(hwloc_topology_t topology, hwloc_const_bitmap_t hwloc_set, int flags)
{
return hwloc_linux_set_pid_cpubind(topology, topology->pid, hwloc_set, flags);
}
static int
hwloc_linux_get_thisproc_cpubind(hwloc_topology_t topology, hwloc_bitmap_t hwloc_set, int flags)
{
return hwloc_linux_get_pid_cpubind(topology, topology->pid, hwloc_set, flags);
}
static int
hwloc_linux_set_thisthread_cpubind(hwloc_topology_t topology, hwloc_const_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
if (topology->pid) {
errno = ENOSYS;
return -1;
}
return hwloc_linux_set_tid_cpubind(topology, 0, hwloc_set);
}
static int
hwloc_linux_get_thisthread_cpubind(hwloc_topology_t topology, hwloc_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
if (topology->pid) {
errno = ENOSYS;
return -1;
}
return hwloc_linux_get_tid_cpubind(topology, 0, hwloc_set);
}
#if HAVE_DECL_PTHREAD_SETAFFINITY_NP
#pragma weak pthread_setaffinity_np
#pragma weak pthread_self
static int
hwloc_linux_set_thread_cpubind(hwloc_topology_t topology, pthread_t tid, hwloc_const_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
int err;
if (topology->pid) {
errno = ENOSYS;
return -1;
}
if (!pthread_self) {
errno = ENOSYS;
return -1;
}
if (tid == pthread_self())
return hwloc_linux_set_tid_cpubind(topology, 0, hwloc_set);
if (!pthread_setaffinity_np) {
errno = ENOSYS;
return -1;
}
#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
{
cpu_set_t *plinux_set;
unsigned cpu;
int last;
size_t setsize;
last = hwloc_bitmap_last(hwloc_set);
if (last == -1) {
errno = EINVAL;
return -1;
}
setsize = CPU_ALLOC_SIZE(last+1);
plinux_set = CPU_ALLOC(last+1);
CPU_ZERO_S(setsize, plinux_set);
hwloc_bitmap_foreach_begin(cpu, hwloc_set)
CPU_SET_S(cpu, setsize, plinux_set);
hwloc_bitmap_foreach_end();
err = pthread_setaffinity_np(tid, setsize, plinux_set);
CPU_FREE(plinux_set);
}
#elif defined(HWLOC_HAVE_CPU_SET)
{
cpu_set_t linux_set;
unsigned cpu;
CPU_ZERO(&linux_set);
hwloc_bitmap_foreach_begin(cpu, hwloc_set)
CPU_SET(cpu, &linux_set);
hwloc_bitmap_foreach_end();
#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
err = pthread_setaffinity_np(tid, &linux_set);
#else 
err = pthread_setaffinity_np(tid, sizeof(linux_set), &linux_set);
#endif 
}
#else 
{
unsigned long mask = hwloc_bitmap_to_ulong(hwloc_set);
#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
err = pthread_setaffinity_np(tid, (void*) &mask);
#else 
err = pthread_setaffinity_np(tid, sizeof(mask), (void*) &mask);
#endif 
}
#endif 
if (err) {
errno = err;
return -1;
}
return 0;
}
#endif 
#if HAVE_DECL_PTHREAD_GETAFFINITY_NP
#pragma weak pthread_getaffinity_np
#pragma weak pthread_self
static int
hwloc_linux_get_thread_cpubind(hwloc_topology_t topology, pthread_t tid, hwloc_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
int err;
if (topology->pid) {
errno = ENOSYS;
return -1;
}
if (!pthread_self) {
errno = ENOSYS;
return -1;
}
if (tid == pthread_self())
return hwloc_linux_get_tid_cpubind(topology, 0, hwloc_set);
if (!pthread_getaffinity_np) {
errno = ENOSYS;
return -1;
}
#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
{
cpu_set_t *plinux_set;
unsigned cpu;
int last;
size_t setsize;
last = hwloc_bitmap_last(topology->levels[0][0]->complete_cpuset);
assert (last != -1);
setsize = CPU_ALLOC_SIZE(last+1);
plinux_set = CPU_ALLOC(last+1);
err = pthread_getaffinity_np(tid, setsize, plinux_set);
if (err) {
CPU_FREE(plinux_set);
errno = err;
return -1;
}
hwloc_bitmap_zero(hwloc_set);
for(cpu=0; cpu<=(unsigned) last; cpu++)
if (CPU_ISSET_S(cpu, setsize, plinux_set))
hwloc_bitmap_set(hwloc_set, cpu);
CPU_FREE(plinux_set);
}
#elif defined(HWLOC_HAVE_CPU_SET)
{
cpu_set_t linux_set;
unsigned cpu;
#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
err = pthread_getaffinity_np(tid, &linux_set);
#else 
err = pthread_getaffinity_np(tid, sizeof(linux_set), &linux_set);
#endif 
if (err) {
errno = err;
return -1;
}
hwloc_bitmap_zero(hwloc_set);
for(cpu=0; cpu<CPU_SETSIZE; cpu++)
if (CPU_ISSET(cpu, &linux_set))
hwloc_bitmap_set(hwloc_set, cpu);
}
#else 
{
unsigned long mask;
#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
err = pthread_getaffinity_np(tid, (void*) &mask);
#else 
err = pthread_getaffinity_np(tid, sizeof(mask), (void*) &mask);
#endif 
if (err) {
errno = err;
return -1;
}
hwloc_bitmap_from_ulong(hwloc_set, mask);
}
#endif 
return 0;
}
#endif 
int
hwloc_linux_get_tid_last_cpu_location(hwloc_topology_t topology __hwloc_attribute_unused, pid_t tid, hwloc_bitmap_t set)
{
char buf[1024] = "";
char name[64];
char *tmp;
int fd, i, err;
if (!tid) {
#ifdef SYS_gettid
tid = syscall(SYS_gettid);
#else
errno = ENOSYS;
return -1;
#endif
}
snprintf(name, sizeof(name), "/proc/%lu/stat", (unsigned long) tid);
fd = open(name, O_RDONLY); 
if (fd < 0) {
errno = ENOSYS;
return -1;
}
err = read(fd, buf, sizeof(buf)-1); 
close(fd);
if (err <= 0) {
errno = ENOSYS;
return -1;
}
buf[err-1] = '\0';
tmp = strrchr(buf, ')');
if (!tmp) {
errno = ENOSYS;
return -1;
}
tmp += 2;
for(i=0; i<36; i++) {
tmp = strchr(tmp, ' ');
if (!tmp) {
errno = ENOSYS;
return -1;
}
tmp++;
}
if (sscanf(tmp, "%d ", &i) != 1) {
errno = ENOSYS;
return -1;
}
hwloc_bitmap_only(set, i);
return 0;
}
struct hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb_data_s {
hwloc_bitmap_t cpuset;
hwloc_bitmap_t tidset;
};
static int
hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb(hwloc_topology_t topology, pid_t tid, void *_data, int idx)
{
struct hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb_data_s *data = _data;
hwloc_bitmap_t cpuset = data->cpuset;
hwloc_bitmap_t tidset = data->tidset;
if (hwloc_linux_get_tid_last_cpu_location(topology, tid, tidset))
return -1;
if (!idx)
hwloc_bitmap_zero(cpuset);
hwloc_bitmap_or(cpuset, cpuset, tidset);
return 0;
}
static int
hwloc_linux_get_pid_last_cpu_location(hwloc_topology_t topology, pid_t pid, hwloc_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
struct hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb_data_s data;
hwloc_bitmap_t tidset = hwloc_bitmap_alloc();
int ret;
data.cpuset = hwloc_set;
data.tidset = tidset;
ret = hwloc_linux_foreach_proc_tid(topology, pid,
hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb,
&data);
hwloc_bitmap_free(tidset);
return ret;
}
static int
hwloc_linux_get_proc_last_cpu_location(hwloc_topology_t topology, pid_t pid, hwloc_bitmap_t hwloc_set, int flags)
{
if (pid == 0)
pid = topology->pid;
if (flags & HWLOC_CPUBIND_THREAD)
return hwloc_linux_get_tid_last_cpu_location(topology, pid, hwloc_set);
else
return hwloc_linux_get_pid_last_cpu_location(topology, pid, hwloc_set, flags);
}
static int
hwloc_linux_get_thisproc_last_cpu_location(hwloc_topology_t topology, hwloc_bitmap_t hwloc_set, int flags)
{
return hwloc_linux_get_pid_last_cpu_location(topology, topology->pid, hwloc_set, flags);
}
static int
hwloc_linux_get_thisthread_last_cpu_location(hwloc_topology_t topology, hwloc_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
if (topology->pid) {
errno = ENOSYS;
return -1;
}
#if HAVE_DECL_SCHED_GETCPU
{
int pu = sched_getcpu();
if (pu >= 0) {
hwloc_bitmap_only(hwloc_set, pu);
return 0;
}
}
#endif
return hwloc_linux_get_tid_last_cpu_location(topology, 0, hwloc_set);
}
static int
hwloc_linux_membind_policy_from_hwloc(int *linuxpolicy, hwloc_membind_policy_t policy, int flags)
{
switch (policy) {
case HWLOC_MEMBIND_DEFAULT:
*linuxpolicy = MPOL_DEFAULT;
break;
case HWLOC_MEMBIND_FIRSTTOUCH:
*linuxpolicy = MPOL_LOCAL;
break;
case HWLOC_MEMBIND_BIND:
if (flags & HWLOC_MEMBIND_STRICT)
*linuxpolicy = MPOL_BIND;
else
*linuxpolicy = MPOL_PREFERRED;
break;
case HWLOC_MEMBIND_INTERLEAVE:
*linuxpolicy = MPOL_INTERLEAVE;
break;
default:
errno = ENOSYS;
return -1;
}
return 0;
}
static int
hwloc_linux_membind_mask_from_nodeset(hwloc_topology_t topology __hwloc_attribute_unused,
hwloc_const_nodeset_t nodeset,
unsigned *max_os_index_p, unsigned long **linuxmaskp)
{
unsigned max_os_index = 0; 
unsigned long *linuxmask;
unsigned i;
hwloc_nodeset_t linux_nodeset = NULL;
if (hwloc_bitmap_isfull(nodeset)) {
linux_nodeset = hwloc_bitmap_alloc();
hwloc_bitmap_only(linux_nodeset, 0);
nodeset = linux_nodeset;
}
max_os_index = hwloc_bitmap_last(nodeset);
if (max_os_index == (unsigned) -1)
max_os_index = 0;
max_os_index = (max_os_index + 1 + HWLOC_BITS_PER_LONG - 1) & ~(HWLOC_BITS_PER_LONG - 1);
linuxmask = calloc(max_os_index/HWLOC_BITS_PER_LONG, sizeof(unsigned long));
if (!linuxmask) {
hwloc_bitmap_free(linux_nodeset);
errno = ENOMEM;
return -1;
}
for(i=0; i<max_os_index/HWLOC_BITS_PER_LONG; i++)
linuxmask[i] = hwloc_bitmap_to_ith_ulong(nodeset, i);
if (linux_nodeset)
hwloc_bitmap_free(linux_nodeset);
*max_os_index_p = max_os_index;
*linuxmaskp = linuxmask;
return 0;
}
static void
hwloc_linux_membind_mask_to_nodeset(hwloc_topology_t topology __hwloc_attribute_unused,
hwloc_nodeset_t nodeset,
unsigned max_os_index, const unsigned long *linuxmask)
{
unsigned i;
#ifdef HWLOC_DEBUG
assert(!(max_os_index%HWLOC_BITS_PER_LONG));
#endif
hwloc_bitmap_zero(nodeset);
for(i=0; i<max_os_index/HWLOC_BITS_PER_LONG; i++)
hwloc_bitmap_set_ith_ulong(nodeset, i, linuxmask[i]);
}
static int
hwloc_linux_set_area_membind(hwloc_topology_t topology, const void *addr, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
unsigned max_os_index; 
unsigned long *linuxmask;
size_t remainder;
int linuxpolicy;
unsigned linuxflags = 0;
int err;
remainder = (uintptr_t) addr & (hwloc_getpagesize()-1);
addr = (char*) addr - remainder;
len += remainder;
err = hwloc_linux_membind_policy_from_hwloc(&linuxpolicy, policy, flags);
if (err < 0)
return err;
if (linuxpolicy == MPOL_DEFAULT) {
return hwloc_mbind((void *) addr, len, linuxpolicy, NULL, 0, 0);
} else if (linuxpolicy == MPOL_LOCAL) {
if (!hwloc_bitmap_isequal(nodeset, hwloc_topology_get_complete_nodeset(topology))) {
errno = EXDEV;
return -1;
}
return hwloc_mbind((void *) addr, len, MPOL_PREFERRED, NULL, 0, 0);
}
err = hwloc_linux_membind_mask_from_nodeset(topology, nodeset, &max_os_index, &linuxmask);
if (err < 0)
goto out;
if (flags & HWLOC_MEMBIND_MIGRATE) {
linuxflags = MPOL_MF_MOVE;
if (flags & HWLOC_MEMBIND_STRICT)
linuxflags |= MPOL_MF_STRICT;
}
err = hwloc_mbind((void *) addr, len, linuxpolicy, linuxmask, max_os_index+1, linuxflags);
if (err < 0)
goto out_with_mask;
free(linuxmask);
return 0;
out_with_mask:
free(linuxmask);
out:
return -1;
}
static void *
hwloc_linux_alloc_membind(hwloc_topology_t topology, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
void *buffer;
int err;
buffer = hwloc_alloc_mmap(topology, len);
if (!buffer)
return NULL;
err = hwloc_linux_set_area_membind(topology, buffer, len, nodeset, policy, flags);
if (err < 0 && (flags & HWLOC_MEMBIND_STRICT)) {
munmap(buffer, len);
return NULL;
}
return buffer;
}
static int
hwloc_linux_set_thisthread_membind(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
unsigned max_os_index; 
unsigned long *linuxmask;
int linuxpolicy;
int err;
err = hwloc_linux_membind_policy_from_hwloc(&linuxpolicy, policy, flags);
if (err < 0)
return err;
if (linuxpolicy == MPOL_DEFAULT) {
return hwloc_set_mempolicy(linuxpolicy, NULL, 0);
} else if (linuxpolicy == MPOL_LOCAL) {
if (!hwloc_bitmap_isequal(nodeset, hwloc_topology_get_complete_nodeset(topology))) {
errno = EXDEV;
return -1;
}
return hwloc_set_mempolicy(MPOL_PREFERRED, NULL, 0);
}
err = hwloc_linux_membind_mask_from_nodeset(topology, nodeset, &max_os_index, &linuxmask);
if (err < 0)
goto out;
if (flags & HWLOC_MEMBIND_MIGRATE) {
unsigned long *fullmask;
fullmask = malloc(max_os_index/HWLOC_BITS_PER_LONG * sizeof(*fullmask));
if (!fullmask)
goto out_with_mask;
memset(fullmask, 0xf, max_os_index/HWLOC_BITS_PER_LONG * sizeof(unsigned long));
err = hwloc_migrate_pages(0, max_os_index+1, fullmask, linuxmask); 
free(fullmask);
if (err < 0 && (flags & HWLOC_MEMBIND_STRICT))
goto out_with_mask;
}
err = hwloc_set_mempolicy(linuxpolicy, linuxmask, max_os_index+1);
if (err < 0)
goto out_with_mask;
free(linuxmask);
return 0;
out_with_mask:
free(linuxmask);
out:
return -1;
}
static int
hwloc_linux_find_kernel_max_numnodes(hwloc_topology_t topology __hwloc_attribute_unused)
{
static int _max_numnodes = -1, max_numnodes;
int linuxpolicy;
int fd;
if (_max_numnodes != -1)
return _max_numnodes;
max_numnodes = HWLOC_BITS_PER_LONG;
fd = open("/sys/devices/system/node/possible", O_RDONLY); 
if (fd >= 0) {
hwloc_bitmap_t possible_bitmap = hwloc_bitmap_alloc();
if (hwloc__read_fd_as_cpulist(fd, possible_bitmap) == 0) {
int max_possible = hwloc_bitmap_last(possible_bitmap);
hwloc_debug_bitmap("possible NUMA nodes are %s\n", possible_bitmap);
if (max_numnodes < max_possible + 1)
max_numnodes = max_possible + 1;
}
close(fd);
hwloc_bitmap_free(possible_bitmap);
}
while (1) {
unsigned long *mask;
int err;
mask = malloc(max_numnodes / HWLOC_BITS_PER_LONG * sizeof(*mask));
if (!mask)
return _max_numnodes = max_numnodes;
err = hwloc_get_mempolicy(&linuxpolicy, mask, max_numnodes, 0, 0);
free(mask);
if (!err || errno != EINVAL)
return _max_numnodes = max_numnodes;
max_numnodes *= 2;
}
}
static int
hwloc_linux_membind_policy_to_hwloc(int linuxpolicy, hwloc_membind_policy_t *policy)
{
switch (linuxpolicy) {
case MPOL_DEFAULT:
case MPOL_LOCAL: 
*policy = HWLOC_MEMBIND_FIRSTTOUCH;
return 0;
case MPOL_PREFERRED:
case MPOL_BIND:
*policy = HWLOC_MEMBIND_BIND;
return 0;
case MPOL_INTERLEAVE:
*policy = HWLOC_MEMBIND_INTERLEAVE;
return 0;
default:
errno = EINVAL;
return -1;
}
}
static int hwloc_linux_mask_is_empty(unsigned max_os_index, unsigned long *linuxmask)
{
unsigned i;
for(i=0; i<max_os_index/HWLOC_BITS_PER_LONG; i++)
if (linuxmask[i])
return 0;
return 1;
}
static int
hwloc_linux_get_thisthread_membind(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t *policy, int flags __hwloc_attribute_unused)
{
unsigned max_os_index;
unsigned long *linuxmask;
int linuxpolicy;
int err;
max_os_index = hwloc_linux_find_kernel_max_numnodes(topology);
linuxmask = malloc(max_os_index/HWLOC_BITS_PER_LONG * sizeof(*linuxmask));;
if (!linuxmask)
goto out;
err = hwloc_get_mempolicy(&linuxpolicy, linuxmask, max_os_index, 0, 0);
if (err < 0)
goto out_with_linuxmask;
if (linuxpolicy == MPOL_PREFERRED && hwloc_linux_mask_is_empty(max_os_index, linuxmask))
linuxpolicy = MPOL_LOCAL;
if (linuxpolicy == MPOL_DEFAULT || linuxpolicy == MPOL_LOCAL) {
hwloc_bitmap_copy(nodeset, hwloc_topology_get_topology_nodeset(topology));
} else {
hwloc_linux_membind_mask_to_nodeset(topology, nodeset, max_os_index, linuxmask);
}
err = hwloc_linux_membind_policy_to_hwloc(linuxpolicy, policy);
if (err < 0)
goto out_with_linuxmask;
free(linuxmask);
return 0;
out_with_linuxmask:
free(linuxmask);
out:
return -1;
}
static int
hwloc_linux_get_area_membind(hwloc_topology_t topology, const void *addr, size_t len, hwloc_nodeset_t nodeset, hwloc_membind_policy_t *policy, int flags __hwloc_attribute_unused)
{
unsigned max_os_index;
unsigned long *linuxmask;
unsigned long *globallinuxmask;
int linuxpolicy = 0, globallinuxpolicy = 0; 
int mixed = 0;
int full = 0;
int first = 1;
int pagesize = hwloc_getpagesize();
char *tmpaddr;
int err;
unsigned i;
max_os_index = hwloc_linux_find_kernel_max_numnodes(topology);
linuxmask = malloc(max_os_index/HWLOC_BITS_PER_LONG * sizeof(*linuxmask));
globallinuxmask = malloc(max_os_index/HWLOC_BITS_PER_LONG * sizeof(*globallinuxmask));
if (!linuxmask || !globallinuxmask)
goto out_with_linuxmasks;
memset(globallinuxmask, 0, sizeof(*globallinuxmask));
for(tmpaddr = (char *)((unsigned long)addr & ~(pagesize-1));
tmpaddr < (char *)addr + len;
tmpaddr += pagesize) {
err = hwloc_get_mempolicy(&linuxpolicy, linuxmask, max_os_index, tmpaddr, MPOL_F_ADDR);
if (err < 0)
goto out_with_linuxmasks;
if (linuxpolicy == MPOL_PREFERRED && hwloc_linux_mask_is_empty(max_os_index, linuxmask))
linuxpolicy = MPOL_LOCAL;
if (first)
globallinuxpolicy = linuxpolicy;
else if (globallinuxpolicy != linuxpolicy)
mixed = 1;
if (full || linuxpolicy == MPOL_DEFAULT || linuxpolicy == MPOL_LOCAL) {
full = 1;
} else {
for(i=0; i<max_os_index/HWLOC_BITS_PER_LONG; i++)
globallinuxmask[i] |= linuxmask[i];
}
first = 0;
}
if (mixed) {
*policy = HWLOC_MEMBIND_MIXED;
} else {
err = hwloc_linux_membind_policy_to_hwloc(linuxpolicy, policy);
if (err < 0)
goto out_with_linuxmasks;
}
if (full) {
hwloc_bitmap_copy(nodeset, hwloc_topology_get_topology_nodeset(topology));
} else {
hwloc_linux_membind_mask_to_nodeset(topology, nodeset, max_os_index, globallinuxmask);
}
free(linuxmask);
free(globallinuxmask);
return 0;
out_with_linuxmasks:
free(linuxmask);
free(globallinuxmask);
return -1;
}
static int
hwloc_linux_get_area_memlocation(hwloc_topology_t topology __hwloc_attribute_unused, const void *addr, size_t len, hwloc_nodeset_t nodeset, int flags __hwloc_attribute_unused)
{
unsigned offset;
unsigned long count;
void **pages;
int *status;
int pagesize = hwloc_getpagesize();
int ret;
unsigned i;
offset = ((unsigned long) addr) & (pagesize-1);
addr = ((char*) addr) - offset;
len += offset;
count = (len + pagesize-1)/pagesize;
pages = malloc(count*sizeof(*pages));
status = malloc(count*sizeof(*status));
if (!pages || !status) {
ret = -1;
goto out_with_pages;
}
for(i=0; i<count; i++)
pages[i] = ((char*)addr) + i*pagesize;
ret = hwloc_move_pages(0, count, pages, NULL, status, 0);
if (ret  < 0)
goto out_with_pages;
hwloc_bitmap_zero(nodeset);
for(i=0; i<count; i++)
if (status[i] >= 0)
hwloc_bitmap_set(nodeset, status[i]);
ret = 0; 
out_with_pages:
free(pages);
free(status);
return ret;
}
static void hwloc_linux__get_allowed_resources(hwloc_topology_t topology, const char *root_path, int root_fd, char **cpuset_namep);
static int hwloc_linux_get_allowed_resources_hook(hwloc_topology_t topology)
{
const char *fsroot_path;
char *cpuset_name = NULL;
int root_fd = -1;
fsroot_path = getenv("HWLOC_FSROOT");
if (!fsroot_path)
fsroot_path = "/";
if (strcmp(fsroot_path, "/")) {
#ifdef HAVE_OPENAT
root_fd = open(fsroot_path, O_RDONLY | O_DIRECTORY);
if (root_fd < 0)
goto out;
#else
errno = ENOSYS;
goto out;
#endif
}
hwloc_linux__get_allowed_resources(topology, fsroot_path, root_fd, &cpuset_name);
if (cpuset_name) {
hwloc__add_info_nodup(&topology->levels[0][0]->infos, &topology->levels[0][0]->infos_count,
"LinuxCgroup", cpuset_name, 1 );
free(cpuset_name);
}
if (root_fd != -1)
close(root_fd);
out:
return -1;
}
void
hwloc_set_linuxfs_hooks(struct hwloc_binding_hooks *hooks,
struct hwloc_topology_support *support)
{
hooks->set_thisthread_cpubind = hwloc_linux_set_thisthread_cpubind;
hooks->get_thisthread_cpubind = hwloc_linux_get_thisthread_cpubind;
hooks->set_thisproc_cpubind = hwloc_linux_set_thisproc_cpubind;
hooks->get_thisproc_cpubind = hwloc_linux_get_thisproc_cpubind;
hooks->set_proc_cpubind = hwloc_linux_set_proc_cpubind;
hooks->get_proc_cpubind = hwloc_linux_get_proc_cpubind;
#if HAVE_DECL_PTHREAD_SETAFFINITY_NP
hooks->set_thread_cpubind = hwloc_linux_set_thread_cpubind;
#endif 
#if HAVE_DECL_PTHREAD_GETAFFINITY_NP
hooks->get_thread_cpubind = hwloc_linux_get_thread_cpubind;
#endif 
hooks->get_thisthread_last_cpu_location = hwloc_linux_get_thisthread_last_cpu_location;
hooks->get_thisproc_last_cpu_location = hwloc_linux_get_thisproc_last_cpu_location;
hooks->get_proc_last_cpu_location = hwloc_linux_get_proc_last_cpu_location;
hooks->set_thisthread_membind = hwloc_linux_set_thisthread_membind;
hooks->get_thisthread_membind = hwloc_linux_get_thisthread_membind;
hooks->get_area_membind = hwloc_linux_get_area_membind;
hooks->set_area_membind = hwloc_linux_set_area_membind;
hooks->get_area_memlocation = hwloc_linux_get_area_memlocation;
hooks->alloc_membind = hwloc_linux_alloc_membind;
hooks->alloc = hwloc_alloc_mmap;
hooks->free_membind = hwloc_free_mmap;
support->membind->firsttouch_membind = 1;
support->membind->bind_membind = 1;
support->membind->interleave_membind = 1;
support->membind->migrate_membind = 1;
hooks->get_allowed_resources = hwloc_linux_get_allowed_resources_hook;
}
struct hwloc_linux_cpuinfo_proc {
unsigned long Pproc;
struct hwloc_info_s *infos;
unsigned infos_count;
};
static void
hwloc_find_linux_cpuset_mntpnt(char **cgroup_mntpnt, char **cpuset_mntpnt, const char *root_path)
{
char *mount_path;
struct mntent mntent;
char *buf;
FILE *fd;
int err;
size_t bufsize;
*cgroup_mntpnt = NULL;
*cpuset_mntpnt = NULL;
if (root_path) {
err = asprintf(&mount_path, "%s/proc/mounts", root_path);
if (err < 0)
return;
fd = setmntent(mount_path, "r");
free(mount_path);
} else {
fd = setmntent("/proc/mounts", "r");
}
if (!fd)
return;
bufsize = hwloc_getpagesize()*4;
buf = malloc(bufsize);
if (!buf)
return;
while (getmntent_r(fd, &mntent, buf, bufsize)) {
if (!strcmp(mntent.mnt_type, "cpuset")) {
hwloc_debug("Found cpuset mount point on %s\n", mntent.mnt_dir);
*cpuset_mntpnt = strdup(mntent.mnt_dir);
break;
} else if (!strcmp(mntent.mnt_type, "cgroup")) {
char *opt, *opts = mntent.mnt_opts;
int cpuset_opt = 0;
int noprefix_opt = 0;
while ((opt = strsep(&opts, ",")) != NULL) {
if (!strcmp(opt, "cpuset"))
cpuset_opt = 1;
else if (!strcmp(opt, "noprefix"))
noprefix_opt = 1;
}
if (!cpuset_opt)
continue;
if (noprefix_opt) {
hwloc_debug("Found cgroup emulating a cpuset mount point on %s\n", mntent.mnt_dir);
*cpuset_mntpnt = strdup(mntent.mnt_dir);
} else {
hwloc_debug("Found cgroup/cpuset mount point on %s\n", mntent.mnt_dir);
*cgroup_mntpnt = strdup(mntent.mnt_dir);
}
break;
}
}
endmntent(fd);
free(buf);
}
static char *
hwloc_read_linux_cpuset_name(int fsroot_fd, hwloc_pid_t pid)
{
#define CPUSET_NAME_LEN 128
char cpuset_name[CPUSET_NAME_LEN];
FILE *file;
int err;
char *tmp;
if (!pid)
file = hwloc_fopen("/proc/self/cgroup", "r", fsroot_fd);
else {
char path[] = "/proc/XXXXXXXXXXX/cgroup";
snprintf(path, sizeof(path), "/proc/%d/cgroup", pid);
file = hwloc_fopen(path, "r", fsroot_fd);
}
if (file) {
#define CGROUP_LINE_LEN 256
char line[CGROUP_LINE_LEN];
while (fgets(line, sizeof(line), file)) {
char *end, *colon = strchr(line, ':');
if (!colon)
continue;
if (strncmp(colon, ":cpuset:", 8))
continue;
fclose(file);
end = strchr(colon, '\n');
if (end)
*end = '\0';
hwloc_debug("Found cgroup-cpuset %s\n", colon+8);
return strdup(colon+8);
}
fclose(file);
}
if (!pid)
err = hwloc_read_path_by_length("/proc/self/cpuset", cpuset_name, sizeof(cpuset_name), fsroot_fd);
else {
char path[] = "/proc/XXXXXXXXXXX/cpuset";
snprintf(path, sizeof(path), "/proc/%d/cpuset", pid);
err = hwloc_read_path_by_length(path, cpuset_name, sizeof(cpuset_name), fsroot_fd);
}
if (err < 0) {
hwloc_debug("%s", "No cgroup or cpuset found\n");
return NULL;
}
tmp = strchr(cpuset_name, '\n');
if (tmp)
*tmp = '\0';
hwloc_debug("Found cpuset %s\n", cpuset_name);
return strdup(cpuset_name);
}
static void
hwloc_admin_disable_set_from_cpuset(int root_fd,
const char *cgroup_mntpnt, const char *cpuset_mntpnt, const char *cpuset_name,
const char *attr_name,
hwloc_bitmap_t admin_enabled_set)
{
#define CPUSET_FILENAME_LEN 256
char cpuset_filename[CPUSET_FILENAME_LEN];
int err;
if (cgroup_mntpnt) {
snprintf(cpuset_filename, CPUSET_FILENAME_LEN, "%s%s/cpuset.%s", cgroup_mntpnt, cpuset_name, attr_name);
hwloc_debug("Trying to read cgroup file <%s>\n", cpuset_filename);
} else if (cpuset_mntpnt) {
snprintf(cpuset_filename, CPUSET_FILENAME_LEN, "%s%s/%s", cpuset_mntpnt, cpuset_name, attr_name);
hwloc_debug("Trying to read cpuset file <%s>\n", cpuset_filename);
}
err = hwloc__read_path_as_cpulist(cpuset_filename, admin_enabled_set, root_fd);
if (err < 0) {
hwloc_debug("failed to read cpuset '%s' attribute '%s'\n", cpuset_name, attr_name);
hwloc_bitmap_fill(admin_enabled_set);
} else {
hwloc_debug_bitmap("cpuset includes %s\n", admin_enabled_set);
}
}
static void
hwloc_parse_meminfo_info(struct hwloc_linux_backend_data_s *data,
const char *path,
uint64_t *local_memory)
{
char *tmp;
char buffer[4096];
unsigned long long number;
if (hwloc_read_path_by_length(path, buffer, sizeof(buffer), data->root_fd) < 0)
return;
tmp = strstr(buffer, "MemTotal: "); 
if (tmp) {
number = strtoull(tmp+10, NULL, 10);
*local_memory = number << 10;
}
}
#define SYSFS_NUMA_NODE_PATH_LEN 128
static void
hwloc_parse_hugepages_info(struct hwloc_linux_backend_data_s *data,
const char *dirpath,
struct hwloc_numanode_attr_s *memory,
uint64_t *remaining_local_memory)
{
DIR *dir;
struct dirent *dirent;
unsigned long index_ = 1; 
char line[64];
char path[SYSFS_NUMA_NODE_PATH_LEN];
dir = hwloc_opendir(dirpath, data->root_fd);
if (dir) {
while ((dirent = readdir(dir)) != NULL) {
int err;
if (strncmp(dirent->d_name, "hugepages-", 10))
continue;
memory->page_types[index_].size = strtoul(dirent->d_name+10, NULL, 0) * 1024ULL;
err = snprintf(path, sizeof(path), "%s/%s/nr_hugepages", dirpath, dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, line, sizeof(line), data->root_fd)) {
memory->page_types[index_].count = strtoull(line, NULL, 0);
*remaining_local_memory -= memory->page_types[index_].count * memory->page_types[index_].size;
index_++;
}
}
closedir(dir);
memory->page_types_len = index_;
}
}
static void
hwloc_get_machine_meminfo(struct hwloc_linux_backend_data_s *data,
struct hwloc_numanode_attr_s *memory)
{
struct stat st;
int has_sysfs_hugepages = 0;
int types = 1; 
uint64_t remaining_local_memory;
int err;
err = hwloc_stat("/sys/kernel/mm/hugepages", &st, data->root_fd);
if (!err) {
types = 1 + st.st_nlink-2;
has_sysfs_hugepages = 1;
}
memory->page_types = calloc(types, sizeof(*memory->page_types));
if (!memory->page_types) {
memory->page_types_len = 0;
return;
}
memory->page_types_len = 1; 
hwloc_parse_meminfo_info(data, "/proc/meminfo",
&memory->local_memory);
remaining_local_memory = memory->local_memory;
if (has_sysfs_hugepages) {
hwloc_parse_hugepages_info(data, "/sys/kernel/mm/hugepages", memory, &remaining_local_memory);
}
memory->page_types[0].size = data->pagesize;
memory->page_types[0].count = remaining_local_memory / memory->page_types[0].size;
}
static void
hwloc_get_sysfs_node_meminfo(struct hwloc_linux_backend_data_s *data,
const char *syspath, int node,
struct hwloc_numanode_attr_s *memory)
{
char path[SYSFS_NUMA_NODE_PATH_LEN];
char meminfopath[SYSFS_NUMA_NODE_PATH_LEN];
struct stat st;
int has_sysfs_hugepages = 0;
int types = 1; 
uint64_t remaining_local_memory;
int err;
sprintf(path, "%s/node%d/hugepages", syspath, node);
err = hwloc_stat(path, &st, data->root_fd);
if (!err) {
types = 1 + st.st_nlink-2;
has_sysfs_hugepages = 1;
}
memory->page_types = calloc(types, sizeof(*memory->page_types));
if (!memory->page_types) {
memory->page_types_len = 0;
return;
}
memory->page_types_len = 1; 
sprintf(meminfopath, "%s/node%d/meminfo", syspath, node);
hwloc_parse_meminfo_info(data, meminfopath,
&memory->local_memory);
remaining_local_memory = memory->local_memory;
if (has_sysfs_hugepages) {
hwloc_parse_hugepages_info(data, path, memory, &remaining_local_memory);
}
memory->page_types[0].size = data->pagesize;
memory->page_types[0].count = remaining_local_memory / memory->page_types[0].size;
}
static int
hwloc_parse_nodes_distances(const char *path, unsigned nbnodes, unsigned *indexes, uint64_t *distances, int fsroot_fd)
{
size_t len = (10+1)*nbnodes;
uint64_t *curdist = distances;
char *string;
unsigned i;
string = malloc(len); 
if (!string)
goto out;
for(i=0; i<nbnodes; i++) {
unsigned osnode = indexes[i];
char distancepath[SYSFS_NUMA_NODE_PATH_LEN];
char *tmp, *next;
unsigned found;
sprintf(distancepath, "%s/node%u/distance", path, osnode);
if (hwloc_read_path_by_length(distancepath, string, len, fsroot_fd) < 0)
goto out_with_string;
tmp = string;
found = 0;
while (tmp) {
unsigned distance = strtoul(tmp, &next, 0); 
if (next == tmp)
break;
*curdist = (uint64_t) distance;
curdist++;
found++;
if (found == nbnodes)
break;
tmp = next+1;
}
if (found != nbnodes)
goto out_with_string;
}
free(string);
return 0;
out_with_string:
free(string);
out:
return -1;
}
static void
hwloc__get_dmi_id_one_info(struct hwloc_linux_backend_data_s *data,
hwloc_obj_t obj,
char *path, unsigned pathlen,
const char *dmi_name, const char *hwloc_name)
{
char dmi_line[64];
strcpy(path+pathlen, dmi_name);
if (hwloc_read_path_by_length(path, dmi_line, sizeof(dmi_line), data->root_fd) < 0)
return;
if (dmi_line[0] != '\0') {
char *tmp = strchr(dmi_line, '\n');
if (tmp)
*tmp = '\0';
hwloc_debug("found %s '%s'\n", hwloc_name, dmi_line);
hwloc_obj_add_info(obj, hwloc_name, dmi_line);
}
}
static void
hwloc__get_dmi_id_info(struct hwloc_linux_backend_data_s *data, hwloc_obj_t obj)
{
char path[128];
unsigned pathlen;
DIR *dir;
strcpy(path, "/sys/devices/virtual/dmi/id");
dir = hwloc_opendir(path, data->root_fd);
if (dir) {
pathlen = 27;
} else {
strcpy(path, "/sys/class/dmi/id");
dir = hwloc_opendir(path, data->root_fd);
if (dir)
pathlen = 17;
else
return;
}
closedir(dir);
path[pathlen++] = '/';
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "product_name", "DMIProductName");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "product_version", "DMIProductVersion");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "product_serial", "DMIProductSerial");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "product_uuid", "DMIProductUUID");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_vendor", "DMIBoardVendor");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_name", "DMIBoardName");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_version", "DMIBoardVersion");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_serial", "DMIBoardSerial");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_asset_tag", "DMIBoardAssetTag");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_vendor", "DMIChassisVendor");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_type", "DMIChassisType");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_version", "DMIChassisVersion");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_serial", "DMIChassisSerial");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_asset_tag", "DMIChassisAssetTag");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "bios_vendor", "DMIBIOSVendor");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "bios_version", "DMIBIOSVersion");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "bios_date", "DMIBIOSDate");
hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "sys_vendor", "DMISysVendor");
}
static void *
hwloc_read_raw(const char *p, const char *p1, size_t *bytes_read, int root_fd)
{
char fname[256];
char *ret = NULL;
struct stat fs;
int file = -1;
snprintf(fname, sizeof(fname), "%s/%s", p, p1);
file = hwloc_open(fname, root_fd);
if (-1 == file) {
goto out_no_close;
}
if (fstat(file, &fs)) {
goto out;
}
ret = (char *) malloc(fs.st_size);
if (NULL != ret) {
ssize_t cb = read(file, ret, fs.st_size);
if (cb == -1) {
free(ret);
ret = NULL;
} else {
if (NULL != bytes_read)
*bytes_read = cb;
}
}
out:
close(file);
out_no_close:
return ret;
}
static char *
hwloc_read_str(const char *p, const char *p1, int root_fd)
{
size_t cb = 0;
char *ret = hwloc_read_raw(p, p1, &cb, root_fd);
if ((NULL != ret) && (0 < cb) && (0 != ret[cb-1])) {
char *tmp = realloc(ret, cb + 1);
if (!tmp) {
free(ret);
return NULL;
}
ret = tmp;
ret[cb] = 0;
}
return ret;
}
static ssize_t
hwloc_read_unit32be(const char *p, const char *p1, uint32_t *buf, int root_fd)
{
size_t cb = 0;
uint32_t *tmp = hwloc_read_raw(p, p1, &cb, root_fd);
if (sizeof(*buf) != cb) {
errno = EINVAL;
free(tmp); 
return -1;
}
*buf = htonl(*tmp);
free(tmp);
return sizeof(*buf);
}
typedef struct {
unsigned int n, allocated;
struct {
hwloc_bitmap_t cpuset;
uint32_t phandle;
uint32_t l2_cache;
char *name;
} *p;
} device_tree_cpus_t;
static void
add_device_tree_cpus_node(device_tree_cpus_t *cpus, hwloc_bitmap_t cpuset,
uint32_t l2_cache, uint32_t phandle, const char *name)
{
if (cpus->n == cpus->allocated) {
void *tmp;
unsigned allocated;
if (!cpus->allocated)
allocated = 64;
else
allocated = 2 * cpus->allocated;
tmp = realloc(cpus->p, allocated * sizeof(cpus->p[0]));
if (!tmp)
return; 
cpus->p = tmp;
cpus->allocated = allocated;
}
cpus->p[cpus->n].phandle = phandle;
cpus->p[cpus->n].cpuset = (NULL == cpuset)?NULL:hwloc_bitmap_dup(cpuset);
cpus->p[cpus->n].l2_cache = l2_cache;
cpus->p[cpus->n].name = strdup(name);
++cpus->n;
}
static int
look_powerpc_device_tree_discover_cache(device_tree_cpus_t *cpus,
uint32_t phandle, unsigned int *level, hwloc_bitmap_t cpuset)
{
unsigned int i;
int ret = -1;
if ((NULL == level) || (NULL == cpuset) || phandle == (uint32_t) -1)
return ret;
for (i = 0; i < cpus->n; ++i) {
if (phandle != cpus->p[i].l2_cache)
continue;
if (NULL != cpus->p[i].cpuset) {
hwloc_bitmap_or(cpuset, cpuset, cpus->p[i].cpuset);
ret = 0;
} else {
++(*level);
if (0 == look_powerpc_device_tree_discover_cache(cpus,
cpus->p[i].phandle, level, cpuset))
ret = 0;
}
}
return ret;
}
static void
try__add_cache_from_device_tree_cpu(struct hwloc_topology *topology,
unsigned int level, hwloc_obj_cache_type_t ctype,
uint32_t cache_line_size, uint32_t cache_size, uint32_t cache_sets,
hwloc_bitmap_t cpuset)
{
struct hwloc_obj *c = NULL;
hwloc_obj_type_t otype;
if (0 == cache_size)
return;
otype = hwloc_cache_type_by_depth_type(level, ctype);
if (otype == HWLOC_OBJ_TYPE_NONE)
return;
if (!hwloc_filter_check_keep_object_type(topology, otype))
return;
c = hwloc_alloc_setup_object(topology, otype, HWLOC_UNKNOWN_INDEX);
c->attr->cache.depth = level;
c->attr->cache.linesize = cache_line_size;
c->attr->cache.size = cache_size;
c->attr->cache.type = ctype;
if (cache_sets == 1)
cache_sets = 0;
if (cache_sets && cache_line_size)
c->attr->cache.associativity = cache_size / (cache_sets * cache_line_size);
else
c->attr->cache.associativity = 0;
c->cpuset = hwloc_bitmap_dup(cpuset);
hwloc_debug_2args_bitmap("cache (%s) depth %u has cpuset %s\n",
ctype == HWLOC_OBJ_CACHE_UNIFIED ? "unified" : (ctype == HWLOC_OBJ_CACHE_DATA ? "data" : "instruction"),
level, c->cpuset);
hwloc_insert_object_by_cpuset(topology, c);
}
static void
try_add_cache_from_device_tree_cpu(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data,
const char *cpu, unsigned int level, hwloc_bitmap_t cpuset)
{
uint32_t d_cache_line_size = 0, d_cache_size = 0, d_cache_sets = 0;
uint32_t i_cache_line_size = 0, i_cache_size = 0, i_cache_sets = 0;
char unified_path[1024];
struct stat statbuf;
int unified;
snprintf(unified_path, sizeof(unified_path), "%s/cache-unified", cpu);
unified = (hwloc_stat(unified_path, &statbuf, data->root_fd) == 0);
hwloc_read_unit32be(cpu, "d-cache-line-size", &d_cache_line_size,
data->root_fd);
hwloc_read_unit32be(cpu, "d-cache-size", &d_cache_size,
data->root_fd);
hwloc_read_unit32be(cpu, "d-cache-sets", &d_cache_sets,
data->root_fd);
hwloc_read_unit32be(cpu, "i-cache-line-size", &i_cache_line_size,
data->root_fd);
hwloc_read_unit32be(cpu, "i-cache-size", &i_cache_size,
data->root_fd);
hwloc_read_unit32be(cpu, "i-cache-sets", &i_cache_sets,
data->root_fd);
if (!unified)
try__add_cache_from_device_tree_cpu(topology, level, HWLOC_OBJ_CACHE_INSTRUCTION,
i_cache_line_size, i_cache_size, i_cache_sets, cpuset);
try__add_cache_from_device_tree_cpu(topology, level, unified ? HWLOC_OBJ_CACHE_UNIFIED : HWLOC_OBJ_CACHE_DATA,
d_cache_line_size, d_cache_size, d_cache_sets, cpuset);
}
static void
look_powerpc_device_tree(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data)
{
device_tree_cpus_t cpus;
const char ofroot[] = "/proc/device-tree/cpus";
unsigned int i;
int root_fd = data->root_fd;
DIR *dt = hwloc_opendir(ofroot, root_fd);
struct dirent *dirent;
if (NULL == dt)
return;
if (data->arch != HWLOC_LINUX_ARCH_POWER) {
closedir(dt);
return;
}
cpus.n = 0;
cpus.p = NULL;
cpus.allocated = 0;
while (NULL != (dirent = readdir(dt))) {
char cpu[256];
char *device_type;
uint32_t reg = -1, l2_cache = -1, phandle = -1;
int err;
if ('.' == dirent->d_name[0])
continue;
err = snprintf(cpu, sizeof(cpu), "%s/%s", ofroot, dirent->d_name);
if ((size_t) err >= sizeof(cpu))
continue;
device_type = hwloc_read_str(cpu, "device_type", root_fd);
if (NULL == device_type)
continue;
hwloc_read_unit32be(cpu, "reg", &reg, root_fd);
if (hwloc_read_unit32be(cpu, "next-level-cache", &l2_cache, root_fd) == -1)
hwloc_read_unit32be(cpu, "l2-cache", &l2_cache, root_fd);
if (hwloc_read_unit32be(cpu, "phandle", &phandle, root_fd) == -1)
if (hwloc_read_unit32be(cpu, "ibm,phandle", &phandle, root_fd) == -1)
hwloc_read_unit32be(cpu, "linux,phandle", &phandle, root_fd);
if (0 == strcmp(device_type, "cache")) {
add_device_tree_cpus_node(&cpus, NULL, l2_cache, phandle, dirent->d_name);
}
else if (0 == strcmp(device_type, "cpu")) {
hwloc_bitmap_t cpuset = NULL;
size_t cb = 0;
uint32_t *threads = hwloc_read_raw(cpu, "ibm,ppc-interrupt-server#s", &cb, root_fd);
uint32_t nthreads = cb / sizeof(threads[0]);
if (NULL != threads) {
cpuset = hwloc_bitmap_alloc();
for (i = 0; i < nthreads; ++i) {
if (hwloc_bitmap_isset(topology->levels[0][0]->complete_cpuset, ntohl(threads[i])))
hwloc_bitmap_set(cpuset, ntohl(threads[i]));
}
free(threads);
} else if ((unsigned int)-1 != reg) {
cpuset = hwloc_bitmap_alloc();
hwloc_bitmap_set(cpuset, reg);
}
if (NULL == cpuset) {
hwloc_debug("%s has no \"reg\" property, skipping\n", cpu);
} else {
struct hwloc_obj *core = NULL;
add_device_tree_cpus_node(&cpus, cpuset, l2_cache, phandle, dirent->d_name);
if (hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_CORE)) {
core = hwloc_alloc_setup_object(topology, HWLOC_OBJ_CORE, (unsigned) reg);
core->cpuset = hwloc_bitmap_dup(cpuset);
hwloc_insert_object_by_cpuset(topology, core);
}
try_add_cache_from_device_tree_cpu(topology, data, cpu, 1, cpuset);
hwloc_bitmap_free(cpuset);
}
}
free(device_type);
}
closedir(dt);
if (0 == cpus.n) {
hwloc_debug("No cores and L2 cache were found in %s, exiting\n", ofroot);
return;
}
#ifdef HWLOC_DEBUG
for (i = 0; i < cpus.n; ++i) {
hwloc_debug("%u: %s  ibm,phandle=%08X l2_cache=%08X ",
i, cpus.p[i].name, cpus.p[i].phandle, cpus.p[i].l2_cache);
if (NULL == cpus.p[i].cpuset) {
hwloc_debug("%s\n", "no cpuset");
} else {
hwloc_debug_bitmap("cpuset %s\n", cpus.p[i].cpuset);
}
}
#endif
for (i = 0; i < cpus.n; ++i) {
unsigned int level = 2;
hwloc_bitmap_t cpuset;
if (NULL != cpus.p[i].cpuset)
continue;
cpuset = hwloc_bitmap_alloc();
if (0 == look_powerpc_device_tree_discover_cache(&cpus,
cpus.p[i].phandle, &level, cpuset)) {
char cpu[256];
snprintf(cpu, sizeof(cpu), "%s/%s", ofroot, cpus.p[i].name);
try_add_cache_from_device_tree_cpu(topology, data, cpu, level, cpuset);
}
hwloc_bitmap_free(cpuset);
}
for (i = 0; i < cpus.n; ++i) {
hwloc_bitmap_free(cpus.p[i].cpuset);
free(cpus.p[i].name);
}
free(cpus.p);
}
struct knl_hwdata {
char memory_mode[32];
char cluster_mode[32];
long long int mcdram_cache_size; 
int mcdram_cache_associativity;
int mcdram_cache_inclusiveness;
int mcdram_cache_line_size;
};
struct knl_distances_summary {
unsigned nb_values; 
struct knl_distances_value {
unsigned occurences;
uint64_t value;
} values[4]; 
};
static int hwloc_knl_distances_value_compar(const void *_v1, const void *_v2)
{
const struct knl_distances_value *v1 = _v1, *v2 = _v2;
return v1->occurences - v2->occurences;
}
static int
hwloc_linux_knl_parse_numa_distances(unsigned nbnodes,
uint64_t *distances,
struct knl_distances_summary *summary)
{
unsigned i, j, k;
summary->nb_values = 1;
summary->values[0].value = 10;
summary->values[0].occurences = nbnodes;
if (nbnodes == 1)
return 0;
if (nbnodes != 2 && nbnodes != 4 && nbnodes != 8) {
fprintf(stderr, "Ignoring KNL NUMA quirk, nbnodes (%u) isn't 2, 4 or 8.\n", nbnodes);
return -1;
}
if (!distances) {
fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix missing.\n");
return -1;
}
for(i=0; i<nbnodes; i++) {
if (distances[i*nbnodes+i] != 10) {
fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix does not contain 10 on the diagonal.\n");
return -1;
}
for(j=i+1; j<nbnodes; j++) {
uint64_t distance = distances[i*nbnodes+j];
if (distance != distances[i+j*nbnodes]) {
fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix isn't symmetric.\n");
return -1;
}
if (distance <= 10) {
fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix contains values <= 10.\n");
return -1;
}
for(k=0; k<summary->nb_values; k++)
if (distance == summary->values[k].value) {
summary->values[k].occurences++;
break;
}
if (k == summary->nb_values) {
if (k == 4) {
fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix contains more than 4 different values.\n");
return -1;
}
summary->values[k].value = distance;
summary->values[k].occurences = 1;
summary->nb_values++;
}
}
}
qsort(summary->values, summary->nb_values, sizeof(struct knl_distances_value), hwloc_knl_distances_value_compar);
if (nbnodes == 2) {
if (summary->nb_values != 2) {
fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix for 2 nodes cannot contain %u different values instead of 2.\n",
summary->nb_values);
return -1;
}
} else if (nbnodes == 4) {
if (summary->nb_values != 2 && summary->nb_values != 4) {
fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix for 8 nodes cannot contain %u different values instead of 2 or 4.\n",
summary->nb_values);
return -1;
}
} else if (nbnodes == 8) {
if (summary->nb_values != 4) {
fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix for 8 nodes cannot contain %u different values instead of 4.\n",
summary->nb_values);
return -1;
}
} else {
abort(); 
}
hwloc_debug("Summary of KNL distance matrix:\n");
for(k=0; k<summary->nb_values; k++)
hwloc_debug("  Found %u times distance %llu\n", summary->values[k].occurences, (unsigned long long) summary->values[k].value);
return 0;
}
static int
hwloc_linux_knl_identify_4nodes(uint64_t *distances,
struct knl_distances_summary *distsum,
unsigned *ddr, unsigned *mcdram) 
{
uint64_t value;
unsigned i;
hwloc_debug("Trying to identify 4 KNL NUMA nodes in SNC-2 cluster mode...\n");
if (distsum->nb_values != 4
|| distsum->values[0].occurences != 1 
|| distsum->values[1].occurences != 2 
|| distsum->values[2].occurences != 3 
|| distsum->values[3].occurences != 4  )
return -1;
ddr[0] = 0;
value = distsum->values[0].value;
ddr[1] = 0;
hwloc_debug("  DDR#0 is NUMAnode#0\n");
for(i=0; i<4; i++)
if (distances[i] == value) {
ddr[1] = i;
hwloc_debug("  DDR#1 is NUMAnode#%u\n", i);
break;
}
if (!ddr[1])
return -1;
value = distsum->values[1].value;
mcdram[0] = mcdram[1] = 0;
for(i=1; i<4; i++) {
if (distances[i] == value) {
hwloc_debug("  MCDRAM#0 is NUMAnode#%u\n", i);
mcdram[0] = i;
} else if (distances[ddr[1]*4+i] == value) {
hwloc_debug("  MCDRAM#1 is NUMAnode#%u\n", i);
mcdram[1] = i;
}
}
if (!mcdram[0] || !mcdram[1])
return -1;
return 0;
}
static int
hwloc_linux_knl_identify_8nodes(uint64_t *distances,
struct knl_distances_summary *distsum,
unsigned *ddr, unsigned *mcdram) 
{
uint64_t value;
unsigned i, nb;
hwloc_debug("Trying to identify 8 KNL NUMA nodes in SNC-4 cluster mode...\n");
if (distsum->nb_values != 4
|| distsum->values[0].occurences != 4 
|| distsum->values[1].occurences != 6 
|| distsum->values[2].occurences != 8 
|| distsum->values[3].occurences != 18  )
return -1;
ddr[0] = 0;
hwloc_debug("  DDR#0 is NUMAnode#0\n");
value = distsum->values[1].value;
ddr[1] = ddr[2] = ddr[3] = 0;
nb = 1;
for(i=0; i<8; i++)
if (distances[i] == value) {
hwloc_debug("  DDR#%u is NUMAnode#%u\n", nb, i);
ddr[nb++] = i;
if (nb == 4)
break;
}
if (nb != 4 || !ddr[1] || !ddr[2] || !ddr[3])
return -1;
value = distsum->values[0].value;
mcdram[0] = mcdram[1] = mcdram[2] = mcdram[3] = 0;
for(i=1; i<8; i++) {
if (distances[i] == value) {
hwloc_debug("  MCDRAM#0 is NUMAnode#%u\n", i);
mcdram[0] = i;
} else if (distances[ddr[1]*8+i] == value) {
hwloc_debug("  MCDRAM#1 is NUMAnode#%u\n", i);
mcdram[1] = i;
} else if (distances[ddr[2]*8+i] == value) {
hwloc_debug("  MCDRAM#2 is NUMAnode#%u\n", i);
mcdram[2] = i;
} else if (distances[ddr[3]*8+i] == value) {
hwloc_debug("  MCDRAM#3 is NUMAnode#%u\n", i);
mcdram[3] = i;
}
}
if (!mcdram[0] || !mcdram[1] || !mcdram[2] || !mcdram[3])
return -1;
return 0;
}
static int
hwloc_linux_knl_read_hwdata_properties(struct hwloc_linux_backend_data_s *data,
struct knl_hwdata *hwdata)
{
char *knl_cache_file;
int version = 0;
char buffer[512] = {0};
char *data_beg = NULL;
if (asprintf(&knl_cache_file, "%s/knl_memoryside_cache", data->dumped_hwdata_dirname) < 0)
return -1;
hwloc_debug("Reading knl cache data from: %s\n", knl_cache_file);
if (hwloc_read_path_by_length(knl_cache_file, buffer, sizeof(buffer), data->root_fd) < 0) {
hwloc_debug("Unable to open KNL data file `%s' (%s)\n", knl_cache_file, strerror(errno));
free(knl_cache_file);
return -1;
}
free(knl_cache_file);
data_beg = &buffer[0];
if (sscanf(data_beg, "version: %d", &version) != 1) {
fprintf(stderr, "Invalid knl_memoryside_cache header, expected \"version: <int>\".\n");
return -1;
}
while (1) {
char *line_end = strstr(data_beg, "\n");
if (!line_end)
break;
if (version >= 1) {
if (!strncmp("cache_size:", data_beg, strlen("cache_size"))) {
sscanf(data_beg, "cache_size: %lld", &hwdata->mcdram_cache_size);
hwloc_debug("read cache_size=%lld\n", hwdata->mcdram_cache_size);
} else if (!strncmp("line_size:", data_beg, strlen("line_size:"))) {
sscanf(data_beg, "line_size: %d", &hwdata->mcdram_cache_line_size);
hwloc_debug("read line_size=%d\n", hwdata->mcdram_cache_line_size);
} else if (!strncmp("inclusiveness:", data_beg, strlen("inclusiveness:"))) {
sscanf(data_beg, "inclusiveness: %d", &hwdata->mcdram_cache_inclusiveness);
hwloc_debug("read inclusiveness=%d\n", hwdata->mcdram_cache_inclusiveness);
} else if (!strncmp("associativity:", data_beg, strlen("associativity:"))) {
sscanf(data_beg, "associativity: %d\n", &hwdata->mcdram_cache_associativity);
hwloc_debug("read associativity=%d\n", hwdata->mcdram_cache_associativity);
}
}
if (version >= 2) {
if (!strncmp("cluster_mode: ", data_beg, strlen("cluster_mode: "))) {
size_t length;
data_beg += strlen("cluster_mode: ");
length = line_end-data_beg;
if (length > sizeof(hwdata->cluster_mode)-1)
length = sizeof(hwdata->cluster_mode)-1;
memcpy(hwdata->cluster_mode, data_beg, length);
hwdata->cluster_mode[length] = '\0';
hwloc_debug("read cluster_mode=%s\n", hwdata->cluster_mode);
} else if (!strncmp("memory_mode: ", data_beg, strlen("memory_mode: "))) {
size_t length;
data_beg += strlen("memory_mode: ");
length = line_end-data_beg;
if (length > sizeof(hwdata->memory_mode)-1)
length = sizeof(hwdata->memory_mode)-1;
memcpy(hwdata->memory_mode, data_beg, length);
hwdata->memory_mode[length] = '\0';
hwloc_debug("read memory_mode=%s\n", hwdata->memory_mode);
}
}
data_beg = line_end + 1;
}
if (hwdata->mcdram_cache_size == -1
|| hwdata->mcdram_cache_line_size == -1
|| hwdata->mcdram_cache_associativity == -1
|| hwdata->mcdram_cache_inclusiveness == -1) {
hwloc_debug("Incorrect file format cache_size=%lld line_size=%d associativity=%d inclusiveness=%d\n",
hwdata->mcdram_cache_size,
hwdata->mcdram_cache_line_size,
hwdata->mcdram_cache_associativity,
hwdata->mcdram_cache_inclusiveness);
hwdata->mcdram_cache_size = -1; 
}
return 0;
}
static void
hwloc_linux_knl_guess_hwdata_properties(struct knl_hwdata *hwdata,
hwloc_obj_t *nodes, unsigned nbnodes,
struct knl_distances_summary *distsum)
{
hwloc_debug("Trying to guess missing KNL configuration information...\n");
hwdata->mcdram_cache_associativity = 1;
hwdata->mcdram_cache_inclusiveness = 1;
hwdata->mcdram_cache_line_size = 64;
if (hwdata->mcdram_cache_size > 0
&& hwdata->cluster_mode[0]
&& hwdata->memory_mode[0])
return;
if (nbnodes == 1) {
if (!hwdata->cluster_mode[0])
strcpy(hwdata->cluster_mode, "Quadrant");
if (!hwdata->memory_mode[0])
strcpy(hwdata->memory_mode, "Cache");
if (hwdata->mcdram_cache_size <= 0)
hwdata->mcdram_cache_size = 16UL*1024*1024*1024;
} else if (nbnodes == 2) {
if (!strcmp(hwdata->memory_mode, "Cache")
|| !strcmp(hwdata->cluster_mode, "SNC2")
|| !hwloc_bitmap_iszero(nodes[1]->cpuset)) { 
if (!hwdata->cluster_mode[0])
strcpy(hwdata->cluster_mode, "SNC2");
if (!hwdata->memory_mode[0])
strcpy(hwdata->memory_mode, "Cache");
if (hwdata->mcdram_cache_size <= 0)
hwdata->mcdram_cache_size = 8UL*1024*1024*1024;
} else {
if (!hwdata->cluster_mode[0])
strcpy(hwdata->cluster_mode, "Quadrant");
if (!hwdata->memory_mode[0]) {
if (hwdata->mcdram_cache_size == 4UL*1024*1024*1024)
strcpy(hwdata->memory_mode, "Hybrid25");
else if (hwdata->mcdram_cache_size == 8UL*1024*1024*1024)
strcpy(hwdata->memory_mode, "Hybrid50");
else
strcpy(hwdata->memory_mode, "Flat");
} else {
if (hwdata->mcdram_cache_size <= 0) {
if (!strcmp(hwdata->memory_mode, "Hybrid25"))
hwdata->mcdram_cache_size = 4UL*1024*1024*1024;
else if (!strcmp(hwdata->memory_mode, "Hybrid50"))
hwdata->mcdram_cache_size = 8UL*1024*1024*1024;
}
}
}
} else if (nbnodes == 4) {
if (!strcmp(hwdata->cluster_mode, "SNC2") || distsum->nb_values == 4) {
if (!hwdata->cluster_mode[0])
strcpy(hwdata->cluster_mode, "SNC2");
if (!hwdata->memory_mode[0]) {
if (hwdata->mcdram_cache_size == 2UL*1024*1024*1024)
strcpy(hwdata->memory_mode, "Hybrid25");
else if (hwdata->mcdram_cache_size == 4UL*1024*1024*1024)
strcpy(hwdata->memory_mode, "Hybrid50");
else
strcpy(hwdata->memory_mode, "Flat");
} else {
if (hwdata->mcdram_cache_size <= 0) {
if (!strcmp(hwdata->memory_mode, "Hybrid25"))
hwdata->mcdram_cache_size = 2UL*1024*1024*1024;
else if (!strcmp(hwdata->memory_mode, "Hybrid50"))
hwdata->mcdram_cache_size = 4UL*1024*1024*1024;
}
}
} else {
if (!hwdata->cluster_mode[0])
strcpy(hwdata->cluster_mode, "SNC4");
if (!hwdata->memory_mode[0])
strcpy(hwdata->memory_mode, "Cache");
if (hwdata->mcdram_cache_size <= 0)
hwdata->mcdram_cache_size = 4UL*1024*1024*1024;
}
} else if (nbnodes == 8) {
if (!hwdata->cluster_mode[0])
strcpy(hwdata->cluster_mode, "SNC4");
if (!hwdata->memory_mode[0]) {
if (hwdata->mcdram_cache_size == 1UL*1024*1024*1024)
strcpy(hwdata->memory_mode, "Hybrid25");
else if (hwdata->mcdram_cache_size == 2UL*1024*1024*1024)
strcpy(hwdata->memory_mode, "Hybrid50");
else
strcpy(hwdata->memory_mode, "Flat");
} else {
if (hwdata->mcdram_cache_size <= 0) {
if (!strcmp(hwdata->memory_mode, "Hybrid25"))
hwdata->mcdram_cache_size = 1UL*1024*1024*1024;
else if (!strcmp(hwdata->memory_mode, "Hybrid50"))
hwdata->mcdram_cache_size = 2UL*1024*1024*1024;
}
}
}
hwloc_debug("  Found cluster=%s memory=%s cache=%lld\n",
hwdata->cluster_mode, hwdata->memory_mode,
hwdata->mcdram_cache_size);
}
static void
hwloc_linux_knl_add_cluster(struct hwloc_topology *topology,
hwloc_obj_t ddr, hwloc_obj_t mcdram,
struct knl_hwdata *knl_hwdata,
int mscache_as_l3,
unsigned *failednodes)
{
hwloc_obj_t cluster = NULL;
if (mcdram) {
mcdram->subtype = strdup("MCDRAM");
hwloc_bitmap_copy(mcdram->cpuset, ddr->cpuset);
cluster = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);
hwloc_obj_add_other_obj_sets(cluster, ddr);
hwloc_obj_add_other_obj_sets(cluster, mcdram);
cluster->subtype = strdup("Cluster");
cluster->attr->group.kind = HWLOC_GROUP_KIND_INTEL_KNL_SUBNUMA_CLUSTER;
cluster = hwloc__insert_object_by_cpuset(topology, NULL, cluster, hwloc_report_os_error);
}
if (cluster) {
hwloc_obj_t res;
res = hwloc__attach_memory_object(topology, cluster, ddr, hwloc_report_os_error);
if (res != ddr) {
(*failednodes)++;
ddr = NULL;
}
res = hwloc__attach_memory_object(topology, cluster, mcdram, hwloc_report_os_error);
if (res != mcdram)
(*failednodes)++;
} else {
hwloc_obj_t res;
res = hwloc__insert_object_by_cpuset(topology, NULL, ddr, hwloc_report_os_error);
if (res != ddr) {
(*failednodes)++;
ddr = NULL;
}
if (mcdram) {
res = hwloc__insert_object_by_cpuset(topology, NULL, mcdram, hwloc_report_os_error);
if (res != mcdram)
(*failednodes)++;
}
}
if (ddr && knl_hwdata->mcdram_cache_size > 0) {
hwloc_obj_t cache = hwloc_alloc_setup_object(topology, HWLOC_OBJ_L3CACHE, HWLOC_UNKNOWN_INDEX);
if (!cache)
return;
cache->attr->cache.depth = 3;
cache->attr->cache.type = HWLOC_OBJ_CACHE_UNIFIED;
cache->attr->cache.size = knl_hwdata->mcdram_cache_size;
cache->attr->cache.linesize = knl_hwdata->mcdram_cache_line_size;
cache->attr->cache.associativity = knl_hwdata->mcdram_cache_associativity;
hwloc_obj_add_info(cache, "Inclusive", knl_hwdata->mcdram_cache_inclusiveness ? "1" : "0");
cache->cpuset = hwloc_bitmap_dup(ddr->cpuset);
cache->nodeset = hwloc_bitmap_dup(ddr->nodeset); 
if (mscache_as_l3) {
cache->subtype = strdup("MemorySideCache");
hwloc_insert_object_by_cpuset(topology, cache);
} else {
cache->type = HWLOC_OBJ_MEMCACHE;
if (cluster)
hwloc__attach_memory_object(topology, cluster, cache, hwloc_report_os_error);
else
hwloc__insert_object_by_cpuset(topology, NULL, cache, hwloc_report_os_error);
}
}
}
static void
hwloc_linux_knl_numa_quirk(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data,
hwloc_obj_t *nodes, unsigned nbnodes,
uint64_t * distances,
unsigned *failednodes)
{
struct knl_hwdata hwdata;
struct knl_distances_summary dist;
unsigned i;
char * fallback_env = getenv("HWLOC_KNL_HDH_FALLBACK");
int fallback = fallback_env ? atoi(fallback_env) : -1; 
char * mscache_as_l3_env = getenv("HWLOC_KNL_MSCACHE_L3");
int mscache_as_l3 = mscache_as_l3_env ? atoi(mscache_as_l3_env) : 1; 
if (*failednodes)
goto error;
if (hwloc_linux_knl_parse_numa_distances(nbnodes, distances, &dist) < 0)
goto error;
hwdata.memory_mode[0] = '\0';
hwdata.cluster_mode[0] = '\0';
hwdata.mcdram_cache_size = -1;
hwdata.mcdram_cache_associativity = -1;
hwdata.mcdram_cache_inclusiveness = -1;
hwdata.mcdram_cache_line_size = -1;
if (fallback == 1)
hwloc_debug("KNL dumped hwdata ignored, forcing fallback to heuristics\n");
else
hwloc_linux_knl_read_hwdata_properties(data, &hwdata);
if (fallback != 0)
hwloc_linux_knl_guess_hwdata_properties(&hwdata, nodes, nbnodes, &dist);
if (strcmp(hwdata.cluster_mode, "All2All")
&& strcmp(hwdata.cluster_mode, "Hemisphere")
&& strcmp(hwdata.cluster_mode, "Quadrant")
&& strcmp(hwdata.cluster_mode, "SNC2")
&& strcmp(hwdata.cluster_mode, "SNC4")) {
fprintf(stderr, "Failed to find a usable KNL cluster mode (%s)\n", hwdata.cluster_mode);
goto error;
}
if (strcmp(hwdata.memory_mode, "Cache")
&& strcmp(hwdata.memory_mode, "Flat")
&& strcmp(hwdata.memory_mode, "Hybrid25")
&& strcmp(hwdata.memory_mode, "Hybrid50")) {
fprintf(stderr, "Failed to find a usable KNL memory mode (%s)\n", hwdata.memory_mode);
goto error;
}
if (mscache_as_l3) {
if (!hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_L3CACHE))
hwdata.mcdram_cache_size = 0;
} else {
if (!hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_MEMCACHE))
hwdata.mcdram_cache_size = 0;
}
hwloc_obj_add_info(topology->levels[0][0], "ClusterMode", hwdata.cluster_mode);
hwloc_obj_add_info(topology->levels[0][0], "MemoryMode", hwdata.memory_mode);
if (!strcmp(hwdata.cluster_mode, "All2All")
|| !strcmp(hwdata.cluster_mode, "Hemisphere")
|| !strcmp(hwdata.cluster_mode, "Quadrant")) {
if (!strcmp(hwdata.memory_mode, "Cache")) {
if (nbnodes != 1) {
fprintf(stderr, "Found %u NUMA nodes instead of 1 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
goto error;
}
hwloc_linux_knl_add_cluster(topology, nodes[0], NULL, &hwdata, mscache_as_l3, failednodes);
} else {
if (nbnodes != 2) {
fprintf(stderr, "Found %u NUMA nodes instead of 2 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
goto error;
}
if (!strcmp(hwdata.memory_mode, "Flat"))
hwdata.mcdram_cache_size = 0;
hwloc_linux_knl_add_cluster(topology, nodes[0], nodes[1], &hwdata, mscache_as_l3, failednodes);
}
} else if (!strcmp(hwdata.cluster_mode, "SNC2")) {
if (!strcmp(hwdata.memory_mode, "Cache")) {
if (nbnodes != 2) {
fprintf(stderr, "Found %u NUMA nodes instead of 2 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
goto error;
}
hwloc_linux_knl_add_cluster(topology, nodes[0], NULL, &hwdata, mscache_as_l3, failednodes);
hwloc_linux_knl_add_cluster(topology, nodes[1], NULL, &hwdata, mscache_as_l3, failednodes);
} else {
unsigned ddr[2], mcdram[2];
if (nbnodes != 4) {
fprintf(stderr, "Found %u NUMA nodes instead of 2 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
goto error;
}
if (hwloc_linux_knl_identify_4nodes(distances, &dist, ddr, mcdram) < 0) {
fprintf(stderr, "Unexpected distance layout for mode %s-%s\n", hwdata.cluster_mode, hwdata.memory_mode);
goto error;
}
if (!strcmp(hwdata.memory_mode, "Flat"))
hwdata.mcdram_cache_size = 0;
hwloc_linux_knl_add_cluster(topology, nodes[ddr[0]], nodes[mcdram[0]], &hwdata, mscache_as_l3, failednodes);
hwloc_linux_knl_add_cluster(topology, nodes[ddr[1]], nodes[mcdram[1]], &hwdata, mscache_as_l3, failednodes);
}
} else if (!strcmp(hwdata.cluster_mode, "SNC4")) {
if (!strcmp(hwdata.memory_mode, "Cache")) {
if (nbnodes != 4) {
fprintf(stderr, "Found %u NUMA nodes instead of 4 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
goto error;
}
hwloc_linux_knl_add_cluster(topology, nodes[0], NULL, &hwdata, mscache_as_l3, failednodes);
hwloc_linux_knl_add_cluster(topology, nodes[1], NULL, &hwdata, mscache_as_l3, failednodes);
hwloc_linux_knl_add_cluster(topology, nodes[2], NULL, &hwdata, mscache_as_l3, failednodes);
hwloc_linux_knl_add_cluster(topology, nodes[3], NULL, &hwdata, mscache_as_l3, failednodes);
} else {
unsigned ddr[4], mcdram[4];
if (nbnodes != 8) {
fprintf(stderr, "Found %u NUMA nodes instead of 2 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
goto error;
}
if (hwloc_linux_knl_identify_8nodes(distances, &dist, ddr, mcdram) < 0) {
fprintf(stderr, "Unexpected distance layout for mode %s-%s\n", hwdata.cluster_mode, hwdata.memory_mode);
goto error;
}
if (!strcmp(hwdata.memory_mode, "Flat"))
hwdata.mcdram_cache_size = 0;
hwloc_linux_knl_add_cluster(topology, nodes[ddr[0]], nodes[mcdram[0]], &hwdata, mscache_as_l3, failednodes);
hwloc_linux_knl_add_cluster(topology, nodes[ddr[1]], nodes[mcdram[1]], &hwdata, mscache_as_l3, failednodes);
hwloc_linux_knl_add_cluster(topology, nodes[ddr[2]], nodes[mcdram[2]], &hwdata, mscache_as_l3, failednodes);
hwloc_linux_knl_add_cluster(topology, nodes[ddr[3]], nodes[mcdram[3]], &hwdata, mscache_as_l3, failednodes);
}
}
return;
error:
for (i = 0; i < nbnodes; i++) {
hwloc_obj_t node = nodes[i];
if (node) {
hwloc_obj_t res_obj = hwloc__insert_object_by_cpuset(topology, NULL, node, hwloc_report_os_error);
if (res_obj != node)
(*failednodes)++;
}
}
}
static int
fixup_cpuless_node_locality_from_distances(unsigned i,
unsigned nbnodes, hwloc_obj_t *nodes, uint64_t *distances)
{
unsigned min = UINT_MAX;
unsigned nb = 0, j;
for(j=0; j<nbnodes; j++) {
if (j==i || !nodes[j])
continue;
if (distances[i*nbnodes+j] < min) {
min = distances[i*nbnodes+j];
nb = 1;
} else if (distances[i*nbnodes+j] == min) {
nb++;
}
}
if (min <= distances[i*nbnodes+i] || min == UINT_MAX || nb == nbnodes-1)
return -1;
for(j=0; j<nbnodes; j++)
if (j!=i && nodes[j] && distances[i*nbnodes+j] == min)
hwloc_bitmap_or(nodes[i]->cpuset, nodes[i]->cpuset, nodes[j]->cpuset);
return 0;
}
static int
read_node_initiators(struct hwloc_linux_backend_data_s *data,
hwloc_obj_t node, unsigned nbnodes, hwloc_obj_t *nodes,
const char *path)
{
char accesspath[SYSFS_NUMA_NODE_PATH_LEN];
DIR *dir;
struct dirent *dirent;
sprintf(accesspath, "%s/node%u/access0/initiators", path, node->os_index);
dir = hwloc_opendir(accesspath, data->root_fd);
if (!dir)
return -1;
while ((dirent = readdir(dir)) != NULL) {
unsigned initiator_os_index;
if (sscanf(dirent->d_name, "node%u", &initiator_os_index) == 1
&& initiator_os_index != node->os_index) {
unsigned j;
for(j=0; j<nbnodes; j++)
if (nodes[j] && nodes[j]->os_index == initiator_os_index) {
hwloc_bitmap_or(node->cpuset, node->cpuset, nodes[j]->cpuset);
break;
}
}
}
closedir(dir);
return 0;
}
static int
read_node_mscaches(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data,
const char *path,
hwloc_obj_t *treep)
{
hwloc_obj_t tree = *treep, node = tree;
unsigned osnode = node->os_index;
char mscpath[SYSFS_NUMA_NODE_PATH_LEN];
DIR *mscdir;
struct dirent *dirent;
sprintf(mscpath, "%s/node%u/memory_side_cache", path, osnode);
mscdir = hwloc_opendir(mscpath, data->root_fd);
if (!mscdir)
return -1;
while ((dirent = readdir(mscdir)) != NULL) {
unsigned depth;
uint64_t size;
unsigned line_size;
unsigned associativity;
hwloc_obj_t cache;
if (strncmp(dirent->d_name, "index", 5))
continue;
depth = atoi(dirent->d_name+5);
sprintf(mscpath, "%s/node%u/memory_side_cache/index%u/size", path, osnode, depth);
if (hwloc_read_path_as_uint64(mscpath, &size, data->root_fd) < 0)
continue;
sprintf(mscpath, "%s/node%u/memory_side_cache/index%u/line_size", path, osnode, depth);
if (hwloc_read_path_as_uint(mscpath, &line_size, data->root_fd) < 0)
continue;
sprintf(mscpath, "%s/node%u/memory_side_cache/index%u/indexing", path, osnode, depth);
if (hwloc_read_path_as_uint(mscpath, &associativity, data->root_fd) < 0)
continue;
cache = hwloc_alloc_setup_object(topology, HWLOC_OBJ_MEMCACHE, HWLOC_UNKNOWN_INDEX);
if (cache) {
cache->nodeset = hwloc_bitmap_dup(node->nodeset);
cache->cpuset = hwloc_bitmap_dup(node->cpuset);
cache->attr->cache.size = size;
cache->attr->cache.depth = depth;
cache->attr->cache.linesize = line_size;
cache->attr->cache.type = HWLOC_OBJ_CACHE_UNIFIED;
cache->attr->cache.associativity = !associativity ? 1  : 0 ;
hwloc_debug_1arg_bitmap("mscache %s has nodeset %s\n",
dirent->d_name, cache->nodeset);
cache->memory_first_child = tree;
tree = cache;
}
}
closedir(mscdir);
*treep = tree;
return 0;
}
static unsigned *
list_sysfsnode(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data,
const char *path,
unsigned *nbnodesp)
{
DIR *dir;
unsigned osnode, nbnodes = 0;
unsigned *indexes, index_;
hwloc_bitmap_t nodeset;
struct dirent *dirent;
nodeset = hwloc__alloc_read_path_as_cpulist("/sys/devices/system/node/online", data->root_fd);
if (nodeset) {
int _nbnodes = hwloc_bitmap_weight(nodeset);
assert(_nbnodes >= 1);
nbnodes = (unsigned)_nbnodes;
hwloc_debug_bitmap("possible NUMA nodes %s\n", nodeset);
goto found;
}
dir = hwloc_opendir(path, data->root_fd);
if (!dir)
return NULL;
nodeset = hwloc_bitmap_alloc();
if (!nodeset) {
closedir(dir);
return NULL;
}
while ((dirent = readdir(dir)) != NULL) {
if (strncmp(dirent->d_name, "node", 4))
continue;
osnode = strtoul(dirent->d_name+4, NULL, 0);
hwloc_bitmap_set(nodeset, osnode);
nbnodes++;
}
closedir(dir);
assert(nbnodes >= 1); 
found:
if (!hwloc_bitmap_iszero(topology->levels[0][0]->nodeset)
&& !hwloc_bitmap_isequal(nodeset, topology->levels[0][0]->nodeset)) {
char *sn, *tn;
hwloc_bitmap_asprintf(&sn, nodeset);
hwloc_bitmap_asprintf(&tn, topology->levels[0][0]->nodeset);
fprintf(stderr, "linux/sysfs: ignoring nodes because nodeset %s doesn't match existing nodeset %s.\n", tn, sn);
free(sn);
free(tn);
hwloc_bitmap_free(nodeset);
return NULL;
}
indexes = calloc(nbnodes, sizeof(*indexes));
if (!indexes) {
hwloc_bitmap_free(nodeset);
return NULL;
}
index_ = 0;
hwloc_bitmap_foreach_begin (osnode, nodeset) {
indexes[index_] = osnode;
index_++;
} hwloc_bitmap_foreach_end();
hwloc_bitmap_free(nodeset);
#ifdef HWLOC_DEBUG
hwloc_debug("%s", "NUMA indexes: ");
for (index_ = 0; index_ < nbnodes; index_++)
hwloc_debug(" %u", indexes[index_]);
hwloc_debug("%s", "\n");
#endif
*nbnodesp = nbnodes;
return indexes;
}
static int
annotate_sysfsnode(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data,
const char *path, unsigned *found)
{
unsigned nbnodes;
hwloc_obj_t * nodes; 
hwloc_obj_t node;
unsigned * indexes;
uint64_t * distances;
unsigned i;
indexes = list_sysfsnode(topology, data, path, &nbnodes);
if (!indexes)
return 0;
nodes = calloc(nbnodes, sizeof(hwloc_obj_t));
distances = malloc(nbnodes*nbnodes*sizeof(*distances));
if (NULL == nodes || NULL == distances) {
free(nodes);
free(indexes);
free(distances);
return 0;
}
for(node=hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, NULL);
node != NULL;
node = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, node)) {
assert(node); 
for(i=0; i<nbnodes; i++)
if (indexes[i] == node->os_index) {
nodes[i] = node;
break;
}
hwloc_get_sysfs_node_meminfo(data, path, node->os_index, &node->attr->numanode);
}
topology->support.discovery->numa = 1;
topology->support.discovery->numa_memory = 1;
topology->support.discovery->disallowed_numa = 1;
if (nbnodes >= 2
&& data->use_numa_distances
&& !hwloc_parse_nodes_distances(path, nbnodes, indexes, distances, data->root_fd)) {
hwloc_internal_distances_add(topology, "NUMALatency", nbnodes, nodes, distances,
HWLOC_DISTANCES_KIND_FROM_OS|HWLOC_DISTANCES_KIND_MEANS_LATENCY,
HWLOC_DISTANCES_ADD_FLAG_GROUP);
} else {
free(nodes);
free(distances);
}
free(indexes);
*found = nbnodes;
return 0;
}
static int
look_sysfsnode(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data,
const char *path, unsigned *found)
{
unsigned osnode;
unsigned nbnodes;
hwloc_obj_t * nodes; 
unsigned nr_trees;
hwloc_obj_t * trees; 
unsigned *indexes;
uint64_t * distances;
hwloc_bitmap_t nodes_cpuset;
unsigned failednodes = 0;
unsigned i;
DIR *dir;
int allow_overlapping_node_cpusets = (getenv("HWLOC_DEBUG_ALLOW_OVERLAPPING_NODE_CPUSETS") != NULL);
int need_memcaches = hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_MEMCACHE);
indexes = list_sysfsnode(topology, data, path, &nbnodes);
if (!indexes)
return 0;
nodes = calloc(nbnodes, sizeof(hwloc_obj_t));
trees = calloc(nbnodes, sizeof(hwloc_obj_t));
distances = malloc(nbnodes*nbnodes*sizeof(*distances));
nodes_cpuset  = hwloc_bitmap_alloc();
if (NULL == nodes || NULL == trees || NULL == distances || NULL == nodes_cpuset) {
free(nodes);
free(trees);
free(indexes);
free(distances);
hwloc_bitmap_free(nodes_cpuset);
nbnodes = 0;
goto out;
}
topology->support.discovery->numa = 1;
topology->support.discovery->numa_memory = 1;
topology->support.discovery->disallowed_numa = 1;
for (i = 0; i < nbnodes; i++) {
hwloc_obj_t node;
char nodepath[SYSFS_NUMA_NODE_PATH_LEN];
hwloc_bitmap_t cpuset;
osnode = indexes[i];
sprintf(nodepath, "%s/node%u/cpumap", path, osnode);
cpuset = hwloc__alloc_read_path_as_cpumask(nodepath, data->root_fd);
if (!cpuset) {
failednodes++;
continue;
}
if (hwloc_bitmap_intersects(nodes_cpuset, cpuset)) {
if (!allow_overlapping_node_cpusets) {
hwloc_debug_1arg_bitmap("node P#%u cpuset %s intersects with previous nodes, ignoring that node.\n", osnode, cpuset);
hwloc_bitmap_free(cpuset);
failednodes++;
continue;
}
fprintf(stderr, "node P#%u cpuset intersects with previous nodes, forcing its acceptance\n", osnode);
}
hwloc_bitmap_or(nodes_cpuset, nodes_cpuset, cpuset);
node = hwloc_alloc_setup_object(topology, HWLOC_OBJ_NUMANODE, osnode);
node->cpuset = cpuset;
node->nodeset = hwloc_bitmap_alloc();
hwloc_bitmap_set(node->nodeset, osnode);
hwloc_get_sysfs_node_meminfo(data, path, osnode, &node->attr->numanode);
nodes[i] = node;
hwloc_debug_1arg_bitmap("os node %u has cpuset %s\n",
osnode, node->cpuset);
}
dir = hwloc_opendir("/proc/driver/nvidia/gpus", data->root_fd);
if (dir) {
struct dirent *dirent;
char *env = getenv("HWLOC_KEEP_NVIDIA_GPU_NUMA_NODES");
int keep = env && atoi(env);
while ((dirent = readdir(dir)) != NULL) {
char nvgpunumapath[300], line[256];
int fd;
snprintf(nvgpunumapath, sizeof(nvgpunumapath), "/proc/driver/nvidia/gpus/%s/numa_status", dirent->d_name);
fd = hwloc_open(nvgpunumapath, data->root_fd);
if (fd >= 0) {
int ret;
ret = read(fd, line, sizeof(line)-1);
line[sizeof(line)-1] = '\0';
if (ret >= 0) {
const char *nvgpu_node_line = strstr(line, "Node:");
if (nvgpu_node_line) {
unsigned nvgpu_node;
const char *value = nvgpu_node_line+5;
while (*value == ' ' || *value == '\t')
value++;
nvgpu_node = atoi(value);
hwloc_debug("os node %u is NVIDIA GPU %s integrated memory\n", nvgpu_node, dirent->d_name);
for(i=0; i<nbnodes; i++) {
hwloc_obj_t node = nodes[i];
if (node && node->os_index == nvgpu_node) {
if (keep) {
char nvgpulocalcpuspath[300];
int err;
node->subtype = strdup("GPUMemory");
hwloc_obj_add_info(node, "PCIBusID", dirent->d_name);
snprintf(nvgpulocalcpuspath, sizeof(nvgpulocalcpuspath), "/sys/bus/pci/devices/%s/local_cpus", dirent->d_name);
err = hwloc__read_path_as_cpumask(nvgpulocalcpuspath, node->cpuset, data->root_fd);
if (err)
hwloc_bitmap_zero(node->cpuset);
} else {
hwloc_free_unlinked_object(node);
nodes[i] = NULL;
}
break;
}
}
}
}
close(fd);
}
}
closedir(dir);
}
dir = hwloc_opendir("/sys/bus/dax/devices/", data->root_fd);
if (dir) {
struct dirent *dirent;
while ((dirent = readdir(dir)) != NULL) {
char daxpath[300];
int tmp;
osnode = (unsigned) -1;
snprintf(daxpath, sizeof(daxpath), "/sys/bus/dax/devices/%s/target_node", dirent->d_name);
if (!hwloc_read_path_as_int(daxpath, &tmp, data->root_fd)) { 
osnode = (unsigned) tmp;
for(i=0; i<nbnodes; i++) {
hwloc_obj_t node = nodes[i];
if (node && node->os_index == osnode)
hwloc_obj_add_info(node, "DAXDevice", dirent->d_name);
}
}
}
closedir(dir);
}
topology->support.discovery->numa = 1;
topology->support.discovery->numa_memory = 1;
topology->support.discovery->disallowed_numa = 1;
hwloc_bitmap_free(nodes_cpuset);
if (nbnodes <= 1) {
data->use_numa_distances = 0;
}
if (!data->use_numa_distances) {
free(distances);
distances = NULL;
}
if (distances && hwloc_parse_nodes_distances(path, nbnodes, indexes, distances, data->root_fd) < 0) {
free(distances);
distances = NULL;
}
free(indexes);
if (data->is_knl) {
char *env = getenv("HWLOC_KNL_NUMA_QUIRK");
int noquirk = (env && !atoi(env));
if (!noquirk) {
hwloc_linux_knl_numa_quirk(topology, data, nodes, nbnodes, distances, &failednodes);
free(distances);
free(nodes);
free(trees);
goto out;
}
}
nr_trees = 0;
for (i = 0; i < nbnodes; i++) {
hwloc_obj_t node = nodes[i];
if (node && !hwloc_bitmap_iszero(node->cpuset)) {
hwloc_obj_t tree;
if (data->use_numa_initiators)
read_node_initiators(data, node, nbnodes, nodes, path);
tree = node;
if (need_memcaches)
read_node_mscaches(topology, data, path, &tree);
trees[nr_trees++] = tree;
}
}
for (i = 0; i < nbnodes; i++) {
hwloc_obj_t node = nodes[i];
if (node && hwloc_bitmap_iszero(node->cpuset)) {
hwloc_obj_t tree;
if (data->use_numa_initiators)
if (!read_node_initiators(data, node, nbnodes, nodes, path))
if (!hwloc_bitmap_iszero(node->cpuset))
goto fixed;
if (distances && data->use_numa_distances_for_cpuless)
fixup_cpuless_node_locality_from_distances(i, nbnodes, nodes, distances);
fixed:
tree = node;
if (need_memcaches)
read_node_mscaches(topology, data, path, &tree);
trees[nr_trees++] = tree;
}
}
for (i = 0; i < nr_trees; i++) {
hwloc_obj_t tree = trees[i];
while (tree) {
hwloc_obj_t cur_obj;
hwloc_obj_t res_obj;
hwloc_obj_type_t cur_type;
cur_obj = tree;
cur_type = cur_obj->type;
tree = cur_obj->memory_first_child;
assert(!cur_obj->next_sibling);
res_obj = hwloc__insert_object_by_cpuset(topology, NULL, cur_obj, hwloc_report_os_error);
if (res_obj != cur_obj && cur_type == HWLOC_OBJ_NUMANODE) {
unsigned j;
for(j=0; j<nbnodes; j++)
if (nodes[j] == cur_obj)
nodes[j] = res_obj;
failednodes++;
}
}
}
free(trees);
if (distances)
hwloc_internal_distances_add(topology, "NUMALatency", nbnodes, nodes, distances,
HWLOC_DISTANCES_KIND_FROM_OS|HWLOC_DISTANCES_KIND_MEANS_LATENCY,
HWLOC_DISTANCES_ADD_FLAG_GROUP);
else
free(nodes);
out:
*found = nbnodes - failednodes;
return 0;
}
static int
look_sysfscpu(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data,
const char *path, int old_filenames,
struct hwloc_linux_cpuinfo_proc * cpuinfo_Lprocs, unsigned cpuinfo_numprocs)
{
hwloc_bitmap_t cpuset; 
hwloc_bitmap_t online_set; 
#define CPU_TOPOLOGY_STR_LEN 128
char str[CPU_TOPOLOGY_STR_LEN];
DIR *dir;
int i,j;
unsigned caches_added;
int threadwithcoreid = data->is_amd_with_CU ? -1 : 0; 
online_set = hwloc__alloc_read_path_as_cpulist("/sys/devices/system/cpu/online", data->root_fd);
if (online_set)
hwloc_debug_bitmap("online CPUs %s\n", online_set);
dir = hwloc_opendir(path, data->root_fd);
if (!dir) {
hwloc_bitmap_free(online_set);
return -1;
} else {
struct dirent *dirent;
cpuset = hwloc_bitmap_alloc();
while ((dirent = readdir(dir)) != NULL) {
unsigned long cpu;
char online[2];
if (strncmp(dirent->d_name, "cpu", 3))
continue;
cpu = strtoul(dirent->d_name+3, NULL, 0);
hwloc_bitmap_set(topology->levels[0][0]->complete_cpuset, cpu);
if (online_set) {
if (!hwloc_bitmap_isset(online_set, cpu)) {
hwloc_debug("os proc %lu is offline\n", cpu);
continue;
}
} else {
sprintf(str, "%s/cpu%lu/online", path, cpu);
if (hwloc_read_path_by_length(str, online, sizeof(online), data->root_fd) == 0) {
if (!atoi(online)) {
hwloc_debug("os proc %lu is offline\n", cpu);
continue;
}
}
}
sprintf(str, "%s/cpu%lu/topology", path, cpu);
if (hwloc_access(str, X_OK, data->root_fd) < 0 && errno == ENOENT) {
hwloc_debug("os proc %lu has no accessible %s/cpu%lu/topology\n",
cpu, path, cpu);
continue;
}
hwloc_bitmap_set(cpuset, cpu);
}
closedir(dir);
}
topology->support.discovery->pu = 1;
topology->support.discovery->disallowed_pu = 1;
hwloc_debug_1arg_bitmap("found %d cpu topologies, cpuset %s\n",
hwloc_bitmap_weight(cpuset), cpuset);
caches_added = 0;
hwloc_bitmap_foreach_begin(i, cpuset) {
int tmpint;
int notfirstofcore = 0; 
int notfirstofdie = 0; 
hwloc_bitmap_t dieset = NULL;
if (hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_CORE)) {
hwloc_bitmap_t coreset;
if (old_filenames)
sprintf(str, "%s/cpu%d/topology/thread_siblings", path, i);
else
sprintf(str, "%s/cpu%d/topology/core_cpus", path, i);
coreset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
if (coreset) {
unsigned mycoreid = (unsigned) -1;
int gotcoreid = 0; 
hwloc_bitmap_and(coreset, coreset, cpuset);
if (hwloc_bitmap_weight(coreset) > 1 && threadwithcoreid == -1) {
unsigned siblingid, siblingcoreid;
mycoreid = (unsigned) -1;
sprintf(str, "%s/cpu%d/topology/core_id", path, i); 
if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
mycoreid = (unsigned) tmpint;
gotcoreid = 1;
siblingid = hwloc_bitmap_first(coreset);
if (siblingid == (unsigned) i)
siblingid = hwloc_bitmap_next(coreset, i);
siblingcoreid = (unsigned) -1;
sprintf(str, "%s/cpu%u/topology/core_id", path, siblingid); 
if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
siblingcoreid = (unsigned) tmpint;
threadwithcoreid = (siblingcoreid != mycoreid);
}
if (hwloc_bitmap_first(coreset) != i)
notfirstofcore = 1;
if (!notfirstofcore || threadwithcoreid) {
struct hwloc_obj *core;
if (!gotcoreid) {
mycoreid = (unsigned) -1;
sprintf(str, "%s/cpu%d/topology/core_id", path, i); 
if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
mycoreid = (unsigned) tmpint;
}
core = hwloc_alloc_setup_object(topology, HWLOC_OBJ_CORE, mycoreid);
if (threadwithcoreid)
hwloc_bitmap_only(coreset, i);
core->cpuset = coreset;
hwloc_debug_1arg_bitmap("os core %u has cpuset %s\n",
mycoreid, core->cpuset);
hwloc_insert_object_by_cpuset(topology, core);
coreset = NULL; 
} else
hwloc_bitmap_free(coreset);
}
}
if (!notfirstofcore 
&& hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_DIE)) {
sprintf(str, "%s/cpu%d/topology/die_cpus", path, i);
dieset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
if (dieset) {
hwloc_bitmap_and(dieset, dieset, cpuset);
if (hwloc_bitmap_first(dieset) != i) {
hwloc_bitmap_free(dieset);
dieset = NULL;
notfirstofdie = 1;
}
}
}
if (!notfirstofdie 
&& hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_PACKAGE)) {
hwloc_bitmap_t packageset;
if (old_filenames)
sprintf(str, "%s/cpu%d/topology/core_siblings", path, i);
else
sprintf(str, "%s/cpu%d/topology/package_cpus", path, i);
packageset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
if (packageset) {
hwloc_bitmap_and(packageset, packageset, cpuset);
if (dieset && hwloc_bitmap_isequal(packageset, dieset)) {
hwloc_bitmap_free(dieset);
dieset = NULL;
}
if (hwloc_bitmap_first(packageset) == i) {
struct hwloc_obj *package;
unsigned mypackageid;
mypackageid = (unsigned) -1;
sprintf(str, "%s/cpu%d/topology/physical_package_id", path, i); 
if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
mypackageid = (unsigned) tmpint;
package = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PACKAGE, mypackageid);
package->cpuset = packageset;
hwloc_debug_1arg_bitmap("os package %u has cpuset %s\n",
mypackageid, packageset);
if (cpuinfo_Lprocs) {
for(j=0; j<(int) cpuinfo_numprocs; j++)
if ((int) cpuinfo_Lprocs[j].Pproc == i) {
hwloc__move_infos(&package->infos, &package->infos_count,
&cpuinfo_Lprocs[j].infos, &cpuinfo_Lprocs[j].infos_count);
}
}
hwloc_insert_object_by_cpuset(topology, package);
packageset = NULL; 
}
hwloc_bitmap_free(packageset);
}
}
if (dieset) {
struct hwloc_obj *die;
unsigned mydieid;
mydieid = (unsigned) -1;
sprintf(str, "%s/cpu%d/topology/die_id", path, i); 
if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
mydieid = (unsigned) tmpint;
die = hwloc_alloc_setup_object(topology, HWLOC_OBJ_DIE, mydieid);
die->cpuset = dieset;
hwloc_debug_1arg_bitmap("os die %u has cpuset %s\n",
mydieid, dieset);
hwloc_insert_object_by_cpuset(topology, die);
}
if (data->arch == HWLOC_LINUX_ARCH_S390
&& hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_GROUP)) {
hwloc_bitmap_t bookset, drawerset;
sprintf(str, "%s/cpu%d/topology/book_siblings", path, i);
bookset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
if (bookset) {
hwloc_bitmap_and(bookset, bookset, cpuset);
if (hwloc_bitmap_first(bookset) == i) {
struct hwloc_obj *book;
unsigned mybookid;
mybookid = (unsigned) -1;
sprintf(str, "%s/cpu%d/topology/book_id", path, i); 
if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0) {
mybookid = (unsigned) tmpint;
book = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, mybookid);
book->cpuset = bookset;
hwloc_debug_1arg_bitmap("os book %u has cpuset %s\n",
mybookid, bookset);
book->subtype = strdup("Book");
book->attr->group.kind = HWLOC_GROUP_KIND_S390_BOOK;
book->attr->group.subkind = 0;
hwloc_insert_object_by_cpuset(topology, book);
bookset = NULL; 
}
}
hwloc_bitmap_free(bookset);
sprintf(str, "%s/cpu%d/topology/drawer_siblings", path, i);
drawerset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
if (drawerset) {
hwloc_bitmap_and(drawerset, drawerset, cpuset);
if (hwloc_bitmap_first(drawerset) == i) {
struct hwloc_obj *drawer;
unsigned mydrawerid;
mydrawerid = (unsigned) -1;
sprintf(str, "%s/cpu%d/topology/drawer_id", path, i); 
if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0) {
mydrawerid = (unsigned) tmpint;
drawer = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, mydrawerid);
drawer->cpuset = drawerset;
hwloc_debug_1arg_bitmap("os drawer %u has cpuset %s\n",
mydrawerid, drawerset);
drawer->subtype = strdup("Drawer");
drawer->attr->group.kind = HWLOC_GROUP_KIND_S390_BOOK;
drawer->attr->group.subkind = 1;
hwloc_insert_object_by_cpuset(topology, drawer);
drawerset = NULL; 
}
}
hwloc_bitmap_free(drawerset);
}
}
}
{
hwloc_bitmap_t threadset;
struct hwloc_obj *thread = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PU, (unsigned) i);
threadset = hwloc_bitmap_alloc();
hwloc_bitmap_only(threadset, i);
thread->cpuset = threadset;
hwloc_debug_1arg_bitmap("thread %d has cpuset %s\n",
i, threadset);
hwloc_insert_object_by_cpuset(topology, thread);
}
for(j=0; j<10; j++) {
char str2[20]; 
hwloc_bitmap_t cacheset;
sprintf(str, "%s/cpu%d/cache/index%d/shared_cpu_map", path, i, j);
cacheset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
if (cacheset) {
if (hwloc_bitmap_iszero(cacheset)) {
hwloc_bitmap_t tmpset;
if (old_filenames)
sprintf(str, "%s/cpu%d/topology/thread_siblings", path, i);
else
sprintf(str, "%s/cpu%d/topology/core_cpus", path, i);
tmpset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
if (tmpset) {
hwloc_bitmap_free(cacheset);
cacheset = tmpset;
}
}
hwloc_bitmap_and(cacheset, cacheset, cpuset);
if (hwloc_bitmap_first(cacheset) == i) {
unsigned kB;
unsigned linesize;
unsigned sets, lines_per_tag;
unsigned depth; 
hwloc_obj_cache_type_t ctype = HWLOC_OBJ_CACHE_UNIFIED; 
hwloc_obj_type_t otype;
struct hwloc_obj *cache;
sprintf(str, "%s/cpu%d/cache/index%d/level", path, i, j); 
if (hwloc_read_path_as_uint(str, &depth, data->root_fd) < 0) {
hwloc_bitmap_free(cacheset);
continue;
}
sprintf(str, "%s/cpu%d/cache/index%d/type", path, i, j);
if (hwloc_read_path_by_length(str, str2, sizeof(str2), data->root_fd) == 0) {
if (!strncmp(str2, "Data", 4))
ctype = HWLOC_OBJ_CACHE_DATA;
else if (!strncmp(str2, "Unified", 7))
ctype = HWLOC_OBJ_CACHE_UNIFIED;
else if (!strncmp(str2, "Instruction", 11))
ctype = HWLOC_OBJ_CACHE_INSTRUCTION;
}
otype = hwloc_cache_type_by_depth_type(depth, ctype);
if (otype == HWLOC_OBJ_TYPE_NONE
|| !hwloc_filter_check_keep_object_type(topology, otype)) {
hwloc_bitmap_free(cacheset);
continue;
}
kB = 0;
sprintf(str, "%s/cpu%d/cache/index%d/size", path, i, j); 
hwloc_read_path_as_uint(str, &kB, data->root_fd);
if (!kB && otype == HWLOC_OBJ_L3CACHE && data->is_knl) {
hwloc_bitmap_free(cacheset);
continue;
}
linesize = 0;
sprintf(str, "%s/cpu%d/cache/index%d/coherency_line_size", path, i, j); 
hwloc_read_path_as_uint(str, &linesize, data->root_fd);
sets = 0;
sprintf(str, "%s/cpu%d/cache/index%d/number_of_sets", path, i, j); 
hwloc_read_path_as_uint(str, &sets, data->root_fd);
lines_per_tag = 1;
sprintf(str, "%s/cpu%d/cache/index%d/physical_line_partition", path, i, j); 
hwloc_read_path_as_uint(str, &lines_per_tag, data->root_fd);
cache = hwloc_alloc_setup_object(topology, otype, HWLOC_UNKNOWN_INDEX);
cache->attr->cache.size = ((uint64_t)kB) << 10;
cache->attr->cache.depth = depth;
cache->attr->cache.linesize = linesize;
cache->attr->cache.type = ctype;
if (!linesize || !lines_per_tag || !sets)
cache->attr->cache.associativity = 0; 
else if (sets == 1)
cache->attr->cache.associativity = 0; 
else
cache->attr->cache.associativity = (kB << 10) / linesize / lines_per_tag / sets;
cache->cpuset = cacheset;
hwloc_debug_1arg_bitmap("cache depth %u has cpuset %s\n",
depth, cacheset);
hwloc_insert_object_by_cpuset(topology, cache);
cacheset = NULL; 
++caches_added;
}
}
hwloc_bitmap_free(cacheset);
}
} hwloc_bitmap_foreach_end();
if (0 == caches_added && data->use_dt)
look_powerpc_device_tree(topology, data);
hwloc_bitmap_free(cpuset);
hwloc_bitmap_free(online_set);
return 0;
}
static int
hwloc_linux_parse_cpuinfo_x86(const char *prefix, const char *value,
struct hwloc_info_s **infos, unsigned *infos_count,
int is_global __hwloc_attribute_unused)
{
if (!strcmp("vendor_id", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUVendor", value);
} else if (!strcmp("model name", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUModel", value);
} else if (!strcmp("model", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUModelNumber", value);
} else if (!strcmp("cpu family", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUFamilyNumber", value);
} else if (!strcmp("stepping", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUStepping", value);
}
return 0;
}
static int
hwloc_linux_parse_cpuinfo_ia64(const char *prefix, const char *value,
struct hwloc_info_s **infos, unsigned *infos_count,
int is_global __hwloc_attribute_unused)
{
if (!strcmp("vendor", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUVendor", value);
} else if (!strcmp("model name", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUModel", value);
} else if (!strcmp("model", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUModelNumber", value);
} else if (!strcmp("family", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUFamilyNumber", value);
}
return 0;
}
static int
hwloc_linux_parse_cpuinfo_arm(const char *prefix, const char *value,
struct hwloc_info_s **infos, unsigned *infos_count,
int is_global __hwloc_attribute_unused)
{
if (!strcmp("Processor", prefix) 
|| !strcmp("model name", prefix) ) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUModel", value);
} else if (!strcmp("CPU implementer", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUImplementer", value);
} else if (!strcmp("CPU architecture", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUArchitecture", value);
} else if (!strcmp("CPU variant", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUVariant", value);
} else if (!strcmp("CPU part", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUPart", value);
} else if (!strcmp("CPU revision", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPURevision", value);
} else if (!strcmp("Hardware", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "HardwareName", value);
} else if (!strcmp("Revision", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "HardwareRevision", value);
} else if (!strcmp("Serial", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "HardwareSerial", value);
}
return 0;
}
static int
hwloc_linux_parse_cpuinfo_ppc(const char *prefix, const char *value,
struct hwloc_info_s **infos, unsigned *infos_count,
int is_global)
{
if (!strcmp("cpu", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "CPUModel", value);
} else if (!strcmp("platform", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "PlatformName", value);
} else if (!strcmp("model", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "PlatformModel", value);
}
else if (!strcasecmp("vendor", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "PlatformVendor", value);
} else if (!strcmp("Board ID", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "PlatformBoardID", value);
} else if (!strcmp("Board", prefix)
|| !strcasecmp("Machine", prefix)) {
if (value[0])
hwloc__add_info_nodup(infos, infos_count, "PlatformModel", value, 1);
} else if (!strcasecmp("Revision", prefix)
|| !strcmp("Hardware rev", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, is_global ? "PlatformRevision" : "CPURevision", value);
} else if (!strcmp("SVR", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "SystemVersionRegister", value);
} else if (!strcmp("PVR", prefix)) {
if (value[0])
hwloc__add_info(infos, infos_count, "ProcessorVersionRegister", value);
}
return 0;
}
static int
hwloc_linux_parse_cpuinfo_generic(const char *prefix, const char *value,
struct hwloc_info_s **infos, unsigned *infos_count,
int is_global __hwloc_attribute_unused)
{
if (!strcmp("model name", prefix)
|| !strcmp("Processor", prefix)
|| !strcmp("chip type", prefix)
|| !strcmp("cpu model", prefix)
|| !strcasecmp("cpu", prefix)) {
if (value[0])
hwloc__add_info_nodup(infos, infos_count, "CPUModel", value, 1);
}
return 0;
}
static int
hwloc_linux_parse_cpuinfo(struct hwloc_linux_backend_data_s *data,
const char *path,
struct hwloc_linux_cpuinfo_proc ** Lprocs_p,
struct hwloc_info_s **global_infos, unsigned *global_infos_count)
{
FILE *fd;
char str[128]; 
char *endptr;
unsigned allocated_Lprocs = 0;
struct hwloc_linux_cpuinfo_proc * Lprocs = NULL;
unsigned numprocs = 0;
int curproc = -1;
int (*parse_cpuinfo_func)(const char *, const char *, struct hwloc_info_s **, unsigned *, int) = NULL;
if (!(fd=hwloc_fopen(path,"r", data->root_fd)))
{
hwloc_debug("could not open %s\n", path);
return -1;
}
#      define PROCESSOR	"processor"
hwloc_debug("\n\n * Topology extraction from %s *\n\n", path);
while (fgets(str, sizeof(str), fd)!=NULL) {
unsigned long Pproc;
char *end, *dot, *prefix, *value;
int noend = 0;
end = strchr(str, '\n');
if (end)
*end = 0;
else
noend = 1;
if (!*str) {
curproc = -1;
continue;
}
dot = strchr(str, ':');
if (!dot)
continue;
if ((*str > 'z' || *str < 'a')
&& (*str > 'Z' || *str < 'A'))
continue;
prefix = str;
end = dot;
while (end[-1] == ' ' || end[-1] == '\t') end--; 
*end = 0;
value = dot+1 + strspn(dot+1, " \t");
#   define getprocnb_begin(field, var)					\
if (!strcmp(field,prefix)) {					\
var = strtoul(value,&endptr,0);					\
if (endptr==value) {						\
hwloc_debug("no number in "field" field of %s\n", path);	\
goto err;							\
} else if (var==ULONG_MAX) {					\
hwloc_debug("too big "field" number in %s\n", path); 		\
goto err;							\
}									\
hwloc_debug(field " %lu\n", var)
#   define getprocnb_end()						\
}
getprocnb_begin(PROCESSOR, Pproc);
curproc = numprocs++;
if (numprocs > allocated_Lprocs) {
struct hwloc_linux_cpuinfo_proc * tmp;
if (!allocated_Lprocs)
allocated_Lprocs = 8;
else
allocated_Lprocs *= 2;
tmp = realloc(Lprocs, allocated_Lprocs * sizeof(*Lprocs));
if (!tmp)
goto err;
Lprocs = tmp;
}
Lprocs[curproc].Pproc = Pproc;
Lprocs[curproc].infos = NULL;
Lprocs[curproc].infos_count = 0;
getprocnb_end() else {
switch (data->arch) {
case HWLOC_LINUX_ARCH_X86:
parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_x86;
break;
case HWLOC_LINUX_ARCH_ARM:
parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_arm;
break;
case HWLOC_LINUX_ARCH_POWER:
parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_ppc;
break;
case HWLOC_LINUX_ARCH_IA64:
parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_ia64;
break;
default:
parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_generic;
}
parse_cpuinfo_func(prefix, value,
curproc >= 0 ? &Lprocs[curproc].infos : global_infos,
curproc >= 0 ? &Lprocs[curproc].infos_count : global_infos_count,
curproc < 0);
}
if (noend) {
if (fscanf(fd,"%*[^\n]") == EOF)
break;
getc(fd);
}
}
fclose(fd);
*Lprocs_p = Lprocs;
return numprocs;
err:
fclose(fd);
free(Lprocs);
*Lprocs_p = NULL;
return -1;
}
static void
hwloc_linux_free_cpuinfo(struct hwloc_linux_cpuinfo_proc * Lprocs, unsigned numprocs,
struct hwloc_info_s *global_infos, unsigned global_infos_count)
{
if (Lprocs) {
unsigned i;
for(i=0; i<numprocs; i++) {
hwloc__free_infos(Lprocs[i].infos, Lprocs[i].infos_count);
}
free(Lprocs);
}
hwloc__free_infos(global_infos, global_infos_count);
}
static void
hwloc__linux_get_mic_sn(struct hwloc_topology *topology, struct hwloc_linux_backend_data_s *data)
{
char line[64], *tmp, *end;
if (hwloc_read_path_by_length("/proc/elog", line, sizeof(line), data->root_fd) < 0)
return;
if (strncmp(line, "Card ", 5))
return;
tmp = line + 5;
end = strchr(tmp, ':');
if (!end)
return;
*end = '\0';
if (tmp[0])
hwloc_obj_add_info(hwloc_get_root_obj(topology), "MICSerialNumber", tmp);
}
static void
hwloc_gather_system_info(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data)
{
FILE *file;
char line[128]; 
const char *env;
memset(&data->utsname, 0, sizeof(data->utsname));
data->fallback_nbprocessors = -1; 
data->pagesize = 4096;
if (topology->is_thissystem) {
uname(&data->utsname);
data->fallback_nbprocessors = hwloc_fallback_nbprocessors(0); 
data->pagesize = hwloc_getpagesize();
}
if (!data->is_real_fsroot) {
file = hwloc_fopen("/proc/hwloc-nofile-info", "r", data->root_fd);
if (file) {
while (fgets(line, sizeof(line), file)) {
char *tmp = strchr(line, '\n');
if (!strncmp("OSName: ", line, 8)) {
if (tmp)
*tmp = '\0';
strncpy(data->utsname.sysname, line+8, sizeof(data->utsname.sysname));
data->utsname.sysname[sizeof(data->utsname.sysname)-1] = '\0';
} else if (!strncmp("OSRelease: ", line, 11)) {
if (tmp)
*tmp = '\0';
strncpy(data->utsname.release, line+11, sizeof(data->utsname.release));
data->utsname.release[sizeof(data->utsname.release)-1] = '\0';
} else if (!strncmp("OSVersion: ", line, 11)) {
if (tmp)
*tmp = '\0';
strncpy(data->utsname.version, line+11, sizeof(data->utsname.version));
data->utsname.version[sizeof(data->utsname.version)-1] = '\0';
} else if (!strncmp("HostName: ", line, 10)) {
if (tmp)
*tmp = '\0';
strncpy(data->utsname.nodename, line+10, sizeof(data->utsname.nodename));
data->utsname.nodename[sizeof(data->utsname.nodename)-1] = '\0';
} else if (!strncmp("Architecture: ", line, 14)) {
if (tmp)
*tmp = '\0';
strncpy(data->utsname.machine, line+14, sizeof(data->utsname.machine));
data->utsname.machine[sizeof(data->utsname.machine)-1] = '\0';
} else if (!strncmp("FallbackNbProcessors: ", line, 22)) {
if (tmp)
*tmp = '\0';
data->fallback_nbprocessors = atoi(line+22);
} else if (!strncmp("PageSize: ", line, 10)) {
if (tmp)
*tmp = '\0';
data->pagesize = strtoull(line+10, NULL, 10);
} else {
hwloc_debug("ignored /proc/hwloc-nofile-info line %s\n", line);
}
}
fclose(file);
}
}
env = getenv("HWLOC_DUMP_NOFILE_INFO");
if (env && *env) {
file = fopen(env, "w");
if (file) {
if (*data->utsname.sysname)
fprintf(file, "OSName: %s\n", data->utsname.sysname);
if (*data->utsname.release)
fprintf(file, "OSRelease: %s\n", data->utsname.release);
if (*data->utsname.version)
fprintf(file, "OSVersion: %s\n", data->utsname.version);
if (*data->utsname.nodename)
fprintf(file, "HostName: %s\n", data->utsname.nodename);
if (*data->utsname.machine)
fprintf(file, "Architecture: %s\n", data->utsname.machine);
fprintf(file, "FallbackNbProcessors: %d\n", data->fallback_nbprocessors);
fprintf(file, "PageSize: %llu\n", (unsigned long long) data->pagesize);
fclose(file);
}
}
#if (defined HWLOC_X86_32_ARCH) || (defined HWLOC_X86_64_ARCH) 
if (topology->is_thissystem)
data->arch = HWLOC_LINUX_ARCH_X86;
#endif
if (data->arch == HWLOC_LINUX_ARCH_UNKNOWN && *data->utsname.machine) {
if (!strcmp(data->utsname.machine, "x86_64")
|| (data->utsname.machine[0] == 'i' && !strcmp(data->utsname.machine+2, "86"))
|| !strcmp(data->utsname.machine, "k1om"))
data->arch = HWLOC_LINUX_ARCH_X86;
else if (!strncmp(data->utsname.machine, "arm", 3))
data->arch = HWLOC_LINUX_ARCH_ARM;
else if (!strncmp(data->utsname.machine, "ppc", 3)
|| !strncmp(data->utsname.machine, "power", 5))
data->arch = HWLOC_LINUX_ARCH_POWER;
else if (!strncmp(data->utsname.machine, "s390", 4))
data->arch = HWLOC_LINUX_ARCH_S390;
else if (!strcmp(data->utsname.machine, "ia64"))
data->arch = HWLOC_LINUX_ARCH_IA64;
}
}
static int
hwloc_linux_try_hardwired_cpuinfo(struct hwloc_backend *backend)
{
struct hwloc_topology *topology = backend->topology;
struct hwloc_linux_backend_data_s *data = backend->private_data;
if (getenv("HWLOC_NO_HARDWIRED_TOPOLOGY"))
return -1;
if (!strcmp(data->utsname.machine, "s64fx")) {
char line[128];
if (hwloc_read_path_by_length("/proc/cpuinfo", line, sizeof(line), data->root_fd) < 0)
return -1;
if (strncmp(line, "cpu\t", 4))
return -1;
if (strstr(line, "Fujitsu SPARC64 VIIIfx"))
return hwloc_look_hardwired_fujitsu_k(topology);
else if (strstr(line, "Fujitsu SPARC64 IXfx"))
return hwloc_look_hardwired_fujitsu_fx10(topology);
else if (strstr(line, "FUJITSU SPARC64 XIfx"))
return hwloc_look_hardwired_fujitsu_fx100(topology);
}
return -1;
}
static void hwloc_linux__get_allowed_resources(hwloc_topology_t topology, const char *root_path, int root_fd, char **cpuset_namep)
{
char *cpuset_mntpnt, *cgroup_mntpnt, *cpuset_name = NULL;
hwloc_find_linux_cpuset_mntpnt(&cgroup_mntpnt, &cpuset_mntpnt, root_path);
if (cgroup_mntpnt || cpuset_mntpnt) {
cpuset_name = hwloc_read_linux_cpuset_name(root_fd, topology->pid);
if (cpuset_name) {
hwloc_admin_disable_set_from_cpuset(root_fd, cgroup_mntpnt, cpuset_mntpnt, cpuset_name, "cpus", topology->allowed_cpuset);
hwloc_admin_disable_set_from_cpuset(root_fd, cgroup_mntpnt, cpuset_mntpnt, cpuset_name, "mems", topology->allowed_nodeset);
}
free(cgroup_mntpnt);
free(cpuset_mntpnt);
}
*cpuset_namep = cpuset_name;
}
static void
hwloc_linux_fallback_pu_level(struct hwloc_backend *backend)
{
struct hwloc_topology *topology = backend->topology;
struct hwloc_linux_backend_data_s *data = backend->private_data;
if (data->fallback_nbprocessors >= 1)
topology->support.discovery->pu = 1;
else
data->fallback_nbprocessors = 1;
hwloc_setup_pu_level(topology, data->fallback_nbprocessors);
}
static const char *find_sysfs_cpu_path(int root_fd, int *old_filenames)
{
if (!hwloc_access("/sys/bus/cpu/devices", R_OK|X_OK, root_fd)) {
if (!hwloc_access("/sys/bus/cpu/devices/cpu0/topology/package_cpus", R_OK, root_fd)
|| !hwloc_access("/sys/bus/cpu/devices/cpu0/topology/core_cpus", R_OK, root_fd)) {
return "/sys/bus/cpu/devices";
}
if (!hwloc_access("/sys/bus/cpu/devices/cpu0/topology/core_siblings", R_OK, root_fd)
|| !hwloc_access("/sys/bus/cpu/devices/cpu0/topology/thread_siblings", R_OK, root_fd)) {
*old_filenames = 1;
return "/sys/bus/cpu/devices";
}
}
if (!hwloc_access("/sys/devices/system/cpu", R_OK|X_OK, root_fd)) {
if (!hwloc_access("/sys/devices/system/cpu/cpu0/topology/package_cpus", R_OK, root_fd)
|| !hwloc_access("/sys/devices/system/cpu/cpu0/topology/core_cpus", R_OK, root_fd)) {
return "/sys/devices/system/cpu";
}
if (!hwloc_access("/sys/devices/system/cpu/cpu0/topology/core_siblings", R_OK, root_fd)
|| !hwloc_access("/sys/devices/system/cpu/cpu0/topology/thread_siblings", R_OK, root_fd)) {
*old_filenames = 1;
return "/sys/devices/system/cpu";
}
}
return NULL;
}
static const char *find_sysfs_node_path(int root_fd)
{
if (!hwloc_access("/sys/bus/node/devices", R_OK|X_OK, root_fd)
&& !hwloc_access("/sys/bus/node/devices/node0/cpumap", R_OK, root_fd))
return "/sys/bus/node/devices";
if (!hwloc_access("/sys/devices/system/node", R_OK|X_OK, root_fd)
&& !hwloc_access("/sys/devices/system/node/node0/cpumap", R_OK, root_fd))
return "/sys/devices/system/node";
return NULL;
}
static int
hwloc_linuxfs_look_cpu(struct hwloc_backend *backend, struct hwloc_disc_status *dstatus)
{
struct hwloc_topology *topology = backend->topology;
struct hwloc_linux_backend_data_s *data = backend->private_data;
unsigned nbnodes;
char *cpuset_name = NULL;
struct hwloc_linux_cpuinfo_proc * Lprocs = NULL;
struct hwloc_info_s *global_infos = NULL;
unsigned global_infos_count = 0;
int numprocs;
int already_pus;
int already_numanodes;
const char *sysfs_cpu_path;
const char *sysfs_node_path;
int old_siblings_filenames = 0;
int err;
sysfs_cpu_path = find_sysfs_cpu_path(data->root_fd, &old_siblings_filenames);
hwloc_debug("Found sysfs cpu files under %s with %s topology filenames\n",
sysfs_cpu_path, old_siblings_filenames ? "old" : "new");
sysfs_node_path = find_sysfs_node_path(data->root_fd);
hwloc_debug("Found sysfs node files under %s\n",
sysfs_node_path);
already_pus = (topology->levels[0][0]->complete_cpuset != NULL
&& !hwloc_bitmap_iszero(topology->levels[0][0]->complete_cpuset));
already_numanodes = (topology->levels[0][0]->complete_nodeset != NULL
&& !hwloc_bitmap_iszero(topology->levels[0][0]->complete_nodeset));
if (already_numanodes)
hwloc_topology_reconnect(topology, 0);
hwloc_alloc_root_sets(topology->levels[0][0]);
hwloc_gather_system_info(topology, data);
numprocs = hwloc_linux_parse_cpuinfo(data, "/proc/cpuinfo", &Lprocs, &global_infos, &global_infos_count);
if (numprocs < 0)
numprocs = 0;
if (data->arch == HWLOC_LINUX_ARCH_X86 && numprocs > 0) {
unsigned i;
const char *cpuvendor = NULL, *cpufamilynumber = NULL, *cpumodelnumber = NULL;
for(i=0; i<Lprocs[0].infos_count; i++) {
if (!strcmp(Lprocs[0].infos[i].name, "CPUVendor")) {
cpuvendor = Lprocs[0].infos[i].value;
} else if (!strcmp(Lprocs[0].infos[i].name, "CPUFamilyNumber")) {
cpufamilynumber = Lprocs[0].infos[i].value;
} else if (!strcmp(Lprocs[0].infos[i].name, "CPUModelNumber")) {
cpumodelnumber = Lprocs[0].infos[i].value;
}
}
if (cpuvendor && !strcmp(cpuvendor, "GenuineIntel")
&& cpufamilynumber && !strcmp(cpufamilynumber, "6")
&& cpumodelnumber && (!strcmp(cpumodelnumber, "87")
|| !strcmp(cpumodelnumber, "133")))
data->is_knl = 1;
if (cpuvendor && !strcmp(cpuvendor, "AuthenticAMD")
&& cpufamilynumber
&& (!strcmp(cpufamilynumber, "21")
|| !strcmp(cpufamilynumber, "22")))
data->is_amd_with_CU = 1;
}
if (!(dstatus->flags & HWLOC_DISC_STATUS_FLAG_GOT_ALLOWED_RESOURCES)) {
hwloc_linux__get_allowed_resources(topology, data->root_path, data->root_fd, &cpuset_name);
dstatus->flags |= HWLOC_DISC_STATUS_FLAG_GOT_ALLOWED_RESOURCES;
}
if (already_pus)
goto cpudone;
err = hwloc_linux_try_hardwired_cpuinfo(backend);
if (!err)
goto cpudone;
hwloc__move_infos(&hwloc_get_root_obj(topology)->infos, &hwloc_get_root_obj(topology)->infos_count,
&global_infos, &global_infos_count);
if (!sysfs_cpu_path) {
hwloc_linux_fallback_pu_level(backend);
if (data->use_dt)
look_powerpc_device_tree(topology, data);
} else {
if (look_sysfscpu(topology, data, sysfs_cpu_path, old_siblings_filenames, Lprocs, numprocs) < 0)
hwloc_linux_fallback_pu_level(backend);
}
cpudone:
hwloc_get_machine_meminfo(data, &topology->machine_memory);
if (sysfs_node_path) {
if (hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE) > 0)
annotate_sysfsnode(topology, data, sysfs_node_path, &nbnodes);
else
look_sysfsnode(topology, data, sysfs_node_path, &nbnodes);
} else
nbnodes = 0;
hwloc__get_dmi_id_info(data, topology->levels[0][0]);
hwloc_obj_add_info(topology->levels[0][0], "Backend", "Linux");
if (cpuset_name) {
hwloc_obj_add_info(topology->levels[0][0], "LinuxCgroup", cpuset_name);
free(cpuset_name);
}
hwloc__linux_get_mic_sn(topology, data);
hwloc_add_uname_info(topology, &data->utsname);
hwloc_linux_free_cpuinfo(Lprocs, numprocs, global_infos, global_infos_count);
return 0;
}
static int
hwloc_linux_backend_get_pci_busid_cpuset(struct hwloc_backend *backend,
struct hwloc_pcidev_attr_s *busid, hwloc_bitmap_t cpuset)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
char path[256];
int err;
snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/local_cpus",
busid->domain, busid->bus,
busid->dev, busid->func);
err = hwloc__read_path_as_cpumask(path, cpuset, data->root_fd);
if (!err && !hwloc_bitmap_iszero(cpuset))
return 0;
return -1;
}
#ifdef HWLOC_HAVE_LINUXIO
#define HWLOC_LINUXFS_OSDEV_FLAG_FIND_VIRTUAL (1U<<0)
#define HWLOC_LINUXFS_OSDEV_FLAG_FIND_USB (1U<<1)
#define HWLOC_LINUXFS_OSDEV_FLAG_BLOCK_WITH_SECTORS (1U<<2)
#define HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS (1U<<31)
static hwloc_obj_t
hwloc_linuxfs_find_osdev_parent(struct hwloc_backend *backend, int root_fd,
const char *osdevpath, unsigned osdev_flags)
{
struct hwloc_topology *topology = backend->topology;
char path[256], buf[10];
int fd;
int foundpci;
unsigned pcidomain = 0, pcibus = 0, pcidev = 0, pcifunc = 0;
unsigned _pcidomain, _pcibus, _pcidev, _pcifunc;
hwloc_bitmap_t cpuset;
const char *tmp;
hwloc_obj_t parent;
char *devicesubdir;
int err;
if (osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS)
devicesubdir = "..";
else
devicesubdir = "device";
err = hwloc_readlink(osdevpath, path, sizeof(path), root_fd);
if (err < 0) {
char olddevpath[256];
snprintf(olddevpath, sizeof(olddevpath), "%s/device", osdevpath);
err = hwloc_readlink(olddevpath, path, sizeof(path), root_fd);
if (err < 0)
return NULL;
}
path[err] = '\0';
if (!(osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_FIND_VIRTUAL)) {
if (strstr(path, "/virtual/"))
return NULL;
}
if (!(osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_FIND_USB)) {
if (strstr(path, "/usb"))
return NULL;
}
tmp = strstr(path, "/pci");
if (!tmp)
goto nopci;
tmp = strchr(tmp+4, '/');
if (!tmp)
goto nopci;
tmp++;
foundpci = 0;
nextpci:
if (sscanf(tmp+1, "%x:%x:%x.%x", &_pcidomain, &_pcibus, &_pcidev, &_pcifunc) == 4) {
foundpci = 1;
pcidomain = _pcidomain;
pcibus = _pcibus;
pcidev = _pcidev;
pcifunc = _pcifunc;
tmp += 13;
goto nextpci;
}
if (sscanf(tmp+1, "%x:%x.%x", &_pcibus, &_pcidev, &_pcifunc) == 3) {
foundpci = 1;
pcidomain = 0;
pcibus = _pcibus;
pcidev = _pcidev;
pcifunc = _pcifunc;
tmp += 8;
goto nextpci;
}
if (foundpci) {
parent = hwloc_pci_find_parent_by_busid(topology, pcidomain, pcibus, pcidev, pcifunc);
if (parent)
return parent;
}
nopci:
snprintf(path, sizeof(path), "%s/%s/numa_node", osdevpath, devicesubdir);
fd = hwloc_open(path, root_fd);
if (fd >= 0) {
err = read(fd, buf, sizeof(buf));
close(fd);
if (err > 0) {
int node = atoi(buf);
if (node >= 0) {
parent = hwloc_get_numanode_obj_by_os_index(topology, (unsigned) node);
if (parent) {
while (hwloc__obj_type_is_memory(parent->type))
parent = parent->parent;
return parent;
}
}
}
}
snprintf(path, sizeof(path), "%s/%s/local_cpus", osdevpath, devicesubdir);
cpuset = hwloc__alloc_read_path_as_cpumask(path, root_fd);
if (cpuset) {
parent = hwloc_find_insert_io_parent_by_complete_cpuset(topology, cpuset);
hwloc_bitmap_free(cpuset);
if (parent)
return parent;
}
return hwloc_get_root_obj(topology);
}
static hwloc_obj_t
hwloc_linux_add_os_device(struct hwloc_backend *backend, struct hwloc_obj *pcidev, hwloc_obj_osdev_type_t type, const char *name)
{
struct hwloc_topology *topology = backend->topology;
struct hwloc_obj *obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_OS_DEVICE, HWLOC_UNKNOWN_INDEX);
obj->name = strdup(name);
obj->attr->osdev.type = type;
hwloc_insert_object_by_parent(topology, pcidev, obj);
return obj;
}
static void
hwloc_linuxfs_block_class_fillinfos(struct hwloc_backend *backend __hwloc_attribute_unused, int root_fd,
struct hwloc_obj *obj, const char *osdevpath, unsigned osdev_flags)
{
#ifdef HWLOC_HAVE_LIBUDEV
struct hwloc_linux_backend_data_s *data = backend->private_data;
#endif
FILE *file;
char path[296]; 
char line[128];
char vendor[64] = "";
char model[64] = "";
char serial[64] = "";
char revision[64] = "";
char blocktype[64] = "";
unsigned sectorsize = 0;
unsigned major_id, minor_id;
char *devicesubdir;
char *tmp;
if (osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS)
devicesubdir = "..";
else
devicesubdir = "device";
snprintf(path, sizeof(path), "%s/size", osdevpath);
if (!hwloc_read_path_by_length(path, line, sizeof(line), root_fd)) {
unsigned long long value = strtoull(line, NULL, 10);
snprintf(line, sizeof(line), "%llu",
(osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_BLOCK_WITH_SECTORS) ? value / 2 : value >> 10);
hwloc_obj_add_info(obj, "Size", line);
}
snprintf(path, sizeof(path), "%s/queue/hw_sector_size", osdevpath);
if (!hwloc_read_path_by_length(path, line, sizeof(line), root_fd)) {
sectorsize = strtoul(line, NULL, 10);
}
snprintf(path, sizeof(path), "%s/%s/devtype", osdevpath, devicesubdir);
if (!hwloc_read_path_by_length(path, line, sizeof(line), root_fd)) {
if (!strncmp(line, "nd_", 3))
strcpy(blocktype, "NVDIMM"); 
}
if (sectorsize) {
snprintf(line, sizeof(line), "%u", sectorsize);
hwloc_obj_add_info(obj, "SectorSize", line);
}
snprintf(path, sizeof(path), "%s/dev", osdevpath);
if (hwloc_read_path_by_length(path, line, sizeof(line), root_fd) < 0)
goto done;
if (sscanf(line, "%u:%u", &major_id, &minor_id) != 2)
goto done;
tmp = strchr(line, '\n');
if (tmp)
*tmp = '\0';
hwloc_obj_add_info(obj, "LinuxDeviceID", line);
#ifdef HWLOC_HAVE_LIBUDEV
if (data->udev) {
struct udev_device *dev;
const char *prop;
dev = udev_device_new_from_subsystem_sysname(data->udev, "block", obj->name);
if (!dev)
goto done;
prop = udev_device_get_property_value(dev, "ID_VENDOR");
if (prop) {
strncpy(vendor, prop, sizeof(vendor));
vendor[sizeof(vendor)-1] = '\0';
}
prop = udev_device_get_property_value(dev, "ID_MODEL");
if (prop) {
strncpy(model, prop, sizeof(model));
model[sizeof(model)-1] = '\0';
}
prop = udev_device_get_property_value(dev, "ID_REVISION");
if (prop) {
strncpy(revision, prop, sizeof(revision));
revision[sizeof(revision)-1] = '\0';
}
prop = udev_device_get_property_value(dev, "ID_SERIAL_SHORT");
if (prop) {
strncpy(serial, prop, sizeof(serial));
serial[sizeof(serial)-1] = '\0';
}
prop = udev_device_get_property_value(dev, "ID_TYPE");
if (prop) {
strncpy(blocktype, prop, sizeof(blocktype));
blocktype[sizeof(blocktype)-1] = '\0';
}
udev_device_unref(dev);
} else
#endif
{
snprintf(path, sizeof(path), "/run/udev/data/b%u:%u", major_id, minor_id);
file = hwloc_fopen(path, "r", root_fd);
if (!file)
goto done;
while (NULL != fgets(line, sizeof(line), file)) {
tmp = strchr(line, '\n');
if (tmp)
*tmp = '\0';
if (!strncmp(line, "E:ID_VENDOR=", strlen("E:ID_VENDOR="))) {
strncpy(vendor, line+strlen("E:ID_VENDOR="), sizeof(vendor));
vendor[sizeof(vendor)-1] = '\0';
} else if (!strncmp(line, "E:ID_MODEL=", strlen("E:ID_MODEL="))) {
strncpy(model, line+strlen("E:ID_MODEL="), sizeof(model));
model[sizeof(model)-1] = '\0';
} else if (!strncmp(line, "E:ID_REVISION=", strlen("E:ID_REVISION="))) {
strncpy(revision, line+strlen("E:ID_REVISION="), sizeof(revision));
revision[sizeof(revision)-1] = '\0';
} else if (!strncmp(line, "E:ID_SERIAL_SHORT=", strlen("E:ID_SERIAL_SHORT="))) {
strncpy(serial, line+strlen("E:ID_SERIAL_SHORT="), sizeof(serial));
serial[sizeof(serial)-1] = '\0';
} else if (!strncmp(line, "E:ID_TYPE=", strlen("E:ID_TYPE="))) {
strncpy(blocktype, line+strlen("E:ID_TYPE="), sizeof(blocktype));
blocktype[sizeof(blocktype)-1] = '\0';
}
}
fclose(file);
}
done:
if (!strcasecmp(vendor, "ATA"))
*vendor = '\0';
if (!*vendor) {
if (!strncasecmp(model, "wd", 2))
strcpy(vendor, "Western Digital");
else if (!strncasecmp(model, "st", 2))
strcpy(vendor, "Seagate");
else if (!strncasecmp(model, "samsung", 7))
strcpy(vendor, "Samsung");
else if (!strncasecmp(model, "sandisk", 7))
strcpy(vendor, "SanDisk");
else if (!strncasecmp(model, "toshiba", 7))
strcpy(vendor, "Toshiba");
}
if (*vendor)
hwloc_obj_add_info(obj, "Vendor", vendor);
if (*model)
hwloc_obj_add_info(obj, "Model", model);
if (*revision)
hwloc_obj_add_info(obj, "Revision", revision);
if (*serial)
hwloc_obj_add_info(obj, "SerialNumber", serial);
if (!strcmp(blocktype, "disk") || !strncmp(obj->name, "nvme", 4))
obj->subtype = strdup("Disk");
else if (!strcmp(blocktype, "NVDIMM")) 
obj->subtype = strdup("NVDIMM");
else if (!strcmp(blocktype, "tape"))
obj->subtype = strdup("Tape");
else if (!strcmp(blocktype, "cd") || !strcmp(blocktype, "floppy") || !strcmp(blocktype, "optical"))
obj->subtype = strdup("Removable Media Device");
else {
}
}
static int
hwloc_linuxfs_lookup_block_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
int root_fd = data->root_fd;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/class/block", root_fd);
if (!dir)
return 0;
osdev_flags |= HWLOC_LINUXFS_OSDEV_FLAG_BLOCK_WITH_SECTORS; 
while ((dirent = readdir(dir)) != NULL) {
char path[256];
struct stat stbuf;
hwloc_obj_t obj, parent;
int err;
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
err = snprintf(path, sizeof(path), "/sys/class/block/%s/partition", dirent->d_name);
if ((size_t) err < sizeof(path)
&& hwloc_stat(path, &stbuf, root_fd) >= 0)
continue;
err = snprintf(path, sizeof(path), "/sys/class/block/%s", dirent->d_name);
if ((size_t) err >= sizeof(path))
continue;
parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
if (!parent)
continue;
obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_BLOCK, dirent->d_name);
hwloc_linuxfs_block_class_fillinfos(backend, root_fd, obj, path, osdev_flags);
}
closedir(dir);
return 0;
}
static int
hwloc_linuxfs_lookup_dax_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
int root_fd = data->root_fd;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/bus/dax/devices", root_fd);
if (dir) {
int found = 0;
while ((dirent = readdir(dir)) != NULL) {
char path[300];
char driver[256];
hwloc_obj_t obj, parent;
int err;
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
found++;
err = snprintf(path, sizeof(path), "/sys/bus/dax/devices/%s/driver", dirent->d_name);
if ((size_t) err >= sizeof(path))
continue;
err = hwloc_readlink(path, driver, sizeof(driver), root_fd);
if (err >= 0) {
driver[err] = '\0';
if (!strcmp(driver+err-5, "/kmem"))
continue;
}
snprintf(path, sizeof(path), "/sys/bus/dax/devices/%s", dirent->d_name);
parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags | HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS);
if (!parent)
continue;
obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_BLOCK, dirent->d_name);
hwloc_linuxfs_block_class_fillinfos(backend, root_fd, obj, path, osdev_flags | HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS);
}
closedir(dir);
if (found)
return 0;
}
dir = hwloc_opendir("/sys/class/dax", root_fd);
if (dir) {
while ((dirent = readdir(dir)) != NULL) {
char path[256];
hwloc_obj_t obj, parent;
int err;
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
err = snprintf(path, sizeof(path), "/sys/class/dax/%s", dirent->d_name);
if ((size_t) err >= sizeof(path))
continue;
parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
if (!parent)
continue;
obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_BLOCK, dirent->d_name);
hwloc_linuxfs_block_class_fillinfos(backend, root_fd, obj, path, osdev_flags);
}
closedir(dir);
}
return 0;
}
static void
hwloc_linuxfs_net_class_fillinfos(int root_fd,
struct hwloc_obj *obj, const char *osdevpath)
{
struct stat st;
char path[296]; 
char address[128];
int err;
snprintf(path, sizeof(path), "%s/address", osdevpath);
if (!hwloc_read_path_by_length(path, address, sizeof(address), root_fd)) {
char *eol = strchr(address, '\n');
if (eol)
*eol = 0;
hwloc_obj_add_info(obj, "Address", address);
}
snprintf(path, sizeof(path), "%s/device/infiniband", osdevpath);
if (!hwloc_stat(path, &st, root_fd)) {
char hexid[16];
snprintf(path, sizeof(path), "%s/dev_port", osdevpath);
err = hwloc_read_path_by_length(path, hexid, sizeof(hexid), root_fd);
if (err < 0) {
snprintf(path, sizeof(path), "%s/dev_id", osdevpath);
err = hwloc_read_path_by_length(path, hexid, sizeof(hexid), root_fd);
}
if (!err) {
char *eoid;
unsigned long port;
port = strtoul(hexid, &eoid, 0);
if (eoid != hexid) {
char portstr[21];
snprintf(portstr, sizeof(portstr), "%lu", port+1);
hwloc_obj_add_info(obj, "Port", portstr);
}
}
}
}
static int
hwloc_linuxfs_lookup_net_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
int root_fd = data->root_fd;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/class/net", root_fd);
if (!dir)
return 0;
while ((dirent = readdir(dir)) != NULL) {
char path[256];
hwloc_obj_t obj, parent;
int err;
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
err = snprintf(path, sizeof(path), "/sys/class/net/%s", dirent->d_name);
if ((size_t) err >= sizeof(path))
continue;
parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
if (!parent)
continue;
obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_NETWORK, dirent->d_name);
hwloc_linuxfs_net_class_fillinfos(root_fd, obj, path);
}
closedir(dir);
return 0;
}
static void
hwloc_linuxfs_infiniband_class_fillinfos(int root_fd,
struct hwloc_obj *obj, const char *osdevpath)
{
char path[296]; 
char guidvalue[20];
unsigned i,j;
snprintf(path, sizeof(path), "%s/node_guid", osdevpath);
if (!hwloc_read_path_by_length(path, guidvalue, sizeof(guidvalue), root_fd)) {
size_t len;
len = strspn(guidvalue, "0123456789abcdefx:");
guidvalue[len] = '\0';
hwloc_obj_add_info(obj, "NodeGUID", guidvalue);
}
snprintf(path, sizeof(path), "%s/sys_image_guid", osdevpath);
if (!hwloc_read_path_by_length(path, guidvalue, sizeof(guidvalue), root_fd)) {
size_t len;
len = strspn(guidvalue, "0123456789abcdefx:");
guidvalue[len] = '\0';
hwloc_obj_add_info(obj, "SysImageGUID", guidvalue);
}
for(i=1; ; i++) {
char statevalue[2];
char lidvalue[11];
char gidvalue[40];
snprintf(path, sizeof(path), "%s/ports/%u/state", osdevpath, i);
if (!hwloc_read_path_by_length(path, statevalue, sizeof(statevalue), root_fd)) {
char statename[32];
statevalue[1] = '\0'; 
snprintf(statename, sizeof(statename), "Port%uState", i);
hwloc_obj_add_info(obj, statename, statevalue);
} else {
break;
}
snprintf(path, sizeof(path), "%s/ports/%u/lid", osdevpath, i);
if (!hwloc_read_path_by_length(path, lidvalue, sizeof(lidvalue), root_fd)) {
char lidname[32];
size_t len;
len = strspn(lidvalue, "0123456789abcdefx");
lidvalue[len] = '\0';
snprintf(lidname, sizeof(lidname), "Port%uLID", i);
hwloc_obj_add_info(obj, lidname, lidvalue);
}
snprintf(path, sizeof(path), "%s/ports/%u/lid_mask_count", osdevpath, i);
if (!hwloc_read_path_by_length(path, lidvalue, sizeof(lidvalue), root_fd)) {
char lidname[32];
size_t len;
len = strspn(lidvalue, "0123456789");
lidvalue[len] = '\0';
snprintf(lidname, sizeof(lidname), "Port%uLMC", i);
hwloc_obj_add_info(obj, lidname, lidvalue);
}
for(j=0; ; j++) {
snprintf(path, sizeof(path), "%s/ports/%u/gids/%u", osdevpath, i, j);
if (!hwloc_read_path_by_length(path, gidvalue, sizeof(gidvalue), root_fd)) {
char gidname[32];
size_t len;
len = strspn(gidvalue, "0123456789abcdefx:");
gidvalue[len] = '\0';
if (strncmp(gidvalue+20, "0000:0000:0000:0000", 19)) {
snprintf(gidname, sizeof(gidname), "Port%uGID%u", i, j);
hwloc_obj_add_info(obj, gidname, gidvalue);
}
} else {
break;
}
}
}
}
static int
hwloc_linuxfs_lookup_infiniband_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
int root_fd = data->root_fd;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/class/infiniband", root_fd);
if (!dir)
return 0;
while ((dirent = readdir(dir)) != NULL) {
char path[256];
hwloc_obj_t obj, parent;
int err;
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
if (!strncmp(dirent->d_name, "scif", 4))
continue;
err = snprintf(path, sizeof(path), "/sys/class/infiniband/%s", dirent->d_name);
if ((size_t) err > sizeof(path))
continue;
parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
if (!parent)
continue;
obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_OPENFABRICS, dirent->d_name);
hwloc_linuxfs_infiniband_class_fillinfos(root_fd, obj, path);
}
closedir(dir);
return 0;
}
static void
hwloc_linuxfs_mic_class_fillinfos(int root_fd,
struct hwloc_obj *obj, const char *osdevpath)
{
char path[296]; 
char family[64];
char sku[64];
char sn[64];
char string[21];
obj->subtype = strdup("MIC");
snprintf(path, sizeof(path), "%s/family", osdevpath);
if (!hwloc_read_path_by_length(path, family, sizeof(family), root_fd)) {
char *eol = strchr(family, '\n');
if (eol)
*eol = 0;
hwloc_obj_add_info(obj, "MICFamily", family);
}
snprintf(path, sizeof(path), "%s/sku", osdevpath);
if (!hwloc_read_path_by_length(path, sku, sizeof(sku), root_fd)) {
char *eol = strchr(sku, '\n');
if (eol)
*eol = 0;
hwloc_obj_add_info(obj, "MICSKU", sku);
}
snprintf(path, sizeof(path), "%s/serialnumber", osdevpath);
if (!hwloc_read_path_by_length(path, sn, sizeof(sn), root_fd)) {
char *eol;
eol = strchr(sn, '\n');
if (eol)
*eol = 0;
hwloc_obj_add_info(obj, "MICSerialNumber", sn);
}
snprintf(path, sizeof(path), "%s/active_cores", osdevpath);
if (!hwloc_read_path_by_length(path, string, sizeof(string), root_fd)) {
unsigned long count = strtoul(string, NULL, 16);
snprintf(string, sizeof(string), "%lu", count);
hwloc_obj_add_info(obj, "MICActiveCores", string);
}
snprintf(path, sizeof(path), "%s/memsize", osdevpath);
if (!hwloc_read_path_by_length(path, string, sizeof(string), root_fd)) {
unsigned long count = strtoul(string, NULL, 16);
snprintf(string, sizeof(string), "%lu", count);
hwloc_obj_add_info(obj, "MICMemorySize", string);
}
}
static int
hwloc_linuxfs_lookup_mic_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
int root_fd = data->root_fd;
unsigned idx;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/class/mic", root_fd);
if (!dir)
return 0;
while ((dirent = readdir(dir)) != NULL) {
char path[256];
hwloc_obj_t obj, parent;
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
if (sscanf(dirent->d_name, "mic%u", &idx) != 1)
continue;
snprintf(path, sizeof(path), "/sys/class/mic/mic%u", idx);
parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
if (!parent)
continue;
obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_COPROC, dirent->d_name);
hwloc_linuxfs_mic_class_fillinfos(root_fd, obj, path);
}
closedir(dir);
return 0;
}
static int
hwloc_linuxfs_lookup_drm_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
int root_fd = data->root_fd;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/class/drm", root_fd);
if (!dir)
return 0;
while ((dirent = readdir(dir)) != NULL) {
char path[256];
hwloc_obj_t parent;
struct stat stbuf;
int err;
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
err = snprintf(path, sizeof(path), "/sys/class/drm/%s/dev", dirent->d_name);
if ((size_t) err < sizeof(path)
&& hwloc_stat(path, &stbuf, root_fd) < 0)
continue;
err = snprintf(path, sizeof(path), "/sys/class/drm/%s", dirent->d_name);
if ((size_t) err >= sizeof(path))
continue;
parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
if (!parent)
continue;
hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_GPU, dirent->d_name);
}
closedir(dir);
return 0;
}
static int
hwloc_linuxfs_lookup_dma_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
int root_fd = data->root_fd;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/class/dma", root_fd);
if (!dir)
return 0;
while ((dirent = readdir(dir)) != NULL) {
char path[256];
hwloc_obj_t parent;
int err;
if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
continue;
err = snprintf(path, sizeof(path), "/sys/class/dma/%s", dirent->d_name);
if ((size_t) err >= sizeof(path))
continue;
parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
if (!parent)
continue;
hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_DMA, dirent->d_name);
}
closedir(dir);
return 0;
}
struct hwloc_firmware_dmi_mem_device_header {
unsigned char type;
unsigned char length;
unsigned char handle[2];
unsigned char phy_mem_handle[2];
unsigned char mem_err_handle[2];
unsigned char tot_width[2];
unsigned char dat_width[2];
unsigned char size[2];
unsigned char ff;
unsigned char dev_set;
unsigned char dev_loc_str_num;
unsigned char bank_loc_str_num;
unsigned char mem_type;
unsigned char type_detail[2];
unsigned char speed[2];
unsigned char manuf_str_num;
unsigned char serial_str_num;
unsigned char asset_tag_str_num;
unsigned char part_num_str_num;
};
static int check_dmi_entry(const char *buffer)
{
if (!*buffer)
return 0;
if (strspn(buffer, " ") == strlen(buffer))
return 0;
return 1;
}
static int
hwloc__get_firmware_dmi_memory_info_one(struct hwloc_topology *topology,
unsigned idx, const char *path, FILE *fd,
struct hwloc_firmware_dmi_mem_device_header *header)
{
unsigned slen;
char buffer[256]; 
unsigned foff; 
unsigned boff; 
unsigned i;
struct hwloc_info_s *infos = NULL;
unsigned infos_count = 0;
hwloc_obj_t misc;
int foundinfo = 0;
foff = header->length;
i = 1;
while (1) {
if (fseek(fd, foff, SEEK_SET) < 0)
break;
if (!fgets(buffer, sizeof(buffer), fd))
break;
boff = 0;
while (1) {
if (!buffer[boff])
goto done;
slen = strlen(buffer+boff);
if (boff + slen+1 == sizeof(buffer))
break;
if (i == header->manuf_str_num) {
if (check_dmi_entry(buffer+boff)) {
hwloc__add_info(&infos, &infos_count, "Vendor", buffer+boff);
foundinfo = 1;
}
}	else if (i == header->serial_str_num) {
if (check_dmi_entry(buffer+boff)) {
hwloc__add_info(&infos, &infos_count, "SerialNumber", buffer+boff);
foundinfo = 1;
}
} else if (i == header->asset_tag_str_num) {
if (check_dmi_entry(buffer+boff)) {
hwloc__add_info(&infos, &infos_count, "AssetTag", buffer+boff);
foundinfo = 1;
}
} else if (i == header->part_num_str_num) {
if (check_dmi_entry(buffer+boff)) {
hwloc__add_info(&infos, &infos_count, "PartNumber", buffer+boff);
foundinfo = 1;
}
} else if (i == header->dev_loc_str_num) {
if (check_dmi_entry(buffer+boff)) {
hwloc__add_info(&infos, &infos_count, "DeviceLocation", buffer+boff);
}
} else if (i == header->bank_loc_str_num) {
if (check_dmi_entry(buffer+boff)) {
hwloc__add_info(&infos, &infos_count, "BankLocation", buffer+boff);
}
} else {
goto done;
}
boff += slen+1;
i++;
}
if (!boff) {
fprintf(stderr, "hwloc could read a DMI firmware entry #%u in %s\n",
i, path);
break;
}
foff += boff;
}
done:
if (!foundinfo) {
goto out_with_infos;
}
misc = hwloc_alloc_setup_object(topology, HWLOC_OBJ_MISC, idx);
if (!misc)
goto out_with_infos;
misc->subtype = strdup("MemoryModule");
hwloc__move_infos(&misc->infos, &misc->infos_count, &infos, &infos_count);
hwloc_insert_object_by_parent(topology, hwloc_get_root_obj(topology), misc);
return 1;
out_with_infos:
hwloc__free_infos(infos, infos_count);
return 0;
}
static int
hwloc__get_firmware_dmi_memory_info(struct hwloc_topology *topology,
struct hwloc_linux_backend_data_s *data)
{
char path[128];
unsigned i;
for(i=0; ; i++) {
FILE *fd;
struct hwloc_firmware_dmi_mem_device_header header;
int err;
snprintf(path, sizeof(path), "/sys/firmware/dmi/entries/17-%u/raw", i);
fd = hwloc_fopen(path, "r", data->root_fd);
if (!fd)
break;
err = fread(&header, sizeof(header), 1, fd);
if (err != 1) {
fclose(fd);
break;
}
if (header.length < sizeof(header)) {
fclose(fd);
break;
}
hwloc__get_firmware_dmi_memory_info_one(topology, i, path, fd, &header);
fclose(fd);
}
return 0;
}
#ifdef HWLOC_HAVE_LINUXPCI
#define HWLOC_PCI_REVISION_ID 0x08
#define HWLOC_PCI_CAP_ID_EXP 0x10
#define HWLOC_PCI_CLASS_NOT_DEFINED 0x0000
static int
hwloc_linuxfs_pci_look_pcidevices(struct hwloc_backend *backend)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
struct hwloc_topology *topology = backend->topology;
hwloc_obj_t tree = NULL;
int root_fd = data->root_fd;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/bus/pci/devices/", root_fd);
if (!dir)
return 0;
while ((dirent = readdir(dir)) != NULL) {
#define CONFIG_SPACE_CACHESIZE 256
unsigned char config_space_cache[CONFIG_SPACE_CACHESIZE];
unsigned domain, bus, dev, func;
unsigned secondary_bus, subordinate_bus;
unsigned short class_id;
hwloc_obj_type_t type;
hwloc_obj_t obj;
struct hwloc_pcidev_attr_s *attr;
unsigned offset;
char path[64];
char value[16];
size_t ret;
int fd, err;
if (sscanf(dirent->d_name, "%04x:%02x:%02x.%01x", &domain, &bus, &dev, &func) != 4)
continue;
if (domain > 0xffff) {
static int warned = 0;
if (!warned)
fprintf(stderr, "Ignoring PCI device with non-16bit domain\n");
warned = 1;
continue;
}
memset(config_space_cache, 0xff, CONFIG_SPACE_CACHESIZE);
err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/config", dirent->d_name);
if ((size_t) err < sizeof(path)) {
fd = hwloc_open(path, root_fd);
if (fd >= 0) {
ret = read(fd, config_space_cache, CONFIG_SPACE_CACHESIZE);
(void) ret; 
close(fd);
}
}
class_id = HWLOC_PCI_CLASS_NOT_DEFINED;
err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/class", dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
class_id = strtoul(value, NULL, 16) >> 8;
type = hwloc_pcidisc_check_bridge_type(class_id, config_space_cache);
if (type == HWLOC_OBJ_BRIDGE) {
if (hwloc_pcidisc_find_bridge_buses(domain, bus, dev, func,
&secondary_bus, &subordinate_bus,
config_space_cache) < 0)
continue;
}
if (type == HWLOC_OBJ_PCI_DEVICE) {
enum hwloc_type_filter_e filter;
hwloc_topology_get_type_filter(topology, HWLOC_OBJ_PCI_DEVICE, &filter);
if (filter == HWLOC_TYPE_FILTER_KEEP_NONE)
continue;
if (filter == HWLOC_TYPE_FILTER_KEEP_IMPORTANT
&& !hwloc_filter_check_pcidev_subtype_important(class_id))
continue;
} else if (type == HWLOC_OBJ_BRIDGE) {
enum hwloc_type_filter_e filter;
hwloc_topology_get_type_filter(topology, HWLOC_OBJ_BRIDGE, &filter);
if (filter == HWLOC_TYPE_FILTER_KEEP_NONE)
continue;
}
obj = hwloc_alloc_setup_object(topology, type, HWLOC_UNKNOWN_INDEX);
if (!obj)
break;
attr = &obj->attr->pcidev;
attr->domain = domain;
attr->bus = bus;
attr->dev = dev;
attr->func = func;
if (type == HWLOC_OBJ_BRIDGE) {
struct hwloc_bridge_attr_s *battr = &obj->attr->bridge;
battr->upstream_type = HWLOC_OBJ_BRIDGE_PCI;
battr->downstream_type = HWLOC_OBJ_BRIDGE_PCI;
battr->downstream.pci.domain = domain;
battr->downstream.pci.secondary_bus = secondary_bus;
battr->downstream.pci.subordinate_bus = subordinate_bus;
}
attr->vendor_id = 0;
attr->device_id = 0;
attr->class_id = class_id;
attr->revision = 0;
attr->subvendor_id = 0;
attr->subdevice_id = 0;
attr->linkspeed = 0;
err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
attr->vendor_id = strtoul(value, NULL, 16);
err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
attr->device_id = strtoul(value, NULL, 16);
err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/subsystem_vendor", dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
attr->subvendor_id = strtoul(value, NULL, 16);
err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/subsystem_device", dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
attr->subdevice_id = strtoul(value, NULL, 16);
attr->revision = config_space_cache[HWLOC_PCI_REVISION_ID];
offset = hwloc_pcidisc_find_cap(config_space_cache, HWLOC_PCI_CAP_ID_EXP);
if (offset > 0 && offset + 20  <= CONFIG_SPACE_CACHESIZE) {
hwloc_pcidisc_find_linkspeed(config_space_cache, offset, &attr->linkspeed);
} else {
float speed = 0.f;
unsigned width = 0;
err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/current_link_speed", dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
speed = hwloc_linux_pci_link_speed_from_string(value);
err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/current_link_width", dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
width = atoi(value);
attr->linkspeed = speed*width/8;
}
hwloc_pcidisc_tree_insert_by_busid(&tree, obj);
}
closedir(dir);
hwloc_pcidisc_tree_attach(backend->topology, tree);
return 0;
}
static int
hwloc_linuxfs_pci_look_pcislots(struct hwloc_backend *backend)
{
struct hwloc_topology *topology = backend->topology;
struct hwloc_linux_backend_data_s *data = backend->private_data;
int root_fd = data->root_fd;
DIR *dir;
struct dirent *dirent;
dir = hwloc_opendir("/sys/bus/pci/slots/", root_fd);
if (dir) {
while ((dirent = readdir(dir)) != NULL) {
char path[64];
char buf[64];
unsigned domain, bus, dev;
int err;
if (dirent->d_name[0] == '.')
continue;
err = snprintf(path, sizeof(path), "/sys/bus/pci/slots/%s/address", dirent->d_name);
if ((size_t) err < sizeof(path)
&& !hwloc_read_path_by_length(path, buf, sizeof(buf), root_fd)
&& sscanf(buf, "%x:%x:%x", &domain, &bus, &dev) == 3) {
hwloc_obj_t obj = hwloc_pci_find_by_busid(topology, domain, bus, dev, 0);
while (obj) {
if (obj->type != HWLOC_OBJ_PCI_DEVICE &&
(obj->type != HWLOC_OBJ_BRIDGE || obj->attr->bridge.upstream_type != HWLOC_OBJ_BRIDGE_PCI))
break;
if (obj->attr->pcidev.domain != domain
|| obj->attr->pcidev.bus != bus
|| obj->attr->pcidev.dev != dev)
break;
hwloc_obj_add_info(obj, "PCISlot", dirent->d_name);
obj = obj->next_sibling;
}
}
}
closedir(dir);
}
return 0;
}
#endif 
#endif 
static int
hwloc_look_linuxfs(struct hwloc_backend *backend, struct hwloc_disc_status *dstatus)
{
struct hwloc_topology *topology = backend->topology;
#ifdef HWLOC_HAVE_LINUXIO
enum hwloc_type_filter_e pfilter, bfilter, ofilter, mfilter;
#endif 
if (dstatus->phase == HWLOC_DISC_PHASE_CPU) {
hwloc_linuxfs_look_cpu(backend, dstatus);
return 0;
}
#ifdef HWLOC_HAVE_LINUXIO
hwloc_topology_get_type_filter(topology, HWLOC_OBJ_PCI_DEVICE, &pfilter);
hwloc_topology_get_type_filter(topology, HWLOC_OBJ_BRIDGE, &bfilter);
hwloc_topology_get_type_filter(topology, HWLOC_OBJ_OS_DEVICE, &ofilter);
hwloc_topology_get_type_filter(topology, HWLOC_OBJ_MISC, &mfilter);
if (dstatus->phase == HWLOC_DISC_PHASE_PCI
&& (bfilter != HWLOC_TYPE_FILTER_KEEP_NONE
|| pfilter != HWLOC_TYPE_FILTER_KEEP_NONE)) {
#ifdef HWLOC_HAVE_LINUXPCI
hwloc_linuxfs_pci_look_pcidevices(backend);
dstatus->excluded_phases |= HWLOC_DISC_PHASE_PCI;
#endif 
}
if (dstatus->phase == HWLOC_DISC_PHASE_ANNOTATE
&& (bfilter != HWLOC_TYPE_FILTER_KEEP_NONE
|| pfilter != HWLOC_TYPE_FILTER_KEEP_NONE)) {
#ifdef HWLOC_HAVE_LINUXPCI
hwloc_linuxfs_pci_look_pcislots(backend);
#endif 
}
if (dstatus->phase == HWLOC_DISC_PHASE_IO
&& ofilter != HWLOC_TYPE_FILTER_KEEP_NONE) {
unsigned osdev_flags = 0;
if (getenv("HWLOC_VIRTUAL_LINUX_OSDEV"))
osdev_flags |= HWLOC_LINUXFS_OSDEV_FLAG_FIND_VIRTUAL;
if (ofilter == HWLOC_TYPE_FILTER_KEEP_ALL)
osdev_flags |= HWLOC_LINUXFS_OSDEV_FLAG_FIND_USB;
hwloc_linuxfs_lookup_block_class(backend, osdev_flags);
hwloc_linuxfs_lookup_dax_class(backend, osdev_flags);
hwloc_linuxfs_lookup_net_class(backend, osdev_flags);
hwloc_linuxfs_lookup_infiniband_class(backend, osdev_flags);
hwloc_linuxfs_lookup_mic_class(backend, osdev_flags);
if (ofilter != HWLOC_TYPE_FILTER_KEEP_IMPORTANT) {
hwloc_linuxfs_lookup_drm_class(backend, osdev_flags);
hwloc_linuxfs_lookup_dma_class(backend, osdev_flags);
}
}
if (dstatus->phase == HWLOC_DISC_PHASE_MISC
&& mfilter != HWLOC_TYPE_FILTER_KEEP_NONE) {
hwloc__get_firmware_dmi_memory_info(topology, backend->private_data);
}
#endif 
return 0;
}
static void
hwloc_linux_backend_disable(struct hwloc_backend *backend)
{
struct hwloc_linux_backend_data_s *data = backend->private_data;
#ifdef HAVE_OPENAT
if (data->root_fd >= 0) {
free(data->root_path);
close(data->root_fd);
}
#endif
#ifdef HWLOC_HAVE_LIBUDEV
if (data->udev)
udev_unref(data->udev);
#endif
free(data);
}
static struct hwloc_backend *
hwloc_linux_component_instantiate(struct hwloc_topology *topology,
struct hwloc_disc_component *component,
unsigned excluded_phases __hwloc_attribute_unused,
const void *_data1 __hwloc_attribute_unused,
const void *_data2 __hwloc_attribute_unused,
const void *_data3 __hwloc_attribute_unused)
{
struct hwloc_backend *backend;
struct hwloc_linux_backend_data_s *data;
const char * fsroot_path;
int root = -1;
char *env;
backend = hwloc_backend_alloc(topology, component);
if (!backend)
goto out;
data = malloc(sizeof(*data));
if (!data) {
errno = ENOMEM;
goto out_with_backend;
}
backend->private_data = data;
backend->discover = hwloc_look_linuxfs;
backend->get_pci_busid_cpuset = hwloc_linux_backend_get_pci_busid_cpuset;
backend->disable = hwloc_linux_backend_disable;
data->arch = HWLOC_LINUX_ARCH_UNKNOWN;
data->is_knl = 0;
data->is_amd_with_CU = 0;
data->use_dt = 0;
data->is_real_fsroot = 1;
data->root_path = NULL;
fsroot_path = getenv("HWLOC_FSROOT");
if (!fsroot_path)
fsroot_path = "/";
if (strcmp(fsroot_path, "/")) {
#ifdef HAVE_OPENAT
int flags;
root = open(fsroot_path, O_RDONLY | O_DIRECTORY);
if (root < 0)
goto out_with_data;
backend->is_thissystem = 0;
data->is_real_fsroot = 0;
data->root_path = strdup(fsroot_path);
flags = fcntl(root, F_GETFD, 0);
if (-1 == flags ||
-1 == fcntl(root, F_SETFD, FD_CLOEXEC | flags)) {
close(root);
root = -1;
goto out_with_data;
}
#else
fprintf(stderr, "Cannot change Linux fsroot without openat() support.\n");
errno = ENOSYS;
goto out_with_data;
#endif
}
data->root_fd = root;
#ifdef HWLOC_HAVE_LIBUDEV
data->udev = NULL;
if (data->is_real_fsroot) {
data->udev = udev_new();
}
#endif
data->dumped_hwdata_dirname = getenv("HWLOC_DUMPED_HWDATA_DIR");
if (!data->dumped_hwdata_dirname)
data->dumped_hwdata_dirname = (char *) RUNSTATEDIR "/hwloc/";
data->use_numa_distances = 1;
data->use_numa_distances_for_cpuless = 1;
data->use_numa_initiators = 1;
env = getenv("HWLOC_USE_NUMA_DISTANCES");
if (env) {
unsigned val = atoi(env);
data->use_numa_distances = !!(val & 3); 
data->use_numa_distances_for_cpuless = !!(val & 2);
data->use_numa_initiators = !!(val & 4);
}
env = getenv("HWLOC_USE_DT");
if (env)
data->use_dt = atoi(env);
return backend;
out_with_data:
#ifdef HAVE_OPENAT
free(data->root_path);
#endif
free(data);
out_with_backend:
free(backend);
out:
return NULL;
}
static struct hwloc_disc_component hwloc_linux_disc_component = {
"linux",
HWLOC_DISC_PHASE_CPU | HWLOC_DISC_PHASE_PCI | HWLOC_DISC_PHASE_IO | HWLOC_DISC_PHASE_MISC | HWLOC_DISC_PHASE_ANNOTATE,
HWLOC_DISC_PHASE_GLOBAL,
hwloc_linux_component_instantiate,
50,
1,
NULL
};
const struct hwloc_component hwloc_linux_component = {
HWLOC_COMPONENT_ABI,
NULL, NULL,
HWLOC_COMPONENT_TYPE_DISC,
0,
&hwloc_linux_disc_component
};
