#include <stdio.h>
#define NEED_OS_FORK
#include "os.h"
#if (!AC_BUILT || HAVE_UNISTD_H) && !_MSC_VER
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#elif _MSC_VER
#include <io.h>
#pragma warning ( disable : 4996 )
#endif
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <assert.h>
#include "memory.h"
#include "logger.h"
#include "params.h"
#include "misc.h"
#include "options.h"
#include "john_mpi.h"
#include "memory.h"
void real_error(const char *file, int line)
{
#ifndef _JOHN_MISC_NO_LOG
log_event("Terminating on error, %s:%d", file, line);
log_done();
#else
fprintf(stderr, "Terminating on error, %s:%d\n", file, line);
#endif
exit(1);
}
void real_error_msg(const char *file, int line,  const char *format, ...)
{
va_list args;
#if defined(HAVE_MPI) && !defined(_JOHN_MISC_NO_LOG)
if (mpi_p > 1)
fprintf(stderr, "%u@%s: ", mpi_id + 1, mpi_name);
else
#elif OS_FORK && !defined(_JOHN_MISC_NO_LOG)
if (options.fork)
fprintf(stderr, "%u: ", options.node_min);
#endif
va_start(args, format);
vfprintf(stderr, format, args);
va_end(args);
real_error(file, line);
}
void real_pexit(const char *file, int line, const char *format, ...)
{
va_list args;
#if !defined(_JOHN_MISC_NO_LOG)
#if HAVE_MPI
if (mpi_p > 1)
fprintf(stderr, "%u@%s: ", mpi_id + 1, mpi_name);
#endif
#if HAVE_MPI && OS_FORK
else
#endif
#if OS_FORK
if (options.fork)
fprintf(stderr, "%u: ", options.node_min);
#endif
#endif
va_start(args, format);
vfprintf(stderr, format, args);
va_end(args);
fprintf(stderr, ": %s\n", strerror(errno));
real_error(file, line);
}
int write_loop(int fd, const char *buffer, int count)
{
int offset, block;
offset = 0;
while (count > 0) {
block = write(fd, &buffer[offset], count);
if (block < 0) {
if (errno == EINTR) continue;
return block;
}
offset += block;
count -= block;
}
return offset;
}
char *fgetl(char *s, int size, FILE *stream)
{
char *res, *pos;
int c;
if ((res = fgets(s, size, stream))) {
if (!*res) return res;
pos = res + strlen(res) - 1;
if (*pos == '\n') {
*pos = 0;
if (pos > res)
if (*--pos == '\r') *pos = 0;
return res;
}
if ((pos-res) + 1 < size) {
while ((++pos - res) + 1 < size)
if (*pos == '\n')
return res; 
}
if ((c = getc(stream)) == '\n') {
if (*pos == '\r') *pos = 0;
} else
while (c != EOF && c != '\n')
c = getc(stream);
}
return res;
}
char *fgetll(char *s, size_t size, FILE *stream)
{
size_t len;
int c;
char *cp;
assert(size <= INT32_MAX);
if (!fgets(s, size, stream))
return NULL;
len = strlen(s);
if (!len)
return s;
if (s[len-1] == '\n') {
s[--len] = 0;
while (len && (s[len-1] == '\n' || s[len-1] == '\r'))
s[--len] = 0;
return s;
}
else if (s[len-1] == '\r') {
s[--len] = 0;
while (len && (s[len-1] == '\n' || s[len-1] == '\r'))
s[--len] = 0;
c = getc(stream);
if (c == EOF)
return s;
if (c != '\n')
ungetc(c, stream);
return s;
}
else if ((len + 1) < size) { 
while (++len < size)
if (s[len] == '\n')
return s; 
do {
c = getc(stream);
} while (c != EOF && c != '\n');
return s;
}
cp = xstrdup(s);
while (1) {
int increase = MIN((((len >> 12) + 1) << 12), 0x40000000);
size_t chunk_len;
void *new_cp;
new_cp = realloc(cp, len + increase);
while (!new_cp) {
increase >>= 2;
if (increase < 0x10000)
pexit("realloc");
new_cp = realloc(cp, len + increase);
if (new_cp)
cp = new_cp;
}
cp = new_cp;
if (!fgets(&cp[len], increase, stream))
return cp;
chunk_len = strlen(&cp[len]);
len += chunk_len;
if (cp[len-1] == '\n') {
cp[--len] = 0;
while (len && (cp[len-1] == '\n' || cp[len-1] == '\r'))
cp[--len] = 0;
return cp;
}
else if (cp[len-1] == '\r') {
cp[--len] = 0;
while (len && (cp[len-1] == '\n' || cp[len-1] == '\r'))
cp[--len] = 0;
c = getc(stream);
if (c == EOF)
return cp;
if (c != '\n')
ungetc(c, stream);
return cp;
}
else if ((chunk_len + 1) < increase) { 
while (++chunk_len < increase)
if (cp[++len] == '\n')
return cp; 
do {
c = getc(stream);
} while (c != EOF && c != '\n');
return cp;
}
}
}
void *strncpy_pad(void *dst, const void *src, size_t size, uint8_t pad)
{
uint8_t *d = dst;
const uint8_t *s = src;
if (!size || size > (((size_t)-1) >> 1))
return dst;
while (*s && size) {
*d++ = *s++;
--size;
}
while (size--)
*d++ = pad;
return dst;
}
char *strnfcpy(char *dst, const char *src, int size)
{
char *dptr;
if (size < 1)
return dst;
dptr = dst;
while (size--)
if (!(*dptr++ = *src++))
break;
return dst;
}
char *strnzcpy(char *dst, const char *src, int size)
{
char *dptr;
if (size < 1)
return dst;
dptr = dst;
while (--size)
if (!(*dptr++ = *src++))
return dst;
*dptr = 0;
return dst;
}
char *strnzcpylwr(char *dst, const char *src, int size)
{
char *dptr;
if (size < 1)
return dst;
dptr = dst;
while (--size) {
if (*src >= 'A' && *src <= 'Z') {
*dptr = *src | 0x20;
} else {
*dptr = *src;
if (!*src)
return dst;
}
dptr++;
src++;
}
*dptr = 0;
return dst;
}
int strnzcpyn(char *dst, const char *src, int size)
{
char *dptr;
if (size < 1)
return 0;
dptr = dst;
while (--size)
if (!(*dptr++ = *src++))
return (dptr-dst)-1;
*dptr = 0;
return (dptr-dst);
}
int strnzcpylwrn(char *dst, const char *src, int size)
{
char *dptr;
if (size < 1)
return 0;
dptr = dst;
while (--size) {
if (*src >= 'A' && *src <= 'Z') {
*dptr = *src | 0x20;
} else {
*dptr = *src;
if (!*src)
return (dptr-dst);
}
dptr++;
src++;
}
*dptr = 0;
return (dptr-dst);
}
char *strnzcat(char *dst, const char *src, int size)
{
char *dptr;
if (size < 1)
return dst;
dptr = dst;
while (size && *dptr) {
size--; dptr++;
}
if (size)
while (--size)
if (!(*dptr++ = *src++))
break;
*dptr = 0;
return dst;
}
char *strnzcatn(char *dst, int size, const char *src, int src_max)
{
char *dptr;
if (size < 1)
return dst;
dptr = dst;
while (size && *dptr)
size--, dptr++;
if (size)
while (--size && src_max--)
if (!(*dptr++ = *src++))
break;
*dptr = 0;
return dst;
}
char *strtokm(char *s1, const char *delims)
{
static char *last = NULL;
char *endp;
if (!s1)
s1 = last;
if (!s1 || *s1 == 0)
return last = NULL;
endp = strpbrk(s1, delims);
if (endp) {
*endp = '\0';
last = endp + 1;
} else
last = NULL;
return s1;
}
unsigned int atou(const char *src)
{
unsigned val;
sscanf(src, "%u", &val);
return val;
}
MAYBE_INLINE const char *_lltoa(int64_t num, char *ret, int ret_sz, int base)
{
char *p = ret, *p1 = ret;
int64_t t;
const char bc[] = "zyxwvutsrqponmlkjihgfedcba987654321"
"0"
"123456789abcdefghijklmnopqrstuvwxyz";
if (--ret_sz < 1)	
return "";	
*ret = 0;
if (base < 2 || base > 36) return ret;
if (num < 0 && --ret_sz < 1)
return ret;
do {
t = num;
num /= base;
*p++ = bc[35 + (t - num * base)];
if (num && p - ret == ret_sz)
break; 
} while (num);
if (t < 0) *p++ = '-'; 
*p-- = 0;
for (; p > p1; ++p1, --p) {
*p1 ^= *p; *p ^= *p1; *p1 ^= *p;
}
return ret;
}
MAYBE_INLINE const char *_ulltoa(uint64_t num, char *ret, int ret_sz, int base)
{
char *p = ret, *p1 = ret;
uint64_t t;
const char bc[] = "0123456789abcdefghijklmnopqrstuvwxyz";
if (--ret_sz < 1)
return "";
*ret = 0;
if (base < 2 || base > 36) return ret;
do {
t = num;
num /= base;
*p++ = bc[t - num * base];
if (num && p - ret == ret_sz)
break; 
} while (num);
*p-- = 0;
for (; p > p1; ++p1, --p) {
*p1 ^= *p; *p ^= *p1; *p1 ^= *p;
}
return ret;
}
const char *jtr_itoa(int val, char *result, int rlen, int base)
{
return _lltoa(val, result, rlen, base);
}
const char *jtr_utoa(unsigned int val, char *result, int rlen, int base)
{
return _ulltoa((uint64_t)val, result, rlen, base);
}
const char *jtr_lltoa(int64_t val, char *result, int rlen, int base)
{
return _lltoa(val, result, rlen, base);
}
const char *jtr_ulltoa(uint64_t val, char *result, int rlen, int base)
{
return _ulltoa(val, result, rlen, base);
}
char *human_prefix(uint64_t num)
{
char *out = mem_alloc_tiny(16, MEM_ALIGN_NONE);
char prefixes[] = "\0KMGTPEZY";
char *p = prefixes;
while (p[1] && (num >= (100 << 10) || (!(num & 1023) && num >= (1 << 10)))) {
num >>= 10;
p++;
}
if (*p)
snprintf(out, 16, "%u %ci", (uint32_t)num, *p);
else
snprintf(out, 16, "%u ", (uint32_t)num);
return out;
}
char *human_speed(uint64_t speed)
{
char *out = mem_alloc_tiny(16, MEM_ALIGN_NONE);
char prefixes[] = "\0KMGTPEZY";
char *p = prefixes;
while (p[1] && speed >= 1000000) {
speed /= 1000;
p++;
}
if (*p)
snprintf(out, 16, "%u%c c/s", (uint32_t)speed, *p);
else
snprintf(out, 16, "%u c/s", (uint32_t)speed);
return out;
}
char *human_prefix_small(double num)
{
char *out = mem_alloc_tiny(16, MEM_ALIGN_NONE);
uint64_t number = num * 1E9;
int whole, milli, micro, nano;
nano = number % 1000;
number /= 1000;
micro = number % 1000;
number /= 1000;
milli = number % 1000;
whole = number / 1000;
if (whole) {
if (milli)
snprintf(out, 16, "%d.%03d ", whole, milli);
else
snprintf(out, 16, "%d ", whole);
} else if (milli) {
if (micro)
snprintf(out, 16, "%d.%03d m", milli, micro);
else
snprintf(out, 16, "%d m", milli);
} else if (micro) {
if (nano)
snprintf(out, 16, "%d.%03d u", micro, nano);
else
snprintf(out, 16, "%d u", micro);
} else
snprintf(out, 16, "%d n", nano);
return out;
}
unsigned int lcm(unsigned int x, unsigned int y)
{
unsigned int tmp, a, b;
a = MAX(x, y);
b = MIN(x, y);
while (b) {
tmp = b;
b = a % b;
a = tmp;
}
return x / a * y;
}
char *ltrim(char *str)
{
char *out = str;
while (*out == ' ' || *out == '\t')
out++;
return out;
}
char *rtrim(char *str)
{
char *out = str + strlen(str) - 1;
while (out >= str && (*out == ' ' || *out == '\t'))
out--;
*(out+1) = '\0';
return str;
}
#ifndef _JOHN_MISC_NO_LOG
int64_t host_total_mem(void)
{
int64_t tot_mem = -1;
#if (!AC_BUILT || HAVE_UNISTD_H) && !_MSC_VER && defined _SC_PAGESIZE && defined _SC_PHYS_PAGES
long pagesize = sysconf(_SC_PAGESIZE);
if (pagesize < 0)
return -1;
long totmem = sysconf(_SC_PHYS_PAGES);
if (totmem < 0)
return -1;
tot_mem = (int64_t)totmem * pagesize;
#endif
return tot_mem;
}
int64_t host_avail_mem(void)
{
int64_t avail_mem = -1;
#if __linux__
FILE *fp;
char buf[LINE_BUFFER_SIZE];
if ((fp = fopen("/proc/meminfo", "r"))) {
while (fgets(buf, LINE_BUFFER_SIZE, fp)) {
if (strstr(buf, "MemAvailable")) {
char *p = strchr(buf, ':');
if (p++)
avail_mem = strtoull(p, NULL, 10) << 10;
break;
}
}
if (avail_mem < 0) {
fseek(fp, 0, SEEK_SET);
while (fgets(buf, LINE_BUFFER_SIZE, fp)) {
if (strstr(buf, "MemFree")) {
char *p = strchr(buf, ':');
if (p++)
avail_mem = strtoull(p, NULL, 10) << 10;
continue;
}
if (strstr(buf, "Buffers")) {
char *p = strchr(buf, ':');
if (p++)
avail_mem += strtoull(p, NULL, 10) << 10;
continue;
}
if (strstr(buf, "Cached")) {
char *p = strchr(buf, ':');
if (p++)
avail_mem += strtoull(p, NULL, 10) << 10;
break;
}
}
}
fclose(fp);
}
#elif __APPLE__
vm_statistics64_data_t vm_stat;
unsigned int count = HOST_VM_INFO64_COUNT;
kern_return_t ret;
mach_port_t myHost = mach_host_self();
vm_size_t pageSize;
if (host_page_size(mach_host_self(), &pageSize) != KERN_SUCCESS)
return -1;
if ((ret = host_statistics64(myHost, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) != KERN_SUCCESS))
return -1;
avail_mem = (vm_stat.free_count + vm_stat.inactive_count + vm_stat.purgeable_count) * pageSize;
#elif (!AC_BUILT || HAVE_UNISTD_H) && !_MSC_VER && defined _SC_PAGESIZE && defined _SC_AVPHYS_PAGES
long pagesize = sysconf(_SC_PAGESIZE);
if (pagesize < 0)
return -1;
long availmem = sysconf(_SC_AVPHYS_PAGES);
if (availmem < 0)
return -1;
avail_mem = (int64_t)availmem * pagesize;
#endif
return avail_mem;
}
int parse_bool(char *string)
{
if (string) {
if (string == OPT_TRISTATE_NO_PARAM || !strcasecmp(string, "y") ||
!strcasecmp(string, "yes") || !strcasecmp(string, "t") ||
!strcasecmp(string, "true") || !strcasecmp(string, "1"))
return 1;
if (string == OPT_TRISTATE_NEGATED || !strcasecmp(string, "n") ||
!strcasecmp(string, "no") || !strcasecmp(string, "f") ||
!strcasecmp(string, "false") || !strcasecmp(string, "0"))
return 0;
}
return -1;
}
#endif
