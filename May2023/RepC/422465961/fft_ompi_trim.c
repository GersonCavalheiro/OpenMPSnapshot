void ort_taskwait(int num){}
void ort_taskenv_free(void *ptr, void *(*task_func)(void *)){}
void ort_leaving_single(){}
void * _ompi_crity;        
void ort_atomic_begin(){}
void ort_atomic_end(){}
struct _noname0_ {
int __val[ 2];
};
struct _IO_FILE;
struct _IO_FILE;
struct _noname1_ {
int __count;
union {
unsigned int __wch;
char __wchb[ 4];
} __value;
};
struct _noname2_ {
long int __pos;
struct _noname1_ __state;
};
struct _noname3_ {
long int __pos;
struct _noname1_ __state;
};
struct _IO_jump_t;
struct _IO_FILE;
struct _IO_marker {
struct _IO_marker * _next;
struct _IO_FILE * _sbuf;
int _pos;
};
enum __codecvt_result {
__codecvt_ok, __codecvt_partial, __codecvt_error, __codecvt_noconv
};
struct _IO_FILE {
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
long int _old_offset;
unsigned short _cur_column;
signed char _vtable_offset;
char _shortbuf[ 1];
void (* _lock);
long int _offset;
void * __pad1;
void * __pad2;
void * __pad3;
void * __pad4;
long unsigned int __pad5;
int _mode;
char _unused2[ 15 * sizeof(int) - 4 * sizeof(void *) - sizeof(long unsigned int )];
};
struct _IO_FILE_plus;
extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
extern int __underflow(struct _IO_FILE (*));
extern int __uflow(struct _IO_FILE (*));
extern int __overflow(struct _IO_FILE (*), int);
extern int _IO_getc(struct _IO_FILE (* __fp));
extern int _IO_putc(int __c, struct _IO_FILE (* __fp));
extern int _IO_feof(struct _IO_FILE (* __fp));
extern int _IO_ferror(struct _IO_FILE (* __fp));
extern int _IO_peekc_locked(struct _IO_FILE (* __fp));
extern void _IO_flockfile(struct _IO_FILE (*));
extern void _IO_funlockfile(struct _IO_FILE (*));
extern int _IO_ftrylockfile(struct _IO_FILE (*));
extern int _IO_vfscanf(struct _IO_FILE (*), const char *, __builtin_va_list , int *);
extern int _IO_vfprintf(struct _IO_FILE (*), const char *, __builtin_va_list );
extern long int _IO_padn(struct _IO_FILE (*), int, long int );
extern long unsigned int _IO_sgetn(struct _IO_FILE (*), void *, long unsigned int );
extern long int _IO_seekoff(struct _IO_FILE (*), long int , int, int);
extern long int _IO_seekpos(struct _IO_FILE (*), long int , int);
extern void _IO_free_backup_area(struct _IO_FILE (*));
extern struct _IO_FILE * stdin;
extern struct _IO_FILE * stdout;
extern struct _IO_FILE * stderr;
extern int remove(const char * __filename);
extern int rename(const char * __old, const char * __new);
extern int renameat(int __oldfd, const char * __old, int __newfd, const char * __new);
extern struct _IO_FILE (* tmpfile(void));
extern char * tmpnam(char * __s);
extern char * tmpnam_r(char * __s);
extern char * tempnam(const char * __dir, const char * __pfx);
extern int fclose(struct _IO_FILE (* __stream));
extern int fflush(struct _IO_FILE (* __stream));
extern int fflush_unlocked(struct _IO_FILE (* __stream));
extern struct _IO_FILE (* fopen(const char * __filename, const char * __modes));
extern struct _IO_FILE (* freopen(const char * __filename, const char * __modes, struct _IO_FILE (* __stream)));
extern struct _IO_FILE (* fdopen(int __fd, const char * __modes));
extern struct _IO_FILE (* fmemopen(void * __s, long unsigned int __len, const char * __modes));
extern struct _IO_FILE (* open_memstream(char ** __bufloc, long unsigned int (* __sizeloc)));
extern void setbuf(struct _IO_FILE (* __stream), char * __buf);
extern int setvbuf(struct _IO_FILE (* __stream), char * __buf, int __modes, long unsigned int __n);
extern void setbuffer(struct _IO_FILE (* __stream), char * __buf, long unsigned int __size);
extern void setlinebuf(struct _IO_FILE (* __stream));
extern int fprintf(struct _IO_FILE (* __stream), const char * __format, ...);
extern int printf(const char * __format, ...);
extern int sprintf(char * __s, const char * __format, ...);
extern int vfprintf(struct _IO_FILE (* __s), const char * __format, __builtin_va_list __arg);
extern int vprintf(const char * __format, __builtin_va_list __arg);
extern int vsprintf(char * __s, const char * __format, __builtin_va_list __arg);
extern int snprintf(char * __s, long unsigned int __maxlen, const char * __format, ...);
extern int vsnprintf(char * __s, long unsigned int __maxlen, const char * __format, __builtin_va_list __arg);
extern int vdprintf(int __fd, const char * __fmt, __builtin_va_list __arg);
extern int dprintf(int __fd, const char * __fmt, ...);
extern int fscanf(struct _IO_FILE (* __stream), const char * __format, ...);
extern int scanf(const char * __format, ...);
extern int sscanf(const char * __s, const char * __format, ...);
extern int __isoc99_fscanf(struct _IO_FILE (* __stream), const char * __format, ...);
extern int __isoc99_scanf(const char * __format, ...);
extern int __isoc99_sscanf(const char * __s, const char * __format, ...);
extern int vfscanf(struct _IO_FILE (* __s), const char * __format, __builtin_va_list __arg);
extern int vscanf(const char * __format, __builtin_va_list __arg);
extern int vsscanf(const char * __s, const char * __format, __builtin_va_list __arg);
extern int __isoc99_vfscanf(struct _IO_FILE (* __s), const char * __format, __builtin_va_list __arg);
extern int __isoc99_vscanf(const char * __format, __builtin_va_list __arg);
extern int __isoc99_vsscanf(const char * __s, const char * __format, __builtin_va_list __arg);
extern int fgetc(struct _IO_FILE (* __stream));
extern int getc(struct _IO_FILE (* __stream));
extern int getchar(void);
extern int getc_unlocked(struct _IO_FILE (* __stream));
extern int getchar_unlocked(void);
extern int fgetc_unlocked(struct _IO_FILE (* __stream));
extern int fputc(int __c, struct _IO_FILE (* __stream));
extern int putc(int __c, struct _IO_FILE (* __stream));
extern int putchar(int __c);
extern int fputc_unlocked(int __c, struct _IO_FILE (* __stream));
extern int putc_unlocked(int __c, struct _IO_FILE (* __stream));
extern int putchar_unlocked(int __c);
extern int getw(struct _IO_FILE (* __stream));
extern int putw(int __w, struct _IO_FILE (* __stream));
extern char * fgets(char * __s, int __n, struct _IO_FILE (* __stream));
extern long int __getdelim(char ** __lineptr, long unsigned int (* __n), int __delimiter, struct _IO_FILE (* __stream));
extern long int getdelim(char ** __lineptr, long unsigned int (* __n), int __delimiter, struct _IO_FILE (* __stream));
extern long int getline(char ** __lineptr, long unsigned int (* __n), struct _IO_FILE (* __stream));
extern int fputs(const char * __s, struct _IO_FILE (* __stream));
extern int puts(const char * __s);
extern int ungetc(int __c, struct _IO_FILE (* __stream));
extern long unsigned int fread(void * __ptr, long unsigned int __size, long unsigned int __n, struct _IO_FILE (* __stream));
extern long unsigned int fwrite(const void * __ptr, long unsigned int __size, long unsigned int __n, struct _IO_FILE (* __s));
extern long unsigned int fread_unlocked(void * __ptr, long unsigned int __size, long unsigned int __n, struct _IO_FILE (* __stream));
extern long unsigned int fwrite_unlocked(const void * __ptr, long unsigned int __size, long unsigned int __n, struct _IO_FILE (* __stream));
extern int fseek(struct _IO_FILE (* __stream), long int __off, int __whence);
extern long int ftell(struct _IO_FILE (* __stream));
extern void rewind(struct _IO_FILE (* __stream));
extern int fseeko(struct _IO_FILE (* __stream), long int __off, int __whence);
extern long int ftello(struct _IO_FILE (* __stream));
extern int fgetpos(struct _IO_FILE (* __stream), struct _noname2_ (* __pos));
extern int fsetpos(struct _IO_FILE (* __stream), const struct _noname2_ (* __pos));
extern void clearerr(struct _IO_FILE (* __stream));
extern int feof(struct _IO_FILE (* __stream));
extern int ferror(struct _IO_FILE (* __stream));
extern void clearerr_unlocked(struct _IO_FILE (* __stream));
extern int feof_unlocked(struct _IO_FILE (* __stream));
extern int ferror_unlocked(struct _IO_FILE (* __stream));
extern void perror(const char * __s);
extern int sys_nerr;
extern const char *const sys_errlist[];
extern int fileno(struct _IO_FILE (* __stream));
extern int fileno_unlocked(struct _IO_FILE (* __stream));
extern struct _IO_FILE (* popen(const char * __command, const char * __modes));
extern int pclose(struct _IO_FILE (* __stream));
extern char * ctermid(char * __s);
extern void flockfile(struct _IO_FILE (* __stream));
extern int ftrylockfile(struct _IO_FILE (* __stream));
extern void funlockfile(struct _IO_FILE (* __stream));
extern int __fpclassify(double __value);
extern int __signbit(double __value);
extern int __isinf(double __value);
extern int __finite(double __value);
extern int __isnan(double __value);
extern int __iseqsig(double __x, double __y);
extern int __issignaling(double __value);
extern double acos(double __x);
extern double __acos(double __x);
extern double asin(double __x);
extern double __asin(double __x);
extern double atan(double __x);
extern double __atan(double __x);
extern double atan2(double __y, double __x);
extern double __atan2(double __y, double __x);
extern double cos(double __x);
extern double __cos(double __x);
extern double sin(double __x);
extern double __sin(double __x);
extern double tan(double __x);
extern double __tan(double __x);
extern double cosh(double __x);
extern double __cosh(double __x);
extern double sinh(double __x);
extern double __sinh(double __x);
extern double tanh(double __x);
extern double __tanh(double __x);
extern double acosh(double __x);
extern double __acosh(double __x);
extern double asinh(double __x);
extern double __asinh(double __x);
extern double atanh(double __x);
extern double __atanh(double __x);
extern double exp(double __x);
extern double __exp(double __x);
extern double frexp(double __x, int * __exponent);
extern double __frexp(double __x, int * __exponent);
extern double ldexp(double __x, int __exponent);
extern double __ldexp(double __x, int __exponent);
extern double log(double __x);
extern double __log(double __x);
extern double log10(double __x);
extern double __log10(double __x);
extern double modf(double __x, double * __iptr);
extern double __modf(double __x, double * __iptr);
extern double expm1(double __x);
extern double __expm1(double __x);
extern double log1p(double __x);
extern double __log1p(double __x);
extern double logb(double __x);
extern double __logb(double __x);
extern double exp2(double __x);
extern double __exp2(double __x);
extern double log2(double __x);
extern double __log2(double __x);
extern double pow(double __x, double __y);
extern double __pow(double __x, double __y);
extern double sqrt(double __x);
extern double __sqrt(double __x);
extern double hypot(double __x, double __y);
extern double __hypot(double __x, double __y);
extern double cbrt(double __x);
extern double __cbrt(double __x);
extern double ceil(double __x);
extern double __ceil(double __x);
extern double fabs(double __x);
extern double __fabs(double __x);
extern double floor(double __x);
extern double __floor(double __x);
extern double fmod(double __x, double __y);
extern double __fmod(double __x, double __y);
extern int isinf(double __value);
extern int finite(double __value);
extern double drem(double __x, double __y);
extern double __drem(double __x, double __y);
extern double significand(double __x);
extern double __significand(double __x);
extern double copysign(double __x, double __y);
extern double __copysign(double __x, double __y);
extern double nan(const char * __tagb);
extern double __nan(const char * __tagb);
extern int isnan(double __value);
extern double j0(double);
extern double __j0(double);
extern double j1(double);
extern double __j1(double);
extern double jn(int, double);
extern double __jn(int, double);
extern double y0(double);
extern double __y0(double);
extern double y1(double);
extern double __y1(double);
extern double yn(int, double);
extern double __yn(int, double);
extern double erf(double);
extern double __erf(double);
extern double erfc(double);
extern double __erfc(double);
extern double lgamma(double);
extern double __lgamma(double);
extern double tgamma(double);
extern double __tgamma(double);
extern double gamma(double);
extern double __gamma(double);
extern double lgamma_r(double, int * __signgamp);
extern double __lgamma_r(double, int * __signgamp);
extern double rint(double __x);
extern double __rint(double __x);
extern double nextafter(double __x, double __y);
extern double __nextafter(double __x, double __y);
extern double nexttoward(double __x, long double __y);
extern double __nexttoward(double __x, long double __y);
extern double remainder(double __x, double __y);
extern double __remainder(double __x, double __y);
extern double scalbn(double __x, int __n);
extern double __scalbn(double __x, int __n);
extern int ilogb(double __x);
extern int __ilogb(double __x);
extern double scalbln(double __x, long int __n);
extern double __scalbln(double __x, long int __n);
extern double nearbyint(double __x);
extern double __nearbyint(double __x);
extern double round(double __x);
extern double __round(double __x);
extern double trunc(double __x);
extern double __trunc(double __x);
extern double remquo(double __x, double __y, int * __quo);
extern double __remquo(double __x, double __y, int * __quo);
extern long int lrint(double __x);
extern long int __lrint(double __x);
extern long long int llrint(double __x);
extern long long int __llrint(double __x);
extern long int lround(double __x);
extern long int __lround(double __x);
extern long long int llround(double __x);
extern long long int __llround(double __x);
extern double fdim(double __x, double __y);
extern double __fdim(double __x, double __y);
extern double fmax(double __x, double __y);
extern double __fmax(double __x, double __y);
extern double fmin(double __x, double __y);
extern double __fmin(double __x, double __y);
extern double fma(double __x, double __y, double __z);
extern double __fma(double __x, double __y, double __z);
extern double scalb(double __x, double __n);
extern double __scalb(double __x, double __n);
extern int __fpclassifyf(float __value);
extern int __signbitf(float __value);
extern int __isinff(float __value);
extern int __finitef(float __value);
extern int __isnanf(float __value);
extern int __iseqsigf(float __x, float __y);
extern int __issignalingf(float __value);
extern float acosf(float __x);
extern float __acosf(float __x);
extern float asinf(float __x);
extern float __asinf(float __x);
extern float atanf(float __x);
extern float __atanf(float __x);
extern float atan2f(float __y, float __x);
extern float __atan2f(float __y, float __x);
extern float cosf(float __x);
extern float __cosf(float __x);
extern float sinf(float __x);
extern float __sinf(float __x);
extern float tanf(float __x);
extern float __tanf(float __x);
extern float coshf(float __x);
extern float __coshf(float __x);
extern float sinhf(float __x);
extern float __sinhf(float __x);
extern float tanhf(float __x);
extern float __tanhf(float __x);
extern float acoshf(float __x);
extern float __acoshf(float __x);
extern float asinhf(float __x);
extern float __asinhf(float __x);
extern float atanhf(float __x);
extern float __atanhf(float __x);
extern float expf(float __x);
extern float __expf(float __x);
extern float frexpf(float __x, int * __exponent);
extern float __frexpf(float __x, int * __exponent);
extern float ldexpf(float __x, int __exponent);
extern float __ldexpf(float __x, int __exponent);
extern float logf(float __x);
extern float __logf(float __x);
extern float log10f(float __x);
extern float __log10f(float __x);
extern float modff(float __x, float * __iptr);
extern float __modff(float __x, float * __iptr);
extern float expm1f(float __x);
extern float __expm1f(float __x);
extern float log1pf(float __x);
extern float __log1pf(float __x);
extern float logbf(float __x);
extern float __logbf(float __x);
extern float exp2f(float __x);
extern float __exp2f(float __x);
extern float log2f(float __x);
extern float __log2f(float __x);
extern float powf(float __x, float __y);
extern float __powf(float __x, float __y);
extern float sqrtf(float __x);
extern float __sqrtf(float __x);
extern float hypotf(float __x, float __y);
extern float __hypotf(float __x, float __y);
extern float cbrtf(float __x);
extern float __cbrtf(float __x);
extern float ceilf(float __x);
extern float __ceilf(float __x);
extern float fabsf(float __x);
extern float __fabsf(float __x);
extern float floorf(float __x);
extern float __floorf(float __x);
extern float fmodf(float __x, float __y);
extern float __fmodf(float __x, float __y);
extern int isinff(float __value);
extern int finitef(float __value);
extern float dremf(float __x, float __y);
extern float __dremf(float __x, float __y);
extern float significandf(float __x);
extern float __significandf(float __x);
extern float copysignf(float __x, float __y);
extern float __copysignf(float __x, float __y);
extern float nanf(const char * __tagb);
extern float __nanf(const char * __tagb);
extern int isnanf(float __value);
extern float j0f(float);
extern float __j0f(float);
extern float j1f(float);
extern float __j1f(float);
extern float jnf(int, float);
extern float __jnf(int, float);
extern float y0f(float);
extern float __y0f(float);
extern float y1f(float);
extern float __y1f(float);
extern float ynf(int, float);
extern float __ynf(int, float);
extern float erff(float);
extern float __erff(float);
extern float erfcf(float);
extern float __erfcf(float);
extern float lgammaf(float);
extern float __lgammaf(float);
extern float tgammaf(float);
extern float __tgammaf(float);
extern float gammaf(float);
extern float __gammaf(float);
extern float lgammaf_r(float, int * __signgamp);
extern float __lgammaf_r(float, int * __signgamp);
extern float rintf(float __x);
extern float __rintf(float __x);
extern float nextafterf(float __x, float __y);
extern float __nextafterf(float __x, float __y);
extern float nexttowardf(float __x, long double __y);
extern float __nexttowardf(float __x, long double __y);
extern float remainderf(float __x, float __y);
extern float __remainderf(float __x, float __y);
extern float scalbnf(float __x, int __n);
extern float __scalbnf(float __x, int __n);
extern int ilogbf(float __x);
extern int __ilogbf(float __x);
extern float scalblnf(float __x, long int __n);
extern float __scalblnf(float __x, long int __n);
extern float nearbyintf(float __x);
extern float __nearbyintf(float __x);
extern float roundf(float __x);
extern float __roundf(float __x);
extern float truncf(float __x);
extern float __truncf(float __x);
extern float remquof(float __x, float __y, int * __quo);
extern float __remquof(float __x, float __y, int * __quo);
extern long int lrintf(float __x);
extern long int __lrintf(float __x);
extern long long int llrintf(float __x);
extern long long int __llrintf(float __x);
extern long int lroundf(float __x);
extern long int __lroundf(float __x);
extern long long int llroundf(float __x);
extern long long int __llroundf(float __x);
extern float fdimf(float __x, float __y);
extern float __fdimf(float __x, float __y);
extern float fmaxf(float __x, float __y);
extern float __fmaxf(float __x, float __y);
extern float fminf(float __x, float __y);
extern float __fminf(float __x, float __y);
extern float fmaf(float __x, float __y, float __z);
extern float __fmaf(float __x, float __y, float __z);
extern float scalbf(float __x, float __n);
extern float __scalbf(float __x, float __n);
extern int __fpclassifyl(long double __value);
extern int __signbitl(long double __value);
extern int __isinfl(long double __value);
extern int __finitel(long double __value);
extern int __isnanl(long double __value);
extern int __iseqsigl(long double __x, long double __y);
extern int __issignalingl(long double __value);
extern long double acosl(long double __x);
extern long double __acosl(long double __x);
extern long double asinl(long double __x);
extern long double __asinl(long double __x);
extern long double atanl(long double __x);
extern long double __atanl(long double __x);
extern long double atan2l(long double __y, long double __x);
extern long double __atan2l(long double __y, long double __x);
extern long double cosl(long double __x);
extern long double __cosl(long double __x);
extern long double sinl(long double __x);
extern long double __sinl(long double __x);
extern long double tanl(long double __x);
extern long double __tanl(long double __x);
extern long double coshl(long double __x);
extern long double __coshl(long double __x);
extern long double sinhl(long double __x);
extern long double __sinhl(long double __x);
extern long double tanhl(long double __x);
extern long double __tanhl(long double __x);
extern long double acoshl(long double __x);
extern long double __acoshl(long double __x);
extern long double asinhl(long double __x);
extern long double __asinhl(long double __x);
extern long double atanhl(long double __x);
extern long double __atanhl(long double __x);
extern long double expl(long double __x);
extern long double __expl(long double __x);
extern long double frexpl(long double __x, int * __exponent);
extern long double __frexpl(long double __x, int * __exponent);
extern long double ldexpl(long double __x, int __exponent);
extern long double __ldexpl(long double __x, int __exponent);
extern long double logl(long double __x);
extern long double __logl(long double __x);
extern long double log10l(long double __x);
extern long double __log10l(long double __x);
extern long double modfl(long double __x, long double * __iptr);
extern long double __modfl(long double __x, long double * __iptr);
extern long double expm1l(long double __x);
extern long double __expm1l(long double __x);
extern long double log1pl(long double __x);
extern long double __log1pl(long double __x);
extern long double logbl(long double __x);
extern long double __logbl(long double __x);
extern long double exp2l(long double __x);
extern long double __exp2l(long double __x);
extern long double log2l(long double __x);
extern long double __log2l(long double __x);
extern long double powl(long double __x, long double __y);
extern long double __powl(long double __x, long double __y);
extern long double sqrtl(long double __x);
extern long double __sqrtl(long double __x);
extern long double hypotl(long double __x, long double __y);
extern long double __hypotl(long double __x, long double __y);
extern long double cbrtl(long double __x);
extern long double __cbrtl(long double __x);
extern long double ceill(long double __x);
extern long double __ceill(long double __x);
extern long double fabsl(long double __x);
extern long double __fabsl(long double __x);
extern long double floorl(long double __x);
extern long double __floorl(long double __x);
extern long double fmodl(long double __x, long double __y);
extern long double __fmodl(long double __x, long double __y);
extern int isinfl(long double __value);
extern int finitel(long double __value);
extern long double dreml(long double __x, long double __y);
extern long double __dreml(long double __x, long double __y);
extern long double significandl(long double __x);
extern long double __significandl(long double __x);
extern long double copysignl(long double __x, long double __y);
extern long double __copysignl(long double __x, long double __y);
extern long double nanl(const char * __tagb);
extern long double __nanl(const char * __tagb);
extern int isnanl(long double __value);
extern long double j0l(long double);
extern long double __j0l(long double);
extern long double j1l(long double);
extern long double __j1l(long double);
extern long double jnl(int, long double);
extern long double __jnl(int, long double);
extern long double y0l(long double);
extern long double __y0l(long double);
extern long double y1l(long double);
extern long double __y1l(long double);
extern long double ynl(int, long double);
extern long double __ynl(int, long double);
extern long double erfl(long double);
extern long double __erfl(long double);
extern long double erfcl(long double);
extern long double __erfcl(long double);
extern long double lgammal(long double);
extern long double __lgammal(long double);
extern long double tgammal(long double);
extern long double __tgammal(long double);
extern long double gammal(long double);
extern long double __gammal(long double);
extern long double lgammal_r(long double, int * __signgamp);
extern long double __lgammal_r(long double, int * __signgamp);
extern long double rintl(long double __x);
extern long double __rintl(long double __x);
extern long double nextafterl(long double __x, long double __y);
extern long double __nextafterl(long double __x, long double __y);
extern long double nexttowardl(long double __x, long double __y);
extern long double __nexttowardl(long double __x, long double __y);
extern long double remainderl(long double __x, long double __y);
extern long double __remainderl(long double __x, long double __y);
extern long double scalbnl(long double __x, int __n);
extern long double __scalbnl(long double __x, int __n);
extern int ilogbl(long double __x);
extern int __ilogbl(long double __x);
extern long double scalblnl(long double __x, long int __n);
extern long double __scalblnl(long double __x, long int __n);
extern long double nearbyintl(long double __x);
extern long double __nearbyintl(long double __x);
extern long double roundl(long double __x);
extern long double __roundl(long double __x);
extern long double truncl(long double __x);
extern long double __truncl(long double __x);
extern long double remquol(long double __x, long double __y, int * __quo);
extern long double __remquol(long double __x, long double __y, int * __quo);
extern long int lrintl(long double __x);
extern long int __lrintl(long double __x);
extern long long int llrintl(long double __x);
extern long long int __llrintl(long double __x);
extern long int lroundl(long double __x);
extern long int __lroundl(long double __x);
extern long long int llroundl(long double __x);
extern long long int __llroundl(long double __x);
extern long double fdiml(long double __x, long double __y);
extern long double __fdiml(long double __x, long double __y);
extern long double fmaxl(long double __x, long double __y);
extern long double __fmaxl(long double __x, long double __y);
extern long double fminl(long double __x, long double __y);
extern long double __fminl(long double __x, long double __y);
extern long double fmal(long double __x, long double __y, long double __z);
extern long double __fmal(long double __x, long double __y, long double __z);
extern long double scalbl(long double __x, long double __n);
extern long double __scalbl(long double __x, long double __n);
extern int signgam;
enum {
FP_NAN = 0, FP_INFINITE = 1, FP_ZERO = 2, FP_SUBNORMAL = 3, FP_NORMAL = 4
};
enum _noname4_ {
P_ALL, P_PID, P_PGID
};
struct _noname5_ {
int quot;
int rem;
};
struct _noname6_ {
long int quot;
long int rem;
};
struct _noname7_ {
long long int quot;
long long int rem;
};
extern long unsigned int __ctype_get_mb_cur_max(void);
extern double atof(const char * __nptr);
extern int atoi(const char * __nptr);
extern long int atol(const char * __nptr);
extern long long int atoll(const char * __nptr);
extern double strtod(const char * __nptr, char ** __endptr);
extern float strtof(const char * __nptr, char ** __endptr);
extern long double strtold(const char * __nptr, char ** __endptr);
extern long int strtol(const char * __nptr, char ** __endptr, int __base);
extern unsigned long int strtoul(const char * __nptr, char ** __endptr, int __base);
extern long long int strtoq(const char * __nptr, char ** __endptr, int __base);
extern unsigned long long int strtouq(const char * __nptr, char ** __endptr, int __base);
extern long long int strtoll(const char * __nptr, char ** __endptr, int __base);
extern unsigned long long int strtoull(const char * __nptr, char ** __endptr, int __base);
extern char * l64a(long int __n);
extern long int a64l(const char * __s);
static unsigned short int __bswap_16(unsigned short int __bsx)
{
return (((unsigned short int) ((((__bsx) >> 8) & 0xff) | (((__bsx) & 0xff) << 8))));
}
static unsigned int __bswap_32(unsigned int __bsx)
{
return (((((__bsx) & 0xff000000) >> 24) | (((__bsx) & 0x00ff0000) >> 8) | (((__bsx) & 0x0000ff00) << 8) | (((__bsx) & 0x000000ff) << 24)));
}
static unsigned long int __bswap_64(unsigned long int __bsx)
{
return (((((__bsx) & 0xff00000000000000ull) >> 56) | (((__bsx) & 0x00ff000000000000ull) >> 40) | (((__bsx) & 0x0000ff0000000000ull) >> 24) | (((__bsx) & 0x000000ff00000000ull) >> 8) | (((__bsx) & 0x00000000ff000000ull) << 8) | (((__bsx) & 0x0000000000ff0000ull) << 24) | (((__bsx) & 0x000000000000ff00ull) << 40) | (((__bsx) & 0x00000000000000ffull) << 56)));
}
static unsigned short int __uint16_identity(unsigned short int __x)
{
return (__x);
}
static unsigned int __uint32_identity(unsigned int __x)
{
return (__x);
}
static unsigned long int __uint64_identity(unsigned long int __x)
{
return (__x);
}
struct _noname8_ {
unsigned long int __val[ (1024 / (8 * sizeof(unsigned long int)))];
};
struct timeval {
long int tv_sec;
long int tv_usec;
};
struct timespec {
long int tv_sec;
long int tv_nsec;
};
struct _noname9_ {
long int (__fds_bits[ 1024 / (8 * (int) sizeof(long int ))]);
};
extern int select(int __nfds, struct _noname9_ (* __readfds), struct _noname9_ (* __writefds), struct _noname9_ (* __exceptfds), struct timeval * __timeout);
extern int pselect(int __nfds, struct _noname9_ (* __readfds), struct _noname9_ (* __writefds), struct _noname9_ (* __exceptfds), const struct timespec * __timeout, const struct _noname8_ (* __sigmask));
extern unsigned int gnu_dev_major(unsigned long int __dev);
extern unsigned int gnu_dev_minor(unsigned long int __dev);
extern unsigned long int gnu_dev_makedev(unsigned int __major, unsigned int __minor);
struct __pthread_rwlock_arch_t {
unsigned int __readers;
unsigned int __writers;
unsigned int __wrphase_futex;
unsigned int __writers_futex;
unsigned int __pad3;
unsigned int __pad4;
int __cur_writer;
int __shared;
signed char __rwelision;
unsigned char __pad1[ 7];
unsigned long int __pad2;
unsigned int __flags;
};
struct __pthread_internal_list {
struct __pthread_internal_list * __prev;
struct __pthread_internal_list * __next;
};
struct __pthread_mutex_s {
int __lock;
unsigned int __count;
int __owner;
unsigned int __nusers;
int __kind;
short __spins;
short __elision;
struct __pthread_internal_list __list;
};
struct __pthread_cond_s {
union {
unsigned long long int __wseq;
struct {
unsigned int __low;
unsigned int __high;
} __wseq32;
} ;
union {
unsigned long long int __g1_start;
struct {
unsigned int __low;
unsigned int __high;
} __g1_start32;
} ;
unsigned int __g_refs[ 2];
unsigned int __g_size[ 2];
unsigned int __g1_orig_size;
unsigned int __wrefs;
unsigned int __g_signals[ 2];
};
union _noname10_ {
char __size[ 4];
int __align;
};
union _noname11_ {
char __size[ 4];
int __align;
};
union pthread_attr_t {
char __size[ 56];
long int __align;
};
union _noname12_ {
struct __pthread_mutex_s __data;
char __size[ 40];
long int __align;
};
union _noname13_ {
struct __pthread_cond_s __data;
char __size[ 48];
long long int __align;
};
union _noname14_ {
struct __pthread_rwlock_arch_t __data;
char __size[ 56];
long int __align;
};
union _noname15_ {
char __size[ 8];
long int __align;
};
union _noname16_ {
char __size[ 32];
long int __align;
};
union _noname17_ {
char __size[ 4];
int __align;
};
extern long int random(void);
extern void srandom(unsigned int __seed);
extern char * initstate(unsigned int __seed, char * __statebuf, long unsigned int __statelen);
extern char * setstate(char * __statebuf);
struct random_data {
signed int (* fptr);
signed int (* rptr);
signed int (* state);
int rand_type;
int rand_deg;
int rand_sep;
signed int (* end_ptr);
};
extern int random_r(struct random_data * __buf, signed int (* __result));
extern int srandom_r(unsigned int __seed, struct random_data * __buf);
extern int initstate_r(unsigned int __seed, char * __statebuf, long unsigned int __statelen, struct random_data * __buf);
extern int setstate_r(char * __statebuf, struct random_data * __buf);
extern int rand(void);
extern void srand(unsigned int __seed);
extern int rand_r(unsigned int * __seed);
extern double drand48(void);
extern double erand48(unsigned short int __xsubi[ 3]);
extern long int lrand48(void);
extern long int nrand48(unsigned short int __xsubi[ 3]);
extern long int mrand48(void);
extern long int jrand48(unsigned short int __xsubi[ 3]);
extern void srand48(long int __seedval);
extern unsigned short int * seed48(unsigned short int __seed16v[ 3]);
extern void lcong48(unsigned short int __param[ 7]);
struct drand48_data {
unsigned short int __x[ 3];
unsigned short int __old_x[ 3];
unsigned short int __c;
unsigned short int __init;
unsigned long long int __a;
};
extern int drand48_r(struct drand48_data * __buffer, double * __result);
extern int erand48_r(unsigned short int __xsubi[ 3], struct drand48_data * __buffer, double * __result);
extern int lrand48_r(struct drand48_data * __buffer, long int * __result);
extern int nrand48_r(unsigned short int __xsubi[ 3], struct drand48_data * __buffer, long int * __result);
extern int mrand48_r(struct drand48_data * __buffer, long int * __result);
extern int jrand48_r(unsigned short int __xsubi[ 3], struct drand48_data * __buffer, long int * __result);
extern int srand48_r(long int __seedval, struct drand48_data * __buffer);
extern int seed48_r(unsigned short int __seed16v[ 3], struct drand48_data * __buffer);
extern int lcong48_r(unsigned short int __param[ 7], struct drand48_data * __buffer);
extern void * malloc(long unsigned int __size);
extern void * calloc(long unsigned int __nmemb, long unsigned int __size);
extern void * realloc(void * __ptr, long unsigned int __size);
extern void free(void * __ptr);
extern void * alloca(long unsigned int __size);
extern void * valloc(long unsigned int __size);
extern int posix_memalign(void ** __memptr, long unsigned int __alignment, long unsigned int __size);
extern void * aligned_alloc(long unsigned int __alignment, long unsigned int __size);
extern void abort(void);
extern int atexit(void (* __func)(void));
extern int at_quick_exit(void (* __func)(void));
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg);
extern void exit(int __status);
extern void quick_exit(int __status);
extern void _Exit(int __status);
extern char * getenv(const char * __name);
extern int putenv(char * __string);
extern int setenv(const char * __name, const char * __value, int __replace);
extern int unsetenv(const char * __name);
extern int clearenv(void);
extern char * mktemp(char * __template);
extern int mkstemp(char * __template);
extern int mkstemps(char * __template, int __suffixlen);
extern char * mkdtemp(char * __template);
extern int system(const char * __command);
extern char * realpath(const char * __name, char * __resolved);
extern void * bsearch(const void * __key, const void * __base, long unsigned int __nmemb, long unsigned int __size, int (* __compar)(const void *, const void *));
extern void qsort(void * __base, long unsigned int __nmemb, long unsigned int __size, int (* __compar)(const void *, const void *));
extern int abs(int __x);
extern long int labs(long int __x);
extern long long int llabs(long long int __x);
extern struct _noname5_ div(int __numer, int __denom);
extern struct _noname6_ ldiv(long int __numer, long int __denom);
extern struct _noname7_ lldiv(long long int __numer, long long int __denom);
extern char * ecvt(double __value, int __ndigit, int * __decpt, int * __sign);
extern char * fcvt(double __value, int __ndigit, int * __decpt, int * __sign);
extern char * gcvt(double __value, int __ndigit, char * __buf);
extern char * qecvt(long double __value, int __ndigit, int * __decpt, int * __sign);
extern char * qfcvt(long double __value, int __ndigit, int * __decpt, int * __sign);
extern char * qgcvt(long double __value, int __ndigit, char * __buf);
extern int ecvt_r(double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, long unsigned int __len);
extern int fcvt_r(double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, long unsigned int __len);
extern int qecvt_r(long double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, long unsigned int __len);
extern int qfcvt_r(long double __value, int __ndigit, int * __decpt, int * __sign, char * __buf, long unsigned int __len);
extern int mblen(const char * __s, long unsigned int __n);
extern int mbtowc(int (* __pwc), const char * __s, long unsigned int __n);
extern int wctomb(char * __s, int __wchar);
extern long unsigned int mbstowcs(int (* __pwcs), const char * __s, long unsigned int __n);
extern long unsigned int wcstombs(char * __s, const int (* __pwcs), long unsigned int __n);
extern int rpmatch(const char * __response);
extern int getsubopt(char ** __optionp, char *const * __tokens, char ** __valuep);
extern int getloadavg(double __loadavg[], int __nelem);
extern void * memcpy(void * __dest, const void * __src, long unsigned int __n);
extern void * memmove(void * __dest, const void * __src, long unsigned int __n);
extern void * memccpy(void * __dest, const void * __src, int __c, long unsigned int __n);
extern void * memset(void * __s, int __c, long unsigned int __n);
extern int memcmp(const void * __s1, const void * __s2, long unsigned int __n);
extern void * memchr(const void * __s, int __c, long unsigned int __n);
extern char * strcpy(char * __dest, const char * __src);
extern char * strncpy(char * __dest, const char * __src, long unsigned int __n);
extern char * strcat(char * __dest, const char * __src);
extern char * strncat(char * __dest, const char * __src, long unsigned int __n);
extern int strcmp(const char * __s1, const char * __s2);
extern int strncmp(const char * __s1, const char * __s2, long unsigned int __n);
extern int strcoll(const char * __s1, const char * __s2);
extern long unsigned int strxfrm(char * __dest, const char * __src, long unsigned int __n);
struct __locale_struct {
struct __locale_data * __locales[ 13];
const unsigned short int * __ctype_b;
const int * __ctype_tolower;
const int * __ctype_toupper;
const char * __names[ 13];
};
extern int strcoll_l(const char * __s1, const char * __s2, struct __locale_struct * __l);
extern long unsigned int strxfrm_l(char * __dest, const char * __src, long unsigned int __n, struct __locale_struct * __l);
extern char * strdup(const char * __s);
extern char * strndup(const char * __string, long unsigned int __n);
extern char * strchr(const char * __s, int __c);
extern char * strrchr(const char * __s, int __c);
extern long unsigned int strcspn(const char * __s, const char * __reject);
extern long unsigned int strspn(const char * __s, const char * __accept);
extern char * strpbrk(const char * __s, const char * __accept);
extern char * strstr(const char * __haystack, const char * __needle);
extern char * strtok(char * __s, const char * __delim);
extern char * __strtok_r(char * __s, const char * __delim, char ** __save_ptr);
extern char * strtok_r(char * __s, const char * __delim, char ** __save_ptr);
extern long unsigned int strlen(const char * __s);
extern long unsigned int strnlen(const char * __string, long unsigned int __maxlen);
extern char * strerror(int __errnum);
extern int __xpg_strerror_r(int __errnum, char * __buf, long unsigned int __buflen);
extern char * strerror_l(int __errnum, struct __locale_struct * __l);
extern int bcmp(const void * __s1, const void * __s2, long unsigned int __n);
extern void bcopy(const void * __src, void * __dest, long unsigned int __n);
extern void bzero(void * __s, long unsigned int __n);
extern char * index(const char * __s, int __c);
extern char * rindex(const char * __s, int __c);
extern int ffs(int __i);
extern int ffsl(long int __l);
extern int ffsll(long long int __ll);
extern int strcasecmp(const char * __s1, const char * __s2);
extern int strncasecmp(const char * __s1, const char * __s2, long unsigned int __n);
extern int strcasecmp_l(const char * __s1, const char * __s2, struct __locale_struct * __loc);
extern int strncasecmp_l(const char * __s1, const char * __s2, long unsigned int __n, struct __locale_struct * __loc);
extern void explicit_bzero(void * __s, long unsigned int __n);
extern char * strsep(char ** __stringp, const char * __delim);
extern char * strsignal(int __sig);
extern char * __stpcpy(char * __dest, const char * __src);
extern char * stpcpy(char * __dest, const char * __src);
extern char * __stpncpy(char * __dest, const char * __src, long unsigned int __n);
extern char * stpncpy(char * __dest, const char * __src, long unsigned int __n);
extern int bots_sequential_flag;
extern int bots_benchmark_flag;
extern int bots_check_flag;
extern int bots_result;
extern int bots_output_format;
extern int bots_print_header;
extern char bots_name[];
extern char bots_parameters[];
extern char bots_model[];
extern char bots_resources[];
extern char bots_exec_date[];
extern char bots_exec_message[];
extern char bots_comp_date[];
extern char bots_comp_message[];
extern char bots_cc[];
extern char bots_cflags[];
extern char bots_ld[];
extern char bots_ldflags[];
extern double bots_time_program;
extern double bots_time_sequential;
extern unsigned long long bots_number_of_tasks;
extern char bots_cutoff[];
extern int bots_cutoff_value;
extern int bots_app_cutoff_value;
extern int bots_app_cutoff_value_1;
extern int bots_app_cutoff_value_2;
extern int bots_arg_size;
extern int bots_arg_size_1;
extern int bots_arg_size_2;
long bots_usecs();
void bots_error(int error, char * message);
void bots_warning(int warning, char * message);
enum _noname18_ {
BOTS_VERBOSE_NONE = 0, BOTS_VERBOSE_DEFAULT, BOTS_VERBOSE_DEBUG
};
extern enum _noname18_ bots_verbose_mode;
int omp_in_parallel(void);
int omp_get_thread_num(void);
void omp_set_num_threads(int num_threads);
int omp_get_num_threads(void);
int omp_get_max_threads(void);
int omp_get_num_procs(void);
void omp_set_dynamic(int dynamic_threads);
int omp_get_dynamic(void);
void omp_set_nested(int nested);
int omp_get_nested(void);
enum omp_sched_t {
omp_sched_static = 1, omp_sched_dynamic = 2, omp_sched_guided = 3, omp_sched_auto = 4
};
enum omp_proc_bind_t {
omp_proc_bind_false = 0, omp_proc_bind_true = 1, omp_proc_bind_master = 2, omp_proc_bind_close = 3, omp_proc_bind_spread = 4
};
void omp_init_lock(void * (* lock));
void omp_destroy_lock(void * (* lock));
void omp_set_lock(void * (* lock));
void omp_unset_lock(void * (* lock));
int omp_test_lock(void * (* lock));
void omp_init_nest_lock(void * (* lock));
void omp_destroy_nest_lock(void * (* lock));
void omp_set_nest_lock(void * (* lock));
void omp_unset_nest_lock(void * (* lock));
int omp_test_nest_lock(void * (* lock));
double omp_get_wtime(void);
double omp_get_wtick(void);
void omp_set_schedule(enum omp_sched_t kind, int chunk);
void omp_get_schedule(enum omp_sched_t (* kind), int * chunk);
int omp_get_thread_limit(void);
void omp_set_max_active_levels(int levels);
int omp_get_max_active_levels(void);
int omp_get_level(void);
int omp_get_ancestor_thread_num(int level);
int omp_get_team_size(int level);
int omp_get_active_level(void);
int omp_in_final(void);
int omp_get_cancellation(void);
enum omp_proc_bind_t omp_get_proc_bind(void);
int omp_get_num_teams(void);
int omp_get_team_num(void);
int omp_is_initial_device(void);
void omp_set_default_device(int device_num);
int omp_get_default_device(void);
int omp_get_num_devices(void);
struct _noname19_ {
double re;
double im;
};
void compute_w_coefficients(int n, int a, int b, struct _noname19_ (* W));
void compute_w_coefficients_seq(int n, int a, int b, struct _noname19_ (* W));
int factor(int n);
void unshuffle(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int r, int m);
void unshuffle_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int r, int m);
void fft_twiddle_gen1(struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int r, int m, int nW, int nWdnti, int nWdntm);
void fft_twiddle_gen(int i, int i1, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int r, int m);
void fft_twiddle_gen_seq(int i, int i1, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int r, int m);
void fft_base_2(struct _noname19_ (* in), struct _noname19_ (* out));
void fft_twiddle_2(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_twiddle_2_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_unshuffle_2(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_unshuffle_2_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_base_4(struct _noname19_ (* in), struct _noname19_ (* out));
void fft_twiddle_4(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_twiddle_4_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_unshuffle_4(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_unshuffle_4_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_base_8(struct _noname19_ (* in), struct _noname19_ (* out));
void fft_twiddle_8(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_twiddle_8_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_unshuffle_8(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_unshuffle_8_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_base_16(struct _noname19_ (* in), struct _noname19_ (* out));
void fft_twiddle_16(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_twiddle_16_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_unshuffle_16(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_unshuffle_16_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_base_32(struct _noname19_ (* in), struct _noname19_ (* out));
void fft_twiddle_32(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_twiddle_32_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m);
void fft_unshuffle_32(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_unshuffle_32_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m);
void fft_aux(int n, struct _noname19_ (* in), struct _noname19_ (* out), int * factors, struct _noname19_ (* W), int nW);
void fft_aux_seq(int n, struct _noname19_ (* in), struct _noname19_ (* out), int * factors, struct _noname19_ (* W), int nW);
void fft(int n, struct _noname19_ (* in), struct _noname19_ (* out));
void fft_seq(int n, struct _noname19_ (* in), struct _noname19_ (* out));
int test_correctness(int n, struct _noname19_ (* out1), struct _noname19_ (* out2));
static void * _taskFunc0_(void *);
static void * _taskFunc1_(void *);
void compute_w_coefficients(int n, int a, int b, struct _noname19_ (* W))
{
register double twoPiOverN;
register int k;
register double s;
register double c;
if (b - a < 128)
{
twoPiOverN = 2.0 * 3.1415926535897932384626434 / n;
for (k = a; k <= b; ++k)
{
c = cos(twoPiOverN * k);
((W[k]).re) = ((W[n - k]).re) = c;
s = sin(twoPiOverN * k);
((W[k]).im) = -s;
((W[n - k]).im) = s;
}
}
else
{
int ab = (a + b) / 2;
_taskFunc0_((void *)0);
_taskFunc1_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc1_(void * __arg)
{
struct __taskenv__ {
int n;
int ab;
int b;
struct _noname19_ (* W);
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int n = _tenv->n;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* W) = _tenv->W;
{
compute_w_coefficients(n, ab + 1, b, W);
CANCEL_task_60 :
;
}
ort_taskenv_free(_tenv, _taskFunc1_);
return ((void *) 0);
}
static void * _taskFunc0_(void * __arg)
{
struct __taskenv__ {
int n;
int a;
int ab;
struct _noname19_ (* W);
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int n = _tenv->n;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* W) = _tenv->W;
{
compute_w_coefficients(n, a, ab, W);
CANCEL_task_58 :
;
}
ort_taskenv_free(_tenv, _taskFunc0_);
return ((void *) 0);
}
void compute_w_coefficients_seq(int n, int a, int b, struct _noname19_ (* W))
{
register double twoPiOverN;
register int k;
register double s;
register double c;
if (b - a < 128)
{
twoPiOverN = 2.0 * 3.1415926535897932384626434 / n;
for (k = a; k <= b; ++k)
{
c = cos(twoPiOverN * k);
((W[k]).re) = ((W[n - k]).re) = c;
s = sin(twoPiOverN * k);
((W[k]).im) = -s;
((W[n - k]).im) = s;
}
}
else
{
int ab = (a + b) / 2;
compute_w_coefficients_seq(n, a, ab, W);
compute_w_coefficients_seq(n, ab + 1, b, W);
}
}
int factor(int n)
{
int r;
if (n < 2)
return (1);
if (n == 64 || n == 128 || n == 256 || n == 1024 || n == 2048 || n == 4096)
return (8);
if ((n & 15) == 0)
return (16);
if ((n & 7) == 0)
return (8);
if ((n & 3) == 0)
return (4);
if ((n & 1) == 0)
return (2);
for (r = 3; r < n; r += 2)
if (n % r == 0)
return (r);
return (n);
}
static void * _taskFunc2_(void *);
static void * _taskFunc3_(void *);
void unshuffle(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int r, int m)
{
int i, j;
int r4 = r & (~0x3);
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if (b - a < 16)
{
ip = in + a * r;
for (i = a; i < b; ++i)
{
jp = out + i;
for (j = 0; j < r4; j += 4)
{
jp[0] = ip[0];
jp[m] = ip[1];
jp[2 * m] = ip[2];
jp[3 * m] = ip[3];
jp += 4 * m;
ip += 4;
}
for ( ; j < r; ++j)
{
*jp = *ip;
ip++;
jp += m;
}
}
}
else
{
int ab = (a + b) / 2;
_taskFunc2_((void *)0);
_taskFunc3_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc3_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
int r;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int r = _tenv->r;
int m = _tenv->m;
{
unshuffle(ab, b, in, out, r, m);
CANCEL_task_137 :
;
}
ort_taskenv_free(_tenv, _taskFunc3_);
return ((void *) 0);
}
static void * _taskFunc2_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
int r;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int r = _tenv->r;
int m = _tenv->m;
{
unshuffle(a, ab, in, out, r, m);
CANCEL_task_135 :
;
}
ort_taskenv_free(_tenv, _taskFunc2_);
return ((void *) 0);
}
void unshuffle_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int r, int m)
{
int i, j;
int r4 = r & (~0x3);
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if (b - a < 16)
{
ip = in + a * r;
for (i = a; i < b; ++i)
{
jp = out + i;
for (j = 0; j < r4; j += 4)
{
jp[0] = ip[0];
jp[m] = ip[1];
jp[2 * m] = ip[2];
jp[3 * m] = ip[3];
jp += 4 * m;
ip += 4;
}
for ( ; j < r; ++j)
{
*jp = *ip;
ip++;
jp += m;
}
}
}
else
{
int ab = (a + b) / 2;
unshuffle_seq(a, ab, in, out, r, m);
unshuffle_seq(ab, b, in, out, r, m);
}
}
void fft_twiddle_gen1(struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int r, int m, int nW, int nWdnti, int nWdntm)
{
int j, k;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
for (k = 0, kp = out; k < r; ++k, kp += m)
{
double r0;
double i0;
double rt;
double it;
double rw;
double iw;
int l1 = nWdnti + nWdntm * k;
int l0;
r0 = i0 = 0.0;
for (j = 0, jp = in, l0 = 0; j < r; ++j, jp += m)
{
rw = ((W[l0]).re);
iw = ((W[l0]).im);
rt = ((*jp).re);
it = ((*jp).im);
r0 += rt * rw - it * iw;
i0 += rt * iw + it * rw;
l0 += l1;
if (l0 > nW)
l0 -= nW;
}
((*kp).re) = r0;
((*kp).im) = i0;
}
}
static void * _taskFunc4_(void *);
static void * _taskFunc5_(void *);
static void * _taskFunc6_(void *);
void fft_twiddle_gen(int i, int i1, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int r, int m)
{
if (i == i1 - 1)
{
_taskFunc4_((void *)0);
}
else
{
int i2 = (i + i1) / 2;
_taskFunc5_((void *)0);
_taskFunc6_((void *)0);
}
ort_taskwait(0);
}
static void * _taskFunc6_(void * __arg)
{
struct __taskenv__ {
int i2;
int i1;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int r;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int i2 = _tenv->i2;
int i1 = _tenv->i1;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int r = _tenv->r;
int m = _tenv->m;
{
fft_twiddle_gen(i2, i1, in, out, W, nW, nWdn, r, m);
CANCEL_task_213 :
;
}
ort_taskenv_free(_tenv, _taskFunc6_);
return ((void *) 0);
}
static void * _taskFunc5_(void * __arg)
{
struct __taskenv__ {
int i;
int i2;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int r;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int i = _tenv->i;
int i2 = _tenv->i2;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int r = _tenv->r;
int m = _tenv->m;
{
fft_twiddle_gen(i, i2, in, out, W, nW, nWdn, r, m);
CANCEL_task_210 :
;
}
ort_taskenv_free(_tenv, _taskFunc5_);
return ((void *) 0);
}
static void * _taskFunc4_(void * __arg)
{
struct __taskenv__ {
struct _noname19_ (* in);
int i;
struct _noname19_ (* out);
struct _noname19_ (* W);
int r;
int m;
int nW;
int nWdn;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
struct _noname19_ (* in) = _tenv->in;
int i = _tenv->i;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int r = _tenv->r;
int m = _tenv->m;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
{
fft_twiddle_gen1(in + i, out + i, W, r, m, nW, nWdn * i, nWdn * m);
CANCEL_task_205 :
;
}
ort_taskenv_free(_tenv, _taskFunc4_);
return ((void *) 0);
}
void fft_twiddle_gen_seq(int i, int i1, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int r, int m)
{
if (i == i1 - 1)
{
fft_twiddle_gen1(in + i, out + i, W, r, m, nW, nWdn * i, nWdn * m);
}
else
{
int i2 = (i + i1) / 2;
fft_twiddle_gen_seq(i, i2, in, out, W, nW, nWdn, r, m);
fft_twiddle_gen_seq(i2, i1, in, out, W, nW, nWdn, r, m);
}
}
void fft_base_2(struct _noname19_ (* in), struct _noname19_ (* out))
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
r1_0 = ((in[0]).re);
i1_0 = ((in[0]).im);
r1_1 = ((in[1]).re);
i1_1 = ((in[1]).im);
((out[0]).re) = (r1_0 + r1_1);
((out[0]).im) = (i1_0 + i1_1);
((out[1]).re) = (r1_0 - r1_1);
((out[1]).im) = (i1_0 - i1_1);
}
static void * _taskFunc7_(void *);
static void * _taskFunc8_(void *);
void fft_twiddle_2(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
r1_0 = ((jp[0 * m]).re);
i1_0 = ((jp[0 * m]).im);
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r1_1 = ((wr * tmpr) - (wi * tmpi));
i1_1 = ((wi * tmpr) + (wr * tmpi));
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[1 * m]).re) = (r1_0 - r1_1);
((kp[1 * m]).im) = (i1_0 - i1_1);
}
}
}
else
{
int ab = (a + b) / 2;
_taskFunc7_((void *)0);
_taskFunc8_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc8_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_2(ab, b, in, out, W, nW, nWdn, m);
CANCEL_task_277 :
;
}
ort_taskenv_free(_tenv, _taskFunc8_);
return ((void *) 0);
}
static void * _taskFunc7_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_2(a, ab, in, out, W, nW, nWdn, m);
CANCEL_task_275 :
;
}
ort_taskenv_free(_tenv, _taskFunc7_);
return ((void *) 0);
}
void fft_twiddle_2_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
r1_0 = ((jp[0 * m]).re);
i1_0 = ((jp[0 * m]).im);
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r1_1 = ((wr * tmpr) - (wi * tmpi));
i1_1 = ((wi * tmpr) + (wr * tmpi));
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[1 * m]).re) = (r1_0 - r1_1);
((kp[1 * m]).im) = (i1_0 - i1_1);
}
}
}
else
{
int ab = (a + b) / 2;
fft_twiddle_2_seq(a, ab, in, out, W, nW, nWdn, m);
fft_twiddle_2_seq(ab, b, in, out, W, nW, nWdn, m);
}
}
static void * _taskFunc9_(void *);
static void * _taskFunc10_(void *);
void fft_unshuffle_2(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 2;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
_taskFunc9_((void *)0);
_taskFunc10_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc10_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_2(ab, b, in, out, m);
CANCEL_task_331 :
;
}
ort_taskenv_free(_tenv, _taskFunc10_);
return ((void *) 0);
}
static void * _taskFunc9_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_2(a, ab, in, out, m);
CANCEL_task_329 :
;
}
ort_taskenv_free(_tenv, _taskFunc9_);
return ((void *) 0);
}
void fft_unshuffle_2_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 2;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
fft_unshuffle_2_seq(a, ab, in, out, m);
fft_unshuffle_2_seq(ab, b, in, out, m);
}
}
void fft_base_4(struct _noname19_ (* in), struct _noname19_ (* out))
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
r2_0 = ((in[0]).re);
i2_0 = ((in[0]).im);
r2_2 = ((in[2]).re);
i2_2 = ((in[2]).im);
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_2 = (r2_0 - r2_2);
i1_2 = (i2_0 - i2_2);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
r2_1 = ((in[1]).re);
i2_1 = ((in[1]).im);
r2_3 = ((in[3]).re);
i2_3 = ((in[3]).im);
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_3 = (r2_1 - r2_3);
i1_3 = (i2_1 - i2_3);
}
((out[0]).re) = (r1_0 + r1_1);
((out[0]).im) = (i1_0 + i1_1);
((out[2]).re) = (r1_0 - r1_1);
((out[2]).im) = (i1_0 - i1_1);
((out[1]).re) = (r1_2 + i1_3);
((out[1]).im) = (i1_2 - r1_3);
((out[3]).re) = (r1_2 - i1_3);
((out[3]).im) = (i1_2 + r1_3);
}
static void * _taskFunc11_(void *);
static void * _taskFunc12_(void *);
void fft_twiddle_4(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
r2_0 = ((jp[0 * m]).re);
i2_0 = ((jp[0 * m]).im);
wr = ((W[2 * l1]).re);
wi = ((W[2 * l1]).im);
tmpr = ((jp[2 * m]).re);
tmpi = ((jp[2 * m]).im);
r2_2 = ((wr * tmpr) - (wi * tmpi));
i2_2 = ((wi * tmpr) + (wr * tmpi));
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_2 = (r2_0 - r2_2);
i1_2 = (i2_0 - i2_2);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r2_1 = ((wr * tmpr) - (wi * tmpi));
i2_1 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[3 * l1]).re);
wi = ((W[3 * l1]).im);
tmpr = ((jp[3 * m]).re);
tmpi = ((jp[3 * m]).im);
r2_3 = ((wr * tmpr) - (wi * tmpi));
i2_3 = ((wi * tmpr) + (wr * tmpi));
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_3 = (r2_1 - r2_3);
i1_3 = (i2_1 - i2_3);
}
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[2 * m]).re) = (r1_0 - r1_1);
((kp[2 * m]).im) = (i1_0 - i1_1);
((kp[1 * m]).re) = (r1_2 + i1_3);
((kp[1 * m]).im) = (i1_2 - r1_3);
((kp[3 * m]).re) = (r1_2 - i1_3);
((kp[3 * m]).im) = (i1_2 + r1_3);
}
}
}
else
{
int ab = (a + b) / 2;
_taskFunc11_((void *)0);
_taskFunc12_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc12_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_4(ab, b, in, out, W, nW, nWdn, m);
CANCEL_task_458 :
;
}
ort_taskenv_free(_tenv, _taskFunc12_);
return ((void *) 0);
}
static void * _taskFunc11_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_4(a, ab, in, out, W, nW, nWdn, m);
CANCEL_task_456 :
;
}
ort_taskenv_free(_tenv, _taskFunc11_);
return ((void *) 0);
}
void fft_twiddle_4_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
r2_0 = ((jp[0 * m]).re);
i2_0 = ((jp[0 * m]).im);
wr = ((W[2 * l1]).re);
wi = ((W[2 * l1]).im);
tmpr = ((jp[2 * m]).re);
tmpi = ((jp[2 * m]).im);
r2_2 = ((wr * tmpr) - (wi * tmpi));
i2_2 = ((wi * tmpr) + (wr * tmpi));
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_2 = (r2_0 - r2_2);
i1_2 = (i2_0 - i2_2);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r2_1 = ((wr * tmpr) - (wi * tmpi));
i2_1 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[3 * l1]).re);
wi = ((W[3 * l1]).im);
tmpr = ((jp[3 * m]).re);
tmpi = ((jp[3 * m]).im);
r2_3 = ((wr * tmpr) - (wi * tmpi));
i2_3 = ((wi * tmpr) + (wr * tmpi));
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_3 = (r2_1 - r2_3);
i1_3 = (i2_1 - i2_3);
}
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[2 * m]).re) = (r1_0 - r1_1);
((kp[2 * m]).im) = (i1_0 - i1_1);
((kp[1 * m]).re) = (r1_2 + i1_3);
((kp[1 * m]).im) = (i1_2 - r1_3);
((kp[3 * m]).re) = (r1_2 - i1_3);
((kp[3 * m]).im) = (i1_2 + r1_3);
}
}
}
else
{
int ab = (a + b) / 2;
fft_twiddle_4_seq(a, ab, in, out, W, nW, nWdn, m);
fft_twiddle_4_seq(ab, b, in, out, W, nW, nWdn, m);
}
}
static void * _taskFunc13_(void *);
static void * _taskFunc14_(void *);
void fft_unshuffle_4(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 4;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
_taskFunc13_((void *)0);
_taskFunc14_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc14_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_4(ab, b, in, out, m);
CANCEL_task_550 :
;
}
ort_taskenv_free(_tenv, _taskFunc14_);
return ((void *) 0);
}
static void * _taskFunc13_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_4(a, ab, in, out, m);
CANCEL_task_548 :
;
}
ort_taskenv_free(_tenv, _taskFunc13_);
return ((void *) 0);
}
void fft_unshuffle_4_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 4;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
fft_unshuffle_4_seq(a, ab, in, out, m);
fft_unshuffle_4_seq(ab, b, in, out, m);
}
}
void fft_base_8(struct _noname19_ (* in), struct _noname19_ (* out))
{
double tmpr;
double tmpi;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
r3_0 = ((in[0]).re);
i3_0 = ((in[0]).im);
r3_4 = ((in[4]).re);
i3_4 = ((in[4]).im);
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_4 = (r3_0 - r3_4);
i2_4 = (i3_0 - i3_4);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
r3_2 = ((in[2]).re);
i3_2 = ((in[2]).im);
r3_6 = ((in[6]).re);
i3_6 = ((in[6]).im);
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_6 = (r3_2 - r3_6);
i2_6 = (i3_2 - i3_6);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_4 = (r2_0 - r2_2);
i1_4 = (i2_0 - i2_2);
r1_2 = (r2_4 + i2_6);
i1_2 = (i2_4 - r2_6);
r1_6 = (r2_4 - i2_6);
i1_6 = (i2_4 + r2_6);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
r3_1 = ((in[1]).re);
i3_1 = ((in[1]).im);
r3_5 = ((in[5]).re);
i3_5 = ((in[5]).im);
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_5 = (r3_1 - r3_5);
i2_5 = (i3_1 - i3_5);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
r3_3 = ((in[3]).re);
i3_3 = ((in[3]).im);
r3_7 = ((in[7]).re);
i3_7 = ((in[7]).im);
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_7 = (r3_3 - r3_7);
i2_7 = (i3_3 - i3_7);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_5 = (r2_1 - r2_3);
i1_5 = (i2_1 - i2_3);
r1_3 = (r2_5 + i2_7);
i1_3 = (i2_5 - r2_7);
r1_7 = (r2_5 - i2_7);
i1_7 = (i2_5 + r2_7);
}
((out[0]).re) = (r1_0 + r1_1);
((out[0]).im) = (i1_0 + i1_1);
((out[4]).re) = (r1_0 - r1_1);
((out[4]).im) = (i1_0 - i1_1);
tmpr = (0.707106781187 * (r1_3 + i1_3));
tmpi = (0.707106781187 * (i1_3 - r1_3));
((out[1]).re) = (r1_2 + tmpr);
((out[1]).im) = (i1_2 + tmpi);
((out[5]).re) = (r1_2 - tmpr);
((out[5]).im) = (i1_2 - tmpi);
((out[2]).re) = (r1_4 + i1_5);
((out[2]).im) = (i1_4 - r1_5);
((out[6]).re) = (r1_4 - i1_5);
((out[6]).im) = (i1_4 + r1_5);
tmpr = (0.707106781187 * (i1_7 - r1_7));
tmpi = (0.707106781187 * (r1_7 + i1_7));
((out[3]).re) = (r1_6 + tmpr);
((out[3]).im) = (i1_6 - tmpi);
((out[7]).re) = (r1_6 - tmpr);
((out[7]).im) = (i1_6 + tmpi);
}
}
static void * _taskFunc15_(void *);
static void * _taskFunc16_(void *);
void fft_twiddle_8(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
r3_0 = ((jp[0 * m]).re);
i3_0 = ((jp[0 * m]).im);
wr = ((W[4 * l1]).re);
wi = ((W[4 * l1]).im);
tmpr = ((jp[4 * m]).re);
tmpi = ((jp[4 * m]).im);
r3_4 = ((wr * tmpr) - (wi * tmpi));
i3_4 = ((wi * tmpr) + (wr * tmpi));
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_4 = (r3_0 - r3_4);
i2_4 = (i3_0 - i3_4);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
wr = ((W[2 * l1]).re);
wi = ((W[2 * l1]).im);
tmpr = ((jp[2 * m]).re);
tmpi = ((jp[2 * m]).im);
r3_2 = ((wr * tmpr) - (wi * tmpi));
i3_2 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[6 * l1]).re);
wi = ((W[6 * l1]).im);
tmpr = ((jp[6 * m]).re);
tmpi = ((jp[6 * m]).im);
r3_6 = ((wr * tmpr) - (wi * tmpi));
i3_6 = ((wi * tmpr) + (wr * tmpi));
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_6 = (r3_2 - r3_6);
i2_6 = (i3_2 - i3_6);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_4 = (r2_0 - r2_2);
i1_4 = (i2_0 - i2_2);
r1_2 = (r2_4 + i2_6);
i1_2 = (i2_4 - r2_6);
r1_6 = (r2_4 - i2_6);
i1_6 = (i2_4 + r2_6);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r3_1 = ((wr * tmpr) - (wi * tmpi));
i3_1 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[5 * l1]).re);
wi = ((W[5 * l1]).im);
tmpr = ((jp[5 * m]).re);
tmpi = ((jp[5 * m]).im);
r3_5 = ((wr * tmpr) - (wi * tmpi));
i3_5 = ((wi * tmpr) + (wr * tmpi));
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_5 = (r3_1 - r3_5);
i2_5 = (i3_1 - i3_5);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
wr = ((W[3 * l1]).re);
wi = ((W[3 * l1]).im);
tmpr = ((jp[3 * m]).re);
tmpi = ((jp[3 * m]).im);
r3_3 = ((wr * tmpr) - (wi * tmpi));
i3_3 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[7 * l1]).re);
wi = ((W[7 * l1]).im);
tmpr = ((jp[7 * m]).re);
tmpi = ((jp[7 * m]).im);
r3_7 = ((wr * tmpr) - (wi * tmpi));
i3_7 = ((wi * tmpr) + (wr * tmpi));
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_7 = (r3_3 - r3_7);
i2_7 = (i3_3 - i3_7);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_5 = (r2_1 - r2_3);
i1_5 = (i2_1 - i2_3);
r1_3 = (r2_5 + i2_7);
i1_3 = (i2_5 - r2_7);
r1_7 = (r2_5 - i2_7);
i1_7 = (i2_5 + r2_7);
}
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[4 * m]).re) = (r1_0 - r1_1);
((kp[4 * m]).im) = (i1_0 - i1_1);
tmpr = (0.707106781187 * (r1_3 + i1_3));
tmpi = (0.707106781187 * (i1_3 - r1_3));
((kp[1 * m]).re) = (r1_2 + tmpr);
((kp[1 * m]).im) = (i1_2 + tmpi);
((kp[5 * m]).re) = (r1_2 - tmpr);
((kp[5 * m]).im) = (i1_2 - tmpi);
((kp[2 * m]).re) = (r1_4 + i1_5);
((kp[2 * m]).im) = (i1_4 - r1_5);
((kp[6 * m]).re) = (r1_4 - i1_5);
((kp[6 * m]).im) = (i1_4 + r1_5);
tmpr = (0.707106781187 * (i1_7 - r1_7));
tmpi = (0.707106781187 * (r1_7 + i1_7));
((kp[3 * m]).re) = (r1_6 + tmpr);
((kp[3 * m]).im) = (i1_6 - tmpi);
((kp[7 * m]).re) = (r1_6 - tmpr);
((kp[7 * m]).im) = (i1_6 + tmpi);
}
}
}
else
{
int ab = (a + b) / 2;
_taskFunc15_((void *)0);
_taskFunc16_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc16_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_8(ab, b, in, out, W, nW, nWdn, m);
CANCEL_task_836 :
;
}
ort_taskenv_free(_tenv, _taskFunc16_);
return ((void *) 0);
}
static void * _taskFunc15_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_8(a, ab, in, out, W, nW, nWdn, m);
CANCEL_task_834 :
;
}
ort_taskenv_free(_tenv, _taskFunc15_);
return ((void *) 0);
}
void fft_twiddle_8_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
r3_0 = ((jp[0 * m]).re);
i3_0 = ((jp[0 * m]).im);
wr = ((W[4 * l1]).re);
wi = ((W[4 * l1]).im);
tmpr = ((jp[4 * m]).re);
tmpi = ((jp[4 * m]).im);
r3_4 = ((wr * tmpr) - (wi * tmpi));
i3_4 = ((wi * tmpr) + (wr * tmpi));
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_4 = (r3_0 - r3_4);
i2_4 = (i3_0 - i3_4);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
wr = ((W[2 * l1]).re);
wi = ((W[2 * l1]).im);
tmpr = ((jp[2 * m]).re);
tmpi = ((jp[2 * m]).im);
r3_2 = ((wr * tmpr) - (wi * tmpi));
i3_2 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[6 * l1]).re);
wi = ((W[6 * l1]).im);
tmpr = ((jp[6 * m]).re);
tmpi = ((jp[6 * m]).im);
r3_6 = ((wr * tmpr) - (wi * tmpi));
i3_6 = ((wi * tmpr) + (wr * tmpi));
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_6 = (r3_2 - r3_6);
i2_6 = (i3_2 - i3_6);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_4 = (r2_0 - r2_2);
i1_4 = (i2_0 - i2_2);
r1_2 = (r2_4 + i2_6);
i1_2 = (i2_4 - r2_6);
r1_6 = (r2_4 - i2_6);
i1_6 = (i2_4 + r2_6);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r3_1 = ((wr * tmpr) - (wi * tmpi));
i3_1 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[5 * l1]).re);
wi = ((W[5 * l1]).im);
tmpr = ((jp[5 * m]).re);
tmpi = ((jp[5 * m]).im);
r3_5 = ((wr * tmpr) - (wi * tmpi));
i3_5 = ((wi * tmpr) + (wr * tmpi));
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_5 = (r3_1 - r3_5);
i2_5 = (i3_1 - i3_5);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
wr = ((W[3 * l1]).re);
wi = ((W[3 * l1]).im);
tmpr = ((jp[3 * m]).re);
tmpi = ((jp[3 * m]).im);
r3_3 = ((wr * tmpr) - (wi * tmpi));
i3_3 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[7 * l1]).re);
wi = ((W[7 * l1]).im);
tmpr = ((jp[7 * m]).re);
tmpi = ((jp[7 * m]).im);
r3_7 = ((wr * tmpr) - (wi * tmpi));
i3_7 = ((wi * tmpr) + (wr * tmpi));
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_7 = (r3_3 - r3_7);
i2_7 = (i3_3 - i3_7);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_5 = (r2_1 - r2_3);
i1_5 = (i2_1 - i2_3);
r1_3 = (r2_5 + i2_7);
i1_3 = (i2_5 - r2_7);
r1_7 = (r2_5 - i2_7);
i1_7 = (i2_5 + r2_7);
}
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[4 * m]).re) = (r1_0 - r1_1);
((kp[4 * m]).im) = (i1_0 - i1_1);
tmpr = (0.707106781187 * (r1_3 + i1_3));
tmpi = (0.707106781187 * (i1_3 - r1_3));
((kp[1 * m]).re) = (r1_2 + tmpr);
((kp[1 * m]).im) = (i1_2 + tmpi);
((kp[5 * m]).re) = (r1_2 - tmpr);
((kp[5 * m]).im) = (i1_2 - tmpi);
((kp[2 * m]).re) = (r1_4 + i1_5);
((kp[2 * m]).im) = (i1_4 - r1_5);
((kp[6 * m]).re) = (r1_4 - i1_5);
((kp[6 * m]).im) = (i1_4 + r1_5);
tmpr = (0.707106781187 * (i1_7 - r1_7));
tmpi = (0.707106781187 * (r1_7 + i1_7));
((kp[3 * m]).re) = (r1_6 + tmpr);
((kp[3 * m]).im) = (i1_6 - tmpi);
((kp[7 * m]).re) = (r1_6 - tmpr);
((kp[7 * m]).im) = (i1_6 + tmpi);
}
}
}
else
{
int ab = (a + b) / 2;
fft_twiddle_8_seq(a, ab, in, out, W, nW, nWdn, m);
fft_twiddle_8_seq(ab, b, in, out, W, nW, nWdn, m);
}
}
static void * _taskFunc17_(void *);
static void * _taskFunc18_(void *);
void fft_unshuffle_8(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 8;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
_taskFunc17_((void *)0);
_taskFunc18_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc18_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_8(ab, b, in, out, m);
CANCEL_task_1020 :
;
}
ort_taskenv_free(_tenv, _taskFunc18_);
return ((void *) 0);
}
static void * _taskFunc17_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_8(a, ab, in, out, m);
CANCEL_task_1018 :
;
}
ort_taskenv_free(_tenv, _taskFunc17_);
return ((void *) 0);
}
void fft_unshuffle_8_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 8;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
fft_unshuffle_8_seq(a, ab, in, out, m);
fft_unshuffle_8_seq(ab, b, in, out, m);
}
}
void fft_base_16(struct _noname19_ (* in), struct _noname19_ (* out))
{
double tmpr;
double tmpi;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
double r1_8;
double i1_8;
double r1_9;
double i1_9;
double r1_10;
double i1_10;
double r1_11;
double i1_11;
double r1_12;
double i1_12;
double r1_13;
double i1_13;
double r1_14;
double i1_14;
double r1_15;
double i1_15;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
double r2_8;
double i2_8;
double r2_10;
double i2_10;
double r2_12;
double i2_12;
double r2_14;
double i2_14;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
double r3_8;
double i3_8;
double r3_12;
double i3_12;
{
double r4_0;
double i4_0;
double r4_8;
double i4_8;
r4_0 = ((in[0]).re);
i4_0 = ((in[0]).im);
r4_8 = ((in[8]).re);
i4_8 = ((in[8]).im);
r3_0 = (r4_0 + r4_8);
i3_0 = (i4_0 + i4_8);
r3_8 = (r4_0 - r4_8);
i3_8 = (i4_0 - i4_8);
}
{
double r4_4;
double i4_4;
double r4_12;
double i4_12;
r4_4 = ((in[4]).re);
i4_4 = ((in[4]).im);
r4_12 = ((in[12]).re);
i4_12 = ((in[12]).im);
r3_4 = (r4_4 + r4_12);
i3_4 = (i4_4 + i4_12);
r3_12 = (r4_4 - r4_12);
i3_12 = (i4_4 - i4_12);
}
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_8 = (r3_0 - r3_4);
i2_8 = (i3_0 - i3_4);
r2_4 = (r3_8 + i3_12);
i2_4 = (i3_8 - r3_12);
r2_12 = (r3_8 - i3_12);
i2_12 = (i3_8 + r3_12);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
double r3_10;
double i3_10;
double r3_14;
double i3_14;
{
double r4_2;
double i4_2;
double r4_10;
double i4_10;
r4_2 = ((in[2]).re);
i4_2 = ((in[2]).im);
r4_10 = ((in[10]).re);
i4_10 = ((in[10]).im);
r3_2 = (r4_2 + r4_10);
i3_2 = (i4_2 + i4_10);
r3_10 = (r4_2 - r4_10);
i3_10 = (i4_2 - i4_10);
}
{
double r4_6;
double i4_6;
double r4_14;
double i4_14;
r4_6 = ((in[6]).re);
i4_6 = ((in[6]).im);
r4_14 = ((in[14]).re);
i4_14 = ((in[14]).im);
r3_6 = (r4_6 + r4_14);
i3_6 = (i4_6 + i4_14);
r3_14 = (r4_6 - r4_14);
i3_14 = (i4_6 - i4_14);
}
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_10 = (r3_2 - r3_6);
i2_10 = (i3_2 - i3_6);
r2_6 = (r3_10 + i3_14);
i2_6 = (i3_10 - r3_14);
r2_14 = (r3_10 - i3_14);
i2_14 = (i3_10 + r3_14);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_8 = (r2_0 - r2_2);
i1_8 = (i2_0 - i2_2);
tmpr = (0.707106781187 * (r2_6 + i2_6));
tmpi = (0.707106781187 * (i2_6 - r2_6));
r1_2 = (r2_4 + tmpr);
i1_2 = (i2_4 + tmpi);
r1_10 = (r2_4 - tmpr);
i1_10 = (i2_4 - tmpi);
r1_4 = (r2_8 + i2_10);
i1_4 = (i2_8 - r2_10);
r1_12 = (r2_8 - i2_10);
i1_12 = (i2_8 + r2_10);
tmpr = (0.707106781187 * (i2_14 - r2_14));
tmpi = (0.707106781187 * (r2_14 + i2_14));
r1_6 = (r2_12 + tmpr);
i1_6 = (i2_12 - tmpi);
r1_14 = (r2_12 - tmpr);
i1_14 = (i2_12 + tmpi);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
double r2_9;
double i2_9;
double r2_11;
double i2_11;
double r2_13;
double i2_13;
double r2_15;
double i2_15;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
double r3_9;
double i3_9;
double r3_13;
double i3_13;
{
double r4_1;
double i4_1;
double r4_9;
double i4_9;
r4_1 = ((in[1]).re);
i4_1 = ((in[1]).im);
r4_9 = ((in[9]).re);
i4_9 = ((in[9]).im);
r3_1 = (r4_1 + r4_9);
i3_1 = (i4_1 + i4_9);
r3_9 = (r4_1 - r4_9);
i3_9 = (i4_1 - i4_9);
}
{
double r4_5;
double i4_5;
double r4_13;
double i4_13;
r4_5 = ((in[5]).re);
i4_5 = ((in[5]).im);
r4_13 = ((in[13]).re);
i4_13 = ((in[13]).im);
r3_5 = (r4_5 + r4_13);
i3_5 = (i4_5 + i4_13);
r3_13 = (r4_5 - r4_13);
i3_13 = (i4_5 - i4_13);
}
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_9 = (r3_1 - r3_5);
i2_9 = (i3_1 - i3_5);
r2_5 = (r3_9 + i3_13);
i2_5 = (i3_9 - r3_13);
r2_13 = (r3_9 - i3_13);
i2_13 = (i3_9 + r3_13);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
double r3_11;
double i3_11;
double r3_15;
double i3_15;
{
double r4_3;
double i4_3;
double r4_11;
double i4_11;
r4_3 = ((in[3]).re);
i4_3 = ((in[3]).im);
r4_11 = ((in[11]).re);
i4_11 = ((in[11]).im);
r3_3 = (r4_3 + r4_11);
i3_3 = (i4_3 + i4_11);
r3_11 = (r4_3 - r4_11);
i3_11 = (i4_3 - i4_11);
}
{
double r4_7;
double i4_7;
double r4_15;
double i4_15;
r4_7 = ((in[7]).re);
i4_7 = ((in[7]).im);
r4_15 = ((in[15]).re);
i4_15 = ((in[15]).im);
r3_7 = (r4_7 + r4_15);
i3_7 = (i4_7 + i4_15);
r3_15 = (r4_7 - r4_15);
i3_15 = (i4_7 - i4_15);
}
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_11 = (r3_3 - r3_7);
i2_11 = (i3_3 - i3_7);
r2_7 = (r3_11 + i3_15);
i2_7 = (i3_11 - r3_15);
r2_15 = (r3_11 - i3_15);
i2_15 = (i3_11 + r3_15);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_9 = (r2_1 - r2_3);
i1_9 = (i2_1 - i2_3);
tmpr = (0.707106781187 * (r2_7 + i2_7));
tmpi = (0.707106781187 * (i2_7 - r2_7));
r1_3 = (r2_5 + tmpr);
i1_3 = (i2_5 + tmpi);
r1_11 = (r2_5 - tmpr);
i1_11 = (i2_5 - tmpi);
r1_5 = (r2_9 + i2_11);
i1_5 = (i2_9 - r2_11);
r1_13 = (r2_9 - i2_11);
i1_13 = (i2_9 + r2_11);
tmpr = (0.707106781187 * (i2_15 - r2_15));
tmpi = (0.707106781187 * (r2_15 + i2_15));
r1_7 = (r2_13 + tmpr);
i1_7 = (i2_13 - tmpi);
r1_15 = (r2_13 - tmpr);
i1_15 = (i2_13 + tmpi);
}
((out[0]).re) = (r1_0 + r1_1);
((out[0]).im) = (i1_0 + i1_1);
((out[8]).re) = (r1_0 - r1_1);
((out[8]).im) = (i1_0 - i1_1);
tmpr = ((0.923879532511 * r1_3) + (0.382683432365 * i1_3));
tmpi = ((0.923879532511 * i1_3) - (0.382683432365 * r1_3));
((out[1]).re) = (r1_2 + tmpr);
((out[1]).im) = (i1_2 + tmpi);
((out[9]).re) = (r1_2 - tmpr);
((out[9]).im) = (i1_2 - tmpi);
tmpr = (0.707106781187 * (r1_5 + i1_5));
tmpi = (0.707106781187 * (i1_5 - r1_5));
((out[2]).re) = (r1_4 + tmpr);
((out[2]).im) = (i1_4 + tmpi);
((out[10]).re) = (r1_4 - tmpr);
((out[10]).im) = (i1_4 - tmpi);
tmpr = ((0.382683432365 * r1_7) + (0.923879532511 * i1_7));
tmpi = ((0.382683432365 * i1_7) - (0.923879532511 * r1_7));
((out[3]).re) = (r1_6 + tmpr);
((out[3]).im) = (i1_6 + tmpi);
((out[11]).re) = (r1_6 - tmpr);
((out[11]).im) = (i1_6 - tmpi);
((out[4]).re) = (r1_8 + i1_9);
((out[4]).im) = (i1_8 - r1_9);
((out[12]).re) = (r1_8 - i1_9);
((out[12]).im) = (i1_8 + r1_9);
tmpr = ((0.923879532511 * i1_11) - (0.382683432365 * r1_11));
tmpi = ((0.923879532511 * r1_11) + (0.382683432365 * i1_11));
((out[5]).re) = (r1_10 + tmpr);
((out[5]).im) = (i1_10 - tmpi);
((out[13]).re) = (r1_10 - tmpr);
((out[13]).im) = (i1_10 + tmpi);
tmpr = (0.707106781187 * (i1_13 - r1_13));
tmpi = (0.707106781187 * (r1_13 + i1_13));
((out[6]).re) = (r1_12 + tmpr);
((out[6]).im) = (i1_12 - tmpi);
((out[14]).re) = (r1_12 - tmpr);
((out[14]).im) = (i1_12 + tmpi);
tmpr = ((0.382683432365 * i1_15) - (0.923879532511 * r1_15));
tmpi = ((0.382683432365 * r1_15) + (0.923879532511 * i1_15));
((out[7]).re) = (r1_14 + tmpr);
((out[7]).im) = (i1_14 - tmpi);
((out[15]).re) = (r1_14 - tmpr);
((out[15]).im) = (i1_14 + tmpi);
}
}
static void * _taskFunc19_(void *);
static void * _taskFunc20_(void *);
void fft_twiddle_16(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
double r1_8;
double i1_8;
double r1_9;
double i1_9;
double r1_10;
double i1_10;
double r1_11;
double i1_11;
double r1_12;
double i1_12;
double r1_13;
double i1_13;
double r1_14;
double i1_14;
double r1_15;
double i1_15;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
double r2_8;
double i2_8;
double r2_10;
double i2_10;
double r2_12;
double i2_12;
double r2_14;
double i2_14;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
double r3_8;
double i3_8;
double r3_12;
double i3_12;
{
double r4_0;
double i4_0;
double r4_8;
double i4_8;
r4_0 = ((jp[0 * m]).re);
i4_0 = ((jp[0 * m]).im);
wr = ((W[8 * l1]).re);
wi = ((W[8 * l1]).im);
tmpr = ((jp[8 * m]).re);
tmpi = ((jp[8 * m]).im);
r4_8 = ((wr * tmpr) - (wi * tmpi));
i4_8 = ((wi * tmpr) + (wr * tmpi));
r3_0 = (r4_0 + r4_8);
i3_0 = (i4_0 + i4_8);
r3_8 = (r4_0 - r4_8);
i3_8 = (i4_0 - i4_8);
}
{
double r4_4;
double i4_4;
double r4_12;
double i4_12;
wr = ((W[4 * l1]).re);
wi = ((W[4 * l1]).im);
tmpr = ((jp[4 * m]).re);
tmpi = ((jp[4 * m]).im);
r4_4 = ((wr * tmpr) - (wi * tmpi));
i4_4 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[12 * l1]).re);
wi = ((W[12 * l1]).im);
tmpr = ((jp[12 * m]).re);
tmpi = ((jp[12 * m]).im);
r4_12 = ((wr * tmpr) - (wi * tmpi));
i4_12 = ((wi * tmpr) + (wr * tmpi));
r3_4 = (r4_4 + r4_12);
i3_4 = (i4_4 + i4_12);
r3_12 = (r4_4 - r4_12);
i3_12 = (i4_4 - i4_12);
}
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_8 = (r3_0 - r3_4);
i2_8 = (i3_0 - i3_4);
r2_4 = (r3_8 + i3_12);
i2_4 = (i3_8 - r3_12);
r2_12 = (r3_8 - i3_12);
i2_12 = (i3_8 + r3_12);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
double r3_10;
double i3_10;
double r3_14;
double i3_14;
{
double r4_2;
double i4_2;
double r4_10;
double i4_10;
wr = ((W[2 * l1]).re);
wi = ((W[2 * l1]).im);
tmpr = ((jp[2 * m]).re);
tmpi = ((jp[2 * m]).im);
r4_2 = ((wr * tmpr) - (wi * tmpi));
i4_2 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[10 * l1]).re);
wi = ((W[10 * l1]).im);
tmpr = ((jp[10 * m]).re);
tmpi = ((jp[10 * m]).im);
r4_10 = ((wr * tmpr) - (wi * tmpi));
i4_10 = ((wi * tmpr) + (wr * tmpi));
r3_2 = (r4_2 + r4_10);
i3_2 = (i4_2 + i4_10);
r3_10 = (r4_2 - r4_10);
i3_10 = (i4_2 - i4_10);
}
{
double r4_6;
double i4_6;
double r4_14;
double i4_14;
wr = ((W[6 * l1]).re);
wi = ((W[6 * l1]).im);
tmpr = ((jp[6 * m]).re);
tmpi = ((jp[6 * m]).im);
r4_6 = ((wr * tmpr) - (wi * tmpi));
i4_6 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[14 * l1]).re);
wi = ((W[14 * l1]).im);
tmpr = ((jp[14 * m]).re);
tmpi = ((jp[14 * m]).im);
r4_14 = ((wr * tmpr) - (wi * tmpi));
i4_14 = ((wi * tmpr) + (wr * tmpi));
r3_6 = (r4_6 + r4_14);
i3_6 = (i4_6 + i4_14);
r3_14 = (r4_6 - r4_14);
i3_14 = (i4_6 - i4_14);
}
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_10 = (r3_2 - r3_6);
i2_10 = (i3_2 - i3_6);
r2_6 = (r3_10 + i3_14);
i2_6 = (i3_10 - r3_14);
r2_14 = (r3_10 - i3_14);
i2_14 = (i3_10 + r3_14);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_8 = (r2_0 - r2_2);
i1_8 = (i2_0 - i2_2);
tmpr = (0.707106781187 * (r2_6 + i2_6));
tmpi = (0.707106781187 * (i2_6 - r2_6));
r1_2 = (r2_4 + tmpr);
i1_2 = (i2_4 + tmpi);
r1_10 = (r2_4 - tmpr);
i1_10 = (i2_4 - tmpi);
r1_4 = (r2_8 + i2_10);
i1_4 = (i2_8 - r2_10);
r1_12 = (r2_8 - i2_10);
i1_12 = (i2_8 + r2_10);
tmpr = (0.707106781187 * (i2_14 - r2_14));
tmpi = (0.707106781187 * (r2_14 + i2_14));
r1_6 = (r2_12 + tmpr);
i1_6 = (i2_12 - tmpi);
r1_14 = (r2_12 - tmpr);
i1_14 = (i2_12 + tmpi);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
double r2_9;
double i2_9;
double r2_11;
double i2_11;
double r2_13;
double i2_13;
double r2_15;
double i2_15;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
double r3_9;
double i3_9;
double r3_13;
double i3_13;
{
double r4_1;
double i4_1;
double r4_9;
double i4_9;
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r4_1 = ((wr * tmpr) - (wi * tmpi));
i4_1 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[9 * l1]).re);
wi = ((W[9 * l1]).im);
tmpr = ((jp[9 * m]).re);
tmpi = ((jp[9 * m]).im);
r4_9 = ((wr * tmpr) - (wi * tmpi));
i4_9 = ((wi * tmpr) + (wr * tmpi));
r3_1 = (r4_1 + r4_9);
i3_1 = (i4_1 + i4_9);
r3_9 = (r4_1 - r4_9);
i3_9 = (i4_1 - i4_9);
}
{
double r4_5;
double i4_5;
double r4_13;
double i4_13;
wr = ((W[5 * l1]).re);
wi = ((W[5 * l1]).im);
tmpr = ((jp[5 * m]).re);
tmpi = ((jp[5 * m]).im);
r4_5 = ((wr * tmpr) - (wi * tmpi));
i4_5 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[13 * l1]).re);
wi = ((W[13 * l1]).im);
tmpr = ((jp[13 * m]).re);
tmpi = ((jp[13 * m]).im);
r4_13 = ((wr * tmpr) - (wi * tmpi));
i4_13 = ((wi * tmpr) + (wr * tmpi));
r3_5 = (r4_5 + r4_13);
i3_5 = (i4_5 + i4_13);
r3_13 = (r4_5 - r4_13);
i3_13 = (i4_5 - i4_13);
}
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_9 = (r3_1 - r3_5);
i2_9 = (i3_1 - i3_5);
r2_5 = (r3_9 + i3_13);
i2_5 = (i3_9 - r3_13);
r2_13 = (r3_9 - i3_13);
i2_13 = (i3_9 + r3_13);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
double r3_11;
double i3_11;
double r3_15;
double i3_15;
{
double r4_3;
double i4_3;
double r4_11;
double i4_11;
wr = ((W[3 * l1]).re);
wi = ((W[3 * l1]).im);
tmpr = ((jp[3 * m]).re);
tmpi = ((jp[3 * m]).im);
r4_3 = ((wr * tmpr) - (wi * tmpi));
i4_3 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[11 * l1]).re);
wi = ((W[11 * l1]).im);
tmpr = ((jp[11 * m]).re);
tmpi = ((jp[11 * m]).im);
r4_11 = ((wr * tmpr) - (wi * tmpi));
i4_11 = ((wi * tmpr) + (wr * tmpi));
r3_3 = (r4_3 + r4_11);
i3_3 = (i4_3 + i4_11);
r3_11 = (r4_3 - r4_11);
i3_11 = (i4_3 - i4_11);
}
{
double r4_7;
double i4_7;
double r4_15;
double i4_15;
wr = ((W[7 * l1]).re);
wi = ((W[7 * l1]).im);
tmpr = ((jp[7 * m]).re);
tmpi = ((jp[7 * m]).im);
r4_7 = ((wr * tmpr) - (wi * tmpi));
i4_7 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[15 * l1]).re);
wi = ((W[15 * l1]).im);
tmpr = ((jp[15 * m]).re);
tmpi = ((jp[15 * m]).im);
r4_15 = ((wr * tmpr) - (wi * tmpi));
i4_15 = ((wi * tmpr) + (wr * tmpi));
r3_7 = (r4_7 + r4_15);
i3_7 = (i4_7 + i4_15);
r3_15 = (r4_7 - r4_15);
i3_15 = (i4_7 - i4_15);
}
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_11 = (r3_3 - r3_7);
i2_11 = (i3_3 - i3_7);
r2_7 = (r3_11 + i3_15);
i2_7 = (i3_11 - r3_15);
r2_15 = (r3_11 - i3_15);
i2_15 = (i3_11 + r3_15);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_9 = (r2_1 - r2_3);
i1_9 = (i2_1 - i2_3);
tmpr = (0.707106781187 * (r2_7 + i2_7));
tmpi = (0.707106781187 * (i2_7 - r2_7));
r1_3 = (r2_5 + tmpr);
i1_3 = (i2_5 + tmpi);
r1_11 = (r2_5 - tmpr);
i1_11 = (i2_5 - tmpi);
r1_5 = (r2_9 + i2_11);
i1_5 = (i2_9 - r2_11);
r1_13 = (r2_9 - i2_11);
i1_13 = (i2_9 + r2_11);
tmpr = (0.707106781187 * (i2_15 - r2_15));
tmpi = (0.707106781187 * (r2_15 + i2_15));
r1_7 = (r2_13 + tmpr);
i1_7 = (i2_13 - tmpi);
r1_15 = (r2_13 - tmpr);
i1_15 = (i2_13 + tmpi);
}
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[8 * m]).re) = (r1_0 - r1_1);
((kp[8 * m]).im) = (i1_0 - i1_1);
tmpr = ((0.923879532511 * r1_3) + (0.382683432365 * i1_3));
tmpi = ((0.923879532511 * i1_3) - (0.382683432365 * r1_3));
((kp[1 * m]).re) = (r1_2 + tmpr);
((kp[1 * m]).im) = (i1_2 + tmpi);
((kp[9 * m]).re) = (r1_2 - tmpr);
((kp[9 * m]).im) = (i1_2 - tmpi);
tmpr = (0.707106781187 * (r1_5 + i1_5));
tmpi = (0.707106781187 * (i1_5 - r1_5));
((kp[2 * m]).re) = (r1_4 + tmpr);
((kp[2 * m]).im) = (i1_4 + tmpi);
((kp[10 * m]).re) = (r1_4 - tmpr);
((kp[10 * m]).im) = (i1_4 - tmpi);
tmpr = ((0.382683432365 * r1_7) + (0.923879532511 * i1_7));
tmpi = ((0.382683432365 * i1_7) - (0.923879532511 * r1_7));
((kp[3 * m]).re) = (r1_6 + tmpr);
((kp[3 * m]).im) = (i1_6 + tmpi);
((kp[11 * m]).re) = (r1_6 - tmpr);
((kp[11 * m]).im) = (i1_6 - tmpi);
((kp[4 * m]).re) = (r1_8 + i1_9);
((kp[4 * m]).im) = (i1_8 - r1_9);
((kp[12 * m]).re) = (r1_8 - i1_9);
((kp[12 * m]).im) = (i1_8 + r1_9);
tmpr = ((0.923879532511 * i1_11) - (0.382683432365 * r1_11));
tmpi = ((0.923879532511 * r1_11) + (0.382683432365 * i1_11));
((kp[5 * m]).re) = (r1_10 + tmpr);
((kp[5 * m]).im) = (i1_10 - tmpi);
((kp[13 * m]).re) = (r1_10 - tmpr);
((kp[13 * m]).im) = (i1_10 + tmpi);
tmpr = (0.707106781187 * (i1_13 - r1_13));
tmpi = (0.707106781187 * (r1_13 + i1_13));
((kp[6 * m]).re) = (r1_12 + tmpr);
((kp[6 * m]).im) = (i1_12 - tmpi);
((kp[14 * m]).re) = (r1_12 - tmpr);
((kp[14 * m]).im) = (i1_12 + tmpi);
tmpr = ((0.382683432365 * i1_15) - (0.923879532511 * r1_15));
tmpi = ((0.382683432365 * r1_15) + (0.923879532511 * i1_15));
((kp[7 * m]).re) = (r1_14 + tmpr);
((kp[7 * m]).im) = (i1_14 - tmpi);
((kp[15 * m]).re) = (r1_14 - tmpr);
((kp[15 * m]).im) = (i1_14 + tmpi);
}
}
}
else
{
int ab = (a + b) / 2;
_taskFunc19_((void *)0);
_taskFunc20_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc20_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_16(ab, b, in, out, W, nW, nWdn, m);
CANCEL_task_1682 :
;
}
ort_taskenv_free(_tenv, _taskFunc20_);
return ((void *) 0);
}
static void * _taskFunc19_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_16(a, ab, in, out, W, nW, nWdn, m);
CANCEL_task_1680 :
;
}
ort_taskenv_free(_tenv, _taskFunc19_);
return ((void *) 0);
}
void fft_twiddle_16_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
double r1_8;
double i1_8;
double r1_9;
double i1_9;
double r1_10;
double i1_10;
double r1_11;
double i1_11;
double r1_12;
double i1_12;
double r1_13;
double i1_13;
double r1_14;
double i1_14;
double r1_15;
double i1_15;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
double r2_8;
double i2_8;
double r2_10;
double i2_10;
double r2_12;
double i2_12;
double r2_14;
double i2_14;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
double r3_8;
double i3_8;
double r3_12;
double i3_12;
{
double r4_0;
double i4_0;
double r4_8;
double i4_8;
r4_0 = ((jp[0 * m]).re);
i4_0 = ((jp[0 * m]).im);
wr = ((W[8 * l1]).re);
wi = ((W[8 * l1]).im);
tmpr = ((jp[8 * m]).re);
tmpi = ((jp[8 * m]).im);
r4_8 = ((wr * tmpr) - (wi * tmpi));
i4_8 = ((wi * tmpr) + (wr * tmpi));
r3_0 = (r4_0 + r4_8);
i3_0 = (i4_0 + i4_8);
r3_8 = (r4_0 - r4_8);
i3_8 = (i4_0 - i4_8);
}
{
double r4_4;
double i4_4;
double r4_12;
double i4_12;
wr = ((W[4 * l1]).re);
wi = ((W[4 * l1]).im);
tmpr = ((jp[4 * m]).re);
tmpi = ((jp[4 * m]).im);
r4_4 = ((wr * tmpr) - (wi * tmpi));
i4_4 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[12 * l1]).re);
wi = ((W[12 * l1]).im);
tmpr = ((jp[12 * m]).re);
tmpi = ((jp[12 * m]).im);
r4_12 = ((wr * tmpr) - (wi * tmpi));
i4_12 = ((wi * tmpr) + (wr * tmpi));
r3_4 = (r4_4 + r4_12);
i3_4 = (i4_4 + i4_12);
r3_12 = (r4_4 - r4_12);
i3_12 = (i4_4 - i4_12);
}
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_8 = (r3_0 - r3_4);
i2_8 = (i3_0 - i3_4);
r2_4 = (r3_8 + i3_12);
i2_4 = (i3_8 - r3_12);
r2_12 = (r3_8 - i3_12);
i2_12 = (i3_8 + r3_12);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
double r3_10;
double i3_10;
double r3_14;
double i3_14;
{
double r4_2;
double i4_2;
double r4_10;
double i4_10;
wr = ((W[2 * l1]).re);
wi = ((W[2 * l1]).im);
tmpr = ((jp[2 * m]).re);
tmpi = ((jp[2 * m]).im);
r4_2 = ((wr * tmpr) - (wi * tmpi));
i4_2 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[10 * l1]).re);
wi = ((W[10 * l1]).im);
tmpr = ((jp[10 * m]).re);
tmpi = ((jp[10 * m]).im);
r4_10 = ((wr * tmpr) - (wi * tmpi));
i4_10 = ((wi * tmpr) + (wr * tmpi));
r3_2 = (r4_2 + r4_10);
i3_2 = (i4_2 + i4_10);
r3_10 = (r4_2 - r4_10);
i3_10 = (i4_2 - i4_10);
}
{
double r4_6;
double i4_6;
double r4_14;
double i4_14;
wr = ((W[6 * l1]).re);
wi = ((W[6 * l1]).im);
tmpr = ((jp[6 * m]).re);
tmpi = ((jp[6 * m]).im);
r4_6 = ((wr * tmpr) - (wi * tmpi));
i4_6 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[14 * l1]).re);
wi = ((W[14 * l1]).im);
tmpr = ((jp[14 * m]).re);
tmpi = ((jp[14 * m]).im);
r4_14 = ((wr * tmpr) - (wi * tmpi));
i4_14 = ((wi * tmpr) + (wr * tmpi));
r3_6 = (r4_6 + r4_14);
i3_6 = (i4_6 + i4_14);
r3_14 = (r4_6 - r4_14);
i3_14 = (i4_6 - i4_14);
}
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_10 = (r3_2 - r3_6);
i2_10 = (i3_2 - i3_6);
r2_6 = (r3_10 + i3_14);
i2_6 = (i3_10 - r3_14);
r2_14 = (r3_10 - i3_14);
i2_14 = (i3_10 + r3_14);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_8 = (r2_0 - r2_2);
i1_8 = (i2_0 - i2_2);
tmpr = (0.707106781187 * (r2_6 + i2_6));
tmpi = (0.707106781187 * (i2_6 - r2_6));
r1_2 = (r2_4 + tmpr);
i1_2 = (i2_4 + tmpi);
r1_10 = (r2_4 - tmpr);
i1_10 = (i2_4 - tmpi);
r1_4 = (r2_8 + i2_10);
i1_4 = (i2_8 - r2_10);
r1_12 = (r2_8 - i2_10);
i1_12 = (i2_8 + r2_10);
tmpr = (0.707106781187 * (i2_14 - r2_14));
tmpi = (0.707106781187 * (r2_14 + i2_14));
r1_6 = (r2_12 + tmpr);
i1_6 = (i2_12 - tmpi);
r1_14 = (r2_12 - tmpr);
i1_14 = (i2_12 + tmpi);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
double r2_9;
double i2_9;
double r2_11;
double i2_11;
double r2_13;
double i2_13;
double r2_15;
double i2_15;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
double r3_9;
double i3_9;
double r3_13;
double i3_13;
{
double r4_1;
double i4_1;
double r4_9;
double i4_9;
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r4_1 = ((wr * tmpr) - (wi * tmpi));
i4_1 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[9 * l1]).re);
wi = ((W[9 * l1]).im);
tmpr = ((jp[9 * m]).re);
tmpi = ((jp[9 * m]).im);
r4_9 = ((wr * tmpr) - (wi * tmpi));
i4_9 = ((wi * tmpr) + (wr * tmpi));
r3_1 = (r4_1 + r4_9);
i3_1 = (i4_1 + i4_9);
r3_9 = (r4_1 - r4_9);
i3_9 = (i4_1 - i4_9);
}
{
double r4_5;
double i4_5;
double r4_13;
double i4_13;
wr = ((W[5 * l1]).re);
wi = ((W[5 * l1]).im);
tmpr = ((jp[5 * m]).re);
tmpi = ((jp[5 * m]).im);
r4_5 = ((wr * tmpr) - (wi * tmpi));
i4_5 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[13 * l1]).re);
wi = ((W[13 * l1]).im);
tmpr = ((jp[13 * m]).re);
tmpi = ((jp[13 * m]).im);
r4_13 = ((wr * tmpr) - (wi * tmpi));
i4_13 = ((wi * tmpr) + (wr * tmpi));
r3_5 = (r4_5 + r4_13);
i3_5 = (i4_5 + i4_13);
r3_13 = (r4_5 - r4_13);
i3_13 = (i4_5 - i4_13);
}
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_9 = (r3_1 - r3_5);
i2_9 = (i3_1 - i3_5);
r2_5 = (r3_9 + i3_13);
i2_5 = (i3_9 - r3_13);
r2_13 = (r3_9 - i3_13);
i2_13 = (i3_9 + r3_13);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
double r3_11;
double i3_11;
double r3_15;
double i3_15;
{
double r4_3;
double i4_3;
double r4_11;
double i4_11;
wr = ((W[3 * l1]).re);
wi = ((W[3 * l1]).im);
tmpr = ((jp[3 * m]).re);
tmpi = ((jp[3 * m]).im);
r4_3 = ((wr * tmpr) - (wi * tmpi));
i4_3 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[11 * l1]).re);
wi = ((W[11 * l1]).im);
tmpr = ((jp[11 * m]).re);
tmpi = ((jp[11 * m]).im);
r4_11 = ((wr * tmpr) - (wi * tmpi));
i4_11 = ((wi * tmpr) + (wr * tmpi));
r3_3 = (r4_3 + r4_11);
i3_3 = (i4_3 + i4_11);
r3_11 = (r4_3 - r4_11);
i3_11 = (i4_3 - i4_11);
}
{
double r4_7;
double i4_7;
double r4_15;
double i4_15;
wr = ((W[7 * l1]).re);
wi = ((W[7 * l1]).im);
tmpr = ((jp[7 * m]).re);
tmpi = ((jp[7 * m]).im);
r4_7 = ((wr * tmpr) - (wi * tmpi));
i4_7 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[15 * l1]).re);
wi = ((W[15 * l1]).im);
tmpr = ((jp[15 * m]).re);
tmpi = ((jp[15 * m]).im);
r4_15 = ((wr * tmpr) - (wi * tmpi));
i4_15 = ((wi * tmpr) + (wr * tmpi));
r3_7 = (r4_7 + r4_15);
i3_7 = (i4_7 + i4_15);
r3_15 = (r4_7 - r4_15);
i3_15 = (i4_7 - i4_15);
}
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_11 = (r3_3 - r3_7);
i2_11 = (i3_3 - i3_7);
r2_7 = (r3_11 + i3_15);
i2_7 = (i3_11 - r3_15);
r2_15 = (r3_11 - i3_15);
i2_15 = (i3_11 + r3_15);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_9 = (r2_1 - r2_3);
i1_9 = (i2_1 - i2_3);
tmpr = (0.707106781187 * (r2_7 + i2_7));
tmpi = (0.707106781187 * (i2_7 - r2_7));
r1_3 = (r2_5 + tmpr);
i1_3 = (i2_5 + tmpi);
r1_11 = (r2_5 - tmpr);
i1_11 = (i2_5 - tmpi);
r1_5 = (r2_9 + i2_11);
i1_5 = (i2_9 - r2_11);
r1_13 = (r2_9 - i2_11);
i1_13 = (i2_9 + r2_11);
tmpr = (0.707106781187 * (i2_15 - r2_15));
tmpi = (0.707106781187 * (r2_15 + i2_15));
r1_7 = (r2_13 + tmpr);
i1_7 = (i2_13 - tmpi);
r1_15 = (r2_13 - tmpr);
i1_15 = (i2_13 + tmpi);
}
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[8 * m]).re) = (r1_0 - r1_1);
((kp[8 * m]).im) = (i1_0 - i1_1);
tmpr = ((0.923879532511 * r1_3) + (0.382683432365 * i1_3));
tmpi = ((0.923879532511 * i1_3) - (0.382683432365 * r1_3));
((kp[1 * m]).re) = (r1_2 + tmpr);
((kp[1 * m]).im) = (i1_2 + tmpi);
((kp[9 * m]).re) = (r1_2 - tmpr);
((kp[9 * m]).im) = (i1_2 - tmpi);
tmpr = (0.707106781187 * (r1_5 + i1_5));
tmpi = (0.707106781187 * (i1_5 - r1_5));
((kp[2 * m]).re) = (r1_4 + tmpr);
((kp[2 * m]).im) = (i1_4 + tmpi);
((kp[10 * m]).re) = (r1_4 - tmpr);
((kp[10 * m]).im) = (i1_4 - tmpi);
tmpr = ((0.382683432365 * r1_7) + (0.923879532511 * i1_7));
tmpi = ((0.382683432365 * i1_7) - (0.923879532511 * r1_7));
((kp[3 * m]).re) = (r1_6 + tmpr);
((kp[3 * m]).im) = (i1_6 + tmpi);
((kp[11 * m]).re) = (r1_6 - tmpr);
((kp[11 * m]).im) = (i1_6 - tmpi);
((kp[4 * m]).re) = (r1_8 + i1_9);
((kp[4 * m]).im) = (i1_8 - r1_9);
((kp[12 * m]).re) = (r1_8 - i1_9);
((kp[12 * m]).im) = (i1_8 + r1_9);
tmpr = ((0.923879532511 * i1_11) - (0.382683432365 * r1_11));
tmpi = ((0.923879532511 * r1_11) + (0.382683432365 * i1_11));
((kp[5 * m]).re) = (r1_10 + tmpr);
((kp[5 * m]).im) = (i1_10 - tmpi);
((kp[13 * m]).re) = (r1_10 - tmpr);
((kp[13 * m]).im) = (i1_10 + tmpi);
tmpr = (0.707106781187 * (i1_13 - r1_13));
tmpi = (0.707106781187 * (r1_13 + i1_13));
((kp[6 * m]).re) = (r1_12 + tmpr);
((kp[6 * m]).im) = (i1_12 - tmpi);
((kp[14 * m]).re) = (r1_12 - tmpr);
((kp[14 * m]).im) = (i1_12 + tmpi);
tmpr = ((0.382683432365 * i1_15) - (0.923879532511 * r1_15));
tmpi = ((0.382683432365 * r1_15) + (0.923879532511 * i1_15));
((kp[7 * m]).re) = (r1_14 + tmpr);
((kp[7 * m]).im) = (i1_14 - tmpi);
((kp[15 * m]).re) = (r1_14 - tmpr);
((kp[15 * m]).im) = (i1_14 + tmpi);
}
}
}
else
{
int ab = (a + b) / 2;
fft_twiddle_16_seq(a, ab, in, out, W, nW, nWdn, m);
fft_twiddle_16_seq(ab, b, in, out, W, nW, nWdn, m);
}
}
static void * _taskFunc21_(void *);
static void * _taskFunc22_(void *);
void fft_unshuffle_16(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 16;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
_taskFunc21_((void *)0);
_taskFunc22_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc22_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_16(ab, b, in, out, m);
CANCEL_task_2082 :
;
}
ort_taskenv_free(_tenv, _taskFunc22_);
return ((void *) 0);
}
static void * _taskFunc21_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_16(a, ab, in, out, m);
CANCEL_task_2080 :
;
}
ort_taskenv_free(_tenv, _taskFunc21_);
return ((void *) 0);
}
void fft_unshuffle_16_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 16;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
fft_unshuffle_16_seq(a, ab, in, out, m);
fft_unshuffle_16_seq(ab, b, in, out, m);
}
}
void fft_base_32(struct _noname19_ (* in), struct _noname19_ (* out))
{
double tmpr;
double tmpi;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
double r1_8;
double i1_8;
double r1_9;
double i1_9;
double r1_10;
double i1_10;
double r1_11;
double i1_11;
double r1_12;
double i1_12;
double r1_13;
double i1_13;
double r1_14;
double i1_14;
double r1_15;
double i1_15;
double r1_16;
double i1_16;
double r1_17;
double i1_17;
double r1_18;
double i1_18;
double r1_19;
double i1_19;
double r1_20;
double i1_20;
double r1_21;
double i1_21;
double r1_22;
double i1_22;
double r1_23;
double i1_23;
double r1_24;
double i1_24;
double r1_25;
double i1_25;
double r1_26;
double i1_26;
double r1_27;
double i1_27;
double r1_28;
double i1_28;
double r1_29;
double i1_29;
double r1_30;
double i1_30;
double r1_31;
double i1_31;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
double r2_8;
double i2_8;
double r2_10;
double i2_10;
double r2_12;
double i2_12;
double r2_14;
double i2_14;
double r2_16;
double i2_16;
double r2_18;
double i2_18;
double r2_20;
double i2_20;
double r2_22;
double i2_22;
double r2_24;
double i2_24;
double r2_26;
double i2_26;
double r2_28;
double i2_28;
double r2_30;
double i2_30;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
double r3_8;
double i3_8;
double r3_12;
double i3_12;
double r3_16;
double i3_16;
double r3_20;
double i3_20;
double r3_24;
double i3_24;
double r3_28;
double i3_28;
{
double r4_0;
double i4_0;
double r4_8;
double i4_8;
double r4_16;
double i4_16;
double r4_24;
double i4_24;
{
double r5_0;
double i5_0;
double r5_16;
double i5_16;
r5_0 = ((in[0]).re);
i5_0 = ((in[0]).im);
r5_16 = ((in[16]).re);
i5_16 = ((in[16]).im);
r4_0 = (r5_0 + r5_16);
i4_0 = (i5_0 + i5_16);
r4_16 = (r5_0 - r5_16);
i4_16 = (i5_0 - i5_16);
}
{
double r5_8;
double i5_8;
double r5_24;
double i5_24;
r5_8 = ((in[8]).re);
i5_8 = ((in[8]).im);
r5_24 = ((in[24]).re);
i5_24 = ((in[24]).im);
r4_8 = (r5_8 + r5_24);
i4_8 = (i5_8 + i5_24);
r4_24 = (r5_8 - r5_24);
i4_24 = (i5_8 - i5_24);
}
r3_0 = (r4_0 + r4_8);
i3_0 = (i4_0 + i4_8);
r3_16 = (r4_0 - r4_8);
i3_16 = (i4_0 - i4_8);
r3_8 = (r4_16 + i4_24);
i3_8 = (i4_16 - r4_24);
r3_24 = (r4_16 - i4_24);
i3_24 = (i4_16 + r4_24);
}
{
double r4_4;
double i4_4;
double r4_12;
double i4_12;
double r4_20;
double i4_20;
double r4_28;
double i4_28;
{
double r5_4;
double i5_4;
double r5_20;
double i5_20;
r5_4 = ((in[4]).re);
i5_4 = ((in[4]).im);
r5_20 = ((in[20]).re);
i5_20 = ((in[20]).im);
r4_4 = (r5_4 + r5_20);
i4_4 = (i5_4 + i5_20);
r4_20 = (r5_4 - r5_20);
i4_20 = (i5_4 - i5_20);
}
{
double r5_12;
double i5_12;
double r5_28;
double i5_28;
r5_12 = ((in[12]).re);
i5_12 = ((in[12]).im);
r5_28 = ((in[28]).re);
i5_28 = ((in[28]).im);
r4_12 = (r5_12 + r5_28);
i4_12 = (i5_12 + i5_28);
r4_28 = (r5_12 - r5_28);
i4_28 = (i5_12 - i5_28);
}
r3_4 = (r4_4 + r4_12);
i3_4 = (i4_4 + i4_12);
r3_20 = (r4_4 - r4_12);
i3_20 = (i4_4 - i4_12);
r3_12 = (r4_20 + i4_28);
i3_12 = (i4_20 - r4_28);
r3_28 = (r4_20 - i4_28);
i3_28 = (i4_20 + r4_28);
}
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_16 = (r3_0 - r3_4);
i2_16 = (i3_0 - i3_4);
tmpr = (0.707106781187 * (r3_12 + i3_12));
tmpi = (0.707106781187 * (i3_12 - r3_12));
r2_4 = (r3_8 + tmpr);
i2_4 = (i3_8 + tmpi);
r2_20 = (r3_8 - tmpr);
i2_20 = (i3_8 - tmpi);
r2_8 = (r3_16 + i3_20);
i2_8 = (i3_16 - r3_20);
r2_24 = (r3_16 - i3_20);
i2_24 = (i3_16 + r3_20);
tmpr = (0.707106781187 * (i3_28 - r3_28));
tmpi = (0.707106781187 * (r3_28 + i3_28));
r2_12 = (r3_24 + tmpr);
i2_12 = (i3_24 - tmpi);
r2_28 = (r3_24 - tmpr);
i2_28 = (i3_24 + tmpi);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
double r3_10;
double i3_10;
double r3_14;
double i3_14;
double r3_18;
double i3_18;
double r3_22;
double i3_22;
double r3_26;
double i3_26;
double r3_30;
double i3_30;
{
double r4_2;
double i4_2;
double r4_10;
double i4_10;
double r4_18;
double i4_18;
double r4_26;
double i4_26;
{
double r5_2;
double i5_2;
double r5_18;
double i5_18;
r5_2 = ((in[2]).re);
i5_2 = ((in[2]).im);
r5_18 = ((in[18]).re);
i5_18 = ((in[18]).im);
r4_2 = (r5_2 + r5_18);
i4_2 = (i5_2 + i5_18);
r4_18 = (r5_2 - r5_18);
i4_18 = (i5_2 - i5_18);
}
{
double r5_10;
double i5_10;
double r5_26;
double i5_26;
r5_10 = ((in[10]).re);
i5_10 = ((in[10]).im);
r5_26 = ((in[26]).re);
i5_26 = ((in[26]).im);
r4_10 = (r5_10 + r5_26);
i4_10 = (i5_10 + i5_26);
r4_26 = (r5_10 - r5_26);
i4_26 = (i5_10 - i5_26);
}
r3_2 = (r4_2 + r4_10);
i3_2 = (i4_2 + i4_10);
r3_18 = (r4_2 - r4_10);
i3_18 = (i4_2 - i4_10);
r3_10 = (r4_18 + i4_26);
i3_10 = (i4_18 - r4_26);
r3_26 = (r4_18 - i4_26);
i3_26 = (i4_18 + r4_26);
}
{
double r4_6;
double i4_6;
double r4_14;
double i4_14;
double r4_22;
double i4_22;
double r4_30;
double i4_30;
{
double r5_6;
double i5_6;
double r5_22;
double i5_22;
r5_6 = ((in[6]).re);
i5_6 = ((in[6]).im);
r5_22 = ((in[22]).re);
i5_22 = ((in[22]).im);
r4_6 = (r5_6 + r5_22);
i4_6 = (i5_6 + i5_22);
r4_22 = (r5_6 - r5_22);
i4_22 = (i5_6 - i5_22);
}
{
double r5_14;
double i5_14;
double r5_30;
double i5_30;
r5_14 = ((in[14]).re);
i5_14 = ((in[14]).im);
r5_30 = ((in[30]).re);
i5_30 = ((in[30]).im);
r4_14 = (r5_14 + r5_30);
i4_14 = (i5_14 + i5_30);
r4_30 = (r5_14 - r5_30);
i4_30 = (i5_14 - i5_30);
}
r3_6 = (r4_6 + r4_14);
i3_6 = (i4_6 + i4_14);
r3_22 = (r4_6 - r4_14);
i3_22 = (i4_6 - i4_14);
r3_14 = (r4_22 + i4_30);
i3_14 = (i4_22 - r4_30);
r3_30 = (r4_22 - i4_30);
i3_30 = (i4_22 + r4_30);
}
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_18 = (r3_2 - r3_6);
i2_18 = (i3_2 - i3_6);
tmpr = (0.707106781187 * (r3_14 + i3_14));
tmpi = (0.707106781187 * (i3_14 - r3_14));
r2_6 = (r3_10 + tmpr);
i2_6 = (i3_10 + tmpi);
r2_22 = (r3_10 - tmpr);
i2_22 = (i3_10 - tmpi);
r2_10 = (r3_18 + i3_22);
i2_10 = (i3_18 - r3_22);
r2_26 = (r3_18 - i3_22);
i2_26 = (i3_18 + r3_22);
tmpr = (0.707106781187 * (i3_30 - r3_30));
tmpi = (0.707106781187 * (r3_30 + i3_30));
r2_14 = (r3_26 + tmpr);
i2_14 = (i3_26 - tmpi);
r2_30 = (r3_26 - tmpr);
i2_30 = (i3_26 + tmpi);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_16 = (r2_0 - r2_2);
i1_16 = (i2_0 - i2_2);
tmpr = ((0.923879532511 * r2_6) + (0.382683432365 * i2_6));
tmpi = ((0.923879532511 * i2_6) - (0.382683432365 * r2_6));
r1_2 = (r2_4 + tmpr);
i1_2 = (i2_4 + tmpi);
r1_18 = (r2_4 - tmpr);
i1_18 = (i2_4 - tmpi);
tmpr = (0.707106781187 * (r2_10 + i2_10));
tmpi = (0.707106781187 * (i2_10 - r2_10));
r1_4 = (r2_8 + tmpr);
i1_4 = (i2_8 + tmpi);
r1_20 = (r2_8 - tmpr);
i1_20 = (i2_8 - tmpi);
tmpr = ((0.382683432365 * r2_14) + (0.923879532511 * i2_14));
tmpi = ((0.382683432365 * i2_14) - (0.923879532511 * r2_14));
r1_6 = (r2_12 + tmpr);
i1_6 = (i2_12 + tmpi);
r1_22 = (r2_12 - tmpr);
i1_22 = (i2_12 - tmpi);
r1_8 = (r2_16 + i2_18);
i1_8 = (i2_16 - r2_18);
r1_24 = (r2_16 - i2_18);
i1_24 = (i2_16 + r2_18);
tmpr = ((0.923879532511 * i2_22) - (0.382683432365 * r2_22));
tmpi = ((0.923879532511 * r2_22) + (0.382683432365 * i2_22));
r1_10 = (r2_20 + tmpr);
i1_10 = (i2_20 - tmpi);
r1_26 = (r2_20 - tmpr);
i1_26 = (i2_20 + tmpi);
tmpr = (0.707106781187 * (i2_26 - r2_26));
tmpi = (0.707106781187 * (r2_26 + i2_26));
r1_12 = (r2_24 + tmpr);
i1_12 = (i2_24 - tmpi);
r1_28 = (r2_24 - tmpr);
i1_28 = (i2_24 + tmpi);
tmpr = ((0.382683432365 * i2_30) - (0.923879532511 * r2_30));
tmpi = ((0.382683432365 * r2_30) + (0.923879532511 * i2_30));
r1_14 = (r2_28 + tmpr);
i1_14 = (i2_28 - tmpi);
r1_30 = (r2_28 - tmpr);
i1_30 = (i2_28 + tmpi);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
double r2_9;
double i2_9;
double r2_11;
double i2_11;
double r2_13;
double i2_13;
double r2_15;
double i2_15;
double r2_17;
double i2_17;
double r2_19;
double i2_19;
double r2_21;
double i2_21;
double r2_23;
double i2_23;
double r2_25;
double i2_25;
double r2_27;
double i2_27;
double r2_29;
double i2_29;
double r2_31;
double i2_31;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
double r3_9;
double i3_9;
double r3_13;
double i3_13;
double r3_17;
double i3_17;
double r3_21;
double i3_21;
double r3_25;
double i3_25;
double r3_29;
double i3_29;
{
double r4_1;
double i4_1;
double r4_9;
double i4_9;
double r4_17;
double i4_17;
double r4_25;
double i4_25;
{
double r5_1;
double i5_1;
double r5_17;
double i5_17;
r5_1 = ((in[1]).re);
i5_1 = ((in[1]).im);
r5_17 = ((in[17]).re);
i5_17 = ((in[17]).im);
r4_1 = (r5_1 + r5_17);
i4_1 = (i5_1 + i5_17);
r4_17 = (r5_1 - r5_17);
i4_17 = (i5_1 - i5_17);
}
{
double r5_9;
double i5_9;
double r5_25;
double i5_25;
r5_9 = ((in[9]).re);
i5_9 = ((in[9]).im);
r5_25 = ((in[25]).re);
i5_25 = ((in[25]).im);
r4_9 = (r5_9 + r5_25);
i4_9 = (i5_9 + i5_25);
r4_25 = (r5_9 - r5_25);
i4_25 = (i5_9 - i5_25);
}
r3_1 = (r4_1 + r4_9);
i3_1 = (i4_1 + i4_9);
r3_17 = (r4_1 - r4_9);
i3_17 = (i4_1 - i4_9);
r3_9 = (r4_17 + i4_25);
i3_9 = (i4_17 - r4_25);
r3_25 = (r4_17 - i4_25);
i3_25 = (i4_17 + r4_25);
}
{
double r4_5;
double i4_5;
double r4_13;
double i4_13;
double r4_21;
double i4_21;
double r4_29;
double i4_29;
{
double r5_5;
double i5_5;
double r5_21;
double i5_21;
r5_5 = ((in[5]).re);
i5_5 = ((in[5]).im);
r5_21 = ((in[21]).re);
i5_21 = ((in[21]).im);
r4_5 = (r5_5 + r5_21);
i4_5 = (i5_5 + i5_21);
r4_21 = (r5_5 - r5_21);
i4_21 = (i5_5 - i5_21);
}
{
double r5_13;
double i5_13;
double r5_29;
double i5_29;
r5_13 = ((in[13]).re);
i5_13 = ((in[13]).im);
r5_29 = ((in[29]).re);
i5_29 = ((in[29]).im);
r4_13 = (r5_13 + r5_29);
i4_13 = (i5_13 + i5_29);
r4_29 = (r5_13 - r5_29);
i4_29 = (i5_13 - i5_29);
}
r3_5 = (r4_5 + r4_13);
i3_5 = (i4_5 + i4_13);
r3_21 = (r4_5 - r4_13);
i3_21 = (i4_5 - i4_13);
r3_13 = (r4_21 + i4_29);
i3_13 = (i4_21 - r4_29);
r3_29 = (r4_21 - i4_29);
i3_29 = (i4_21 + r4_29);
}
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_17 = (r3_1 - r3_5);
i2_17 = (i3_1 - i3_5);
tmpr = (0.707106781187 * (r3_13 + i3_13));
tmpi = (0.707106781187 * (i3_13 - r3_13));
r2_5 = (r3_9 + tmpr);
i2_5 = (i3_9 + tmpi);
r2_21 = (r3_9 - tmpr);
i2_21 = (i3_9 - tmpi);
r2_9 = (r3_17 + i3_21);
i2_9 = (i3_17 - r3_21);
r2_25 = (r3_17 - i3_21);
i2_25 = (i3_17 + r3_21);
tmpr = (0.707106781187 * (i3_29 - r3_29));
tmpi = (0.707106781187 * (r3_29 + i3_29));
r2_13 = (r3_25 + tmpr);
i2_13 = (i3_25 - tmpi);
r2_29 = (r3_25 - tmpr);
i2_29 = (i3_25 + tmpi);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
double r3_11;
double i3_11;
double r3_15;
double i3_15;
double r3_19;
double i3_19;
double r3_23;
double i3_23;
double r3_27;
double i3_27;
double r3_31;
double i3_31;
{
double r4_3;
double i4_3;
double r4_11;
double i4_11;
double r4_19;
double i4_19;
double r4_27;
double i4_27;
{
double r5_3;
double i5_3;
double r5_19;
double i5_19;
r5_3 = ((in[3]).re);
i5_3 = ((in[3]).im);
r5_19 = ((in[19]).re);
i5_19 = ((in[19]).im);
r4_3 = (r5_3 + r5_19);
i4_3 = (i5_3 + i5_19);
r4_19 = (r5_3 - r5_19);
i4_19 = (i5_3 - i5_19);
}
{
double r5_11;
double i5_11;
double r5_27;
double i5_27;
r5_11 = ((in[11]).re);
i5_11 = ((in[11]).im);
r5_27 = ((in[27]).re);
i5_27 = ((in[27]).im);
r4_11 = (r5_11 + r5_27);
i4_11 = (i5_11 + i5_27);
r4_27 = (r5_11 - r5_27);
i4_27 = (i5_11 - i5_27);
}
r3_3 = (r4_3 + r4_11);
i3_3 = (i4_3 + i4_11);
r3_19 = (r4_3 - r4_11);
i3_19 = (i4_3 - i4_11);
r3_11 = (r4_19 + i4_27);
i3_11 = (i4_19 - r4_27);
r3_27 = (r4_19 - i4_27);
i3_27 = (i4_19 + r4_27);
}
{
double r4_7;
double i4_7;
double r4_15;
double i4_15;
double r4_23;
double i4_23;
double r4_31;
double i4_31;
{
double r5_7;
double i5_7;
double r5_23;
double i5_23;
r5_7 = ((in[7]).re);
i5_7 = ((in[7]).im);
r5_23 = ((in[23]).re);
i5_23 = ((in[23]).im);
r4_7 = (r5_7 + r5_23);
i4_7 = (i5_7 + i5_23);
r4_23 = (r5_7 - r5_23);
i4_23 = (i5_7 - i5_23);
}
{
double r5_15;
double i5_15;
double r5_31;
double i5_31;
r5_15 = ((in[15]).re);
i5_15 = ((in[15]).im);
r5_31 = ((in[31]).re);
i5_31 = ((in[31]).im);
r4_15 = (r5_15 + r5_31);
i4_15 = (i5_15 + i5_31);
r4_31 = (r5_15 - r5_31);
i4_31 = (i5_15 - i5_31);
}
r3_7 = (r4_7 + r4_15);
i3_7 = (i4_7 + i4_15);
r3_23 = (r4_7 - r4_15);
i3_23 = (i4_7 - i4_15);
r3_15 = (r4_23 + i4_31);
i3_15 = (i4_23 - r4_31);
r3_31 = (r4_23 - i4_31);
i3_31 = (i4_23 + r4_31);
}
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_19 = (r3_3 - r3_7);
i2_19 = (i3_3 - i3_7);
tmpr = (0.707106781187 * (r3_15 + i3_15));
tmpi = (0.707106781187 * (i3_15 - r3_15));
r2_7 = (r3_11 + tmpr);
i2_7 = (i3_11 + tmpi);
r2_23 = (r3_11 - tmpr);
i2_23 = (i3_11 - tmpi);
r2_11 = (r3_19 + i3_23);
i2_11 = (i3_19 - r3_23);
r2_27 = (r3_19 - i3_23);
i2_27 = (i3_19 + r3_23);
tmpr = (0.707106781187 * (i3_31 - r3_31));
tmpi = (0.707106781187 * (r3_31 + i3_31));
r2_15 = (r3_27 + tmpr);
i2_15 = (i3_27 - tmpi);
r2_31 = (r3_27 - tmpr);
i2_31 = (i3_27 + tmpi);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_17 = (r2_1 - r2_3);
i1_17 = (i2_1 - i2_3);
tmpr = ((0.923879532511 * r2_7) + (0.382683432365 * i2_7));
tmpi = ((0.923879532511 * i2_7) - (0.382683432365 * r2_7));
r1_3 = (r2_5 + tmpr);
i1_3 = (i2_5 + tmpi);
r1_19 = (r2_5 - tmpr);
i1_19 = (i2_5 - tmpi);
tmpr = (0.707106781187 * (r2_11 + i2_11));
tmpi = (0.707106781187 * (i2_11 - r2_11));
r1_5 = (r2_9 + tmpr);
i1_5 = (i2_9 + tmpi);
r1_21 = (r2_9 - tmpr);
i1_21 = (i2_9 - tmpi);
tmpr = ((0.382683432365 * r2_15) + (0.923879532511 * i2_15));
tmpi = ((0.382683432365 * i2_15) - (0.923879532511 * r2_15));
r1_7 = (r2_13 + tmpr);
i1_7 = (i2_13 + tmpi);
r1_23 = (r2_13 - tmpr);
i1_23 = (i2_13 - tmpi);
r1_9 = (r2_17 + i2_19);
i1_9 = (i2_17 - r2_19);
r1_25 = (r2_17 - i2_19);
i1_25 = (i2_17 + r2_19);
tmpr = ((0.923879532511 * i2_23) - (0.382683432365 * r2_23));
tmpi = ((0.923879532511 * r2_23) + (0.382683432365 * i2_23));
r1_11 = (r2_21 + tmpr);
i1_11 = (i2_21 - tmpi);
r1_27 = (r2_21 - tmpr);
i1_27 = (i2_21 + tmpi);
tmpr = (0.707106781187 * (i2_27 - r2_27));
tmpi = (0.707106781187 * (r2_27 + i2_27));
r1_13 = (r2_25 + tmpr);
i1_13 = (i2_25 - tmpi);
r1_29 = (r2_25 - tmpr);
i1_29 = (i2_25 + tmpi);
tmpr = ((0.382683432365 * i2_31) - (0.923879532511 * r2_31));
tmpi = ((0.382683432365 * r2_31) + (0.923879532511 * i2_31));
r1_15 = (r2_29 + tmpr);
i1_15 = (i2_29 - tmpi);
r1_31 = (r2_29 - tmpr);
i1_31 = (i2_29 + tmpi);
}
((out[0]).re) = (r1_0 + r1_1);
((out[0]).im) = (i1_0 + i1_1);
((out[16]).re) = (r1_0 - r1_1);
((out[16]).im) = (i1_0 - i1_1);
tmpr = ((0.980785280403 * r1_3) + (0.195090322016 * i1_3));
tmpi = ((0.980785280403 * i1_3) - (0.195090322016 * r1_3));
((out[1]).re) = (r1_2 + tmpr);
((out[1]).im) = (i1_2 + tmpi);
((out[17]).re) = (r1_2 - tmpr);
((out[17]).im) = (i1_2 - tmpi);
tmpr = ((0.923879532511 * r1_5) + (0.382683432365 * i1_5));
tmpi = ((0.923879532511 * i1_5) - (0.382683432365 * r1_5));
((out[2]).re) = (r1_4 + tmpr);
((out[2]).im) = (i1_4 + tmpi);
((out[18]).re) = (r1_4 - tmpr);
((out[18]).im) = (i1_4 - tmpi);
tmpr = ((0.831469612303 * r1_7) + (0.55557023302 * i1_7));
tmpi = ((0.831469612303 * i1_7) - (0.55557023302 * r1_7));
((out[3]).re) = (r1_6 + tmpr);
((out[3]).im) = (i1_6 + tmpi);
((out[19]).re) = (r1_6 - tmpr);
((out[19]).im) = (i1_6 - tmpi);
tmpr = (0.707106781187 * (r1_9 + i1_9));
tmpi = (0.707106781187 * (i1_9 - r1_9));
((out[4]).re) = (r1_8 + tmpr);
((out[4]).im) = (i1_8 + tmpi);
((out[20]).re) = (r1_8 - tmpr);
((out[20]).im) = (i1_8 - tmpi);
tmpr = ((0.55557023302 * r1_11) + (0.831469612303 * i1_11));
tmpi = ((0.55557023302 * i1_11) - (0.831469612303 * r1_11));
((out[5]).re) = (r1_10 + tmpr);
((out[5]).im) = (i1_10 + tmpi);
((out[21]).re) = (r1_10 - tmpr);
((out[21]).im) = (i1_10 - tmpi);
tmpr = ((0.382683432365 * r1_13) + (0.923879532511 * i1_13));
tmpi = ((0.382683432365 * i1_13) - (0.923879532511 * r1_13));
((out[6]).re) = (r1_12 + tmpr);
((out[6]).im) = (i1_12 + tmpi);
((out[22]).re) = (r1_12 - tmpr);
((out[22]).im) = (i1_12 - tmpi);
tmpr = ((0.195090322016 * r1_15) + (0.980785280403 * i1_15));
tmpi = ((0.195090322016 * i1_15) - (0.980785280403 * r1_15));
((out[7]).re) = (r1_14 + tmpr);
((out[7]).im) = (i1_14 + tmpi);
((out[23]).re) = (r1_14 - tmpr);
((out[23]).im) = (i1_14 - tmpi);
((out[8]).re) = (r1_16 + i1_17);
((out[8]).im) = (i1_16 - r1_17);
((out[24]).re) = (r1_16 - i1_17);
((out[24]).im) = (i1_16 + r1_17);
tmpr = ((0.980785280403 * i1_19) - (0.195090322016 * r1_19));
tmpi = ((0.980785280403 * r1_19) + (0.195090322016 * i1_19));
((out[9]).re) = (r1_18 + tmpr);
((out[9]).im) = (i1_18 - tmpi);
((out[25]).re) = (r1_18 - tmpr);
((out[25]).im) = (i1_18 + tmpi);
tmpr = ((0.923879532511 * i1_21) - (0.382683432365 * r1_21));
tmpi = ((0.923879532511 * r1_21) + (0.382683432365 * i1_21));
((out[10]).re) = (r1_20 + tmpr);
((out[10]).im) = (i1_20 - tmpi);
((out[26]).re) = (r1_20 - tmpr);
((out[26]).im) = (i1_20 + tmpi);
tmpr = ((0.831469612303 * i1_23) - (0.55557023302 * r1_23));
tmpi = ((0.831469612303 * r1_23) + (0.55557023302 * i1_23));
((out[11]).re) = (r1_22 + tmpr);
((out[11]).im) = (i1_22 - tmpi);
((out[27]).re) = (r1_22 - tmpr);
((out[27]).im) = (i1_22 + tmpi);
tmpr = (0.707106781187 * (i1_25 - r1_25));
tmpi = (0.707106781187 * (r1_25 + i1_25));
((out[12]).re) = (r1_24 + tmpr);
((out[12]).im) = (i1_24 - tmpi);
((out[28]).re) = (r1_24 - tmpr);
((out[28]).im) = (i1_24 + tmpi);
tmpr = ((0.55557023302 * i1_27) - (0.831469612303 * r1_27));
tmpi = ((0.55557023302 * r1_27) + (0.831469612303 * i1_27));
((out[13]).re) = (r1_26 + tmpr);
((out[13]).im) = (i1_26 - tmpi);
((out[29]).re) = (r1_26 - tmpr);
((out[29]).im) = (i1_26 + tmpi);
tmpr = ((0.382683432365 * i1_29) - (0.923879532511 * r1_29));
tmpi = ((0.382683432365 * r1_29) + (0.923879532511 * i1_29));
((out[14]).re) = (r1_28 + tmpr);
((out[14]).im) = (i1_28 - tmpi);
((out[30]).re) = (r1_28 - tmpr);
((out[30]).im) = (i1_28 + tmpi);
tmpr = ((0.195090322016 * i1_31) - (0.980785280403 * r1_31));
tmpi = ((0.195090322016 * r1_31) + (0.980785280403 * i1_31));
((out[15]).re) = (r1_30 + tmpr);
((out[15]).im) = (i1_30 - tmpi);
((out[31]).re) = (r1_30 - tmpr);
((out[31]).im) = (i1_30 + tmpi);
}
}
static void * _taskFunc23_(void *);
static void * _taskFunc24_(void *);
void fft_twiddle_32(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
double r1_8;
double i1_8;
double r1_9;
double i1_9;
double r1_10;
double i1_10;
double r1_11;
double i1_11;
double r1_12;
double i1_12;
double r1_13;
double i1_13;
double r1_14;
double i1_14;
double r1_15;
double i1_15;
double r1_16;
double i1_16;
double r1_17;
double i1_17;
double r1_18;
double i1_18;
double r1_19;
double i1_19;
double r1_20;
double i1_20;
double r1_21;
double i1_21;
double r1_22;
double i1_22;
double r1_23;
double i1_23;
double r1_24;
double i1_24;
double r1_25;
double i1_25;
double r1_26;
double i1_26;
double r1_27;
double i1_27;
double r1_28;
double i1_28;
double r1_29;
double i1_29;
double r1_30;
double i1_30;
double r1_31;
double i1_31;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
double r2_8;
double i2_8;
double r2_10;
double i2_10;
double r2_12;
double i2_12;
double r2_14;
double i2_14;
double r2_16;
double i2_16;
double r2_18;
double i2_18;
double r2_20;
double i2_20;
double r2_22;
double i2_22;
double r2_24;
double i2_24;
double r2_26;
double i2_26;
double r2_28;
double i2_28;
double r2_30;
double i2_30;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
double r3_8;
double i3_8;
double r3_12;
double i3_12;
double r3_16;
double i3_16;
double r3_20;
double i3_20;
double r3_24;
double i3_24;
double r3_28;
double i3_28;
{
double r4_0;
double i4_0;
double r4_8;
double i4_8;
double r4_16;
double i4_16;
double r4_24;
double i4_24;
{
double r5_0;
double i5_0;
double r5_16;
double i5_16;
r5_0 = ((jp[0 * m]).re);
i5_0 = ((jp[0 * m]).im);
wr = ((W[16 * l1]).re);
wi = ((W[16 * l1]).im);
tmpr = ((jp[16 * m]).re);
tmpi = ((jp[16 * m]).im);
r5_16 = ((wr * tmpr) - (wi * tmpi));
i5_16 = ((wi * tmpr) + (wr * tmpi));
r4_0 = (r5_0 + r5_16);
i4_0 = (i5_0 + i5_16);
r4_16 = (r5_0 - r5_16);
i4_16 = (i5_0 - i5_16);
}
{
double r5_8;
double i5_8;
double r5_24;
double i5_24;
wr = ((W[8 * l1]).re);
wi = ((W[8 * l1]).im);
tmpr = ((jp[8 * m]).re);
tmpi = ((jp[8 * m]).im);
r5_8 = ((wr * tmpr) - (wi * tmpi));
i5_8 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[24 * l1]).re);
wi = ((W[24 * l1]).im);
tmpr = ((jp[24 * m]).re);
tmpi = ((jp[24 * m]).im);
r5_24 = ((wr * tmpr) - (wi * tmpi));
i5_24 = ((wi * tmpr) + (wr * tmpi));
r4_8 = (r5_8 + r5_24);
i4_8 = (i5_8 + i5_24);
r4_24 = (r5_8 - r5_24);
i4_24 = (i5_8 - i5_24);
}
r3_0 = (r4_0 + r4_8);
i3_0 = (i4_0 + i4_8);
r3_16 = (r4_0 - r4_8);
i3_16 = (i4_0 - i4_8);
r3_8 = (r4_16 + i4_24);
i3_8 = (i4_16 - r4_24);
r3_24 = (r4_16 - i4_24);
i3_24 = (i4_16 + r4_24);
}
{
double r4_4;
double i4_4;
double r4_12;
double i4_12;
double r4_20;
double i4_20;
double r4_28;
double i4_28;
{
double r5_4;
double i5_4;
double r5_20;
double i5_20;
wr = ((W[4 * l1]).re);
wi = ((W[4 * l1]).im);
tmpr = ((jp[4 * m]).re);
tmpi = ((jp[4 * m]).im);
r5_4 = ((wr * tmpr) - (wi * tmpi));
i5_4 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[20 * l1]).re);
wi = ((W[20 * l1]).im);
tmpr = ((jp[20 * m]).re);
tmpi = ((jp[20 * m]).im);
r5_20 = ((wr * tmpr) - (wi * tmpi));
i5_20 = ((wi * tmpr) + (wr * tmpi));
r4_4 = (r5_4 + r5_20);
i4_4 = (i5_4 + i5_20);
r4_20 = (r5_4 - r5_20);
i4_20 = (i5_4 - i5_20);
}
{
double r5_12;
double i5_12;
double r5_28;
double i5_28;
wr = ((W[12 * l1]).re);
wi = ((W[12 * l1]).im);
tmpr = ((jp[12 * m]).re);
tmpi = ((jp[12 * m]).im);
r5_12 = ((wr * tmpr) - (wi * tmpi));
i5_12 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[28 * l1]).re);
wi = ((W[28 * l1]).im);
tmpr = ((jp[28 * m]).re);
tmpi = ((jp[28 * m]).im);
r5_28 = ((wr * tmpr) - (wi * tmpi));
i5_28 = ((wi * tmpr) + (wr * tmpi));
r4_12 = (r5_12 + r5_28);
i4_12 = (i5_12 + i5_28);
r4_28 = (r5_12 - r5_28);
i4_28 = (i5_12 - i5_28);
}
r3_4 = (r4_4 + r4_12);
i3_4 = (i4_4 + i4_12);
r3_20 = (r4_4 - r4_12);
i3_20 = (i4_4 - i4_12);
r3_12 = (r4_20 + i4_28);
i3_12 = (i4_20 - r4_28);
r3_28 = (r4_20 - i4_28);
i3_28 = (i4_20 + r4_28);
}
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_16 = (r3_0 - r3_4);
i2_16 = (i3_0 - i3_4);
tmpr = (0.707106781187 * (r3_12 + i3_12));
tmpi = (0.707106781187 * (i3_12 - r3_12));
r2_4 = (r3_8 + tmpr);
i2_4 = (i3_8 + tmpi);
r2_20 = (r3_8 - tmpr);
i2_20 = (i3_8 - tmpi);
r2_8 = (r3_16 + i3_20);
i2_8 = (i3_16 - r3_20);
r2_24 = (r3_16 - i3_20);
i2_24 = (i3_16 + r3_20);
tmpr = (0.707106781187 * (i3_28 - r3_28));
tmpi = (0.707106781187 * (r3_28 + i3_28));
r2_12 = (r3_24 + tmpr);
i2_12 = (i3_24 - tmpi);
r2_28 = (r3_24 - tmpr);
i2_28 = (i3_24 + tmpi);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
double r3_10;
double i3_10;
double r3_14;
double i3_14;
double r3_18;
double i3_18;
double r3_22;
double i3_22;
double r3_26;
double i3_26;
double r3_30;
double i3_30;
{
double r4_2;
double i4_2;
double r4_10;
double i4_10;
double r4_18;
double i4_18;
double r4_26;
double i4_26;
{
double r5_2;
double i5_2;
double r5_18;
double i5_18;
wr = ((W[2 * l1]).re);
wi = ((W[2 * l1]).im);
tmpr = ((jp[2 * m]).re);
tmpi = ((jp[2 * m]).im);
r5_2 = ((wr * tmpr) - (wi * tmpi));
i5_2 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[18 * l1]).re);
wi = ((W[18 * l1]).im);
tmpr = ((jp[18 * m]).re);
tmpi = ((jp[18 * m]).im);
r5_18 = ((wr * tmpr) - (wi * tmpi));
i5_18 = ((wi * tmpr) + (wr * tmpi));
r4_2 = (r5_2 + r5_18);
i4_2 = (i5_2 + i5_18);
r4_18 = (r5_2 - r5_18);
i4_18 = (i5_2 - i5_18);
}
{
double r5_10;
double i5_10;
double r5_26;
double i5_26;
wr = ((W[10 * l1]).re);
wi = ((W[10 * l1]).im);
tmpr = ((jp[10 * m]).re);
tmpi = ((jp[10 * m]).im);
r5_10 = ((wr * tmpr) - (wi * tmpi));
i5_10 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[26 * l1]).re);
wi = ((W[26 * l1]).im);
tmpr = ((jp[26 * m]).re);
tmpi = ((jp[26 * m]).im);
r5_26 = ((wr * tmpr) - (wi * tmpi));
i5_26 = ((wi * tmpr) + (wr * tmpi));
r4_10 = (r5_10 + r5_26);
i4_10 = (i5_10 + i5_26);
r4_26 = (r5_10 - r5_26);
i4_26 = (i5_10 - i5_26);
}
r3_2 = (r4_2 + r4_10);
i3_2 = (i4_2 + i4_10);
r3_18 = (r4_2 - r4_10);
i3_18 = (i4_2 - i4_10);
r3_10 = (r4_18 + i4_26);
i3_10 = (i4_18 - r4_26);
r3_26 = (r4_18 - i4_26);
i3_26 = (i4_18 + r4_26);
}
{
double r4_6;
double i4_6;
double r4_14;
double i4_14;
double r4_22;
double i4_22;
double r4_30;
double i4_30;
{
double r5_6;
double i5_6;
double r5_22;
double i5_22;
wr = ((W[6 * l1]).re);
wi = ((W[6 * l1]).im);
tmpr = ((jp[6 * m]).re);
tmpi = ((jp[6 * m]).im);
r5_6 = ((wr * tmpr) - (wi * tmpi));
i5_6 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[22 * l1]).re);
wi = ((W[22 * l1]).im);
tmpr = ((jp[22 * m]).re);
tmpi = ((jp[22 * m]).im);
r5_22 = ((wr * tmpr) - (wi * tmpi));
i5_22 = ((wi * tmpr) + (wr * tmpi));
r4_6 = (r5_6 + r5_22);
i4_6 = (i5_6 + i5_22);
r4_22 = (r5_6 - r5_22);
i4_22 = (i5_6 - i5_22);
}
{
double r5_14;
double i5_14;
double r5_30;
double i5_30;
wr = ((W[14 * l1]).re);
wi = ((W[14 * l1]).im);
tmpr = ((jp[14 * m]).re);
tmpi = ((jp[14 * m]).im);
r5_14 = ((wr * tmpr) - (wi * tmpi));
i5_14 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[30 * l1]).re);
wi = ((W[30 * l1]).im);
tmpr = ((jp[30 * m]).re);
tmpi = ((jp[30 * m]).im);
r5_30 = ((wr * tmpr) - (wi * tmpi));
i5_30 = ((wi * tmpr) + (wr * tmpi));
r4_14 = (r5_14 + r5_30);
i4_14 = (i5_14 + i5_30);
r4_30 = (r5_14 - r5_30);
i4_30 = (i5_14 - i5_30);
}
r3_6 = (r4_6 + r4_14);
i3_6 = (i4_6 + i4_14);
r3_22 = (r4_6 - r4_14);
i3_22 = (i4_6 - i4_14);
r3_14 = (r4_22 + i4_30);
i3_14 = (i4_22 - r4_30);
r3_30 = (r4_22 - i4_30);
i3_30 = (i4_22 + r4_30);
}
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_18 = (r3_2 - r3_6);
i2_18 = (i3_2 - i3_6);
tmpr = (0.707106781187 * (r3_14 + i3_14));
tmpi = (0.707106781187 * (i3_14 - r3_14));
r2_6 = (r3_10 + tmpr);
i2_6 = (i3_10 + tmpi);
r2_22 = (r3_10 - tmpr);
i2_22 = (i3_10 - tmpi);
r2_10 = (r3_18 + i3_22);
i2_10 = (i3_18 - r3_22);
r2_26 = (r3_18 - i3_22);
i2_26 = (i3_18 + r3_22);
tmpr = (0.707106781187 * (i3_30 - r3_30));
tmpi = (0.707106781187 * (r3_30 + i3_30));
r2_14 = (r3_26 + tmpr);
i2_14 = (i3_26 - tmpi);
r2_30 = (r3_26 - tmpr);
i2_30 = (i3_26 + tmpi);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_16 = (r2_0 - r2_2);
i1_16 = (i2_0 - i2_2);
tmpr = ((0.923879532511 * r2_6) + (0.382683432365 * i2_6));
tmpi = ((0.923879532511 * i2_6) - (0.382683432365 * r2_6));
r1_2 = (r2_4 + tmpr);
i1_2 = (i2_4 + tmpi);
r1_18 = (r2_4 - tmpr);
i1_18 = (i2_4 - tmpi);
tmpr = (0.707106781187 * (r2_10 + i2_10));
tmpi = (0.707106781187 * (i2_10 - r2_10));
r1_4 = (r2_8 + tmpr);
i1_4 = (i2_8 + tmpi);
r1_20 = (r2_8 - tmpr);
i1_20 = (i2_8 - tmpi);
tmpr = ((0.382683432365 * r2_14) + (0.923879532511 * i2_14));
tmpi = ((0.382683432365 * i2_14) - (0.923879532511 * r2_14));
r1_6 = (r2_12 + tmpr);
i1_6 = (i2_12 + tmpi);
r1_22 = (r2_12 - tmpr);
i1_22 = (i2_12 - tmpi);
r1_8 = (r2_16 + i2_18);
i1_8 = (i2_16 - r2_18);
r1_24 = (r2_16 - i2_18);
i1_24 = (i2_16 + r2_18);
tmpr = ((0.923879532511 * i2_22) - (0.382683432365 * r2_22));
tmpi = ((0.923879532511 * r2_22) + (0.382683432365 * i2_22));
r1_10 = (r2_20 + tmpr);
i1_10 = (i2_20 - tmpi);
r1_26 = (r2_20 - tmpr);
i1_26 = (i2_20 + tmpi);
tmpr = (0.707106781187 * (i2_26 - r2_26));
tmpi = (0.707106781187 * (r2_26 + i2_26));
r1_12 = (r2_24 + tmpr);
i1_12 = (i2_24 - tmpi);
r1_28 = (r2_24 - tmpr);
i1_28 = (i2_24 + tmpi);
tmpr = ((0.382683432365 * i2_30) - (0.923879532511 * r2_30));
tmpi = ((0.382683432365 * r2_30) + (0.923879532511 * i2_30));
r1_14 = (r2_28 + tmpr);
i1_14 = (i2_28 - tmpi);
r1_30 = (r2_28 - tmpr);
i1_30 = (i2_28 + tmpi);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
double r2_9;
double i2_9;
double r2_11;
double i2_11;
double r2_13;
double i2_13;
double r2_15;
double i2_15;
double r2_17;
double i2_17;
double r2_19;
double i2_19;
double r2_21;
double i2_21;
double r2_23;
double i2_23;
double r2_25;
double i2_25;
double r2_27;
double i2_27;
double r2_29;
double i2_29;
double r2_31;
double i2_31;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
double r3_9;
double i3_9;
double r3_13;
double i3_13;
double r3_17;
double i3_17;
double r3_21;
double i3_21;
double r3_25;
double i3_25;
double r3_29;
double i3_29;
{
double r4_1;
double i4_1;
double r4_9;
double i4_9;
double r4_17;
double i4_17;
double r4_25;
double i4_25;
{
double r5_1;
double i5_1;
double r5_17;
double i5_17;
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r5_1 = ((wr * tmpr) - (wi * tmpi));
i5_1 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[17 * l1]).re);
wi = ((W[17 * l1]).im);
tmpr = ((jp[17 * m]).re);
tmpi = ((jp[17 * m]).im);
r5_17 = ((wr * tmpr) - (wi * tmpi));
i5_17 = ((wi * tmpr) + (wr * tmpi));
r4_1 = (r5_1 + r5_17);
i4_1 = (i5_1 + i5_17);
r4_17 = (r5_1 - r5_17);
i4_17 = (i5_1 - i5_17);
}
{
double r5_9;
double i5_9;
double r5_25;
double i5_25;
wr = ((W[9 * l1]).re);
wi = ((W[9 * l1]).im);
tmpr = ((jp[9 * m]).re);
tmpi = ((jp[9 * m]).im);
r5_9 = ((wr * tmpr) - (wi * tmpi));
i5_9 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[25 * l1]).re);
wi = ((W[25 * l1]).im);
tmpr = ((jp[25 * m]).re);
tmpi = ((jp[25 * m]).im);
r5_25 = ((wr * tmpr) - (wi * tmpi));
i5_25 = ((wi * tmpr) + (wr * tmpi));
r4_9 = (r5_9 + r5_25);
i4_9 = (i5_9 + i5_25);
r4_25 = (r5_9 - r5_25);
i4_25 = (i5_9 - i5_25);
}
r3_1 = (r4_1 + r4_9);
i3_1 = (i4_1 + i4_9);
r3_17 = (r4_1 - r4_9);
i3_17 = (i4_1 - i4_9);
r3_9 = (r4_17 + i4_25);
i3_9 = (i4_17 - r4_25);
r3_25 = (r4_17 - i4_25);
i3_25 = (i4_17 + r4_25);
}
{
double r4_5;
double i4_5;
double r4_13;
double i4_13;
double r4_21;
double i4_21;
double r4_29;
double i4_29;
{
double r5_5;
double i5_5;
double r5_21;
double i5_21;
wr = ((W[5 * l1]).re);
wi = ((W[5 * l1]).im);
tmpr = ((jp[5 * m]).re);
tmpi = ((jp[5 * m]).im);
r5_5 = ((wr * tmpr) - (wi * tmpi));
i5_5 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[21 * l1]).re);
wi = ((W[21 * l1]).im);
tmpr = ((jp[21 * m]).re);
tmpi = ((jp[21 * m]).im);
r5_21 = ((wr * tmpr) - (wi * tmpi));
i5_21 = ((wi * tmpr) + (wr * tmpi));
r4_5 = (r5_5 + r5_21);
i4_5 = (i5_5 + i5_21);
r4_21 = (r5_5 - r5_21);
i4_21 = (i5_5 - i5_21);
}
{
double r5_13;
double i5_13;
double r5_29;
double i5_29;
wr = ((W[13 * l1]).re);
wi = ((W[13 * l1]).im);
tmpr = ((jp[13 * m]).re);
tmpi = ((jp[13 * m]).im);
r5_13 = ((wr * tmpr) - (wi * tmpi));
i5_13 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[29 * l1]).re);
wi = ((W[29 * l1]).im);
tmpr = ((jp[29 * m]).re);
tmpi = ((jp[29 * m]).im);
r5_29 = ((wr * tmpr) - (wi * tmpi));
i5_29 = ((wi * tmpr) + (wr * tmpi));
r4_13 = (r5_13 + r5_29);
i4_13 = (i5_13 + i5_29);
r4_29 = (r5_13 - r5_29);
i4_29 = (i5_13 - i5_29);
}
r3_5 = (r4_5 + r4_13);
i3_5 = (i4_5 + i4_13);
r3_21 = (r4_5 - r4_13);
i3_21 = (i4_5 - i4_13);
r3_13 = (r4_21 + i4_29);
i3_13 = (i4_21 - r4_29);
r3_29 = (r4_21 - i4_29);
i3_29 = (i4_21 + r4_29);
}
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_17 = (r3_1 - r3_5);
i2_17 = (i3_1 - i3_5);
tmpr = (0.707106781187 * (r3_13 + i3_13));
tmpi = (0.707106781187 * (i3_13 - r3_13));
r2_5 = (r3_9 + tmpr);
i2_5 = (i3_9 + tmpi);
r2_21 = (r3_9 - tmpr);
i2_21 = (i3_9 - tmpi);
r2_9 = (r3_17 + i3_21);
i2_9 = (i3_17 - r3_21);
r2_25 = (r3_17 - i3_21);
i2_25 = (i3_17 + r3_21);
tmpr = (0.707106781187 * (i3_29 - r3_29));
tmpi = (0.707106781187 * (r3_29 + i3_29));
r2_13 = (r3_25 + tmpr);
i2_13 = (i3_25 - tmpi);
r2_29 = (r3_25 - tmpr);
i2_29 = (i3_25 + tmpi);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
double r3_11;
double i3_11;
double r3_15;
double i3_15;
double r3_19;
double i3_19;
double r3_23;
double i3_23;
double r3_27;
double i3_27;
double r3_31;
double i3_31;
{
double r4_3;
double i4_3;
double r4_11;
double i4_11;
double r4_19;
double i4_19;
double r4_27;
double i4_27;
{
double r5_3;
double i5_3;
double r5_19;
double i5_19;
wr = ((W[3 * l1]).re);
wi = ((W[3 * l1]).im);
tmpr = ((jp[3 * m]).re);
tmpi = ((jp[3 * m]).im);
r5_3 = ((wr * tmpr) - (wi * tmpi));
i5_3 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[19 * l1]).re);
wi = ((W[19 * l1]).im);
tmpr = ((jp[19 * m]).re);
tmpi = ((jp[19 * m]).im);
r5_19 = ((wr * tmpr) - (wi * tmpi));
i5_19 = ((wi * tmpr) + (wr * tmpi));
r4_3 = (r5_3 + r5_19);
i4_3 = (i5_3 + i5_19);
r4_19 = (r5_3 - r5_19);
i4_19 = (i5_3 - i5_19);
}
{
double r5_11;
double i5_11;
double r5_27;
double i5_27;
wr = ((W[11 * l1]).re);
wi = ((W[11 * l1]).im);
tmpr = ((jp[11 * m]).re);
tmpi = ((jp[11 * m]).im);
r5_11 = ((wr * tmpr) - (wi * tmpi));
i5_11 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[27 * l1]).re);
wi = ((W[27 * l1]).im);
tmpr = ((jp[27 * m]).re);
tmpi = ((jp[27 * m]).im);
r5_27 = ((wr * tmpr) - (wi * tmpi));
i5_27 = ((wi * tmpr) + (wr * tmpi));
r4_11 = (r5_11 + r5_27);
i4_11 = (i5_11 + i5_27);
r4_27 = (r5_11 - r5_27);
i4_27 = (i5_11 - i5_27);
}
r3_3 = (r4_3 + r4_11);
i3_3 = (i4_3 + i4_11);
r3_19 = (r4_3 - r4_11);
i3_19 = (i4_3 - i4_11);
r3_11 = (r4_19 + i4_27);
i3_11 = (i4_19 - r4_27);
r3_27 = (r4_19 - i4_27);
i3_27 = (i4_19 + r4_27);
}
{
double r4_7;
double i4_7;
double r4_15;
double i4_15;
double r4_23;
double i4_23;
double r4_31;
double i4_31;
{
double r5_7;
double i5_7;
double r5_23;
double i5_23;
wr = ((W[7 * l1]).re);
wi = ((W[7 * l1]).im);
tmpr = ((jp[7 * m]).re);
tmpi = ((jp[7 * m]).im);
r5_7 = ((wr * tmpr) - (wi * tmpi));
i5_7 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[23 * l1]).re);
wi = ((W[23 * l1]).im);
tmpr = ((jp[23 * m]).re);
tmpi = ((jp[23 * m]).im);
r5_23 = ((wr * tmpr) - (wi * tmpi));
i5_23 = ((wi * tmpr) + (wr * tmpi));
r4_7 = (r5_7 + r5_23);
i4_7 = (i5_7 + i5_23);
r4_23 = (r5_7 - r5_23);
i4_23 = (i5_7 - i5_23);
}
{
double r5_15;
double i5_15;
double r5_31;
double i5_31;
wr = ((W[15 * l1]).re);
wi = ((W[15 * l1]).im);
tmpr = ((jp[15 * m]).re);
tmpi = ((jp[15 * m]).im);
r5_15 = ((wr * tmpr) - (wi * tmpi));
i5_15 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[31 * l1]).re);
wi = ((W[31 * l1]).im);
tmpr = ((jp[31 * m]).re);
tmpi = ((jp[31 * m]).im);
r5_31 = ((wr * tmpr) - (wi * tmpi));
i5_31 = ((wi * tmpr) + (wr * tmpi));
r4_15 = (r5_15 + r5_31);
i4_15 = (i5_15 + i5_31);
r4_31 = (r5_15 - r5_31);
i4_31 = (i5_15 - i5_31);
}
r3_7 = (r4_7 + r4_15);
i3_7 = (i4_7 + i4_15);
r3_23 = (r4_7 - r4_15);
i3_23 = (i4_7 - i4_15);
r3_15 = (r4_23 + i4_31);
i3_15 = (i4_23 - r4_31);
r3_31 = (r4_23 - i4_31);
i3_31 = (i4_23 + r4_31);
}
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_19 = (r3_3 - r3_7);
i2_19 = (i3_3 - i3_7);
tmpr = (0.707106781187 * (r3_15 + i3_15));
tmpi = (0.707106781187 * (i3_15 - r3_15));
r2_7 = (r3_11 + tmpr);
i2_7 = (i3_11 + tmpi);
r2_23 = (r3_11 - tmpr);
i2_23 = (i3_11 - tmpi);
r2_11 = (r3_19 + i3_23);
i2_11 = (i3_19 - r3_23);
r2_27 = (r3_19 - i3_23);
i2_27 = (i3_19 + r3_23);
tmpr = (0.707106781187 * (i3_31 - r3_31));
tmpi = (0.707106781187 * (r3_31 + i3_31));
r2_15 = (r3_27 + tmpr);
i2_15 = (i3_27 - tmpi);
r2_31 = (r3_27 - tmpr);
i2_31 = (i3_27 + tmpi);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_17 = (r2_1 - r2_3);
i1_17 = (i2_1 - i2_3);
tmpr = ((0.923879532511 * r2_7) + (0.382683432365 * i2_7));
tmpi = ((0.923879532511 * i2_7) - (0.382683432365 * r2_7));
r1_3 = (r2_5 + tmpr);
i1_3 = (i2_5 + tmpi);
r1_19 = (r2_5 - tmpr);
i1_19 = (i2_5 - tmpi);
tmpr = (0.707106781187 * (r2_11 + i2_11));
tmpi = (0.707106781187 * (i2_11 - r2_11));
r1_5 = (r2_9 + tmpr);
i1_5 = (i2_9 + tmpi);
r1_21 = (r2_9 - tmpr);
i1_21 = (i2_9 - tmpi);
tmpr = ((0.382683432365 * r2_15) + (0.923879532511 * i2_15));
tmpi = ((0.382683432365 * i2_15) - (0.923879532511 * r2_15));
r1_7 = (r2_13 + tmpr);
i1_7 = (i2_13 + tmpi);
r1_23 = (r2_13 - tmpr);
i1_23 = (i2_13 - tmpi);
r1_9 = (r2_17 + i2_19);
i1_9 = (i2_17 - r2_19);
r1_25 = (r2_17 - i2_19);
i1_25 = (i2_17 + r2_19);
tmpr = ((0.923879532511 * i2_23) - (0.382683432365 * r2_23));
tmpi = ((0.923879532511 * r2_23) + (0.382683432365 * i2_23));
r1_11 = (r2_21 + tmpr);
i1_11 = (i2_21 - tmpi);
r1_27 = (r2_21 - tmpr);
i1_27 = (i2_21 + tmpi);
tmpr = (0.707106781187 * (i2_27 - r2_27));
tmpi = (0.707106781187 * (r2_27 + i2_27));
r1_13 = (r2_25 + tmpr);
i1_13 = (i2_25 - tmpi);
r1_29 = (r2_25 - tmpr);
i1_29 = (i2_25 + tmpi);
tmpr = ((0.382683432365 * i2_31) - (0.923879532511 * r2_31));
tmpi = ((0.382683432365 * r2_31) + (0.923879532511 * i2_31));
r1_15 = (r2_29 + tmpr);
i1_15 = (i2_29 - tmpi);
r1_31 = (r2_29 - tmpr);
i1_31 = (i2_29 + tmpi);
}
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[16 * m]).re) = (r1_0 - r1_1);
((kp[16 * m]).im) = (i1_0 - i1_1);
tmpr = ((0.980785280403 * r1_3) + (0.195090322016 * i1_3));
tmpi = ((0.980785280403 * i1_3) - (0.195090322016 * r1_3));
((kp[1 * m]).re) = (r1_2 + tmpr);
((kp[1 * m]).im) = (i1_2 + tmpi);
((kp[17 * m]).re) = (r1_2 - tmpr);
((kp[17 * m]).im) = (i1_2 - tmpi);
tmpr = ((0.923879532511 * r1_5) + (0.382683432365 * i1_5));
tmpi = ((0.923879532511 * i1_5) - (0.382683432365 * r1_5));
((kp[2 * m]).re) = (r1_4 + tmpr);
((kp[2 * m]).im) = (i1_4 + tmpi);
((kp[18 * m]).re) = (r1_4 - tmpr);
((kp[18 * m]).im) = (i1_4 - tmpi);
tmpr = ((0.831469612303 * r1_7) + (0.55557023302 * i1_7));
tmpi = ((0.831469612303 * i1_7) - (0.55557023302 * r1_7));
((kp[3 * m]).re) = (r1_6 + tmpr);
((kp[3 * m]).im) = (i1_6 + tmpi);
((kp[19 * m]).re) = (r1_6 - tmpr);
((kp[19 * m]).im) = (i1_6 - tmpi);
tmpr = (0.707106781187 * (r1_9 + i1_9));
tmpi = (0.707106781187 * (i1_9 - r1_9));
((kp[4 * m]).re) = (r1_8 + tmpr);
((kp[4 * m]).im) = (i1_8 + tmpi);
((kp[20 * m]).re) = (r1_8 - tmpr);
((kp[20 * m]).im) = (i1_8 - tmpi);
tmpr = ((0.55557023302 * r1_11) + (0.831469612303 * i1_11));
tmpi = ((0.55557023302 * i1_11) - (0.831469612303 * r1_11));
((kp[5 * m]).re) = (r1_10 + tmpr);
((kp[5 * m]).im) = (i1_10 + tmpi);
((kp[21 * m]).re) = (r1_10 - tmpr);
((kp[21 * m]).im) = (i1_10 - tmpi);
tmpr = ((0.382683432365 * r1_13) + (0.923879532511 * i1_13));
tmpi = ((0.382683432365 * i1_13) - (0.923879532511 * r1_13));
((kp[6 * m]).re) = (r1_12 + tmpr);
((kp[6 * m]).im) = (i1_12 + tmpi);
((kp[22 * m]).re) = (r1_12 - tmpr);
((kp[22 * m]).im) = (i1_12 - tmpi);
tmpr = ((0.195090322016 * r1_15) + (0.980785280403 * i1_15));
tmpi = ((0.195090322016 * i1_15) - (0.980785280403 * r1_15));
((kp[7 * m]).re) = (r1_14 + tmpr);
((kp[7 * m]).im) = (i1_14 + tmpi);
((kp[23 * m]).re) = (r1_14 - tmpr);
((kp[23 * m]).im) = (i1_14 - tmpi);
((kp[8 * m]).re) = (r1_16 + i1_17);
((kp[8 * m]).im) = (i1_16 - r1_17);
((kp[24 * m]).re) = (r1_16 - i1_17);
((kp[24 * m]).im) = (i1_16 + r1_17);
tmpr = ((0.980785280403 * i1_19) - (0.195090322016 * r1_19));
tmpi = ((0.980785280403 * r1_19) + (0.195090322016 * i1_19));
((kp[9 * m]).re) = (r1_18 + tmpr);
((kp[9 * m]).im) = (i1_18 - tmpi);
((kp[25 * m]).re) = (r1_18 - tmpr);
((kp[25 * m]).im) = (i1_18 + tmpi);
tmpr = ((0.923879532511 * i1_21) - (0.382683432365 * r1_21));
tmpi = ((0.923879532511 * r1_21) + (0.382683432365 * i1_21));
((kp[10 * m]).re) = (r1_20 + tmpr);
((kp[10 * m]).im) = (i1_20 - tmpi);
((kp[26 * m]).re) = (r1_20 - tmpr);
((kp[26 * m]).im) = (i1_20 + tmpi);
tmpr = ((0.831469612303 * i1_23) - (0.55557023302 * r1_23));
tmpi = ((0.831469612303 * r1_23) + (0.55557023302 * i1_23));
((kp[11 * m]).re) = (r1_22 + tmpr);
((kp[11 * m]).im) = (i1_22 - tmpi);
((kp[27 * m]).re) = (r1_22 - tmpr);
((kp[27 * m]).im) = (i1_22 + tmpi);
tmpr = (0.707106781187 * (i1_25 - r1_25));
tmpi = (0.707106781187 * (r1_25 + i1_25));
((kp[12 * m]).re) = (r1_24 + tmpr);
((kp[12 * m]).im) = (i1_24 - tmpi);
((kp[28 * m]).re) = (r1_24 - tmpr);
((kp[28 * m]).im) = (i1_24 + tmpi);
tmpr = ((0.55557023302 * i1_27) - (0.831469612303 * r1_27));
tmpi = ((0.55557023302 * r1_27) + (0.831469612303 * i1_27));
((kp[13 * m]).re) = (r1_26 + tmpr);
((kp[13 * m]).im) = (i1_26 - tmpi);
((kp[29 * m]).re) = (r1_26 - tmpr);
((kp[29 * m]).im) = (i1_26 + tmpi);
tmpr = ((0.382683432365 * i1_29) - (0.923879532511 * r1_29));
tmpi = ((0.382683432365 * r1_29) + (0.923879532511 * i1_29));
((kp[14 * m]).re) = (r1_28 + tmpr);
((kp[14 * m]).im) = (i1_28 - tmpi);
((kp[30 * m]).re) = (r1_28 - tmpr);
((kp[30 * m]).im) = (i1_28 + tmpi);
tmpr = ((0.195090322016 * i1_31) - (0.980785280403 * r1_31));
tmpi = ((0.195090322016 * r1_31) + (0.980785280403 * i1_31));
((kp[15 * m]).re) = (r1_30 + tmpr);
((kp[15 * m]).im) = (i1_30 - tmpi);
((kp[31 * m]).re) = (r1_30 - tmpr);
((kp[31 * m]).im) = (i1_30 + tmpi);
}
}
}
else
{
int ab = (a + b) / 2;
_taskFunc23_((void *)0);
_taskFunc24_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc24_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_32(ab, b, in, out, W, nW, nWdn, m);
CANCEL_task_3624 :
;
}
ort_taskenv_free(_tenv, _taskFunc24_);
return ((void *) 0);
}
static void * _taskFunc23_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int nWdn;
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int nWdn = _tenv->nWdn;
int m = _tenv->m;
{
fft_twiddle_32(a, ab, in, out, W, nW, nWdn, m);
CANCEL_task_3622 :
;
}
ort_taskenv_free(_tenv, _taskFunc23_);
return ((void *) 0);
}
void fft_twiddle_32_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), struct _noname19_ (* W), int nW, int nWdn, int m)
{
int l1, i;
struct _noname19_ (* jp);
struct _noname19_ (* kp);
double tmpr;
double tmpi;
double wr;
double wi;
if ((b - a) < 128)
{
for (i = a, l1 = nWdn * i, kp = out + i; i < b; i++, l1 += nWdn, kp++)
{
jp = in + i;
{
double r1_0;
double i1_0;
double r1_1;
double i1_1;
double r1_2;
double i1_2;
double r1_3;
double i1_3;
double r1_4;
double i1_4;
double r1_5;
double i1_5;
double r1_6;
double i1_6;
double r1_7;
double i1_7;
double r1_8;
double i1_8;
double r1_9;
double i1_9;
double r1_10;
double i1_10;
double r1_11;
double i1_11;
double r1_12;
double i1_12;
double r1_13;
double i1_13;
double r1_14;
double i1_14;
double r1_15;
double i1_15;
double r1_16;
double i1_16;
double r1_17;
double i1_17;
double r1_18;
double i1_18;
double r1_19;
double i1_19;
double r1_20;
double i1_20;
double r1_21;
double i1_21;
double r1_22;
double i1_22;
double r1_23;
double i1_23;
double r1_24;
double i1_24;
double r1_25;
double i1_25;
double r1_26;
double i1_26;
double r1_27;
double i1_27;
double r1_28;
double i1_28;
double r1_29;
double i1_29;
double r1_30;
double i1_30;
double r1_31;
double i1_31;
{
double r2_0;
double i2_0;
double r2_2;
double i2_2;
double r2_4;
double i2_4;
double r2_6;
double i2_6;
double r2_8;
double i2_8;
double r2_10;
double i2_10;
double r2_12;
double i2_12;
double r2_14;
double i2_14;
double r2_16;
double i2_16;
double r2_18;
double i2_18;
double r2_20;
double i2_20;
double r2_22;
double i2_22;
double r2_24;
double i2_24;
double r2_26;
double i2_26;
double r2_28;
double i2_28;
double r2_30;
double i2_30;
{
double r3_0;
double i3_0;
double r3_4;
double i3_4;
double r3_8;
double i3_8;
double r3_12;
double i3_12;
double r3_16;
double i3_16;
double r3_20;
double i3_20;
double r3_24;
double i3_24;
double r3_28;
double i3_28;
{
double r4_0;
double i4_0;
double r4_8;
double i4_8;
double r4_16;
double i4_16;
double r4_24;
double i4_24;
{
double r5_0;
double i5_0;
double r5_16;
double i5_16;
r5_0 = ((jp[0 * m]).re);
i5_0 = ((jp[0 * m]).im);
wr = ((W[16 * l1]).re);
wi = ((W[16 * l1]).im);
tmpr = ((jp[16 * m]).re);
tmpi = ((jp[16 * m]).im);
r5_16 = ((wr * tmpr) - (wi * tmpi));
i5_16 = ((wi * tmpr) + (wr * tmpi));
r4_0 = (r5_0 + r5_16);
i4_0 = (i5_0 + i5_16);
r4_16 = (r5_0 - r5_16);
i4_16 = (i5_0 - i5_16);
}
{
double r5_8;
double i5_8;
double r5_24;
double i5_24;
wr = ((W[8 * l1]).re);
wi = ((W[8 * l1]).im);
tmpr = ((jp[8 * m]).re);
tmpi = ((jp[8 * m]).im);
r5_8 = ((wr * tmpr) - (wi * tmpi));
i5_8 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[24 * l1]).re);
wi = ((W[24 * l1]).im);
tmpr = ((jp[24 * m]).re);
tmpi = ((jp[24 * m]).im);
r5_24 = ((wr * tmpr) - (wi * tmpi));
i5_24 = ((wi * tmpr) + (wr * tmpi));
r4_8 = (r5_8 + r5_24);
i4_8 = (i5_8 + i5_24);
r4_24 = (r5_8 - r5_24);
i4_24 = (i5_8 - i5_24);
}
r3_0 = (r4_0 + r4_8);
i3_0 = (i4_0 + i4_8);
r3_16 = (r4_0 - r4_8);
i3_16 = (i4_0 - i4_8);
r3_8 = (r4_16 + i4_24);
i3_8 = (i4_16 - r4_24);
r3_24 = (r4_16 - i4_24);
i3_24 = (i4_16 + r4_24);
}
{
double r4_4;
double i4_4;
double r4_12;
double i4_12;
double r4_20;
double i4_20;
double r4_28;
double i4_28;
{
double r5_4;
double i5_4;
double r5_20;
double i5_20;
wr = ((W[4 * l1]).re);
wi = ((W[4 * l1]).im);
tmpr = ((jp[4 * m]).re);
tmpi = ((jp[4 * m]).im);
r5_4 = ((wr * tmpr) - (wi * tmpi));
i5_4 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[20 * l1]).re);
wi = ((W[20 * l1]).im);
tmpr = ((jp[20 * m]).re);
tmpi = ((jp[20 * m]).im);
r5_20 = ((wr * tmpr) - (wi * tmpi));
i5_20 = ((wi * tmpr) + (wr * tmpi));
r4_4 = (r5_4 + r5_20);
i4_4 = (i5_4 + i5_20);
r4_20 = (r5_4 - r5_20);
i4_20 = (i5_4 - i5_20);
}
{
double r5_12;
double i5_12;
double r5_28;
double i5_28;
wr = ((W[12 * l1]).re);
wi = ((W[12 * l1]).im);
tmpr = ((jp[12 * m]).re);
tmpi = ((jp[12 * m]).im);
r5_12 = ((wr * tmpr) - (wi * tmpi));
i5_12 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[28 * l1]).re);
wi = ((W[28 * l1]).im);
tmpr = ((jp[28 * m]).re);
tmpi = ((jp[28 * m]).im);
r5_28 = ((wr * tmpr) - (wi * tmpi));
i5_28 = ((wi * tmpr) + (wr * tmpi));
r4_12 = (r5_12 + r5_28);
i4_12 = (i5_12 + i5_28);
r4_28 = (r5_12 - r5_28);
i4_28 = (i5_12 - i5_28);
}
r3_4 = (r4_4 + r4_12);
i3_4 = (i4_4 + i4_12);
r3_20 = (r4_4 - r4_12);
i3_20 = (i4_4 - i4_12);
r3_12 = (r4_20 + i4_28);
i3_12 = (i4_20 - r4_28);
r3_28 = (r4_20 - i4_28);
i3_28 = (i4_20 + r4_28);
}
r2_0 = (r3_0 + r3_4);
i2_0 = (i3_0 + i3_4);
r2_16 = (r3_0 - r3_4);
i2_16 = (i3_0 - i3_4);
tmpr = (0.707106781187 * (r3_12 + i3_12));
tmpi = (0.707106781187 * (i3_12 - r3_12));
r2_4 = (r3_8 + tmpr);
i2_4 = (i3_8 + tmpi);
r2_20 = (r3_8 - tmpr);
i2_20 = (i3_8 - tmpi);
r2_8 = (r3_16 + i3_20);
i2_8 = (i3_16 - r3_20);
r2_24 = (r3_16 - i3_20);
i2_24 = (i3_16 + r3_20);
tmpr = (0.707106781187 * (i3_28 - r3_28));
tmpi = (0.707106781187 * (r3_28 + i3_28));
r2_12 = (r3_24 + tmpr);
i2_12 = (i3_24 - tmpi);
r2_28 = (r3_24 - tmpr);
i2_28 = (i3_24 + tmpi);
}
{
double r3_2;
double i3_2;
double r3_6;
double i3_6;
double r3_10;
double i3_10;
double r3_14;
double i3_14;
double r3_18;
double i3_18;
double r3_22;
double i3_22;
double r3_26;
double i3_26;
double r3_30;
double i3_30;
{
double r4_2;
double i4_2;
double r4_10;
double i4_10;
double r4_18;
double i4_18;
double r4_26;
double i4_26;
{
double r5_2;
double i5_2;
double r5_18;
double i5_18;
wr = ((W[2 * l1]).re);
wi = ((W[2 * l1]).im);
tmpr = ((jp[2 * m]).re);
tmpi = ((jp[2 * m]).im);
r5_2 = ((wr * tmpr) - (wi * tmpi));
i5_2 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[18 * l1]).re);
wi = ((W[18 * l1]).im);
tmpr = ((jp[18 * m]).re);
tmpi = ((jp[18 * m]).im);
r5_18 = ((wr * tmpr) - (wi * tmpi));
i5_18 = ((wi * tmpr) + (wr * tmpi));
r4_2 = (r5_2 + r5_18);
i4_2 = (i5_2 + i5_18);
r4_18 = (r5_2 - r5_18);
i4_18 = (i5_2 - i5_18);
}
{
double r5_10;
double i5_10;
double r5_26;
double i5_26;
wr = ((W[10 * l1]).re);
wi = ((W[10 * l1]).im);
tmpr = ((jp[10 * m]).re);
tmpi = ((jp[10 * m]).im);
r5_10 = ((wr * tmpr) - (wi * tmpi));
i5_10 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[26 * l1]).re);
wi = ((W[26 * l1]).im);
tmpr = ((jp[26 * m]).re);
tmpi = ((jp[26 * m]).im);
r5_26 = ((wr * tmpr) - (wi * tmpi));
i5_26 = ((wi * tmpr) + (wr * tmpi));
r4_10 = (r5_10 + r5_26);
i4_10 = (i5_10 + i5_26);
r4_26 = (r5_10 - r5_26);
i4_26 = (i5_10 - i5_26);
}
r3_2 = (r4_2 + r4_10);
i3_2 = (i4_2 + i4_10);
r3_18 = (r4_2 - r4_10);
i3_18 = (i4_2 - i4_10);
r3_10 = (r4_18 + i4_26);
i3_10 = (i4_18 - r4_26);
r3_26 = (r4_18 - i4_26);
i3_26 = (i4_18 + r4_26);
}
{
double r4_6;
double i4_6;
double r4_14;
double i4_14;
double r4_22;
double i4_22;
double r4_30;
double i4_30;
{
double r5_6;
double i5_6;
double r5_22;
double i5_22;
wr = ((W[6 * l1]).re);
wi = ((W[6 * l1]).im);
tmpr = ((jp[6 * m]).re);
tmpi = ((jp[6 * m]).im);
r5_6 = ((wr * tmpr) - (wi * tmpi));
i5_6 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[22 * l1]).re);
wi = ((W[22 * l1]).im);
tmpr = ((jp[22 * m]).re);
tmpi = ((jp[22 * m]).im);
r5_22 = ((wr * tmpr) - (wi * tmpi));
i5_22 = ((wi * tmpr) + (wr * tmpi));
r4_6 = (r5_6 + r5_22);
i4_6 = (i5_6 + i5_22);
r4_22 = (r5_6 - r5_22);
i4_22 = (i5_6 - i5_22);
}
{
double r5_14;
double i5_14;
double r5_30;
double i5_30;
wr = ((W[14 * l1]).re);
wi = ((W[14 * l1]).im);
tmpr = ((jp[14 * m]).re);
tmpi = ((jp[14 * m]).im);
r5_14 = ((wr * tmpr) - (wi * tmpi));
i5_14 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[30 * l1]).re);
wi = ((W[30 * l1]).im);
tmpr = ((jp[30 * m]).re);
tmpi = ((jp[30 * m]).im);
r5_30 = ((wr * tmpr) - (wi * tmpi));
i5_30 = ((wi * tmpr) + (wr * tmpi));
r4_14 = (r5_14 + r5_30);
i4_14 = (i5_14 + i5_30);
r4_30 = (r5_14 - r5_30);
i4_30 = (i5_14 - i5_30);
}
r3_6 = (r4_6 + r4_14);
i3_6 = (i4_6 + i4_14);
r3_22 = (r4_6 - r4_14);
i3_22 = (i4_6 - i4_14);
r3_14 = (r4_22 + i4_30);
i3_14 = (i4_22 - r4_30);
r3_30 = (r4_22 - i4_30);
i3_30 = (i4_22 + r4_30);
}
r2_2 = (r3_2 + r3_6);
i2_2 = (i3_2 + i3_6);
r2_18 = (r3_2 - r3_6);
i2_18 = (i3_2 - i3_6);
tmpr = (0.707106781187 * (r3_14 + i3_14));
tmpi = (0.707106781187 * (i3_14 - r3_14));
r2_6 = (r3_10 + tmpr);
i2_6 = (i3_10 + tmpi);
r2_22 = (r3_10 - tmpr);
i2_22 = (i3_10 - tmpi);
r2_10 = (r3_18 + i3_22);
i2_10 = (i3_18 - r3_22);
r2_26 = (r3_18 - i3_22);
i2_26 = (i3_18 + r3_22);
tmpr = (0.707106781187 * (i3_30 - r3_30));
tmpi = (0.707106781187 * (r3_30 + i3_30));
r2_14 = (r3_26 + tmpr);
i2_14 = (i3_26 - tmpi);
r2_30 = (r3_26 - tmpr);
i2_30 = (i3_26 + tmpi);
}
r1_0 = (r2_0 + r2_2);
i1_0 = (i2_0 + i2_2);
r1_16 = (r2_0 - r2_2);
i1_16 = (i2_0 - i2_2);
tmpr = ((0.923879532511 * r2_6) + (0.382683432365 * i2_6));
tmpi = ((0.923879532511 * i2_6) - (0.382683432365 * r2_6));
r1_2 = (r2_4 + tmpr);
i1_2 = (i2_4 + tmpi);
r1_18 = (r2_4 - tmpr);
i1_18 = (i2_4 - tmpi);
tmpr = (0.707106781187 * (r2_10 + i2_10));
tmpi = (0.707106781187 * (i2_10 - r2_10));
r1_4 = (r2_8 + tmpr);
i1_4 = (i2_8 + tmpi);
r1_20 = (r2_8 - tmpr);
i1_20 = (i2_8 - tmpi);
tmpr = ((0.382683432365 * r2_14) + (0.923879532511 * i2_14));
tmpi = ((0.382683432365 * i2_14) - (0.923879532511 * r2_14));
r1_6 = (r2_12 + tmpr);
i1_6 = (i2_12 + tmpi);
r1_22 = (r2_12 - tmpr);
i1_22 = (i2_12 - tmpi);
r1_8 = (r2_16 + i2_18);
i1_8 = (i2_16 - r2_18);
r1_24 = (r2_16 - i2_18);
i1_24 = (i2_16 + r2_18);
tmpr = ((0.923879532511 * i2_22) - (0.382683432365 * r2_22));
tmpi = ((0.923879532511 * r2_22) + (0.382683432365 * i2_22));
r1_10 = (r2_20 + tmpr);
i1_10 = (i2_20 - tmpi);
r1_26 = (r2_20 - tmpr);
i1_26 = (i2_20 + tmpi);
tmpr = (0.707106781187 * (i2_26 - r2_26));
tmpi = (0.707106781187 * (r2_26 + i2_26));
r1_12 = (r2_24 + tmpr);
i1_12 = (i2_24 - tmpi);
r1_28 = (r2_24 - tmpr);
i1_28 = (i2_24 + tmpi);
tmpr = ((0.382683432365 * i2_30) - (0.923879532511 * r2_30));
tmpi = ((0.382683432365 * r2_30) + (0.923879532511 * i2_30));
r1_14 = (r2_28 + tmpr);
i1_14 = (i2_28 - tmpi);
r1_30 = (r2_28 - tmpr);
i1_30 = (i2_28 + tmpi);
}
{
double r2_1;
double i2_1;
double r2_3;
double i2_3;
double r2_5;
double i2_5;
double r2_7;
double i2_7;
double r2_9;
double i2_9;
double r2_11;
double i2_11;
double r2_13;
double i2_13;
double r2_15;
double i2_15;
double r2_17;
double i2_17;
double r2_19;
double i2_19;
double r2_21;
double i2_21;
double r2_23;
double i2_23;
double r2_25;
double i2_25;
double r2_27;
double i2_27;
double r2_29;
double i2_29;
double r2_31;
double i2_31;
{
double r3_1;
double i3_1;
double r3_5;
double i3_5;
double r3_9;
double i3_9;
double r3_13;
double i3_13;
double r3_17;
double i3_17;
double r3_21;
double i3_21;
double r3_25;
double i3_25;
double r3_29;
double i3_29;
{
double r4_1;
double i4_1;
double r4_9;
double i4_9;
double r4_17;
double i4_17;
double r4_25;
double i4_25;
{
double r5_1;
double i5_1;
double r5_17;
double i5_17;
wr = ((W[1 * l1]).re);
wi = ((W[1 * l1]).im);
tmpr = ((jp[1 * m]).re);
tmpi = ((jp[1 * m]).im);
r5_1 = ((wr * tmpr) - (wi * tmpi));
i5_1 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[17 * l1]).re);
wi = ((W[17 * l1]).im);
tmpr = ((jp[17 * m]).re);
tmpi = ((jp[17 * m]).im);
r5_17 = ((wr * tmpr) - (wi * tmpi));
i5_17 = ((wi * tmpr) + (wr * tmpi));
r4_1 = (r5_1 + r5_17);
i4_1 = (i5_1 + i5_17);
r4_17 = (r5_1 - r5_17);
i4_17 = (i5_1 - i5_17);
}
{
double r5_9;
double i5_9;
double r5_25;
double i5_25;
wr = ((W[9 * l1]).re);
wi = ((W[9 * l1]).im);
tmpr = ((jp[9 * m]).re);
tmpi = ((jp[9 * m]).im);
r5_9 = ((wr * tmpr) - (wi * tmpi));
i5_9 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[25 * l1]).re);
wi = ((W[25 * l1]).im);
tmpr = ((jp[25 * m]).re);
tmpi = ((jp[25 * m]).im);
r5_25 = ((wr * tmpr) - (wi * tmpi));
i5_25 = ((wi * tmpr) + (wr * tmpi));
r4_9 = (r5_9 + r5_25);
i4_9 = (i5_9 + i5_25);
r4_25 = (r5_9 - r5_25);
i4_25 = (i5_9 - i5_25);
}
r3_1 = (r4_1 + r4_9);
i3_1 = (i4_1 + i4_9);
r3_17 = (r4_1 - r4_9);
i3_17 = (i4_1 - i4_9);
r3_9 = (r4_17 + i4_25);
i3_9 = (i4_17 - r4_25);
r3_25 = (r4_17 - i4_25);
i3_25 = (i4_17 + r4_25);
}
{
double r4_5;
double i4_5;
double r4_13;
double i4_13;
double r4_21;
double i4_21;
double r4_29;
double i4_29;
{
double r5_5;
double i5_5;
double r5_21;
double i5_21;
wr = ((W[5 * l1]).re);
wi = ((W[5 * l1]).im);
tmpr = ((jp[5 * m]).re);
tmpi = ((jp[5 * m]).im);
r5_5 = ((wr * tmpr) - (wi * tmpi));
i5_5 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[21 * l1]).re);
wi = ((W[21 * l1]).im);
tmpr = ((jp[21 * m]).re);
tmpi = ((jp[21 * m]).im);
r5_21 = ((wr * tmpr) - (wi * tmpi));
i5_21 = ((wi * tmpr) + (wr * tmpi));
r4_5 = (r5_5 + r5_21);
i4_5 = (i5_5 + i5_21);
r4_21 = (r5_5 - r5_21);
i4_21 = (i5_5 - i5_21);
}
{
double r5_13;
double i5_13;
double r5_29;
double i5_29;
wr = ((W[13 * l1]).re);
wi = ((W[13 * l1]).im);
tmpr = ((jp[13 * m]).re);
tmpi = ((jp[13 * m]).im);
r5_13 = ((wr * tmpr) - (wi * tmpi));
i5_13 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[29 * l1]).re);
wi = ((W[29 * l1]).im);
tmpr = ((jp[29 * m]).re);
tmpi = ((jp[29 * m]).im);
r5_29 = ((wr * tmpr) - (wi * tmpi));
i5_29 = ((wi * tmpr) + (wr * tmpi));
r4_13 = (r5_13 + r5_29);
i4_13 = (i5_13 + i5_29);
r4_29 = (r5_13 - r5_29);
i4_29 = (i5_13 - i5_29);
}
r3_5 = (r4_5 + r4_13);
i3_5 = (i4_5 + i4_13);
r3_21 = (r4_5 - r4_13);
i3_21 = (i4_5 - i4_13);
r3_13 = (r4_21 + i4_29);
i3_13 = (i4_21 - r4_29);
r3_29 = (r4_21 - i4_29);
i3_29 = (i4_21 + r4_29);
}
r2_1 = (r3_1 + r3_5);
i2_1 = (i3_1 + i3_5);
r2_17 = (r3_1 - r3_5);
i2_17 = (i3_1 - i3_5);
tmpr = (0.707106781187 * (r3_13 + i3_13));
tmpi = (0.707106781187 * (i3_13 - r3_13));
r2_5 = (r3_9 + tmpr);
i2_5 = (i3_9 + tmpi);
r2_21 = (r3_9 - tmpr);
i2_21 = (i3_9 - tmpi);
r2_9 = (r3_17 + i3_21);
i2_9 = (i3_17 - r3_21);
r2_25 = (r3_17 - i3_21);
i2_25 = (i3_17 + r3_21);
tmpr = (0.707106781187 * (i3_29 - r3_29));
tmpi = (0.707106781187 * (r3_29 + i3_29));
r2_13 = (r3_25 + tmpr);
i2_13 = (i3_25 - tmpi);
r2_29 = (r3_25 - tmpr);
i2_29 = (i3_25 + tmpi);
}
{
double r3_3;
double i3_3;
double r3_7;
double i3_7;
double r3_11;
double i3_11;
double r3_15;
double i3_15;
double r3_19;
double i3_19;
double r3_23;
double i3_23;
double r3_27;
double i3_27;
double r3_31;
double i3_31;
{
double r4_3;
double i4_3;
double r4_11;
double i4_11;
double r4_19;
double i4_19;
double r4_27;
double i4_27;
{
double r5_3;
double i5_3;
double r5_19;
double i5_19;
wr = ((W[3 * l1]).re);
wi = ((W[3 * l1]).im);
tmpr = ((jp[3 * m]).re);
tmpi = ((jp[3 * m]).im);
r5_3 = ((wr * tmpr) - (wi * tmpi));
i5_3 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[19 * l1]).re);
wi = ((W[19 * l1]).im);
tmpr = ((jp[19 * m]).re);
tmpi = ((jp[19 * m]).im);
r5_19 = ((wr * tmpr) - (wi * tmpi));
i5_19 = ((wi * tmpr) + (wr * tmpi));
r4_3 = (r5_3 + r5_19);
i4_3 = (i5_3 + i5_19);
r4_19 = (r5_3 - r5_19);
i4_19 = (i5_3 - i5_19);
}
{
double r5_11;
double i5_11;
double r5_27;
double i5_27;
wr = ((W[11 * l1]).re);
wi = ((W[11 * l1]).im);
tmpr = ((jp[11 * m]).re);
tmpi = ((jp[11 * m]).im);
r5_11 = ((wr * tmpr) - (wi * tmpi));
i5_11 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[27 * l1]).re);
wi = ((W[27 * l1]).im);
tmpr = ((jp[27 * m]).re);
tmpi = ((jp[27 * m]).im);
r5_27 = ((wr * tmpr) - (wi * tmpi));
i5_27 = ((wi * tmpr) + (wr * tmpi));
r4_11 = (r5_11 + r5_27);
i4_11 = (i5_11 + i5_27);
r4_27 = (r5_11 - r5_27);
i4_27 = (i5_11 - i5_27);
}
r3_3 = (r4_3 + r4_11);
i3_3 = (i4_3 + i4_11);
r3_19 = (r4_3 - r4_11);
i3_19 = (i4_3 - i4_11);
r3_11 = (r4_19 + i4_27);
i3_11 = (i4_19 - r4_27);
r3_27 = (r4_19 - i4_27);
i3_27 = (i4_19 + r4_27);
}
{
double r4_7;
double i4_7;
double r4_15;
double i4_15;
double r4_23;
double i4_23;
double r4_31;
double i4_31;
{
double r5_7;
double i5_7;
double r5_23;
double i5_23;
wr = ((W[7 * l1]).re);
wi = ((W[7 * l1]).im);
tmpr = ((jp[7 * m]).re);
tmpi = ((jp[7 * m]).im);
r5_7 = ((wr * tmpr) - (wi * tmpi));
i5_7 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[23 * l1]).re);
wi = ((W[23 * l1]).im);
tmpr = ((jp[23 * m]).re);
tmpi = ((jp[23 * m]).im);
r5_23 = ((wr * tmpr) - (wi * tmpi));
i5_23 = ((wi * tmpr) + (wr * tmpi));
r4_7 = (r5_7 + r5_23);
i4_7 = (i5_7 + i5_23);
r4_23 = (r5_7 - r5_23);
i4_23 = (i5_7 - i5_23);
}
{
double r5_15;
double i5_15;
double r5_31;
double i5_31;
wr = ((W[15 * l1]).re);
wi = ((W[15 * l1]).im);
tmpr = ((jp[15 * m]).re);
tmpi = ((jp[15 * m]).im);
r5_15 = ((wr * tmpr) - (wi * tmpi));
i5_15 = ((wi * tmpr) + (wr * tmpi));
wr = ((W[31 * l1]).re);
wi = ((W[31 * l1]).im);
tmpr = ((jp[31 * m]).re);
tmpi = ((jp[31 * m]).im);
r5_31 = ((wr * tmpr) - (wi * tmpi));
i5_31 = ((wi * tmpr) + (wr * tmpi));
r4_15 = (r5_15 + r5_31);
i4_15 = (i5_15 + i5_31);
r4_31 = (r5_15 - r5_31);
i4_31 = (i5_15 - i5_31);
}
r3_7 = (r4_7 + r4_15);
i3_7 = (i4_7 + i4_15);
r3_23 = (r4_7 - r4_15);
i3_23 = (i4_7 - i4_15);
r3_15 = (r4_23 + i4_31);
i3_15 = (i4_23 - r4_31);
r3_31 = (r4_23 - i4_31);
i3_31 = (i4_23 + r4_31);
}
r2_3 = (r3_3 + r3_7);
i2_3 = (i3_3 + i3_7);
r2_19 = (r3_3 - r3_7);
i2_19 = (i3_3 - i3_7);
tmpr = (0.707106781187 * (r3_15 + i3_15));
tmpi = (0.707106781187 * (i3_15 - r3_15));
r2_7 = (r3_11 + tmpr);
i2_7 = (i3_11 + tmpi);
r2_23 = (r3_11 - tmpr);
i2_23 = (i3_11 - tmpi);
r2_11 = (r3_19 + i3_23);
i2_11 = (i3_19 - r3_23);
r2_27 = (r3_19 - i3_23);
i2_27 = (i3_19 + r3_23);
tmpr = (0.707106781187 * (i3_31 - r3_31));
tmpi = (0.707106781187 * (r3_31 + i3_31));
r2_15 = (r3_27 + tmpr);
i2_15 = (i3_27 - tmpi);
r2_31 = (r3_27 - tmpr);
i2_31 = (i3_27 + tmpi);
}
r1_1 = (r2_1 + r2_3);
i1_1 = (i2_1 + i2_3);
r1_17 = (r2_1 - r2_3);
i1_17 = (i2_1 - i2_3);
tmpr = ((0.923879532511 * r2_7) + (0.382683432365 * i2_7));
tmpi = ((0.923879532511 * i2_7) - (0.382683432365 * r2_7));
r1_3 = (r2_5 + tmpr);
i1_3 = (i2_5 + tmpi);
r1_19 = (r2_5 - tmpr);
i1_19 = (i2_5 - tmpi);
tmpr = (0.707106781187 * (r2_11 + i2_11));
tmpi = (0.707106781187 * (i2_11 - r2_11));
r1_5 = (r2_9 + tmpr);
i1_5 = (i2_9 + tmpi);
r1_21 = (r2_9 - tmpr);
i1_21 = (i2_9 - tmpi);
tmpr = ((0.382683432365 * r2_15) + (0.923879532511 * i2_15));
tmpi = ((0.382683432365 * i2_15) - (0.923879532511 * r2_15));
r1_7 = (r2_13 + tmpr);
i1_7 = (i2_13 + tmpi);
r1_23 = (r2_13 - tmpr);
i1_23 = (i2_13 - tmpi);
r1_9 = (r2_17 + i2_19);
i1_9 = (i2_17 - r2_19);
r1_25 = (r2_17 - i2_19);
i1_25 = (i2_17 + r2_19);
tmpr = ((0.923879532511 * i2_23) - (0.382683432365 * r2_23));
tmpi = ((0.923879532511 * r2_23) + (0.382683432365 * i2_23));
r1_11 = (r2_21 + tmpr);
i1_11 = (i2_21 - tmpi);
r1_27 = (r2_21 - tmpr);
i1_27 = (i2_21 + tmpi);
tmpr = (0.707106781187 * (i2_27 - r2_27));
tmpi = (0.707106781187 * (r2_27 + i2_27));
r1_13 = (r2_25 + tmpr);
i1_13 = (i2_25 - tmpi);
r1_29 = (r2_25 - tmpr);
i1_29 = (i2_25 + tmpi);
tmpr = ((0.382683432365 * i2_31) - (0.923879532511 * r2_31));
tmpi = ((0.382683432365 * r2_31) + (0.923879532511 * i2_31));
r1_15 = (r2_29 + tmpr);
i1_15 = (i2_29 - tmpi);
r1_31 = (r2_29 - tmpr);
i1_31 = (i2_29 + tmpi);
}
((kp[0 * m]).re) = (r1_0 + r1_1);
((kp[0 * m]).im) = (i1_0 + i1_1);
((kp[16 * m]).re) = (r1_0 - r1_1);
((kp[16 * m]).im) = (i1_0 - i1_1);
tmpr = ((0.980785280403 * r1_3) + (0.195090322016 * i1_3));
tmpi = ((0.980785280403 * i1_3) - (0.195090322016 * r1_3));
((kp[1 * m]).re) = (r1_2 + tmpr);
((kp[1 * m]).im) = (i1_2 + tmpi);
((kp[17 * m]).re) = (r1_2 - tmpr);
((kp[17 * m]).im) = (i1_2 - tmpi);
tmpr = ((0.923879532511 * r1_5) + (0.382683432365 * i1_5));
tmpi = ((0.923879532511 * i1_5) - (0.382683432365 * r1_5));
((kp[2 * m]).re) = (r1_4 + tmpr);
((kp[2 * m]).im) = (i1_4 + tmpi);
((kp[18 * m]).re) = (r1_4 - tmpr);
((kp[18 * m]).im) = (i1_4 - tmpi);
tmpr = ((0.831469612303 * r1_7) + (0.55557023302 * i1_7));
tmpi = ((0.831469612303 * i1_7) - (0.55557023302 * r1_7));
((kp[3 * m]).re) = (r1_6 + tmpr);
((kp[3 * m]).im) = (i1_6 + tmpi);
((kp[19 * m]).re) = (r1_6 - tmpr);
((kp[19 * m]).im) = (i1_6 - tmpi);
tmpr = (0.707106781187 * (r1_9 + i1_9));
tmpi = (0.707106781187 * (i1_9 - r1_9));
((kp[4 * m]).re) = (r1_8 + tmpr);
((kp[4 * m]).im) = (i1_8 + tmpi);
((kp[20 * m]).re) = (r1_8 - tmpr);
((kp[20 * m]).im) = (i1_8 - tmpi);
tmpr = ((0.55557023302 * r1_11) + (0.831469612303 * i1_11));
tmpi = ((0.55557023302 * i1_11) - (0.831469612303 * r1_11));
((kp[5 * m]).re) = (r1_10 + tmpr);
((kp[5 * m]).im) = (i1_10 + tmpi);
((kp[21 * m]).re) = (r1_10 - tmpr);
((kp[21 * m]).im) = (i1_10 - tmpi);
tmpr = ((0.382683432365 * r1_13) + (0.923879532511 * i1_13));
tmpi = ((0.382683432365 * i1_13) - (0.923879532511 * r1_13));
((kp[6 * m]).re) = (r1_12 + tmpr);
((kp[6 * m]).im) = (i1_12 + tmpi);
((kp[22 * m]).re) = (r1_12 - tmpr);
((kp[22 * m]).im) = (i1_12 - tmpi);
tmpr = ((0.195090322016 * r1_15) + (0.980785280403 * i1_15));
tmpi = ((0.195090322016 * i1_15) - (0.980785280403 * r1_15));
((kp[7 * m]).re) = (r1_14 + tmpr);
((kp[7 * m]).im) = (i1_14 + tmpi);
((kp[23 * m]).re) = (r1_14 - tmpr);
((kp[23 * m]).im) = (i1_14 - tmpi);
((kp[8 * m]).re) = (r1_16 + i1_17);
((kp[8 * m]).im) = (i1_16 - r1_17);
((kp[24 * m]).re) = (r1_16 - i1_17);
((kp[24 * m]).im) = (i1_16 + r1_17);
tmpr = ((0.980785280403 * i1_19) - (0.195090322016 * r1_19));
tmpi = ((0.980785280403 * r1_19) + (0.195090322016 * i1_19));
((kp[9 * m]).re) = (r1_18 + tmpr);
((kp[9 * m]).im) = (i1_18 - tmpi);
((kp[25 * m]).re) = (r1_18 - tmpr);
((kp[25 * m]).im) = (i1_18 + tmpi);
tmpr = ((0.923879532511 * i1_21) - (0.382683432365 * r1_21));
tmpi = ((0.923879532511 * r1_21) + (0.382683432365 * i1_21));
((kp[10 * m]).re) = (r1_20 + tmpr);
((kp[10 * m]).im) = (i1_20 - tmpi);
((kp[26 * m]).re) = (r1_20 - tmpr);
((kp[26 * m]).im) = (i1_20 + tmpi);
tmpr = ((0.831469612303 * i1_23) - (0.55557023302 * r1_23));
tmpi = ((0.831469612303 * r1_23) + (0.55557023302 * i1_23));
((kp[11 * m]).re) = (r1_22 + tmpr);
((kp[11 * m]).im) = (i1_22 - tmpi);
((kp[27 * m]).re) = (r1_22 - tmpr);
((kp[27 * m]).im) = (i1_22 + tmpi);
tmpr = (0.707106781187 * (i1_25 - r1_25));
tmpi = (0.707106781187 * (r1_25 + i1_25));
((kp[12 * m]).re) = (r1_24 + tmpr);
((kp[12 * m]).im) = (i1_24 - tmpi);
((kp[28 * m]).re) = (r1_24 - tmpr);
((kp[28 * m]).im) = (i1_24 + tmpi);
tmpr = ((0.55557023302 * i1_27) - (0.831469612303 * r1_27));
tmpi = ((0.55557023302 * r1_27) + (0.831469612303 * i1_27));
((kp[13 * m]).re) = (r1_26 + tmpr);
((kp[13 * m]).im) = (i1_26 - tmpi);
((kp[29 * m]).re) = (r1_26 - tmpr);
((kp[29 * m]).im) = (i1_26 + tmpi);
tmpr = ((0.382683432365 * i1_29) - (0.923879532511 * r1_29));
tmpi = ((0.382683432365 * r1_29) + (0.923879532511 * i1_29));
((kp[14 * m]).re) = (r1_28 + tmpr);
((kp[14 * m]).im) = (i1_28 - tmpi);
((kp[30 * m]).re) = (r1_28 - tmpr);
((kp[30 * m]).im) = (i1_28 + tmpi);
tmpr = ((0.195090322016 * i1_31) - (0.980785280403 * r1_31));
tmpi = ((0.195090322016 * r1_31) + (0.980785280403 * i1_31));
((kp[15 * m]).re) = (r1_30 + tmpr);
((kp[15 * m]).im) = (i1_30 - tmpi);
((kp[31 * m]).re) = (r1_30 - tmpr);
((kp[31 * m]).im) = (i1_30 + tmpi);
}
}
}
else
{
int ab = (a + b) / 2;
fft_twiddle_32_seq(a, ab, in, out, W, nW, nWdn, m);
fft_twiddle_32_seq(ab, b, in, out, W, nW, nWdn, m);
}
}
static void * _taskFunc25_(void *);
static void * _taskFunc26_(void *);
void fft_unshuffle_32(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 32;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
_taskFunc25_((void *)0);
_taskFunc26_((void *)0);
ort_taskwait(0);
}
}
static void * _taskFunc26_(void * __arg)
{
struct __taskenv__ {
int ab;
int b;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int ab = _tenv->ab;
int b = _tenv->b;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_32(ab, b, in, out, m);
CANCEL_task_4520 :
;
}
ort_taskenv_free(_tenv, _taskFunc26_);
return ((void *) 0);
}
static void * _taskFunc25_(void * __arg)
{
struct __taskenv__ {
int a;
int ab;
struct _noname19_ (* in);
struct _noname19_ (* out);
int m;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int a = _tenv->a;
int ab = _tenv->ab;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
int m = _tenv->m;
{
fft_unshuffle_32(a, ab, in, out, m);
CANCEL_task_4518 :
;
}
ort_taskenv_free(_tenv, _taskFunc25_);
return ((void *) 0);
}
void fft_unshuffle_32_seq(int a, int b, struct _noname19_ (* in), struct _noname19_ (* out), int m)
{
int i;
const struct _noname19_ (* ip);
struct _noname19_ (* jp);
if ((b - a) < 128)
{
ip = in + a * 32;
for (i = a; i < b; ++i)
{
jp = out + i;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
jp += 2 * m;
jp[0] = ip[0];
jp[m] = ip[1];
ip += 2;
}
}
else
{
int ab = (a + b) / 2;
fft_unshuffle_32_seq(a, ab, in, out, m);
fft_unshuffle_32_seq(ab, b, in, out, m);
}
}
static void * _taskFunc27_(void *);
static void * _taskFunc28_(void *);
static void * _taskFunc29_(void *);
static void * _taskFunc30_(void *);
static void * _taskFunc31_(void *);
static void * _taskFunc32_(void *);
static void * _taskFunc33_(void *);
static void * _taskFunc34_(void *);
static void * _taskFunc35_(void *);
static void * _taskFunc36_(void *);
static void * _taskFunc37_(void *);
static void * _taskFunc38_(void *);
void fft_aux(int n, struct _noname19_ (* in), struct _noname19_ (* out), int * factors, struct _noname19_ (* W), int nW)
{
int r, m;
int k;
if (n == 32)
{
fft_base_32(in, out);
return;
}
if (n == 16)
{
fft_base_16(in, out);
return;
}
if (n == 8)
{
fft_base_8(in, out);
return;
}
if (n == 4)
{
fft_base_4(in, out);
return;
}
if (n == 2)
{
fft_base_2(in, out);
return;
}
r = *factors;
m = n / r;
if (r < n)
{
if (r == 32)
{
_taskFunc27_((void *)0);
}
else
if (r == 16)
{
_taskFunc28_((void *)0);
}
else
if (r == 8)
{
_taskFunc29_((void *)0);
}
else
if (r == 4)
{
_taskFunc30_((void *)0);
}
else
if (r == 2)
{
_taskFunc31_((void *)0);
}
else
unshuffle(0, m, in, out, r, m);
ort_taskwait(0);
for (k = 0; k < n; k += m)
{
_taskFunc32_((void *)0);
}
ort_taskwait(0);
}
if (r == 2)
{
_taskFunc33_((void *)0);
}
else
if (r == 4)
{
_taskFunc34_((void *)0);
}
else
if (r == 8)
{
_taskFunc35_((void *)0);
}
else
if (r == 16)
{
_taskFunc36_((void *)0);
}
else
if (r == 32)
{
_taskFunc37_((void *)0);
}
else
{
_taskFunc38_((void *)0);
}
ort_taskwait(0);
return;
}
static void * _taskFunc38_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int n;
int r;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int n = _tenv->n;
int r = _tenv->r;
{
fft_twiddle_gen(0, m, in, out, W, nW, nW / n, r, m);
CANCEL_task_4705 :
;
}
ort_taskenv_free(_tenv, _taskFunc38_);
return ((void *) 0);
}
static void * _taskFunc37_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int n;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int n = _tenv->n;
{
fft_twiddle_32(0, m, in, out, W, nW, nW / n, m);
CANCEL_task_4702 :
;
}
ort_taskenv_free(_tenv, _taskFunc37_);
return ((void *) 0);
}
static void * _taskFunc36_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int n;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int n = _tenv->n;
{
fft_twiddle_16(0, m, in, out, W, nW, nW / n, m);
CANCEL_task_4699 :
;
}
ort_taskenv_free(_tenv, _taskFunc36_);
return ((void *) 0);
}
static void * _taskFunc35_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int n;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int n = _tenv->n;
{
fft_twiddle_8(0, m, in, out, W, nW, nW / n, m);
CANCEL_task_4696 :
;
}
ort_taskenv_free(_tenv, _taskFunc35_);
return ((void *) 0);
}
static void * _taskFunc34_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int n;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int n = _tenv->n;
{
fft_twiddle_4(0, m, in, out, W, nW, nW / n, m);
CANCEL_task_4693 :
;
}
ort_taskenv_free(_tenv, _taskFunc34_);
return ((void *) 0);
}
static void * _taskFunc33_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
struct _noname19_ (* W);
int nW;
int n;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
int n = _tenv->n;
{
fft_twiddle_2(0, m, in, out, W, nW, nW / n, m);
CANCEL_task_4690 :
;
}
ort_taskenv_free(_tenv, _taskFunc33_);
return ((void *) 0);
}
static void * _taskFunc32_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* out);
int k;
struct _noname19_ (* in);
int * factors;
struct _noname19_ (* W);
int nW;
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* out) = _tenv->out;
int k = _tenv->k;
struct _noname19_ (* in) = _tenv->in;
int * factors = _tenv->factors;
struct _noname19_ (* W) = _tenv->W;
int nW = _tenv->nW;
{
fft_aux(m, out + k, in + k, factors + 1, W, nW);
CANCEL_task_4680 :
;
}
ort_taskenv_free(_tenv, _taskFunc32_);
return ((void *) 0);
}
static void * _taskFunc31_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
{
fft_unshuffle_2(0, m, in, out, m);
CANCEL_task_4672 :
;
}
ort_taskenv_free(_tenv, _taskFunc31_);
return ((void *) 0);
}
static void * _taskFunc30_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
{
fft_unshuffle_4(0, m, in, out, m);
CANCEL_task_4669 :
;
}
ort_taskenv_free(_tenv, _taskFunc30_);
return ((void *) 0);
}
static void * _taskFunc29_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
{
fft_unshuffle_8(0, m, in, out, m);
CANCEL_task_4666 :
;
}
ort_taskenv_free(_tenv, _taskFunc29_);
return ((void *) 0);
}
static void * _taskFunc28_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
{
fft_unshuffle_16(0, m, in, out, m);
CANCEL_task_4663 :
;
}
ort_taskenv_free(_tenv, _taskFunc28_);
return ((void *) 0);
}
static void * _taskFunc27_(void * __arg)
{
struct __taskenv__ {
int m;
struct _noname19_ (* in);
struct _noname19_ (* out);
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int m = _tenv->m;
struct _noname19_ (* in) = _tenv->in;
struct _noname19_ (* out) = _tenv->out;
{
fft_unshuffle_32(0, m, in, out, m);
CANCEL_task_4660 :
;
}
ort_taskenv_free(_tenv, _taskFunc27_);
return ((void *) 0);
}
void fft_aux_seq(int n, struct _noname19_ (* in), struct _noname19_ (* out), int * factors, struct _noname19_ (* W), int nW)
{
int r, m;
int k;
if (n == 32)
{
fft_base_32(in, out);
return;
}
if (n == 16)
{
fft_base_16(in, out);
return;
}
if (n == 8)
{
fft_base_8(in, out);
return;
}
if (n == 4)
{
fft_base_4(in, out);
return;
}
if (n == 2)
{
fft_base_2(in, out);
return;
}
r = *factors;
m = n / r;
if (r < n)
{
if (r == 32)
fft_unshuffle_32_seq(0, m, in, out, m);
else
if (r == 16)
fft_unshuffle_16_seq(0, m, in, out, m);
else
if (r == 8)
fft_unshuffle_8_seq(0, m, in, out, m);
else
if (r == 4)
fft_unshuffle_4_seq(0, m, in, out, m);
else
if (r == 2)
fft_unshuffle_2_seq(0, m, in, out, m);
else
unshuffle_seq(0, m, in, out, r, m);
for (k = 0; k < n; k += m)
{
fft_aux_seq(m, out + k, in + k, factors + 1, W, nW);
}
}
if (r == 2)
fft_twiddle_2_seq(0, m, in, out, W, nW, nW / n, m);
else
if (r == 4)
fft_twiddle_4_seq(0, m, in, out, W, nW, nW / n, m);
else
if (r == 8)
fft_twiddle_8_seq(0, m, in, out, W, nW, nW / n, m);
else
if (r == 16)
fft_twiddle_16_seq(0, m, in, out, W, nW, nW / n, m);
else
if (r == 32)
fft_twiddle_32_seq(0, m, in, out, W, nW, nW / n, m);
else
fft_twiddle_gen_seq(0, m, in, out, W, nW, nW / n, r, m);
return;
}
static void * _taskFunc39_(void *);
static void * _thrFunc0_(void *);
static void * _taskFunc40_(void *);
static void * _thrFunc1_(void *);
void fft(int n, struct _noname19_ (* in), struct _noname19_ (* out))
{
int factors[ 40];
int * p = factors;
int l = n;
int r;
struct _noname19_ (* W);
{
if (bots_verbose_mode >= BOTS_VERBOSE_DEFAULT)
{
fprintf(stdout, "Computing coefficients ");
}
}
;
W = (struct _noname19_ (*)) malloc((n + 1) * sizeof(struct _noname19_ ));
{
struct __shvt__ {
int (* n);
struct _noname19_ (* (* W));
} _shvars;
_shvars.n = &n;
_shvars.W = &W;
ort_execute_parallel(_thrFunc0_, (void *) &_shvars, -1, 0, 1);
}
{
if (bots_verbose_mode >= BOTS_VERBOSE_DEFAULT)
{
fprintf(stdout, " completed!\n");
}
}
;
do
{
r = factor(l);
*p++ = r;
l /= r;
}
while (l > 1);
{
if (bots_verbose_mode >= BOTS_VERBOSE_DEFAULT)
{
fprintf(stdout, "Computing FFT ");
}
}
;
{
struct __shvt__ {
int (* n);
struct _noname19_ (* (* in));
struct _noname19_ (* (* out));
int (* factors)[ 40];
struct _noname19_ (* (* W));
} _shvars;
_shvars.n = &n;
_shvars.in = &in;
_shvars.out = &out;
_shvars.factors = &factors;
_shvars.W = &W;
ort_execute_parallel(_thrFunc1_, (void *) &_shvars, -1, 0, 1);
}
{
if (bots_verbose_mode >= BOTS_VERBOSE_DEFAULT)
{
fprintf(stdout, " completed!\n");
}
}
;
free(W);
return;
}
static void * _thrFunc1_(void * __arg)
{
struct __shvt__ {
int (* n);
struct _noname19_ (* (* in));
struct _noname19_ (* (* out));
int (* factors)[ 40];
struct _noname19_ (* (* W));
};
struct __shvt__ * _shvars = (struct __shvt__ *) __arg;
int (* n) = _shvars->n;
struct _noname19_ (* (* in)) = _shvars->in;
struct _noname19_ (* (* out)) = _shvars->out;
int (* factors)[ 40] = _shvars->factors;
struct _noname19_ (* (* W)) = _shvars->W;
{
if (ort_mysingle(1))
_taskFunc40_((void *)0);
ort_leaving_single();
}
CANCEL_parallel_4806 :
ort_taskwait(2);
return ((void *) 0);
}
static void * _taskFunc40_(void * __arg)
{
struct __taskenv__ {
int (* n);
struct _noname19_ (* (* in));
struct _noname19_ (* (* out));
int (* factors)[ 40];
struct _noname19_ (* (* W));
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int (* n) = _tenv->n;
struct _noname19_ (* (* in)) = _tenv->in;
struct _noname19_ (* (* out)) = _tenv->out;
int (* factors)[ 40] = _tenv->factors;
struct _noname19_ (* (* W)) = _tenv->W;
{
fft_aux((*n), (*in), (*out), (*factors), (*W), (*n));
CANCEL_task_4808 :
;
}
ort_taskenv_free(_tenv, _taskFunc40_);
return ((void *) 0);
}
static void * _thrFunc0_(void * __arg)
{
struct __shvt__ {
int (* n);
struct _noname19_ (* (* W));
};
struct __shvt__ * _shvars = (struct __shvt__ *) __arg;
int (* n) = _shvars->n;
struct _noname19_ (* (* W)) = _shvars->W;
{
if (ort_mysingle(1))
_taskFunc39_((void *)0);
ort_leaving_single();
}
CANCEL_parallel_4789 :
ort_taskwait(2);
return ((void *) 0);
}
static void * _taskFunc39_(void * __arg)
{
struct __taskenv__ {
int (* n);
struct _noname19_ (* (* W));
};
struct __taskenv__ * _tenv = (struct __taskenv__ *) __arg;
int (* n) = _tenv->n;
struct _noname19_ (* (* W)) = _tenv->W;
{
compute_w_coefficients((*n), 0, (*n) / 2, (*W));
CANCEL_task_4791 :
;
}
ort_taskenv_free(_tenv, _taskFunc39_);
return ((void *) 0);
}
void fft_seq(int n, struct _noname19_ (* in), struct _noname19_ (* out))
{
int factors[ 40];
int * p = factors;
int l = n;
int r;
struct _noname19_ (* W);
W = (struct _noname19_ (*)) malloc((n + 1) * sizeof(struct _noname19_ ));
compute_w_coefficients_seq(n, 0, n / 2, W);
do
{
r = factor(l);
*p++ = r;
l /= r;
}
while (l > 1);
fft_aux_seq(n, in, out, factors, W, n);
free(W);
return;
}
int test_correctness(int n, struct _noname19_ (* out1), struct _noname19_ (* out2))
{
int i;
double a, d, error = 0.0;
for (i = 0; i < n; ++i)
{
a = sqrt((((out1[i]).re) - ((out2[i]).re)) * (((out1[i]).re) - ((out2[i]).re)) + (((out1[i]).im) - ((out2[i]).im)) * (((out1[i]).im) - ((out2[i]).im)));
d = sqrt(((out2[i]).re) * ((out2[i]).re) + ((out2[i]).im) * ((out2[i]).im));
if (d < -1.0e-10 || d > 1.0e-10)
a /= d;
if (a > error)
error = a;
}
{
if (bots_verbose_mode >= BOTS_VERBOSE_DEFAULT)
{
fprintf(stdout, "relative error=%e\n", error);
}
}
;
if (error > 1e-3)
return (2);
else
return (1);
}
