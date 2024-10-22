#ifndef _PTHREAD_H
#define	_PTHREAD_H
#pragma ident	"@(#)pthread.h	1.26	98/04/12 SMI"
#ifndef	_ASM
#include <sys/types.h>
#include <time.h>
#include <sched.h>
#endif	
#ifdef	__cplusplus
extern "C" {
#endif
#define	PTHREAD_CREATE_DETACHED		0x40	
#define	PTHREAD_CREATE_JOINABLE		0
#define	PTHREAD_SCOPE_SYSTEM		0x01	
#define	PTHREAD_SCOPE_PROCESS		0
#define	PTHREAD_INHERIT_SCHED		1
#define	PTHREAD_EXPLICIT_SCHED		0
#define	PTHREAD_PROCESS_SHARED		1	
#define	PTHREAD_PROCESS_PRIVATE		0	
#define	DEFAULT_TYPE			PTHREAD_PROCESS_PRIVATE
#define	PTHREAD_MUTEX_NORMAL		0x0
#define	PTHREAD_MUTEX_ERRORCHECK	0x2
#define	PTHREAD_MUTEX_RECURSIVE		0x4
#define	PTHREAD_MUTEX_DEFAULT		PTHREAD_MUTEX_NORMAL
#define	PTHREAD_MUTEX_INITIALIZER	{{{0}, 0}, {{{0}}}, 0}
#define	PTHREAD_COND_INITIALIZER	{{{0}, 0}, 0}	
#define	PTHREAD_RWLOCK_INITIALIZER	{0, 0, 0, {0, 0, 0}, {0, 0}, {0, 0}}
#define	PTHREAD_CANCEL_ENABLE		0x00
#define	PTHREAD_CANCEL_DISABLE		0x01
#define	PTHREAD_CANCEL_DEFERRED		0x00
#define	PTHREAD_CANCEL_ASYNCHRONOUS	0x02
#define	PTHREAD_CANCELED		(void *)-19
#define	PTHREAD_ONCE_NOTDONE	0
#define	PTHREAD_ONCE_DONE	1
#define	PTHREAD_ONCE_INIT	{0, 0, 0, PTHREAD_ONCE_NOTDONE}
#ifndef	_ASM
typedef struct _cleanup {
uintptr_t	pthread_cleanup_pad[4];
} _cleanup_t;
#ifdef	__STDC__
void	__pthread_cleanup_push(void (*routine)(void *), void *args,
caddr_t fp, _cleanup_t *info);
void	__pthread_cleanup_pop(int ex, _cleanup_t *info);
caddr_t	_getfp(void);
#else	
void	__pthread_cleanup_push();
void	__pthread_cleanup_pop();
caddr_t	_getfp();
#endif	
#define	pthread_cleanup_push(routine, args) { \
_cleanup_t _cleanup_info; \
__pthread_cleanup_push((void (*)(void *))(routine), (void *)(args), \
(caddr_t)_getfp(), &_cleanup_info);
#define	pthread_cleanup_pop(ex) \
__pthread_cleanup_pop(ex, &_cleanup_info); \
}
#ifdef	__STDC__
extern int pthread_attr_init(pthread_attr_t *);
extern int pthread_attr_destroy(pthread_attr_t *);
extern int pthread_attr_setstacksize(pthread_attr_t *, size_t);
extern int pthread_attr_getstacksize(const pthread_attr_t *, size_t *);
extern int pthread_attr_setstackaddr(pthread_attr_t *, void *);
extern int pthread_attr_getstackaddr(const pthread_attr_t *, void **);
extern int pthread_attr_setdetachstate(pthread_attr_t *, int);
extern int pthread_attr_getdetachstate(const pthread_attr_t *, int *);
extern int pthread_attr_setscope(pthread_attr_t *, int);
extern int pthread_attr_getscope(const pthread_attr_t *, int *);
extern int pthread_attr_setinheritsched(pthread_attr_t *, int);
extern int pthread_attr_getinheritsched(const pthread_attr_t *, int *);
extern int pthread_attr_setschedpolicy(pthread_attr_t *, int);
extern int pthread_attr_getschedpolicy(const pthread_attr_t *, int *);
extern int pthread_attr_setschedparam(pthread_attr_t *,
const struct sched_param *);
extern int pthread_attr_getschedparam(const pthread_attr_t *,
struct sched_param *);
extern int pthread_create(pthread_t *, const pthread_attr_t *,
void * (*)(void *), void *);
extern int pthread_once(pthread_once_t *, void (*)(void));
extern int pthread_join(pthread_t, void **);
extern int pthread_detach(pthread_t);
extern void pthread_exit(void *);
extern int pthread_cancel(pthread_t);
extern int pthread_setschedparam(pthread_t, int, const struct sched_param *);
extern int pthread_getschedparam(pthread_t, int *, struct sched_param *);
extern int pthread_setcancelstate(int, int *);
extern int pthread_setcanceltype(int, int *);
extern void pthread_testcancel(void);
extern int pthread_equal(pthread_t, pthread_t);
extern int pthread_key_create(pthread_key_t *, void (*)(void *));
extern int pthread_key_delete(pthread_key_t);
extern int pthread_setspecific(pthread_key_t, const void *);
extern void *pthread_getspecific(pthread_key_t);
extern pthread_t pthread_self(void);
extern int pthread_mutexattr_init(pthread_mutexattr_t *);
extern int pthread_mutexattr_destroy(pthread_mutexattr_t *);
extern int pthread_mutexattr_setpshared(pthread_mutexattr_t *, int);
extern int pthread_mutexattr_getpshared(const pthread_mutexattr_t *, int *);
extern int pthread_mutexattr_setprotocol(pthread_mutexattr_t *, int);
extern int pthread_mutexattr_getprotocol(const pthread_mutexattr_t *, int *);
extern int pthread_mutexattr_setprioceiling(pthread_mutexattr_t *, int);
extern int pthread_mutexattr_getprioceiling(const pthread_mutexattr_t *, int *);
extern int pthread_mutex_init(pthread_mutex_t *, const pthread_mutexattr_t *);
extern int pthread_mutex_destroy(pthread_mutex_t *);
extern int pthread_mutex_lock(pthread_mutex_t *);
extern int pthread_mutex_unlock(pthread_mutex_t *);
extern int pthread_mutex_trylock(pthread_mutex_t *);
extern int pthread_mutex_setprioceiling(pthread_mutex_t *, int, int *);
extern int pthread_mutex_getprioceiling(pthread_mutex_t *, int *);
extern int pthread_condattr_init(pthread_condattr_t *);
extern int pthread_condattr_destroy(pthread_condattr_t *);
extern int pthread_condattr_setpshared(pthread_condattr_t *, int);
extern int pthread_condattr_getpshared(const pthread_condattr_t *, int *);
extern int pthread_cond_init(pthread_cond_t *, const pthread_condattr_t *);
extern int pthread_cond_destroy(pthread_cond_t *);
extern int pthread_cond_broadcast(pthread_cond_t *);
extern int pthread_cond_signal(pthread_cond_t *);
extern int pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);
extern int pthread_cond_timedwait(pthread_cond_t *, pthread_mutex_t *,
const struct timespec *);
extern int pthread_attr_getguardsize(const pthread_attr_t *, size_t *);
extern int pthread_attr_setguardsize(pthread_attr_t *, size_t);
extern int pthread_getconcurrency(void);
extern int pthread_setconcurrency(int newval);
extern int pthread_mutexattr_settype(pthread_mutexattr_t *, int);
extern int pthread_mutexattr_gettype(const pthread_mutexattr_t *, int *);
extern int pthread_rwlock_init(pthread_rwlock_t *,
const pthread_rwlockattr_t *);
extern int pthread_rwlock_destroy(pthread_rwlock_t *);
extern int pthread_rwlock_rdlock(pthread_rwlock_t *);
extern int pthread_rwlock_tryrdlock(pthread_rwlock_t *);
extern int pthread_rwlock_wrlock(pthread_rwlock_t *);
extern int pthread_rwlock_trywrlock(pthread_rwlock_t *);
extern int pthread_rwlock_unlock(pthread_rwlock_t *);
extern int pthread_rwlockattr_init(pthread_rwlockattr_t *);
extern int pthread_rwlockattr_destroy(pthread_rwlockattr_t *);
extern int pthread_rwlockattr_getpshared(const pthread_rwlockattr_t *, int *);
extern int pthread_rwlockattr_setpshared(pthread_rwlockattr_t *, int);
#else	
extern int pthread_attr_init();
extern int pthread_attr_destroy();
extern int pthread_attr_setstacksize();
extern int pthread_attr_getstacksize();
extern int pthread_attr_setstackaddr();
extern int pthread_attr_getstackaddr();
extern int pthread_attr_setdetachstate();
extern int pthread_attr_getdetachstate();
extern int pthread_attr_setscope();
extern int pthread_attr_getscope();
extern int pthread_attr_setinheritsched();
extern int pthread_attr_getinheritsched();
extern int pthread_attr_setschedpolicy();
extern int pthread_attr_getschedpolicy();
extern int pthread_attr_setschedparam();
extern int pthread_attr_getschedparam();
extern int pthread_create();
extern int pthread_once();
extern int pthread_join();
extern int pthread_detach();
extern void pthread_exit();
extern int pthread_cancel();
extern int pthread_setschedparam();
extern int pthread_getschedparam();
extern int pthread_setcancelstate();
extern int pthread_setcanceltype();
extern void pthread_testcancel();
extern int pthread_equal();
extern int pthread_key_create();
extern int pthread_key_delete();
extern int pthread_setspecific();
extern void *pthread_getspecific();
extern pthread_t pthread_self();
extern int pthread_mutexattr_init();
extern int pthread_mutexattr_destroy();
extern int pthread_mutexattr_setpshared();
extern int pthread_mutexattr_getpshared();
extern int pthread_mutexattr_setprotocol();
extern int pthread_mutexattr_getprotocol();
extern int pthread_mutexattr_setprioceiling();
extern int pthread_mutexattr_getprioceiling();
extern int pthread_mutex_init();
extern int pthread_mutex_destroy();
extern int pthread_mutex_lock();
extern int pthread_mutex_unlock();
extern int pthread_mutex_trylock();
extern int pthread_mutex_setprioceiling();
extern int pthread_mutex_getprioceiling();
extern int pthread_condattr_init();
extern int pthread_condattr_destroy();
extern int pthread_condattr_setpshared();
extern int pthread_condattr_getpshared();
extern int pthread_cond_init();
extern int pthread_cond_destroy();
extern int pthread_cond_broadcast();
extern int pthread_cond_signal();
extern int pthread_cond_wait();
extern int pthread_cond_timedwait();
extern int pthread_attr_getguardsize();
extern int pthread_attr_setguardsize();
extern int pthread_getconcurrency();
extern int pthread_setconcurrency();
extern int pthread_mutexattr_settype();
extern int pthread_mutexattr_gettype();
extern int pthread_rwlock_init();
extern int pthread_rwlock_destroy();
extern int pthread_rwlock_rdlock();
extern int pthread_rwlock_tryrdlock();
extern int pthread_rwlock_wrlock();
extern int pthread_rwlock_trywrlock();
extern int pthread_rwlock_unlock();
extern int pthread_rwlockattr_init();
extern int pthread_rwlockattr_destroy();
extern int pthread_rwlockattr_getpshared();
extern int pthread_rwlockattr_setpshared();
#endif	
#endif	
#ifdef	__cplusplus
}
#endif
#endif	
