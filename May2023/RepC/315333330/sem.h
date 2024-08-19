#ifndef GOMP_SEM_H
#define GOMP_SEM_H 1
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility push(default)
#endif
#include <semaphore.h>
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility pop
#endif
#ifdef HAVE_BROKEN_POSIX_SEMAPHORES
#include <pthread.h>
struct gomp_sem
{
pthread_mutex_t	mutex;
pthread_cond_t	cond;
int			value;
};
typedef struct gomp_sem gomp_sem_t;
extern void gomp_sem_init (gomp_sem_t *sem, int value);
extern void gomp_sem_wait (gomp_sem_t *sem);
extern void gomp_sem_post (gomp_sem_t *sem);
extern void gomp_sem_destroy (gomp_sem_t *sem);
#else 
typedef sem_t gomp_sem_t;
static inline void gomp_sem_init (gomp_sem_t *sem, int value)
{
sem_init (sem, 0, value);
}
extern void gomp_sem_wait (gomp_sem_t *sem);
static inline void gomp_sem_post (gomp_sem_t *sem)
{
sem_post (sem);
}
static inline void gomp_sem_destroy (gomp_sem_t *sem)
{
sem_destroy (sem);
}
#endif 
#endif 
