#ifndef OACC_INT_H
#define OACC_INT_H 1
#include "openacc.h"
#include "config.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility push(hidden)
#endif
static inline enum acc_device_t
acc_device_type (enum offload_target_type type)
{
return (enum acc_device_t) type;
}
struct goacc_thread
{
struct gomp_device_descr *base_dev;
struct gomp_device_descr *dev;
struct gomp_device_descr *saved_bound_dev;
struct target_mem_desc *mapped_data;
struct goacc_thread *next;
void *target_tls;
};
#if defined HAVE_TLS || defined USE_EMUTLS
extern __thread struct goacc_thread *goacc_tls_data;
static inline struct goacc_thread *
goacc_thread (void)
{
return goacc_tls_data;
}
#else
extern pthread_key_t goacc_tls_key;
static inline struct goacc_thread *
goacc_thread (void)
{
return pthread_getspecific (goacc_tls_key);
}
#endif
void goacc_register (struct gomp_device_descr *) __GOACC_NOTHROW;
void goacc_attach_host_thread_to_device (int);
void goacc_runtime_initialize (void);
void goacc_save_and_set_bind (acc_device_t);
void goacc_restore_bind (void);
void goacc_lazy_initialize (void);
void goacc_host_init (void);
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility pop
#endif
#endif
