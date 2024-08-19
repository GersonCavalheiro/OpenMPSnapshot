#include "libgomp.h"
#include "oacc-int.h"
#include "openacc.h"
#include <assert.h>
#include <stdlib.h>
#include <strings.h>
#include <stdbool.h>
#include <string.h>
static gomp_mutex_t acc_device_lock;
static struct gomp_device_descr *cached_base_dev = NULL;
#if defined HAVE_TLS || defined USE_EMUTLS
__thread struct goacc_thread *goacc_tls_data;
#else
pthread_key_t goacc_tls_key;
#endif
static pthread_key_t goacc_cleanup_key;
static struct goacc_thread *goacc_threads;
static gomp_mutex_t goacc_thread_lock;
static struct gomp_device_descr *dispatchers[_ACC_device_hwm] = { 0 };
attribute_hidden void
goacc_register (struct gomp_device_descr *disp)
{
if (disp->target_id != 0)
return;
gomp_mutex_lock (&acc_device_lock);
assert (acc_device_type (disp->type) != acc_device_none
&& acc_device_type (disp->type) != acc_device_default
&& acc_device_type (disp->type) != acc_device_not_host);
assert (!dispatchers[disp->type]);
dispatchers[disp->type] = disp;
gomp_mutex_unlock (&acc_device_lock);
}
static const char *
get_openacc_name (const char *name)
{
if (strcmp (name, "nvptx") == 0)
return "nvidia";
else
return name;
}
static const char *
name_of_acc_device_t (enum acc_device_t type)
{
switch (type)
{
case acc_device_none: return "none";
case acc_device_default: return "default";
case acc_device_host: return "host";
case acc_device_not_host: return "not_host";
case acc_device_nvidia: return "nvidia";
default: gomp_fatal ("unknown device type %u", (unsigned) type);
}
}
static struct gomp_device_descr *
resolve_device (acc_device_t d, bool fail_is_error)
{
acc_device_t d_arg = d;
switch (d)
{
case acc_device_default:
{
if (goacc_device_type)
{
while (++d != _ACC_device_hwm)
if (dispatchers[d]
&& !strcasecmp (goacc_device_type,
get_openacc_name (dispatchers[d]->name))
&& dispatchers[d]->get_num_devices_func () > 0)
goto found;
if (fail_is_error)
{
gomp_mutex_unlock (&acc_device_lock);
gomp_fatal ("device type %s not supported", goacc_device_type);
}
else
return NULL;
}
d = acc_device_not_host;
}
case acc_device_not_host:
while (++d != _ACC_device_hwm)
if (dispatchers[d] && dispatchers[d]->get_num_devices_func () > 0)
goto found;
if (d_arg == acc_device_default)
{
d = acc_device_host;
goto found;
}
if (fail_is_error)
{
gomp_mutex_unlock (&acc_device_lock);
gomp_fatal ("no device found");
}
else
return NULL;
break;
case acc_device_host:
break;
default:
if (d > _ACC_device_hwm)
{
if (fail_is_error)
goto unsupported_device;
else
return NULL;
}
break;
}
found:
assert (d != acc_device_none
&& d != acc_device_default
&& d != acc_device_not_host);
if (dispatchers[d] == NULL && fail_is_error)
{
unsupported_device:
gomp_mutex_unlock (&acc_device_lock);
gomp_fatal ("device type %s not supported", name_of_acc_device_t (d));
}
return dispatchers[d];
}
static void
acc_dev_num_out_of_range (acc_device_t d, int ord, int ndevs)
{
if (ndevs == 0)
gomp_fatal ("no devices of type %s available", name_of_acc_device_t (d));
else
gomp_fatal ("device %u out of range", ord);
}
static struct gomp_device_descr *
acc_init_1 (acc_device_t d)
{
struct gomp_device_descr *base_dev, *acc_dev;
int ndevs;
base_dev = resolve_device (d, true);
ndevs = base_dev->get_num_devices_func ();
if (ndevs <= 0 || goacc_device_num >= ndevs)
acc_dev_num_out_of_range (d, goacc_device_num, ndevs);
acc_dev = &base_dev[goacc_device_num];
gomp_mutex_lock (&acc_dev->lock);
if (acc_dev->state == GOMP_DEVICE_INITIALIZED)
{
gomp_mutex_unlock (&acc_dev->lock);
gomp_fatal ("device already active");
}
gomp_init_device (acc_dev);
gomp_mutex_unlock (&acc_dev->lock);
return base_dev;
}
static void
acc_shutdown_1 (acc_device_t d)
{
struct gomp_device_descr *base_dev;
struct goacc_thread *walk;
int ndevs, i;
bool devices_active = false;
base_dev = resolve_device (d, true);
ndevs = base_dev->get_num_devices_func ();
for (i = 0; i < ndevs; i++)
{
struct gomp_device_descr *acc_dev = &base_dev[i];
gomp_mutex_lock (&acc_dev->lock);
gomp_unload_device (acc_dev);
gomp_mutex_unlock (&acc_dev->lock);
}
gomp_mutex_lock (&goacc_thread_lock);
for (walk = goacc_threads; walk != NULL; walk = walk->next)
{
if (walk->target_tls)
base_dev->openacc.destroy_thread_data_func (walk->target_tls);
walk->target_tls = NULL;
if (walk->mapped_data)
{
gomp_mutex_unlock (&goacc_thread_lock);
gomp_fatal ("shutdown in 'acc data' region");
}
if (walk->saved_bound_dev)
{
gomp_mutex_unlock (&goacc_thread_lock);
gomp_fatal ("shutdown during host fallback");
}
if (walk->dev)
{
gomp_mutex_lock (&walk->dev->lock);
gomp_free_memmap (&walk->dev->mem_map);
gomp_mutex_unlock (&walk->dev->lock);
walk->dev = NULL;
walk->base_dev = NULL;
}
}
gomp_mutex_unlock (&goacc_thread_lock);
bool ret = true;
for (i = 0; i < ndevs; i++)
{
struct gomp_device_descr *acc_dev = &base_dev[i];
gomp_mutex_lock (&acc_dev->lock);
if (acc_dev->state == GOMP_DEVICE_INITIALIZED)
{
devices_active = true;
ret &= acc_dev->fini_device_func (acc_dev->target_id);
acc_dev->state = GOMP_DEVICE_UNINITIALIZED;
}
gomp_mutex_unlock (&acc_dev->lock);
}
if (!ret)
gomp_fatal ("device finalization failed");
if (!devices_active)
gomp_fatal ("no device initialized");
}
static struct goacc_thread *
goacc_new_thread (void)
{
struct goacc_thread *thr = gomp_malloc (sizeof (struct goacc_thread));
#if defined HAVE_TLS || defined USE_EMUTLS
goacc_tls_data = thr;
#else
pthread_setspecific (goacc_tls_key, thr);
#endif
pthread_setspecific (goacc_cleanup_key, thr);
gomp_mutex_lock (&goacc_thread_lock);
thr->next = goacc_threads;
goacc_threads = thr;
gomp_mutex_unlock (&goacc_thread_lock);
return thr;
}
static void
goacc_destroy_thread (void *data)
{
struct goacc_thread *thr = data, *walk, *prev;
gomp_mutex_lock (&goacc_thread_lock);
if (thr)
{
struct gomp_device_descr *acc_dev = thr->dev;
if (acc_dev && thr->target_tls)
{
acc_dev->openacc.destroy_thread_data_func (thr->target_tls);
thr->target_tls = NULL;
}
assert (!thr->mapped_data);
for (prev = NULL, walk = goacc_threads; walk;
prev = walk, walk = walk->next)
if (walk == thr)
{
if (prev == NULL)
goacc_threads = walk->next;
else
prev->next = walk->next;
free (thr);
break;
}
assert (walk);
}
gomp_mutex_unlock (&goacc_thread_lock);
}
void
goacc_attach_host_thread_to_device (int ord)
{
struct goacc_thread *thr = goacc_thread ();
struct gomp_device_descr *acc_dev = NULL, *base_dev = NULL;
int num_devices;
if (thr && thr->dev && (thr->dev->target_id == ord || ord < 0))
return;
if (ord < 0)
ord = goacc_device_num;
if (thr && thr->base_dev)
base_dev = thr->base_dev;
else
{
assert (cached_base_dev);
base_dev = cached_base_dev;
}
num_devices = base_dev->get_num_devices_func ();
if (num_devices <= 0 || ord >= num_devices)
acc_dev_num_out_of_range (acc_device_type (base_dev->type), ord,
num_devices);
if (!thr)
thr = goacc_new_thread ();
thr->base_dev = base_dev;
thr->dev = acc_dev = &base_dev[ord];
thr->saved_bound_dev = NULL;
thr->mapped_data = NULL;
thr->target_tls
= acc_dev->openacc.create_thread_data_func (ord);
acc_dev->openacc.async_set_async_func (acc_async_sync);
}
void
acc_init (acc_device_t d)
{
gomp_init_targets_once ();
gomp_mutex_lock (&acc_device_lock);
cached_base_dev = acc_init_1 (d);
gomp_mutex_unlock (&acc_device_lock);
goacc_attach_host_thread_to_device (-1);
}
ialias (acc_init)
void
acc_shutdown (acc_device_t d)
{
gomp_init_targets_once ();
gomp_mutex_lock (&acc_device_lock);
acc_shutdown_1 (d);
gomp_mutex_unlock (&acc_device_lock);
}
ialias (acc_shutdown)
int
acc_get_num_devices (acc_device_t d)
{
int n = 0;
struct gomp_device_descr *acc_dev;
if (d == acc_device_none)
return 0;
gomp_init_targets_once ();
gomp_mutex_lock (&acc_device_lock);
acc_dev = resolve_device (d, false);
gomp_mutex_unlock (&acc_device_lock);
if (!acc_dev)
return 0;
n = acc_dev->get_num_devices_func ();
if (n < 0)
n = 0;
return n;
}
ialias (acc_get_num_devices)
void
acc_set_device_type (acc_device_t d)
{
struct gomp_device_descr *base_dev, *acc_dev;
struct goacc_thread *thr = goacc_thread ();
gomp_init_targets_once ();
gomp_mutex_lock (&acc_device_lock);
cached_base_dev = base_dev = resolve_device (d, true);
acc_dev = &base_dev[goacc_device_num];
gomp_mutex_lock (&acc_dev->lock);
if (acc_dev->state == GOMP_DEVICE_UNINITIALIZED)
gomp_init_device (acc_dev);
gomp_mutex_unlock (&acc_dev->lock);
gomp_mutex_unlock (&acc_device_lock);
if (thr && thr->base_dev != base_dev)
{
thr->base_dev = thr->dev = NULL;
if (thr->mapped_data)
gomp_fatal ("acc_set_device_type in 'acc data' region");
}
goacc_attach_host_thread_to_device (-1);
}
ialias (acc_set_device_type)
acc_device_t
acc_get_device_type (void)
{
acc_device_t res = acc_device_none;
struct gomp_device_descr *dev;
struct goacc_thread *thr = goacc_thread ();
if (thr && thr->base_dev)
res = acc_device_type (thr->base_dev->type);
else
{
gomp_init_targets_once ();
gomp_mutex_lock (&acc_device_lock);
dev = resolve_device (acc_device_default, true);
gomp_mutex_unlock (&acc_device_lock);
res = acc_device_type (dev->type);
}
assert (res != acc_device_default
&& res != acc_device_not_host);
return res;
}
ialias (acc_get_device_type)
int
acc_get_device_num (acc_device_t d)
{
const struct gomp_device_descr *dev;
struct goacc_thread *thr = goacc_thread ();
if (d >= _ACC_device_hwm)
gomp_fatal ("unknown device type %u", (unsigned) d);
gomp_init_targets_once ();
gomp_mutex_lock (&acc_device_lock);
dev = resolve_device (d, true);
gomp_mutex_unlock (&acc_device_lock);
if (thr && thr->base_dev == dev && thr->dev)
return thr->dev->target_id;
return goacc_device_num;
}
ialias (acc_get_device_num)
void
acc_set_device_num (int ord, acc_device_t d)
{
struct gomp_device_descr *base_dev, *acc_dev;
int num_devices;
gomp_init_targets_once ();
if (ord < 0)
ord = goacc_device_num;
if ((int) d == 0)
goacc_attach_host_thread_to_device (ord);
else
{
gomp_mutex_lock (&acc_device_lock);
cached_base_dev = base_dev = resolve_device (d, true);
num_devices = base_dev->get_num_devices_func ();
if (num_devices <= 0 || ord >= num_devices)
acc_dev_num_out_of_range (d, ord, num_devices);
acc_dev = &base_dev[ord];
gomp_mutex_lock (&acc_dev->lock);
if (acc_dev->state == GOMP_DEVICE_UNINITIALIZED)
gomp_init_device (acc_dev);
gomp_mutex_unlock (&acc_dev->lock);
gomp_mutex_unlock (&acc_device_lock);
goacc_attach_host_thread_to_device (ord);
}
goacc_device_num = ord;
}
ialias (acc_set_device_num)
int __attribute__ ((__optimize__ ("O2")))
acc_on_device (acc_device_t dev)
{
return __builtin_acc_on_device (dev);
}
ialias (acc_on_device)
attribute_hidden void
goacc_runtime_initialize (void)
{
gomp_mutex_init (&acc_device_lock);
#if !(defined HAVE_TLS || defined USE_EMUTLS)
pthread_key_create (&goacc_tls_key, NULL);
#endif
pthread_key_create (&goacc_cleanup_key, goacc_destroy_thread);
cached_base_dev = NULL;
goacc_threads = NULL;
gomp_mutex_init (&goacc_thread_lock);
goacc_host_init ();
}
attribute_hidden void
goacc_save_and_set_bind (acc_device_t d)
{
struct goacc_thread *thr = goacc_thread ();
assert (!thr->saved_bound_dev);
thr->saved_bound_dev = thr->dev;
thr->dev = dispatchers[d];
}
attribute_hidden void
goacc_restore_bind (void)
{
struct goacc_thread *thr = goacc_thread ();
thr->dev = thr->saved_bound_dev;
thr->saved_bound_dev = NULL;
}
attribute_hidden void
goacc_lazy_initialize (void)
{
struct goacc_thread *thr = goacc_thread ();
if (thr && thr->dev)
return;
if (!cached_base_dev)
acc_init (acc_device_default);
else
goacc_attach_host_thread_to_device (-1);
}
