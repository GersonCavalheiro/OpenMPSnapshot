#ifndef _OPENACC_H
#define _OPENACC_H 1
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#if __cplusplus >= 201103
# define __GOACC_NOTHROW noexcept
#elif __cplusplus
# define __GOACC_NOTHROW throw ()
#else 
# define __GOACC_NOTHROW __attribute__ ((__nothrow__))
#endif
typedef enum acc_device_t {
acc_device_none = 0,
acc_device_default = 1,
acc_device_host = 2,
acc_device_not_host = 4,
acc_device_nvidia = 5,
_ACC_device_hwm,
_ACC_highest = __INT_MAX__,
_ACC_neg = -1
} acc_device_t;
typedef enum acc_async_t {
acc_async_noval = -1,
acc_async_sync  = -2
} acc_async_t;
int acc_get_num_devices (acc_device_t) __GOACC_NOTHROW;
void acc_set_device_type (acc_device_t) __GOACC_NOTHROW;
acc_device_t acc_get_device_type (void) __GOACC_NOTHROW;
void acc_set_device_num (int, acc_device_t) __GOACC_NOTHROW;
int acc_get_device_num (acc_device_t) __GOACC_NOTHROW;
int acc_async_test (int) __GOACC_NOTHROW;
int acc_async_test_all (void) __GOACC_NOTHROW;
void acc_wait (int) __GOACC_NOTHROW;
void acc_async_wait (int) __GOACC_NOTHROW;
void acc_wait_async (int, int) __GOACC_NOTHROW;
void acc_wait_all (void) __GOACC_NOTHROW;
void acc_async_wait_all (void) __GOACC_NOTHROW;
void acc_wait_all_async (int) __GOACC_NOTHROW;
void acc_init (acc_device_t) __GOACC_NOTHROW;
void acc_shutdown (acc_device_t) __GOACC_NOTHROW;
#ifdef __cplusplus
int acc_on_device (int __arg) __GOACC_NOTHROW;
#else
int acc_on_device (acc_device_t __arg) __GOACC_NOTHROW;
#endif
void *acc_malloc (size_t) __GOACC_NOTHROW;
void acc_free (void *) __GOACC_NOTHROW;
void *acc_copyin (void *, size_t) __GOACC_NOTHROW;
void *acc_present_or_copyin (void *, size_t) __GOACC_NOTHROW;
void *acc_pcopyin (void *, size_t) __GOACC_NOTHROW;
void *acc_create (void *, size_t) __GOACC_NOTHROW;
void *acc_present_or_create (void *, size_t) __GOACC_NOTHROW;
void *acc_pcreate (void *, size_t) __GOACC_NOTHROW;
void acc_copyout (void *, size_t) __GOACC_NOTHROW;
void acc_delete (void *, size_t) __GOACC_NOTHROW;
void acc_update_device (void *, size_t) __GOACC_NOTHROW;
void acc_update_self (void *, size_t) __GOACC_NOTHROW;
void acc_map_data (void *, void *, size_t) __GOACC_NOTHROW;
void acc_unmap_data (void *) __GOACC_NOTHROW;
void *acc_deviceptr (void *) __GOACC_NOTHROW;
void *acc_hostptr (void *) __GOACC_NOTHROW;
int acc_is_present (void *, size_t) __GOACC_NOTHROW;
void acc_memcpy_to_device (void *, void *, size_t) __GOACC_NOTHROW;
void acc_memcpy_from_device (void *, void *, size_t) __GOACC_NOTHROW;
void *acc_get_current_cuda_device (void) __GOACC_NOTHROW;
void *acc_get_current_cuda_context (void) __GOACC_NOTHROW;
void *acc_get_cuda_stream (int) __GOACC_NOTHROW;
int acc_set_cuda_stream (int, void *) __GOACC_NOTHROW;
#ifdef __cplusplus
}
#pragma acc routine seq
inline int acc_on_device (acc_device_t __arg) __GOACC_NOTHROW
{
return acc_on_device ((int) __arg);
}
#endif
#endif 
