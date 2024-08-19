#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef void* pevent_t;
#define PEVENT_WAIT_INFINITE (~((uint64_t)0));
pevent_t pevent_create(bool manualReset, bool initialState);
int pevent_destroy(pevent_t event);
int pevent_wait(pevent_t event, uint64_t milliseconds);
int pevent_set(pevent_t event);
int pevent_reset(pevent_t event);
#ifdef WFMO
int pevent_wait_multiple(pevent_t *events, int count, bool waitAll, uint64_t milliseconds);
int pevent_wait_multiple_get_index(pevent_t *events, int count, bool waitAll, uint64_t milliseconds, int *index);
#endif
#ifdef PULSE
int pevent_pulse(pevent_t event);
#endif
#ifdef __cplusplus
}
#endif
