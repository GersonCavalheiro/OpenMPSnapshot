
#pragma once

#include "platform.h"
#include "mutex.h"
#include "alloc.h"
#include "vector.h"
#include <vector>

namespace embree
{

typedef struct opaque_thread_t* thread_t;


typedef void (*thread_func)(void*);


thread_t createThread(thread_func f, void* arg, size_t stack_size = 0, ssize_t threadID = -1);


void setAffinity(ssize_t affinity);


void yield();


void join(thread_t tid);


void destroyThread(thread_t tid);


typedef struct opaque_tls_t* tls_t;


tls_t createTls();


void setTls(tls_t tls, void* const ptr);


void* getTls(tls_t tls);


void destroyTls(tls_t tls);
}
