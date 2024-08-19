

#pragma once

#include "rawspeedconfig.h"       
#include "ThreadSafetyAnalysis.h" 

#ifdef HAVE_OPENMP
#include <omp.h> 
#endif

namespace rawspeed {

#ifdef HAVE_OPENMP

class CAPABILITY("mutex") Mutex final {
omp_lock_t mutex;

public:
explicit Mutex() { omp_init_lock(&mutex); }

Mutex(const Mutex&) = delete;
Mutex(Mutex&&) = delete;
Mutex& operator=(const Mutex&) = delete;
Mutex& operator=(Mutex&&) = delete;

~Mutex() { omp_destroy_lock(&mutex); }

void Lock() ACQUIRE() { omp_set_lock(&mutex); }

void Unlock() RELEASE() { omp_unset_lock(&mutex); }

bool TryLock() TRY_ACQUIRE(true) { return omp_test_lock(&mutex); }

const Mutex& operator!() const { return *this; }
};

#else

class CAPABILITY("mutex") Mutex final {
public:
explicit Mutex() = default;

Mutex(const Mutex&) = delete;
Mutex(Mutex&&) = delete;
Mutex& operator=(const Mutex&) = delete;
Mutex& operator=(Mutex&&) = delete;

~Mutex() = default;

void Lock() const ACQUIRE() {
}

void Unlock() const RELEASE() {
}

bool TryLock() const TRY_ACQUIRE(true) {
return true;
}

const Mutex& operator!() const { return *this; }
};

#endif

class SCOPED_CAPABILITY MutexLocker final {
Mutex* mut;

public:
explicit MutexLocker(Mutex* mu) ACQUIRE(mu) : mut(mu) { mu->Lock(); }

MutexLocker(const MutexLocker&) = delete;
MutexLocker(MutexLocker&&) = delete;
MutexLocker& operator=(const MutexLocker&) = delete;
MutexLocker& operator=(MutexLocker&&) = delete;

~MutexLocker() RELEASE() { mut->Unlock(); }
};

} 
