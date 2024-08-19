
#pragma once

#include <mutex>

#ifdef KRATOS_SMP_OPENMP
#include <omp.h>
#endif

#include "includes/define.h"


namespace Kratos
{



class LockObject
{
public:

LockObject() noexcept
{
#ifdef KRATOS_SMP_OPENMP
omp_init_lock(&mLock);
#endif
}

LockObject(LockObject const& rOther) = delete;

~LockObject() noexcept
{
#ifdef KRATOS_SMP_OPENMP
omp_destroy_lock(&mLock);
#endif
}


LockObject& operator=(LockObject const& rOther) = delete;


inline void lock() const
{
#ifdef KRATOS_SMP_CXX11
mLock.lock();
#elif KRATOS_SMP_OPENMP
omp_set_lock(&mLock);
#endif
}

KRATOS_DEPRECATED_MESSAGE("Please use lock instead")
inline void SetLock() const
{
this->lock();
}

inline void unlock() const
{
#ifdef KRATOS_SMP_CXX11
mLock.unlock();
#elif KRATOS_SMP_OPENMP
omp_unset_lock(&mLock);
#endif
}

KRATOS_DEPRECATED_MESSAGE("Please use unlock instead")
inline void UnSetLock() const
{
this->unlock();
}

inline bool try_lock() const
{
#ifdef KRATOS_SMP_CXX11
return mLock.try_lock();
#elif KRATOS_SMP_OPENMP
return omp_test_lock(&mLock);
#endif
return true;
}


private:

#ifdef KRATOS_SMP_CXX11
mutable std::mutex mLock;
#elif KRATOS_SMP_OPENMP
mutable omp_lock_t mLock;
#endif


}; 



}  
