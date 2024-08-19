

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_MUTEX_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_MUTEX_H_


#include <chrono>
#include <condition_variable>
#include <mutex>
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

#undef mutex_lock

enum LinkerInitialized { LINKER_INITIALIZED };

class condition_variable;

class LOCKABLE mutex {
public:
mutex();
explicit mutex(LinkerInitialized x) {}

void lock() EXCLUSIVE_LOCK_FUNCTION();
bool try_lock() EXCLUSIVE_TRYLOCK_FUNCTION(true);
void unlock() UNLOCK_FUNCTION();

void lock_shared() SHARED_LOCK_FUNCTION();
bool try_lock_shared() SHARED_TRYLOCK_FUNCTION(true);
void unlock_shared() UNLOCK_FUNCTION();

struct external_mu_space {
void* space[2];
};

private:
friend class condition_variable;
external_mu_space mu_;
};

class SCOPED_LOCKABLE mutex_lock {
public:
typedef ::tensorflow::mutex mutex_type;

explicit mutex_lock(mutex_type& mu) EXCLUSIVE_LOCK_FUNCTION(mu) : mu_(&mu) {
mu_->lock();
}

mutex_lock(mutex_type& mu, std::try_to_lock_t) EXCLUSIVE_LOCK_FUNCTION(mu)
: mu_(&mu) {
if (!mu.try_lock()) {
mu_ = nullptr;
}
}

mutex_lock(mutex_lock&& ml) noexcept EXCLUSIVE_LOCK_FUNCTION(ml.mu_)
: mu_(ml.mu_) {
ml.mu_ = nullptr;
}
~mutex_lock() UNLOCK_FUNCTION() {
if (mu_ != nullptr) {
mu_->unlock();
}
}
mutex_type* mutex() { return mu_; }

operator bool() const { return mu_ != nullptr; }

private:
mutex_type* mu_;
};

#define mutex_lock(x) static_assert(0, "mutex_lock_decl_missing_var_name");

class SCOPED_LOCKABLE tf_shared_lock {
public:
typedef ::tensorflow::mutex mutex_type;

explicit tf_shared_lock(mutex_type& mu) SHARED_LOCK_FUNCTION(mu) : mu_(&mu) {
mu_->lock_shared();
}

tf_shared_lock(mutex_type& mu, std::try_to_lock_t) SHARED_LOCK_FUNCTION(mu)
: mu_(&mu) {
if (!mu.try_lock_shared()) {
mu_ = nullptr;
}
}

tf_shared_lock(tf_shared_lock&& ml) noexcept SHARED_LOCK_FUNCTION(ml.mu_)
: mu_(ml.mu_) {
ml.mu_ = nullptr;
}
~tf_shared_lock() UNLOCK_FUNCTION() {
if (mu_ != nullptr) {
mu_->unlock_shared();
}
}
mutex_type* mutex() { return mu_; }

operator bool() const { return mu_ != nullptr; }

private:
mutex_type* mu_;
};

#define tf_shared_lock(x) \
static_assert(0, "tf_shared_lock_decl_missing_var_name");

class condition_variable {
public:
condition_variable();

void wait(mutex_lock& lock);
template <class Rep, class Period>
std::cv_status wait_for(mutex_lock& lock,
std::chrono::duration<Rep, Period> dur) {
return wait_until_system_clock(lock,
std::chrono::system_clock::now() + dur);
}
void notify_one();
void notify_all();

struct external_cv_space {
void* space[2];
};

private:
friend ConditionResult WaitForMilliseconds(mutex_lock* mu,
condition_variable* cv, int64 ms);
std::cv_status wait_until_system_clock(
mutex_lock& lock,
const std::chrono::system_clock::time_point timeout_time);
external_cv_space cv_;
};

inline ConditionResult WaitForMilliseconds(mutex_lock* mu,
condition_variable* cv, int64 ms) {
std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
return (s == std::cv_status::timeout) ? kCond_Timeout : kCond_MaybeNotified;
}

}  

#endif  
