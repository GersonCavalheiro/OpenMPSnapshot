
#pragma once

#include <iostream>
#include <array>
#include <iterator>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <future>
#include <thread>
#include <mutex>

#ifdef KRATOS_SMP_OPENMP
#include <omp.h>
#endif

#include "includes/define.h"
#include "includes/global_variables.h"
#include "includes/lock_object.h"

#define KRATOS_CRITICAL_SECTION const std::lock_guard scope_lock(ParallelUtilities::GetGlobalLock());

#define KRATOS_PREPARE_CATCH_THREAD_EXCEPTION std::stringstream err_stream;

#define KRATOS_CATCH_THREAD_EXCEPTION \
} catch(Exception& e) { \
KRATOS_CRITICAL_SECTION \
err_stream << "Thread #" << i << " caught exception: " << e.what(); \
} catch(std::exception& e) { \
KRATOS_CRITICAL_SECTION \
err_stream << "Thread #" << i << " caught exception: " << e.what(); \
} catch(...) { \
KRATOS_CRITICAL_SECTION \
err_stream << "Thread #" << i << " caught unknown exception:"; \
}

#define KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION \
const std::string& err_msg = err_stream.str(); \
KRATOS_ERROR_IF_NOT(err_msg.empty()) << "The following errors occured in a parallel region!\n" << err_msg << std::endl;

namespace Kratos
{


class KRATOS_API(KRATOS_CORE) ParallelUtilities
{
public:



[[nodiscard]] static int GetNumThreads();


static void SetNumThreads(const int NumThreads);


[[nodiscard]] static int GetNumProcs();



[[nodiscard]] static LockObject& GetGlobalLock();


private:

static LockObject* mspGlobalLock;

static int* mspNumThreads;


ParallelUtilities() = delete;


static int InitializeNumberOfThreads();


static int& GetNumberOfThreads();

}; 


/
template<class TIterator, int MaxThreads=Globals::MaxAllowedThreads>
class BlockPartition
{
public:

BlockPartition(TIterator it_begin,
TIterator it_end,
int Nchunks = ParallelUtilities::GetNumThreads())
{
static_assert(std::is_same_v<typename std::iterator_traits<TIterator>::iterator_category, std::random_access_iterator_tag>);
KRATOS_ERROR_IF(Nchunks < 1) << "Number of chunks must be > 0 (and not " << Nchunks << ")" << std::endl;

const std::ptrdiff_t size_container = it_end-it_begin;

if (size_container == 0) {
mNchunks = Nchunks;
} else {
mNchunks = std::min(static_cast<int>(size_container), Nchunks);
}
const std::ptrdiff_t block_partition_size = size_container / mNchunks;
mBlockPartition[0] = it_begin;
mBlockPartition[mNchunks] = it_end;
for (int i=1; i<mNchunks; i++) {
mBlockPartition[i] = mBlockPartition[i-1] + block_partition_size;
}
}


template <class TUnaryFunction>
inline void for_each(TUnaryFunction&& f)
{
KRATOS_PREPARE_CATCH_THREAD_EXCEPTION

#pragma omp parallel for
for (int i=0; i<mNchunks; ++i) {
KRATOS_TRY
for (auto it = mBlockPartition[i]; it != mBlockPartition[i+1]; ++it) {
f(*it); 
}
KRATOS_CATCH_THREAD_EXCEPTION
}

KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION
}


template <class TReducer, class TUnaryFunction>
[[nodiscard]] inline typename TReducer::return_type for_each(TUnaryFunction &&f)
{
KRATOS_PREPARE_CATCH_THREAD_EXCEPTION

TReducer global_reducer;
#pragma omp parallel for
for (int i=0; i<mNchunks; ++i) {
KRATOS_TRY
TReducer local_reducer;
for (auto it = mBlockPartition[i]; it != mBlockPartition[i+1]; ++it) {
local_reducer.LocalReduce(f(*it));
}
global_reducer.ThreadSafeReduce(local_reducer);
KRATOS_CATCH_THREAD_EXCEPTION
}

KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION

return global_reducer.GetValue();
}


template <class TThreadLocalStorage, class TFunction>
inline void for_each(const TThreadLocalStorage& rThreadLocalStoragePrototype, TFunction &&f)
{
static_assert(std::is_copy_constructible<TThreadLocalStorage>::value, "TThreadLocalStorage must be copy constructible!");

KRATOS_PREPARE_CATCH_THREAD_EXCEPTION

#pragma omp parallel
{
TThreadLocalStorage thread_local_storage(rThreadLocalStoragePrototype);

#pragma omp for
for(int i=0; i<mNchunks; ++i){
KRATOS_TRY
for (auto it = mBlockPartition[i]; it != mBlockPartition[i+1]; ++it){
f(*it, thread_local_storage); 
}
KRATOS_CATCH_THREAD_EXCEPTION
}
}
KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION
}


template <class TReducer, class TThreadLocalStorage, class TFunction>
[[nodiscard]] inline typename TReducer::return_type for_each(const TThreadLocalStorage& rThreadLocalStoragePrototype, TFunction &&f)
{
static_assert(std::is_copy_constructible<TThreadLocalStorage>::value, "TThreadLocalStorage must be copy constructible!");

KRATOS_PREPARE_CATCH_THREAD_EXCEPTION

TReducer global_reducer;

#pragma omp parallel
{
TThreadLocalStorage thread_local_storage(rThreadLocalStoragePrototype);

#pragma omp for
for (int i=0; i<mNchunks; ++i) {
KRATOS_TRY
TReducer local_reducer;
for (auto it = mBlockPartition[i]; it != mBlockPartition[i+1]; ++it) {
local_reducer.LocalReduce(f(*it, thread_local_storage));
}
global_reducer.ThreadSafeReduce(local_reducer);
KRATOS_CATCH_THREAD_EXCEPTION
}
}
KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION
return global_reducer.GetValue();
}

private:
int mNchunks;
std::array<TIterator, MaxThreads> mBlockPartition;
};


template <class TIterator,
class TFunction,
std::enable_if_t<std::is_same_v<typename std::iterator_traits<TIterator>::iterator_category,std::random_access_iterator_tag>,bool> = true>
void block_for_each(TIterator itBegin, TIterator itEnd, TFunction&& rFunction)
{
BlockPartition<TIterator>(itBegin, itEnd).for_each(std::forward<TFunction>(rFunction));
}


template <class TReduction,
class TIterator,
class TFunction,
std::enable_if_t<std::is_same_v<typename std::iterator_traits<TIterator>::iterator_category,std::random_access_iterator_tag>,bool> = true>
[[nodiscard]] typename TReduction::return_type block_for_each(TIterator itBegin, TIterator itEnd, TFunction&& rFunction)
{
return  BlockPartition<TIterator>(itBegin, itEnd).template for_each<TReduction>(std::forward<TFunction>(std::forward<TFunction>(rFunction)));
}


template <class TIterator,
class TTLS,
class TFunction,
std::enable_if_t<std::is_same_v<typename std::iterator_traits<TIterator>::iterator_category,std::random_access_iterator_tag>,bool> = true>
void block_for_each(TIterator itBegin, TIterator itEnd, const TTLS& rTLS, TFunction &&rFunction)
{
BlockPartition<TIterator>(itBegin, itEnd).for_each(rTLS, std::forward<TFunction>(rFunction));
}


template <class TReduction,
class TIterator,
class TTLS,
class TFunction,
std::enable_if_t<std::is_same_v<typename std::iterator_traits<TIterator>::iterator_category,std::random_access_iterator_tag>,bool> = true>
[[nodiscard]] typename TReduction::return_type block_for_each(TIterator itBegin, TIterator itEnd, const TTLS& tls, TFunction&& rFunction)
{
return BlockPartition<TIterator>(itBegin, itEnd).template for_each<TReduction>(tls, std::forward<TFunction>(std::forward<TFunction>(rFunction)));
}


template <class TContainerType, class TFunctionType>
void block_for_each(TContainerType &&v, TFunctionType &&func)
{
block_for_each(v.begin(), v.end(), std::forward<TFunctionType>(func));
}


template <class TReducer, class TContainerType, class TFunctionType>
[[nodiscard]] typename TReducer::return_type block_for_each(TContainerType &&v, TFunctionType &&func)
{
return block_for_each<TReducer>(v.begin(), v.end(), std::forward<TFunctionType>(func));
}


template <class TContainerType, class TThreadLocalStorage, class TFunctionType>
void block_for_each(TContainerType &&v, const TThreadLocalStorage& tls, TFunctionType &&func)
{
block_for_each(v.begin(), v.end(), tls, std::forward<TFunctionType>(func));
}


template <class TReducer, class TContainerType, class TThreadLocalStorage, class TFunctionType>
[[nodiscard]] typename TReducer::return_type block_for_each(TContainerType &&v, const TThreadLocalStorage& tls, TFunctionType &&func)
{
return block_for_each<TReducer>(v.begin(), v.end(), tls, std::forward<TFunctionType>(func));
}

/
template<class TIndexType=std::size_t, int TMaxThreads=Globals::MaxAllowedThreads>
class IndexPartition
{
public:


IndexPartition(TIndexType Size,
int Nchunks = ParallelUtilities::GetNumThreads())
{
KRATOS_ERROR_IF(Nchunks < 1) << "Number of chunks must be > 0 (and not " << Nchunks << ")" << std::endl;

if (Size == 0) {
mNchunks = Nchunks;
} else {
mNchunks = std::min(static_cast<int>(Size), Nchunks);
}

const int block_partition_size = Size / mNchunks;
mBlockPartition[0] = 0;
mBlockPartition[mNchunks] = Size;
for (int i=1; i<mNchunks; i++) {
mBlockPartition[i] = mBlockPartition[i-1] + block_partition_size;
}

}

template <class TUnaryFunction>
inline void for_pure_c11(TUnaryFunction &&f)
{
std::vector< std::future<void> > runners(mNchunks);
const auto& partition = mBlockPartition;
for (int i=0; i<mNchunks; ++i) {
runners[i] = std::async(std::launch::async, [&partition, i,  &f]()
{
for (auto k = partition[i]; k < partition[i+1]; ++k) {
f(k);
}
});
}

for(int i=0; i<mNchunks; ++i) {
try {
runners[i].get();
}
catch(Exception& e) {
KRATOS_ERROR << std::endl << "THREAD number: " << i << " caught exception " << e.what() << std::endl;
} catch(std::exception& e) {
KRATOS_ERROR << std::endl << "THREAD number: " << i << " caught exception " << e.what() << std::endl;
} catch(...) {
KRATOS_ERROR << std::endl << "unknown error" << std::endl;
}
}
}


template <class TUnaryFunction>
inline void for_each(TUnaryFunction &&f)
{
KRATOS_PREPARE_CATCH_THREAD_EXCEPTION

#pragma omp parallel for
for (int i=0; i<mNchunks; ++i) {
KRATOS_TRY
for (auto k = mBlockPartition[i]; k < mBlockPartition[i+1]; ++k) {
f(k); 
}
KRATOS_CATCH_THREAD_EXCEPTION
}
KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION
}


template <class TReducer, class TUnaryFunction>
[[nodiscard]] inline typename TReducer::return_type for_each(TUnaryFunction &&f)
{
KRATOS_PREPARE_CATCH_THREAD_EXCEPTION

TReducer global_reducer;
#pragma omp parallel for
for (int i=0; i<mNchunks; ++i) {
KRATOS_TRY
TReducer local_reducer;
for (auto k = mBlockPartition[i]; k < mBlockPartition[i+1]; ++k) {
local_reducer.LocalReduce(f(k));
}
global_reducer.ThreadSafeReduce(local_reducer);
KRATOS_CATCH_THREAD_EXCEPTION
}
KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION
return global_reducer.GetValue();
}



template <class TThreadLocalStorage, class TFunction>
inline void for_each(const TThreadLocalStorage& rThreadLocalStoragePrototype, TFunction &&f)
{
static_assert(std::is_copy_constructible<TThreadLocalStorage>::value, "TThreadLocalStorage must be copy constructible!");

KRATOS_PREPARE_CATCH_THREAD_EXCEPTION

#pragma omp parallel
{
TThreadLocalStorage thread_local_storage(rThreadLocalStoragePrototype);

#pragma omp for
for (int i=0; i<mNchunks; ++i) {
KRATOS_TRY
for (auto k = mBlockPartition[i]; k < mBlockPartition[i+1]; ++k) {
f(k, thread_local_storage); 
}
KRATOS_CATCH_THREAD_EXCEPTION
}
}
KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION
}


template <class TReducer, class TThreadLocalStorage, class TFunction>
[[nodiscard]] inline typename TReducer::return_type for_each(const TThreadLocalStorage& rThreadLocalStoragePrototype, TFunction &&f)
{
static_assert(std::is_copy_constructible<TThreadLocalStorage>::value, "TThreadLocalStorage must be copy constructible!");

KRATOS_PREPARE_CATCH_THREAD_EXCEPTION

TReducer global_reducer;

#pragma omp parallel
{
TThreadLocalStorage thread_local_storage(rThreadLocalStoragePrototype);

#pragma omp for
for (int i=0; i<mNchunks; ++i) {
KRATOS_TRY
TReducer local_reducer;
for (auto k = mBlockPartition[i]; k < mBlockPartition[i+1]; ++k) {
local_reducer.LocalReduce(f(k, thread_local_storage));
}
global_reducer.ThreadSafeReduce(local_reducer);
KRATOS_CATCH_THREAD_EXCEPTION
}
}
KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION

return global_reducer.GetValue();
}

private:
int mNchunks;
std::array<TIndexType, TMaxThreads> mBlockPartition;
};

} 

#undef KRATOS_PREPARE_CATCH_THREAD_EXCEPTION
#undef KRATOS_CATCH_THREAD_EXCEPTION
#undef KRATOS_CHECK_AND_THROW_THREAD_EXCEPTION
