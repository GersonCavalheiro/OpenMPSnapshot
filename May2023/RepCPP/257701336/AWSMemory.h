

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/UnreferencedParam.h>
#include <aws/core/utils/memory/MemorySystemInterface.h>

#include <memory>
#include <cstdlib>
#include <algorithm>
#include <type_traits>

namespace Aws
{
namespace Utils
{
namespace Memory
{

AWS_CORE_API void InitializeAWSMemorySystem(MemorySystemInterface& memorySystem);


AWS_CORE_API void ShutdownAWSMemorySystem(void);


AWS_CORE_API MemorySystemInterface* GetMemorySystem();

} 
} 


AWS_CORE_API void* Malloc(const char* allocationTag, size_t allocationSize);


AWS_CORE_API void Free(void* memoryPtr);


template<typename T, typename ...ArgTypes>
T* New(const char* allocationTag, ArgTypes&&... args)
{
void *rawMemory = Malloc(allocationTag, sizeof(T));
T *constructedMemory = new (rawMemory) T(std::forward<ArgTypes>(args)...);
return constructedMemory;
}

#if defined(_MSC_VER) && !defined(_CPPRTTI)
template<typename T>
void Delete(T* pointerToT)
{
if (pointerToT == nullptr)
{
return;
}
pointerToT->~T();
Free(pointerToT);
}
#else

template<typename T>
typename std::enable_if<!std::is_polymorphic<T>::value>::type Delete(T* pointerToT)
{
if (pointerToT == nullptr)
{
return;
}
pointerToT->~T();
Free(pointerToT);
}

template<typename T>
typename std::enable_if<std::is_polymorphic<T>::value>::type Delete(T* pointerToT)
{
if (pointerToT == nullptr)
{
return;
}
void* mostDerivedT = dynamic_cast<void*>(pointerToT);
pointerToT->~T();
Free(mostDerivedT);
}
#endif 

template<typename T>
bool ShouldConstructArrayMembers()
{
return std::is_class<T>::value;
}

template<typename T>
bool ShouldDestroyArrayMembers()
{
return !std::is_trivially_destructible<T>::value;
}


template<typename T>
T* NewArray(std::size_t amount, const char* allocationTag)
{
if (amount > 0)
{
bool constructMembers = ShouldConstructArrayMembers<T>();
bool trackMemberCount = ShouldDestroyArrayMembers<T>();

std::size_t allocationSize = amount * sizeof(T);
#if defined(_MSC_VER) && _MSC_VER < 1900
std::size_t headerSize = (std::max)(sizeof(std::size_t), __alignof(T));
#else
std::size_t headerSize = (std::max)(sizeof(std::size_t), alignof(T));
#endif

if (trackMemberCount)
{
allocationSize += headerSize;
}

void* rawMemory = Malloc(allocationTag, allocationSize);
T* pointerToT = nullptr;

if (trackMemberCount)
{
std::size_t* pointerToAmount = reinterpret_cast<std::size_t*>(rawMemory);
*pointerToAmount = amount;
pointerToT = reinterpret_cast<T*>(reinterpret_cast<char*>(pointerToAmount) + headerSize);

}
else
{
pointerToT = reinterpret_cast<T*>(rawMemory);
}

if (constructMembers)
{
for (std::size_t i = 0; i < amount; ++i)
{
new (pointerToT + i) T;
}
}

return pointerToT;
}

return nullptr;
}


template<typename T>
void DeleteArray(T* pointerToTArray)
{
if (pointerToTArray == nullptr)
{
return;
}

bool destroyMembers = ShouldDestroyArrayMembers<T>();
void* rawMemory = nullptr;

if (destroyMembers)
{
#if defined(_MSC_VER) && _MSC_VER < 1900
std::size_t headerSize = (std::max)(sizeof(std::size_t), __alignof(T));
#else
std::size_t headerSize = (std::max)(sizeof(std::size_t), alignof(T));
#endif

std::size_t *pointerToAmount = reinterpret_cast<std::size_t*>(reinterpret_cast<char*>(pointerToTArray) - headerSize);
std::size_t amount = *pointerToAmount;

for (std::size_t i = amount; i > 0; --i)
{
(pointerToTArray + i - 1)->~T();
}
rawMemory = reinterpret_cast<void *>(pointerToAmount);
}
else
{
rawMemory = reinterpret_cast<void *>(pointerToTArray);
}

Free(rawMemory);
}


template<typename T>
struct Deleter
{
Deleter() {}

template<class U, class = typename std::enable_if<std::is_convertible<U *, T *>::value, void>::type>
Deleter(const Deleter<U>&)
{
}

void operator()(T *pointerToT) const
{
static_assert(0 < sizeof(T), "can't delete an incomplete type");
Aws::Delete(pointerToT);
}
};

template< typename T > using UniquePtr = std::unique_ptr< T, Deleter< T > >;


template<typename T, typename ...ArgTypes>
UniquePtr<T> MakeUnique(const char* allocationTag, ArgTypes&&... args)
{
return UniquePtr<T>(Aws::New<T>(allocationTag, std::forward<ArgTypes>(args)...));
}

template<typename T>
struct ArrayDeleter
{
ArrayDeleter() {}

template<class U, class = typename std::enable_if<std::is_convertible<U *, T *>::value, void>::type>
ArrayDeleter(const ArrayDeleter<U>&)
{
}

void operator()(T *pointerToTArray) const
{
static_assert(0 < sizeof(T), "can't delete an incomplete type");
Aws::DeleteArray(pointerToTArray);
}
};

template< typename T > using UniqueArrayPtr = std::unique_ptr< T, ArrayDeleter< T > >;


template<typename T, typename ...ArgTypes>
UniqueArrayPtr<T> MakeUniqueArray(std::size_t amount, const char* allocationTag, ArgTypes&&... args)
{
return UniqueArrayPtr<T>(Aws::NewArray<T>(amount, allocationTag, std::forward<ArgTypes>(args)...));
}


} 

