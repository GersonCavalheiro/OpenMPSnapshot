#pragma once

#ifndef __UT_SMALLARRAY_H_INCLUDED__
#define __UT_SMALLARRAY_H_INCLUDED__

#include "array.h"
#include "SYS_Types.h"
#include <utility>
#include <stddef.h>

template <typename T, size_t MAX_BYTES = 64>
class UT_SmallArray : public UT_Array<T>
{
enum { MAX_ELEMS = MAX_BYTES/sizeof(T) < 1 ? 1 : MAX_BYTES/sizeof(T) };

public:

#if defined(__clang__) || defined(_MSC_VER)
#define UT_SMALL_ARRAY_SIZE_ASSERT()    \
using ThisT = UT_SmallArray<T,MAX_BYTES>; \
static_assert(offsetof(ThisT, myBuffer) == sizeof(UT_Array<T>), \
"In order for UT_Array's checks for whether it needs to free the buffer to work, " \
"the buffer must be exactly following the base class memory.")
#else
#define UT_SMALL_ARRAY_SIZE_ASSERT()    \
UT_ASSERT_P(!UT_Array<T>::isHeapBuffer());
#endif

UT_SmallArray()
: UT_Array<T>(0)
{
UT_Array<T>::unsafeShareData((T*)myBuffer, 0, MAX_ELEMS);
UT_SMALL_ARRAY_SIZE_ASSERT();
}

explicit UT_SmallArray(const UT_Array<T> &copy)
: UT_Array<T>(0)
{
UT_Array<T>::unsafeShareData((T*)myBuffer, 0, MAX_ELEMS);
UT_SMALL_ARRAY_SIZE_ASSERT();
UT_Array<T>::operator=(copy);
}
explicit UT_SmallArray(const UT_SmallArray<T,MAX_BYTES> &copy)
: UT_Array<T>(0)
{
UT_Array<T>::unsafeShareData((T*)myBuffer, 0, MAX_ELEMS);
UT_SMALL_ARRAY_SIZE_ASSERT();
UT_Array<T>::operator=(copy);
}

UT_SmallArray(UT_Array<T> &&movable) noexcept
{
UT_Array<T>::unsafeShareData((T*)myBuffer, 0, MAX_ELEMS);
UT_SMALL_ARRAY_SIZE_ASSERT();
UT_Array<T>::operator=(std::move(movable));
}
UT_SmallArray(UT_SmallArray<T,MAX_BYTES> &&movable) noexcept
{
UT_Array<T>::unsafeShareData((T*)myBuffer, 0, MAX_ELEMS);
UT_SMALL_ARRAY_SIZE_ASSERT();
UT_Array<T>::operator=(std::move(movable));
}

explicit UT_SmallArray(std::initializer_list<T> init)
{
UT_Array<T>::unsafeShareData((T*)myBuffer, 0, MAX_ELEMS);
UT_SMALL_ARRAY_SIZE_ASSERT();
UT_Array<T>::operator=(init);
}

#undef UT_SMALL_ARRAY_SIZE_ASSERT

UT_SmallArray<T,MAX_BYTES> &
operator=(const UT_SmallArray<T,MAX_BYTES> &copy)
{
UT_Array<T>::operator=(copy);
return *this;
}
UT_SmallArray<T,MAX_BYTES> &
operator=(const UT_Array<T> &copy)
{
UT_Array<T>::operator=(copy);
return *this;
}

UT_SmallArray<T,MAX_BYTES> &
operator=(UT_SmallArray<T,MAX_BYTES> &&movable)
{
UT_Array<T>::operator=(std::move(movable));
return *this;
}
UT_SmallArray<T,MAX_BYTES> &
operator=(UT_Array<T> &&movable)
{
UT_Array<T>::operator=(std::move(movable));
return *this;
}

UT_SmallArray<T,MAX_BYTES> &
operator=(std::initializer_list<T> src)
{
UT_Array<T>::operator=(src);
return *this;
}
private:
alignas(T) char myBuffer[MAX_ELEMS*sizeof(T)];
};

#endif 
