#pragma once

#ifndef __UT_ARRAY_H_INCLUDED__
#define __UT_ARRAY_H_INCLUDED__

#include "SYS_Types.h"

#include <algorithm>
#include <functional>
#include <type_traits>
#include <string.h>

template <typename T>
static inline T
UTbumpAlloc(T current_size)
{
constexpr T SMALL_ALLOC(16);
constexpr T BIG_ALLOC(128);

if (current_size < T(8))
{
return (current_size < T(4)) ? T(4) : T(8);
}
if (current_size < T(BIG_ALLOC))
{
return (current_size + T(SMALL_ALLOC)) & ~T(SMALL_ALLOC-1);
}
if (current_size < T(BIG_ALLOC * 8))
{
return (current_size + T(BIG_ALLOC)) & ~T(BIG_ALLOC-1);
}

T bump = current_size >> 3; 
current_size += bump;
return current_size;
}

template <typename T>
class UT_Array
{
public:
typedef T value_type;

typedef int (*Comparator)(const T *, const T *);

explicit UT_Array(const UT_Array<T> &a);

UT_Array(UT_Array<T> &&a) noexcept;

UT_Array(exint capacity, exint size)
{
myData = capacity ? allocateCapacity(capacity) : NULL;
if (capacity < size)
size = capacity;
mySize = size;
myCapacity = capacity;
trivialConstructRange(myData, mySize);
}

explicit UT_Array(exint capacity = 0) : myCapacity(capacity), mySize(0)
{
myData = capacity ? allocateCapacity(capacity) : NULL;
}

explicit UT_Array(std::initializer_list<T> init);

~UT_Array();

void	    swap(UT_Array<T> &other);

exint           append(void) { return insert(mySize); }
exint           append(const T &t) { return appendImpl(t); }
exint           append(T &&t) { return appendImpl(std::move(t)); }
void            append(const T *pt, exint count);
void	    appendMultiple(const T &t, exint count);
exint	    insert(exint index);
exint	    insert(const T &t, exint i)
{ return insertImpl(t, i); }
exint	    insert(T &&t, exint i)
{ return insertImpl(std::move(t), i); }

template <typename... S>
exint	    emplace_back(S&&... s);

exint	    concat(const UT_Array<T> &a);

exint	    multipleInsert(exint index, exint count);

exint	    insertAt(const T &t, exint index)
{ return insertImpl(t, index); }

bool	    isValidIndex(exint index) const
{ return (index >= 0 && index < mySize); }

exint	    removeIndex(exint index)
{
return isValidIndex(index) ? removeAt(index) : -1;
}
void	    removeLast()
{
if (mySize) removeAt(mySize-1);
}

void	    removeRange(exint begin_i, exint end_i);

void            extractRange(exint begin_i, exint end_i,
UT_Array<T>& dest);

template <typename IsEqual>
exint	    removeIf(IsEqual is_equal);

template <typename IsEqual>
void	    collapseIf(IsEqual is_equal)
{
removeIf(is_equal);
setCapacity(size());
}

void	    move(exint srcIdx, exint destIdx, exint howMany);

void	    cycle(exint howMany);

void	    constant(const T &v);
void	    zero();

exint	    index(const T &t) const { return &t - myData; }
exint	    safeIndex(const T &t) const
{
return (&t >= myData && &t < (myData + mySize))
? &t - myData : -1;
}

void            setCapacity(exint newcapacity);
void            setCapacityIfNeeded(exint mincapacity)
{
if (capacity() < mincapacity)
setCapacity(mincapacity);
}
void            bumpCapacity(exint mincapacity)
{
if (capacity() >= mincapacity)
return;
exint bumped = UTbumpAlloc(capacity());
exint newcapacity = mincapacity;
if (bumped > mincapacity)
newcapacity = bumped;
setCapacity(newcapacity);
}

void            bumpSize(exint newsize)
{
bumpCapacity(newsize);
setSize(newsize);
}
void            bumpEntries(exint newsize)
{
bumpSize(newsize);
}

exint           capacity() const { return myCapacity; }
exint           size() const     { return mySize; }
exint           entries() const  { return mySize; }
bool            isEmpty() const  { return mySize==0; }

void            setSize(exint newsize)
{
if (newsize < 0)
newsize = 0;
if (newsize == mySize)
return;
setCapacityIfNeeded(newsize);
if (mySize > newsize)
trivialDestructRange(myData + newsize, mySize - newsize);
else 
trivialConstructRange(myData + mySize, newsize - mySize);
mySize = newsize;
}
void            entries(exint newsize)
{
setSize(newsize);
}
void            setSizeNoInit(exint newsize)
{
if (newsize < 0)
newsize = 0;
if (newsize == mySize)
return;
setCapacityIfNeeded(newsize);
if (mySize > newsize)
trivialDestructRange(myData + newsize, mySize - newsize);
else if (!isPOD()) 
trivialConstructRange(myData + mySize, newsize - mySize);
mySize = newsize;
}

void            truncate(exint maxsize)
{
if (maxsize >= 0 && size() > maxsize)
setSize(maxsize);
}
void            clear() {
trivialDestructRange(myData, mySize);
mySize = 0;
}

UT_Array<T> &   operator=(const UT_Array<T> &a);

UT_Array<T> &   operator=(std::initializer_list<T> ilist);

UT_Array<T> &   operator=(UT_Array<T> &&a);

bool            operator==(const UT_Array<T> &a) const;
bool            operator!=(const UT_Array<T> &a) const;

T &		    operator()(exint i)
{
UT_ASSERT_P(i >= 0 && i < mySize);
return myData[i];
}
const T &	    operator()(exint i) const
{
UT_ASSERT_P(i >= 0 && i < mySize);
return myData[i];
}

T &		    operator[](exint i)
{
UT_ASSERT_P(i >= 0 && i < mySize);
return myData[i];
}
const T &	    operator[](exint i) const
{
UT_ASSERT_P(i >= 0 && i < mySize);
return myData[i];
}

T &             forcedRef(exint i)
{
UT_ASSERT_P(i >= 0);
if (i >= mySize)
bumpSize(i+1);
return myData[i];
}

T               forcedGet(exint i) const
{
return (i >= 0 && i < mySize) ? myData[i] : T();
}

T &		    last()
{
UT_ASSERT_P(mySize);
return myData[mySize-1];
}
const T &	    last() const
{
UT_ASSERT_P(mySize);
return myData[mySize-1];
}

T *		    getArray() const		    { return myData; }
const T *	    getRawArray() const		    { return myData; }

T *		    array()			    { return myData; }
const T *	    array() const		    { return myData; }

T *		    data()			    { return myData; }
const T *	    data() const		    { return myData; }

T *		    aliasArray(T *newdata)
{ T *data = myData; myData = newdata; return data; }

template <typename IT, bool FORWARD>
class base_iterator : 
public std::iterator<std::random_access_iterator_tag, T, exint> 
{
public:
typedef IT&		reference;
typedef IT*		pointer;

base_iterator() : myCurrent(NULL), myEnd(NULL) {}

template<typename EIT>
base_iterator(const base_iterator<EIT, FORWARD> &src)
: myCurrent(src.myCurrent), myEnd(src.myEnd) {}

pointer	operator->() const 
{ return FORWARD ? myCurrent : myCurrent - 1; }

reference	operator*() const
{ return FORWARD ? *myCurrent : myCurrent[-1]; }

reference	item() const
{ return FORWARD ? *myCurrent : myCurrent[-1]; }

reference	operator[](exint n) const
{ return FORWARD ? myCurrent[n] : myCurrent[-n - 1]; } 

base_iterator &operator++()
{
if (FORWARD) ++myCurrent; else --myCurrent;
return *this;
}
base_iterator operator++(int)
{
base_iterator tmp = *this;
if (FORWARD) ++myCurrent; else --myCurrent;
return tmp;
}
base_iterator &operator--()
{
if (FORWARD) --myCurrent; else ++myCurrent;
return *this;
}
base_iterator operator--(int)
{
base_iterator tmp = *this;
if (FORWARD) --myCurrent; else ++myCurrent;
return tmp;
}

base_iterator &operator+=(exint n)   
{
if (FORWARD)
myCurrent += n;
else
myCurrent -= n;
return *this;
}
base_iterator operator+(exint n) const
{
if (FORWARD)
return base_iterator(myCurrent + n, myEnd);
else
return base_iterator(myCurrent - n, myEnd);
}

base_iterator &operator-=(exint n)
{ return (*this) += (-n); }
base_iterator operator-(exint n) const
{ return (*this) + (-n); }

bool	 atEnd() const		{ return myCurrent == myEnd; }
void	 advance()		{ this->operator++(); }

template<typename ITR, bool FR>
bool 	 operator==(const base_iterator<ITR, FR> &r) const
{ return myCurrent == r.myCurrent; }

template<typename ITR, bool FR>
bool 	 operator!=(const base_iterator<ITR, FR> &r) const
{ return myCurrent != r.myCurrent; }

template<typename ITR>
bool	 operator<(const base_iterator<ITR, FORWARD> &r) const
{
if (FORWARD) 
return myCurrent < r.myCurrent;
else
return r.myCurrent < myCurrent;
}

template<typename ITR>
bool	 operator>(const base_iterator<ITR, FORWARD> &r) const
{
if (FORWARD) 
return myCurrent > r.myCurrent;
else
return r.myCurrent > myCurrent;
}

template<typename ITR>
bool	 operator<=(const base_iterator<ITR, FORWARD> &r) const
{
if (FORWARD) 
return myCurrent <= r.myCurrent;
else
return r.myCurrent <= myCurrent;
}

template<typename ITR>
bool	 operator>=(const base_iterator<ITR, FORWARD> &r) const
{
if (FORWARD) 
return myCurrent >= r.myCurrent;
else
return r.myCurrent >= myCurrent;
}

template<typename ITR>
exint	 operator-(const base_iterator<ITR, FORWARD> &r) const
{
if (FORWARD) 
return exint(myCurrent - r.myCurrent);
else
return exint(r.myCurrent - myCurrent);
}


protected:
friend class UT_Array<T>;
base_iterator(IT *c, IT *e) : myCurrent(c), myEnd(e) {}
private:

IT			*myCurrent;
IT			*myEnd;
};

typedef base_iterator<T, true>		iterator;
typedef base_iterator<const T, true>	const_iterator;
typedef base_iterator<T, false>		reverse_iterator;
typedef base_iterator<const T, false>	const_reverse_iterator;
typedef const_iterator	traverser; 

iterator		begin()
{
return iterator(myData, myData + mySize);
}
iterator		end()
{
return iterator(myData + mySize,
myData + mySize);
}

const_iterator	begin() const
{
return const_iterator(myData, myData + mySize);
}
const_iterator	end() const
{
return const_iterator(myData + mySize,
myData + mySize);
}

reverse_iterator	rbegin()
{
return reverse_iterator(myData + mySize,
myData);
}
reverse_iterator	rend()
{
return reverse_iterator(myData, myData);
}
const_reverse_iterator rbegin() const
{
return const_reverse_iterator(myData + mySize,
myData);
}
const_reverse_iterator rend() const
{
return const_reverse_iterator(myData, myData);
}

void		removeItem(const reverse_iterator &it)
{
removeAt(&it.item() - myData);
}


void	    unsafeShareData(UT_Array<T> &src)
{
myData = src.myData;
myCapacity = src.myCapacity;
mySize = src.mySize;
}
void	    unsafeShareData(T *src, exint srcsize)
{
myData = src;
myCapacity = srcsize;
mySize = srcsize;
}
void	    unsafeShareData(T *src, exint size, exint capacity)
{
myData = src;
mySize = size;
myCapacity = capacity;
}
void	    unsafeClearData()
{
myData = NULL;
myCapacity = 0;
mySize = 0;
}

inline bool	    isHeapBuffer() const
{
return (myData != (T *)(((char*)this) + sizeof(*this)));
}
inline bool	    isHeapBuffer(T* data) const
{
return (data != (T *)(((char*)this) + sizeof(*this)));
}

protected:
static constexpr SYS_FORCE_INLINE bool isPOD()
{
return std::is_pod<T>::value;
}

template <typename S>
exint           appendImpl(S &&s);

template <typename S>
exint           insertImpl(S &&s, exint index);

template <typename... S>
static void	    construct(T &dst, S&&... s)
{
new (&dst) T(std::forward<S>(s)...);
}

static void	    copyConstruct(T &dst, const T &src)
{
if (isPOD())
dst = src;
else
new (&dst) T(src);
}
static void	    copyConstructRange(T *dst, const T *src, exint n)
{
if (isPOD())
{
if (n > 0)
{
::memcpy((void *)dst, (const void *)src,
n * sizeof(T));
}
}
else
{
for (exint i = 0; i < n; i++)
new (&dst[i]) T(src[i]);
}
}

static void	    trivialConstruct(T &dst)
{
if (!isPOD())
new (&dst) T();
else
memset((void *)&dst, 0, sizeof(T));
}
static void	    trivialConstructRange(T *dst, exint n)
{
if (!isPOD())
{
for (exint i = 0; i < n; i++)
new (&dst[i]) T();
}
else if (n == 1)
{
memset((void *)dst, 0, sizeof(T));
}
else
memset((void *)dst, 0, sizeof(T) * n);
}

static void	    trivialDestruct(T &dst)
{
if (!isPOD())
dst.~T();
}
static void	    trivialDestructRange(T *dst, exint n)
{
if (!isPOD())
{
for (exint i = 0; i < n; i++)
dst[i].~T();
}
}

private:
T *myData;

exint myCapacity;

exint mySize;

exint	    removeAt(exint index);

T *		    allocateCapacity(exint num_items);
};

#include "arrayImpl.h"

#endif 
