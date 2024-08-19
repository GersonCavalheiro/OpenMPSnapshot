#pragma once

#include "entity.hpp"
#include "types.hpp"
#include <array>
#include <bitset>
#include <cassert>
#include <set>
#include <type_traits>



template <class Type, class StoragePool>
class MarkedPoolIterator
{
public:
using iterator_category = typename FlatPtrHashSet<Type>::const_iterator::iterator_category;
using difference_type = typename FlatPtrHashSet<Type>::const_iterator::difference_type;
using value_type = typename FlatPtrHashSet<Type>::const_iterator::value_type;
using pointer = typename FlatPtrHashSet<Type>::const_iterator::pointer;
using reference = typename FlatPtrHashSet<Type>::const_iterator::reference;

private:
StoragePool& pool; 
int lockedID; 
const FlatPtrHashSet<Type>& entries; 
typename FlatPtrHashSet<Type>::const_iterator iter; 

inline void lock()
{
assert(lockedID == -1);
if (iter != entries.end())
{
lockedID = (*iter)->getID();
pool.lock(lockedID);
}
}

inline void unlock()
{
if (lockedID != -1)
{
pool.unlock(lockedID);
lockedID = -1;
}
}

public:
inline MarkedPoolIterator(StoragePool& pool, const FlatPtrHashSet<Type>& entries, typename FlatPtrHashSet<Type>::const_iterator iter)
: pool(pool)
, lockedID(-1)
, entries(entries)
, iter(iter)
{
lock();
}

inline ~MarkedPoolIterator()
{
unlock();
}

inline reference operator*() const { return *iter; }
inline pointer operator->() { return iter.operator->(); }

inline MarkedPoolIterator<Type, StoragePool>& operator++()
{
++iter;
unlock();
lock();
return *this;
}

inline friend bool operator==(const MarkedPoolIterator<Type, StoragePool>& a, const MarkedPoolIterator<Type, StoragePool>& b)
{
return a.iter == b.iter;
};
inline friend bool operator!=(const MarkedPoolIterator<Type, StoragePool>& a, const MarkedPoolIterator<Type, StoragePool>& b)
{
return a.iter != b.iter;
};
};



template <typename T>
struct IReadOnlyPool
{
virtual T* get(int index) = 0;

virtual Pair<size_t, size_t> bounds() const = 0;
};

template <typename T>
struct PoolEventHandler
{
virtual void onPoolEntryCreated(T& entry) { }
virtual void onPoolEntryDestroyed(T& entry) { }
};

template <typename T>
struct IPool : IReadOnlyPool<T>
{
using Iterator = MarkedPoolIterator<T, IPool<T>>;

virtual void release(int index) = 0;

virtual void lock(int index) = 0;

virtual bool unlock(int index) = 0;

virtual IEventDispatcher<PoolEventHandler<T>>& getPoolEventDispatcher() = 0;

inline Iterator begin()
{
return Iterator(*this, entries(), entries().begin());
}

inline Iterator end()
{
return Iterator(*this, entries(), entries().end());
}

inline size_t count()
{
return entries().size();
}

protected:
virtual const FlatPtrHashSet<T>& entries() = 0;
};

template <typename T>
struct IPoolComponent : public IComponent, public IPool<T>
{
ComponentType componentType() const override { return ComponentType::Pool; }
};
