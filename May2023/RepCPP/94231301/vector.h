
#pragma once

#include "alloc.h"

namespace embree
{
template<typename T, typename allocator>
class vector_t
{
public:
typedef T value_type;
typedef T* iterator;
typedef const T* const_iterator;

__forceinline vector_t () 
: size_active(0), size_alloced(0), items(nullptr) {}

__forceinline explicit vector_t (size_t sz) 
: size_active(0), size_alloced(0), items(nullptr) { internal_resize_init(sz); }

template<typename M>
__forceinline explicit vector_t (M alloc, size_t sz) 
: alloc(alloc), size_active(0), size_alloced(0), items(nullptr) { internal_resize_init(sz); }

__forceinline ~vector_t() {
clear();
}

__forceinline vector_t (const vector_t& other)
{
size_active = other.size_active;
size_alloced = other.size_alloced;
items = alloc.allocate(size_alloced);
for (size_t i=0; i<size_active; i++) 
::new (&items[i]) value_type(other.items[i]);
}

__forceinline vector_t (vector_t&& other)
: alloc(std::move(other.alloc))
{
size_active = other.size_active; other.size_active = 0;
size_alloced = other.size_alloced; other.size_alloced = 0;
items = other.items; other.items = nullptr;
}

__forceinline vector_t& operator=(const vector_t& other) 
{
resize(other.size_active);
for (size_t i=0; i<size_active; i++) 
::new (&items[i]) value_type(other.items[i]);
return *this;
}

__forceinline vector_t& operator=(vector_t&& other) 
{
alloc = std::move(other.alloc);
size_active = other.size_active; other.size_active = 0;
size_alloced = other.size_alloced; other.size_alloced = 0;
items = other.items; other.items = nullptr;
return *this;
}



__forceinline       iterator begin()       { return items; };
__forceinline const_iterator begin() const { return items; };

__forceinline       iterator end  ()       { return items+size_active; };
__forceinline const_iterator end  () const { return items+size_active; };




__forceinline bool   empty    () const { return size_active == 0; }
__forceinline size_t size     () const { return size_active; }
__forceinline size_t capacity () const { return size_alloced; }


__forceinline void resize(size_t new_size) {
internal_resize(new_size,internal_grow_size(new_size));
}

__forceinline void reserve(size_t new_alloced) 
{

if (new_alloced <= size_alloced) 
return;


internal_resize(size_active,new_alloced);
}

__forceinline void shrink_to_fit() {
internal_resize(size_active,size_active);
}



__forceinline       T& operator[](size_t i)       { assert(i < size_active); return items[i]; }
__forceinline const T& operator[](size_t i) const { assert(i < size_active); return items[i]; }

__forceinline       T& at(size_t i)       { assert(i < size_active); return items[i]; }
__forceinline const T& at(size_t i) const { assert(i < size_active); return items[i]; }

__forceinline T& front() const { assert(size_active > 0); return items[0]; };
__forceinline T& back () const { assert(size_active > 0); return items[size_active-1]; };

__forceinline       T* data()       { return items; };
__forceinline const T* data() const { return items; };




__forceinline void push_back(const T& nt) 
{
const T v = nt; 
internal_resize(size_active,internal_grow_size(size_active+1));
::new (&items[size_active++]) T(v);
}

__forceinline void pop_back() 
{
assert(!empty());
size_active--;
alloc.destroy(&items[size_active]);
}

__forceinline void clear() 
{

for (size_t i=0; i<size_active; i++)
alloc.destroy(&items[i]);


alloc.deallocate(items,size_alloced); 
items = nullptr;
size_active = size_alloced = 0;
}



friend bool operator== (const vector_t& a, const vector_t& b) 
{
if (a.size() != b.size()) return false;
for (size_t i=0; i<a.size(); i++)
if (a[i] != b[i])
return false;
return true;
}

friend bool operator!= (const vector_t& a, const vector_t& b) {
return !(a==b);
}

private:

__forceinline void internal_resize_init(size_t new_active)
{
assert(size_active == 0); 
assert(size_alloced == 0);
assert(items == nullptr);
if (new_active == 0) return;
items = alloc.allocate(new_active);
for (size_t i=0; i<new_active; i++) ::new (&items[i]) T();
size_active = new_active;
size_alloced = new_active;
}

__forceinline void internal_resize(size_t new_active, size_t new_alloced)
{
assert(new_active <= new_alloced); 


if (new_active < size_active) 
{
for (size_t i=new_active; i<size_active; i++)
alloc.destroy(&items[i]);
size_active = new_active;
}


if (new_alloced == size_alloced) {
for (size_t i=size_active; i<new_active; i++) ::new (&items[i]) T;
size_active = new_active;
return;
}


T* old_items = items;
items = alloc.allocate(new_alloced);
for (size_t i=0; i<size_active; i++) {
::new (&items[i]) T(std::move(old_items[i]));
alloc.destroy(&old_items[i]);
}
for (size_t i=size_active; i<new_active; i++) {
::new (&items[i]) T;
}
alloc.deallocate(old_items,size_alloced);
size_active = new_active;
size_alloced = new_alloced;
}

__forceinline size_t internal_grow_size(size_t new_alloced)
{

if (new_alloced <= size_alloced) 
return size_alloced;


size_t new_size_alloced = size_alloced;
while (new_size_alloced < new_alloced) {
new_size_alloced = 2*new_size_alloced;
if (new_size_alloced == 0) new_size_alloced = 1;
}
return new_size_alloced;
}

private:
allocator alloc;
size_t size_active;    
size_t size_alloced;   
T* items;              
};


template<typename T>
using vector = vector_t<T,std::allocator<T>>;


template<typename T>
using avector = vector_t<T,aligned_allocator<T,std::alignment_of<T>::value> >;


template<typename T>
using ovector = vector_t<T,os_allocator<T> >;
}
