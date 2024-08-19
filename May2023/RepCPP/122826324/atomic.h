

#ifndef __TBB_atomic_H
#define __TBB_atomic_H

#include "tbb_stddef.h"
#include <cstddef>

#if _MSC_VER
#define __TBB_LONG_LONG __int64
#else
#define __TBB_LONG_LONG long long
#endif 

#include "tbb_machine.h"

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4244 4267)
#endif

namespace tbb {

enum memory_semantics {
full_fence,
acquire,
release,
relaxed
};

namespace internal {

#if __TBB_ATTRIBUTE_ALIGNED_PRESENT
#define __TBB_DECL_ATOMIC_FIELD(t,f,a) t f  __attribute__ ((aligned(a)));
#elif __TBB_DECLSPEC_ALIGN_PRESENT
#define __TBB_DECL_ATOMIC_FIELD(t,f,a) __declspec(align(a)) t f;
#else
#error Do not know syntax for forcing alignment.
#endif

template<size_t S>
struct atomic_rep;           

template<>
struct atomic_rep<1> {       
typedef int8_t word;
int8_t value;
};
template<>
struct atomic_rep<2> {       
typedef int16_t word;
__TBB_DECL_ATOMIC_FIELD(int16_t,value,2)
};
template<>
struct atomic_rep<4> {       
#if _MSC_VER && !_WIN64
typedef intptr_t word;
#else
typedef int32_t word;
#endif
__TBB_DECL_ATOMIC_FIELD(int32_t,value,4)
};
#if __TBB_64BIT_ATOMICS
template<>
struct atomic_rep<8> {       
typedef int64_t word;
__TBB_DECL_ATOMIC_FIELD(int64_t,value,8)
};
#endif

template<size_t Size, memory_semantics M>
struct atomic_traits;        

#define __TBB_DECL_FENCED_ATOMIC_PRIMITIVES(S,M)                                                         \
template<> struct atomic_traits<S,M> {                                                               \
typedef atomic_rep<S>::word word;                                                                \
inline static word compare_and_swap( volatile void* location, word new_value, word comparand ) { \
return __TBB_machine_cmpswp##S##M(location,new_value,comparand);                             \
}                                                                                                \
inline static word fetch_and_add( volatile void* location, word addend ) {                       \
return __TBB_machine_fetchadd##S##M(location,addend);                                        \
}                                                                                                \
inline static word fetch_and_store( volatile void* location, word value ) {                      \
return __TBB_machine_fetchstore##S##M(location,value);                                       \
}                                                                                                \
};

#define __TBB_DECL_ATOMIC_PRIMITIVES(S)                                                                  \
template<memory_semantics M>                                                                         \
struct atomic_traits<S,M> {                                                                          \
typedef atomic_rep<S>::word word;                                                                \
inline static word compare_and_swap( volatile void* location, word new_value, word comparand ) { \
return __TBB_machine_cmpswp##S(location,new_value,comparand);                                \
}                                                                                                \
inline static word fetch_and_add( volatile void* location, word addend ) {                       \
return __TBB_machine_fetchadd##S(location,addend);                                           \
}                                                                                                \
inline static word fetch_and_store( volatile void* location, word value ) {                      \
return __TBB_machine_fetchstore##S(location,value);                                          \
}                                                                                                \
};

template<memory_semantics M>
struct atomic_load_store_traits;    

#define __TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(M)                      \
template<> struct atomic_load_store_traits<M> {                     \
template <typename T>                                           \
inline static T load( const volatile T& location ) {            \
return __TBB_load_##M( location );                          \
}                                                               \
template <typename T>                                           \
inline static void store( volatile T& location, T value ) {     \
__TBB_store_##M( location, value );                         \
}                                                               \
}

#if __TBB_USE_FENCED_ATOMICS
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(1,full_fence)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(2,full_fence)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(4,full_fence)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(1,acquire)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(2,acquire)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(4,acquire)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(1,release)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(2,release)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(4,release)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(1,relaxed)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(2,relaxed)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(4,relaxed)
#if __TBB_64BIT_ATOMICS
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(8,full_fence)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(8,acquire)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(8,release)
__TBB_DECL_FENCED_ATOMIC_PRIMITIVES(8,relaxed)
#endif
#else 
__TBB_DECL_ATOMIC_PRIMITIVES(1)
__TBB_DECL_ATOMIC_PRIMITIVES(2)
__TBB_DECL_ATOMIC_PRIMITIVES(4)
#if __TBB_64BIT_ATOMICS
__TBB_DECL_ATOMIC_PRIMITIVES(8)
#endif
#endif 

__TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(full_fence);
__TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(acquire);
__TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(release);
__TBB_DECL_ATOMIC_LOAD_STORE_PRIMITIVES(relaxed);


#define __TBB_MINUS_ONE(T) (T(T(0)-T(1)))


template<typename T>
struct atomic_impl {
protected:
atomic_rep<sizeof(T)> rep;
private:
union converter {
T value;
typename atomic_rep<sizeof(T)>::word bits;
};
public:
typedef T value_type;

template<memory_semantics M>
value_type fetch_and_store( value_type value ) {
converter u, w;
u.value = value;
w.bits = internal::atomic_traits<sizeof(value_type),M>::fetch_and_store(&rep.value,u.bits);
return w.value;
}

value_type fetch_and_store( value_type value ) {
return fetch_and_store<full_fence>(value);
}

template<memory_semantics M>
value_type compare_and_swap( value_type value, value_type comparand ) {
converter u, v, w;
u.value = value;
v.value = comparand;
w.bits = internal::atomic_traits<sizeof(value_type),M>::compare_and_swap(&rep.value,u.bits,v.bits);
return w.value;
}

value_type compare_and_swap( value_type value, value_type comparand ) {
return compare_and_swap<full_fence>(value,comparand);
}

operator value_type() const volatile {                
converter w;
w.bits = __TBB_load_with_acquire( rep.value );
return w.value;
}

template<memory_semantics M>
value_type load () const {
converter u;
u.bits = internal::atomic_load_store_traits<M>::load( rep.value );
return u.value;
}

value_type load () const {
return load<acquire>();
}

template<memory_semantics M>
void store ( value_type value ) {
converter u;
u.value = value;
internal::atomic_load_store_traits<M>::store( rep.value, u.bits );
}

void store ( value_type value ) {
store<release>( value );
}

protected:
value_type store_with_release( value_type rhs ) {
converter u;
u.value = rhs;
__TBB_store_with_release(rep.value,u.bits);
return rhs;
}
};


template<typename I, typename D, typename StepType>
struct atomic_impl_with_arithmetic: atomic_impl<I> {
public:
typedef I value_type;

template<memory_semantics M>
value_type fetch_and_add( D addend ) {
return value_type(internal::atomic_traits<sizeof(value_type),M>::fetch_and_add( &this->rep.value, addend*sizeof(StepType) ));
}

value_type fetch_and_add( D addend ) {
return fetch_and_add<full_fence>(addend);
}

template<memory_semantics M>
value_type fetch_and_increment() {
return fetch_and_add<M>(1);
}

value_type fetch_and_increment() {
return fetch_and_add(1);
}

template<memory_semantics M>
value_type fetch_and_decrement() {
return fetch_and_add<M>(__TBB_MINUS_ONE(D));
}

value_type fetch_and_decrement() {
return fetch_and_add(__TBB_MINUS_ONE(D));
}

public:
value_type operator+=( D value ) {
return fetch_and_add(value)+value;
}

value_type operator-=( D value ) {
return operator+=(D(0)-value);
}

value_type operator++() {
return fetch_and_add(1)+1;
}

value_type operator--() {
return fetch_and_add(__TBB_MINUS_ONE(D))-1;
}

value_type operator++(int) {
return fetch_and_add(1);
}

value_type operator--(int) {
return fetch_and_add(__TBB_MINUS_ONE(D));
}
};

} 


template<typename T>
struct atomic: internal::atomic_impl<T> {
T operator=( T rhs ) {
return this->store_with_release(rhs);
}
atomic<T>& operator=( const atomic<T>& rhs ) {this->store_with_release(rhs); return *this;}
};

#define __TBB_DECL_ATOMIC(T) \
template<> struct atomic<T>: internal::atomic_impl_with_arithmetic<T,T,char> {  \
T operator=( T rhs ) {return store_with_release(rhs);}  \
atomic<T>& operator=( const atomic<T>& rhs ) {store_with_release(rhs); return *this;}  \
};

#if __TBB_64BIT_ATOMICS
__TBB_DECL_ATOMIC(__TBB_LONG_LONG)
__TBB_DECL_ATOMIC(unsigned __TBB_LONG_LONG)
#else
#endif
__TBB_DECL_ATOMIC(long)
__TBB_DECL_ATOMIC(unsigned long)

#if _MSC_VER && !_WIN64

#define __TBB_DECL_ATOMIC_ALT(T,U) \
template<> struct atomic<T>: internal::atomic_impl_with_arithmetic<T,T,char> {  \
T operator=( U rhs ) {return store_with_release(T(rhs));}  \
atomic<T>& operator=( const atomic<T>& rhs ) {store_with_release(rhs); return *this;}  \
};
__TBB_DECL_ATOMIC_ALT(unsigned,size_t)
__TBB_DECL_ATOMIC_ALT(int,ptrdiff_t)
#else
__TBB_DECL_ATOMIC(unsigned)
__TBB_DECL_ATOMIC(int)
#endif 

__TBB_DECL_ATOMIC(unsigned short)
__TBB_DECL_ATOMIC(short)
__TBB_DECL_ATOMIC(char)
__TBB_DECL_ATOMIC(signed char)
__TBB_DECL_ATOMIC(unsigned char)

#if !_MSC_VER || defined(_NATIVE_WCHAR_T_DEFINED)
__TBB_DECL_ATOMIC(wchar_t)
#endif 

template<typename T> struct atomic<T*>: internal::atomic_impl_with_arithmetic<T*,ptrdiff_t,T> {
T* operator=( T* rhs ) {
return this->store_with_release(rhs);
}
atomic<T*>& operator=( const atomic<T*>& rhs ) {
this->store_with_release(rhs); return *this;
}
T* operator->() const {
return (*this);
}
};

template<> struct atomic<void*>: internal::atomic_impl<void*> {
void* operator=( void* rhs ) {
return this->store_with_release(rhs);
}
atomic<void*>& operator=( const atomic<void*>& rhs ) {
this->store_with_release(rhs); return *this;
}
};


template <memory_semantics M, typename T>
T load ( const atomic<T>& a ) { return a.template load<M>(); }

template <memory_semantics M, typename T>
void store ( atomic<T>& a, T value ) { return a.template store<M>(value); }

namespace interface6{
template<typename T>
atomic<T> make_atomic(T t) {
atomic<T> a;
store<relaxed>(a,t);
return a;
}
}
using interface6::make_atomic;

namespace internal {

template<typename T>
inline atomic<T>& as_atomic( T& t ) {
return (atomic<T>&)t;
}
} 

} 

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning (pop)
#endif 

#endif 
