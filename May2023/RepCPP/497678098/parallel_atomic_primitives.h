

#ifndef SEQAN_PARALLEL_PARALLEL_ATOMIC_PRIMITIVES_H_
#define SEQAN_PARALLEL_PARALLEL_ATOMIC_PRIMITIVES_H_

#if defined(STDLIB_VS)
#include <intrin.h>
#endif  

namespace seqan {





template <typename T>
struct Atomic
{
typedef std::atomic<T> Type;
};

















#ifndef SEQAN_CACHE_LINE_SIZE
#define SEQAN_CACHE_LINE_SIZE 128
#endif

#if defined(STDLIB_VS)


#if !defined(COMPILER_CLANG)
#pragma intrinsic(_InterlockedOr, _InterlockedXor, _InterlockedCompareExchange)
#endif


template <typename T, typename S>
inline T _atomicOr(T volatile &x, ConstInt<sizeof(char)>, S y) { return _InterlockedOr8(reinterpret_cast<char volatile *>(&x), y); }
template <typename T, typename S>
inline T _atomicXor(T volatile &x, ConstInt<sizeof(char)>, S y) { return _InterlockedXor8(reinterpret_cast<char volatile *>(&x), y); }

template <typename T>
inline T _atomicInc(T volatile &x, ConstInt<sizeof(short)>) { return InterlockedIncrement16(reinterpret_cast<short volatile *>(&x)); }
template <typename T>
inline T _atomicDec(T volatile &x, ConstInt<sizeof(short)>) { return InterlockedDecrement16(reinterpret_cast<short volatile *>(&x)); }
template <typename T, typename S>
inline T _atomicAdd(T volatile &x, ConstInt<sizeof(short)>, S y) { return InterlockedExchangeAdd16(reinterpret_cast<short volatile *>(&x), y); }
template <typename T, typename S>
inline T _atomicOr(T volatile &x, ConstInt<sizeof(short)>, S y) { return _InterlockedOr16(reinterpret_cast<short volatile *>(&x), y); }
template <typename T, typename S>
inline T _atomicXor(T volatile &x, ConstInt<sizeof(short)>, S y) { return _InterlockedXor16(reinterpret_cast<short volatile *>(&x), y); }
template <typename T, typename S, typename U>
inline T _atomicCas(T volatile &x, ConstInt<sizeof(short)>, S cmp, U y) { return _InterlockedCompareExchange16(reinterpret_cast<short volatile *>(&x), y, cmp); }

template <typename T>
inline T _atomicInc(T volatile &x, ConstInt<sizeof(LONG)>) { return InterlockedIncrement(reinterpret_cast<LONG volatile *>(&x)); }
template <typename T>
inline T* _atomicInc(T* volatile &x, ConstInt<sizeof(LONG)>) { InterlockedExchangeAdd(reinterpret_cast<LONG volatile *>(&x), sizeof(LONG)); return x; }
template <typename T>
inline T _atomicDec(T volatile &x, ConstInt<sizeof(LONG)>) { return InterlockedDecrement(reinterpret_cast<LONG volatile *>(&x)); }
template <typename T>
inline T* _atomicDec(T* volatile &x, ConstInt<sizeof(LONG)>) { InterlockedExchangeAdd(reinterpret_cast<LONG volatile *>(&x), -sizeof(LONG)); return x; }
template <typename T, typename S>
inline T _atomicAdd(T volatile &x, ConstInt<sizeof(LONG)>, S y) { return InterlockedExchangeAdd(reinterpret_cast<LONG volatile *>(&x), y); }
template <typename T, typename S>
inline T _atomicOr(T volatile &x, ConstInt<sizeof(long)>, S y) { return _InterlockedOr(reinterpret_cast<long volatile *>(&x), y); }
template <typename T, typename S>
inline T _atomicXor(T volatile &x, ConstInt<sizeof(long)>, S y) { return _InterlockedXor(reinterpret_cast<long volatile *>(&x), y); }
template <typename T, typename S, typename U>
inline T _atomicCas(T volatile &x, ConstInt<sizeof(long)>, S cmp, U y) { return _InterlockedCompareExchange(reinterpret_cast<long volatile *>(&x), y, cmp); }

#ifdef _WIN64
template <typename T>
inline T _atomicInc(T volatile &x, ConstInt<sizeof(LONGLONG)>) { return InterlockedIncrement64(reinterpret_cast<LONGLONG volatile *>(&x)); }
template <typename T>
inline T* _atomicInc(T* volatile &x, ConstInt<sizeof(LONGLONG)>) { InterlockedExchangeAdd64(reinterpret_cast<LONGLONG volatile *>(&x), sizeof(LONGLONG)); return x; }
template <typename T>
inline T _atomicDec(T volatile &x, ConstInt<sizeof(LONGLONG)>) { return InterlockedDecrement64(reinterpret_cast<LONGLONG volatile *>(&x)); }
template <typename T>
inline T* _atomicDec(T* volatile &x, ConstInt<sizeof(LONGLONG)>) { InterlockedExchangeAdd64(reinterpret_cast<LONGLONG volatile *>(&x), -sizeof(LONGLONG)); return x; }
template <typename T, typename S>
inline T _atomicAdd(T volatile &x, ConstInt<sizeof(LONGLONG)>, S y) { return InterlockedExchangeAdd64(reinterpret_cast<LONGLONG volatile *>(&x), y); }
template <typename T, typename S>
inline T _atomicOr(T volatile &x, ConstInt<sizeof(int64_t)>, S y) { return _InterlockedOr64(reinterpret_cast<int64_t volatile *>(&x), y); }
template <typename T, typename S>
inline T _atomicXor(T volatile &x, ConstInt<sizeof(int64_t)>, S y) { return _InterlockedXor64(reinterpret_cast<int64_t volatile *>(&x), y); }
template <typename T, typename S, typename U>
inline T _atomicCas(T volatile &x, ConstInt<sizeof(int64_t)>, S cmp, U y) { return _InterlockedCompareExchange64(reinterpret_cast<int64_t volatile *>(&x), y, cmp); }
#endif  

template <typename T>
inline T atomicInc(T volatile & x) { return _atomicInc(x, ConstInt<sizeof(T)>()); }
template <typename T>
inline T atomicDec(T volatile & x) { return _atomicDec(x, ConstInt<sizeof(T)>()); }
template <typename T, typename S>
inline T atomicAdd(T volatile &x, S y) { return _atomicAdd(x, ConstInt<sizeof(T)>(), y); }
template <typename T, typename S>
inline T atomicOr(T volatile &x, S y) { return _atomicOr(x, ConstInt<sizeof(T)>(), y); }
template <typename T, typename S>
inline T atomicXor(T volatile &x, S y) { return _atomicXor(x, ConstInt<sizeof(T)>(), y); }
template <typename T, typename S, typename U>
inline T atomicCas(T volatile &x, S cmp, U y) { return _atomicCas(x, ConstInt<sizeof(T)>(), cmp, y); }
template <typename T, typename S, typename U>
inline bool atomicCasBool(T volatile &x, S cmp, U y) { return _atomicCas(x, ConstInt<sizeof(T)>(), cmp, y) == cmp; }

template <typename T>
inline T atomicPostInc(T volatile & x) { return atomicInc(x) - 1; }
template <typename T>
inline T atomicPostDec(T volatile & x) { return atomicDec(x) + 1; }


#else  


template <typename T>
inline T atomicInc(T volatile & x)
{
return __sync_add_and_fetch(&x, 1);
}

template <typename T>
inline T atomicPostInc(T volatile & x)
{
return __sync_fetch_and_add(&x, 1);
}

template <typename T>
inline T atomicDec(T volatile & x)
{
return __sync_add_and_fetch(&x, -1);
}

template <typename T>
inline T atomicPostDec(T volatile & x)
{
return __sync_fetch_and_add(&x, -1);
}

template <typename T1, typename T2>
inline T1 atomicAdd(T1 volatile & x, T2 y)
{
return __sync_add_and_fetch(&x, y);
}

template <typename T>
inline T atomicOr(T volatile & x, T y)
{
return __sync_or_and_fetch(&x, y);
}

template <typename T>
inline T atomicXor(T volatile & x, T y)
{
return __sync_xor_and_fetch(&x, y);
}

template <typename T>
inline T atomicCas(T volatile & x, T cmp, T y)
{
return __sync_val_compare_and_swap(&x, cmp, y);
}

template <typename T>
inline bool atomicCasBool(T volatile & x, T cmp, T y)
{
return __sync_bool_compare_and_swap(&x, cmp, y);
}

template <typename T>
inline T atomicSwap(T volatile & x, T y)
{
return __sync_lock_test_and_set(x, y);
}


template <typename T>
inline T * atomicInc(T * volatile & x)
{
return (T *) __sync_add_and_fetch((size_t volatile *)&x, sizeof(T));
}

template <typename T>
inline T * atomicPostInc(T * volatile & x)
{
return (T *) __sync_fetch_and_add((size_t volatile *)&x, sizeof(T));
}

template <typename T>
inline T * atomicDec(T * volatile & x)
{
return (T *) __sync_add_and_fetch((size_t volatile *)&x, -sizeof(T));
}

template <typename T>
inline T * atomicPostDec(T * volatile & x)
{
return (T *) __sync_fetch_and_add((size_t volatile *)&x, -sizeof(T));
}

template <typename T1, typename T2>
inline T1 * atomicAdd(T1 * volatile & x, T2 y)
{
return (T1 *) __sync_add_and_fetch((size_t volatile *)&x, y * sizeof(T2));
}

#endif  



template <typename T>   inline T atomicInc(T          & x,             Serial)      { return ++x;                    }
template <typename T>   inline T atomicPostInc(T      & x,             Serial)      { return x++;                    }
template <typename T>   inline T atomicDec(T          & x,             Serial)      { return --x;                    }
template <typename T>   inline T atomicPostDec(T      & x,             Serial)      { return x--;                    }
template <typename T>   inline T atomicOr (T          & x, T y,        Serial)      { return x |= y;                 }
template <typename T>   inline T atomicXor(T          & x, T y,        Serial)      { return x ^= y;                 }
template <typename T>   inline T atomicCas(T          & x, T cmp, T y, Serial)      { if (x == cmp) x = y; return x; }
template <typename T>   inline bool atomicCasBool(T volatile & x, T, T y, Serial)   { x = y; return true;            }

template <typename T>   inline T atomicInc(T volatile & x,             Parallel)    { return atomicInc(x);           }
template <typename T>   inline T atomicPostInc(T volatile & x,         Parallel)    { return atomicPostInc(x);       }
template <typename T>   inline T atomicDec(T volatile & x,             Parallel)    { return atomicDec(x);           }
template <typename T>   inline T atomicPostDec(T volatile & x,         Parallel)    { return atomicPostDec(x);       }
template <typename T>   inline T atomicOr (T volatile & x, T y,        Parallel)    { return atomicOr(x, y);         }
template <typename T>   inline T atomicXor(T volatile & x, T y,        Parallel)    { return atomicXor(x, y);        }
template <typename T>   inline T atomicCas(T volatile & x, T cmp, T y, Parallel)    { return atomicCas(x, cmp, y);   }
template <typename T>   inline bool atomicCasBool(T volatile & x, T cmp, T y, Parallel) { return atomicCasBool(x, cmp, y); }

template <typename T1, typename T2>   inline T1 atomicAdd(T1          & x, T2 y, Serial)    { return x = x + y; }
template <typename T1, typename T2>   inline T1 atomicAdd(T1 volatile & x, T2 y, Parallel)  { return atomicAdd(x, y); }



template <typename T>   inline T atomicInc(std::atomic<T>        & x     )        { return ++x;                    }
template <typename T>   inline T atomicPostInc(std::atomic<T>    & x     )        { return x++;                    }
template <typename T>   inline T atomicDec(std::atomic<T>        & x     )        { return --x;                    }
template <typename T>   inline T atomicPostDec(std::atomic<T>    & x     )        { return x--;                    }
template <typename T>   inline T atomicOr (std::atomic<T>        & x, T y)        { return x |= y;                 }
template <typename T>   inline T atomicXor(std::atomic<T>        & x, T y)        { return x ^= y;                 }
template <typename T>   inline T atomicCas(std::atomic<T>        & x, T cmp, T y, Serial)   { if (x == cmp) x = y;             return x;   }
template <typename T>   inline T atomicCas(std::atomic<T>        & x, T cmp, T y, Parallel) { x.compare_exchange_weak(cmp, y); return cmp; }
template <typename T>   inline bool atomicCasBool(std::atomic<T> & x, T    , T y, Serial)   { x = y; return true;                          }
template <typename T>   inline bool atomicCasBool(std::atomic<T> & x, T cmp, T y, Parallel) { return x.compare_exchange_weak(cmp, y);      }

} 

#endif  
