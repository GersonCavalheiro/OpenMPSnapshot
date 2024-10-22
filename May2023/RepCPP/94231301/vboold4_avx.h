
#pragma once

namespace embree
{

template<>
struct vboold<4>
{
typedef vboold4 Bool;

enum  { size = 4 };       
union {                   
__m256d v;
struct { __m128d vl,vh; };
long long i[4];
};


__forceinline vboold() {}
__forceinline vboold(const vboold4& a) { v = a.v; }
__forceinline vboold4& operator =(const vboold4& a) { v = a.v; return *this; }

__forceinline vboold(__m256d a) : v(a) {}
__forceinline vboold(__m256i a) : v(_mm256_castsi256_pd(a)) {}

__forceinline operator const __m256() const { return _mm256_castpd_ps(v); }
__forceinline operator const __m256i() const { return _mm256_castpd_si256(v); }
__forceinline operator const __m256d() const { return v; }

__forceinline vboold(int a)
{
assert(a >= 0 && a <= 255);
#if defined (__AVX2__)
const __m256i mask = _mm256_set_epi64x(0x8, 0x4, 0x2, 0x1);
const __m256i b = _mm256_set1_epi64x(a);
const __m256i c = _mm256_and_si256(b,mask);
v = _mm256_castsi256_pd(_mm256_cmpeq_epi64(c,mask));
#else
vl = _mm_lookupmask_pd[a & 0x3];
vh = _mm_lookupmask_pd[a >> 2];
#endif
}

__forceinline vboold(__m128d a, __m128d b) : vl(a), vh(b) {}


__forceinline vboold(FalseTy) : v(_mm256_setzero_pd()) {}
__forceinline vboold(TrueTy)  : v(_mm256_cmp_pd(_mm256_setzero_pd(), _mm256_setzero_pd(), _CMP_EQ_OQ)) {}


__forceinline bool       operator [](size_t index) const { assert(index < 4); return (_mm256_movemask_pd(v) >> index) & 1; }
__forceinline long long& operator [](size_t index)       { assert(index < 4); return i[index]; }
};


__forceinline vboold4 operator !(const vboold4& a) { return _mm256_xor_pd(a, vboold4(embree::True)); }


__forceinline vboold4 operator &(const vboold4& a, const vboold4& b) { return _mm256_and_pd(a, b); }
__forceinline vboold4 operator |(const vboold4& a, const vboold4& b) { return _mm256_or_pd (a, b); }
__forceinline vboold4 operator ^(const vboold4& a, const vboold4& b) { return _mm256_xor_pd(a, b); }

__forceinline vboold4& operator &=(vboold4& a, const vboold4& b) { return a = a & b; }
__forceinline vboold4& operator |=(vboold4& a, const vboold4& b) { return a = a | b; }
__forceinline vboold4& operator ^=(vboold4& a, const vboold4& b) { return a = a ^ b; }


__forceinline vboold4 operator !=(const vboold4& a, const vboold4& b) { return _mm256_xor_pd(a, b); }
__forceinline vboold4 operator ==(const vboold4& a, const vboold4& b) { return _mm256_xor_pd(_mm256_xor_pd(a,b),vboold4(embree::True)); }

__forceinline vboold4 select(const vboold4& mask, const vboold4& t, const vboold4& f) {
return _mm256_blendv_pd(f, t, mask); 
}


__forceinline vboold4 unpacklo(const vboold4& a, const vboold4& b) { return _mm256_unpacklo_pd(a, b); }
__forceinline vboold4 unpackhi(const vboold4& a, const vboold4& b) { return _mm256_unpackhi_pd(a, b); }


#if defined(__AVX2__)
template<int i0, int i1, int i2, int i3>
__forceinline vboold4 shuffle(const vboold4& v) {
return _mm256_permute4x64_pd(v, _MM_SHUFFLE(i3, i2, i1, i0));
}

template<int i>
__forceinline vboold4 shuffle(const vboold4& v) {
return _mm256_permute4x64_pd(v, _MM_SHUFFLE(i, i, i, i));
}
#endif



__forceinline bool reduce_and(const vboold4& a) { return _mm256_movemask_pd(a) == (unsigned int)0xf; }
__forceinline bool reduce_or (const vboold4& a) { return !_mm256_testz_pd(a,a); }

__forceinline bool all (const vboold4& a) { return _mm256_movemask_pd(a) == (unsigned int)0xf; }
__forceinline bool any (const vboold4& a) { return !_mm256_testz_pd(a,a); }
__forceinline bool none(const vboold4& a) { return _mm256_testz_pd(a,a) != 0; }

__forceinline bool all (const vboold4& valid, const vboold4& b) { return all((!valid) | b); }
__forceinline bool any (const vboold4& valid, const vboold4& b) { return any(valid & b); }
__forceinline bool none(const vboold4& valid, const vboold4& b) { return none(valid & b); }

__forceinline unsigned int movemask(const vboold4& a) { return _mm256_movemask_pd(a); }
__forceinline size_t       popcnt  (const vboold4& a) { return __popcnt((size_t)_mm256_movemask_pd(a)); }


__forceinline bool get(const vboold4& a, size_t index) { return a[index]; }
__forceinline void set  (vboold4& a, size_t index)     { a[index] = -1; }
__forceinline void clear(vboold4& a, size_t index)     { a[index] =  0; }


inline std::ostream& operator <<(std::ostream& cout, const vboold4& a) {
return cout << "<" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ", "
<< a[4] << ", " << a[5] << ", " << a[6] << ", " << a[7] << ">";
}
}
