
#pragma once

namespace embree
{

template<>
struct vboold<8>
{
typedef vboold8 Bool;
typedef vint8   Int;

enum { size = 8 }; 
__mmask8 v;        


__forceinline vboold() {}
__forceinline vboold(const vboold8& t) { v = t.v; }
__forceinline vboold8& operator =(const vboold8& f) { v = f.v; return *this; }

__forceinline vboold(const __mmask8& t) { v = t; }
__forceinline operator __mmask8() const { return v; }

__forceinline vboold(bool b) { v = b ? 0xff : 0x00; }
__forceinline vboold(int t)  { v = (__mmask8)t; }
__forceinline vboold(unsigned int t) { v = (__mmask8)t; }


__forceinline __m128i mask8() const {
#if defined(__AVX512BW__)
return _mm_movm_epi8(v);
#else
const __m512i f = _mm512_set1_epi64(0);
const __m512i t = _mm512_set1_epi64(-1);
const __m512i m =  _mm512_mask_or_epi64(f,v,t,t); 
return _mm512_cvtepi64_epi8(m);
#endif
}


__forceinline __m512i mask64() const { 
#if defined(__AVX512DQ__)
return _mm512_movm_epi64(v);
#else
const __m512i f = _mm512_set1_epi64(0);
const __m512i t = _mm512_set1_epi64(-1);
return _mm512_mask_or_epi64(f,v,t,t); 
#endif
}


__forceinline vboold(FalseTy) : v(0x00) {}
__forceinline vboold(TrueTy)  : v(0xff) {}
};


__forceinline vboold8 operator !(const vboold8& a) { return _mm512_knot(a); }


__forceinline vboold8 operator &(const vboold8& a, const vboold8& b) { return _mm512_kand(a, b); }
__forceinline vboold8 operator |(const vboold8& a, const vboold8& b) { return _mm512_kor(a, b); }
__forceinline vboold8 operator ^(const vboold8& a, const vboold8& b) { return _mm512_kxor(a, b); }

__forceinline vboold8 andn(const vboold8& a, const vboold8& b) { return _mm512_kandn(b, a); }


__forceinline vboold8& operator &=(vboold8& a, const vboold8& b) { return a = a & b; }
__forceinline vboold8& operator |=(vboold8& a, const vboold8& b) { return a = a | b; }
__forceinline vboold8& operator ^=(vboold8& a, const vboold8& b) { return a = a ^ b; }


__forceinline vboold8 operator !=(const vboold8& a, const vboold8& b) { return _mm512_kxor(a, b); }
__forceinline vboold8 operator ==(const vboold8& a, const vboold8& b) { return _mm512_kxnor(a, b); }

__forceinline vboold8 select(const vboold8& s, const vboold8& a, const vboold8& b) {
return _mm512_kor(_mm512_kand(s, a), _mm512_kandn(s, b));
}


__forceinline int all (const vboold8& a) { return a.v == 0xff; }
__forceinline int any (const vboold8& a) { return _mm512_kortestz(a, a) == 0; }
__forceinline int none(const vboold8& a) { return _mm512_kortestz(a, a) != 0; }

__forceinline int all (const vboold8& valid, const vboold8& b) { return all((!valid) | b); }
__forceinline int any (const vboold8& valid, const vboold8& b) { return any(valid & b); }
__forceinline int none(const vboold8& valid, const vboold8& b) { return none(valid & b); }

__forceinline size_t movemask(const vboold8& a) { return _mm512_kmov(a); }
__forceinline size_t popcnt  (const vboold8& a) { return __popcnt(a.v); }


__forceinline unsigned int toInt(const vboold8& a) { return mm512_mask2int(a); }


__forceinline bool get(const vboold8& a, size_t index) { assert(index < 8); return (toInt(a) >> index) & 1; }
__forceinline void set(vboold8& a, size_t index)       { assert(index < 8); a |= 1 << index; }
__forceinline void clear(vboold8& a, size_t index)     { assert(index < 8); a = andn(a, 1 << index); }


inline std::ostream& operator <<(std::ostream& cout, const vboold8& a)
{
cout << "<";
for (size_t i=0; i<8; i++) {
if ((a.v >> i) & 1) cout << "1"; else cout << "0";
}
return cout << ">";
}
}
