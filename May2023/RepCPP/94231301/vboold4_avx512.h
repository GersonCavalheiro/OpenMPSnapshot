
#pragma once

namespace embree
{

template<>
struct vboold<4>
{
typedef vboold4 Bool;
typedef vint4   Int;

enum { size = 4 }; 
__mmask8 v;        


__forceinline vboold() {}
__forceinline vboold(const vboold4& t) { v = t.v; }
__forceinline vboold4& operator =(const vboold4& f) { v = f.v; return *this; }

__forceinline vboold(const __mmask8 &t) { v = t; }
__forceinline operator __mmask8() const { return v; }

__forceinline vboold(bool b) { v = b ? 0xf : 0x0; }
__forceinline vboold(int t)  { v = (__mmask8)t; }
__forceinline vboold(unsigned int t) { v = (__mmask8)t; }


__forceinline __m128i mask8() const {
return _mm_movm_epi8(v);
}


__forceinline __m128i mask32() const {
return _mm_movm_epi32(v);
}


__forceinline __m256i mask64() const {
return _mm256_movm_epi64(v);
}


__forceinline vboold(FalseTy) : v(0x0) {}
__forceinline vboold(TrueTy)  : v(0xf) {}
};


__forceinline vboold4 operator !(const vboold4& a) { return _mm512_kandn(a, 0xf); }


__forceinline vboold4 operator &(const vboold4& a, const vboold4& b) { return _mm512_kand(a, b); }
__forceinline vboold4 operator |(const vboold4& a, const vboold4& b) { return _mm512_kor(a, b); }
__forceinline vboold4 operator ^(const vboold4& a, const vboold4& b) { return _mm512_kxor(a, b); }

__forceinline vboold4 andn(const vboold4& a, const vboold4& b) { return _mm512_kandn(b, a); }


__forceinline vboold4& operator &=(vboold4& a, const vboold4& b) { return a = a & b; }
__forceinline vboold4& operator |=(vboold4& a, const vboold4& b) { return a = a | b; }
__forceinline vboold4& operator ^=(vboold4& a, const vboold4& b) { return a = a ^ b; }


__forceinline vboold4 operator !=(const vboold4& a, const vboold4& b) { return _mm512_kxor(a, b); }
__forceinline vboold4 operator ==(const vboold4& a, const vboold4& b) { return _mm512_kand(_mm512_kxnor(a, b), 0xf); }

__forceinline vboold4 select(const vboold4& s, const vboold4& a, const vboold4& b) {
return _mm512_kor(_mm512_kand(s, a), _mm512_kandn(s, b));
}


__forceinline int all (const vboold4& a) { return a.v == 0xf; }
__forceinline int any (const vboold4& a) { return _mm512_kortestz(a, a) == 0; }
__forceinline int none(const vboold4& a) { return _mm512_kortestz(a, a) != 0; }

__forceinline int all (const vboold4& valid, const vboold4& b) { return all((!valid) | b); }
__forceinline int any (const vboold4& valid, const vboold4& b) { return any(valid & b); }
__forceinline int none(const vboold4& valid, const vboold4& b) { return none(valid & b); }

__forceinline size_t movemask(const vboold4& a) { return _mm512_kmov(a); }
__forceinline size_t popcnt  (const vboold4& a) { return __popcnt(a.v); }


__forceinline unsigned int toInt(const vboold4& a) { return mm512_mask2int(a); }


__forceinline bool get(const vboold4& a, size_t index) { assert(index < 4); return (toInt(a) >> index) & 1; }
__forceinline void set(vboold4& a, size_t index)       { assert(index < 4); a |= 1 << index; }
__forceinline void clear(vboold4& a, size_t index)     { assert(index < 4); a = andn(a, 1 << index); }


inline std::ostream& operator <<(std::ostream& cout, const vboold4& a)
{
cout << "<";
for (size_t i=0; i<4; i++) {
if ((a.v >> i) & 1) cout << "1"; else cout << "0";
}
return cout << ">";
}
}
