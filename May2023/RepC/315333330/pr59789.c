#pragma GCC push_options
#pragma GCC target("sse2")
typedef int __v4si __attribute__ ((__vector_size__ (16)));
typedef long long __m128i __attribute__ ((__vector_size__ (16), __may_alias__));
extern __inline __m128i __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_set_epi32 (int __q3, int __q2, int __q1, int __q0) 
{
return __extension__ (__m128i)(__v4si){ __q0, __q1, __q2, __q3 };
}
#pragma GCC pop_options
__m128i
f1(void)
{ 
return _mm_set_epi32 (0, 0, 0, 0); 
}
