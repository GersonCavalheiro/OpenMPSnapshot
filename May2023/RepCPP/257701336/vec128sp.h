

















#ifndef _H_VEC128SP
#define _H_VEC128SP

#include <altivec.h>
#include "veclib_types.h"





static const vector bool int expand_bit_to_word_masks[16] = {
#ifdef __LITTLE_ENDIAN__
{ 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u }, 
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u }, 
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, 
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, 
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, 
{ 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, 
{ 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u }, 
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u }, 
{ 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, 
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, 
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu }, 
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu }, 
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu }, 
{ 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu }, 
{ 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }, 
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }  
#elif __BIG_ENDIAN__
{ 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u }, 
{ 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, 
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, 
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu }, 
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, 
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu }, 
{ 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u }, 
{ 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }, 
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u }, 
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, 
{ 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, 
{ 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu }, 
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, 
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu }, 
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u }, 
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }  
#endif
};

static const vector unsigned char permute_highest_word_to_words_masks[16] = {

#ifdef __LITTLE_ENDIAN__
{ 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, 
{ 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, 
{ 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, 
{ 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, 
{ 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00 }, 
{ 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00 }, 
{ 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00 }, 
{ 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00 }, 
{ 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10 }, 
{ 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10 }, 
{ 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10 }, 
{ 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10 }, 
{ 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10 }, 
{ 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10 }, 
{ 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10 }, 
{ 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10 }  
#elif __BIG_ENDIAN__
{ 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, 
{ 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13 }, 
{ 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00 }, 
{ 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13 }, 
{ 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, 
{ 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13 }, 
{ 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00 }, 
{ 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13 }, 
{ 0x11,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, 
{ 0x11,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13 }, 
{ 0x11,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00 }, 
{ 0x11,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13 }, 
{ 0x11,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, 
{ 0x11,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13 }, 
{ 0x11,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00 }, 
{ 0x11,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13 }  
#endif
};





VECLIB_INLINE __m128 vec_load4sp (float const* address)
{
return (__m128) vec_ld (0, (vector float*)address);
}


VECLIB_INLINE __m128 vec_loaduandinsertupper2spinto4sp (__m128 a, __m64 const* from)
{
__m64_union from_union; from_union.as_m64 = *from;
__m128_union a_union; a_union.as_m128 = a;
__m128_union result_union;
#ifdef __LITTLE_ENDIAN__
result_union.as_m64[0] = a_union.as_m64[0];
result_union.as_float[2] = from_union.as_float[0];
result_union.as_float[3] = from_union.as_float[1];
#elif __BIG_ENDIAN__
result_union.as_m64[1] = a_union.as_m64[1];
result_union.as_float[0] = from_union.as_float[0];
result_union.as_float[1] = from_union.as_float[1];
#endif
return result_union.as_m128;
}


VECLIB_INLINE __m128 vec_loadandinsertupper2spinto4sp (__m128 a, __m64 const* from)
{
return vec_loaduandinsertupper2spinto4sp (a, from);
}


VECLIB_INLINE __m128 vec_loaduandinsertlower2spinto4sp (__m128 a, __m64 const* from)
{
__m64_union from_union; from_union.as_m64 = *from;
__m128_union a_union; a_union.as_m128 = a;
__m128_union result_union;
#ifdef __LITTLE_ENDIAN__
result_union.as_float[0] = from_union.as_float[0];
result_union.as_float[1] = from_union.as_float[1];
result_union.as_m64[1] = a_union.as_m64[1];
#elif __BIG_ENDIAN__
result_union.as_float[2] = from_union.as_float[0];
result_union.as_float[3] = from_union.as_float[1];
result_union.as_m64[0] = a_union.as_m64[0];
#endif
return result_union.as_m128;
}


VECLIB_INLINE __m128 vec_loadandinsertlower2spinto4sp (__m128 a, __m64 const* from)
{
return vec_loaduandinsertlower2spinto4sp (a, from);
}


VECLIB_INLINE __m128 vec_loaduzero1sp3z (float const* from)
{
__m128_union result_union;
result_union.as_m128 = (__m128) vec_splats ((float) 0.0);
#ifdef __LITTLE_ENDIAN__
result_union.as_float[0] = *from;
#elif __BIG_ENDIAN__
result_union.as_float[3] = *from;
#endif
return result_union.as_m128;
}


VECLIB_INLINE __m128 vec_loadzero1sp3zu (float const* from)
{
return vec_loaduzero1sp3z (from);
}


VECLIB_INLINE __m128 vec_loadu4sp (float const* from)
{
#if __LITTLE_ENDIAN__

return (__m128) *(vector float*) from;
#elif __BIG_ENDIAN__
__m128 result;
__m128 temp_ld0 = vec_ld (0, from);
__m128 temp_ld16 = vec_ld (16, from);
vector unsigned char permute_selector = vec_lvsl (0, (float *)from);
result = (__m128) vec_perm (temp_ld0, temp_ld16, permute_selector);
return result;
#endif
}


VECLIB_INLINE __m128 vec_load4spu (float const* from)
{
return vec_loadu4sp (from);
}


VECLIB_INLINE __m128 vec_loadreverse4sp (float const* from)
{
__m128 result = vec_ld (0, from);
vector unsigned char permute_vector = {
0x1C, 0x1D, 0x1E, 0x1F,  0x18, 0x19, 0x1A, 0x1B,  0x14, 0x15, 0x16, 0x17,  0x10, 0x11, 0x12, 0x13
};
result = vec_perm (result, result, permute_vector);
return result;
}


VECLIB_INLINE __m128 vec_loadsplat4sp (float const* from)
{
return (__m128) vec_splats (*from);
}





VECLIB_INLINE __m128 vec_zero4sp (void)
{
return (__m128) vec_splats ((float) 0);
}


VECLIB_INLINE __m128 vec_splat4sp (float scalar)
{
#ifdef __ibmxl__
return (__m128) vec_splats (scalar);
#elif __GNUC__


__m128_union t;
t.as_float[0] = scalar;
t.as_float[1] = scalar;
t.as_float[2] = scalar;
t.as_float[3] = scalar;
return t.as_m128;
#else
#error Compiler not supported yet.
#endif
}


VECLIB_INLINE __m128 vec_set4sp (float f3, float f2, float f1, float f0)
{
__m128_union t;
#ifdef __LITTLE_ENDIAN__
t.as_float[0] = f0;
t.as_float[1] = f1;
t.as_float[2] = f2;
t.as_float[3] = f3;
#elif __BIG_ENDIAN__
t.as_float[0] = f3;
t.as_float[1] = f2;
t.as_float[2] = f1;
t.as_float[3] = f0;
#endif
return t.as_m128;
}


VECLIB_INLINE __m128 vec_setreverse4sp (float f3, float f2, float f1, float f0)
{
return (vec_set4sp (f0, f1, f2, f3));
}


VECLIB_INLINE __m128 vec_set1sp3z (float scalar)
{
__m128_union t;
#ifdef __BIG_ENDIAN__
t.as_float[3] = scalar;
t.as_float[2] = 0;
t.as_float[1] = 0;
t.as_float[0] = 0;
#elif __LITTLE_ENDIAN__
t.as_float[0] = scalar;
t.as_float[1] = 0;
t.as_float[2] = 0;
t.as_float[3] = 0;
#endif
return t.as_m128;
}


VECLIB_INLINE __m128 vec_setbot1spscalar4sp (float scalar)
{
return vec_set1sp3z (scalar);
}




VECLIB_INLINE void vec_store4sp (float* address, __m128 v)
{
vec_st (v, 0, address);
}


VECLIB_INLINE void vec_storeupper2spof4sp (__m64* to, __m128 from)
{
__m128_union from_union; from_union.as_m128 = from;
#ifdef __LITTLE_ENDIAN__
*to = from_union.as_m64[1];
#elif __BIG_ENDIAN__
*to = from_union.as_m64[0];
#endif
}


VECLIB_INLINE void vec_storelower2spof4sp (__m64* to, __m128 from)
{
__m128_union from_union; from_union.as_m128 = from;
#ifdef __LITTLE_ENDIAN__
*to = from_union.as_m64[0];
#elif __BIG_ENDIAN__
*to = from_union.as_m64[1];
#endif
}




VECLIB_INLINE void vec_storeu4spto1sp (float* address, __m128 v)
{
__m128_union t;
t.as_vector_float = v;
unsigned int element_number;
#ifdef __LITTLE_ENDIAN__
#ifdef __LOWER_MEANS_SCALAR_NOT_LOWER__
element_number = 3;
#else

element_number = 0;
#endif
#elif __BIG_ENDIAN__
#ifdef __LOWER_MEANS_SCALAR_NOT_LOWER__
element_number = 0;
#else

element_number = 3;
#endif
#endif
*address = t.as_float[element_number];
}




VECLIB_INLINE void vec_store4spto1sp (float* address, __m128 v)
{
vec_storeu4spto1sp (address, v);
}


VECLIB_INLINE void vec_storereverse4sp (float* address, __m128 data)
{
__m128_union t;
float temp;
t.as_m128 = data;

temp = t.as_float[0];
t.as_float[0] = t.as_float[3];
t.as_float[3] = temp;

temp = t.as_float[1];
t.as_float[1] = t.as_float[2];
t.as_float[2] = temp;

vec_st (t.as_m128, 0, address);
}



VECLIB_INLINE float vec_store1spof4sp (__m128 from)
{
__m128_union from_union; from_union.as_m128 = from;
float result;
#ifdef __LITTLE_ENDIAN__
result = from_union.as_float[0];
#elif __BIG_ENDIAN__
result = from_union.as_float[3];
#endif
return result;
}


VECLIB_INLINE void vec_storeu4sp (float* to, __m128 from)
{
#if __LITTLE_ENDIAN__

*(vector float*) to = (vector float) from;
#elif __BIG_ENDIAN__

vector signed char all_one = vec_splat_s8( -1 );
vector signed char all_zero = vec_splat_s8( 0 );

vector unsigned char permute_vector = vec_lvsr (0, (unsigned char *) to);

vector unsigned char select_vector = vec_perm ((vector unsigned char) all_zero, (vector unsigned char) all_one, permute_vector);



vector unsigned char low = vec_ld (0, (unsigned char *) to);
vector unsigned char high = vec_ld (16, (unsigned char *) to);

vector unsigned char temp_low = vec_perm (low, (vector unsigned char) from, permute_vector);
low = vec_sel (low, temp_low, select_vector);
high = vec_perm ((vector unsigned char) from, high, permute_vector);

vec_st (low, 0, (unsigned char *) to);
vec_st (high, 16, (unsigned char *) to);
#endif
}


VECLIB_INLINE void vec_store4spstream (float* to, __m128 from)
{
vec_st (from, 0, to);
#ifdef __ibmxl__

__dcbt ((void *) to);
#endif
}


VECLIB_INLINE void vec_storesplat1nto4sp (float* to, __m128 from)
{
vector unsigned char permute_vector = {
#ifdef __LITTLE_ENDIAN__
0x10, 0x11, 0x12, 0x13,  0x10, 0x11, 0x12, 0x13,  0x10, 0x11, 0x12, 0x13,  0x10, 0x11, 0x12, 0x13
#elif __BIG_ENDIAN__
0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F
#endif
};
from = (__m128) vec_perm (from, from, permute_vector);
vec_st (from, 0, to);
}





VECLIB_INLINE __m128 vec_insert1spintolower4sp (__m128 into, __m128 from)
{
static const vector unsigned int permute_selector =
#ifdef __LITTLE_ENDIAN__
{ 0x13121110u, 0x07060504u, 0x0B0A0908u, 0x0F0E0D0Cu };
#elif __BIG_ENDIAN__
{ 0x00010203u, 0x04050607u, 0x08090A0Bu, 0x1C1D1E1Fu };
#endif
return (__m128) vec_perm ((vector float) into, (vector float) from, (vector unsigned char) permute_selector);
}


VECLIB_INLINE __m128 vec_insertlowerto4sp (__m128 into, __m128 from)
{
return vec_insert1spintolower4sp (into, from);
}



VECLIB_INLINE __m128 vec_insert4sp (__m128 into, __m128 from, const intlit8 control)
{
int extract_selector = (control >> 6) & 0x3;
int insert_selector = (control >> 4) & 0x3;
int zero_selector = control & 0xF;
static const vector unsigned char extract_selectors[4] = {
#ifdef __LITTLE_ENDIAN__
{ 0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03 }, 
{ 0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07 }, 
{ 0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B }, 
{ 0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F }, 
#elif __BIG_ENDIAN__
{ 0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F }, 
{ 0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B }, 
{ 0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07 }, 
{ 0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03 }, 
#endif
};
vector float extracted = vec_perm (from, from, extract_selectors[extract_selector]);
static const vector unsigned int insert_selectors[4] = {
#ifdef __LITTLE_ENDIAN__
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u }, 
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, 
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, 
{ 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }  
#elif __BIG_ENDIAN__
{ 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, 
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, 
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, 
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u }, 
#endif
};
vector float inserted = vec_sel (into, extracted, insert_selectors[insert_selector]);
static const vector unsigned int zero_selectors [16] = {

#ifdef __LITTLE_ENDIAN__


{ 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u },  
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u },  
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u },  
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u },  
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u },  
{ 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u },  
{ 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u },  
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u },  
{ 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu },  
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu },  
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu },  
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu },  
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu },  
{ 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu },  
{ 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },  
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },  
#elif __BIG_ENDIAN__


{ 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u },  
{ 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu },  
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u },  
{ 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu },  
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u },  
{ 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu },  
{ 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u },  
{ 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },  
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u },  
{ 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu },  
{ 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u },  
{ 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu },  
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u },  
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu },  
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u },  
{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },  
#endif
};
return (__m128) vec_sel (inserted, vec_splats (0.0f), zero_selectors[zero_selector]);
}





VECLIB_INLINE int vec_extractupperbit4sp (__m128 v)
{
__m128_all_union t;
t.as_m128 = v;
int result = 0;
#ifdef __LITTLE_ENDIAN__
result |= ((t.as_float[3] < 0) ? 1:0 ) << 3;
result |= ((t.as_float[2] < 0) ? 1:0 ) << 2;
result |= ((t.as_float[1] < 0) ? 1:0 ) << 1;
result |= ((t.as_float[0] < 0) ? 1:0 ) ;
#elif __BIG_ENDIAN__
result |= ((t.as_float[0] < 0) ? 1:0 ) << 3;
result |= ((t.as_float[1] < 0) ? 1:0 ) << 2;
result |= ((t.as_float[2] < 0) ? 1:0 ) << 1;
result |= ((t.as_float[3] < 0) ? 1:0 ) ;
#endif
return result;
}


VECLIB_INLINE int vec_extract4sp (__m128 from, const intlit2 element_number) {
__m128_union from_union;  from_union.as_m128 = from;
return from_union.as_hex[element_number & 0x3];
}


VECLIB_INLINE __m128 vec_extract4spfrom8sp (__m256 from, const intlit1 element_number)
{
if ((element_number & 1) == 0)
{
#ifdef __LITTLE_ENDIAN__
return from.m128_0;
#elif __BIG_ENDIAN__
return from.m128_1;
#endif
} else
{
#ifdef __LITTLE_ENDIAN__
return from.m128_1;
#elif __BIG_ENDIAN__
return from.m128_0;
#endif
}
}





VECLIB_INLINE __m128 vec_convert2wto4sp (__m64 lower, __m64 upper)
{
__m64_union lower_union;
__m64_union upper_union;
lower_union.as_m64 = lower;
upper_union.as_m64 = upper;
lower_union.as_float[0] = (float) lower_union.as_int[0];
lower_union.as_float[1] = (float) lower_union.as_int[1];
upper_union.as_float[0] = (float) upper_union.as_int[0];
upper_union.as_float[1] = (float) upper_union.as_int[1];
__m128_union result;
result.as_m64[0] = lower_union.as_m64;
result.as_m64[1] = upper_union.as_m64;
return result.as_m128;
}


VECLIB_INLINE __m128 vec_convert22wto4sp (__m64 lower, __m64 upper)
{
return vec_convert2wto4sp (lower, upper);
}


VECLIB_INLINE __m128 vec_convert4swto4sp (__m128i v)
{
__m128i_union t;
t.as_m128i=v;
return (__m128) vec_ctf (t.as_vector_signed_int, 0u);
}


VECLIB_INLINE __m128 vec_convert1swtolower1of4sp (__m128 v, int a)
{  
__m128_union result;
result.as_m128 = v;
result.as_float[0] = (float) a;
return result.as_m128;
}


VECLIB_INLINE __m128 vec_convert2swtolower2of4sp (__m128 v , __m64 lower)
{
__m64_union lower_union;
lower_union.as_m64 = lower;
lower_union.as_float[0] = (float) lower_union.as_int[0];
lower_union.as_float[1] = (float) lower_union.as_int[1];
__m128_union result;
result.as_m128 = v;
result.as_m64[0] = lower_union.as_m64;
return result.as_m128;
}


VECLIB_INLINE __m128 vec_convert1sdtolower1of4sp (__m128 v , long long int l)
{
__m128_union result;
result.as_m128 = v;
result.as_float[0] = (float) l;
return result.as_m128;
}


VECLIB_INLINE __m128 vec_convertlower4of8sbto4sp(__m64 lower)
{
__m64_union lower_union;
lower_union.as_m64 = lower;
__m128_union result;
result.as_float[0] = (float) lower_union.as_signed_char[0];
result.as_float[1] = (float) lower_union.as_signed_char[1];
result.as_float[2] = (float) lower_union.as_signed_char[2];
result.as_float[3] = (float) lower_union.as_signed_char[3];
return result.as_m128;
}


VECLIB_INLINE __m128 vec_convertlower4of8ubto4sp(__m64 lower)
{
__m64_all_union lower_union;
lower_union.as_m64 = lower;
__m128_union result;
result.as_float[0] = (float) lower_union.as_unsigned_char[0];
result.as_float[1] = (float) lower_union.as_unsigned_char[1];
result.as_float[2] = (float) lower_union.as_unsigned_char[2];
result.as_float[3] = (float) lower_union.as_unsigned_char[3];
return result.as_m128;
}


VECLIB_INLINE __m128 vec_convert4shto4sp (__m64 lower)
{
__m64_union lower_union;
lower_union.as_m64 = lower;
__m128_union result;
result.as_float[0] = (float) lower_union.as_short[0];
result.as_float[1] = (float) lower_union.as_short[1];
result.as_float[2] = (float) lower_union.as_short[2];
result.as_float[3] = (float) lower_union.as_short[3];
return result.as_m128;
}


VECLIB_INLINE __m128 vec_convert4uhto4sp (__m64 lower)
{
__m64_all_union lower_union;
lower_union.as_m64 = lower;
__m128_union result;
result.as_float[0] = (float) lower_union.as_unsigned_short[0];
result.as_float[1] = (float) lower_union.as_unsigned_short[1];
result.as_float[2] = (float) lower_union.as_unsigned_short[2];
result.as_float[3] = (float) lower_union.as_unsigned_short[3];
return result.as_m128;
}








VECLIB_INLINE __m128 vec_partialhorizontal2sp (__m128 left, __m128 right)
{






#ifdef __LITTLE_ENDIAN__
static vector unsigned char addend_1_permute_mask = (vector unsigned char)
{ 0x07,0x06,0x05,0x04, 0x0F,0x0E,0x0D,0x0C, 0x17,0x16,0x15,0x14, 0x1F,0x1E,0x1D,0x1C };
static vector unsigned char addend_2_permute_mask = (vector unsigned char)
{ 0x03,0x02,0x01,0x00, 0x0B,0x0A,0x09,0x08, 0x1B,0x1A,0x19,0x18, 0x13,0x12,0x11,0x10 };
#elif __BIG_ENDIAN__
static vector unsigned char addend_1_permute_mask = (vector unsigned char)
{ 0x04,0x05,0x06,0x07, 0x0C,0x0D,0x0E,0x0F, 0x14,0x15,0x16,0x17, 0x1C,0x1D,0x1E,0x1F };
static vector unsigned char addend_2_permute_mask = (vector unsigned char)
{ 0x00,0x01,0x02,0x03, 0x08,0x09,0x0A,0x0B, 0x10,0x11,0x12,0x13, 0x18,0x19,0x1A,0x1B };
#endif
vector float addend_1 = vec_perm (left, right, addend_1_permute_mask);
vector float addend_2 = vec_perm (left, right, addend_2_permute_mask);
return (__m128) vec_add (addend_1, addend_2);
}


VECLIB_INLINE __m128 vec_partialhorizontal22sp (__m128 left, __m128 right)
{
return vec_partialhorizontal2sp (left, right);
}


VECLIB_INLINE __m128 vec_add4sp (__m128 left, __m128 right)
{
return (__m128) vec_add ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_subtract4sp (__m128 left, __m128 right)
{
return (__m128) vec_sub ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_multiply4sp (__m128 left, __m128 right)
{
return (__m128) vec_mul ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_divide4sp (__m128 left, __m128 right)
{
return (__m128) vec_div ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_max4sp (__m128 left, __m128 right)
{
return (__m128) vec_max ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_min4sp (__m128 left, __m128 right)
{
return (__m128) vec_min ((vector float) left, (vector float) right);
}




VECLIB_INLINE __m128 vec_log24sp (__m128 v)

{
return vec_loge (v);
}



#ifdef __ibmxl__
VECLIB_NOINLINE
#else
VECLIB_INLINE
#endif
__m128 vec_dotproduct4sp (__m128 left, __m128 right, const intlit8 multiply_and_result_masks)
{
#ifdef __ibmxl__
#pragma option_override (vec_dotproduct4sp, "opt(level,0)")
#endif
__m128_union left_union; left_union.as_m128 = left;
__m128_union right_union; right_union.as_m128 = right;
__m128_all_union result_union;
static const vector unsigned char all64s =
{ 64,64,64,64, 64,64,64,64, 64,64,64,64, 64,64,64,64 };
static const vector unsigned char all32s =
{ 32,32,32,32, 32,32,32,32, 32,32,32,32, 32,32,32,32 };

unsigned int multiply_mask = (multiply_and_result_masks & 0xF0) >> 4;
unsigned int result_mask   =  multiply_and_result_masks & 0x0F;
__m128_all_union multiply_element_mask;
__m128_all_union result_element_mask;
multiply_element_mask.as_vector_bool_int = (vector bool int) expand_bit_to_word_masks[multiply_mask];
result_element_mask.as_vector_unsigned_int = (vector unsigned int) permute_highest_word_to_words_masks[result_mask];

__m128_all_union masked_left;
masked_left.as_m128  = (__m128) vec_and (left, multiply_element_mask.as_vector_bool_int);
__m128_all_union masked_right;
masked_right.as_m128 = (__m128) vec_and (right, multiply_element_mask.as_vector_bool_int);
__m128_all_union products;
products.as_m128 = (__m128) vec_madd (masked_left .as_vector_float,
masked_right.as_vector_float,
vec_splats (0.f));

__m128_all_union t;
#ifdef USE_VEC_SLD
t.as_m128 = (__m128) vec_add (products.as_vector_float,
vec_sld (products.as_vector_float, products.as_vector_float, 64/8));
#else
t.as_m128 = (__m128) vec_add (products.as_vector_float,
vec_slo (products.as_vector_float, all64s));
#endif
__m128_all_union s;
#ifdef USE_VEC_SLD
s.as_m128 = (__m128) vec_add (t.as_vector_float,
vec_sld (t.as_vector_float, t.as_vector_float, 32/8));
#else
s.as_m128 = (__m128) vec_add (t.as_vector_float,
vec_slo (t.as_vector_float, all32s));
#endif

result_union.as_vector_float = vec_perm (vec_splats (0.f), s.as_vector_float,
result_element_mask.as_vector_unsigned_char);
return result_union.as_m128;
}


VECLIB_INLINE __m128 vec_add1spof4sp (__m128 left, __m128 right)
{
vector float all_zero = vec_splats ((float) 0.0);
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float extracted_right = vec_sel (all_zero, (vector float) right, select_vector);
vector float result = vec_add (extracted_right, (vector float) left); 
return (__m128) result;
}


VECLIB_INLINE __m128 vec_squareroot4sp (__m128 v)
{
return (__m128) vec_sqrt (v);
}


VECLIB_INLINE __m128 vec_subtract1spof4sp (__m128 left, __m128 right)
{
vector float all_zero = vec_splats ((float) 0.0);
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float extracted_right = vec_sel (all_zero, (vector float) right, select_vector);
vector float result = vec_sub ((vector float) left, extracted_right); 
return (__m128) result;
}



VECLIB_INLINE __m128 vec_multiply1spof4sp (__m128 left, __m128 right)
{
vector float all_ones = vec_splats ((float) 1.0);
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float extracted_right = vec_sel (all_ones, (vector float) right, select_vector);
vector float result = vec_mul ((vector float) left, extracted_right);
return (__m128) result;
}


#ifdef VECLIB_VSX
VECLIB_INLINE __m128 vec_divide1spof4sp (__m128 left, __m128 right)
{
vector float all_ones = vec_splats ((float) 1.0);
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float extracted_right = vec_sel (all_ones, (vector float) right, select_vector);
vector float result = vec_div ((vector float) left, extracted_right);
return (__m128) result;
}
#endif



VECLIB_INLINE __m128 vec_squareroot1spof4sp (__m128 v)
{
vector float all_zero = vec_splats ((float) 0.0);
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float first_res = vec_sqrt ((vector float) v);
vector bool int is_zero_mask = vec_cmpeq (all_zero, (vector float) v);
vector float inter_res = (__m128) vec_sel (first_res, all_zero, is_zero_mask);
vector float result = vec_sel ((vector float) v, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_max1spof4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_max ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_min1spof4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_min ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}



VECLIB_INLINE __m128 vec_reciprocalestimate4sp (__m128 v)
{
return (__m128) vec_re ((vector float) v);
}



VECLIB_INLINE __m128 vec_reciprocalestimate1spof4sp (__m128 v)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_re ((vector float) v);
vector float result = vec_sel ((vector float) v, inter_res, select_vector);
return (__m128) result;
}



VECLIB_INLINE __m128 vec_reciprocalsquarerootestimate4sp (__m128 v)
{
return (__m128) vec_rsqrte ((vector float) v);
}




VECLIB_INLINE __m128 vec_reciprocalsquarerootestimate1spof4sp (__m128 v)
{

vector float all_zero = vec_splats ((float) 1.0);
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float extracted_right = vec_sel (all_zero, (vector float) v, select_vector);
vector float inter_res = vec_rsqrte ((vector float) extracted_right);
vector float result = vec_sel ((vector float) v, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_Floor4sp (__m128 v) {
return vec_floor (v);
}




VECLIB_INLINE __m128 vec_bitwiseor4sp (__m128 left, __m128 right)
{
return (__m128) vec_or (left, right);
}


VECLIB_INLINE __m128 vec_bitwiseand4sp (__m128 left, __m128 right)
{
return (__m128) vec_and (left, right);
}


VECLIB_INLINE __m128 vec_bitand4sp (__m128 left, __m128 right)
{
return vec_bitwiseand4sp (left, right);
}


VECLIB_INLINE __m128 vec_bitwiseandnotleft4sp (__m128 left, __m128 right)
{
return (__m128) vec_andc (right, left);
}


VECLIB_INLINE __m128 vec_bitandnotleft4sp (__m128 left, __m128 right)
{
return vec_bitwiseandnotleft4sp (left, right);
}


VECLIB_INLINE __m128 vec_bitwisexor4sp (__m128 left, __m128 right)
{
return (__m128) vec_xor ((vector float) left, (vector float) right);
}






VECLIB_INLINE __m128 vec_shufflepermute4sp (__m128 left, __m128 right, unsigned int element_selectors)
{
unsigned long element_selector_10 =  element_selectors       & 0x03;
unsigned long element_selector_32 = (element_selectors >> 2) & 0x03;
unsigned long element_selector_54 = (element_selectors >> 4) & 0x03;
unsigned long element_selector_76 = (element_selectors >> 6) & 0x03;
#ifdef __LITTLE_ENDIAN__
const static unsigned int permute_selectors_from_left_operand  [4] = { 0x03020100u, 0x07060504u, 0x0B0A0908u, 0x0F0E0D0Cu };
const static unsigned int permute_selectors_from_right_operand [4] = { 0x13121110u, 0x17161514u, 0x1B1A1918u, 0x1F1E1D1Cu };
#elif __BIG_ENDIAN__
const static unsigned int permute_selectors_from_left_operand  [4] = { 0x00010203u, 0x04050607u, 0x08090A0Bu, 0x0C0D0E0Fu };
const static unsigned int permute_selectors_from_right_operand [4] = { 0x10111213u, 0x14151617u, 0x18191A1Bu, 0x1C1D1E1Fu };
#endif
__m128i_union permute_selectors;
#ifdef __LITTLE_ENDIAN__
permute_selectors.as_int[0] = permute_selectors_from_left_operand [element_selector_10];
permute_selectors.as_int[1] = permute_selectors_from_left_operand [element_selector_32];
permute_selectors.as_int[2] = permute_selectors_from_right_operand[element_selector_54];
permute_selectors.as_int[3] = permute_selectors_from_right_operand[element_selector_76];
#elif __BIG_ENDIAN__
permute_selectors.as_int[3] = permute_selectors_from_left_operand [element_selector_10];
permute_selectors.as_int[2] = permute_selectors_from_left_operand [element_selector_32];
permute_selectors.as_int[1] = permute_selectors_from_right_operand[element_selector_54];
permute_selectors.as_int[0] = permute_selectors_from_right_operand[element_selector_76];
#endif
return (vector float) vec_perm ((vector unsigned char) left, (vector unsigned char) right,
permute_selectors.as_vector_unsigned_char);
}




VECLIB_INLINE __m128 vec_shufflepermute44sp (__m128 left, __m128 right, unsigned int element_selectors)
{
return vec_shufflepermute4sp (left, right, element_selectors);
}


VECLIB_INLINE __m128 vec_blendpermute4sp (__m128 left, __m128 right, const intlit4 mask)
{
static const vector unsigned char permute_selector[16] = {

#ifdef __LITTLE_ENDIAN__
{ 0x0F,0x0E,0x0D,0x0C, 0x0B,0x0A,0x09,0x08, 0x07,0x06,0x05,0x04, 0x03,0x02,0x01,0x00 }, 
{ 0x1F,0x1E,0x1D,0x1C, 0x0B,0x0A,0x09,0x08, 0x07,0x06,0x05,0x04, 0x03,0x02,0x01,0x00 }, 
{ 0x0F,0x0E,0x0D,0x0C, 0x1B,0x1A,0x19,0x18, 0x07,0x06,0x05,0x04, 0x03,0x02,0x01,0x00 }, 
{ 0x1F,0x1E,0x1D,0x1C, 0x1B,0x1A,0x19,0x18, 0x07,0x06,0x05,0x04, 0x03,0x02,0x01,0x00 }, 
{ 0x0F,0x0E,0x0D,0x0C, 0x0B,0x0A,0x09,0x08, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }, 
{ 0x1F,0x1E,0x1D,0x1C, 0x0B,0x0A,0x09,0x08, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }, 
{ 0x0F,0x0E,0x0D,0x0C, 0x1B,0x1A,0x19,0x18, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }, 
{ 0x1F,0x1E,0x1D,0x1C, 0x1B,0x1A,0x19,0x18, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }, 
{ 0x0F,0x0E,0x0D,0x0C, 0x0B,0x0A,0x09,0x08, 0x07,0x06,0x05,0x04, 0x13,0x02,0x01,0x00 }, 
{ 0x1F,0x1E,0x1D,0x1C, 0x0B,0x0A,0x09,0x08, 0x07,0x06,0x05,0x04, 0x13,0x02,0x01,0x10 }, 
{ 0x0F,0x0E,0x0D,0x0C, 0x1B,0x1A,0x19,0x18, 0x07,0x06,0x05,0x04, 0x13,0x02,0x11,0x00 }, 
{ 0x1F,0x1E,0x1D,0x1C, 0x1B,0x1A,0x19,0x18, 0x07,0x06,0x05,0x04, 0x13,0x02,0x11,0x01 }, 
{ 0x0F,0x0E,0x0D,0x0C, 0x0B,0x0A,0x09,0x08, 0x17,0x16,0x15,0x14, 0x13,0x12,0x01,0x00 }, 
{ 0x1F,0x1E,0x1D,0x1C, 0x0B,0x0A,0x09,0x08, 0x17,0x16,0x15,0x14, 0x13,0x12,0x01,0x10 }, 
{ 0x0F,0x0E,0x0D,0x0C, 0x1B,0x1A,0x19,0x18, 0x17,0x16,0x15,0x14, 0x13,0x12,0x11,0x00 }, 
{ 0x1F,0x1E,0x1D,0x1C, 0x1B,0x1A,0x19,0x18, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }  
#elif __BIG_ENDIAN__
{ 0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B, 0x0C,0x0D,0x0E,0x0F }, 
{ 0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B, 0x1C,0x1D,0x1E,0x1F }, 
{ 0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B, 0x0C,0x0D,0x0E,0x0F }, 
{ 0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B, 0x1C,0x1D,0x1E,0x1F }, 
{ 0x00,0x01,0x02,0x03, 0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B, 0x0C,0x0D,0x0E,0x0F }, 
{ 0x00,0x01,0x02,0x03, 0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B, 0x1C,0x1D,0x1E,0x1F }, 
{ 0x00,0x01,0x02,0x03, 0x14,0x15,0x16,0x17, 0x18,0x19,0x1A,0x1B, 0x0C,0x0D,0x0E,0x0F }, 
{ 0x00,0x01,0x02,0x03, 0x14,0x15,0x16,0x17, 0x18,0x19,0x1A,0x1B, 0x1C,0x1D,0x1E,0x1F }, 
{ 0x10,0x11,0x12,0x13, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B, 0x0C,0x0D,0x0E,0x0F }, 
{ 0x10,0x11,0x12,0x13, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B, 0x1C,0x1D,0x1E,0x1F }, 
{ 0x10,0x11,0x12,0x13, 0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B, 0x0C,0x0D,0x0E,0x0F }, 
{ 0x10,0x11,0x12,0x13, 0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B, 0x1C,0x1D,0x1E,0x1F }, 
{ 0x10,0x11,0x12,0x13, 0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B, 0x0C,0x0D,0x0E,0x0F }, 
{ 0x10,0x11,0x12,0x13, 0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B, 0x1C,0x1D,0x1E,0x1F }, 
{ 0x10,0x11,0x12,0x13, 0x14,0x15,0x16,0x17, 0x18,0x19,0x1A,0x1B, 0x0C,0x0D,0x0E,0x0F }, 
{ 0x10,0x11,0x12,0x13, 0x14,0x15,0x16,0x17, 0x18,0x19,0x1A,0x1B, 0x1C,0x1D,0x1E,0x1F }  
#endif
};
return (__m128) vec_perm ((vector float) left, (vector float) right, permute_selector[mask & 0xF]);
}


VECLIB_INLINE __m128 vec_blendpermute44sp (__m128 left, __m128 right, const intlit4 mask)
{
return vec_blendpermute4sp (left, right, mask);
}


VECLIB_INLINE __m128 vec_permutevr4sp (__m128 left, __m128 right, __m128 mask)
{

vector bool int select_mask = vec_cmplt ((vector signed int) mask, vec_splats (0));  
return (__m128) vec_sel (left, right, select_mask);
}


VECLIB_INLINE __m128 vec_permutevr44sp (__m128 left, __m128 right, __m128 mask)
{
return vec_permutevr4sp (left, right, mask);
}


VECLIB_INLINE __m128 vec_extractupper2spinsertlower2spof4sp (__m128 upper_to_upper, __m128 upper_to_lower)
{
vector unsigned char permute_selector = {
#ifdef __LITTLE_ENDIAN__
0x18, 0x19, 0x1A, 0x1B,  0x1C, 0x1D, 0x1E, 0x1F,  0x08, 0x09, 0xA, 0xB,    0x0C, 0x0D, 0x0E, 0x0F
#elif __BIG_ENDIAN__
0x00, 0x01, 0x02, 0x03,  0x04, 0x05, 0x06, 0x07,  0x10, 0x11, 0x12, 0x13,  0x14, 0x15, 0x16, 0x17
#endif
};
return vec_perm ((vector float) upper_to_upper, (vector float) upper_to_lower, permute_selector);
}


VECLIB_INLINE __m128  vec_extractoddsptoevensp (__m128 a) {
vector unsigned char permute_selector = {
#ifdef __LITTLE_ENDIAN__
0x04, 0x05, 0x6, 0x7,  0x04, 0x05, 0x6, 0x7,   0x0C, 0x0D, 0x0E, 0x0F,   0x0C, 0x0D, 0x0E, 0x0F
#elif __BIG_ENDIAN__
0x00, 0x01, 0x2, 0x3,   0x00, 0x01, 0x2, 0x3,   0x08, 0x09, 0x0A, 0x0B,   0x08, 0x09, 0x0A, 0x0B
#endif
};
return vec_perm(a, a, permute_selector);
}


VECLIB_INLINE __m128 vec_extractevensptooddsp (__m128 a) {
vector unsigned char permute_selector = {
#ifdef __LITTLE_ENDIAN__
0x00, 0x01, 0x2, 0x3,   0x00, 0x01, 0x2, 0x3,   0x08, 0x09, 0x0A, 0x0B,   0x08, 0x09, 0x0A, 0x0B
#elif __BIG_ENDIAN__
0x04, 0x05, 0x6, 0x7,  0x04, 0x05, 0x6, 0x7,   0x0C, 0x0D, 0x0E, 0x0F,   0x0C, 0x0D, 0x0E, 0x0F
#endif
};
return vec_perm(a, a, permute_selector);
}


VECLIB_INLINE __m128 vec_extractlower2spinsertupper2spof4sp (__m128 lower_to_lower, __m128 lower_to_upper)
{
vector unsigned char permute_selector = {
#ifdef __LITTLE_ENDIAN__
0x10, 0x11, 0x12, 0x13,  0x14, 0x15, 0x16, 0x17,  0x00, 0x01, 0x02, 0x03,  0x04, 0x05, 0x06, 0x07
#elif __BIG_ENDIAN__
0x08, 0x09, 0x0A, 0x0B,  0x0C, 0x0D, 0x0E, 0x0F,  0x18, 0x19, 0x1A, 0x1B,  0x1C, 0x1D, 0x1E, 0x1F
#endif
};
return vec_perm ((vector float) lower_to_upper, (vector float) lower_to_lower, permute_selector);
}




VECLIB_INLINE __m128 vec_comparenotnans4sp (__m128 left, __m128 right);




VECLIB_INLINE __m128 vec_compareeq4sp (__m128 left, __m128 right)
{
return (__m128) vec_cmpeq ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_compareeq_4sp (__m128 left, __m128 right)
{
return vec_compareeq4sp (right, left);
}


VECLIB_INLINE __m128 vec_comparene4sp (__m128 left, __m128 right)
{
__m128_union leftx;
leftx.as_m128 = vec_compareeq4sp (left, right);
__m128_union rightx;
rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);
return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}


VECLIB_INLINE __m128 vec_compareeq1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_compareeq4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_comparene1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparene4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE int vec_compareeq1of4sptobool (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_compareeq4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);


return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}


VECLIB_INLINE int vec_comparene1of4sptobool (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparene4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}


VECLIB_INLINE int vec_compareeq1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_compareeq4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}


VECLIB_INLINE int vec_comparene1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparene4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}




VECLIB_INLINE __m128 vec_comparelt4sp (__m128 left, __m128 right)
{
return (__m128) vec_cmplt ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_comparelt_4sp (__m128 left, __m128 right)
{
return vec_comparelt4sp (left, right);
}


VECLIB_INLINE __m128 vec_comparenotlt4sp (__m128 left, __m128 right)
{
__m128_union leftx;
leftx.as_m128 = vec_comparelt4sp (left, right); 
__m128_union rightx;
rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);

return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}


VECLIB_INLINE __m128 vec_comparelt1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparelt4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_comparenotlt1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparenotlt4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE int vec_comparelt1of4sptobool (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparelt4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}


VECLIB_INLINE int vec_comparelt1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparelt4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}




VECLIB_INLINE __m128 vec_comparele4sp (__m128 left, __m128 right)
{
return (__m128) vec_cmple ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_comparele_4sp (__m128 left, __m128 right)
{
return vec_comparele4sp (left, right);
}


VECLIB_INLINE __m128 vec_comparenotle4sp (__m128 left, __m128 right)
{
__m128_union leftx;
leftx.as_m128 = vec_comparele4sp (left, right); 
__m128_union rightx;
rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);
return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}


VECLIB_INLINE __m128 vec_comparele1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparele4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_comparenotle1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparenotle4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE int vec_comparele1of4sptobool (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparele4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}


VECLIB_INLINE int vec_comparele1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparele4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}




VECLIB_INLINE __m128 vec_comparegt4sp (__m128 left, __m128 right)
{
return (__m128) vec_cmpgt ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_comparegt_4sp (__m128 left, __m128 right)
{
return vec_comparegt4sp (left, right);
}


VECLIB_INLINE __m128 vec_comparenotgt4sp (__m128 left, __m128 right)
{
__m128_union leftx;
leftx.as_m128 = vec_comparegt4sp (left, right);
__m128_union rightx;
rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);

return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}


VECLIB_INLINE __m128 vec_comparegtlower1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparegt4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_comparenotgtlower1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparenotgt4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE int vec_comparegt1of4sptobool (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparegt4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}


VECLIB_INLINE int vec_comparegt1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparegt4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}




VECLIB_INLINE __m128 vec_comparege4sp(__m128 left, __m128 right)
{
return (__m128) vec_cmpge ((vector float) left, (vector float) right);
}


VECLIB_INLINE __m128 vec_comparenotge4sp(__m128 left, __m128 right)
{
__m128_union leftx;
leftx.as_m128 = vec_comparege4sp (left, right);
__m128_union rightx;
rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);
return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}


VECLIB_INLINE __m128 vec_comparegelower1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparege4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_comparenotgelower1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparenotge4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE int vec_comparege1of4sptobool (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparege4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}


VECLIB_INLINE int vec_comparege1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
__m128_union res_union;
__m128_union nan_union;
res_union.as_m128 = vec_comparege4sp(left, right);
nan_union.as_m128 = vec_comparenotnans4sp(left, right);

return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}




VECLIB_INLINE __m128 vec_comparenotnans4sp (__m128 left, __m128 right)
{
__m128 left_mask = vec_compareeq4sp(left, left);
__m128 right_mask = vec_compareeq4sp(right, right);
return (__m128) vec_and (left_mask, right_mask);
}


VECLIB_INLINE __m128 vec_comparenans4sp (__m128 left, __m128 right)
{
__m128_union leftx;
leftx.as_m128 = vec_comparenotnans4sp (left, right);
__m128_union rightx;
rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);
return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}


VECLIB_INLINE __m128 vec_comparenotnanslower1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparenotnans4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}


VECLIB_INLINE __m128 vec_comparenanslower1of4sp (__m128 left, __m128 right)
{
vector unsigned int select_vector = {
0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
};
vector float inter_res = vec_comparenans4sp ((vector float) left, (vector float) right);
vector float result = vec_sel ((vector float) left, inter_res, select_vector);
return (__m128) result;
}




VECLIB_INLINE __m128 vec_cast1qto4sp (__m128i v)
{
__m128_all_union v_union;
v_union.as_m128i = v;
return (__m128) v_union.as_m128;
}


#ifdef VECLIB_VSX
VECLIB_INLINE __m128 vec_cast4of8spto4sp (__m256 from)
{
__m256_union from_union;  from_union.as_m256 = from;
#ifdef __LITTLE_ENDIAN__
return from_union.as_m128[0];
#elif __BIG_ENDIAN__
return from_union.as_m128[1];
#endif
}
#endif


VECLIB_INLINE __m128 vec_Cast2dpto4sp (__m128d from) {
__m128_all_union newFrom; newFrom.as_m128d = from;
return newFrom.as_m128;
}




VECLIB_INLINE __m128 vec_addsub4sp (__m128 left, __m128 right) {
__m128_all_union negation;
negation.as_vector_unsigned_int = (vector unsigned int) {
#ifdef __LITTLE_ENDIAN__
0x80000000u, 0x00000000u, 0x80000000u, 0x00000000u
#elif __BIG_ENDIAN__
0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u
#endif
};
__m128 tempResult = (vector float) vec_xor (right, negation.as_vector_float);
return (vector float) vec_add (left, tempResult);
}


VECLIB_INLINE __m128 vec_horizontaladd4sp (__m128 lower, __m128 upper) {
vector unsigned char transformation2 = {
#ifdef __LITTLE_ENDIAN__
0x00, 0x01, 0x02, 0x03,  0x08, 0x09, 0x0A, 0x0B,  0x10, 0x11, 0x12, 0x13,   0x18, 0x19, 0x1A, 0x1B
#elif __BIG_ENDIAN__
0x14, 0x15, 0x16, 0x17,  0x1C, 0x1D, 0x1E, 0x1F,  0x04, 0x05, 0x06, 0x07,   0x0C, 0x0D, 0x0E, 0x0F
#endif
};
vector unsigned char transformation1 = {
#ifdef __LITTLE_ENDIAN__
0x04, 0x05, 0x06, 0x07,  0x0C, 0x0D, 0x0E, 0x0F,  0x14, 0x15, 0x16, 0x17,    0x1C, 0x1D, 0x1E, 0x1F
#elif __BIG_ENDIAN__
0x10, 0x11, 0x12, 0x13,  0x18, 0x19, 0x1A, 0x1B,  0x00, 0x01, 0x02, 0x03,    0x08, 0x09, 0x0A, 0x0B
#endif
};
return (vector float) vec_add (vec_perm ((vector float) lower, (vector float) upper, transformation1),
vec_perm ((vector float) lower, (vector float) upper, transformation2));
}


VECLIB_INLINE __m128 vec_horizontalsub4sp (__m128 lower, __m128 upper) {
vector unsigned char transformation2 = {
#ifdef __LITTLE_ENDIAN__
0x00, 0x01, 0x02, 0x03,  0x08, 0x09, 0x0A, 0x0B,  0x10, 0x11, 0x12, 0x13,   0x18, 0x19, 0x1A, 0x1B
#elif __BIG_ENDIAN__
0x10, 0x11, 0x12, 0x13,  0x18, 0x19, 0x1A, 0x1B,  0x00, 0x01, 0x02, 0x03,    0x08, 0x09, 0x0A, 0x0B
#endif
};
vector unsigned char transformation1 = {
#ifdef __LITTLE_ENDIAN__
0x04, 0x05, 0x06, 0x07,  0x0C, 0x0D, 0x0E, 0x0F,  0x14, 0x15, 0x16, 0x17,    0x1C, 0x1D, 0x1E, 0x1F
#elif __BIG_ENDIAN__
0x14, 0x15, 0x16, 0x17,  0x1C, 0x1D, 0x1E, 0x1F,  0x04, 0x05, 0x06, 0x07,   0x0C, 0x0D, 0x0E, 0x0F
#endif
};
return (vector float) vec_sub (vec_perm ((vector float) lower, (vector float) upper, transformation2),
vec_perm ((vector float) lower, (vector float) upper, transformation1));
}




VECLIB_INLINE __m128 vec_unpackupper2spto4sp (__m128 even, __m128 odd)
{
static const vector unsigned char permute_selector = {
#ifdef __LITTLE_ENDIAN__
0x08, 0x09, 0x0A, 0x0B,  0x18, 0x19, 0x1A, 0x1B,  0x0C, 0x0D, 0x0E, 0x0F,  0x1C, 0x1D, 0x1E, 0x1F
#elif __BIG_ENDIAN__
0x10, 0x11, 0x12, 0x13,  0x00, 0x01, 0x02, 0x03,  0x14, 0x15, 0x16, 0x17,  0x04, 0x05, 0x06, 0x07
#endif
};
return vec_perm (even, odd, permute_selector);
}


VECLIB_INLINE __m128 vec_unpackupper22spto4sp (__m128 even, __m128 odd)
{
return vec_unpackupper2spto4sp (even, odd);
}


VECLIB_INLINE __m128 vec_unpacklower2spto4sp (__m128 even, __m128 odd)
{
static const vector unsigned char permute_selector = {
#ifdef __LITTLE_ENDIAN__
0x00, 0x01, 0x02, 0x03,  0x10, 0x11, 0x12, 0x13,  0x04, 0x05, 0x06, 0x07,  0x14, 0x15, 0x16, 0x17
#elif __BIG_ENDIAN__
0x18, 0x19, 0x1A, 0x1B,  0x08, 0x09, 0x0A, 0x0B,  0x1C, 0x1D, 0x1E, 0x1F,  0x0C, 0x0D, 0x0E, 0x0F
#endif
};
return vec_perm (even, odd, permute_selector);
}


VECLIB_INLINE __m128 vec_unpacklower22spto4sp (__m128 even, __m128 odd)
{
return vec_unpacklower2spto4sp (even, odd);
}

#endif
