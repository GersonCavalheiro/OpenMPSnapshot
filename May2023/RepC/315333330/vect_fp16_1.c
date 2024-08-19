#pragma GCC target ("arch=armv8.2-a+fp16")
_Float16
sum_Float16 (_Float16 *__restrict__ __attribute__ ((__aligned__ (16))) a,
_Float16 *__restrict__ __attribute__ ((__aligned__ (16))) b,
_Float16 *__restrict__ __attribute__ ((__aligned__ (16))) c)
{
for (int i = 0; i < 256; i++)
a[i] = b[i] + c[i];
}
_Float16
sum_fp16 (__fp16 *__restrict__ __attribute__ ((__aligned__ (16))) a,
__fp16 *__restrict__ __attribute__ ((__aligned__ (16))) b,
__fp16 *__restrict__ __attribute__ ((__aligned__ (16))) c)
{
for (int i = 0; i < 256; i++)
a[i] = b[i] + c[i];
}
