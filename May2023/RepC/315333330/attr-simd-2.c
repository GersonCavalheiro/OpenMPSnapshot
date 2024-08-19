#pragma omp declare simd
extern
#ifdef __cplusplus
"C"
#endif
__attribute__((__simd__))
int simd_attr (void)
{
return 0;
}
