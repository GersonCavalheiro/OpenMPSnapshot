#pragma once
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
DEFAULT=-42,
FALLBACK=0, 
SSE=1,  
SSE2=2, 
SSE3=3, 
SSSE3=4, 
SSE4=5,
SSE42=6, 
AVX=7, 
AVX2=8,  
AVX512F=9,
NUM_ISA  
} isa;  
static inline void cpuid (int output[4], int functionnumber) {	
#if defined(__GNUC__) || defined(__clang__)              
int a, b, c, d;
__asm("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "a"(functionnumber),"c"(0) );
output[0] = a;
output[1] = b;
output[2] = c;
output[3] = d;
#else                                                      
__asm {
mov eax, functionnumber
xor ecx, ecx
cpuid;
mov esi, output
mov [esi],    eax
mov [esi+4],  ebx
mov [esi+8],  ecx
mov [esi+12], edx
}
#endif
}
static inline int64_t xgetbv (int ctr) {	
#if (defined (__INTEL_COMPILER) && __INTEL_COMPILER >= 1200) 
return _xgetbv(ctr);                                   
#elif defined(__GNUC__)                                    
uint32_t a, d;
__asm("xgetbv" : "=a"(a),"=d"(d) : "c"(ctr) : );
return a | (((uint64_t) d) << 32);
#else  
uint32_t a, d;
__asm {
mov ecx, ctr
_emit 0x0f
_emit 0x01
_emit 0xd0 ; 
mov a, eax
mov d, edx
}
return a | (((uint64_t) d) << 32);
#endif
}
extern int runtime_instrset_detect(void);
extern int get_max_usable_isa(void);
#ifdef __cplusplus
}
#endif
