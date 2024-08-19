



#pragma once

#if defined (__cplusplus)
extern "C" {
#endif



#include <stddef.h>   
typedef enum { XXH_OK=0, XXH_ERROR } XXH_errorcode;




#ifdef XXH_NAMESPACE
#  define XXH_CAT(A,B) A##B
#  define XXH_NAME2(A,B) XXH_CAT(A,B)
#  define XXH32 XXH_NAME2(XXH_NAMESPACE, XXH32)
#  define XXH64 XXH_NAME2(XXH_NAMESPACE, XXH64)
#  define XXH32_createState XXH_NAME2(XXH_NAMESPACE, XXH32_createState)
#  define XXH64_createState XXH_NAME2(XXH_NAMESPACE, XXH64_createState)
#  define XXH32_freeState XXH_NAME2(XXH_NAMESPACE, XXH32_freeState)
#  define XXH64_freeState XXH_NAME2(XXH_NAMESPACE, XXH64_freeState)
#  define XXH32_reset XXH_NAME2(XXH_NAMESPACE, XXH32_reset)
#  define XXH64_reset XXH_NAME2(XXH_NAMESPACE, XXH64_reset)
#  define XXH32_update XXH_NAME2(XXH_NAMESPACE, XXH32_update)
#  define XXH64_update XXH_NAME2(XXH_NAMESPACE, XXH64_update)
#  define XXH32_digest XXH_NAME2(XXH_NAMESPACE, XXH32_digest)
#  define XXH64_digest XXH_NAME2(XXH_NAMESPACE, XXH64_digest)
#endif




unsigned int       XXH32 (const void* input, size_t length, unsigned seed);
unsigned long long XXH64 (const void* input, size_t length, unsigned long long seed);






typedef struct {
long long ll[ 6];
} XXH32_state_t;
typedef struct {
long long ll[11];
} XXH64_state_t;



XXH32_state_t* XXH32_createState(void);
XXH_errorcode  XXH32_freeState(XXH32_state_t* statePtr);

XXH64_state_t* XXH64_createState(void);
XXH_errorcode  XXH64_freeState(XXH64_state_t* statePtr);




XXH_errorcode XXH32_reset  (XXH32_state_t* statePtr, unsigned seed);
XXH_errorcode XXH32_update (XXH32_state_t* statePtr, const void* input, size_t length);
unsigned int  XXH32_digest (const XXH32_state_t* statePtr);

XXH_errorcode      XXH64_reset  (XXH64_state_t* statePtr, unsigned long long seed);
XXH_errorcode      XXH64_update (XXH64_state_t* statePtr, const void* input, size_t length);
unsigned long long XXH64_digest (const XXH64_state_t* statePtr);




#if defined (__cplusplus)
}
#endif
