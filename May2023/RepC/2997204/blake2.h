#ifndef __BLAKE2_H__
#define __BLAKE2_H__
#include <stddef.h>
#include <stdint.h>
#include "aligned.h"
#include "jumbo.h"
#if defined(__cplusplus)
extern "C" {
#endif
enum blake2s_constant
{
BLAKE2S_BLOCKBYTES = 64,
BLAKE2S_OUTBYTES   = 32,
BLAKE2S_KEYBYTES   = 32,
BLAKE2S_SALTBYTES  = 8,
BLAKE2S_PERSONALBYTES = 8
};
enum blake2b_constant
{
BLAKE2B_BLOCKBYTES = 128,
BLAKE2B_OUTBYTES   = 64,
BLAKE2B_KEYBYTES   = 64,
BLAKE2B_SALTBYTES  = 16,
BLAKE2B_PERSONALBYTES = 16
};
typedef struct __blake2s_param
{
uint8_t  digest_length; 
uint8_t  key_length;    
uint8_t  fanout;        
uint8_t  depth;         
uint32_t leaf_length;   
uint8_t  node_offset[6];
uint8_t  node_depth;    
uint8_t  inner_length;  
uint8_t  salt[BLAKE2B_SALTBYTES]; 
uint8_t  personal[BLAKE2S_PERSONALBYTES];  
} blake2s_param;
typedef struct JTR_ALIGN( 64 ) __blake2s_state
{
uint32_t h[8];
uint32_t t[2];
uint32_t f[2];
uint8_t  buf[2 * BLAKE2S_BLOCKBYTES];
size_t   buflen;
uint8_t  last_node;
} blake2s_state ;
typedef struct __blake2b_param
{
uint8_t  digest_length; 
uint8_t  key_length;    
uint8_t  fanout;        
uint8_t  depth;         
uint32_t leaf_length;   
uint64_t node_offset;   
uint8_t  node_depth;    
uint8_t  inner_length;  
uint8_t  reserved[14];  
uint8_t  salt[BLAKE2B_SALTBYTES]; 
uint8_t  personal[BLAKE2B_PERSONALBYTES];  
} blake2b_param;
typedef struct JTR_ALIGN( 64 ) __blake2b_state
{
uint64_t h[8];
uint64_t t[2];
uint64_t f[2];
uint8_t  buf[2 * BLAKE2B_BLOCKBYTES];
size_t   buflen;
uint8_t  last_node;
} blake2b_state;
#if defined(JOHN_NO_SIMD) || (!defined(__SSE2__) && !defined(__SSE4_1__) && !defined(__XOP__))
typedef struct __blake2sp_state
#else
typedef struct JTR_ALIGN( 64 ) __blake2sp_state
#endif
{
blake2s_state S[8][1];
blake2s_state R[1];
uint8_t buf[8 * BLAKE2S_BLOCKBYTES];
size_t  buflen;
} blake2sp_state;
#if defined(JOHN_NO_SIMD) || (!defined(__SSE2__) && !defined(__SSE4_1__) && !defined(__XOP__))
typedef struct __blake2bp_state
#else
typedef struct JTR_ALIGN( 64 ) __blake2bp_state
#endif
{
blake2b_state S[4][1];
blake2b_state R[1];
uint8_t buf[4 * BLAKE2B_BLOCKBYTES];
size_t  buflen;
} blake2bp_state;
int blake2s_init( blake2s_state *S, const uint8_t outlen );
int blake2s_init_key( blake2s_state *S, const uint8_t outlen, const void *key, const uint8_t keylen );
int blake2s_init_param( blake2s_state *S, const blake2s_param *P );
int blake2s_update( blake2s_state *S, const uint8_t *in, uint64_t inlen );
int blake2s_final( blake2s_state *S, uint8_t *out, uint8_t outlen );
int blake2b_init( blake2b_state *S, const uint8_t outlen );
int blake2b_init_key( blake2b_state *S, const uint8_t outlen, const void *key, const uint8_t keylen );
int blake2b_init_param( blake2b_state *S, const blake2b_param *P );
int blake2b_update( blake2b_state *S, const uint8_t *in, uint64_t inlen );
int blake2b_final( blake2b_state *S, uint8_t *out, uint8_t outlen );
int blake2sp_init( blake2sp_state *S, const uint8_t outlen );
int blake2sp_init_key( blake2sp_state *S, const uint8_t outlen, const void *key, const uint8_t keylen );
int blake2sp_update( blake2sp_state *S, const uint8_t *in, uint64_t inlen );
int blake2sp_final( blake2sp_state *S, uint8_t *out, uint8_t outlen );
int blake2bp_init( blake2bp_state *S, const uint8_t outlen );
int blake2bp_init_key( blake2bp_state *S, const uint8_t outlen, const void *key, const uint8_t keylen );
int blake2bp_update( blake2bp_state *S, const uint8_t *in, uint64_t inlen );
int blake2bp_final( blake2bp_state *S, uint8_t *out, uint8_t outlen );
int blake2s( uint8_t *out, const void *in, const void *key, const uint8_t outlen, const uint64_t inlen, uint8_t keylen );
int blake2b( uint8_t *out, const void *in, const void *key, const uint8_t outlen, const uint64_t inlen, uint8_t keylen );
int blake2sp( uint8_t *out, const void *in, const void *key, const uint8_t outlen, const uint64_t inlen, uint8_t keylen );
int blake2bp( uint8_t *out, const void *in, const void *key, const uint8_t outlen, const uint64_t inlen, uint8_t keylen );
inline static int blake2( uint8_t *out, const void *in, const void *key, const uint8_t outlen, const uint64_t inlen, uint8_t keylen )
{
return blake2b( out, in, key, outlen, inlen, keylen );
}
#if defined(__cplusplus)
}
#endif
#endif
