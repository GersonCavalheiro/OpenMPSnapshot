#ifndef _HMAC_MD5_H
#include "aligned.h"
#include "md5.h"
#if defined(__SUNPRO_C)
#pragma align ARCH_SIZE (k_ipad, k_opad)
#endif
typedef struct {
JTR_ALIGN(ARCH_SIZE) unsigned char k_ipad[64];
JTR_ALIGN(ARCH_SIZE) unsigned char k_opad[64];
MD5_CTX ctx;
} HMACMD5Context;
extern void hmac_md5_init_rfc2104(const unsigned char *key, int key_len, HMACMD5Context *ctx);
extern void hmac_md5_init_limK_to_64(const unsigned char*, int, HMACMD5Context*);
extern void hmac_md5_init_K16(const unsigned char*, HMACMD5Context*);
extern void hmac_md5_update(const unsigned char*, int, HMACMD5Context*);
extern void hmac_md5_final(unsigned char*, HMACMD5Context*);
extern void hmac_md5(const unsigned char *key, const unsigned char *data, int data_len, unsigned char *digest);
#endif 
