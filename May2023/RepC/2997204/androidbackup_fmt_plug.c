#if FMT_EXTERNS_H
extern struct fmt_main fmt_ab;
#elif FMT_REGISTERS_H
john_register_one(&fmt_ab);
#else
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define OMP_SCALE               4  
#include "formats.h"
#include "misc.h"
#include "common.h"
#include "params.h"
#include "options.h"
#include "sha2.h"
#include "aes.h"
#include "pbkdf2_hmac_sha1.h"
#include "androidbackup_common.h"
#define FORMAT_LABEL            "AndroidBackup"
#define FORMAT_NAME             ""
#define FORMAT_TAG              "$ab$"
#define TAG_LENGTH              (sizeof(FORMAT_TAG) - 1)
#ifdef SIMD_COEF_64
#define ALGORITHM_NAME          "PBKDF2-SHA1 " SHA1_ALGORITHM_NAME " AES"
#else
#define ALGORITHM_NAME          "PBKDF2-SHA1 32/" ARCH_BITS_STR " AES"
#endif
#define BENCHMARK_COMMENT       ""
#define BENCHMARK_LENGTH        0x107
#define PLAINTEXT_LENGTH        125
#define BINARY_SIZE             0
#define BINARY_ALIGN            1
#define SALT_SIZE               sizeof(struct custom_salt)
#define SALT_ALIGN              sizeof(uint32_t)
#ifdef SIMD_COEF_32
#define MIN_KEYS_PER_CRYPT      SSE_GROUP_SZ_SHA1
#define MAX_KEYS_PER_CRYPT      (SSE_GROUP_SZ_SHA1 * 4)
#else
#define MIN_KEYS_PER_CRYPT      1
#define MAX_KEYS_PER_CRYPT      4
#endif
#define SALTLEN                 64
#define IVLEN                   16
#define TAGLEN                  16
#define MAX_MASTERKEYBLOB_LEN   128
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static int *saved_len;
static int any_cracked, *cracked;
static size_t cracked_size;
static struct custom_salt *cur_salt;
static void init(struct fmt_main *self)
{
omp_autotune(self, OMP_SCALE);
saved_key = mem_calloc(sizeof(*saved_key), self->params.max_keys_per_crypt);
saved_len = mem_calloc(self->params.max_keys_per_crypt, sizeof(*saved_len));
cracked_size = sizeof(*cracked) * self->params.max_keys_per_crypt;
any_cracked = 0;
cracked = mem_calloc(cracked_size, 1);
}
static void done(void)
{
MEM_FREE(saved_key);
MEM_FREE(saved_len);
MEM_FREE(cracked);
}
static int check_password(unsigned char *user_key, struct custom_salt *cs)
{
unsigned char iv[IVLEN];
unsigned char out[MAX_MASTERKEYBLOB_LEN];
AES_KEY aeskey;
int pad_byte;
int len;
memcpy(iv, cs->iv, cs->iv_length);
AES_set_decrypt_key(user_key, 256, &aeskey);
AES_cbc_encrypt(cs->masterkey_blob, out, cs->masterkey_blob_length, &aeskey, iv, AES_DECRYPT);
len = out[0];
if (len != IVLEN)  
return 0;
if (check_pkcs_pad(out, cs->masterkey_blob_length, 16) < 0)
return 0;
pad_byte = out[cs->masterkey_blob_length - 1];
if (pad_byte > 8)
return 1;
return 0;
}
static void set_salt(void *salt)
{
cur_salt = (struct custom_salt *)salt;
}
static int crypt_all(int *pcount, struct db_salt *salt)
{
const int count = *pcount;
int index;
if (any_cracked) {
memset(cracked, 0, cracked_size);
any_cracked = 0;
}
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (index = 0; index < count; index += MIN_KEYS_PER_CRYPT) {
unsigned char pkey[MIN_KEYS_PER_CRYPT][32];
int i;
#ifdef SIMD_COEF_32
int len[MIN_KEYS_PER_CRYPT];
unsigned char *pin[MIN_KEYS_PER_CRYPT], *pout[MIN_KEYS_PER_CRYPT];
for (i = 0; i < MIN_KEYS_PER_CRYPT; ++i) {
len[i] = strlen(saved_key[i+index]);
pin[i] = (unsigned char*)saved_key[i+index];
pout[i] = pkey[i];
}
pbkdf2_sha1_sse((const unsigned char **)pin, len, cur_salt->user_salt, cur_salt->user_salt_length, cur_salt->iterations, pout, 32, 0);
#else
for (i = 0; i < MIN_KEYS_PER_CRYPT; i++) {
pbkdf2_sha1((unsigned char *)saved_key[index+i],
strlen(saved_key[index+i]),
cur_salt->user_salt, cur_salt->user_salt_length, cur_salt->iterations,
pkey[i], 32, 0);
}
#endif
for (i = 0; i < MIN_KEYS_PER_CRYPT; i++) {
if (check_password(pkey[i], cur_salt)) {
cracked[index+i] = 1;
#ifdef _OPENMP
#pragma omp atomic
#endif
any_cracked |= 1;
} else {
cracked[index+i] = 0;
}
}
}
return count;
}
static int cmp_all(void *binary, int count)
{
return any_cracked;
}
static int cmp_one(void *binary, int index)
{
return cracked[index];
}
static int cmp_exact(char *source, int index)
{
return 1;
}
static void set_key(char *key, int index)
{
saved_len[index] = strnzcpyn(saved_key[index], key, PLAINTEXT_LENGTH + 1);
}
static char *get_key(int index)
{
return saved_key[index];
}
struct fmt_main fmt_ab = {
{
FORMAT_LABEL,
FORMAT_NAME,
ALGORITHM_NAME,
BENCHMARK_COMMENT,
BENCHMARK_LENGTH,
0,
PLAINTEXT_LENGTH,
BINARY_SIZE,
BINARY_ALIGN,
SALT_SIZE,
SALT_ALIGN,
MIN_KEYS_PER_CRYPT,
MAX_KEYS_PER_CRYPT,
FMT_CASE | FMT_8_BIT | FMT_OMP,
{
"iteration count",
},
{ FORMAT_TAG },
ab_tests
}, {
init,
done,
fmt_default_reset,
fmt_default_prepare,
ab_valid,
fmt_default_split,
fmt_default_binary,
ab_get_salt,
{
ab_iteration_count,
},
fmt_default_source,
{
fmt_default_binary_hash
},
fmt_default_salt_hash,
NULL,
set_salt,
set_key,
get_key,
fmt_default_clear_keys,
crypt_all,
{
fmt_default_get_hash
},
cmp_all,
cmp_one,
cmp_exact
}
};
#endif 
