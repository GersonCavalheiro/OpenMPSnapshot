#if FMT_EXTERNS_H
extern struct fmt_main fmt_encdadatavault_pbkdf2;
#elif FMT_REGISTERS_H
john_register_one(&fmt_encdadatavault_pbkdf2);
#else
#ifdef _OPENMP
#include <omp.h>
#endif
#include "md5.h"
#include "pbkdf2_hmac_sha256.h"
#include "encdatavault_common.h"
#define FORMAT_LABEL_PBKDF2       	"ENCDataVault-PBKDF2"
#define FORMAT_NAME               	""
#ifdef SIMD_COEF_32
#define ALGORITHM_NAME_PBKDF2     	"PBKDF2-HMAC-SHA256 " SHA256_ALGORITHM_NAME
#else
#define ALGORITHM_NAME_PBKDF2     	"PBKDF2-HMAC-SHA256 32/" ARCH_BITS_STR
#endif
#define BENCHMARK_COMMENT         	""
#define BENCHMARK_LENGTH          	0x107
#define PLAINTEXT_LENGTH          	125
#define BINARY_SIZE               	0
#define BINARY_ALIGN              	1
#define SALT_SIZE                 	sizeof(custom_salt)
#define SALT_ALIGN                	4
#ifdef SIMD_COEF_32
#define MIN_KEYS_PER_PBKDF2_CRYPT	SSE_GROUP_SZ_SHA256
#define MAX_KEYS_PER_PBKDF2_CRYPT   SSE_GROUP_SZ_SHA256
#else
#define MIN_KEYS_PER_PBKDF2_CRYPT   1
#define MAX_KEYS_PER_PBKDF2_CRYPT   1
#endif
#ifndef OMP_SCALE
#define OMP_SCALE               	1
#endif
static struct fmt_tests encdatavault_pbkdf2_tests[] = {
{ "$encdv-pbkdf2$1$1$eb012112b3561e6e$01c7dca58660d6ae$32$9b1c17f7a467cf4d13b73dfd3ce12b3dd51cb980cc1f0ccfad4e47e9c9f3d774$100000", "123456Aa!"},
{ "$encdv-pbkdf2$3$4$4f1d3889c629968c$e8b68bb804b94a23$32$fa79c83e85a41973799522f525e1316ca07ab663def47b816da76205dc1e3e80$100000$0443baa02c7e00870a24f0c16ff0454c56ad079e334bb6fb52bf39de56485dc07f4827d8d83b6f3bf135062e2869d58b8ade04db0716c75d2519797cdbb1d9d75ffe3fc156edfd3bb95747fc0286c149c3c90da8a4b4cdf11838b680b6e9b26da3a0c93f29f8bff6e70f07e6077cda620f7b808793dc127395a11bfc068f1895", "123456789ABCDEf"},
{ NULL }
};
static custom_salt *cur_salt;
static int *cracked;
static int any_cracked;
static size_t cracked_size;
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static void init(struct fmt_main *self)
{
omp_autotune(self, OMP_SCALE);
saved_key = mem_calloc(sizeof(*saved_key), self->params.max_keys_per_crypt);
cracked = mem_calloc(sizeof(*cracked), self->params.max_keys_per_crypt);
cracked_size = sizeof(*cracked) * self->params.max_keys_per_crypt;
any_cracked = 0;
}
static void done(void)
{
MEM_FREE(saved_key);
MEM_FREE(cracked);
}
static int valid_pbkdf2(char *ciphertext, struct fmt_main *self)
{
return valid_common(ciphertext, self, 1);
}
static void *get_salt_pbkdf2(char *ciphertext)
{
return get_salt_common(ciphertext, 1);
}
static void set_salt(void *salt)
{
cur_salt = (custom_salt *)salt;
}
static int crypt_all_pbkdf2(int *pcount, struct db_salt *salt)
{
const int count = *pcount;
int index;
int nb_keys;
if (any_cracked) {
memset(cracked, 0, cracked_size);
any_cracked = 0;
}
nb_keys = 1 << (cur_salt->algo_id - 1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (index = 0; index < count; index += MIN_KEYS_PER_PBKDF2_CRYPT) {
int i, j, key_len;
buffer_128 kdf_out[MIN_KEYS_PER_PBKDF2_CRYPT][ENC_MAX_KEY_NUM];
buffer_128 tmp;
buffer_128 ivs[ENC_MAX_KEY_NUM];
unsigned char result[ENC_KEY_SIZE * ENC_MAX_KEY_NUM] = { 0 };
unsigned char master[MIN_KEYS_PER_PBKDF2_CRYPT][ENC_KEY_SIZE * ENC_MAX_KEY_NUM];
if (cur_salt->version == 1) {
key_len = nb_keys * ENC_KEY_SIZE;
} else {
key_len = ENC_MAX_KEY_NUM * ENC_KEY_SIZE;
}
#ifdef SIMD_COEF_32
int lens[MIN_KEYS_PER_PBKDF2_CRYPT];
unsigned char *pin[MIN_KEYS_PER_PBKDF2_CRYPT], *pout[MIN_KEYS_PER_PBKDF2_CRYPT];
for (i = 0; i < MIN_KEYS_PER_PBKDF2_CRYPT; ++i) {
lens[i] = strlen(saved_key[index + i]);
pin[i] = (unsigned char *)saved_key[index + i];
pout[i] = master[i];
}
pbkdf2_sha256_sse((const unsigned char **)pin, lens, cur_salt->salt, cur_salt->salt_length,
cur_salt->iterations, pout, key_len, 0);
for (i = 0; i < MIN_KEYS_PER_PBKDF2_CRYPT; ++i) {
for (j = 0; j < ENC_MAX_KEY_NUM; j++) {
memcpy(kdf_out[i][j].u8, pout[i] + (j * ENC_KEY_SIZE), ENC_KEY_SIZE);
}
}
#else
for (i = 0; i < MIN_KEYS_PER_PBKDF2_CRYPT; ++i) {
pbkdf2_sha256((unsigned char *)saved_key[index + i], strlen(saved_key[index + i]), cur_salt->salt,
cur_salt->salt_length, cur_salt->iterations, master[i], key_len, 0);
for (j = 0; j < ENC_MAX_KEY_NUM; j++) {
memcpy(kdf_out[i][j].u8, master[i] + (j * ENC_KEY_SIZE), ENC_KEY_SIZE);
}
}
#endif
for (i = 0; i < MIN_KEYS_PER_PBKDF2_CRYPT; ++i) {
if (cur_salt->version == 1) {
memcpy(ivs[0].u8, cur_salt->iv, ENC_NONCE_SIZE);
for (j = 1; j < nb_keys; j++) {
memcpy(ivs[j].u8, cur_salt->iv, ENC_NONCE_SIZE);
ivs[j].u64[0] ^= kdf_out[i][j].u64[0];
}
enc_aes_ctr_iterated(cur_salt->encrypted_data, result, kdf_out[i][0].u8, ivs, AES_BLOCK_SIZE,
nb_keys, 1);
if (!memcmp(result + 4, "\xd2\xc3\xb4\xa1\x00\x00", MIN(cur_salt->encrypted_data_length, ENC_SIG_SIZE - 2))) {
cracked[index + i] = 1;
#ifdef _OPENMP
#pragma omp atomic
#endif
any_cracked |= 1;
}
} else {
ivs[0].u64[0] = 0;
for (j = 1; j < ENC_MAX_KEY_NUM; j++) {
ivs[j].u64[0] = kdf_out[i][ENC_MAX_KEY_NUM - j].u64[0];
}
enc_aes_ctr_iterated(cur_salt->keychain, result, kdf_out[i][0].u8, ivs, ENC_KEYCHAIN_SIZE,
ENC_MAX_KEY_NUM, 0);
memcpy(ivs[0].u8, cur_salt->iv, ENC_NONCE_SIZE);
for (j = 1; j < nb_keys; j++) {
memcpy(ivs[j].u8, cur_salt->iv, ENC_NONCE_SIZE);
memcpy(tmp.u8, result + j * 16, ENC_NONCE_SIZE);
ivs[j].u64[0] ^= tmp.u64[0];
}
enc_aes_ctr_iterated(cur_salt->encrypted_data, result, result, ivs, AES_BLOCK_SIZE, nb_keys, 1);
if (!memcmp(result + 4, "\xd2\xc3\xb4\xa1\x00\x00", MIN(cur_salt->encrypted_data_length, ENC_SIG_SIZE - 2))) {
cracked[index + i] = 1;
#ifdef _OPENMP
#pragma omp atomic
#endif
any_cracked |= 1;
}
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
return cracked[index];
}
static void set_key(char *key, int index)
{
strnzcpy(saved_key[index], key, sizeof(saved_key[index]));
}
static char *get_key(int index)
{
return saved_key[index];
}
static unsigned int tunable_cost_iterations(void *_salt)
{
custom_salt *cs = (custom_salt *)_salt;
return cs->iterations;
}
struct fmt_main fmt_encdadatavault_pbkdf2 = {
{
FORMAT_LABEL_PBKDF2,
FORMAT_NAME,
ALGORITHM_NAME_PBKDF2,
BENCHMARK_COMMENT,
BENCHMARK_LENGTH,
0,
PLAINTEXT_LENGTH,
BINARY_SIZE,
BINARY_ALIGN,
SALT_SIZE,
SALT_ALIGN,
MIN_KEYS_PER_PBKDF2_CRYPT,
MAX_KEYS_PER_PBKDF2_CRYPT,
FMT_CASE | FMT_8_BIT | FMT_OMP | FMT_NOT_EXACT,
{ "iterations" },
{ FORMAT_TAG_PBKDF2},
encdatavault_pbkdf2_tests
}, {
init,
done,
fmt_default_reset,
fmt_default_prepare,
valid_pbkdf2,
fmt_default_split,
fmt_default_binary,
get_salt_pbkdf2,
{ tunable_cost_iterations },
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
crypt_all_pbkdf2,
{
fmt_default_get_hash
},
cmp_all,
cmp_one,
cmp_exact
}
};
#endif                          
