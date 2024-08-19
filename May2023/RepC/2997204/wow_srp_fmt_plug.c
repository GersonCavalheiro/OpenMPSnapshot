#if AC_BUILT
#include "autoconfig.h"
#endif
#if HAVE_LIBGMP || HAVE_LIBCRYPTO 
#if FMT_EXTERNS_H
extern struct fmt_main fmt_blizzard;
#elif FMT_REGISTERS_H
john_register_one(&fmt_blizzard);
#else
#include <string.h>
#ifdef HAVE_LIBGMP
#if HAVE_GMP_GMP_H
#include <gmp/gmp.h>
#else
#include <gmp.h>
#endif
#define EXP_STR " GMP-exp"
#else
#include <openssl/bn.h>
#define EXP_STR " oSSL-exp"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "arch.h"
#include "sha.h"
#include "sha2.h"
#include "params.h"
#include "common.h"
#include "formats.h"
#include "unicode.h" 
#include "johnswap.h"
#ifndef OMP_SCALE
#define OMP_SCALE          256	
#endif
#define FORMAT_LABEL		"WoWSRP"
#define FORMAT_NAME		"Battlenet"
#define ALGORITHM_NAME		"SHA1 32/" ARCH_BITS_STR EXP_STR
#define BENCHMARK_COMMENT	""
#define BENCHMARK_LENGTH	8
#define WOWSIG			"$WoWSRP$"
#define WOWSIGLEN		(sizeof(WOWSIG)-1)
#define PLAINTEXT_LENGTH	16
#define CIPHERTEXT_LENGTH	64
#define BINARY_SIZE		4
#define BINARY_ALIGN		4
#define FULL_BINARY_SIZE	32
#define USERNAMELEN             32
#define ONLY_SALT_SIZE		(64+3)
#define SALT_SIZE		(1 + ONLY_SALT_SIZE + USERNAMELEN + 1)
#define SALT_ALIGN		1
#define MIN_KEYS_PER_CRYPT	1
#define MAX_KEYS_PER_CRYPT	1
static struct fmt_tests tests[] = {
{WOWSIG"6D00CD214C8473C7F4E9DC77AE8FC6B3944298C48C7454E6BB8296952DCFE78D$73616C74", "PASSWORD", {"SOLAR"}},
{WOWSIG"A35DCC134159A34F1D411DA7F38AB064B617D5DBDD9258FE2F23D5AB1CF3F685$73616C7432", "PASSWORD2", {"DIZ"}},
{WOWSIG"A35DCC134159A34F1D411DA7F38AB064B617D5DBDD9258FE2F23D5AB1CF3F685$73616C7432*DIZ", "PASSWORD2"},
{"$WoWSRP$01C7F618E4589F3229D764580FDBF0D579D7CB1C071F11C856BDDA9E41946530$36354172646F744A366A7A58386D4D6E*JOHN", "PASSWORD"},
{"$WoWSRP$1C7F618E4589F3229D764580FDBF0D579D7CB1C071F11C856BDDA9E41946530$36354172646F744A366A7A58386D4D6E*JOHN", "PASSWORD"},
{NULL}
};
#ifdef HAVE_LIBGMP
typedef struct t_SRP_CTX {
mpz_t z_mod, z_base, z_exp, z_rop;
} SRP_CTX;
#else
typedef struct t_SRP_CTX {
BIGNUM *z_mod, *z_base, *z_exp, *z_rop;
BN_CTX *BN_ctx;
}SRP_CTX;
#endif
static SRP_CTX *pSRP_CTX;
static unsigned char saved_salt[SALT_SIZE];
static unsigned char user_id[USERNAMELEN];
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static uint32_t (*crypt_out)[8];
static int max_keys_per_crypt;
static void init(struct fmt_main *self)
{
int i;
omp_autotune(self, OMP_SCALE);
saved_key = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*saved_key));
crypt_out = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*crypt_out));
pSRP_CTX  = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*pSRP_CTX));
max_keys_per_crypt = self->params.max_keys_per_crypt;
for (i = 0; i < max_keys_per_crypt; ++i) {
#ifdef HAVE_LIBGMP
mpz_init_set_str(pSRP_CTX[i].z_mod, "112624315653284427036559548610503669920632123929604336254260115573677366691719", 10);
mpz_init_set_str(pSRP_CTX[i].z_base, "47", 10);
mpz_init_set_str(pSRP_CTX[i].z_exp, "1", 10);
mpz_init(pSRP_CTX[i].z_rop);
mpz_mul_2exp(pSRP_CTX[i].z_exp, pSRP_CTX[i].z_exp, 159);
#else
pSRP_CTX[i].z_mod=BN_new();
BN_dec2bn(&pSRP_CTX[i].z_mod, "112624315653284427036559548610503669920632123929604336254260115573677366691719");
pSRP_CTX[i].z_base=BN_new();
BN_set_word(pSRP_CTX[i].z_base, 47);
pSRP_CTX[i].z_exp=BN_new();
pSRP_CTX[i].z_rop=BN_new();
pSRP_CTX[i].BN_ctx = BN_CTX_new();
#endif
}
}
static void done(void)
{
int i;
for (i = 0; i < max_keys_per_crypt; ++i) {
#ifdef HAVE_LIBGMP
mpz_clear(pSRP_CTX[i].z_mod);
mpz_clear(pSRP_CTX[i].z_base);
mpz_clear(pSRP_CTX[i].z_exp);
mpz_clear(pSRP_CTX[i].z_rop);
#else
BN_clear_free(pSRP_CTX[i].z_mod);
BN_clear_free(pSRP_CTX[i].z_base);
BN_clear_free(pSRP_CTX[i].z_exp);
BN_clear_free(pSRP_CTX[i].z_rop);
BN_CTX_free(pSRP_CTX[i].BN_ctx);
#endif
}
MEM_FREE(pSRP_CTX);
MEM_FREE(crypt_out);
MEM_FREE(saved_key);
}
static int valid(char *ciphertext, struct fmt_main *self)
{
char *p, *q;
if (strncmp(ciphertext, WOWSIG, WOWSIGLEN))
return 0;
q = p = &ciphertext[WOWSIGLEN];
while (atoi16[ARCH_INDEX(*q)] != 0x7F)
q++;
if (q-p > CIPHERTEXT_LENGTH)
return 0;
if (*q != '$')
return 0;
++q;
p = strchr(q, '*');
if (!p)
return 0;
if (((p - q) & 1))
return 0;
if (p - q >= 2 * ONLY_SALT_SIZE)
return 0;
while (atoi16[ARCH_INDEX(*q)] != 0x7F)
q++;
if (q != p)
return 0;
if (strlen(&p[1]) > USERNAMELEN)
return 0;
return 1;
}
static void StripZeros(const char *ct, char *ct2, const int ct2_size) {
int i;
for (i = 0; i < WOWSIGLEN && i < (ct2_size - 1); ++i)
*ct2++ = *ct++;
while (*ct == '0')
++ct;
while (*ct && i < (ct2_size - 1)) {
*ct2++ = *ct++;
i++;
}
*ct2 = 0;
}
static char *prepare(char *split_fields[10], struct fmt_main *pFmt) {
static char ct[128+32+1];
char *cp;
if (!split_fields[1][0] || strncmp(split_fields[1], WOWSIG, WOWSIGLEN))
return split_fields[1];
cp = strchr(split_fields[1], '*');
if (cp) {
if (split_fields[1][WOWSIGLEN] == '0') {
StripZeros(split_fields[1], ct, sizeof(ct));
return ct;
}
return split_fields[1];
}
if (strnlen(split_fields[1], 129) <= 128) {
strnzcpy(ct, split_fields[1], 128);
cp = &ct[strlen(ct)];
*cp++ = '*';
strnzcpy(cp, split_fields[0], USERNAMELEN);
enc_strupper(cp);
if (ct[WOWSIGLEN] == '0') {
char ct2[128+32+1];
StripZeros(ct, ct2, sizeof(ct2));
strcpy(ct, ct2);
}
return ct;
}
return split_fields[1];
}
static char *split(char *ciphertext, int index, struct fmt_main *pFmt) {
static char ct[128+32+1];
char *cp;
strnzcpy(ct, ciphertext, 128+32+1);
cp = strchr(ct, '*');
if (cp) *cp = 0;
strupr(&ct[WOWSIGLEN]);
if (cp) *cp = '*';
if (ct[WOWSIGLEN] == '0') {
char ct2[128+32+1];
StripZeros(ct, ct2, sizeof(ct2));
strcpy(ct, ct2);
}
return ct;
}
static void *get_binary(char *ciphertext)
{
static union {
unsigned char b[FULL_BINARY_SIZE];
uint32_t dummy[1];
} out;
char *p, *q;
int i;
p = &ciphertext[WOWSIGLEN];
q = strchr(p, '$');
memset(out.b, 0, sizeof(out.b));
while (*p == '0')
++p;
if ((q-p)&1) {
out.b[0] = atoi16[ARCH_INDEX(*p)];
++p;
} else {
out.b[0] =
(atoi16[ARCH_INDEX(*p)] << 4) |
atoi16[ARCH_INDEX(p[1])];
p += 2;
}
for (i = 1; i < FULL_BINARY_SIZE; i++) {
out.b[i] =
(atoi16[ARCH_INDEX(*p)] << 4) |
atoi16[ARCH_INDEX(p[1])];
p += 2;
if (p >= q)
break;
}
return out.b;
}
static void *get_salt(char *ciphertext)
{
static union {
unsigned char b[SALT_SIZE];
uint32_t dummy;
} out;
char *p;
int length=0;
memset(out.b, 0, SALT_SIZE);
p = strchr(&ciphertext[WOWSIGLEN], '$') + 1;
while (atoi16[ARCH_INDEX(*p++)] != 0x7f)
length++;
p = strchr(&ciphertext[WOWSIGLEN], '$') + 1;
if ((length&1)&&atoi16[ARCH_INDEX(*p)] != 0x7f) {
length=0;
out.b[++length] = atoi16[ARCH_INDEX(*p)];
++p;
} else
length = 0;
while (atoi16[ARCH_INDEX(*p)] != 0x7f && atoi16[ARCH_INDEX(p[1])] != 0x7f) {
out.b[++length] =
(atoi16[ARCH_INDEX(*p)] << 4) |
atoi16[ARCH_INDEX(p[1])];
p += 2;
}
out.b[0] = length;
if (*p) {
++p;
memcpy(out.b + length+1, p, strlen(p)+1);
}
return out.b;
}
#define COMMON_GET_HASH_VAR crypt_out
#include "common-get-hash.h"
static int salt_hash(void *salt)
{
unsigned int hash = 0;
char *p = (char *)salt;
while (*p) {
hash <<= 1;
hash += (unsigned char)*p++;
if (hash >> SALT_HASH_LOG) {
hash ^= hash >> SALT_HASH_LOG;
hash &= (SALT_HASH_SIZE - 1);
}
}
hash ^= hash >> SALT_HASH_LOG;
hash &= (SALT_HASH_SIZE - 1);
return hash;
}
static void set_salt(void *salt)
{
unsigned char *cp = (unsigned char*)salt;
memcpy(saved_salt, &cp[1], *cp);
saved_salt[*cp] = 0;
strcpy((char*)user_id, (char*)&cp[*cp+1]);
}
static void set_key(char *key, int index)
{
strnzcpy(saved_key[index], key, sizeof(*saved_key));
enc_strupper(saved_key[index]);
}
static char *get_key(int index)
{
return saved_key[index];
}
static int crypt_all(int *pcount, struct db_salt *salt)
{
const int count = *pcount;
int j;
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (j = 0; j < count; ++j) {
SHA_CTX ctx;
unsigned char Tmp[20];
memset(crypt_out[j], 0, sizeof(crypt_out[j]));
SHA1_Init(&ctx);
SHA1_Update(&ctx, user_id, strlen((char*)user_id));
SHA1_Update(&ctx, ":", 1);
SHA1_Update(&ctx, saved_key[j], strlen(saved_key[j]));
SHA1_Final(Tmp, &ctx);
SHA1_Init(&ctx);
SHA1_Update(&ctx, saved_salt, strlen((char*)saved_salt));
SHA1_Update(&ctx, Tmp, 20);
SHA1_Final(Tmp, &ctx);
#ifdef HAVE_LIBGMP
{
unsigned char HashStr[80], *p;
int i, todo;
p = HashStr;
for (i = 0; i < 20; ++i) {
*p++ = itoa16[Tmp[i]>>4];
*p++ = itoa16[Tmp[i]&0xF];
}
*p = 0;
mpz_set_str(pSRP_CTX[j].z_exp, (char*)HashStr, 16);
mpz_powm (pSRP_CTX[j].z_rop, pSRP_CTX[j].z_base, pSRP_CTX[j].z_exp, pSRP_CTX[j].z_mod );
mpz_get_str ((char*)HashStr, 16, pSRP_CTX[j].z_rop);
p = HashStr;
todo = strlen((char*)p);
if (todo&1) {
((unsigned char*)(crypt_out[j]))[0] = atoi16[ARCH_INDEX(*p)];
++p;
--todo;
} else {
((unsigned char*)(crypt_out[j]))[0] =
(atoi16[ARCH_INDEX(*p)] << 4) |
atoi16[ARCH_INDEX(p[1])];
p += 2;
todo -= 2;
}
todo >>= 1;
for (i = 1; i <= todo; i++) {
((unsigned char*)(crypt_out[j]))[i] =
(atoi16[ARCH_INDEX(*p)] << 4) |
atoi16[ARCH_INDEX(p[1])];
p += 2;
}
}
#else
pSRP_CTX[j].z_exp = BN_bin2bn(Tmp,20,pSRP_CTX[j].z_exp);
BN_mod_exp(pSRP_CTX[j].z_rop, pSRP_CTX[j].z_base, pSRP_CTX[j].z_exp, pSRP_CTX[j].z_mod, pSRP_CTX[j].BN_ctx);
BN_bn2bin(pSRP_CTX[j].z_rop, (unsigned char*)(crypt_out[j]));
#endif
}
return count;
}
static int cmp_all(void *binary, int count)
{
int i;
for (i = 0; i < count; ++i) {
if (*((uint32_t*)binary) == *((uint32_t*)(crypt_out[i])))
return 1;
}
return 0;
}
static int cmp_one(void *binary, int index)
{
return *((uint32_t*)binary) == *((uint32_t*)(crypt_out[index]));
}
static int cmp_exact(char *source, int index)
{
return !memcmp(get_binary(source), crypt_out[index], BINARY_SIZE);
}
struct fmt_main fmt_blizzard = {
{
FORMAT_LABEL,
FORMAT_NAME,
ALGORITHM_NAME,
BENCHMARK_COMMENT,
BENCHMARK_LENGTH,
8,
PLAINTEXT_LENGTH,
BINARY_SIZE,
BINARY_ALIGN,
SALT_SIZE,
SALT_ALIGN,
MIN_KEYS_PER_CRYPT,
MAX_KEYS_PER_CRYPT,
FMT_8_BIT | FMT_SPLIT_UNIFIES_CASE | FMT_OMP,
{ NULL },
{ WOWSIG },
tests
}, {
init,
done,
fmt_default_reset,
prepare,
valid,
split,
get_binary,
get_salt,
{ NULL },
fmt_default_source,
{
fmt_default_binary_hash_0,
fmt_default_binary_hash_1,
fmt_default_binary_hash_2,
fmt_default_binary_hash_3,
fmt_default_binary_hash_4,
fmt_default_binary_hash_5,
fmt_default_binary_hash_6
},
salt_hash,
NULL,
set_salt,
set_key,
get_key,
fmt_default_clear_keys,
crypt_all,
{
#define COMMON_GET_HASH_LINK
#include "common-get-hash.h"
},
cmp_all,
cmp_one,
cmp_exact
}
};
#endif 
#endif 
