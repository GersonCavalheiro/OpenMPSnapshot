#if AC_BUILT
#include "autoconfig.h"
#endif
#if HAVE_LIBCRYPTO
#if FMT_EXTERNS_H
extern struct fmt_main fmt_NETLM;
#elif FMT_REGISTERS_H
john_register_one(&fmt_NETLM);
#else
#include <string.h>
#include <openssl/des.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "misc.h"
#include "common.h"
#include "formats.h"
#include "memory.h"
#include "unicode.h"
#ifndef OMP_SCALE
#define OMP_SCALE            128 
#endif
#ifndef uchar
#define uchar unsigned char
#endif
#define FORMAT_LABEL         "netlm"
#define FORMAT_NAME          "LM C/R"
#define FORMAT_TAG           "$NETLM$"
#define FORMAT_TAG_LEN       (sizeof(FORMAT_TAG)-1)
#define ALGORITHM_NAME       "DES 32/" ARCH_BITS_STR
#define BENCHMARK_COMMENT    ""
#define BENCHMARK_LENGTH     7
#define PLAINTEXT_LENGTH     14
#define PARTIAL_BINARY_SIZE  8
#define BINARY_SIZE          24
#define BINARY_ALIGN         4
#define SALT_SIZE            8
#define SALT_ALIGN           4
#define CIPHERTEXT_LENGTH    48
#define TOTAL_LENGTH         8 + 2 * SALT_SIZE + CIPHERTEXT_LENGTH
#define MIN_KEYS_PER_CRYPT   1
#define MAX_KEYS_PER_CRYPT   1024
static struct fmt_tests tests[] = {
{"", "G3RG3P00!",      {"User", "", "", "6E1EC36D3417CE9E09A4424309F116C4C991948DAEB4ADAD", "ntlm-hash", "1122334455667788"} },
{"$NETLM$1122334455667788$16A7FDFE0CA109B937BFFB041F0E5B2D8B94A97D3FCA1A18", "hiyagerge"},
{"$NETLM$1122334455667788$B3A1B87DBBD4DF3CFA296198DD390C2F4E2E93C5C07B1D8B", "MEDUSAFGDUMP12"},
{"$NETLM$1122334455667788$0836F085B124F33895875FB1951905DD2F85252CC731BB25", "cory21"},
{"$NETLM$1122334455667788$6E1EC36D3417CE9E09A4424309F116C4C991948DAEB4ADAD", "G3RG3P00!"},
{"", "HIYAGERGE",      {"User", "", "", "16A7FDFE0CA109B937BFFB041F0E5B2D8B94A97D3FCA1A18", "ntlm-hash", "1122334455667788"} },
{"", "MEDUSAFGDUMP12", {"User", "", "", "B3A1B87DBBD4DF3CFA296198DD390C2F4E2E93C5C07B1D8B", "ntlm-hash", "1122334455667788"} },
{"", "CORY21",         {"User", "", "", "0836F085B124F33895875FB1951905DD2F85252CC731BB25", "ntlm-hash", "1122334455667788"} },
{"$NETLM$1122334455667788$0836f085b124f33895875fb1951905dd2f85252cc731bb25", "CORY21"},
{NULL}
};
static uchar (*saved_key)[21];
static uchar (*saved_plain)[PLAINTEXT_LENGTH + 1];
static uchar (*output)[PARTIAL_BINARY_SIZE];
static uchar *challenge;
static void init(struct fmt_main *self)
{
omp_autotune(self, OMP_SCALE);
saved_key = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*saved_key));
saved_plain = mem_calloc(self->params.max_keys_per_crypt,
sizeof(*saved_plain));
output = mem_calloc(self->params.max_keys_per_crypt, sizeof(*output));
}
static void done(void)
{
MEM_FREE(output);
MEM_FREE(saved_plain);
MEM_FREE(saved_key);
}
static int valid(char *ciphertext, struct fmt_main *self)
{
char *pos;
if (strncmp(ciphertext, FORMAT_TAG, FORMAT_TAG_LEN)!=0) return 0;
if (strlen(ciphertext) < TOTAL_LENGTH) return 0;
if (ciphertext[23] != '$') return 0;
if (strncmp(&ciphertext[24 + 2 * SALT_SIZE],
"00000000000000000000000000000000", 32) == 0)
return 0; 
for (pos = &ciphertext[24]; atoi16[ARCH_INDEX(*pos)] != 0x7F; pos++)
;
if (!*pos && pos - ciphertext - 24 == CIPHERTEXT_LENGTH)
return 1;
else
return 0;
}
static char *prepare(char *split_fields[10], struct fmt_main *self)
{
char *cp;
char *srv_challenge = split_fields[3];
char *nethashv2     = split_fields[4];
char *cli_challenge = split_fields[5];
if (!strncmp(split_fields[1], FORMAT_TAG, FORMAT_TAG_LEN))
return split_fields[1];
if (!srv_challenge || !nethashv2 || !cli_challenge)
return split_fields[1];
if (strlen(srv_challenge) != CIPHERTEXT_LENGTH)
return split_fields[1];
if (!strncmp(srv_challenge, nethashv2, 48))
return split_fields[1];
if (strlen(nethashv2) > 31) {
if (!strncmp(&nethashv2[32], "0101000000000000", 16))
return split_fields[1];
}
cp = mem_alloc(7+strlen(srv_challenge)+1+strlen(cli_challenge)+1);
sprintf(cp, "%s%s$%s", FORMAT_TAG, cli_challenge, srv_challenge);
if (valid(cp,self)) {
char *cp2 = str_alloc_copy(cp);
MEM_FREE(cp);
return cp2;
}
MEM_FREE(cp);
return split_fields[1];
}
static char *split(char *ciphertext, int index, struct fmt_main *self)
{
static char out[TOTAL_LENGTH + 1];
memset(out, 0, TOTAL_LENGTH + 1);
memcpy(out, ciphertext, TOTAL_LENGTH);
strlwr(&out[FORMAT_TAG_LEN]); 
return out;
}
static void *get_binary(char *ciphertext)
{
static uchar *binary;
int i;
if (!binary) binary = mem_alloc_tiny(BINARY_SIZE, MEM_ALIGN_WORD);
ciphertext+=24;
for (i=0; i<BINARY_SIZE; i++)
{
binary[i] = (atoi16[ARCH_INDEX(ciphertext[i*2])])<<4;
binary[i] |= (atoi16[ARCH_INDEX(ciphertext[i*2+1])]);
}
return binary;
}
inline static void setup_des_key(unsigned char key_56[], DES_key_schedule *ks)
{
DES_cblock key;
key[0] = key_56[0];
key[1] = (key_56[0] << 7) | (key_56[1] >> 1);
key[2] = (key_56[1] << 6) | (key_56[2] >> 2);
key[3] = (key_56[2] << 5) | (key_56[3] >> 3);
key[4] = (key_56[3] << 4) | (key_56[4] >> 4);
key[5] = (key_56[4] << 3) | (key_56[5] >> 5);
key[6] = (key_56[5] << 2) | (key_56[6] >> 6);
key[7] = (key_56[6] << 1);
DES_set_key_unchecked(&key, ks);
}
static int crypt_all(int *pcount, struct db_salt *salt)
{
int count = *pcount;
DES_key_schedule ks;
int i;
#ifdef _OPENMP
#pragma omp parallel for default(none) private(i, ks) shared(count, output, challenge, saved_key)
#endif
for (i = 0; i < count; i++) {
setup_des_key(saved_key[i], &ks);
DES_ecb_encrypt((DES_cblock*)challenge, (DES_cblock*)output[i],
&ks, DES_ENCRYPT);
}
return count;
}
static int cmp_all(void *binary, int count)
{
int index;
for (index = 0; index < count; index++)
if (!memcmp(output[index], binary, PARTIAL_BINARY_SIZE))
return 1;
return 0;
}
static int cmp_one(void *binary, int index)
{
return !memcmp(output[index], binary, PARTIAL_BINARY_SIZE);
}
static int cmp_exact(char *source, int index)
{
DES_key_schedule ks;
uchar binary[BINARY_SIZE];
memset(&saved_key[index][16], 0, 5);
setup_des_key(saved_key[index], &ks);
DES_ecb_encrypt((DES_cblock*)challenge, (DES_cblock*)binary, &ks, DES_ENCRYPT);
setup_des_key(&saved_key[index][7], &ks);
DES_ecb_encrypt((DES_cblock*)challenge, (DES_cblock*)&binary[8], &ks, DES_ENCRYPT);
setup_des_key(&saved_key[index][14], &ks);
DES_ecb_encrypt((DES_cblock*)challenge, (DES_cblock*)&binary[16], &ks, DES_ENCRYPT);
return (!memcmp(binary, get_binary(source), BINARY_SIZE));
}
static void *get_salt(char *ciphertext)
{
static unsigned char *binary_salt;
int i;
if (!binary_salt) binary_salt = mem_alloc_tiny(SALT_SIZE, MEM_ALIGN_WORD);
ciphertext += FORMAT_TAG_LEN;
for (i = 0; i < SALT_SIZE; ++i)
binary_salt[i] = (atoi16[ARCH_INDEX(ciphertext[i*2])] << 4) + atoi16[ARCH_INDEX(ciphertext[i*2+1])];
return (void*)binary_salt;
}
static void set_salt(void *salt)
{
challenge = salt;
}
static void netlm_set_key(char *key, int index)
{
const unsigned char magic[] = {0x4b, 0x47, 0x53, 0x21, 0x40, 0x23, 0x24, 0x25};
DES_key_schedule ks;
strncpy((char *)saved_plain[index], key, sizeof(saved_plain[index]));
saved_plain[index][sizeof(saved_plain[index])-1] = 0;
enc_strupper((char*)saved_plain[index]);
setup_des_key(saved_plain[index], &ks);
DES_ecb_encrypt((DES_cblock*)magic, (DES_cblock*)saved_key[index], &ks, DES_ENCRYPT);
setup_des_key(&saved_plain[index][7], &ks);
DES_ecb_encrypt((DES_cblock*)magic, (DES_cblock*)&saved_key[index][8], &ks, DES_ENCRYPT);
}
static char *get_key(int index)
{
return (char*)saved_plain[index];
}
static int salt_hash(void *salt)
{
return *(uint32_t *)salt & (SALT_HASH_SIZE - 1);
}
#define COMMON_GET_HASH_VAR output
#include "common-get-hash.h"
struct fmt_main fmt_NETLM = {
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
FMT_8_BIT | FMT_TRUNC | FMT_SPLIT_UNIFIES_CASE | FMT_OMP | FMT_OMP_BAD,
{ NULL },
{ FORMAT_TAG },
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
netlm_set_key,
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
