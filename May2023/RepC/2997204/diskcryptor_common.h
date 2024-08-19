#include "common.h"
#include "formats.h"
#include "jumbo.h"
#define FORMAT_TAG              "$diskcryptor$"
#define TAG_LENGTH              (sizeof(FORMAT_TAG) - 1)
struct custom_salt {
uint32_t type;
uint32_t iterations;
uint32_t saltlen;
unsigned char salt[64];
unsigned char header[2048];
};
#define PKCS5_SALT_SIZE         64
#define DISKKEY_SIZE            256
#define MAX_KEY_SIZE            (32*3)
#define PKCS_DERIVE_MAX         (MAX_KEY_SIZE*2)
#define CF_CIPHERS_NUM          7  
#if defined(__GNUC__) && !defined(__MINGW32__)
#define PACKED __attribute__ ((__packed__))
#else
#define PACKED
#pragma pack(push,1)
#endif
struct dc_header {
uint8_t  salt[PKCS5_SALT_SIZE]; 
uint32_t sign;                  
uint32_t hdr_crc;               
uint16_t version;               
uint32_t flags;                 
uint32_t disk_id;               
int32_t  alg_1;                 
uint8_t  key_1[DISKKEY_SIZE];   
int32_t  alg_2;                 
uint8_t  key_2[DISKKEY_SIZE];   
uint64_t stor_off;              
uint64_t use_size;              
uint64_t tmp_size;              
uint8_t  tmp_wp_mode;           
uint8_t  reserved[1422 - 1];
} PACKED;
#if !defined(__GNUC__) || defined(__MINGW32__)
#pragma pack(pop)
#endif
extern struct fmt_tests diskcryptor_tests[];
int diskcryptor_valid(char *ciphertext, struct fmt_main *self);
void *diskcryptor_get_salt(char *ciphertext);
unsigned int diskcryptor_iteration_count(void *salt);
int diskcryptor_decrypt_data(unsigned char *key, struct custom_salt *cur_salt);
