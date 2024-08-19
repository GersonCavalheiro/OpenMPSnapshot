#include "arch.h"
#if !AC_BUILT && !__MIC__
#define HAVE_LIBZ 1 
#endif
#if HAVE_LIBZ
#if FMT_EXTERNS_H
extern struct fmt_main fmt_pkzip;
#elif FMT_REGISTERS_H
john_register_one(&fmt_pkzip);
#else
#include <string.h>
#include <zlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common.h"
#include "misc.h"
#include "formats.h"
#define USE_PKZIP_MAGIC 1
#include "pkzip.h"
#include "pkzip_inffixed.h"  
#include "loader.h"
#define FORMAT_LABEL        "PKZIP"
#define FORMAT_NAME         ""
#define ALGORITHM_NAME      "32/" ARCH_BITS_STR
#define FORMAT_TAG          "$pkzip$"
#define FORMAT_TAG2         "$pkzip2$"
#define FORMAT_TAG_LEN      (sizeof(FORMAT_TAG)-1)
#define FORMAT_TAG2_LEN     (sizeof(FORMAT_TAG2)-1)
#define BENCHMARK_COMMENT   ""
#define BENCHMARK_LENGTH    7
#define PLAINTEXT_LENGTH    63
#define BINARY_SIZE         0
#define BINARY_ALIGN        1
#define SALT_SIZE           (sizeof(PKZ_SALT*))
#define SALT_ALIGN          (sizeof(uint64_t))
#define MIN_KEYS_PER_CRYPT  1
#define MAX_KEYS_PER_CRYPT  64
#ifndef OMP_SCALE
#define OMP_SCALE           32 
#endif
#define PKZIP_USE_MULT_TABLE
#if ARCH_LITTLE_ENDIAN
#define KB1 0
#define KB2 3
#else
#define KB1 3
#define KB2 0
#endif
static struct fmt_tests tests[] = {
{"$pkzip$1*1*2*0*e4*1c5*eda7a8de*0*4c*8*e4*eda7*194883130e4c7419bd735c53dec36f0c4b6de6daefea0f507d67ff7256a49b5ea93ccfd9b12f2ee99053ee0b1c9e1c2b88aeaeb6bd4e60094a1ea118785d4ded6dae94cade41199330f4f11b37cba7cda5d69529bdfa43e2700ba517bd2f7ff4a0d4b3d7f2559690ec044deb818c44844d6dd50adbebf02cec663ae8ebb0dde05d2abc31eaf6de36a2fc19fda65dd6a7e449f669d1f8c75e9daa0a3f7be8feaa43bf84762d6dbcc9424285a93cedfa3a75dadc11e969065f94fe3991bc23c9b09eaa5318aa29fa02e83b6bee26cafec0a5e189242ac9e562c7a5ed673f599cefcd398617*$/pkzip$", "password" },
{"$pkzip$1*1*2*0*e4*1c5*eda7a8de*0*4c*8*e4*eda7*581f798527109cbadfca0b3318435a000be84366caf9723f841a2b13e27c2ed8cdb5628705a98c3fbbfb34552ed498c51a172641bf231f9948bca304a6be2138ab718f6a5b1c513a2fb80c49030ff1a404f7bd04dd47c684317adea4107e5d70ce13edc356c60bebd532418e0855428f9dd582265956e39a0b446a10fd8b7ffb2b4af559351bbd549407381c0d2acc270f3bcaffb275cbe2f628cb09e2978e87cd023d4ccb50caaa92b6c952ba779980d65f59f664dde2451cc456d435188be59301a5df1b1b4fed6b7509196334556c44208a9d7e2d9e237f591d6c9fc467b408bf0aaa*$/pkzip$", "password" },
{"$pkzip$1*2*2*0*e4*1c5*eda7a8de*0*47*8*e4*4bb6*436c9ffa4328870f6272349b591095e1b1126420c3041744650282bc4f575d0d4a5fc5fb34724e6a1cde742192387b9ed749ab5c72cd6bb0206f102e9216538f095fb773661cfde82c2e2a619332998124648bf4cd0da56279f0c297567d9b5d684125ee92920dd513fd18c27afba2a9633614f75d8f8b9a14095e3fafe8165330871287222e6681dd9c0f830cf5d464457b257d0900eed29107fad8af3ac4f87cf5af5183ff0516ccd9aeac1186006c8d11b18742dfb526aadbf2906772fbfe8fb18798967fd397a724d59f6fcd4c32736550986d227a6b447ef70585c049a1a4d7bf25*$/pkzip$", "password" },
{"$pkzip$1*2*2*0*e4*1c5*eda7a8de*0*47*8*e4*4bb6*436c9ffa4328870f6272349b591095e1b1126420c3041744650282bc4f575d0d4a5fc5fb34724e6a1cde742192387b9ed749ab5c72cd6bb0206f102e9216538f095fb773661cfde82c2e2a619332998124648bf4cd0da56279f0c297567d9b5d684125ee92920dd513fd18c27afba2a9633614f75d8f8b9a14095e3fafe8165330871287222e6681dd9c0f830cf5d464457b257d0900eed29107fad8af3ac4f87cf5af5183ff0516ccd9aeac1186006c8d11b18742dfb526aadbf2906772fbfe8fb18798967fd397a724d59f6fcd4c32736550986d227a6b447ef70585c049a1a4d7bf25*$/pkzip$", "password"},
{"$pkzip$3*1*1*0*8*24*4001*8986ec4d693e86c1a42c1bd2e6a994cb0b98507a6ec937fe0a41681c02fe52c61e3cc046*1*0*8*24*4003*a087adcda58de2e14e73db0043a4ff0ed3acc6a9aee3985d7cb81d5ddb32b840ea2057d9*2*0*e4*1c5*eda7a8de*0*4c*8*e4*eda7*89a792af804bf38e31fdccc8919a75ab6eb75d1fd6e7ecefa3c5b9c78c3d50d656f42e582af95882a38168a8493b2de5031bb8b39797463cb4769a955a2ba72abe48ee75b103f93ef9984ae740559b9bd84cf848d693d86acabd84749853675fb1a79edd747867ef52f4ee82435af332d43f0d0bb056c49384d740523fa75b86a6d29a138da90a8de31dbfa89f2f6b0550c2b47c43d907395904453ddf42a665b5f7662de170986f89d46d944b519e1db9d13d4254a6b0a5ac02b3cfdd468d7a4965e4af05699a920e6f3ddcedb57d956a6b2754835b14e174070ba6aec4882d581c9f30*$/pkzip$", "3!files"},
{"$pkzip$1*1*2*0*163*2b5*cd154083*0*26*8*163*cd15*d6b094794b40116a8b387c10159225d776f815b178186e51faf16fa981fddbffdfa22f6c6f32d2f81dab35e141f2899841991f3cb8d53f8ee1f1d85657f7c7a82ebb2d63182803c6beee00e0bf6c72edeeb1b00dc9f07f917bb8544cc0e96ca01503cd0fb6632c296cebe3fb9b64543925daae6b7ea95cfd27c42f6f3465e0ab2c812b9aeeb15209ce3b691f27ea43a7a77b89c2387e31c4775866a044b6da783af8ddb72784ccaff4d9a246db96484e865ea208ade290b0131b4d2dd21f172693e6b5c90f2eb9b67572b55874b6d3a78763212b248629e744c07871a6054e24ef74b6d779e44970e1619df223b4e5a72a189bef40682b62be6fb7f65e087ca6ee19d1ebfc259fa7e3d98f3cb99347689f8360294352accffb146edafa9e91afba1f119f95145738ac366b332743d4ff40d49fac42b8758c43b0af5b60b8a1c63338359ffbff432774f2c92de3f8c49bd4611e134db98e6a3f2cfb148d2b20f75abab6*$/pkzip$", "passwort"},
{"$pkzip$1*1*2*0*163*2b6*46abc149*0*28*8*163*46ab*0f539b23b761a347a329f362f7f1f0249515f000404c77ec0b0ffe06f29140e8fa3e8e5a6354e57f3252fae3d744212d4d425dc44389dd4450aa9a4f2f3c072bee39d6ac6662620812978f7ab166c66e1acb703602707ab2da96bb28033485ec192389f213e48eda8fc7d9dad1965b097fafebfda6703117db90e0295db9a653058cb28215c3245e6e0f6ad321065bf7b8cc5f66f6f2636e0d02ea35a6ba64bbf0191c308098fd836e278abbce7f10c3360a0a682663f59f92d9c2dcfc87cde2aae27ea18a14d2e4a0752b6b51e7a5c4c8c2bab88f4fb0aba27fb20e448655021bb3ac63752fdb01e6b7c99f9223f9e15d71eb1bd8e323f522fc3da467ff0aae1aa17824085d5d6f1cdfc9c7c689cd7cb057005d94ba691f388484cfb842c8775baac220a5490ed945c8b0414dbfc4589254b856aade49f1aa386db86e9fc87e6475b452bd72c5e2122df239f8c2fd462ca54c1a5bddac36918c5f5cf0cc94aa6ee820*$/pkzip$", "Credit11"},
{"$pkzip$1*1*2*0*163*2b6*46abc149*0*26*8*163*46ab*7ea9a6b07ddc9419439311702b4800e7e1f620b0ab8535c5aa3b14287063557b176cf87a800b8ee496643c0b54a77684929cc160869db4443edc44338294458f1b6c8f056abb0fa27a5e5099e19a07735ff73dc91c6b20b05c023b3ef019529f6f67584343ac6d86fa3d12113f3d374b047efe90e2a325c0901598f31f7fb2a31a615c51ea8435a97d07e0bd4d4afbd228231dbc5e60bf1116ce49d6ce2547b63a1b057f286401acb7c21afbb673f3e26bc1b2114ab0b581f039c2739c7dd0af92c986fc4831b6c294783f1abb0765cf754eada132df751cf94cad7f29bb2fec0c7c47a7177dea82644fc17b455ba2b4ded6d9a24e268fcc4545cae73b14ceca1b429d74d1ebb6947274d9b0dcfb2e1ac6f6b7cd2be8f6141c3295c0dbe25b65ff89feb62cb24bd5be33853b88b8ac839fdd295f71e17a7ae1f054e27ba5e60ca03c6601b85c3055601ce41a33127938440600aaa16cfdd31afaa909fd80afc8690aaf*$/pkzip$", "7J0rdan!!"},
{"$pkzip$1*2*2*0*6b*73*8e687a5b*0*46*8*6b*0d9d*636fedc7a78a7f80cda8542441e71092d87d13da94c93848c230ea43fab5978759e506110b77bd4bc10c95bc909598a10adfd4febc0d42f3cd31e4fec848d6f49ab24bb915cf939fb1ce09326378bb8ecafde7d3fe06b6013628a779e017be0f0ad278a5b04e41807ae9fc*$/pkzip$", "c00rslit3!"},
{"$pkzip2$3*2*1*2*8*c0*7224*72f6*6195f9f3401076b22f006105c4323f7ac8bb8ebf8d570dc9c7f13ddacd8f071783f6bef08e09ce4f749af00178e56bc948ada1953a0263c706fd39e96bb46731f827a764c9d55945a89b952f0503747703d40ed4748a8e5c31cb7024366d0ef2b0eb4232e250d343416c12c7cbc15d41e01e986857d320fb6a2d23f4c44201c808be107912dbfe4586e3bf2c966d926073078b92a2a91568081daae85cbcddec75692485d0e89994634c71090271ac7b4a874ede424dafe1de795075d2916eae*1*6*8*c0*26ee*461b*944bebb405b5eab4322a9ce6f7030ace3d8ec776b0a989752cf29569acbdd1fb3f5bd5fe7e4775d71f9ba728bf6c17aad1516f3aebf096c26f0c40e19a042809074caa5ae22f06c7dcd1d8e3334243bca723d20875bd80c54944712562c4ff5fdb25be5f4eed04f75f79584bfd28f8b786dd82fd0ffc760893dac4025f301c2802b79b3cb6bbdf565ceb3190849afdf1f17688b8a65df7bc53bc83b01a15c375e34970ae080307638b763fb10783b18b5dec78d8dfac58f49e3c3be62d6d54f9*2*0*2a*1e*4a204eab*ce8*2c*0*2a*4a20*7235*6b6e1a8de47449a77e6f0d126b217d6b2b72227c0885f7dc10a2fb3e7cb0e611c5c219a78f98a9069f30*$/pkzip2$", "123456"},
{"$pkzip$1*1*2*0*14*6*775f54d8*0*47*8*14*8cd0*11b75efed56a5795f07c509268a88b4a6ff362ef*$/pkzip$", "test"},
{NULL}
};
static char (*saved_key)[PLAINTEXT_LENGTH + 1];
static u32  *K12;
static PKZ_SALT *salt;
static u8 *chk;
static int dirty=1;
#if USE_PKZIP_MAGIC
static ZIP_SIGS SIGS[256];
#endif
#ifdef PKZIP_USE_MULT_TABLE
static u8 mult_tab[16384];
#define PKZ_MULT(b,w) b^mult_tab[(u16)(w.u)>>2]
#else
inline u8 PKZ_MULT(u8 b, MY_WORD w) {u16 t = w.u|2; return b ^ (u8)(((u16)(t*(t^1))>>8)); }
#endif
extern struct fmt_main fmt_pkzip;
static const char *ValidateZipContents(FILE *in, long offset, u32 offex, int len, u32 crc);
static int valid(char *ciphertext, struct fmt_main *self)
{
c8 *p, *cp, *cpkeep;
int cnt, ret=0;
u64 data_len;
u32 crc;
FILE *in;
const char *sFailStr;
long offset;
u32 offex;
int type;
u64 complen = 0;
int type2 = 0;
if (strncmp(ciphertext, FORMAT_TAG, FORMAT_TAG_LEN)) {
if (!strncmp(ciphertext, FORMAT_TAG2, FORMAT_TAG2_LEN))
type2 = 1;
else
return ret;
}
cpkeep = xstrdup(ciphertext);
cp = cpkeep;
p = &cp[FORMAT_TAG_LEN];
if (type2)
++p;
if ((cp = strtokm(p, "*")) == NULL || !cp[0] || !ishexlc_oddOK(cp)) {
sFailStr = "Out of data, reading count of hashes field";
goto Bail;
}
sscanf(cp, "%x", &cnt);
if (cnt < 1 || cnt > MAX_PKZ_FILES) {
sFailStr = "Count of hashes field out of range";
goto Bail;
}
if ((cp = strtokm(NULL, "*")) == NULL || cp[0] < '0' || cp[0] > '2' || cp[1]) {
sFailStr = "Number of valid hash bytes empty or out of range";
goto Bail;
}
while (cnt--) {
if ((cp = strtokm(NULL, "*")) == NULL || cp[0]<'1' || cp[0]>'3' || cp[1]) {
sFailStr = "Invalid data enumeration type";
goto Bail;
}
type = cp[0] - '0';
if ((cp = strtokm(NULL, "*")) == NULL || !cp[0] || !ishexlc_oddOK(cp)) {
sFailStr = "Invalid type enumeration";
goto Bail;
}
if (type > 1) {
if ((cp = strtokm(NULL, "*")) == NULL || !cp[0] || !ishexlc_oddOK(cp)) {
sFailStr = "Invalid compressed length";
goto Bail;
}
sscanf(cp, "%"PRIx64, &complen);
if ((cp = strtokm(NULL, "*")) == NULL || !cp[0] || !ishexlc_oddOK(cp)) {
sFailStr = "Invalid data length value";
goto Bail;
}
if ((cp = strtokm(NULL, "*")) == NULL || !cp[0] || !ishexlc_oddOK(cp)) {
sFailStr = "Invalid CRC value";
goto Bail;
}
sscanf(cp, "%x", &crc);
if ((cp = strtokm(NULL, "*")) == NULL || !cp[0] || !ishexlc_oddOK(cp)) {
sFailStr = "Invalid offset length";
goto Bail;
}
sscanf(cp, "%lx", &offset);
if ((cp = strtokm(NULL, "*")) == NULL || !cp[0] || !ishexlc_oddOK(cp)) {
sFailStr = "Invalid offset length";
goto Bail;
}
sscanf(cp, "%x", &offex);
}
if ((cp = strtokm(NULL, "*")) == NULL || (cp[0] != '0' && cp[0] != '8') || cp[1]) {
sFailStr = "Compression type enumeration";
goto Bail;
}
if ((cp = strtokm(NULL, "*")) == NULL || !cp[0] || !ishexlc_oddOK(cp)) {
sFailStr = "Invalid data length value";
goto Bail;
}
sscanf(cp, "%"PRIx64, &data_len);
if ((cp = strtokm(NULL, "*")) == NULL || !ishexlc(cp) || strlen(cp) != 4) {
sFailStr = "invalid checksum value";
goto Bail;
}
if (type2) {
if ((cp = strtokm(NULL, "*")) == NULL || !ishexlc(cp) || strlen(cp) != 4) {
sFailStr = "invalid checksum2 value";
goto Bail;}
}
if ((cp = strtokm(NULL, "*")) == NULL) goto Bail;
if (type > 1) {
if (type == 3) {
if (strlen(cp) != data_len) {
sFailStr = "invalid checksum value";
goto Bail;
}
in = fopen(cp, "rb"); 
if (!in) {
if (!ldr_in_pot)
fprintf(stderr, "Error loading a pkzip hash line. The ZIP file '%s' could NOT be found\n", cp);
return 0;
}
sFailStr = ValidateZipContents(in, offset, offex, complen, crc);
if (*sFailStr) {
fprintf(stderr, "pkzip validation failed [%s] Hash is %s\n", sFailStr, ciphertext);
fclose(in);
return 0;
}
fseek(in, offset+offex, SEEK_SET);
if (complen < 16*1024) {
void *tbuf = mem_alloc(complen);
if (fread(tbuf, 1, complen, in) != complen) {
MEM_FREE(tbuf);
fclose(in);
return 0;
}
data_len = complen;
MEM_FREE(tbuf);
}
fclose(in);
} else {
if (complen != data_len) {
sFailStr = "length of full data does not match the salt len";
goto Bail;
}
if (!ishexlc(cp) || strlen(cp) != data_len<<1) {
sFailStr = "invalid inline data";
goto Bail;
}
}
} else {
if (!ishexlc(cp) || strlen(cp) != data_len<<1) {
sFailStr = "invalid partial data";
goto Bail;
}
}
}
if ((cp = strtokm(NULL, "*")) == NULL) goto Bail;
if (strtokm(NULL, "") != NULL) goto Bail;
if (type2) ret = !strcmp(cp, "$/pkzip2$");
else       ret = !strcmp(cp, "$/pkzip$");
Bail:;
#ifdef ZIP_DEBUG
if (!ret) fprintf(stderr, "pkzip validation failed [%s]  Hash is %.64s\n", sFailStr, ciphertext);
#endif
MEM_FREE(cpkeep);
return ret;
}
static const char *ValidateZipContents(FILE *fp, long offset, u32 offex, int _len, u32 _crc)
{
u32 id;
u16 version, flags, method, modtm, moddt, namelen, exlen;
u32 crc, complen, uncomplen;
if (fseek(fp, offset, SEEK_SET) != 0)
return "Not able to seek to specified offset in the .zip file, to read the zip blob data.";
id = fget32LE(fp);
if (id != 0x04034b50U)
return "Compressed zip file offset does not point to start of zip blob";
version = fget16LE(fp);
flags = fget16LE(fp);
method = fget16LE(fp);
modtm = fget16LE(fp);
moddt = fget16LE(fp);
crc = fget32LE(fp);
complen = fget32LE(fp);
uncomplen = fget32LE(fp);
namelen = fget16LE(fp);
exlen = fget16LE(fp);
(void)uncomplen;
(void)modtm;
(void)moddt;
if (_crc == crc && _len == complen &&  (0x14 == version || 0xA == version) && (flags & 1) && (method == 8 || method == 0) && offex==30+namelen+exlen)
return "";
return "We could NOT find the internal zip data in this ZIP file";
}
static u8 *buf_copy (char *p, int len)
{
u8 *op = mem_alloc_tiny(len, MEM_ALIGN_NONE);
memcpy(op, p, len);
return op;
}
static void init(struct fmt_main *self)
{
#ifdef PKZIP_USE_MULT_TABLE
unsigned short n=0;
#endif
omp_autotune(self, OMP_SCALE);
saved_key = mem_calloc(sizeof(*saved_key), self->params.max_keys_per_crypt);
K12 = mem_calloc(sizeof(*K12) * 3, self->params.max_keys_per_crypt);
chk = mem_calloc(sizeof(*chk), self->params.max_keys_per_crypt);
#ifdef PKZIP_USE_MULT_TABLE
for (n = 0; n < 16384; n++)
mult_tab[n] = (((unsigned)(n*4+3) * (n*4+2)) >> 8) & 0xff;
#endif
#if USE_PKZIP_MAGIC
SIGS[1].magic_signature[0] = (u8*)str_alloc_copy("\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1");
SIGS[1].magic_sig_len[0] = 8;
SIGS[1].magic_signature[1] = buf_copy("\x50\x4B\x03\x04\x14\x00\x06\x00\x08", 10);  
SIGS[1].magic_sig_len[1] = 9;
SIGS[1].magic_signature[2] = buf_copy("\x09\x04\x06\x00\x00\x00\x10\x00\xF6\x05\x5C\x00", 13); 
SIGS[1].magic_sig_len[2] = 12;
SIGS[1].magic_signature[3] = buf_copy("\x09\x02\x06\x00\x00\x00\x10\x00\xB9\x04\x5C\x00", 13); 
SIGS[1].magic_sig_len[3] = 12;
SIGS[1].magic_signature[4] = buf_copy("\x50\x4B\x03\x04\x14\x00\x00\x00\x00\x00", 11); 
SIGS[1].magic_sig_len[4] = 10;
SIGS[1].magic_signature[5] = buf_copy("\x31\xBE\x00\x00\x00\xAB\x00\x00", 9); 
SIGS[1].magic_sig_len[5] = 8;
SIGS[1].magic_signature[6] = (u8*)str_alloc_copy("\x12\x34\x56\x78\x90\xFF"); 
SIGS[1].magic_sig_len[6] = 6;
SIGS[1].magic_signature[7] = (u8*)str_alloc_copy("\x7F\xFE\x34\x0A");  
SIGS[1].magic_sig_len[7] = 4;
SIGS[1].magic_count = 8;
SIGS[1].max_len = 12;
SIGS[2].magic_signature[0] = (u8*)str_alloc_copy("MZ");
SIGS[2].magic_sig_len[0] = 2;
SIGS[2].magic_count = 1;
SIGS[2].max_len = 2;
SIGS[3].magic_signature[0] = (u8*)str_alloc_copy("\x50\x4B\x03\x04");
SIGS[3].magic_sig_len[0] = 4;
SIGS[3].magic_count = 1;
SIGS[3].max_len = 4;
SIGS[4].magic_signature[0] = (u8*)str_alloc_copy("BM");
SIGS[4].magic_sig_len[0] = 2;
SIGS[4].magic_count = 1;
SIGS[4].max_len = 2;
SIGS[5].magic_signature[0] = (u8*)str_alloc_copy("GIF87a");
SIGS[5].magic_sig_len[0] = 6;
SIGS[5].magic_signature[1] = (u8*)str_alloc_copy("GIF89a");
SIGS[5].magic_sig_len[1] = 6;
SIGS[5].magic_count = 2;
SIGS[5].max_len = 6;
SIGS[6].magic_signature[0] = (u8*)str_alloc_copy("%PDF");
SIGS[6].magic_sig_len[0] = 4;
SIGS[6].magic_count = 1;
SIGS[6].max_len = 4;
SIGS[7].magic_signature[0] = (u8*)str_alloc_copy("\x1F\x8B\x08");
SIGS[7].magic_sig_len[0] = 3;
SIGS[7].magic_count = 1;
SIGS[7].max_len = 3;
SIGS[8].magic_signature[0] = (u8*)str_alloc_copy("BZh");
SIGS[8].magic_sig_len[0] = 3;
SIGS[8].magic_signature[1] = (u8*)str_alloc_copy("BZ0");
SIGS[8].magic_sig_len[1] = 3;
SIGS[8].magic_count = 2;
SIGS[8].max_len = 3;
SIGS[9].magic_signature[0] = (u8*)str_alloc_copy("FLV\x01");
SIGS[9].magic_sig_len[0] = 4;
SIGS[9].magic_count = 1;
SIGS[9].max_len = 4;
SIGS[10].magic_signature[0] = (u8*)str_alloc_copy("FWS");
SIGS[10].magic_sig_len[0] = 3;
SIGS[10].magic_signature[1] = (u8*)str_alloc_copy("CWS");
SIGS[10].magic_sig_len[1] = 3;
SIGS[10].magic_signature[2] = (u8*)str_alloc_copy("ZWS");
SIGS[10].magic_sig_len[2] = 3;
SIGS[10].magic_count = 3;
SIGS[10].max_len = 3;
SIGS[11].magic_signature[0] = (u8*)str_alloc_copy("ID3");
SIGS[11].magic_sig_len[0] = 3;
SIGS[11].magic_count = 1;
SIGS[11].max_len = 3;
SIGS[12].magic_signature[0] = (u8*)str_alloc_copy("!BDN");
SIGS[12].magic_sig_len[0] = 4;
SIGS[12].magic_count = 1;
SIGS[12].max_len = 4;
SIGS[255].max_len = 64;
#endif
}
static void done(void)
{
MEM_FREE(chk);
MEM_FREE(K12);
MEM_FREE(saved_key);
}
static void set_salt(void *_salt)
{
int i;
int need_fixup = 0;
long tot_len = 0;
salt = *((PKZ_SALT**)_salt);
for (i = 0; i < MAX_PKZ_FILES; i++) {
if (!salt->H[i].h) {
need_fixup = 1;
break;
}
}
if (need_fixup) {
for (i = 0; i < MAX_PKZ_FILES; i++) {
salt->H[i].h = &salt->zip_data[i + tot_len];
tot_len += salt->H[i].datlen;
}
}
}
static void *get_salt(char *ciphertext)
{
static union alignment {
unsigned char c[8];
uint64_t a[1];	
} a;
unsigned char *salt_p = a.c;
PKZ_SALT *salt, *psalt;
long offset=0;
char *H[MAX_PKZ_FILES] = { 0 };
long ex_len[MAX_PKZ_FILES] = { 0 };
long tot_len;
u32 offex;
size_t i, j;
c8 *p, *cp, *cpalloc = (char*)mem_alloc(strlen(ciphertext)+1);
int type2 = 0;
salt = mem_calloc(1, sizeof(PKZ_SALT));
cp = cpalloc;
strcpy(cp, ciphertext);
if (!strncmp(cp, FORMAT_TAG, FORMAT_TAG_LEN))
p = &cp[FORMAT_TAG_LEN];
else {
p = &cp[FORMAT_TAG2_LEN];
type2 = 1;
}
cp = strtokm(p, "*");
sscanf(cp, "%x", &(salt->cnt));
cp = strtokm(NULL, "*");
sscanf(cp, "%x", &(salt->chk_bytes));
for (i = 0; i < salt->cnt; ++i) {
int data_enum;
salt->H[i].type = type2 ? 2 : 1;
cp = strtokm(NULL, "*");
data_enum = *cp - '0';
cp = strtokm(NULL, "*");
#if USE_PKZIP_MAGIC
{
unsigned jnk;
sscanf(cp, "%x", &jnk);
salt->H[i].magic = (unsigned char)jnk;
}
salt->H[i].pSig = &SIGS[salt->H[i].magic];
#endif
if (data_enum > 1) {
cp = strtokm(NULL, "*");
sscanf(cp, "%"PRIx64, &(salt->compLen));
cp = strtokm(NULL, "*");
sscanf(cp, "%"PRIx64, &(salt->deCompLen));
cp = strtokm(NULL, "*");
sscanf(cp, "%x", &(salt->crc32));
cp = strtokm(NULL, "*");
sscanf(cp, "%lx", &offset);
cp = strtokm(NULL, "*");
sscanf(cp, "%x", &offex);
}
cp = strtokm(NULL, "*");
sscanf(cp, "%x", &(salt->H[i].compType));
cp = strtokm(NULL, "*");
sscanf(cp, "%"PRIx64, &(salt->H[i].datlen));
cp = strtokm(NULL, "*");
for (j = 0; j < 4; ++j) {
salt->H[i].c <<= 4;
salt->H[i].c |= atoi16[ARCH_INDEX(cp[j])];
}
if (type2) {
cp = strtokm(NULL, "*");
for (j = 0; j < 4; ++j) {
salt->H[i].c2 <<= 4;
salt->H[i].c2 |= atoi16[ARCH_INDEX(cp[j])];
}
}
cp = strtokm(NULL, "*");
if (data_enum > 1) {
if (data_enum == 3) {
FILE *fp;
fp = fopen(cp, "rb");
if (!fp) {
fprintf(stderr, "Error opening file for pkzip data:  %s\n", cp);
MEM_FREE(cpalloc);
return 0;
}
fseek(fp, offset+offex, SEEK_SET);
if (salt->compLen < 16*1024) {
ex_len[i] = salt->compLen;
H[i] = mem_alloc(salt->compLen);
if (fread(H[i], 1, salt->compLen, fp) != salt->compLen) {
fprintf(stderr, "Error reading zip file for pkzip data:  %s\n", cp);
fclose(fp);
MEM_FREE(cpalloc);
return 0;
}
fclose(fp);
salt->H[i].datlen = salt->compLen;
}
else {
strnzcpy(salt->fname, (const char *)cp, sizeof(salt->fname));
salt->offset = offset+offex;
ex_len[i] = 384;
H[i] = mem_alloc(384);
if (fread(H[i], 1, 384, fp) != 384) {
fprintf(stderr, "Error reading zip file for pkzip data:  %s\n", cp);
fclose(fp);
MEM_FREE(cpalloc);
return 0;
}
fclose(fp);
salt->H[i].datlen = 384;
}
} else {
ex_len[i] = salt->compLen;
H[i] = mem_alloc(salt->compLen);
for (j = 0; j < salt->H[i].datlen; ++j)
H[i][j] = (atoi16[ARCH_INDEX(cp[j*2])]<<4) + atoi16[ARCH_INDEX(cp[j*2+1])];
}
salt->compType = salt->H[i].compType;
salt->H[i].full_zip = 1;
salt->full_zip_idx = i;
} else {
ex_len[i] = salt->H[i].datlen;
H[i] = mem_alloc(salt->H[i].datlen);
for (j = 0; j < salt->H[i].datlen; ++j)
H[i][j] = (atoi16[ARCH_INDEX(cp[j*2])]<<4) + atoi16[ARCH_INDEX(cp[j*2+1])];
}
}
MEM_FREE(cpalloc);
j = 0;
for (i = 0; i < salt->cnt; ++i) {
if (salt->H[i].compType == 8) {
if (salt->cnt == 1 && salt->chk_bytes == 1)
j += 10;
else
break;
}
j += 1;
}
if (j >= 20)
j = 0;
if (j && salt->chk_bytes == 2 && salt->cnt > 1)
j = 0;  
if (j && salt->chk_bytes == 1 && salt->cnt == 3)
j = 0;  
if (!j) {
for (i = 0; i < salt->cnt; ++i)
salt->H[i].magic = 0;	
}
tot_len = 0;
for (i = 0; i < salt->cnt; i++)
tot_len += ex_len[i];
tot_len += salt->cnt - 1;
psalt = mem_calloc(1, sizeof(PKZ_SALT) + tot_len);
memcpy(psalt, salt, sizeof(*salt));
tot_len = 0;
for (i = 0; i < salt->cnt; i++) {
memcpy(psalt->zip_data + i + tot_len, H[i], ex_len[i]);
tot_len += ex_len[i];
MEM_FREE(H[i]);
}
tot_len += salt->cnt - 1;
MEM_FREE(salt);
psalt->dsalt.salt_alloc_needs_free = 1;  
memcpy(salt_p, &psalt, sizeof(psalt));
psalt->dsalt.salt_cmp_offset = SALT_CMP_OFF(PKZ_SALT, cnt);
psalt->dsalt.salt_cmp_size = SALT_CMP_SIZE(PKZ_SALT, cnt, zip_data, tot_len);
return salt_p;
}
static void set_key(char *key, int index)
{
strnzcpy(saved_key[index], key, sizeof(*saved_key));
dirty = 1;
}
static char *get_key(int index)
{
return saved_key[index];
}
static int cmp_one(void *binary, int idx)
{
return chk[idx] == 1;
}
static int cmp_all(void *binary, int count)
{
int i,j;
for (i=j=0; i<count; ++i)
j+=chk[i]; 
return j;
}
static int get_next_decrypted_block(u8 *in, int sizeof_n, FILE *fp, u32 *inp_used, MY_WORD *pkey0, MY_WORD *pkey1, MY_WORD *pkey2)
{
u32 new_bytes = sizeof_n, k;
u8 C;
if (*inp_used >= salt->compLen)
return 0;
if (*inp_used + new_bytes > salt->compLen)
new_bytes = salt->compLen - *inp_used;
*inp_used += new_bytes;
if (fread(in, 1, new_bytes, fp) != new_bytes)
return 0;
for (k = 0; k < new_bytes; ++k) {
C = PKZ_MULT(in[k],(*pkey2));
pkey0->u = jtr_crc32 (pkey0->u, C);
pkey1->u = (pkey1->u + pkey0->c[KB1]) * 134775813 + 1;
pkey2->u = jtr_crc32 (pkey2->u, pkey1->c[KB2]);
in[k] = C;
}
return new_bytes;
}
#define CHUNK (64*1024)
static int cmp_exact_loadfile(int index)
{
int ret;
u32 have, k;
z_stream strm;
unsigned char in[CHUNK];
unsigned char out[CHUNK];
FILE *fp;
MY_WORD key0, key1, key2;
u8 *b, C;
u32 inp_used, decomp_len=0;
u32 crc = 0xFFFFFFFF;
fp = fopen(salt->fname, "rb");
if (!fp) {
fprintf(stderr, "\nERROR, the zip file: %s has been removed.\nWe are a possible password has been found, but FULL validation can not be done!\n", salt->fname);
return 1;
}
if (fseek(fp, salt->offset, SEEK_SET)) {
fprintf(stderr, "\nERROR, the zip file: %s fseek() failed.\nWe are a possible password has been found, but FULL validation can not be done!\n", salt->fname);
fclose(fp);
return 1;
}
key0.u = K12[index*3], key1.u = K12[index*3+1], key2.u = K12[index*3+2];
k=12;
if (fread(in, 1, 12, fp) != 12) {
fprintf(stderr, "\nERROR, the zip file: %s fread() failed.\nWe are a possible password has been found, but FULL validation can not be done!\n", salt->fname);
fclose(fp);
return 1;
}
b = salt->H[salt->full_zip_idx].h;
do {
C = PKZ_MULT(*b++,key2);
key0.u = jtr_crc32 (key0.u, C);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
}
while(--k);
inp_used = 12;
if (salt->H[salt->full_zip_idx].compType == 0) {
int avail_in;
crc = 0xFFFFFFFF;
avail_in = get_next_decrypted_block(in, CHUNK, fp, &inp_used, &key0, &key1, &key2);
while (avail_in) {
for (k = 0; k < avail_in; ++k)
crc = jtr_crc32(crc,in[k]);
avail_in = get_next_decrypted_block(in, CHUNK, fp, &inp_used, &key0, &key1, &key2);
}
fclose(fp);
return ~crc == salt->crc32;
}
strm.zalloc = Z_NULL;
strm.zfree = Z_NULL;
strm.opaque = Z_NULL;
strm.avail_in = 0;
strm.next_in = Z_NULL;
ret = inflateInit2(&strm, -15);
if (ret != Z_OK) 
perror("Error, initializing the libz inflateInit2() system\n");
do {
strm.avail_in = get_next_decrypted_block(in, CHUNK, fp, &inp_used, &key0, &key1, &key2);
if (ferror(fp)) {
inflateEnd(&strm);
fclose(fp);
fprintf(stderr, "\nERROR, the zip file: %s fread() failed.\nWe are a possible password has been found, but FULL validation can not be done!\n", salt->fname);
return 1;
}
if (strm.avail_in == 0)
break;
strm.next_in = in;
do {
strm.avail_out = CHUNK;
strm.next_out = out;
ret = inflate(&strm, Z_NO_FLUSH);
switch (ret) {
case Z_NEED_DICT:
case Z_DATA_ERROR:
case Z_MEM_ERROR:
inflateEnd(&strm);
fclose(fp);
return 0;
}
have = CHUNK - strm.avail_out;
for (k = 0; k < have; ++k)
crc = jtr_crc32(crc,out[k]);
decomp_len += have;
} while (strm.avail_out == 0);
} while (ret != Z_STREAM_END);
inflateEnd(&strm);
fclose(fp);
return ret == Z_STREAM_END && inp_used == salt->compLen && decomp_len == salt->deCompLen && salt->crc32 == ~crc;
}
static int cmp_exact(char *source, int index)
{
const u8 *b;
u8 C, *decompBuf, *decrBuf, *B;
u32 k, crc;
MY_WORD key0, key1, key2;
z_stream strm;
int ret;
if (salt->H[salt->full_zip_idx].full_zip == 0)
return 1;
#ifdef ZIP_DEBUG
fprintf(stderr, "FULL zip test being done. (pass=%s)\n", saved_key[index]);
#endif
if (salt->fname[0] == 0) {
decrBuf = mem_alloc(salt->compLen-12);
key0.u = K12[index*3], key1.u = K12[index*3+1], key2.u = K12[index*3+2];
b = salt->H[salt->full_zip_idx].h;
k=12;
do {
C = PKZ_MULT(*b++,key2);
key0.u = jtr_crc32 (key0.u, C);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
}
while(--k);
B = decrBuf;
k = salt->compLen-12;
do {
C = PKZ_MULT(*b++,key2);
key0.u = jtr_crc32 (key0.u, C);
*B++ = C;
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
} while (--k);
if (salt->H[salt->full_zip_idx].compType == 0) {
crc = 0xFFFFFFFF;
for (k = 0; k < salt->compLen-12; ++k)
crc = jtr_crc32(crc,decrBuf[k]);
MEM_FREE(decrBuf);
return ~crc == salt->crc32;
}
strm.zalloc = Z_NULL;
strm.zfree = Z_NULL;
strm.opaque = Z_NULL;
strm.next_in = Z_NULL;
strm.avail_in = 0;
ret = inflateInit2(&strm, -15); 
if (ret != Z_OK)
perror("Error, initializing the libz inflateInit2() system\n");
decompBuf = mem_alloc(salt->deCompLen);
strm.next_in = decrBuf;
strm.avail_in = salt->compLen-12;
strm.avail_out = salt->deCompLen;
strm.next_out = decompBuf;
ret = inflate(&strm, Z_SYNC_FLUSH);
inflateEnd(&strm);
if (ret != Z_STREAM_END || strm.total_out != salt->deCompLen) {
MEM_FREE(decompBuf);
MEM_FREE(decrBuf);
return 0;
}
crc = 0xFFFFFFFF;
for (k = 0; k < strm.total_out; ++k)
crc = jtr_crc32(crc,decompBuf[k]);
MEM_FREE(decompBuf);
MEM_FREE(decrBuf);
return ~crc == salt->crc32;
}
return cmp_exact_loadfile(index);
}
#if USE_PKZIP_MAGIC
const char exBytesUTF8[64] = {
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,4,4,4,4,5,5,5,5
};
static int isLegalUTF8_char(const u8 *source, int length)
{
u8 a;
int len;
const u8 *srcptr;
if (*source < 0xC0)
return 1;
len = exBytesUTF8[*source&0x3f];
srcptr = source+len;
if (len+1 > length)
return -1;
switch (len) {
default: return -1;
case 4: if ((a = (*--srcptr)) < 0x80 || a > 0xBF) return -1;
case 3: if ((a = (*--srcptr)) < 0x80 || a > 0xBF) return -1;
case 2: if ((a = (*--srcptr)) < 0x80 || a > 0xBF) return -1;
switch (*source) {
case 0xE0: if (a < 0xA0) return -1;
break;
case 0xED: if (a > 0x9F) return -1;
break;
case 0xF0: if (a < 0x90) return -1;
break;
case 0xF4: if (a > 0x8F) return -1;
}
case 1: if (*source >= 0x80 && *source < 0xC2) return -1;
}
if (*source > 0xF4) return -1;
return len+1;
}
static int validate_ascii(const u8 *out, int inplen)
{
int i;
int unicode=0;
for (i = 0; i < inplen-1; ++i) {
if (out[i] > 0x7E) {
if (unicode)
return 0; 
if (out[i] > 0xC0) {
int len;
if (i > inplen-4)
return 1;
len = isLegalUTF8_char(&out[i], 5);
if (len < 0) return 0;
i += (len-1);
}
else {
if (i) {
if (out[0] == 0xEF && out[1] == 0xBB && out[2] == 0xBF) {
i = 2;
continue;
}
if (out[0] == 0xFF && out[1] == 0xFE) {
unicode = 1;
i++;
continue;
}
if (out[0] == 0xFE && out[1] == 0xFF) {
unicode = 1;
i += 2;
continue;
}
if (out[0] == 0xFF && out[1] == 0xFE && out[2] == 0 && out[3] == 0) {
unicode = 3;
i += 3;
continue;
}
if (out[0] == 0 && out[1] == 0 && out[2] == 0xFE && out[3] == 0xFF) {
unicode = 3;
i += 6;
continue;
}
if (out[1] <= 0x7E && out[1] >= 0x20) {
++i;
continue;
}
return 0;
}
}
} else if (out[i] < 0x20) {
if (out[i]!='\n' && out[i]!='\r' && out[i]!='\t' && out[i]!=0x1B)
return 0;
}
i += unicode; 
}
return 1;
}
static int CheckSigs(const u8 *p, int len, ZIP_SIGS *pSig)
{
int i, j;
for (i = 0; i < pSig->magic_count; ++i) {
int fnd = 1;
u8 *pS = pSig->magic_signature[i];
for (j = 0; j < pSig->magic_sig_len[i]; ++j) {
if (p[j] != pS[j]) {
fnd = 0;
break;
}
}
if (fnd)
return 1;
}
return 0;
}
#endif
MAYBE_INLINE static int check_inflate_CODE2(u8 *next)
{
u32 bits, hold, thisget, have, i;
int left;
u32 ncode;
u32 ncount[2];	
u8 *count;		
#if (ARCH_LITTLE_ENDIAN==1) && (ARCH_ALLOWS_UNALIGNED==1)
hold = *((u32*)next);
#else
hold = *next + (((u32)next[1])<<8) + (((u32)next[2])<<16) + (((u32)next[3])<<24);
#endif
next += 3;	
hold >>= 3;	
count = (u8*)ncount;
if (257+(hold&0x1F) > 286)
return 0;	
hold >>= 5;
if (1+(hold&0x1F) > 30)
return 0;		
hold >>= 5;
ncode = 4+(hold&0xF);
hold >>= 4;
hold += ((u32)(*++next)) << 15;
hold += ((u32)(*++next)) << 23;
bits = 31;
have = 0;
ncount[0] = ncount[1] = 0;
for (;;) {
if (have+7>ncode)
thisget = ncode-have;
else
thisget = 7;
have += thisget;
bits -= thisget*3;
while (thisget--) {
++count[hold&7];
hold>>=3;
}
if (have == ncode)
break;
hold += ((u32)(*++next)) << bits;
bits += 8;
hold += ((u32)(*++next)) << bits;
bits += 8;
}
count[0] = 0;
if (!ncount[0] && !ncount[1])
return 0; 
left = 1;
for (i = 1; i <= 7; ++i) {
left <<= 1;
left -= count[i];
if (left < 0)
return 0;	
}
if (left > 0)
return 0;		
return 1;			
}
MAYBE_INLINE static int check_inflate_CODE1(u8 *next, int left)
{
u32 whave = 0, op, bits, hold,len;
code here;
#if (ARCH_LITTLE_ENDIAN==1) && (ARCH_ALLOWS_UNALIGNED==1)
hold = *((u32*)next);
#else
hold = *next + (((u32)next[1])<<8) + (((u32)next[2])<<16) + (((u32)next[3])<<24);
#endif
next += 3; 
left -= 4;
hold >>= 3;  
bits = 32-3;
for (;;) {
if (bits < 15) {
if (left < 2)
return 1;	
left -= 2;
hold += (u32)(*++next) << bits;
bits += 8;
hold += (u32)(*++next) << bits;
bits += 8;
}
here=lenfix[hold & 0x1FF];
op = (unsigned)(here.bits);
hold >>= op;
bits -= op;
op = (unsigned)(here.op);
if (op == 0)							
++whave;
else if (op & 16) {						
len = (unsigned)(here.val);
op &= 15;							
if (op) {
if (bits < op) {
if (!left)
return 1;	
--left;
hold += (u32)(*++next) << bits;
bits += 8;
}
len += (unsigned)hold & ((1U << op) - 1);
hold >>= op;
bits -= op;
}
if (bits < 15) {
if (left < 2)
return 1;	
left -= 2;
hold += (u32)(*++next) << bits;
bits += 8;
hold += (u32)(*++next) << bits;
bits += 8;
}
here = distfix[hold & 0x1F];
op = (unsigned)(here.bits);
hold >>= op;
bits -= op;
op = (unsigned)(here.op);
if (op & 16) {                      
u32 dist = (unsigned)(here.val);
op &= 15;                       
if (bits < op) {
if (!left)
return 1;	
--left;
hold += (u32)(*++next) << bits;
bits += 8;
if (bits < op) {
if (!left)
return 1;	
--left;
hold += (u32)(*++next) << bits;
bits += 8;
}
}
dist += (unsigned)hold & ((1U << op) - 1);
if (dist > whave)
return 0;  
hold >>= op;
bits -= op;
/
}
else if (op & 32) {
if (left == 0)
return 1;
return 0;
}
else {
return 0; 
}
/
here = distfix[here.val + (hold & ((1U << op) - 1))];
goto dodist;
}
else
return 0;		
}
else if (op & 64) {
return 0;
}
else if (op & 32) {
return 0;
}
else {
return 0; 
}
#endif
static int crypt_all(int *pcount, struct db_salt *_salt)
{
const int _count = *pcount;
int idx;
#if (ZIP_DEBUG==2)
static int CNT, FAILED, FAILED2;
++CNT;
#endif
#ifdef _OPENMP
#pragma omp parallel for private(idx)
#endif
for (idx = 0; idx < _count; ++idx) {
int cur_hash_count = salt->cnt;
int cur_hash_idx = -1;
MY_WORD key0, key1, key2;
u8 C;
const u8 *b;
u8 curDecryBuf[256];
#if USE_PKZIP_MAGIC
u8 curInfBuf[128];
#endif
int k, SigChecked;
u16 e, v1, v2;
z_stream strm;
int ret;
if (dirty) {
u8 *p = (u8*)saved_key[idx];
key0.u = 0x12345678UL; key1.u = 0x23456789UL; key2.u = 0x34567890UL;
do {
key0.u = jtr_crc32 (key0.u, *p++);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
} while (*p);
K12[idx*3] = key0.u, K12[idx*3+1] = key1.u, K12[idx*3+2] = key2.u;
goto SkipKeyLoadInit;
}
do
{
key0.u = K12[idx*3], key1.u = K12[idx*3+1], key2.u = K12[idx*3+2];
SkipKeyLoadInit:;
b = salt->H[++cur_hash_idx].h;
k=11;
e = salt->H[cur_hash_idx].c;
do
{
C = PKZ_MULT(*b++,key2);
key0.u = jtr_crc32 (key0.u, C);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
}
while(--k);
if (salt->H[cur_hash_idx].type == 2) {
u16 e2 = salt->H[cur_hash_idx].c2;
if (salt->chk_bytes == 2 && C != (e & 0xff) && C != (e2 & 0xff))
goto Failed_Bailout;
C = PKZ_MULT(*b++, key2);
if (C != (e >> 8) && C != (e2 >> 8))
goto Failed_Bailout;
} else {
if (salt->chk_bytes == 2 && C != (e & 0xff))
goto Failed_Bailout;
C = PKZ_MULT(*b++, key2);
if (C != (e >> 8))
goto Failed_Bailout;
}
key0.u = jtr_crc32 (key0.u, C);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
e = 0;
C = PKZ_MULT(*b++,key2);
SigChecked = 0;
if (salt->H[cur_hash_idx].compType == 0) {
#if USE_PKZIP_MAGIC
if (salt->H[cur_hash_idx].pSig->max_len) {
int len = salt->H[cur_hash_idx].pSig->max_len;
if (len > salt->H[cur_hash_idx].datlen-12)
len = salt->H[cur_hash_idx].datlen-12;
SigChecked = 1;
curDecryBuf[0] = C;
for (; e < len;) {
key0.u = jtr_crc32 (key0.u, curDecryBuf[e]);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
curDecryBuf[++e] = PKZ_MULT(*b++,key2);
}
if (salt->H[cur_hash_idx].magic == 255) {
if (!validate_ascii(&curDecryBuf[5], len-5))
goto Failed_Bailout;
} else {
if (!CheckSigs(curDecryBuf, len, salt->H[cur_hash_idx].pSig))
goto Failed_Bailout;
}
}
#endif
continue;
}
#if 1
if ((C & 6) == 6)
goto Failed_Bailout;
#endif
if ((C & 6) == 0) {
if (C > 1)
goto Failed_Bailout;
curDecryBuf[0] = C;
for (e = 0; e <= 4;) {
key0.u = jtr_crc32 (key0.u, curDecryBuf[e]);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
curDecryBuf[++e] = PKZ_MULT(*b++,key2);
}
v1 = curDecryBuf[1] | (((u16)curDecryBuf[2])<<8);
v2 = curDecryBuf[3] | (((u16)curDecryBuf[4])<<8);
if (v1 != (v2^0xFFFF))
goto Failed_Bailout;
#if USE_PKZIP_MAGIC
if (salt->H[cur_hash_idx].pSig->max_len) {
int len = salt->H[cur_hash_idx].pSig->max_len + 5;
if (len > salt->H[cur_hash_idx].datlen-12)
len = salt->H[cur_hash_idx].datlen-12;
SigChecked = 1;
for (; e < len;) {
key0.u = jtr_crc32 (key0.u, curDecryBuf[e]);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
curDecryBuf[++e] = PKZ_MULT(*b++,key2);
}
if (salt->H[cur_hash_idx].magic == 255) {
if (!validate_ascii(&curDecryBuf[5], len-5))
goto Failed_Bailout;
} else {
if (!CheckSigs(&curDecryBuf[5], len-5, salt->H[cur_hash_idx].pSig))
goto Failed_Bailout;
}
}
#endif
}
else {
curDecryBuf[0] = C;
if ((C & 6) == 4) { 
#if (ZIP_DEBUG==2)
static unsigned count, found;
++count;
#endif
for (; e < 10;) {
key0.u = jtr_crc32 (key0.u, curDecryBuf[e]);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
curDecryBuf[++e] = PKZ_MULT(*b++,key2);
}
if (!check_inflate_CODE2(curDecryBuf))
goto Failed_Bailout;
#if (ZIP_DEBUG==2)
fprintf(stderr, "CODE2 Pass=%s  count = %u, found = %u\n", saved_key[idx], count, ++found);
#endif
}
else {
int til;
#if (ZIP_DEBUG==2)
static unsigned count, found;
++count;
#endif
til = 36;
if (salt->H[cur_hash_idx].datlen-12 < til)
til = salt->H[cur_hash_idx].datlen-12;
for (; e < til;) {
key0.u = jtr_crc32 (key0.u, curDecryBuf[e]);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
curDecryBuf[++e] = PKZ_MULT(*b++,key2);
}
if (!check_inflate_CODE1(curDecryBuf, til))
goto Failed_Bailout;
#if (ZIP_DEBUG==2)
fprintf(stderr, "CODE1 Pass=%s  count = %u, found = %u\n", saved_key[idx], count, ++found);
#endif
}
}
#if USE_PKZIP_MAGIC
if (!SigChecked && salt->H[cur_hash_idx].pSig->max_len) {
int til = 180;
if (salt->H[cur_hash_idx].datlen-12 < til)
til = salt->H[cur_hash_idx].datlen-12;
for (; e < til;) {
key0.u = jtr_crc32 (key0.u, curDecryBuf[e]);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
curDecryBuf[++e] = PKZ_MULT(*b++,key2);
}
strm.zalloc = Z_NULL;
strm.zfree = Z_NULL;
strm.opaque = Z_NULL;
strm.next_in = Z_NULL;
strm.avail_in = til;
ret = inflateInit2(&strm, -15); 
if (ret != Z_OK)
perror("Error, initializing the libz inflateInit2() system\n");
strm.next_in = curDecryBuf;
strm.avail_out = sizeof(curInfBuf);
strm.next_out = curInfBuf;
ret = inflate(&strm, Z_SYNC_FLUSH);
inflateEnd(&strm);
if (ret != Z_OK) {
if (ret == Z_STREAM_END && salt->deCompLen == strm.total_out)
; 
else
goto Failed_Bailout;
}
if (!strm.total_out)
goto Failed_Bailout;
ret = salt->H[cur_hash_idx].pSig->max_len;
if (salt->H[cur_hash_idx].magic == 255) {
if (!validate_ascii(curInfBuf, strm.total_out))
goto Failed_Bailout;
} else {
if (strm.total_out < ret)
goto Failed_Bailout;
if (!CheckSigs(curInfBuf, strm.total_out, salt->H[cur_hash_idx].pSig))
goto Failed_Bailout;
}
}
#endif
if (salt->H[cur_hash_idx].full_zip) {
u8 inflateBufTmp[1024];
if (salt->compLen > 240 && salt->H[cur_hash_idx].datlen >= 200) {
for (;e < 200;) {
key0.u = jtr_crc32 (key0.u, curDecryBuf[e]);
key1.u = (key1.u + key0.c[KB1]) * 134775813 + 1;
key2.u = jtr_crc32 (key2.u, key1.c[KB2]);
curDecryBuf[++e] = PKZ_MULT(*b++,key2);
}
strm.zalloc = Z_NULL;
strm.zfree = Z_NULL;
strm.opaque = Z_NULL;
strm.next_in = Z_NULL;
strm.avail_in = e;
ret = inflateInit2(&strm, -15); 
if (ret != Z_OK)
perror("Error, initializing the libz inflateInit2() system\n");
strm.next_in = curDecryBuf;
strm.avail_out = sizeof(inflateBufTmp);
strm.next_out = inflateBufTmp;
ret = inflate(&strm, Z_SYNC_FLUSH);
inflateEnd(&strm);
if (ret != Z_OK) {
#if (ZIP_DEBUG==2)
fprintf(stderr, "fail=%d fail2=%d tot="LLd"\n", ++FAILED, FAILED2, ((long long)CNT)*_count);
#endif
goto Failed_Bailout;
}
}
goto KnownSuccess;
}
}
while(--cur_hash_count);
KnownSuccess: ;
chk[idx] = 1;
continue;
Failed_Bailout: ;
chk[idx] = 0;
}
dirty = 0;
return _count;
}
struct fmt_main fmt_pkzip = {
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
FMT_CASE | FMT_8_BIT | FMT_OMP | FMT_DYNA_SALT | FMT_HUGE_INPUT,
{ NULL },
{ FORMAT_TAG, FORMAT_TAG2 },
tests
}, {
init,
done,
fmt_default_reset,
fmt_default_prepare,
valid,
fmt_default_split,
fmt_default_binary,
get_salt,
{ NULL },
fmt_default_source,
{
fmt_default_binary_hash
},
fmt_default_dyna_salt_hash,
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
#else
#if !defined(FMT_EXTERNS_H) && !defined(FMT_REGISTERS_H)
#ifdef __GNUC__
#warning pkzip format requires zlib to function. The format has been disabled
#elif _MSC_VER
#pragma message(": warning pkzip format requires zlib to function. The format has been disabled :")
#endif
#endif
#endif 
