#ifdef _MSC_VER
#define inline _inline
#endif
#include <stdint.h>
#include "arch.h"
#include "johnswap.h"
#include "hccap.h"
#pragma pack(1)
#define TCPDUMP_MAGIC           0xa1b2c3d4
#define TCPDUMP_CIGAM           0xd4c3b2a1
#define PCAPNGBLOCKTYPE         0x0a0d0d0a
#define PCAPNGMAGICNUMBER       0x1a2b3c4d
#define PCAPNGMAGICNUMBERBE     0x4d3c2b1a
#define LINKTYPE_ETHERNET       1
#define LINKTYPE_IEEE802_11     105
#define LINKTYPE_PRISM_HEADER   119
#define LINKTYPE_RADIOTAP_HDR   127
#define LINKTYPE_PPI_HDR        192
typedef struct pcap_hdr_s {
uint32_t magic_number;   
uint16_t version_major;  
uint16_t version_minor;  
int32_t  thiszone;       
uint32_t sigfigs;        
uint32_t snaplen;        
uint32_t network;        
} pcap_hdr_t;
typedef struct pcaprec_hdr_s {
uint32_t ts_sec;         
uint32_t ts_usec;        
uint32_t snap_len;       
uint32_t orig_len;       
} pcaprec_hdr_t;
typedef struct block_header_s {
uint32_t	block_type;	
uint32_t	total_length;	
} block_header_t;
#define	BH_SIZE (sizeof(block_header_t))
typedef struct option_header_s {
uint16_t		option_code;	
uint16_t		option_length;	
} option_header_t;
#define	OH_SIZE (sizeof(option_header_t))
typedef struct section_header_block_s {
uint32_t	byte_order_magic;	
uint16_t	major_version;		
uint16_t	minor_version;		
int64_t	section_length;		
} section_header_block_t;
#define	SHB_SIZE (sizeof(section_header_block_t))
typedef struct interface_description_block_s {
uint16_t	linktype;	
uint16_t	reserved;	
uint32_t	snaplen;	
} interface_description_block_t;
#define	IDB_SIZE (sizeof(interface_description_block_t))
typedef struct packet_block_s {
uint16_t	interface_id;	
uint16_t	drops_count;	
uint32_t	timestamp_high;	
uint32_t	timestamp_low;	
uint32_t	caplen;	
uint32_t	len;	
} packet_block_t;
#define	PB_SIZE (sizeof(packet_block_t))
typedef struct simple_packet_block_s {
uint32_t	len;  
} simple_packet_block_t;
#define	SPB_SIZE (sizeof(simple_packet_block_t))
typedef struct name_resolution_block_s {
uint16_t	record_type;    
uint16_t	record_length;  
} name_resolution_block_t;
#define	NRB_SIZE (sizeof(name_resolution_block_t))
typedef struct interface_statistics_block_s {
uint32_t	interface_id;     
uint32_t	timestamp_high;   
uint32_t	timestamp_low;    
} interface_statistics_block_t;
#define	ISB_SIZE (sizeof(interface_statistics_block_t))
typedef struct enhanced_packet_block_s {
uint32_t	interface_id;     
uint32_t	timestamp_high;   
uint32_t	timestamp_low;    
uint32_t	caplen;           
uint32_t	len;              
} enhanced_packet_block_t;
#define	EPB_SIZE (sizeof(enhanced_packet_block_t))
typedef struct ieee802_1x_frame_hdr_s {
uint16_t frame_ctl;
uint16_t duration;
uint8_t  addr1[6]; 
uint8_t  addr2[6]; 
uint8_t  addr3[6]; 
uint16_t seq;
} ieee802_1x_frame_hdr_t;
typedef struct ieee802_1x_frame_ctl_s {
uint16_t version  : 2;
uint16_t type     : 2;
uint16_t subtype  : 4;
uint16_t toDS     : 1;
uint16_t fromDS   : 1;
uint16_t morefrag : 1;
uint16_t retry    : 1;
uint16_t powman   : 1;
uint16_t moredata : 1;
uint16_t protfram : 1;
uint16_t order    : 1;
} ieee802_1x_frame_ctl_t;
typedef struct ieee802_1x_eapol_s {
uint8_t ver; 
uint8_t type; 
uint16_t length;  
uint8_t key_descr; 
union {
struct {
uint16_t KeyDescr	: 3; 
uint16_t KeyType	: 1; 
uint16_t KeyIdx	: 2; 
uint16_t Install	: 1; 
uint16_t KeyACK	: 1; 
uint16_t KeyMIC	: 1; 
uint16_t Secure	: 1;
uint16_t Error	: 1;
uint16_t Reqst	: 1;
uint16_t EncKeyDat: 1;
} key_info;
uint16_t key_info_u16;	
};
uint16_t key_len;
uint64_t replay_cnt;
uint8_t wpa_nonce[32];
uint8_t wpa_keyiv[16];
uint8_t wpa_keyrsc[8];
uint8_t wpa_keyid[8];
uint8_t wpa_keymic[16];
uint16_t wpa_keydatlen;
} ieee802_1x_eapol_t;
typedef struct keydata_s {
uint8_t tagtype;
uint8_t taglen;
uint8_t oui[3];
uint8_t oui_type;
uint8_t data[1];
} keydata_t;
typedef struct eapol_keydata_s {
ieee802_1x_eapol_t auth;
keydata_t tag[1];
} eapol_keydata_t;
typedef struct ieee802_1x_auth_s {
uint16_t algo;
uint16_t seq;
uint16_t status;
} ieee802_1x_auth_t;
typedef struct ieee802_1x_beacon_tag_s {
uint8_t  tagtype;
uint8_t  taglen;
uint8_t  tag[1];
} ieee802_1x_beacon_tag_t;
typedef struct ieee802_1x_beacon_data_s {
uint32_t time1;
uint32_t time2;
uint16_t interval;
uint16_t caps;
ieee802_1x_beacon_tag_t tags[1];
} ieee802_1x_beacon_data_t;
typedef struct ieee802_1x_assocreq_s {
uint16_t capa;
uint16_t interval;
ieee802_1x_beacon_tag_t tags[1];
} ieee802_1x_assocreq_t;
typedef struct ieee802_1x_reassocreq_s {
uint16_t capa;
uint16_t interval;
uint8_t  addr3[6];
ieee802_1x_beacon_tag_t tags[1];
} ieee802_1x_reassocreq_t;
typedef struct eapext_s {
uint8_t  version;
uint8_t  type;
uint16_t len;
uint8_t  eapcode;
uint8_t  eapid;
uint16_t eaplen;
uint8_t  eaptype;
} eapext_t;
#define EAP_CODE_RESP       2
#define EAP_TYPE_ID         1
inline static uint16_t swap16u(uint16_t v) {
return ((v>>8)|((v&0xFF)<<8));
}
inline static uint32_t swap32u(uint32_t v) {
return JOHNSWAP(v);
}
inline static uint64_t swap64u(uint64_t v) {
return JOHNSWAP64(v);
}
typedef struct essid_s {
int prio; 
int essid_len;
char essid[32 + 1];
uint8_t bssid[6];
} essid_t;
typedef struct handshake_s {
uint64_t ts64;
int eapol_size;
ieee802_1x_eapol_t *eapol;
} handshake_t;
typedef struct WPA4way_s {
uint64_t rc;
uint32_t anonce_msb;
uint32_t anonce_lsb;
int8_t fuzz;
uint8_t endian; 
int handshake_done;
int pmkid_done;
handshake_t M[5];
uint8_t bssid[6];
uint8_t staid[6];
} WPA4way_t;
#define IVSONLY_MAGIC           "\xBF\xCA\x84\xD4"
#define IVS2_MAGIC              "\xAE\x78\xD1\xFF"
#define IVS2_EXTENSION          "ivs"
#define IVS2_VERSION             1
#define IVS2_BSSID      0x0001
#define IVS2_ESSID      0x0002
#define IVS2_WPA        0x0004
#define IVS2_XOR        0x0008
#define IVS2_PTW        0x0010
#define IVS2_CLR        0x0020
struct ivs2_filehdr
{
uint16_t version;
};
struct ivs2_pkthdr
{
uint16_t  flags;
uint16_t  len;
};
#pragma pack() 
struct ivs2_WPA_hdsk
{
uint8_t stmac[6];     
uint8_t snonce[32];   
uint8_t anonce[32];   
uint8_t keymic[16];   
uint8_t eapol[256];   
uint32_t eapol_size;  
uint8_t keyver;       
uint8_t state;        
};
static void dump_hex(char *msg, void *x, unsigned int size)
{
unsigned int i;
fprintf(stderr, "%s : ", msg);
for (i = 0; i < size; i++) {
fprintf(stderr, "%.2x", ((uint8_t*)x)[i]);
if ((i % 4) == 3)
fprintf(stderr, " ");
}
fprintf(stderr, "\n");
}
#define safe_realloc(p, len) do {	  \
if (!(p = realloc(p, len))) { \
fprintf(stderr, "%s:%d: realloc of "Zu" bytes failed\n", \
__FILE__, __LINE__, (size_t)len); \
exit(EXIT_FAILURE); \
} \
} while (0)
#define safe_malloc(p, len) do {	  \
if (!(p = malloc(len))) { \
fprintf(stderr, "%s:%d: malloc of "Zu" bytes failed\n", \
__FILE__, __LINE__, (size_t)len); \
exit(EXIT_FAILURE); \
} \
} while (0)
