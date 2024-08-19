#ifndef UAF_RAW_H
#define UAF_RAW_H
#include <stddef.h>
typedef struct {
unsigned char len;
char s[31];
} vstring32;			
typedef struct {
unsigned char len;
char s[63];
} vstring64;			
typedef unsigned int uafrec_qword[2];	
typedef int uafrec_bintime[2];	
typedef struct uafrec_flags_def {
unsigned disctly : 1;	
unsigned defcli : 1;	
unsigned lockpwd : 1;
unsigned restricted : 1;   
unsigned disacnt : 1;
unsigned diswelcome : 1;
unsigned dismail : 1;
unsigned nomail : 1;	
unsigned genpwd : 1;
unsigned pwd_expired : 1;
unsigned pwd2_expired : 1;
unsigned audit : 1;		
unsigned disreport : 1;     
unsigned disreconnect : 1;
unsigned autologin : 1;	
unsigned disforce_pwd_change : 1;
unsigned captive : 1;	
unsigned disimage : 1;
unsigned dispwddic : 1;
unsigned dispwdhis : 1;
unsigned devclsval : 1;	
unsigned extauth : 1;	
unsigned migratepwd : 1;	
unsigned vmsauth : 1;	
unsigned dispwdsynch : 1;	
unsigned pwdmix : 1;	
} uaf_flags_bitset;
struct uaf_rec {
unsigned char rtype;	
unsigned char version;	
unsigned short usrdatoff;   
char username[32];		
struct {
unsigned short int mem;
unsigned short int grp;
} uic;			
unsigned int sub_id;	
uafrec_qword parent_id;	
char account[32];		
vstring32 owner;		
vstring32 defdev;		
vstring64 defdir;		
vstring64 lgicmd;		
vstring32 defcli;		
vstring32 clitables;
uafrec_qword pwd, pwd2;	
unsigned short int logfails;
unsigned short int salt;	
unsigned char encrypt, encrypt2;	
unsigned char pwd_length;	
unsigned char fill_1;	
uafrec_bintime expiration;	
uafrec_bintime pwd_lifetime;	
uafrec_bintime pwd_date, pwd2_date;	
uafrec_bintime lastlogin_i, lastlogin_n;
uafrec_qword priv;		
uafrec_qword def_priv;		
char min_class[20];		
char max_class[20];
union {
uaf_flags_bitset flags;
unsigned flagbits;
};
unsigned char fixed_fill[128+44];
unsigned char user_fill[768];
};
#define UAF_REC_SIZE (sizeof(struct uaf_rec))
#define UAF_REC_MIN_USRDATOFF (offsetof(struct uaf_rec,user_fill))
#define UAF_RECTYPE_USER 1
#define UAF_REC_USER_VERSION 1
#define UAF_FLG_DISACNT
#ifdef VMS
#include <uaf070def>		
#pragma assert non_zero(sizeof(struct uaf_rec) == sizeof(uaf070)) "uaf_rec is wrong size"
#endif
#endif
