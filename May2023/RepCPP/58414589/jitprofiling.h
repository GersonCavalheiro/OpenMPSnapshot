

#ifndef __JITPROFILING_H__
#define __JITPROFILING_H__






typedef enum iJIT_jvm_event
{
iJVM_EVENT_TYPE_SHUTDOWN = 2,               

iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED = 13,  

iJVM_EVENT_TYPE_METHOD_UNLOAD_START,    


iJVM_EVENT_TYPE_METHOD_UPDATE,   


iJVM_EVENT_TYPE_METHOD_INLINE_LOAD_FINISHED, 


iJVM_EVENT_TYPE_METHOD_UPDATE_V2,


iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V2 = 21, 

iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V3       
} iJIT_JVM_EVENT;


typedef enum _iJIT_IsProfilingActiveFlags
{
iJIT_NOTHING_RUNNING           = 0x0000,    
iJIT_SAMPLING_ON               = 0x0001,    
} iJIT_IsProfilingActiveFlags;


typedef struct _LineNumberInfo
{
unsigned int Offset;     
unsigned int LineNumber; 

} *pLineNumberInfo, LineNumberInfo;


typedef enum _iJIT_CodeArchitecture
{
iJIT_CA_NATIVE = 0, 

iJIT_CA_32,         

iJIT_CA_64          

} iJIT_CodeArchitecture;

#pragma pack(push, 8)


typedef struct _iJIT_Method_Load
{
unsigned int method_id; 

char* method_name; 

void* method_load_address; 

unsigned int method_size; 

unsigned int line_number_size; 

pLineNumberInfo line_number_table; 

unsigned int class_id; 

char* class_file_name; 

char* source_file_name; 

} *piJIT_Method_Load, iJIT_Method_Load;


typedef struct _iJIT_Method_Load_V2
{
unsigned int method_id; 

char* method_name; 

void* method_load_address; 

unsigned int method_size; 

unsigned int line_number_size; 

pLineNumberInfo line_number_table; 

char* class_file_name; 

char* source_file_name; 

char* module_name; 

} *piJIT_Method_Load_V2, iJIT_Method_Load_V2;


typedef struct _iJIT_Method_Load_V3
{
unsigned int method_id; 

char* method_name; 

void* method_load_address; 

unsigned int method_size; 

unsigned int line_number_size; 

pLineNumberInfo line_number_table; 

char* class_file_name; 

char* source_file_name; 

char* module_name; 

iJIT_CodeArchitecture module_arch; 

} *piJIT_Method_Load_V3, iJIT_Method_Load_V3;


typedef struct _iJIT_Method_Inline_Load
{
unsigned int method_id; 

unsigned int parent_method_id; 

char* method_name; 

void* method_load_address;  

unsigned int method_size; 

unsigned int line_number_size; 

pLineNumberInfo line_number_table; 

char* class_file_name; 

char* source_file_name; 

} *piJIT_Method_Inline_Load, iJIT_Method_Inline_Load;



typedef enum _iJIT_SegmentType
{
iJIT_CT_UNKNOWN = 0,

iJIT_CT_CODE,           

iJIT_CT_DATA,           

iJIT_CT_KEEP,           
iJIT_CT_EOF
} iJIT_SegmentType;



typedef struct _iJIT_Method_Update
{
void* load_address;         

unsigned int size;          

iJIT_SegmentType type;      

const char* data_format;    
} *piJIT_Method_Update, iJIT_Method_Update;



#pragma pack(pop)


#ifdef __cplusplus
extern "C" {
#endif 

#ifndef JITAPI_CDECL
#  if defined WIN32 || defined _WIN32
#    define JITAPI_CDECL __cdecl
#  else 
#    if defined _M_IX86 || defined __i386__
#      define JITAPI_CDECL __attribute__ ((cdecl))
#    else  
#      define JITAPI_CDECL 
#    endif 
#  endif 
#endif 

#define JITAPI JITAPI_CDECL



unsigned int JITAPI iJIT_GetNewMethodID(void);


iJIT_IsProfilingActiveFlags JITAPI iJIT_IsProfilingActive(void);


int JITAPI iJIT_NotifyEvent(iJIT_JVM_EVENT event_type, void *EventSpecificData);

#ifdef __cplusplus
}
#endif 




#endif 
