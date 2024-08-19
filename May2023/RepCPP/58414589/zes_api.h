
#ifndef _ZES_API_H
#define _ZES_API_H
#if defined(__cplusplus)
#pragma once
#endif

#include "ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(__GNUC__)
#pragma region common
#endif
typedef ze_driver_handle_t zes_driver_handle_t;

typedef ze_device_handle_t zes_device_handle_t;

typedef struct _zes_sched_handle_t *zes_sched_handle_t;

typedef struct _zes_perf_handle_t *zes_perf_handle_t;

typedef struct _zes_pwr_handle_t *zes_pwr_handle_t;

typedef struct _zes_freq_handle_t *zes_freq_handle_t;

typedef struct _zes_engine_handle_t *zes_engine_handle_t;

typedef struct _zes_standby_handle_t *zes_standby_handle_t;

typedef struct _zes_firmware_handle_t *zes_firmware_handle_t;

typedef struct _zes_mem_handle_t *zes_mem_handle_t;

typedef struct _zes_fabric_port_handle_t *zes_fabric_port_handle_t;

typedef struct _zes_temp_handle_t *zes_temp_handle_t;

typedef struct _zes_psu_handle_t *zes_psu_handle_t;

typedef struct _zes_fan_handle_t *zes_fan_handle_t;

typedef struct _zes_led_handle_t *zes_led_handle_t;

typedef struct _zes_ras_handle_t *zes_ras_handle_t;

typedef struct _zes_diag_handle_t *zes_diag_handle_t;

typedef enum _zes_structure_type_t
{
ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1,     
ZES_STRUCTURE_TYPE_PCI_PROPERTIES = 0x2,        
ZES_STRUCTURE_TYPE_PCI_BAR_PROPERTIES = 0x3,    
ZES_STRUCTURE_TYPE_DIAG_PROPERTIES = 0x4,       
ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES = 0x5,     
ZES_STRUCTURE_TYPE_FABRIC_PORT_PROPERTIES = 0x6,
ZES_STRUCTURE_TYPE_FAN_PROPERTIES = 0x7,        
ZES_STRUCTURE_TYPE_FIRMWARE_PROPERTIES = 0x8,   
ZES_STRUCTURE_TYPE_FREQ_PROPERTIES = 0x9,       
ZES_STRUCTURE_TYPE_LED_PROPERTIES = 0xa,        
ZES_STRUCTURE_TYPE_MEM_PROPERTIES = 0xb,        
ZES_STRUCTURE_TYPE_PERF_PROPERTIES = 0xc,       
ZES_STRUCTURE_TYPE_POWER_PROPERTIES = 0xd,      
ZES_STRUCTURE_TYPE_PSU_PROPERTIES = 0xe,        
ZES_STRUCTURE_TYPE_RAS_PROPERTIES = 0xf,        
ZES_STRUCTURE_TYPE_SCHED_PROPERTIES = 0x10,     
ZES_STRUCTURE_TYPE_SCHED_TIMEOUT_PROPERTIES = 0x11, 
ZES_STRUCTURE_TYPE_SCHED_TIMESLICE_PROPERTIES = 0x12,   
ZES_STRUCTURE_TYPE_STANDBY_PROPERTIES = 0x13,   
ZES_STRUCTURE_TYPE_TEMP_PROPERTIES = 0x14,      
ZES_STRUCTURE_TYPE_DEVICE_STATE = 0x15,         
ZES_STRUCTURE_TYPE_PROCESS_STATE = 0x16,        
ZES_STRUCTURE_TYPE_PCI_STATE = 0x17,            
ZES_STRUCTURE_TYPE_FABRIC_PORT_CONFIG = 0x18,   
ZES_STRUCTURE_TYPE_FABRIC_PORT_STATE = 0x19,    
ZES_STRUCTURE_TYPE_FAN_CONFIG = 0x1a,           
ZES_STRUCTURE_TYPE_FREQ_STATE = 0x1b,           
ZES_STRUCTURE_TYPE_OC_CAPABILITIES = 0x1c,      
ZES_STRUCTURE_TYPE_LED_STATE = 0x1d,            
ZES_STRUCTURE_TYPE_MEM_STATE = 0x1e,            
ZES_STRUCTURE_TYPE_PSU_STATE = 0x1f,            
ZES_STRUCTURE_TYPE_BASE_STATE = 0x20,           
ZES_STRUCTURE_TYPE_RAS_CONFIG = 0x21,           
ZES_STRUCTURE_TYPE_RAS_STATE = 0x22,            
ZES_STRUCTURE_TYPE_TEMP_CONFIG = 0x23,          
ZES_STRUCTURE_TYPE_PCI_BAR_PROPERTIES_1_2 = 0x24,   
ZES_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_structure_type_t;

typedef struct _zes_base_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    

} zes_base_properties_t;

typedef struct _zes_base_desc_t
{
zes_structure_type_t stype;                     
const void* pNext;                              

} zes_base_desc_t;

typedef struct _zes_base_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              

} zes_base_state_t;

typedef struct _zes_base_config_t
{
zes_structure_type_t stype;                     
const void* pNext;                              

} zes_base_config_t;

typedef struct _zes_base_capability_t
{
zes_structure_type_t stype;                     
const void* pNext;                              

} zes_base_capability_t;

typedef struct _zes_base_properties_t zes_base_properties_t;

typedef struct _zes_base_desc_t zes_base_desc_t;

typedef struct _zes_base_state_t zes_base_state_t;

typedef struct _zes_base_config_t zes_base_config_t;

typedef struct _zes_base_capability_t zes_base_capability_t;

typedef struct _zes_device_state_t zes_device_state_t;

typedef struct _zes_device_properties_t zes_device_properties_t;

typedef struct _zes_process_state_t zes_process_state_t;

typedef struct _zes_pci_address_t zes_pci_address_t;

typedef struct _zes_pci_speed_t zes_pci_speed_t;

typedef struct _zes_pci_properties_t zes_pci_properties_t;

typedef struct _zes_pci_state_t zes_pci_state_t;

typedef struct _zes_pci_bar_properties_t zes_pci_bar_properties_t;

typedef struct _zes_pci_bar_properties_1_2_t zes_pci_bar_properties_1_2_t;

typedef struct _zes_pci_stats_t zes_pci_stats_t;

typedef struct _zes_diag_test_t zes_diag_test_t;

typedef struct _zes_diag_properties_t zes_diag_properties_t;

typedef struct _zes_engine_properties_t zes_engine_properties_t;

typedef struct _zes_engine_stats_t zes_engine_stats_t;

typedef struct _zes_fabric_port_id_t zes_fabric_port_id_t;

typedef struct _zes_fabric_port_speed_t zes_fabric_port_speed_t;

typedef struct _zes_fabric_port_properties_t zes_fabric_port_properties_t;

typedef struct _zes_fabric_link_type_t zes_fabric_link_type_t;

typedef struct _zes_fabric_port_config_t zes_fabric_port_config_t;

typedef struct _zes_fabric_port_state_t zes_fabric_port_state_t;

typedef struct _zes_fabric_port_throughput_t zes_fabric_port_throughput_t;

typedef struct _zes_fan_speed_t zes_fan_speed_t;

typedef struct _zes_fan_temp_speed_t zes_fan_temp_speed_t;

typedef struct _zes_fan_speed_table_t zes_fan_speed_table_t;

typedef struct _zes_fan_properties_t zes_fan_properties_t;

typedef struct _zes_fan_config_t zes_fan_config_t;

typedef struct _zes_firmware_properties_t zes_firmware_properties_t;

typedef struct _zes_freq_properties_t zes_freq_properties_t;

typedef struct _zes_freq_range_t zes_freq_range_t;

typedef struct _zes_freq_state_t zes_freq_state_t;

typedef struct _zes_freq_throttle_time_t zes_freq_throttle_time_t;

typedef struct _zes_oc_capabilities_t zes_oc_capabilities_t;

typedef struct _zes_led_properties_t zes_led_properties_t;

typedef struct _zes_led_color_t zes_led_color_t;

typedef struct _zes_led_state_t zes_led_state_t;

typedef struct _zes_mem_properties_t zes_mem_properties_t;

typedef struct _zes_mem_state_t zes_mem_state_t;

typedef struct _zes_mem_bandwidth_t zes_mem_bandwidth_t;

typedef struct _zes_perf_properties_t zes_perf_properties_t;

typedef struct _zes_power_properties_t zes_power_properties_t;

typedef struct _zes_power_energy_counter_t zes_power_energy_counter_t;

typedef struct _zes_power_sustained_limit_t zes_power_sustained_limit_t;

typedef struct _zes_power_burst_limit_t zes_power_burst_limit_t;

typedef struct _zes_power_peak_limit_t zes_power_peak_limit_t;

typedef struct _zes_energy_threshold_t zes_energy_threshold_t;

typedef struct _zes_psu_properties_t zes_psu_properties_t;

typedef struct _zes_psu_state_t zes_psu_state_t;

typedef struct _zes_ras_properties_t zes_ras_properties_t;

typedef struct _zes_ras_state_t zes_ras_state_t;

typedef struct _zes_ras_config_t zes_ras_config_t;

typedef struct _zes_sched_properties_t zes_sched_properties_t;

typedef struct _zes_sched_timeout_properties_t zes_sched_timeout_properties_t;

typedef struct _zes_sched_timeslice_properties_t zes_sched_timeslice_properties_t;

typedef struct _zes_standby_properties_t zes_standby_properties_t;

typedef struct _zes_temp_properties_t zes_temp_properties_t;

typedef struct _zes_temp_threshold_t zes_temp_threshold_t;

typedef struct _zes_temp_config_t zes_temp_config_t;


#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region device
#endif
#ifndef ZES_STRING_PROPERTY_SIZE
#define ZES_STRING_PROPERTY_SIZE  64
#endif 

typedef uint32_t zes_engine_type_flags_t;
typedef enum _zes_engine_type_flag_t
{
ZES_ENGINE_TYPE_FLAG_OTHER = ZE_BIT(0),         
ZES_ENGINE_TYPE_FLAG_COMPUTE = ZE_BIT(1),       
ZES_ENGINE_TYPE_FLAG_3D = ZE_BIT(2),            
ZES_ENGINE_TYPE_FLAG_MEDIA = ZE_BIT(3),         
ZES_ENGINE_TYPE_FLAG_DMA = ZE_BIT(4),           
ZES_ENGINE_TYPE_FLAG_RENDER = ZE_BIT(5),        
ZES_ENGINE_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_engine_type_flag_t;

typedef enum _zes_repair_status_t
{
ZES_REPAIR_STATUS_UNSUPPORTED = 0,              
ZES_REPAIR_STATUS_NOT_PERFORMED = 1,            
ZES_REPAIR_STATUS_PERFORMED = 2,                
ZES_REPAIR_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_repair_status_t;

typedef uint32_t zes_reset_reason_flags_t;
typedef enum _zes_reset_reason_flag_t
{
ZES_RESET_REASON_FLAG_WEDGED = ZE_BIT(0),       
ZES_RESET_REASON_FLAG_REPAIR = ZE_BIT(1),       
ZES_RESET_REASON_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_reset_reason_flag_t;

typedef struct _zes_device_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
zes_reset_reason_flags_t reset;                 
zes_repair_status_t repaired;                   

} zes_device_state_t;

typedef struct _zes_device_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_device_properties_t core;                    
uint32_t numSubdevices;                         
char serialNumber[ZES_STRING_PROPERTY_SIZE];    
char boardNumber[ZES_STRING_PROPERTY_SIZE];     
char brandName[ZES_STRING_PROPERTY_SIZE];       
char modelName[ZES_STRING_PROPERTY_SIZE];       
char vendorName[ZES_STRING_PROPERTY_SIZE];      
char driverVersion[ZES_STRING_PROPERTY_SIZE];   

} zes_device_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetProperties(
zes_device_handle_t hDevice,                    
zes_device_properties_t* pProperties            
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetState(
zes_device_handle_t hDevice,                    
zes_device_state_t* pState                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceReset(
zes_device_handle_t hDevice,                    
ze_bool_t force                                 
);

typedef struct _zes_process_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
uint32_t processId;                             
uint64_t memSize;                               
uint64_t sharedSize;                            
zes_engine_type_flags_t engines;                

} zes_process_state_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceProcessesGetState(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_process_state_t* pProcesses                 
);

typedef struct _zes_pci_address_t
{
uint32_t domain;                                
uint32_t bus;                                   
uint32_t device;                                
uint32_t function;                              

} zes_pci_address_t;

typedef struct _zes_pci_speed_t
{
int32_t gen;                                    
int32_t width;                                  
int64_t maxBandwidth;                           

} zes_pci_speed_t;

typedef struct _zes_pci_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_pci_address_t address;                      
zes_pci_speed_t maxSpeed;                       
ze_bool_t haveBandwidthCounters;                
ze_bool_t havePacketCounters;                   
ze_bool_t haveReplayCounters;                   

} zes_pci_properties_t;

typedef enum _zes_pci_link_status_t
{
ZES_PCI_LINK_STATUS_UNKNOWN = 0,                
ZES_PCI_LINK_STATUS_GOOD = 1,                   
ZES_PCI_LINK_STATUS_QUALITY_ISSUES = 2,         
ZES_PCI_LINK_STATUS_STABILITY_ISSUES = 3,       
ZES_PCI_LINK_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_status_t;

typedef uint32_t zes_pci_link_qual_issue_flags_t;
typedef enum _zes_pci_link_qual_issue_flag_t
{
ZES_PCI_LINK_QUAL_ISSUE_FLAG_REPLAYS = ZE_BIT(0),   
ZES_PCI_LINK_QUAL_ISSUE_FLAG_SPEED = ZE_BIT(1), 
ZES_PCI_LINK_QUAL_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_qual_issue_flag_t;

typedef uint32_t zes_pci_link_stab_issue_flags_t;
typedef enum _zes_pci_link_stab_issue_flag_t
{
ZES_PCI_LINK_STAB_ISSUE_FLAG_RETRAINING = ZE_BIT(0),
ZES_PCI_LINK_STAB_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_stab_issue_flag_t;

typedef struct _zes_pci_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
zes_pci_link_status_t status;                   
zes_pci_link_qual_issue_flags_t qualityIssues;  
zes_pci_link_stab_issue_flags_t stabilityIssues;
zes_pci_speed_t speed;                          

} zes_pci_state_t;

typedef enum _zes_pci_bar_type_t
{
ZES_PCI_BAR_TYPE_MMIO = 0,                      
ZES_PCI_BAR_TYPE_ROM = 1,                       
ZES_PCI_BAR_TYPE_MEM = 2,                       
ZES_PCI_BAR_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_pci_bar_type_t;

typedef struct _zes_pci_bar_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_pci_bar_type_t type;                        
uint32_t index;                                 
uint64_t base;                                  
uint64_t size;                                  

} zes_pci_bar_properties_t;

typedef struct _zes_pci_bar_properties_1_2_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_pci_bar_type_t type;                        
uint32_t index;                                 
uint64_t base;                                  
uint64_t size;                                  
ze_bool_t resizableBarSupported;                
ze_bool_t resizableBarEnabled;                  

} zes_pci_bar_properties_1_2_t;

typedef struct _zes_pci_stats_t
{
uint64_t timestamp;                             
uint64_t replayCounter;                         
uint64_t packetCounter;                         
uint64_t rxCounter;                             
uint64_t txCounter;                             
zes_pci_speed_t speed;                          

} zes_pci_stats_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetProperties(
zes_device_handle_t hDevice,                    
zes_pci_properties_t* pProperties               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetState(
zes_device_handle_t hDevice,                    
zes_pci_state_t* pState                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetBars(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_pci_bar_properties_t* pProperties           
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetStats(
zes_device_handle_t hDevice,                    
zes_pci_stats_t* pStats                         
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region diagnostics
#endif
typedef enum _zes_diag_result_t
{
ZES_DIAG_RESULT_NO_ERRORS = 0,                  
ZES_DIAG_RESULT_ABORT = 1,                      
ZES_DIAG_RESULT_FAIL_CANT_REPAIR = 2,           
ZES_DIAG_RESULT_REBOOT_FOR_REPAIR = 3,          
ZES_DIAG_RESULT_FORCE_UINT32 = 0x7fffffff

} zes_diag_result_t;

#ifndef ZES_DIAG_FIRST_TEST_INDEX
#define ZES_DIAG_FIRST_TEST_INDEX  0x0
#endif 

#ifndef ZES_DIAG_LAST_TEST_INDEX
#define ZES_DIAG_LAST_TEST_INDEX  0xFFFFFFFF
#endif 

typedef struct _zes_diag_test_t
{
uint32_t index;                                 
char name[ZES_STRING_PROPERTY_SIZE];            

} zes_diag_test_t;

typedef struct _zes_diag_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
char name[ZES_STRING_PROPERTY_SIZE];            
ze_bool_t haveTests;                            

} zes_diag_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumDiagnosticTestSuites(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_diag_handle_t* phDiagnostics                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsGetProperties(
zes_diag_handle_t hDiagnostics,                 
zes_diag_properties_t* pProperties              
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsGetTests(
zes_diag_handle_t hDiagnostics,                 
uint32_t* pCount,                               
zes_diag_test_t* pTests                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsRunTests(
zes_diag_handle_t hDiagnostics,                 
uint32_t startIndex,                            
uint32_t endIndex,                              
zes_diag_result_t* pResult                      
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region engine
#endif
typedef enum _zes_engine_group_t
{
ZES_ENGINE_GROUP_ALL = 0,                       
ZES_ENGINE_GROUP_COMPUTE_ALL = 1,               
ZES_ENGINE_GROUP_MEDIA_ALL = 2,                 
ZES_ENGINE_GROUP_COPY_ALL = 3,                  
ZES_ENGINE_GROUP_COMPUTE_SINGLE = 4,            
ZES_ENGINE_GROUP_RENDER_SINGLE = 5,             
ZES_ENGINE_GROUP_MEDIA_DECODE_SINGLE = 6,       
ZES_ENGINE_GROUP_MEDIA_ENCODE_SINGLE = 7,       
ZES_ENGINE_GROUP_COPY_SINGLE = 8,               
ZES_ENGINE_GROUP_MEDIA_ENHANCEMENT_SINGLE = 9,  
ZES_ENGINE_GROUP_3D_SINGLE = 10,                
ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL = 11,    
ZES_ENGINE_GROUP_RENDER_ALL = 12,               
ZES_ENGINE_GROUP_3D_ALL = 13,                   
ZES_ENGINE_GROUP_FORCE_UINT32 = 0x7fffffff

} zes_engine_group_t;

typedef struct _zes_engine_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_engine_group_t type;                        
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           

} zes_engine_properties_t;

typedef struct _zes_engine_stats_t
{
uint64_t activeTime;                            
uint64_t timestamp;                             

} zes_engine_stats_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumEngineGroups(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_engine_handle_t* phEngine                   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesEngineGetProperties(
zes_engine_handle_t hEngine,                    
zes_engine_properties_t* pProperties            
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesEngineGetActivity(
zes_engine_handle_t hEngine,                    
zes_engine_stats_t* pStats                      
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region events
#endif
typedef uint32_t zes_event_type_flags_t;
typedef enum _zes_event_type_flag_t
{
ZES_EVENT_TYPE_FLAG_DEVICE_DETACH = ZE_BIT(0),  
ZES_EVENT_TYPE_FLAG_DEVICE_ATTACH = ZE_BIT(1),  
ZES_EVENT_TYPE_FLAG_DEVICE_SLEEP_STATE_ENTER = ZE_BIT(2),   
ZES_EVENT_TYPE_FLAG_DEVICE_SLEEP_STATE_EXIT = ZE_BIT(3),
ZES_EVENT_TYPE_FLAG_FREQ_THROTTLED = ZE_BIT(4), 
ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED = ZE_BIT(5),   
ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL = ZE_BIT(6),  
ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 = ZE_BIT(7),
ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 = ZE_BIT(8),
ZES_EVENT_TYPE_FLAG_MEM_HEALTH = ZE_BIT(9),     
ZES_EVENT_TYPE_FLAG_FABRIC_PORT_HEALTH = ZE_BIT(10),
ZES_EVENT_TYPE_FLAG_PCI_LINK_HEALTH = ZE_BIT(11),   
ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS = ZE_BIT(12),
ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS = ZE_BIT(13),  
ZES_EVENT_TYPE_FLAG_DEVICE_RESET_REQUIRED = ZE_BIT(14), 
ZES_EVENT_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_event_type_flag_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEventRegister(
zes_device_handle_t hDevice,                    
zes_event_type_flags_t events                   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverEventListen(
ze_driver_handle_t hDriver,                     
uint32_t timeout,                               
uint32_t count,                                 
zes_device_handle_t* phDevices,                 
uint32_t* pNumDeviceEvents,                     
zes_event_type_flags_t* pEvents                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverEventListenEx(
ze_driver_handle_t hDriver,                     
uint64_t timeout,                               
uint32_t count,                                 
zes_device_handle_t* phDevices,                 
uint32_t* pNumDeviceEvents,                     
zes_event_type_flags_t* pEvents                 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region fabric
#endif
#ifndef ZES_MAX_FABRIC_PORT_MODEL_SIZE
#define ZES_MAX_FABRIC_PORT_MODEL_SIZE  256
#endif 

#ifndef ZES_MAX_FABRIC_LINK_TYPE_SIZE
#define ZES_MAX_FABRIC_LINK_TYPE_SIZE  256
#endif 

typedef enum _zes_fabric_port_status_t
{
ZES_FABRIC_PORT_STATUS_UNKNOWN = 0,             
ZES_FABRIC_PORT_STATUS_HEALTHY = 1,             
ZES_FABRIC_PORT_STATUS_DEGRADED = 2,            
ZES_FABRIC_PORT_STATUS_FAILED = 3,              
ZES_FABRIC_PORT_STATUS_DISABLED = 4,            
ZES_FABRIC_PORT_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_status_t;

typedef uint32_t zes_fabric_port_qual_issue_flags_t;
typedef enum _zes_fabric_port_qual_issue_flag_t
{
ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_LINK_ERRORS = ZE_BIT(0),
ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_SPEED = ZE_BIT(1),  
ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_qual_issue_flag_t;

typedef uint32_t zes_fabric_port_failure_flags_t;
typedef enum _zes_fabric_port_failure_flag_t
{
ZES_FABRIC_PORT_FAILURE_FLAG_FAILED = ZE_BIT(0),
ZES_FABRIC_PORT_FAILURE_FLAG_TRAINING_TIMEOUT = ZE_BIT(1),  
ZES_FABRIC_PORT_FAILURE_FLAG_FLAPPING = ZE_BIT(2),  
ZES_FABRIC_PORT_FAILURE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_failure_flag_t;

typedef struct _zes_fabric_port_id_t
{
uint32_t fabricId;                              
uint32_t attachId;                              
uint8_t portNumber;                             

} zes_fabric_port_id_t;

typedef struct _zes_fabric_port_speed_t
{
int64_t bitRate;                                
int32_t width;                                  

} zes_fabric_port_speed_t;

typedef struct _zes_fabric_port_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
char model[ZES_MAX_FABRIC_PORT_MODEL_SIZE];     
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
zes_fabric_port_id_t portId;                    
zes_fabric_port_speed_t maxRxSpeed;             
zes_fabric_port_speed_t maxTxSpeed;             

} zes_fabric_port_properties_t;

typedef struct _zes_fabric_link_type_t
{
char desc[ZES_MAX_FABRIC_LINK_TYPE_SIZE];       

} zes_fabric_link_type_t;

typedef struct _zes_fabric_port_config_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
ze_bool_t enabled;                              
ze_bool_t beaconing;                            

} zes_fabric_port_config_t;

typedef struct _zes_fabric_port_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
zes_fabric_port_status_t status;                
zes_fabric_port_qual_issue_flags_t qualityIssues;   
zes_fabric_port_failure_flags_t failureReasons; 
zes_fabric_port_id_t remotePortId;              
zes_fabric_port_speed_t rxSpeed;                
zes_fabric_port_speed_t txSpeed;                

} zes_fabric_port_state_t;

typedef struct _zes_fabric_port_throughput_t
{
uint64_t timestamp;                             
uint64_t rxCounter;                             
uint64_t txCounter;                             

} zes_fabric_port_throughput_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumFabricPorts(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_fabric_port_handle_t* phPort                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetProperties(
zes_fabric_port_handle_t hPort,                 
zes_fabric_port_properties_t* pProperties       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetLinkType(
zes_fabric_port_handle_t hPort,                 
zes_fabric_link_type_t* pLinkType               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetConfig(
zes_fabric_port_handle_t hPort,                 
zes_fabric_port_config_t* pConfig               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortSetConfig(
zes_fabric_port_handle_t hPort,                 
const zes_fabric_port_config_t* pConfig         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetState(
zes_fabric_port_handle_t hPort,                 
zes_fabric_port_state_t* pState                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetThroughput(
zes_fabric_port_handle_t hPort,                 
zes_fabric_port_throughput_t* pThroughput       
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region fan
#endif
typedef enum _zes_fan_speed_mode_t
{
ZES_FAN_SPEED_MODE_DEFAULT = 0,                 
ZES_FAN_SPEED_MODE_FIXED = 1,                   
ZES_FAN_SPEED_MODE_TABLE = 2,                   
ZES_FAN_SPEED_MODE_FORCE_UINT32 = 0x7fffffff

} zes_fan_speed_mode_t;

typedef enum _zes_fan_speed_units_t
{
ZES_FAN_SPEED_UNITS_RPM = 0,                    
ZES_FAN_SPEED_UNITS_PERCENT = 1,                
ZES_FAN_SPEED_UNITS_FORCE_UINT32 = 0x7fffffff

} zes_fan_speed_units_t;

typedef struct _zes_fan_speed_t
{
int32_t speed;                                  
zes_fan_speed_units_t units;                    

} zes_fan_speed_t;

typedef struct _zes_fan_temp_speed_t
{
uint32_t temperature;                           
zes_fan_speed_t speed;                          

} zes_fan_temp_speed_t;

#ifndef ZES_FAN_TEMP_SPEED_PAIR_COUNT
#define ZES_FAN_TEMP_SPEED_PAIR_COUNT  32
#endif 

typedef struct _zes_fan_speed_table_t
{
int32_t numPoints;                              
zes_fan_temp_speed_t table[ZES_FAN_TEMP_SPEED_PAIR_COUNT];  

} zes_fan_speed_table_t;

typedef struct _zes_fan_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
ze_bool_t canControl;                           
uint32_t supportedModes;                        
uint32_t supportedUnits;                        
int32_t maxRPM;                                 
int32_t maxPoints;                              

} zes_fan_properties_t;

typedef struct _zes_fan_config_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
zes_fan_speed_mode_t mode;                      
zes_fan_speed_t speedFixed;                     
zes_fan_speed_table_t speedTable;               

} zes_fan_config_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumFans(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_fan_handle_t* phFan                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetProperties(
zes_fan_handle_t hFan,                          
zes_fan_properties_t* pProperties               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetConfig(
zes_fan_handle_t hFan,                          
zes_fan_config_t* pConfig                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetDefaultMode(
zes_fan_handle_t hFan                           
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetFixedSpeedMode(
zes_fan_handle_t hFan,                          
const zes_fan_speed_t* speed                    
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetSpeedTableMode(
zes_fan_handle_t hFan,                          
const zes_fan_speed_table_t* speedTable         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetState(
zes_fan_handle_t hFan,                          
zes_fan_speed_units_t units,                    
int32_t* pSpeed                                 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region firmware
#endif
typedef struct _zes_firmware_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
ze_bool_t canControl;                           
char name[ZES_STRING_PROPERTY_SIZE];            
char version[ZES_STRING_PROPERTY_SIZE];         

} zes_firmware_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumFirmwares(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_firmware_handle_t* phFirmware               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFirmwareGetProperties(
zes_firmware_handle_t hFirmware,                
zes_firmware_properties_t* pProperties          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFirmwareFlash(
zes_firmware_handle_t hFirmware,                
void* pImage,                                   
uint32_t size                                   
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region frequency
#endif
typedef enum _zes_freq_domain_t
{
ZES_FREQ_DOMAIN_GPU = 0,                        
ZES_FREQ_DOMAIN_MEMORY = 1,                     
ZES_FREQ_DOMAIN_FORCE_UINT32 = 0x7fffffff

} zes_freq_domain_t;

typedef struct _zes_freq_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_freq_domain_t type;                         
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
ze_bool_t canControl;                           
ze_bool_t isThrottleEventSupported;             
double min;                                     
double max;                                     

} zes_freq_properties_t;

typedef struct _zes_freq_range_t
{
double min;                                     
double max;                                     

} zes_freq_range_t;

typedef uint32_t zes_freq_throttle_reason_flags_t;
typedef enum _zes_freq_throttle_reason_flag_t
{
ZES_FREQ_THROTTLE_REASON_FLAG_AVE_PWR_CAP = ZE_BIT(0),  
ZES_FREQ_THROTTLE_REASON_FLAG_BURST_PWR_CAP = ZE_BIT(1),
ZES_FREQ_THROTTLE_REASON_FLAG_CURRENT_LIMIT = ZE_BIT(2),
ZES_FREQ_THROTTLE_REASON_FLAG_THERMAL_LIMIT = ZE_BIT(3),
ZES_FREQ_THROTTLE_REASON_FLAG_PSU_ALERT = ZE_BIT(4),
ZES_FREQ_THROTTLE_REASON_FLAG_SW_RANGE = ZE_BIT(5), 
ZES_FREQ_THROTTLE_REASON_FLAG_HW_RANGE = ZE_BIT(6), 
ZES_FREQ_THROTTLE_REASON_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_freq_throttle_reason_flag_t;

typedef struct _zes_freq_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
double currentVoltage;                          
double request;                                 
double tdp;                                     
double efficient;                               
double actual;                                  
zes_freq_throttle_reason_flags_t throttleReasons;   

} zes_freq_state_t;

typedef struct _zes_freq_throttle_time_t
{
uint64_t throttleTime;                          
uint64_t timestamp;                             

} zes_freq_throttle_time_t;

typedef enum _zes_oc_mode_t
{
ZES_OC_MODE_OFF = 0,                            
ZES_OC_MODE_OVERRIDE = 1,                       
ZES_OC_MODE_INTERPOLATIVE = 2,                  
ZES_OC_MODE_FIXED = 3,                          
ZES_OC_MODE_FORCE_UINT32 = 0x7fffffff

} zes_oc_mode_t;

typedef struct _zes_oc_capabilities_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
ze_bool_t isOcSupported;                        
double maxFactoryDefaultFrequency;              
double maxFactoryDefaultVoltage;                
double maxOcFrequency;                          
double minOcVoltageOffset;                      
double maxOcVoltageOffset;                      
double maxOcVoltage;                            
ze_bool_t isTjMaxSupported;                     
ze_bool_t isIccMaxSupported;                    
ze_bool_t isHighVoltModeCapable;                
ze_bool_t isHighVoltModeEnabled;                
ze_bool_t isExtendedModeSupported;              
ze_bool_t isFixedModeSupported;                 

} zes_oc_capabilities_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumFrequencyDomains(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_freq_handle_t* phFrequency                  
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetProperties(
zes_freq_handle_t hFrequency,                   
zes_freq_properties_t* pProperties              
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetAvailableClocks(
zes_freq_handle_t hFrequency,                   
uint32_t* pCount,                               
double* phFrequency                             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetRange(
zes_freq_handle_t hFrequency,                   
zes_freq_range_t* pLimits                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencySetRange(
zes_freq_handle_t hFrequency,                   
const zes_freq_range_t* pLimits                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetState(
zes_freq_handle_t hFrequency,                   
zes_freq_state_t* pState                        
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetThrottleTime(
zes_freq_handle_t hFrequency,                   
zes_freq_throttle_time_t* pThrottleTime         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetCapabilities(
zes_freq_handle_t hFrequency,                   
zes_oc_capabilities_t* pOcCapabilities          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetFrequencyTarget(
zes_freq_handle_t hFrequency,                   
double* pCurrentOcFrequency                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetFrequencyTarget(
zes_freq_handle_t hFrequency,                   
double CurrentOcFrequency                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetVoltageTarget(
zes_freq_handle_t hFrequency,                   
double* pCurrentVoltageTarget,                  
double* pCurrentVoltageOffset                   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetVoltageTarget(
zes_freq_handle_t hFrequency,                   
double CurrentVoltageTarget,                    
double CurrentVoltageOffset                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetMode(
zes_freq_handle_t hFrequency,                   
zes_oc_mode_t CurrentOcMode                     
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetMode(
zes_freq_handle_t hFrequency,                   
zes_oc_mode_t* pCurrentOcMode                   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetIccMax(
zes_freq_handle_t hFrequency,                   
double* pOcIccMax                               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetIccMax(
zes_freq_handle_t hFrequency,                   
double ocIccMax                                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetTjMax(
zes_freq_handle_t hFrequency,                   
double* pOcTjMax                                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetTjMax(
zes_freq_handle_t hFrequency,                   
double ocTjMax                                  
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region led
#endif
typedef struct _zes_led_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
ze_bool_t canControl;                           
ze_bool_t haveRGB;                              

} zes_led_properties_t;

typedef struct _zes_led_color_t
{
double red;                                     
double green;                                   
double blue;                                    

} zes_led_color_t;

typedef struct _zes_led_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
ze_bool_t isOn;                                 
zes_led_color_t color;                          

} zes_led_state_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumLeds(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_led_handle_t* phLed                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedGetProperties(
zes_led_handle_t hLed,                          
zes_led_properties_t* pProperties               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedGetState(
zes_led_handle_t hLed,                          
zes_led_state_t* pState                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedSetState(
zes_led_handle_t hLed,                          
ze_bool_t enable                                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedSetColor(
zes_led_handle_t hLed,                          
const zes_led_color_t* pColor                   
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region memory
#endif
typedef enum _zes_mem_type_t
{
ZES_MEM_TYPE_HBM = 0,                           
ZES_MEM_TYPE_DDR = 1,                           
ZES_MEM_TYPE_DDR3 = 2,                          
ZES_MEM_TYPE_DDR4 = 3,                          
ZES_MEM_TYPE_DDR5 = 4,                          
ZES_MEM_TYPE_LPDDR = 5,                         
ZES_MEM_TYPE_LPDDR3 = 6,                        
ZES_MEM_TYPE_LPDDR4 = 7,                        
ZES_MEM_TYPE_LPDDR5 = 8,                        
ZES_MEM_TYPE_SRAM = 9,                          
ZES_MEM_TYPE_L1 = 10,                           
ZES_MEM_TYPE_L3 = 11,                           
ZES_MEM_TYPE_GRF = 12,                          
ZES_MEM_TYPE_SLM = 13,                          
ZES_MEM_TYPE_GDDR4 = 14,                        
ZES_MEM_TYPE_GDDR5 = 15,                        
ZES_MEM_TYPE_GDDR5X = 16,                       
ZES_MEM_TYPE_GDDR6 = 17,                        
ZES_MEM_TYPE_GDDR6X = 18,                       
ZES_MEM_TYPE_GDDR7 = 19,                        
ZES_MEM_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_mem_type_t;

typedef enum _zes_mem_loc_t
{
ZES_MEM_LOC_SYSTEM = 0,                         
ZES_MEM_LOC_DEVICE = 1,                         
ZES_MEM_LOC_FORCE_UINT32 = 0x7fffffff

} zes_mem_loc_t;

typedef enum _zes_mem_health_t
{
ZES_MEM_HEALTH_UNKNOWN = 0,                     
ZES_MEM_HEALTH_OK = 1,                          
ZES_MEM_HEALTH_DEGRADED = 2,                    
ZES_MEM_HEALTH_CRITICAL = 3,                    
ZES_MEM_HEALTH_REPLACE = 4,                     
ZES_MEM_HEALTH_FORCE_UINT32 = 0x7fffffff

} zes_mem_health_t;

typedef struct _zes_mem_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_mem_type_t type;                            
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
zes_mem_loc_t location;                         
uint64_t physicalSize;                          
int32_t busWidth;                               
int32_t numChannels;                            

} zes_mem_properties_t;

typedef struct _zes_mem_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
zes_mem_health_t health;                        
uint64_t free;                                  
uint64_t size;                                  

} zes_mem_state_t;

typedef struct _zes_mem_bandwidth_t
{
uint64_t readCounter;                           
uint64_t writeCounter;                          
uint64_t maxBandwidth;                          
uint64_t timestamp;                             

} zes_mem_bandwidth_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumMemoryModules(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_mem_handle_t* phMemory                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetProperties(
zes_mem_handle_t hMemory,                       
zes_mem_properties_t* pProperties               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetState(
zes_mem_handle_t hMemory,                       
zes_mem_state_t* pState                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetBandwidth(
zes_mem_handle_t hMemory,                       
zes_mem_bandwidth_t* pBandwidth                 
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region performance
#endif
typedef struct _zes_perf_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
zes_engine_type_flags_t engines;                

} zes_perf_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumPerformanceFactorDomains(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_perf_handle_t* phPerf                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorGetProperties(
zes_perf_handle_t hPerf,                        
zes_perf_properties_t* pProperties              
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorGetConfig(
zes_perf_handle_t hPerf,                        
double* pFactor                                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorSetConfig(
zes_perf_handle_t hPerf,                        
double factor                                   
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region power
#endif
typedef struct _zes_power_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
ze_bool_t canControl;                           
ze_bool_t isEnergyThresholdSupported;           
int32_t defaultLimit;                           
int32_t minLimit;                               
int32_t maxLimit;                               

} zes_power_properties_t;

typedef struct _zes_power_energy_counter_t
{
uint64_t energy;                                
uint64_t timestamp;                             

} zes_power_energy_counter_t;

typedef struct _zes_power_sustained_limit_t
{
ze_bool_t enabled;                              
int32_t power;                                  
int32_t interval;                               

} zes_power_sustained_limit_t;

typedef struct _zes_power_burst_limit_t
{
ze_bool_t enabled;                              
int32_t power;                                  

} zes_power_burst_limit_t;

typedef struct _zes_power_peak_limit_t
{
int32_t powerAC;                                
int32_t powerDC;                                

} zes_power_peak_limit_t;

typedef struct _zes_energy_threshold_t
{
ze_bool_t enable;                               
double threshold;                               
uint32_t processId;                             

} zes_energy_threshold_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumPowerDomains(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_pwr_handle_t* phPower                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetCardPowerDomain(
zes_device_handle_t hDevice,                    
zes_pwr_handle_t* phPower                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetProperties(
zes_pwr_handle_t hPower,                        
zes_power_properties_t* pProperties             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetEnergyCounter(
zes_pwr_handle_t hPower,                        
zes_power_energy_counter_t* pEnergy             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetLimits(
zes_pwr_handle_t hPower,                        
zes_power_sustained_limit_t* pSustained,        
zes_power_burst_limit_t* pBurst,                
zes_power_peak_limit_t* pPeak                   
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerSetLimits(
zes_pwr_handle_t hPower,                        
const zes_power_sustained_limit_t* pSustained,  
const zes_power_burst_limit_t* pBurst,          
const zes_power_peak_limit_t* pPeak             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetEnergyThreshold(
zes_pwr_handle_t hPower,                        
zes_energy_threshold_t* pThreshold              
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerSetEnergyThreshold(
zes_pwr_handle_t hPower,                        
double threshold                                
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region psu
#endif
typedef enum _zes_psu_voltage_status_t
{
ZES_PSU_VOLTAGE_STATUS_UNKNOWN = 0,             
ZES_PSU_VOLTAGE_STATUS_NORMAL = 1,              
ZES_PSU_VOLTAGE_STATUS_OVER = 2,                
ZES_PSU_VOLTAGE_STATUS_UNDER = 3,               
ZES_PSU_VOLTAGE_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_psu_voltage_status_t;

typedef struct _zes_psu_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
ze_bool_t haveFan;                              
int32_t ampLimit;                               

} zes_psu_properties_t;

typedef struct _zes_psu_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
zes_psu_voltage_status_t voltStatus;            
ze_bool_t fanFailed;                            
int32_t temperature;                            
int32_t current;                                

} zes_psu_state_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumPsus(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_psu_handle_t* phPsu                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPsuGetProperties(
zes_psu_handle_t hPsu,                          
zes_psu_properties_t* pProperties               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesPsuGetState(
zes_psu_handle_t hPsu,                          
zes_psu_state_t* pState                         
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region ras
#endif
typedef enum _zes_ras_error_type_t
{
ZES_RAS_ERROR_TYPE_CORRECTABLE = 0,             
ZES_RAS_ERROR_TYPE_UNCORRECTABLE = 1,           
ZES_RAS_ERROR_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_ras_error_type_t;

typedef enum _zes_ras_error_cat_t
{
ZES_RAS_ERROR_CAT_RESET = 0,                    
ZES_RAS_ERROR_CAT_PROGRAMMING_ERRORS = 1,       
ZES_RAS_ERROR_CAT_DRIVER_ERRORS = 2,            
ZES_RAS_ERROR_CAT_COMPUTE_ERRORS = 3,           
ZES_RAS_ERROR_CAT_NON_COMPUTE_ERRORS = 4,       
ZES_RAS_ERROR_CAT_CACHE_ERRORS = 5,             
ZES_RAS_ERROR_CAT_DISPLAY_ERRORS = 6,           
ZES_RAS_ERROR_CAT_FORCE_UINT32 = 0x7fffffff

} zes_ras_error_cat_t;

#ifndef ZES_MAX_RAS_ERROR_CATEGORY_COUNT
#define ZES_MAX_RAS_ERROR_CATEGORY_COUNT  7
#endif 

typedef struct _zes_ras_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_ras_error_type_t type;                      
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           

} zes_ras_properties_t;

typedef struct _zes_ras_state_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
uint64_t category[ZES_MAX_RAS_ERROR_CATEGORY_COUNT];

} zes_ras_state_t;

typedef struct _zes_ras_config_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
uint64_t totalThreshold;                        
zes_ras_state_t detailedThresholds;             

} zes_ras_config_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumRasErrorSets(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_ras_handle_t* phRas                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetProperties(
zes_ras_handle_t hRas,                          
zes_ras_properties_t* pProperties               
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetConfig(
zes_ras_handle_t hRas,                          
zes_ras_config_t* pConfig                       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasSetConfig(
zes_ras_handle_t hRas,                          
const zes_ras_config_t* pConfig                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetState(
zes_ras_handle_t hRas,                          
ze_bool_t clear,                                
zes_ras_state_t* pState                         
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region scheduler
#endif
typedef enum _zes_sched_mode_t
{
ZES_SCHED_MODE_TIMEOUT = 0,                     
ZES_SCHED_MODE_TIMESLICE = 1,                   
ZES_SCHED_MODE_EXCLUSIVE = 2,                   
ZES_SCHED_MODE_COMPUTE_UNIT_DEBUG = 3,          
ZES_SCHED_MODE_FORCE_UINT32 = 0x7fffffff

} zes_sched_mode_t;

typedef struct _zes_sched_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
ze_bool_t canControl;                           
zes_engine_type_flags_t engines;                
uint32_t supportedModes;                        

} zes_sched_properties_t;

#ifndef ZES_SCHED_WATCHDOG_DISABLE
#define ZES_SCHED_WATCHDOG_DISABLE  (~(0ULL))
#endif 

typedef struct _zes_sched_timeout_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
uint64_t watchdogTimeout;                       

} zes_sched_timeout_properties_t;

typedef struct _zes_sched_timeslice_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
uint64_t interval;                              
uint64_t yieldTimeout;                          

} zes_sched_timeslice_properties_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumSchedulers(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_sched_handle_t* phScheduler                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetProperties(
zes_sched_handle_t hScheduler,                  
zes_sched_properties_t* pProperties             
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetCurrentMode(
zes_sched_handle_t hScheduler,                  
zes_sched_mode_t* pMode                         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetTimeoutModeProperties(
zes_sched_handle_t hScheduler,                  
ze_bool_t getDefaults,                          
zes_sched_timeout_properties_t* pConfig         
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetTimesliceModeProperties(
zes_sched_handle_t hScheduler,                  
ze_bool_t getDefaults,                          
zes_sched_timeslice_properties_t* pConfig       
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetTimeoutMode(
zes_sched_handle_t hScheduler,                  
zes_sched_timeout_properties_t* pProperties,    
ze_bool_t* pNeedReload                          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetTimesliceMode(
zes_sched_handle_t hScheduler,                  
zes_sched_timeslice_properties_t* pProperties,  
ze_bool_t* pNeedReload                          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetExclusiveMode(
zes_sched_handle_t hScheduler,                  
ze_bool_t* pNeedReload                          
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetComputeUnitDebugMode(
zes_sched_handle_t hScheduler,                  
ze_bool_t* pNeedReload                          
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region standby
#endif
typedef enum _zes_standby_type_t
{
ZES_STANDBY_TYPE_GLOBAL = 0,                    
ZES_STANDBY_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_standby_type_t;

typedef struct _zes_standby_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_standby_type_t type;                        
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           

} zes_standby_properties_t;

typedef enum _zes_standby_promo_mode_t
{
ZES_STANDBY_PROMO_MODE_DEFAULT = 0,             
ZES_STANDBY_PROMO_MODE_NEVER = 1,               
ZES_STANDBY_PROMO_MODE_FORCE_UINT32 = 0x7fffffff

} zes_standby_promo_mode_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumStandbyDomains(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_standby_handle_t* phStandby                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbyGetProperties(
zes_standby_handle_t hStandby,                  
zes_standby_properties_t* pProperties           
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbyGetMode(
zes_standby_handle_t hStandby,                  
zes_standby_promo_mode_t* pMode                 
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbySetMode(
zes_standby_handle_t hStandby,                  
zes_standby_promo_mode_t mode                   
);

#if !defined(__GNUC__)
#pragma endregion
#endif
#if !defined(__GNUC__)
#pragma region temperature
#endif
typedef enum _zes_temp_sensors_t
{
ZES_TEMP_SENSORS_GLOBAL = 0,                    
ZES_TEMP_SENSORS_GPU = 1,                       
ZES_TEMP_SENSORS_MEMORY = 2,                    
ZES_TEMP_SENSORS_GLOBAL_MIN = 3,                
ZES_TEMP_SENSORS_GPU_MIN = 4,                   
ZES_TEMP_SENSORS_MEMORY_MIN = 5,                
ZES_TEMP_SENSORS_FORCE_UINT32 = 0x7fffffff

} zes_temp_sensors_t;

typedef struct _zes_temp_properties_t
{
zes_structure_type_t stype;                     
void* pNext;                                    
zes_temp_sensors_t type;                        
ze_bool_t onSubdevice;                          
uint32_t subdeviceId;                           
double maxTemperature;                          
ze_bool_t isCriticalTempSupported;              
ze_bool_t isThreshold1Supported;                
ze_bool_t isThreshold2Supported;                

} zes_temp_properties_t;

typedef struct _zes_temp_threshold_t
{
ze_bool_t enableLowToHigh;                      
ze_bool_t enableHighToLow;                      
double threshold;                               

} zes_temp_threshold_t;

typedef struct _zes_temp_config_t
{
zes_structure_type_t stype;                     
const void* pNext;                              
ze_bool_t enableCritical;                       
zes_temp_threshold_t threshold1;                
zes_temp_threshold_t threshold2;                

} zes_temp_config_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEnumTemperatureSensors(
zes_device_handle_t hDevice,                    
uint32_t* pCount,                               
zes_temp_handle_t* phTemperature                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetProperties(
zes_temp_handle_t hTemperature,                 
zes_temp_properties_t* pProperties              
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetConfig(
zes_temp_handle_t hTemperature,                 
zes_temp_config_t* pConfig                      
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureSetConfig(
zes_temp_handle_t hTemperature,                 
const zes_temp_config_t* pConfig                
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetState(
zes_temp_handle_t hTemperature,                 
double* pTemperature                            
);

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} 
#endif

#endif 