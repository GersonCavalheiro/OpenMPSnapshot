

#include "ittnotify_config.h"

#if ITT_PLATFORM==ITT_PLATFORM_WIN

#if defined _MSC_VER

#pragma warning (disable: 593)   
#pragma warning (disable: 344)   
#pragma warning (disable: 174)   
#pragma warning (disable: 4127)  
#pragma warning (disable: 4306)  

#endif

#endif 

#if defined __INTEL_COMPILER

#pragma warning (disable: 869)  
#pragma warning (disable: 1418) 
#pragma warning (disable: 1419) 

#endif 
