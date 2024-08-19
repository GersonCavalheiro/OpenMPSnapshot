#ifndef _OBJC_TEST_SUITE_TYPES_H_
#define _OBJC_TEST_SUITE_TYPES_H_
#ifndef __NEXT_RUNTIME__
typedef void * TNS_STRING_REF_T;
#else 
#include "next-abi.h"
#ifdef NEXT_OBJC_USE_NEW_INTERFACE
#include <objc/runtime.h>
#else
#include <objc/objc-runtime.h>
#endif
#undef  nil
#define nil ((id)0)
#ifndef NULL
#define NULL 0
#endif
#ifdef __OBJC2__
typedef Class TNS_STRING_REF_T;
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
typedef struct objc_class TNS_STRING_REF_T;
#pragma GCC diagnostic pop
#endif
#endif  
#endif 
