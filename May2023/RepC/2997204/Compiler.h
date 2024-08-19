#ifndef __7Z_COMPILER_H
#define __7Z_COMPILER_H
#ifdef _MSC_VER
#ifdef UNDER_CE
#define RPC_NO_WINDOWS_H
#pragma warning(disable : 4201) 
#pragma warning(disable : 4214) 
#endif
#if _MSC_VER >= 1300
#pragma warning(disable : 4996) 
#else
#pragma warning(disable : 4511) 
#pragma warning(disable : 4512) 
#pragma warning(disable : 4514) 
#pragma warning(disable : 4702) 
#pragma warning(disable : 4710) 
#pragma warning(disable : 4714) 
#pragma warning(disable : 4786) 
#endif
#endif
#define UNUSED_VAR(x) (void)x;
#endif
