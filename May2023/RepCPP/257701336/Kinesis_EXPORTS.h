

#pragma once

#ifdef _MSC_VER
#pragma warning (disable : 4503)
#endif 

#if defined (USE_WINDOWS_DLL_SEMANTICS) || defined (WIN32)
#ifdef _MSC_VER
#pragma warning(disable : 4251)
#endif 

#ifdef USE_IMPORT_EXPORT
#ifdef AWS_KINESIS_EXPORTS
#define AWS_KINESIS_API __declspec(dllexport)
#else
#define AWS_KINESIS_API __declspec(dllimport)
#endif 
#else
#define AWS_KINESIS_API
#endif 
#else 
#define AWS_KINESIS_API
#endif 
