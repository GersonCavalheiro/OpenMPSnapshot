

#ifndef SHR_UTILS_H
#define SHR_UTILS_H


#ifdef _WIN32
#pragma message ("Note: including windows.h")
#pragma message ("Note: including math.h")
#pragma message ("Note: including assert.h")
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#endif

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


inline int ConvertSMVer2Cores(int major, int minor)
{
typedef struct {
int SM; 
int Cores;
} sSMtoCores;

sSMtoCores nGpuArchCoresPerSM[] = 
{ { 0x10,  8 }, 
{ 0x11,  8 }, 
{ 0x12,  8 }, 
{ 0x13,  8 }, 
{ 0x20, 32 }, 
{ 0x21, 48 }, 
{ 0x30, 192}, 
{ 0x35, 192}, 
{   -1, -1 }
};

int index = 0;
while (nGpuArchCoresPerSM[index].SM != -1) {
if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
return nGpuArchCoresPerSM[index].Cores;
}
index++;
}
printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
return nGpuArchCoresPerSM[7].Cores;
}


#define DEFAULTLOGFILE "SdkConsoleLog.txt"
#define MASTERLOGFILE "SdkMasterLog.csv"
enum LOGMODES 
{
LOGCONSOLE = 1, 
LOGFILE    = 2, 
LOGBOTH    = 3, 
APPENDMODE = 4, 
MASTER     = 8, 
ERRORMSG   = 16, 
CLOSELOG   = 32  
};
#define HDASHLINE "-----------------------------------------------------------\n"

enum shrBOOL
{
shrFALSE = 0,
shrTRUE = 1
};

#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)    
#define TOPCLAMP(a, b) (a < b ? a:b)	    

#define shrCheckErrorEX(a, b, c) __shrCheckErrorEX(a, b, c, __FILE__ , __LINE__) 

#define shrCheckError(a, b) shrCheckErrorEX(a, b, 0) 

#define shrExitEX(a, b, c) __shrExitEX(a, b, c)

#define shrEXIT(a, b)        __shrExitEX(a, b, EXIT_SUCCESS)

#define ARGCHECK(a) if((a) != shrTRUE)return shrFALSE 

#define STDERROR "file %s, line %i\n\n" , __FILE__ , __LINE__

extern "C" void shrFree(void* ptr);

extern "C" int shrLogEx(int iLogMode, int iErrNum, const char* cFormatString, ...);

extern "C" int shrLog(const char* cFormatString, ...);

extern "C" double shrDeltaT(int iCounterID);

extern "C" void shrSetLogFileName (const char* cOverRideName);

extern "C" void shrFillArray(float* pfData, int iSize);

extern "C" void shrPrintArray(float* pfData, int iSize);

extern "C" char* shrFindFilePath(const char* filename, const char* executablePath);

extern "C" shrBOOL shrReadFilef( const char* filename, float** data, unsigned int* len, 
bool verbose = false);

extern "C" shrBOOL shrReadFiled( const char* filename, double** data, unsigned int* len, 
bool verbose = false);

extern "C" shrBOOL shrReadFilei( const char* filename, int** data, unsigned int* len, bool verbose = false);

extern "C" shrBOOL shrReadFileui( const char* filename, unsigned int** data, 
unsigned int* len, bool verbose = false);

extern "C" shrBOOL shrReadFileb( const char* filename, char** data, unsigned int* len, 
bool verbose = false);

extern "C" shrBOOL shrReadFileub( const char* filename, unsigned char** data, 
unsigned int* len, bool verbose = false);

extern "C" shrBOOL shrWriteFilef( const char* filename, const float* data, unsigned int len,
const float epsilon, bool verbose = false);

extern "C" shrBOOL shrWriteFiled( const char* filename, const float* data, unsigned int len,
const double epsilon, bool verbose = false);

extern "C" shrBOOL shrWriteFilei( const char* filename, const int* data, unsigned int len,
bool verbose = false);

extern "C" shrBOOL shrWriteFileui( const char* filename, const unsigned int* data, 
unsigned int len, bool verbose = false);

extern "C" shrBOOL shrWriteFileb( const char* filename, const char* data, unsigned int len, 
bool verbose = false);

extern "C" shrBOOL shrWriteFileub( const char* filename, const unsigned char* data,
unsigned int len, bool verbose = false);

extern "C" shrBOOL shrLoadPPM4ub(const char* file, unsigned char** OutData, 
unsigned int *w, unsigned int *h);

extern "C" shrBOOL shrSavePPM4ub( const char* file, unsigned char *data, 
unsigned int w, unsigned int h);

extern "C" shrBOOL shrSavePGMub( const char* file, unsigned char *data, 
unsigned int w, unsigned int h); 

extern "C" shrBOOL shrLoadPGMub( const char* file, unsigned char** data,
unsigned int *w,unsigned int *h);


extern "C" shrBOOL shrCheckCmdLineFlag( const int argc, const char** argv, 
const char* flag_name);

extern "C" shrBOOL shrGetCmdLineArgumenti( const int argc, const char** argv, 
const char* arg_name, int* val);

extern "C" shrBOOL shrGetCmdLineArgumentu( const int argc, const char** argv, 
const char* arg_name, unsigned int* val);

extern "C" shrBOOL shrGetCmdLineArgumentf( const int argc, const char** argv, 
const char* arg_name, float* val);

extern "C" shrBOOL shrGetCmdLineArgumentstr( const int argc, const char** argv, 
const char* arg_name, char** val);

extern "C" shrBOOL shrGetCmdLineArgumentListstr( const int argc, const char** argv, 
const char* arg_name, char** val, 
unsigned int* len);

extern "C" shrBOOL shrComparef( const float* reference, const float* data,
const unsigned int len);

extern "C" shrBOOL shrComparei( const int* reference, const int* data, 
const unsigned int len ); 

extern "C" shrBOOL shrCompareuit( const unsigned int* reference, const unsigned int* data,
const unsigned int len, const float epsilon, const float threshold );

extern "C" shrBOOL shrCompareub( const unsigned char* reference, const unsigned char* data,
const unsigned int len ); 

extern "C" shrBOOL shrCompareubt( const unsigned char* reference, const unsigned char* data,
const unsigned int len, const float epsilon, const float threshold );

extern "C" shrBOOL shrCompareube( const unsigned char* reference, const unsigned char* data,
const unsigned int len, const float epsilon );

extern "C" shrBOOL shrComparefe( const float* reference, const float* data,
const unsigned int len, const float epsilon );

extern "C" shrBOOL shrComparefet( const float* reference, const float* data,
const unsigned int len, const float epsilon, const float threshold );

extern "C" shrBOOL shrCompareL2fe( const float* reference, const float* data,
const unsigned int len, const float epsilon );

extern "C" shrBOOL shrComparePPM( const char *src_file, const char *ref_file, const float epsilon, const float threshold);

extern "C" shrBOOL shrComparePGM( const char *src_file, const char *ref_file, const float epsilon, const float threshold);

extern "C" unsigned char* shrLoadRawFile(const char* filename, size_t size);

extern "C" size_t shrRoundUp(int group_size, int global_size);

inline void __shrCheckErrorEX(int iSample, int iReference, void (*pCleanup)(int), const char* cFile, const int iLine)
{
if (iReference != iSample)
{
shrLogEx(LOGBOTH | ERRORMSG, iSample, "line %i , in file %s !!!\n\n" , iLine, cFile); 
if (pCleanup != NULL)
{
pCleanup(EXIT_FAILURE);
}
else 
{
shrLogEx(LOGBOTH | CLOSELOG, 0, "Exiting...\n");
exit(EXIT_FAILURE);
}
}
}

inline void __shrExitEX(int argc, const char** argv, int iExitCode)
{
#ifdef WIN32
if (!shrCheckCmdLineFlag(argc, argv, "noprompt") && !shrCheckCmdLineFlag(argc, argv, "qatest")) 
#else 
if (shrCheckCmdLineFlag(argc, argv, "prompt") && !shrCheckCmdLineFlag(argc, argv, "qatest")) 
#endif
{
shrLogEx(LOGBOTH | CLOSELOG, 0, "\nPress <Enter> to Quit...\n");                  
getchar();                                                           
}       
else 
{
shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", argv[0]); 
}
fflush(stderr);                                                         
exit(iExitCode);
}

#endif
