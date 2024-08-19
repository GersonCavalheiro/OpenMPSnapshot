
#pragma once

#define CACHELINE_SIZE 64

#if !defined(PAGE_SIZE)
#define PAGE_SIZE 4096
#endif

#define PAGE_SIZE_2M (2*1024*1024)
#define PAGE_SIZE_4K (4*1024)

#include "platform.h"


#if defined (__AVX512VL__)
#  define isa avx512skx
#  define ISA AVX512SKX
#  define ISA_STR "AVX512SKX"
#elif defined (__AVX512F__)
#  define isa avx512knl
#  define ISA AVX512KNL
#  define ISA_STR "AVX512KNL"
#elif defined (__AVX2__)
#  define isa avx2
#  define ISA AVX2
#  define ISA_STR "AVX2"
#elif defined(__AVXI__)
#  define isa avxi
#  define ISA AVXI
#  define ISA_STR "AVXI"
#elif defined(__AVX__)
#  define isa avx
#  define ISA AVX
#  define ISA_STR "AVX"
#elif defined (__SSE4_2__)
#  define isa sse42
#  define ISA SSE42
#  define ISA_STR "SSE4.2"
#elif defined (__SSE4_1__)
#  define isa sse41
#  define ISA SSE41
#  define ISA_STR "SSE4.1"
#elif defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__)
#  define isa sse2
#  define ISA SSE2
#  define ISA_STR "SSE2"
#elif defined(__SSE__)
#  define isa sse
#  define ISA SSE
#  define ISA_STR "SSE"
#else 
#error Unknown ISA
#endif

namespace embree
{
enum CPUModel {
CPU_UNKNOWN,
CPU_CORE1,
CPU_CORE2,
CPU_CORE_NEHALEM,
CPU_CORE_SANDYBRIDGE,
CPU_HASWELL,
CPU_KNIGHTS_LANDING,
CPU_SKYLAKE
};


std::string getExecutableFileName();


std::string getPlatformName();


std::string getCompilerName();


std::string getCPUVendor();


CPUModel getCPUModel(); 


std::string stringOfCPUModel(CPUModel model);


static const int CPU_FEATURE_SSE    = 1 << 0;
static const int CPU_FEATURE_SSE2   = 1 << 1;
static const int CPU_FEATURE_SSE3   = 1 << 2;
static const int CPU_FEATURE_SSSE3  = 1 << 3;
static const int CPU_FEATURE_SSE41  = 1 << 4;
static const int CPU_FEATURE_SSE42  = 1 << 5; 
static const int CPU_FEATURE_POPCNT = 1 << 6;
static const int CPU_FEATURE_AVX    = 1 << 7;
static const int CPU_FEATURE_F16C   = 1 << 8;
static const int CPU_FEATURE_RDRAND = 1 << 9;
static const int CPU_FEATURE_AVX2   = 1 << 10;
static const int CPU_FEATURE_FMA3   = 1 << 11;
static const int CPU_FEATURE_LZCNT  = 1 << 12;
static const int CPU_FEATURE_BMI1   = 1 << 13;
static const int CPU_FEATURE_BMI2   = 1 << 14;
static const int CPU_FEATURE_AVX512F = 1 << 16;
static const int CPU_FEATURE_AVX512DQ = 1 << 17;    
static const int CPU_FEATURE_AVX512PF = 1 << 18;
static const int CPU_FEATURE_AVX512ER = 1 << 19;
static const int CPU_FEATURE_AVX512CD = 1 << 20;
static const int CPU_FEATURE_AVX512BW = 1 << 21;
static const int CPU_FEATURE_AVX512VL = 1 << 22;
static const int CPU_FEATURE_AVX512IFMA = 1 << 23;
static const int CPU_FEATURE_AVX512VBMI = 1 << 24;


int getCPUFeatures();


std::string stringOfCPUFeatures(int features);


std::string supportedTargetList (int isa);


static const int SSE    = CPU_FEATURE_SSE; 
static const int SSE2   = SSE | CPU_FEATURE_SSE2;
static const int SSE3   = SSE2 | CPU_FEATURE_SSE3;
static const int SSSE3  = SSE3 | CPU_FEATURE_SSSE3;
static const int SSE41  = SSSE3 | CPU_FEATURE_SSE41;
static const int SSE42  = SSE41 | CPU_FEATURE_SSE42 | CPU_FEATURE_POPCNT;
static const int AVX    = SSE42 | CPU_FEATURE_AVX;
static const int AVXI   = AVX | CPU_FEATURE_F16C | CPU_FEATURE_RDRAND;
static const int AVX2   = AVXI | CPU_FEATURE_AVX2 | CPU_FEATURE_FMA3 | CPU_FEATURE_BMI1 | CPU_FEATURE_BMI2 | CPU_FEATURE_LZCNT;
static const int AVX512KNL = AVX2 | CPU_FEATURE_AVX512F | CPU_FEATURE_AVX512PF | CPU_FEATURE_AVX512ER | CPU_FEATURE_AVX512CD;
static const int AVX512SKX = AVX2 | CPU_FEATURE_AVX512F | CPU_FEATURE_AVX512DQ | CPU_FEATURE_AVX512CD | CPU_FEATURE_AVX512BW | CPU_FEATURE_AVX512VL;


std::string stringOfISA(int features);


unsigned int getNumberOfLogicalThreads();


int getTerminalWidth();


double getSeconds();


void sleepSeconds(double t);
}
