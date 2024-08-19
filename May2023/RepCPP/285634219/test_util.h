


#pragma once

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#undef small            
#else
#include <sys/resource.h>
#endif

#include <hip/hip_runtime.h>

#include <stdio.h>
#include <math.h>
#include <float.h>

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <limits>

#include "mersenne.h"






#define AssertEquals(a, b) if ((a) != (b)) { std::cerr << "\n(" << __FILE__ << ": " << __LINE__ << ")\n"; exit(1);}





struct CommandLineArgs
{

std::vector<std::string>    keys;
std::vector<std::string>    values;
std::vector<std::string>    args;
hipDeviceProp_t              deviceProp;
float                       device_giga_bandwidth;
size_t                      device_free_physmem;
size_t                      device_total_physmem;


CommandLineArgs(int argc, char **argv) :
keys(10),
values(10)
{
using namespace std;

unsigned int mersenne_init[4]=  {0x123, 0x234, 0x345, 0x456};
mersenne::init_by_array(mersenne_init, 4);

for (int i = 1; i < argc; i++)
{
string arg = argv[i];

if ((arg[0] != '-') || (arg[1] != '-'))
{
args.push_back(arg);
continue;
}

string::size_type pos;
string key, val;
if ((pos = arg.find('=')) == string::npos) {
key = string(arg, 2, arg.length() - 2);
val = "";
} else {
key = string(arg, 2, pos - 2);
val = string(arg, pos + 1, arg.length() - 1);
}

keys.push_back(key);
values.push_back(val);
}
}



bool CheckCmdLineFlag(const char* arg_name)
{
using namespace std;

for (int i = 0; i < int(keys.size()); ++i)
{
if (keys[i] == string(arg_name))
return true;
}
return false;
}



template <typename T>
int NumNakedArgs()
{
return args.size();
}



template <typename T>
void GetCmdLineArgument(int index, T &val)
{
using namespace std;
if (index < args.size()) {
istringstream str_stream(args[index]);
str_stream >> val;
}
}


template <typename T>
void GetCmdLineArgument(const char *arg_name, T &val)
{
using namespace std;

for (int i = 0; i < int(keys.size()); ++i)
{
if (keys[i] == string(arg_name))
{
istringstream str_stream(values[i]);
str_stream >> val;
}
}
}



template <typename T>
void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
{
using namespace std;

if (CheckCmdLineFlag(arg_name))
{
vals.clear();

for (int i = 0; i < keys.size(); ++i)
{
if (keys[i] == string(arg_name))
{
string val_string(values[i]);
istringstream str_stream(val_string);
string::size_type old_pos = 0;
string::size_type new_pos = 0;

T val;
while ((new_pos = val_string.find(',', old_pos)) != string::npos)
{
if (new_pos != old_pos)
{
str_stream.width(new_pos - old_pos);
str_stream >> val;
vals.push_back(val);
}

str_stream.ignore(1);
old_pos = new_pos + 1;
}

str_stream >> val;
vals.push_back(val);
}
}
}
}



int ParsedArgc()
{
return (int) keys.size();
}



};



int g_num_rand_samples = 0;


template <typename T>
bool IsNaN(T val) { return false; }

template<>
bool IsNaN<float>(float val)
{
volatile unsigned int bits = reinterpret_cast<unsigned int &>(val);

return (((bits >= 0x7F800001) && (bits <= 0x7FFFFFFF)) || 
((bits >= 0xFF800001) && (bits <= 0xFFFFFFFF)));
}



template<>
bool IsNaN<float2>(float2 val)
{
return (IsNaN(val.y) || IsNaN(val.x));
}

template<>
bool IsNaN<float3>(float3 val)
{
return (IsNaN(val.z) || IsNaN(val.y) || IsNaN(val.x));
}

template<>
bool IsNaN<float4>(float4 val)
{
return (IsNaN(val.y) || IsNaN(val.x) || IsNaN(val.w) || IsNaN(val.z));
}

template<>
bool IsNaN<double>(double val)
{
volatile unsigned long long bits = *reinterpret_cast<unsigned long long *>(&val);

return (((bits >= 0x7FF0000000000001) && (bits <= 0x7FFFFFFFFFFFFFFF)) || 
((bits >= 0xFFF0000000000001) && (bits <= 0xFFFFFFFFFFFFFFFF)));
}



template<>
bool IsNaN<double2>(double2 val)
{
return (IsNaN(val.y) || IsNaN(val.x));
}

template<>
bool IsNaN<double3>(double3 val)
{
return (IsNaN(val.z) || IsNaN(val.y) || IsNaN(val.x));
}

template<>
bool IsNaN<double4>(double4 val)
{
return (IsNaN(val.y) || IsNaN(val.x) || IsNaN(val.w) || IsNaN(val.z));
}










template <typename T>
T CoutCast(T val) { return val; }

int CoutCast(char val) { return val; }

int CoutCast(unsigned char val) { return val; }

int CoutCast(signed char val) { return val; }






enum GenMode
{
UNIFORM,            
INTEGER_SEED,       
};


template <typename T>
__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, T &value, int index = 0)
{
switch (gen_mode)
{
case UNIFORM:
value = 2;
break;
case INTEGER_SEED:
default:
value = (T) index;
break;
}
}



__host__ __device__ __forceinline__ void InitValue(GenMode gen_mode, bool &value, int index = 0)
{
switch (gen_mode)
{
case UNIFORM:
value = true;
break;
case INTEGER_SEED:
default:
value = (index > 0);
break;
}
}



template <typename K>
void RandomBits(
K &key,
int entropy_reduction = 0,
int begin_bit = 0,
int end_bit = sizeof(K) * 8)
{
const int NUM_BYTES = sizeof(K);
const int WORD_BYTES = sizeof(unsigned int);
const int NUM_WORDS = (NUM_BYTES + WORD_BYTES - 1) / WORD_BYTES;

unsigned int word_buff[NUM_WORDS];

if (entropy_reduction == -1)
{
memset((void *) &key, 0, sizeof(key));
return;
}

if (end_bit < 0)
end_bit = sizeof(K) * 8;

while (true) 
{
for (int j = 0; j < NUM_WORDS; j++)
{
int current_bit = j * WORD_BYTES * 8;

unsigned int word = 0xffffffff;
word &= 0xffffffff << std::max(0, begin_bit - current_bit);
word &= 0xffffffff >> std::max(0, (current_bit + (WORD_BYTES * 8)) - end_bit);

for (int i = 0; i <= entropy_reduction; i++)
{
word &= mersenne::genrand_int32();
g_num_rand_samples++;                
}

word_buff[j] = word;
}

memcpy(&key, word_buff, sizeof(K));

K copy = key;
if (!IsNaN(copy))
break;          
}
}







template <typename S, typename T, typename OffsetT>
int CompareResults(T* computed, S* reference, OffsetT len, bool verbose = true)
{
for (OffsetT i = 0; i < len; i++)
{
if (computed[i] != reference[i])
{
if (verbose) std::cout << "INCORRECT: [" << i << "]: "
<< CoutCast(computed[i]) << " != "
<< CoutCast(reference[i]);
return 1;
}
}
return 0;
}



template <typename OffsetT>
int CompareResults(float* computed, float* reference, OffsetT len, bool verbose = true)
{
for (OffsetT i = 0; i < len; i++)
{
if (computed[i] != reference[i])
{
float difference = std::abs(computed[i]-reference[i]);
float fraction = difference / std::abs(reference[i]);

if (fraction > 0.0001)
{
if (verbose) std::cout << "INCORRECT: [" << i << "]: "
<< "(computed) " << CoutCast(computed[i]) << " != "
<< CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
return 1;
}
}
}
return 0;
}



template <typename OffsetT>
int CompareResults(double* computed, double* reference, OffsetT len, bool verbose = true)
{
for (OffsetT i = 0; i < len; i++)
{
if (computed[i] != reference[i])
{
double difference = std::abs(computed[i]-reference[i]);
double fraction = difference / std::abs(reference[i]);

if (fraction > 0.0001)
{
if (verbose) std::cout << "INCORRECT: [" << i << "]: "
<< CoutCast(computed[i]) << " != "
<< CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
return 1;
}
}
}
return 0;
}




template <typename S, typename T>
int CompareDeviceResults(
S *h_reference,
T *d_data,
size_t num_items,
bool verbose = true,
bool display_data = false)
{
T *h_data = (T*) malloc(num_items * sizeof(T));

hipMemcpy(h_data, d_data, sizeof(T) * num_items, hipMemcpyDeviceToHost);

if (display_data)
{
printf("Reference:\n");
for (int i = 0; i < int(num_items); i++)
{
std::cout << CoutCast(h_reference[i]) << ", ";
}
printf("\n\nComputed:\n");
for (int i = 0; i < int(num_items); i++)
{
std::cout << CoutCast(h_data[i]) << ", ";
}
printf("\n\n");
}

int retval = CompareResults(h_data, h_reference, num_items, verbose);

if (h_data) free(h_data);

return retval;
}




struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

LARGE_INTEGER ll_freq;
LARGE_INTEGER ll_start;
LARGE_INTEGER ll_stop;

CpuTimer()
{
QueryPerformanceFrequency(&ll_freq);
}

void Start()
{
QueryPerformanceCounter(&ll_start);
}

void Stop()
{
QueryPerformanceCounter(&ll_stop);
}

float ElapsedMillis()
{
double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

return float((stop - start) * 1000);
}

#else

rusage start;
rusage stop;

void Start()
{
getrusage(RUSAGE_SELF, &start);
}

void Stop()
{
getrusage(RUSAGE_SELF, &stop);
}

float ElapsedMillis()
{
float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

return (sec * 1000) + (usec / 1000);
}

#endif
};

struct GpuTimer
{
hipEvent_t start;
hipEvent_t stop;

GpuTimer()
{
hipEventCreate(&start);
hipEventCreate(&stop);
}

~GpuTimer()
{
hipEventDestroy(start);
hipEventDestroy(stop);
}

void Start()
{
hipEventRecord(start, 0);
}

void Stop()
{
hipEventRecord(stop, 0);
}

float ElapsedMillis()
{
float elapsed;
hipEventSynchronize(stop);
hipEventElapsedTime(&elapsed, start, stop);
return elapsed;
}
};
