



#ifndef COMMON_H
#define COMMON_H
#include <string>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include "ogr_api.h"
#include "mpi.h"
#include <algorithm>
#include <iostream>

#include <queue>  
#ifdef WINDOWS
#define _WINSOCKAPI_    
#include <windows.h>
#include <direct.h>
#include <time.h>
#else
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <cstdio>
#include <dlfcn.h>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#endif 
#ifdef SUPPORT_OMP
#include <omp.h>
#endif 

#define MCW MPI_COMM_WORLD
#define MAX_STRING_LENGTH 255
#define MAXLN 4096

using std::cout;
using std::endl;
using std::nothrow;

using std::queue;
using std::vector;

using std::ifstream;
using std::ios;

using std::abs;
using std::fabs;
using std::sqrt;


#define NOTFINISHED 0
#define FINISHED 1

#define TDVERSION "5.3.9"

enum DATA_TYPE {
SHORT_TYPE,
LONG_TYPE,
FLOAT_TYPE
};

struct node {
int x;
int y;
};
inline bool operator==(const node& n1, const node& n2)
{
return (n1.x == n2.x) && (n1.y == n2.y);
}
const double PI = 3.14159265359;
const double SQRT2 = 1.4142135623730951;
const int16_t MISSINGSHORT = -32768;
const int32_t MISSINGLONG = -2147483647;
const float MISSINGFLOAT = -1 * FLT_MAX;
const float MINEPS = 1E-5f;

const float DEFAULTNODATA = -9999.f;  
const int DEFAULTNODATA_INT = -9999;
const int OMPTHREADS = 4;
const float ZERO = 1.0e-12F;

const int d1[9] = {0, 1, 1, 0, -1, -1, -1, 0, 1}; 
const int d2[9] = {0, 0, -1, -1, -1, 0, 1, 1, 1}; 
const double e = 0.;
const double ne = PI * 0.25;
const double n = PI * 0.5;
const double nw = PI * 0.75;
const double w = PI;
const double sw = PI * 1.25;
const double s = PI * 1.5;
const double se = PI * 1.75;
const double dinfang[9] = {0., e, ne, n, nw, w, sw, s, se};

const int esri_flowdir[9] = {-1, 1, 128, 64, 32, 16, 8, 4, 2};


int nameadd(char *, char *, const char *);
double prop(float a, int k, double dx1, double dy1);
void getLayername(char *inputogrfile, char *layername);
const char *getOGRdrivername(char *datasrcnew);
void getlayerfail(OGRDataSourceH hDS1, char *outletsds, int outletslyr);
int readoutlets(char *outletsds,
char *lyrname,
int uselayername,
int outletslyr,
OGRSpatialReferenceH hSRSRaster,
int *noutlets,
double *&x,
double *&y);
int readoutlets(char *outletsds,
char *lyrname,
int uselayername,
int outletslyr,
OGRSpatialReferenceH hSRSraster,
int *noutlets,
double *&x,
double *&y,
int *&id);





template<typename T>
bool Initialize1DArray(int row, T *&data, T initialValue) {
if (nullptr != data) {
cout << "The input 1D array pointer is not nullptr, without initialized!" << endl;
return false;
}
data = new(nothrow)T[row];
if (nullptr == data) {
cout << "Bad memory allocated during 1D array initialization!" << endl;
return false;
}
for (int i = 0; i < row; i++) {
data[i] = initialValue;
}
return true;
}

template<typename T>
bool Initialize1DArray(int row, T *&data, const T *iniData) {
if (nullptr != data) {
cout << "The input 1D array pointer is not nullptr, without initialized!" << endl;
return false;
}
data = new(nothrow)T[row];
if (nullptr == data) {
cout << "Bad memory allocated during 1D array initialization!" << endl;
return false;
}
if (nullptr == iniData) {
cout << "The input parameter iniData MUST NOT be nullptr!" << endl;
return false;
}
for (int i = 0; i < row; i++) {
data[i] = iniData[i];
}
return true;
}

template<typename T>
bool Initialize2DArray(int row, int col, T **&data, T initialValue) {
if (nullptr != data) {
cout << "The input 2D array pointer is not nullptr, without initialized!" << endl;
return false;
}
data = new(nothrow)T *[row];
if (nullptr == data) {
cout << "Bad memory allocated during 2D array initialization!" << endl;
return false;
}
int badAlloc = 0;
for (int i = 0; i < row; i++) {
data[i] = new(nothrow)T[col];
if (nullptr == data[i]) {
badAlloc++;
}
for (int j = 0; j < col; j++) {
data[i][j] = initialValue;
}
}
if (badAlloc > 0) {
cout << "Bad memory allocated during 2D array initialization!" << endl;
Release2DArray(row, data);
return false;
}
return true;
}

template<typename T>
bool Initialize2DArray(int row, int col, T **&data, const T *const *iniData) {
if (nullptr != data) {
cout << "The input 2D array pointer is not nullptr, without initialized!" << endl;
return false;
}
data = new(nothrow)T *[row];
if (nullptr == data) {
cout << "Bad memory allocated during 2D array initialization!" << endl;
return false;
}
int badAlloc = 0;
int errorAccess = 0;
#pragma omp parallel for reduction(+:badAlloc, errorAccess)
for (int i = 0; i < row; i++) {
data[i] = new(nothrow)T[col];
if (nullptr == data[i]) {
badAlloc++;
}
if (nullptr == iniData[i]) {
errorAccess++;
}
else {
for (int j = 0; j < col; j++) {
data[i][j] = iniData[i][j];
}
}
}
if (badAlloc > 0) {
cout << "Bad memory allocated during 2D array initialization!" << endl;
Release2DArray(row, data);
return false;
}
if (errorAccess > 0) {
cout << "nullptr pointer existed in iniData during 2D array initialization!" << endl;
Release2DArray(row, data);
return false;
}
return true;
}

template<typename T>
void Release1DArray(T *&data) {
if (nullptr != data) {
delete[] data;
data = nullptr;
}
}


template<typename T>
void Release2DArray(int row, T **&data) {
if (nullptr == data) {
return;
}
for (int i = 0; i < row; i++) {
if (data[i] != nullptr) {
delete[] data[i];
data[i] = nullptr;
}
}
delete[] data;
data = nullptr;
}

char* convertStringToCharPtr(const std::string& s);

double TimeCounting();
#ifdef MSVC
#define stringcat strcat_s
#define stringcpy strcpy_s
#define stringscan sscanf_s
#define stringprintf sprintf_s
#else
#define stringcat strcat
#define stringcpy strcpy
#define stringscan sscanf
#define stringprintf sprintf
#endif 

#endif 
