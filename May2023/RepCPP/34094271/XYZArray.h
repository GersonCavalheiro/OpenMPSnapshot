
#ifndef XYZ_ARRAY_H
#define XYZ_ARRAY_H

#include "BasicTypes.h"
#include <string.h> 
#include <stdio.h> 
#include <cmath>
#include <utility>      
#include <algorithm>      
#ifdef _OPENMP
#include <omp.h>
#endif
#include "EnsemblePreprocessor.h"

class BoxDimensions;



class XYZArray
{
public:
friend class BoxDimensions;

XYZArray(void) : x(NULL), y(NULL), z(NULL), count(0),
allocDone(false) {}

explicit XYZArray(const uint n)
{
allocDone = false;
Init(n);
}

XYZArray(const std::vector<XYZ> & AOS)
{
allocDone = false;
Init(AOS.size());
for(int i = 0; i < AOS.size(); ++i){
x[i] = AOS[i].x;
y[i] = AOS[i].y;
z[i] = AOS[i].z;
}
}

XYZArray(const XYZArray& other);
XYZArray& operator=(XYZArray other);
bool operator==(const XYZArray & other);

friend void swap(XYZArray& a1, XYZArray& a2);
~XYZArray(void)
{
Uninit();
}

void Uninit();

void Init(const uint n);

uint Count() const
{
return count;
}

XYZ operator[](const uint i) const
{
return Get(i);
}

XYZ Get(const uint i) const
{
return XYZ(x[i], y[i], z[i]);
}

double Min(const uint i) const
{
double result = x[i];
if (result > y[i])
result = y[i];
if (result > z[i])
result = z[i];
return result;
}

double Max(const uint i) const
{
double result = x[i];
if (y[i] > result)
result = y[i];
if (z[i] > result)
result = z[i];
return result;
}


void Set(const uint i, const double a, const double b, const double c)
{
x[i] = a;
y[i] = b;
z[i] = c;
}

void SetRange(const uint start, const uint stop,
const double a, const double b, const double c);

void Set(const uint i, XYZ const& val)
{
Set(i, val.x, val.y, val.z);
}

void SetRange(const uint start, const uint stop, XYZ const& val);

void ResetRange(const uint val, const uint stop);

void Reset();


void Add(XYZArray & src, const uint i, const uint srcI)
{
x[i] += src.x[srcI];
y[i] += src.y[srcI];
z[i] += src.z[srcI];
}

XYZ Sum(const XYZArray & src, const uint i, const uint srcI) const
{
return XYZ(x[i] + src.x[srcI], y[i] + src.y[srcI], z[i] + src.z[srcI]);
}

void Sub(XYZArray & src, const uint i, const uint srcI)
{
x[i] -= src.x[srcI];
y[i] -= src.y[srcI];
z[i] -= src.z[srcI];
}

XYZ Difference(const uint i, const uint j) const
{
return XYZ(x[i] - x[j], y[i] - y[j], z[i] - z[j]);
}

double DifferenceX(const uint i, const uint j) const
{
return x[i] - x[j];
}

double DifferenceY(const uint i, const uint j) const
{
return y[i] - y[j];
}

double DifferenceZ(const uint i, const uint j) const
{
return z[i] - z[j];
}

double Length(const uint i) const
{
return sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
}

double AdjointMatrix(XYZArray &Inv);

XYZ Difference(const uint i, XYZArray const& other,
const uint otherI) const
{
return XYZ( x[i] - other.x[otherI], y[i] - other.y[otherI],
z[i] - other.z[otherI]);
}

double DifferenceX(const uint i, XYZArray const& other,
const uint otherI) const
{
return x[i] - other.x[otherI];
}

double DifferenceY(const uint i, XYZArray const& other,
const uint otherI) const
{
return y[i] - other.y[otherI];
}

double DifferenceZ(const uint i, XYZArray const& other,
const uint otherI) const
{
return z[i] - other.z[otherI];
}


void Add(const uint i, const uint j)
{
x[i] += x[j];
y[i] += y[j];
z[i] += z[j];
}

void Sub(const uint i, const uint j)
{
x[i] -= x[j];
y[i] -= y[j];
z[i] -= z[j];
}

void Add(const uint i, const double a, const double b, const double c)
{
x[i] += a;
y[i] += b;
z[i] += c;
}

void Add(const uint i, XYZ const& val)
{
x[i] += val.x;
y[i] += val.y;
z[i] += val.z;
}

void Sub(const uint i, XYZ const& val)
{
x[i] -= val.x;
y[i] -= val.y;
z[i] -= val.z;
}

void Sub(const uint i, const double a, const double b, const double c)
{
x[i] -= a;
y[i] -= b;
z[i] -= c;
}

void Scale(const uint i, const double a, const double b, const double c)
{
x[i] *= a;
y[i] *= b;
z[i] *= c;
}

void Add(const uint i, const double val)
{
x[i] += val;
y[i] += val;
z[i] += val;
}

void Sub(const uint i, const double val)
{
x[i] -= val;
y[i] -= val;
z[i] -= val;
}

void Scale(const uint i, const double val)
{
x[i] *= val;
y[i] *= val;
z[i] *= val;
}

void Scale(const uint i, XYZ const& val)
{
x[i] *= val.x;
y[i] *= val.y;
z[i] *= val.z;
}

void AddRange(const uint start, const uint stop, XYZ const& val);

void SubRange(const uint start, const uint stop, XYZ const& val);

void ScaleRange(const uint start, const uint stop, XYZ const& val);

void AddRange(const uint start, const uint stop, const double val);

void SubRange(const uint start, const uint stop, const double val);

void ScaleRange(const uint start, const uint stop, const double val);

void Set(XYZArray & dest, const uint srcIndex, const uint destIndex) const
{
dest.x[destIndex] = x[srcIndex];
dest.y[destIndex] = y[srcIndex];
dest.z[destIndex] = z[srcIndex];
}

void AddAll(XYZ const& val);

void SubAll(XYZ const& val);

void ScaleAll(XYZ const& val);

void AddAll(const double val);

void SubAll(const double val);

void ScaleAll(const double val);


void CopyRange(XYZArray & dest, const uint srcIndex, const uint destIndex,
const uint len) const;

double * x, * y, * z;

protected:
uint count;
bool allocDone;
};


inline XYZArray::XYZArray(XYZArray const& other)
{
count = other.count;
x = new double[count];
y = new double[count];
z = new double[count];
allocDone = true;
memcpy(x, other.x, sizeof(double) * count);
memcpy(y, other.y, sizeof(double) * count);
memcpy(z, other.z, sizeof(double) * count);
}

inline XYZArray& XYZArray::operator=(XYZArray other)
{
swap(*this, other);
return *this;
}

inline bool XYZArray::operator==(const XYZArray & other){
bool result = true;
for(uint i = 0; i < count; i++) {
result &= (x[i] == other.x[i]);
result &= (y[i] == other.y[i]);
result &= (z[i] == other.z[i]);
}
return result;
}


inline void swap(XYZArray& a1, XYZArray& a2)
{
using std::swap;
swap(a1.x, a2.x);
swap(a1.y, a2.y);
swap(a1.z, a2.z);
swap(a1.count, a2.count);
swap(a1.allocDone, a2.allocDone);
}

inline void XYZArray::Uninit()
{
if (x != NULL)
delete[] x;
if (y != NULL)
delete[] y;
if (z != NULL)
delete[] z;
allocDone = false;
}

inline void XYZArray::SetRange(const uint start, const uint stop,
const double a, const double b, const double c)
{
SetRange(start, stop, XYZ(a, b, c));
}

inline void XYZArray::SetRange(const uint start, const uint stop,
XYZ const& val)
{
for(uint i = start; i < stop ; ++i) {
x[i] = val.x;
y[i] = val.y;
z[i] = val.z;
}
}

inline void XYZArray::ResetRange(const uint val, const uint stop)
{
#ifdef _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel default(none) shared(val, stop)
#else
#pragma omp parallel default(none)
#endif
#endif
{
memset(this->x, val, stop * sizeof(double));
memset(this->y, val, stop * sizeof(double));
memset(this->z, val, stop * sizeof(double));
}
}

inline void XYZArray::Reset()
{
ResetRange(0, count);
}

inline void XYZArray::Init(const uint n)
{
count = n;

if (allocDone) {
allocDone = false;
if (x != NULL)
delete[] x;
if (y != NULL)
delete[] y;
if (z != NULL)
delete[] z;
}

x = new double[n];
y = new double[n];
z = new double[n];

for(uint i = 0; i < n; i++) {
x[i] = 0.0;
y[i] = 0.0;
z[i] = 0.0;
}

allocDone = true;
}

inline void XYZArray::AddRange(const uint start, const uint stop,
XYZ const& val)
{
for(uint i = start; i < stop ; ++i) {
x[i] += val.x;
y[i] += val.y;
z[i] += val.z;
}
}

inline void XYZArray::SubRange(const uint start, const uint stop,
XYZ const& val)
{
for(uint i = start; i < stop ; ++i) {
x[i] -= val.x;
y[i] -= val.y;
z[i] -= val.z;
}
}

inline void XYZArray::ScaleRange(const uint start, const uint stop,
XYZ const& val)
{
for(uint i = start; i < stop ; ++i) {
x[i] *= val.x;
y[i] *= val.y;
z[i] *= val.z;
}
}

inline void XYZArray::AddRange(const uint start, const uint stop,
const double val)
{
for(uint i = start; i < stop ; ++i) {
x[i] += val;
y[i] += val;
z[i] += val;
}
}

inline void XYZArray::SubRange(const uint start, const uint stop,
const double val)
{
for(uint i = start; i < stop ; ++i) {
x[i] -= val;
y[i] -= val;
z[i] -= val;
}
}

inline void XYZArray::ScaleRange(const uint start, const uint stop,
const double val)
{
for(uint i = start; i < stop ; ++i) {
x[i] *= val;
y[i] *= val;
z[i] *= val;
}
}

inline void XYZArray::AddAll(XYZ const& val)
{
AddRange(0, count, val);
}

inline void XYZArray::SubAll(XYZ const& val)
{
SubRange(0, count, val);
}

inline void XYZArray::ScaleAll(XYZ const& val)
{
ScaleRange(0, count, val);
}

inline void XYZArray::AddAll(const double val)
{
AddRange(0, count, val);
}

inline void XYZArray::SubAll(const double val)
{
SubRange(0, count, val);
}

inline void XYZArray::ScaleAll(const double val)
{
ScaleRange(0, count, val);
}

inline void XYZArray::CopyRange(XYZArray & dest, const uint srcIndex,
const uint destIndex, const uint len) const
{
#ifdef _OPENMP
#if GCC_VERSION >= 90000
#pragma omp parallel default(none) shared(dest, len, srcIndex, destIndex)
#else
#pragma omp parallel default(none) shared(dest)
#endif
#endif
{
memcpy(dest.x + destIndex, x + srcIndex, len * sizeof(double));
memcpy(dest.y + destIndex, y + srcIndex, len * sizeof(double));
memcpy(dest.z + destIndex, z + srcIndex, len * sizeof(double));
}
}

inline double XYZArray::AdjointMatrix(XYZArray &Inv)
{
Inv.x[0] = y[1] * z[2] - y[2] * z[1];
Inv.y[0] = y[2] * z[0] - y[0] * z[2];
Inv.z[0] = y[0] * z[1] - y[1] * z[0];

Inv.x[1] = x[2] * z[1] - x[1] * z[2];
Inv.y[1] = x[0] * z[2] - x[2] * z[0];
Inv.z[1] = x[1] * z[0] - x[0] * z[1];

Inv.x[2] = x[1] * y[2] - x[2] * y[1];
Inv.y[2] = x[2] * y[0] - x[0] * y[2];
Inv.z[2] = x[0] * y[1] - x[1] * y[0];

double det = x[0] * Inv.x[0] + x[1] * Inv.y[0] + x[2] * Inv.z[0];
return det;
}

#endif 
