#pragma once

#ifndef __HDK_UT_SolidAngle_h__
#define __HDK_UT_SolidAngle_h__

#include "BVH.h"

#include "fixedVector.h"
#include "SYS_Math.h"
#include <memory>

namespace HDK_Sample {

template<typename T>
using UT_Vector2T = UT_FixedVector<T,2>;
template<typename T>
using UT_Vector3T = UT_FixedVector<T,3>;

template <typename T>
SYS_FORCE_INLINE T cross(const UT_Vector2T<T> &v1, const UT_Vector2T<T> &v2)
{
return v1[0]*v2[1] - v1[1]*v2[0];
}

template <typename T>
SYS_FORCE_INLINE
UT_Vector3T<T> cross(const UT_Vector3T<T> &v1, const UT_Vector3T<T> &v2)
{
UT_Vector3T<T> result;
result[0] = v1[1]*v2[2] - v1[2]*v2[1];
result[1] = v1[2]*v2[0] - v1[0]*v2[2];
result[2] = v1[0]*v2[1] - v1[1]*v2[0];
return result;
}

template<typename T>
T UTsignedSolidAngleTri(
const UT_Vector3T<T> &a,
const UT_Vector3T<T> &b,
const UT_Vector3T<T> &c,
const UT_Vector3T<T> &query)
{
UT_Vector3T<T> qa = a-query;
UT_Vector3T<T> qb = b-query;
UT_Vector3T<T> qc = c-query;

const T alength = qa.length();
const T blength = qb.length();
const T clength = qc.length();

if (alength == 0 || blength == 0 || clength == 0)
return T(0);

qa /= alength;
qb /= blength;
qc /= clength;

const T numerator = dot(qa, cross(qb-qa, qc-qa));

if (numerator == 0)
return T(0);

const T denominator = T(1) + dot(qa,qb) + dot(qa,qc) + dot(qb,qc);

return T(2)*SYSatan2(numerator, denominator);
}

template<typename T>
T UTsignedSolidAngleQuad(
const UT_Vector3T<T> &a,
const UT_Vector3T<T> &b,
const UT_Vector3T<T> &c,
const UT_Vector3T<T> &d,
const UT_Vector3T<T> &query)
{
UT_Vector3T<T> v[4] = {
a-query,
b-query,
c-query,
d-query
};

const T lengths[4] = {
v[0].length(),
v[1].length(),
v[2].length(),
v[3].length()
};

if (lengths[0] == T(0) || lengths[1] == T(0) || lengths[2] == T(0) || lengths[3] == T(0))
return T(0);

v[0] /= lengths[0];
v[1] /= lengths[1];
v[2] /= lengths[2];
v[3] /= lengths[3];


const UT_Vector3T<T> diag02 = v[2]-v[0];
const UT_Vector3T<T> diag13 = v[3]-v[1];
const UT_Vector3T<T> v01 = v[1]-v[0];
const UT_Vector3T<T> v23 = v[3]-v[2];

T bary[4];
bary[0] = dot(v[3],cross(v23,diag13));
bary[1] = -dot(v[2],cross(v23,diag02));
bary[2] = -dot(v[1],cross(v01,diag13));
bary[3] = dot(v[0],cross(v01,diag02));

const T dot01 = dot(v[0],v[1]);
const T dot12 = dot(v[1],v[2]);
const T dot23 = dot(v[2],v[3]);
const T dot30 = dot(v[3],v[0]);

T omega = T(0);

if (bary[0]*bary[2] < bary[1]*bary[3])
{
const T numerator012 = bary[3];
const T numerator023 = bary[1];
const T dot02 = dot(v[0],v[2]);

if (numerator012 != T(0))
{
const T denominator012 = T(1) + dot01 + dot12 + dot02;
omega = SYSatan2(numerator012, denominator012);
}
if (numerator023 != T(0))
{
const T denominator023 = T(1) + dot02 + dot23 + dot30;
omega += SYSatan2(numerator023, denominator023);
}
}
else
{
const T numerator013 = -bary[2];
const T numerator123 = -bary[0];
const T dot13 = dot(v[1],v[3]);

if (numerator013 != T(0))
{
const T denominator013 = T(1) + dot01 + dot13 + dot30;
omega = SYSatan2(numerator013, denominator013);
}
if (numerator123 != T(0))
{
const T denominator123 = T(1) + dot12 + dot23 + dot13;
omega += SYSatan2(numerator123, denominator123);
}
}
return T(2)*omega;
}

template<typename T,typename S>
class UT_SolidAngle
{
public:
UT_SolidAngle();
~UT_SolidAngle();

UT_SolidAngle(
const int ntriangles,
const int *const triangle_points,
const int npoints,
const UT_Vector3T<S> *const positions,
const int order = 2)
: UT_SolidAngle()
{ init(ntriangles, triangle_points, npoints, positions, order); }

void init(
const int ntriangles,
const int *const triangle_points,
const int npoints,
const UT_Vector3T<S> *const positions,
const int order = 2);

void clear();

bool isClear() const
{ return myNTriangles == 0; }

T computeSolidAngle(const UT_Vector3T<T> &query_point, const T accuracy_scale = T(2.0)) const;

private:
struct BoxData;

static constexpr uint BVH_N = 4;
UT_BVH<BVH_N> myTree;
int myNBoxes;
int myOrder;
std::unique_ptr<BoxData[]> myData;
int myNTriangles;
const int *myTrianglePoints;
int myNPoints;
const UT_Vector3T<S> *myPositions;
};

template<typename T>
T UTsignedAngleSegment(
const UT_Vector2T<T> &a,
const UT_Vector2T<T> &b,
const UT_Vector2T<T> &query)
{
UT_Vector2T<T> qa = a-query;
UT_Vector2T<T> qb = b-query;

if (qa.isZero() || qb.isZero())
return T(0);

const T numerator = cross(qa, qb);

if (numerator == 0)
return T(0);

const T denominator = dot(qa,qb);

return SYSatan2(numerator, denominator);
}

template<typename T,typename S>
class UT_SubtendedAngle
{
public:
UT_SubtendedAngle();
~UT_SubtendedAngle();

UT_SubtendedAngle(
const int nsegments,
const int *const segment_points,
const int npoints,
const UT_Vector2T<S> *const positions,
const int order = 2)
: UT_SubtendedAngle()
{ init(nsegments, segment_points, npoints, positions, order); }

void init(
const int nsegments,
const int *const segment_points,
const int npoints,
const UT_Vector2T<S> *const positions,
const int order = 2);

void clear();

bool isClear() const
{ return myNSegments == 0; }

T computeAngle(const UT_Vector2T<T> &query_point, const T accuracy_scale = T(2.0)) const;

private:
struct BoxData;

static constexpr uint BVH_N = 4;
UT_BVH<BVH_N> myTree;
int myNBoxes;
int myOrder;
std::unique_ptr<BoxData[]> myData;
int myNSegments;
const int *mySegmentPoints;
int myNPoints;
const UT_Vector2T<S> *myPositions;
};

} 
#endif
