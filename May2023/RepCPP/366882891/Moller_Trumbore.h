#pragma once

#include "geometry.h"

#define EPSILON 0.000001
#define CULLING

bool rayTriangleIntersect(
const vector3D& orig, const vector3D& dir,
tryangle tr,
float& dist)
{

vector3D v0v2 = tr.Get3() - tr.Get1();
vector3D v0v1 = tr.Get2() - tr.Get1();
vector3D pvec = dir.cross(v0v2);
float det = v0v1.dot(pvec);
#ifdef CULLING 
if (det < EPSILON) return false;
#else 
if (det < EPSILON && det > -EPSILON) return false;
#endif 
float invDet = 1 / det;

vector3D tvec = orig - tr.Get1();
float u = tvec.dot(pvec) * invDet;
if (u < 0 || u > 1) return false;

vector3D qvec = tvec.cross(v0v1);
float v = dir.dot(qvec) * invDet;
if (v < 0 || u + v > 1) return false;

dist = v0v2.dot(qvec) * invDet;

return true;
}
