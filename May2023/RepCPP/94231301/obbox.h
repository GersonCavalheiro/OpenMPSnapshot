
#pragma once

#include "bbox.h"
#include "linearspace3.h"

namespace embree
{

template<typename T>
struct OBBox 
{
public:

__forceinline OBBox () {}

__forceinline OBBox (EmptyTy) 
: space(one), bounds(empty) {}

__forceinline OBBox (const BBox<T>& bounds) 
: space(one), bounds(bounds) {}

__forceinline OBBox (const LinearSpace3<T>& space, const BBox<T>& bounds) 
: space(space), bounds(bounds) {}

friend std::ostream& operator<<(std::ostream& cout, const OBBox& p) {
return std::cout << "{ space = " << p.space << ", bounds = " << p.bounds << "}";
}

public:
LinearSpace3<T> space; 
BBox<T> bounds;        
};

typedef OBBox<Vec3f> OBBox3f;
typedef OBBox<Vec3fa> OBBox3fa;
}
