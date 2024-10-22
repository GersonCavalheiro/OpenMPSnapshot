#pragma once

#ifndef __HDK_UT_BVH_h__
#define __HDK_UT_BVH_h__

#include "fixedVector.h"
#include "smallArray.h"
#include "SYS_Types.h"
#include <limits>
#include <memory>

template<typename T> class UT_Array;
class v4uf;
class v4uu;

namespace HDK_Sample {

namespace UT {

template<typename T,uint NAXES>
struct Box {
T vals[NAXES][2];

SYS_FORCE_INLINE Box() noexcept = default;
SYS_FORCE_INLINE constexpr Box(const Box &other) noexcept = default;
SYS_FORCE_INLINE constexpr Box(Box &&other) noexcept = default;
SYS_FORCE_INLINE Box& operator=(const Box &other) noexcept = default;
SYS_FORCE_INLINE Box& operator=(Box &&other) noexcept = default;

template<typename S>
SYS_FORCE_INLINE Box(const Box<S,NAXES>& other) noexcept {
static_assert((std::is_pod<Box<T,NAXES>>::value) || !std::is_pod<T>::value,
"UT::Box should be POD, for better performance in UT_Array, etc.");

for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = T(other.vals[axis][0]);
vals[axis][1] = T(other.vals[axis][1]);
}
}
template<typename S,bool INSTANTIATED>
SYS_FORCE_INLINE Box(const UT_FixedVector<S,NAXES,INSTANTIATED>& pt) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = pt[axis];
vals[axis][1] = pt[axis];
}
}
template<typename S>
SYS_FORCE_INLINE Box& operator=(const Box<S,NAXES>& other) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = T(other.vals[axis][0]);
vals[axis][1] = T(other.vals[axis][1]);
}
return *this;
}

SYS_FORCE_INLINE const T* operator[](const size_t axis) const noexcept {
UT_ASSERT_P(axis < NAXES);
return vals[axis];
}
SYS_FORCE_INLINE T* operator[](const size_t axis) noexcept {
UT_ASSERT_P(axis < NAXES);
return vals[axis];
}

SYS_FORCE_INLINE void initBounds() noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = std::numeric_limits<T>::max();
vals[axis][1] = -std::numeric_limits<T>::max();
}
}
SYS_FORCE_INLINE void initBounds(const Box<T,NAXES>& src) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = src.vals[axis][0];
vals[axis][1] = src.vals[axis][1];
}
}
SYS_FORCE_INLINE void initBoundsUnordered(const Box<T,NAXES>& src0, const Box<T,NAXES>& src1) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = SYSmin(src0.vals[axis][0], src1.vals[axis][0]);
vals[axis][1] = SYSmax(src0.vals[axis][1], src1.vals[axis][1]);
}
}
SYS_FORCE_INLINE void combine(const Box<T,NAXES>& src) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
T& minv = vals[axis][0];
T& maxv = vals[axis][1];
const T curminv = src.vals[axis][0];
const T curmaxv = src.vals[axis][1];
minv = (minv < curminv) ? minv : curminv;
maxv = (maxv > curmaxv) ? maxv : curmaxv;
}
}
SYS_FORCE_INLINE void enlargeBounds(const Box<T,NAXES>& src) noexcept {
combine(src);
}

template<typename S,bool INSTANTIATED>
SYS_FORCE_INLINE
void initBounds(const UT_FixedVector<S,NAXES,INSTANTIATED>& pt) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = pt[axis];
vals[axis][1] = pt[axis];
}
}
template<bool INSTANTIATED>
SYS_FORCE_INLINE
void initBounds(const UT_FixedVector<T,NAXES,INSTANTIATED>& min, const UT_FixedVector<T,NAXES,INSTANTIATED>& max) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = min[axis];
vals[axis][1] = max[axis];
}
}
template<bool INSTANTIATED>
SYS_FORCE_INLINE
void initBoundsUnordered(const UT_FixedVector<T,NAXES,INSTANTIATED>& p0, const UT_FixedVector<T,NAXES,INSTANTIATED>& p1) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = SYSmin(p0[axis], p1[axis]);
vals[axis][1] = SYSmax(p0[axis], p1[axis]);
}
}
template<bool INSTANTIATED>
SYS_FORCE_INLINE
void enlargeBounds(const UT_FixedVector<T,NAXES,INSTANTIATED>& pt) noexcept {
for (uint axis = 0; axis < NAXES; ++axis) {
vals[axis][0] = SYSmin(vals[axis][0], pt[axis]);
vals[axis][1] = SYSmax(vals[axis][1], pt[axis]);
}
}

SYS_FORCE_INLINE
UT_FixedVector<T,NAXES> getMin() const noexcept {
UT_FixedVector<T,NAXES> v;
for (uint axis = 0; axis < NAXES; ++axis) {
v[axis] = vals[axis][0];
}
return v;
}

SYS_FORCE_INLINE
UT_FixedVector<T,NAXES> getMax() const noexcept {
UT_FixedVector<T,NAXES> v;
for (uint axis = 0; axis < NAXES; ++axis) {
v[axis] = vals[axis][1];
}
return v;
}

T diameter2() const noexcept {
T diff = (vals[0][1]-vals[0][0]);
T sum = diff*diff;
for (uint axis = 1; axis < NAXES; ++axis) {
diff = (vals[axis][1]-vals[axis][0]);
sum += diff*diff;
}
return sum;
}
T volume() const noexcept {
T product = (vals[0][1]-vals[0][0]);
for (uint axis = 1; axis < NAXES; ++axis) {
product *= (vals[axis][1]-vals[axis][0]);
}
return product;
}
T half_surface_area() const noexcept {
if (NAXES==1) {
return (vals[0][1]-vals[0][0]);
}
if (NAXES==2) {
const T d0 = (vals[0][1]-vals[0][0]);
const T d1 = (vals[1][1]-vals[1][0]);
return d0 + d1;
}
if (NAXES==3) {
const T d0 = (vals[0][1]-vals[0][0]);
const T d1 = (vals[1][1]-vals[1][0]);
const T d2 = (vals[2][1]-vals[2][0]);
return d0*d1 + d1*d2 + d2*d0;
}
if (NAXES==4) {
const T d0 = (vals[0][1]-vals[0][0]);
const T d1 = (vals[1][1]-vals[1][0]);
const T d2 = (vals[2][1]-vals[2][0]);
const T d3 = (vals[3][1]-vals[3][0]);
const T d0d1 = d0*d1;
const T d2d3 = d2*d3;
return d0d1*(d2+d3) + d2d3*(d0+d1);
}

T sum = 0;
for (uint skipped_axis = 0; skipped_axis < NAXES; ++skipped_axis) {
T product = 1;
for (uint axis = 0; axis < NAXES; ++axis) {
if (axis != skipped_axis) {
product *= (vals[axis][1]-vals[axis][0]);
}
}
sum += product;
}
return sum;
}
T axis_sum() const noexcept {
T sum = (vals[0][1]-vals[0][0]);
for (uint axis = 1; axis < NAXES; ++axis) {
sum += (vals[axis][1]-vals[axis][0]);
}
return sum;
}
template<bool INSTANTIATED0,bool INSTANTIATED1>
SYS_FORCE_INLINE void intersect(
T &box_tmin,
T &box_tmax,
const UT_FixedVector<uint,NAXES,INSTANTIATED0> &signs,
const UT_FixedVector<T,NAXES,INSTANTIATED1> &origin,
const UT_FixedVector<T,NAXES,INSTANTIATED1> &inverse_direction
) const noexcept {
for (int axis = 0; axis < NAXES; ++axis)
{
uint sign = signs[axis];
T t1 = (vals[axis][sign]   - origin[axis]) * inverse_direction[axis];
T t2 = (vals[axis][sign^1] - origin[axis]) * inverse_direction[axis];
box_tmin = SYSmax(t1, box_tmin);
box_tmax = SYSmin(t2, box_tmax);
}
}
SYS_FORCE_INLINE void intersect(const Box& other, Box& dest) const noexcept {
for (int axis = 0; axis < NAXES; ++axis)
{
dest.vals[axis][0] = SYSmax(vals[axis][0], other.vals[axis][0]);
dest.vals[axis][1] = SYSmin(vals[axis][1], other.vals[axis][1]);
}
}
template<bool INSTANTIATED>
SYS_FORCE_INLINE T minDistance2(
const UT_FixedVector<T,NAXES,INSTANTIATED> &p
) const noexcept {
T diff = SYSmax(SYSmax(vals[0][0]-p[0], p[0]-vals[0][1]), T(0.0f));
T d2 = diff*diff;
for (int axis = 1; axis < NAXES; ++axis)
{
diff = SYSmax(SYSmax(vals[axis][0]-p[axis], p[axis]-vals[axis][1]), T(0.0f));
d2 += diff*diff;
}
return d2;
}
template<bool INSTANTIATED>
SYS_FORCE_INLINE T maxDistance2(
const UT_FixedVector<T,NAXES,INSTANTIATED> &p
) const noexcept {
T diff = SYSmax(p[0]-vals[0][0], vals[0][1]-p[0]);
T d2 = diff*diff;
for (int axis = 1; axis < NAXES; ++axis)
{
diff = SYSmax(p[axis]-vals[axis][0], vals[axis][1]-p[axis]);
d2 += diff*diff;
}
return d2;
}
};

enum class BVH_Heuristic {
BOX_PERIMETER,

BOX_AREA,

BOX_VOLUME,

BOX_RADIUS,

BOX_RADIUS2,

BOX_RADIUS3,

MEDIAN_MAX_AXIS
};

template<uint N>
class BVH {
public:
using INT_TYPE = uint;
struct Node {
INT_TYPE child[N];

static constexpr INT_TYPE theN = N;
static constexpr INT_TYPE EMPTY = INT_TYPE(-1);
static constexpr INT_TYPE INTERNAL_BIT = (INT_TYPE(1)<<(sizeof(INT_TYPE)*8 - 1));
SYS_FORCE_INLINE static INT_TYPE markInternal(INT_TYPE internal_node_num) noexcept {
return internal_node_num | INTERNAL_BIT;
}
SYS_FORCE_INLINE static bool isInternal(INT_TYPE node_int) noexcept {
return (node_int & INTERNAL_BIT) != 0;
}
SYS_FORCE_INLINE static INT_TYPE getInternalNum(INT_TYPE node_int) noexcept {
return node_int & ~INTERNAL_BIT;
}
};
private:
struct FreeDeleter {
SYS_FORCE_INLINE void operator()(Node* p) const {
if (p) {
free(p);
}
}
};

std::unique_ptr<Node[],FreeDeleter> myRoot;
INT_TYPE myNumNodes;
public:
SYS_FORCE_INLINE BVH() noexcept : myRoot(nullptr), myNumNodes(0) {}

template<BVH_Heuristic H,typename T,uint NAXES,typename BOX_TYPE,typename SRC_INT_TYPE=INT_TYPE>
void init(const BOX_TYPE* boxes, const INT_TYPE nboxes, SRC_INT_TYPE* indices=nullptr, bool reorder_indices=false, INT_TYPE max_items_per_leaf=1) noexcept;

template<BVH_Heuristic H,typename T,uint NAXES,typename BOX_TYPE,typename SRC_INT_TYPE=INT_TYPE>
void init(Box<T,NAXES> axes_minmax, const BOX_TYPE* boxes, INT_TYPE nboxes, SRC_INT_TYPE* indices=nullptr, bool reorder_indices=false, INT_TYPE max_items_per_leaf=1) noexcept;

SYS_FORCE_INLINE
INT_TYPE getNumNodes() const noexcept
{
return myNumNodes;
}
SYS_FORCE_INLINE
const Node *getNodes() const noexcept
{
return myRoot.get();
}

SYS_FORCE_INLINE
void clear() noexcept {
myRoot.reset();
myNumNodes = 0;
}

template<typename LOCAL_DATA,typename FUNCTORS>
void traverse(
FUNCTORS &functors,
LOCAL_DATA *data_for_parent=nullptr) const noexcept;

template<typename LOCAL_DATA,typename FUNCTORS>
void traverseParallel(
INT_TYPE parallel_threshold,
FUNCTORS &functors,
LOCAL_DATA *data_for_parent=nullptr) const noexcept;

template<typename LOCAL_DATA,typename FUNCTORS>
void traverseVector(
FUNCTORS &functors,
LOCAL_DATA *data_for_parent=nullptr) const noexcept;

void debugDump() const;

template<typename SRC_INT_TYPE>
static void createTrivialIndices(SRC_INT_TYPE* indices, const INT_TYPE n) noexcept;

private:
template<typename LOCAL_DATA,typename FUNCTORS>
void traverseHelper(
INT_TYPE nodei,
INT_TYPE parent_nodei,
FUNCTORS &functors,
LOCAL_DATA *data_for_parent=nullptr) const noexcept;

template<typename LOCAL_DATA,typename FUNCTORS>
void traverseParallelHelper(
INT_TYPE nodei,
INT_TYPE parent_nodei,
INT_TYPE parallel_threshold,
INT_TYPE next_node_id,
FUNCTORS &functors,
LOCAL_DATA *data_for_parent=nullptr) const noexcept;

template<typename LOCAL_DATA,typename FUNCTORS>
void traverseVectorHelper(
INT_TYPE nodei,
INT_TYPE parent_nodei,
FUNCTORS &functors,
LOCAL_DATA *data_for_parent=nullptr) const noexcept;

template<typename T,uint NAXES,typename BOX_TYPE,typename SRC_INT_TYPE>
static void computeFullBoundingBox(Box<T,NAXES>& axes_minmax, const BOX_TYPE* boxes, const INT_TYPE nboxes, SRC_INT_TYPE* indices) noexcept;

template<BVH_Heuristic H,typename T,uint NAXES,typename BOX_TYPE,typename SRC_INT_TYPE>
static void initNode(UT_Array<Node>& nodes, Node &node, const Box<T,NAXES>& axes_minmax, const BOX_TYPE* boxes, SRC_INT_TYPE* indices, const INT_TYPE nboxes) noexcept;

template<BVH_Heuristic H,typename T,uint NAXES,typename BOX_TYPE,typename SRC_INT_TYPE>
static void initNodeReorder(UT_Array<Node>& nodes, Node &node, const Box<T,NAXES>& axes_minmax, const BOX_TYPE* boxes, SRC_INT_TYPE* indices, const INT_TYPE nboxes, const INT_TYPE indices_offset, const INT_TYPE max_items_per_leaf) noexcept;

template<BVH_Heuristic H,typename T,uint NAXES,typename BOX_TYPE,typename SRC_INT_TYPE>
static void multiSplit(const Box<T,NAXES>& axes_minmax, const BOX_TYPE* boxes, SRC_INT_TYPE* indices, INT_TYPE nboxes, SRC_INT_TYPE* sub_indices[N+1], Box<T,NAXES> sub_boxes[N]) noexcept;

template<BVH_Heuristic H,typename T,uint NAXES,typename BOX_TYPE,typename SRC_INT_TYPE>
static void split(const Box<T,NAXES>& axes_minmax, const BOX_TYPE* boxes, SRC_INT_TYPE* indices, INT_TYPE nboxes, SRC_INT_TYPE*& split_indices, Box<T,NAXES>* split_boxes) noexcept;

template<INT_TYPE PARALLEL_THRESHOLD, typename SRC_INT_TYPE>
static void adjustParallelChildNodes(INT_TYPE nparallel, UT_Array<Node>& nodes, Node& node, UT_Array<Node>* parallel_nodes, SRC_INT_TYPE* sub_indices) noexcept;

template<typename T,typename BOX_TYPE,typename SRC_INT_TYPE>
static void nthElement(const BOX_TYPE* boxes, SRC_INT_TYPE* indices, const SRC_INT_TYPE* indices_end, const uint axis, SRC_INT_TYPE*const nth) noexcept;

template<typename T,typename BOX_TYPE,typename SRC_INT_TYPE>
static void partitionByCentre(const BOX_TYPE* boxes, SRC_INT_TYPE*const indices, const SRC_INT_TYPE*const indices_end, const uint axis, const T pivotx2, SRC_INT_TYPE*& ppivot_start, SRC_INT_TYPE*& ppivot_end) noexcept;

SYS_FORCE_INLINE static INT_TYPE nodeEstimate(const INT_TYPE nboxes) noexcept {
return nboxes/2 + nboxes/(2*(N-1));
}

template<BVH_Heuristic H,typename T, uint NAXES>
SYS_FORCE_INLINE static T unweightedHeuristic(const Box<T, NAXES>& box) noexcept {
if (H == BVH_Heuristic::BOX_PERIMETER) {
return box.axis_sum();
}
if (H == BVH_Heuristic::BOX_AREA) {
return box.half_surface_area();
}
if (H == BVH_Heuristic::BOX_VOLUME) {
return box.volume();
}
if (H == BVH_Heuristic::BOX_RADIUS) {
T diameter2 = box.diameter2();
return SYSsqrt(diameter2);
}
if (H == BVH_Heuristic::BOX_RADIUS2) {
return box.diameter2();
}
if (H == BVH_Heuristic::BOX_RADIUS3) {
T diameter2 = box.diameter2();
return diameter2*SYSsqrt(diameter2);
}
UT_ASSERT_MSG(0, "BVH_Heuristic::MEDIAN_MAX_AXIS should be handled separately by caller!");
return T(1);
}

static constexpr INT_TYPE NSPANS = 16;
static constexpr INT_TYPE NSPLITS = NSPANS-1;

static constexpr INT_TYPE MIN_FRACTION = 16;
};

} 

template<uint N>
using UT_BVH = UT::BVH<N>;

} 
#endif
