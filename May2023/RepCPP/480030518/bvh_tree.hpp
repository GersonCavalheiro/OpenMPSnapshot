

#ifndef CPP_RAYTRACING_BVH_TREE_HPP
#define CPP_RAYTRACING_BVH_TREE_HPP

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <vector>

#include "axis_aligned_bounding_box.hpp"
#include "entities/base.hpp"

namespace cpp_raytracing {


template <Dimension DIMENSION>
class BVHTree {
private:

struct Node {
using Iter = typename std::vector<const Entity<DIMENSION>*>::iterator;


const Entity<DIMENSION>* value = nullptr;

std::unique_ptr<Node> left{};

std::unique_ptr<Node> right{};

AxisAlignedBoundingBox<DIMENSION> bounds{Vec<DIMENSION>{},
Vec<DIMENSION>{}};

Node() = default;
Node(const Iter first, const Iter last) {

const auto span = std::distance(first, last);

if (span == 0) {
return;
}
if (span == 1) {
value = *first;
bounds = value->bounding_box().value();
} else {
const int axis = rand() % 3;
const auto comp = pseudo_comparators.at(axis);
std::sort(first, last, comp);

const auto mid = std::next(first, span / 2);
if (first != mid) {
#pragma omp task shared(left)
{ left = std::make_unique<Node>(first, mid); }
}
if (last != mid) {
#pragma omp task shared(right)
{ right = std::make_unique<Node>(mid, last); }
}
#pragma omp taskwait

if (left && right) {
bounds = surrounding_box(left->bounds, right->bounds);
} else if (left && !right) {
bounds = left->bounds;
} else if (!left && right) {
bounds = right->bounds;
} else {
}
}
}

void hit_record(const Geometry<DIMENSION>& geometry,
const RaySegment<DIMENSION>& ray_segment,
const Scalar t_min,
HitRecord<DIMENSION>& closest_record) const {
if (bounds.hit(ray_segment, t_min, ray_segment.t_max())) {
if (value) {
HitRecord<DIMENSION> record =
value->hit_record(geometry, ray_segment, t_min);
if (record.t < closest_record.t) {
closest_record = record;
}
}
if (left) {
left->hit_record(geometry, ray_segment, t_min,
closest_record);
}
if (right) {
right->hit_record(geometry, ray_segment, t_min,
closest_record);
}
}
}


std::size_t size() const {
std::size_t count = value != nullptr ? 1 : 0;
if (left) {
count += left->size();
}
if (right) {
count += right->size();
}
return count;
}
};

public:

template <typename Container>
BVHTree(const Container& container) {
std::vector<const Entity<DIMENSION>*> bounded_entities{};

for (const auto& e : container) {
if (e->is_bounded()) {
bounded_entities.push_back(e.get());
} else {
_unbounded_entities.push_back(e.get());
}
}


#pragma omp parallel sections shared(bounded_entities)
{ _root = Node(bounded_entities.begin(), bounded_entities.end()); }
}


HitRecord<DIMENSION> hit_record(const Geometry<DIMENSION>& geometry,
const RaySegment<DIMENSION>& ray_segment,
const Scalar t_min = 0.0) const {
HitRecord<DIMENSION> closest_record = {.t = infinity};
_root.hit_record(geometry, ray_segment, t_min, closest_record);
for (auto& unbounded_entity : _unbounded_entities) {
HitRecord<DIMENSION> record =
unbounded_entity->hit_record(geometry, ray_segment, t_min);
if (record.t < closest_record.t) {
closest_record = record;
}
}
return closest_record;
}


std::optional<AxisAlignedBoundingBox<DIMENSION>> bounding_box() const {
if (_unbounded_entities.size() > 0) {
return std::nullopt;
}
return _root.bounds;
}


std::size_t size_bounded() const {
return _root.size();
}


std::size_t size_unbounded() const {
return _unbounded_entities.size();
}


std::size_t size() const {
return size_bounded() + size_unbounded();
}

private:

static bool pseudo_comparator(const std::size_t axis,
const Entity<DIMENSION>* e1,
const Entity<DIMENSION>* e2) {
auto const b1 = e1->bounding_box();
auto const b2 = e2->bounding_box();
if (b1 && b2) {
return b1->min()[axis] < b2->min()[axis];
}
return false; 
}


static bool pseudo_comparator_x(const Entity<DIMENSION>* e1,
const Entity<DIMENSION>* e2) {
return pseudo_comparator(0, e1, e2);
}

static bool pseudo_comparator_y(const Entity<DIMENSION>* e1,
const Entity<DIMENSION>* e2) {
return pseudo_comparator(1, e1, e2);
}

static bool pseudo_comparator_z(const Entity<DIMENSION>* e1,
const Entity<DIMENSION>* e2) {
return pseudo_comparator(2, e1, e2);
}

using pseudo_cmp_t = bool(const Entity<DIMENSION>* e1,
const Entity<DIMENSION>* e2);

static constexpr std::array<pseudo_cmp_t*, 3> pseudo_comparators = {
pseudo_comparator_x, pseudo_comparator_y, pseudo_comparator_z};

private:
std::vector<const Entity<DIMENSION>*> _unbounded_entities{};
Node _root;
};


using BVHTree3D = BVHTree<Dimension{3}>;

} 

#endif
