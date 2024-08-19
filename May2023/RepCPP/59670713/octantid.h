#pragma once

#include "octreebuilder_api.h"

#include <vector>
#include <iosfwd>

#include "mortoncode.h"

namespace octreebuilder {
class LinearOctree;


class OCTREEBUILDER_API OctantID {
public:
OctantID();
OctantID(const Vector3i& coord, uint level);
OctantID(morton_t mcode, uint level);


morton_t mcode() const;


uint level() const;


Vector3i coord() const;


OctantID parent() const;


OctantID ancestorAtLevel(uint level) const;


::std::vector<OctantID> children() const;


bool isDecendantOf(const OctantID& possibleAncestor) const;


::std::vector<OctantID> potentialNeighbours(const LinearOctree& octree) const;


::std::vector<OctantID> potentialNeighboursWithoutSiblings(const LinearOctree& octree) const;


bool isBoundaryOctant(const LinearOctree& block, const LinearOctree& globalTree) const;


bool isBoundaryOctant(const Vector3i& blockLLF, const Vector3i& blockURB, const Vector3i& treeLLF, const Vector3i& treeURB) const;


::std::vector<OctantID> getSearchKeys(const LinearOctree& octree) const;

private:
morton_t m_mcode;
uint m_level;
};

bool OCTREEBUILDER_API operator<(const OctantID& left, const OctantID& right);
bool OCTREEBUILDER_API operator<=(const OctantID& left, const OctantID& right);
bool OCTREEBUILDER_API operator>(const OctantID& left, const OctantID& right);
bool OCTREEBUILDER_API operator>=(const OctantID& left, const OctantID& right);
bool OCTREEBUILDER_API operator==(const OctantID& a, const OctantID& b);
bool OCTREEBUILDER_API operator!=(const OctantID& a, const OctantID& b);
OCTREEBUILDER_API ::std::ostream& operator<<(::std::ostream& s, const OctantID& n);
}

namespace std {
template <>
struct hash<octreebuilder::OctantID> {
typedef octreebuilder::OctantID argument_type;
typedef octreebuilder::morton_t result_type;
result_type operator()(argument_type const& octant) const {
return octant.mcode();
}
};
}
