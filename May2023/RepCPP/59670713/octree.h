#pragma once

#include "octreebuilder_api.h"
#include "octreenode.h"

#include "vector3i.h"

#include <vector>
#include <memory>
#include <iosfwd>

namespace octreebuilder {

class LinearOctree;


class OCTREEBUILDER_API Octree {
public:

virtual Vector3i getMaxXYZ() const = 0;


virtual uint getDepth() const = 0;


virtual uint getMaxLevel() const = 0;


virtual size_t getNumNodes() const = 0;


virtual OctreeNode getNode(const size_t& i) const = 0;


virtual OctreeNode tryGetNodeAt(const Vector3i& llf, uint level) const = 0;


virtual ::std::vector<OctreeNode> getNeighbourNodes(const OctreeNode& n, OctreeNode::Face sharedFace) const = 0;

enum class OctreeState { VALID, INCOMPLETE, OVERLAPPING, UNSORTED, UNBALANCED };


virtual OctreeState checkState() const = 0;

virtual ~Octree();
};

OCTREEBUILDER_API ::std::ostream& operator<<(::std::ostream& s, const OctreeNode& n);
OCTREEBUILDER_API ::std::ostream& operator<<(::std::ostream& s, const Octree& tree);
OCTREEBUILDER_API ::std::ostream& operator<<(::std::ostream& s, const ::std::unique_ptr<Octree>& tree);
OCTREEBUILDER_API ::std::ostream& operator<<(::std::ostream& s, const Octree::OctreeState& octreeState);
}
