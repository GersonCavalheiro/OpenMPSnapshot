#pragma once

#pragma once

#include "octreebuilder_api.h"
#include "octree.h"
#include "box.h"
#include "linearoctree.h"

#include <vector>
#include <unordered_set>

namespace octreebuilder {

class OCTREEBUILDER_API OctreeImpl : public Octree {
public:
OctreeImpl(::std::vector<::std::unordered_set<morton_t>> tree);


OctreeImpl(LinearOctree&& linearOctree);

virtual Vector3i getMaxXYZ() const override;

virtual uint getDepth() const override;

virtual uint getMaxLevel() const override;

virtual size_t getNumNodes() const override;

virtual OctreeNode getNode(const size_t& i) const override;

virtual OctreeNode tryGetNodeAt(const Vector3i& llf, uint level) const override;

virtual ::std::vector<OctreeNode> getNeighbourNodes(const OctreeNode& n, OctreeNode::Face sharedFace) const override;

virtual OctreeState checkState() const override;

private:
::std::vector<::std::unordered_set<morton_t>> m_tree;
LinearOctree m_linearTree;
Box m_bounding;
};
}
