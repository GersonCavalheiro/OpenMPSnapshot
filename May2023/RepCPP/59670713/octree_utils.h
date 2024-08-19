#pragma once

#include "octreebuilder_api.h"
#include "octreebuilder.h"

#include "octantid.h"
#include "linearoctree.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>

namespace octreebuilder {

OCTREEBUILDER_API LinearOctree balanceTree(const LinearOctree& octree);


OCTREEBUILDER_API void createBalancedSubtree(LinearOctree& tree, uint maxLevel = ::std::numeric_limits<uint>::max());


OCTREEBUILDER_API LinearOctree
createBalancedSubtree(const OctantID& root, const ::std::vector<OctantID>& levelZeroLeafs, uint maxLevel = ::std::numeric_limits<uint>::max());


struct OCTREEBUILDER_API Partition {
Partition(const OctantID& rootOctant, const ::std::vector<LinearOctree>& partitionList);
OctantID root;
::std::vector<LinearOctree> partitions;
};


OCTREEBUILDER_API Partition computePartition(const OctantID& globalRoot, const ::std::vector<OctantID>& levelZeroLeafs, const int numThreads);


OCTREEBUILDER_API LinearOctree mergeUnbalancedCompleteTreeWithBalancedIncompleteTree(const LinearOctree& unbalancedTree, const LinearOctree& balancedTree);


OCTREEBUILDER_API ::std::vector<OctantID> completeRegion(const OctantID& start, const OctantID& end);


OCTREEBUILDER_API LinearOctree computeBlocksFromRegions(const OctantID& globalRoot, ::std::vector<::std::vector<OctantID>> completedRegions);


OCTREEBUILDER_API OctantID nearestCommonAncestor(const OctantID& a, const OctantID& b);


OCTREEBUILDER_API ::std::vector<OctantID> completeSubtree(const OctantID& root, uint lowestLevel, const ::std::unordered_set<OctantID>& keys);


OCTREEBUILDER_API LinearOctree createBalancedOctreeParallel(const OctantID& root, const ::std::vector<OctantID>& levelZeroLeafs, const int numThreads,
const uint maxLevel = ::std::numeric_limits<uint>::max());
}
