#pragma once

#include "octreebuilder_api.h"

#include <vector>
#include <unordered_map>
#include <iosfwd>

#include "octantid.h"

namespace octreebuilder {


class OCTREEBUILDER_API LinearOctree {
public:
typedef ::std::vector<OctantID> container_type;


LinearOctree();


LinearOctree(const OctantID& root, const container_type& leafs = {});


LinearOctree(const OctantID& root, const size_t& numLeafs);


const OctantID& root() const;


uint depth() const;


const container_type& leafs() const;


void insert(const OctantID& octant);
void insert(container_type::const_iterator begin, container_type::const_iterator end);


bool hasLeaf(const OctantID& octant) const;


::std::vector<OctantID> replaceWithChildren(const OctantID& octant);


void replaceWithSubtree(const OctantID& octant, const ::std::vector<OctantID>& subtree);


bool maximumLowerBound(const OctantID& octant, OctantID& lowerBound) const;


bool insideTreeBounds(const OctantID& octant) const;


OctantID deepestLastDecendant() const;


OctantID deepestFirstDecendant() const;


void sortAndRemove();


void reserve(const size_t numLeafs);

private:
OctantID m_root;
OctantID m_deepestLastDecendant;
container_type m_leafs;
::std::unordered_map<morton_t, uint> m_toRemove;
};

OCTREEBUILDER_API ::std::ostream& operator<<(::std::ostream& s, const LinearOctree& octree);
}
