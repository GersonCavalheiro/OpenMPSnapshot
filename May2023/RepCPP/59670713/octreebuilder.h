#pragma once

#include "octreebuilder_api.h"

#include "vector3i.h"

#include "mortoncode.h"

#include <memory>
#include <limits>

namespace octreebuilder {

class Octree;


class OCTREEBUILDER_API OctreeBuilder {
public:

explicit OctreeBuilder(const Vector3i& maxXYZ, uint maxLevel = ::std::numeric_limits<uint>::max());


virtual morton_t addLevelZeroLeaf(const Vector3i& c) = 0;

virtual ::std::unique_ptr<Octree> finishBuilding() = 0;

virtual ~OctreeBuilder();

protected:
Vector3i m_maxXYZ;

uint maxLevel();

private:
uint m_maxLevel;
};
}
