#pragma once



#include "octreebuilder_api.h"

#include "mortoncode.h"

#include <array>
#include <vector>

namespace octreebuilder {


OCTREEBUILDER_API bool fitsInMortonCode(const Vector3i& maxXYZ);


OCTREEBUILDER_API uint getOctreeDepthForBounding(const Vector3i& maxXYZ);


OCTREEBUILDER_API Vector3i getMaxXYZForOctreeDepth(const uint& depth);


OCTREEBUILDER_API coord_t getOctantSizeForLevel(const uint& level);


OCTREEBUILDER_API morton_t getMortonCodeForCoordinate(const Vector3i& coordinate);


OCTREEBUILDER_API Vector3i getCoordinateForMortonCode(const morton_t& code);


OCTREEBUILDER_API morton_t getMortonCodeForParent(const morton_t& current_code, const uint& currentLevel);


OCTREEBUILDER_API morton_t getMortonCodeForAncestor(const morton_t& current_code, const uint& currentLevel, const uint& ancestorLevel);


OCTREEBUILDER_API ::std::array<morton_t, 8> getMortonCodesForChildren(const morton_t& parent, const uint& parentLevel);


OCTREEBUILDER_API ::std::vector<morton_t> getMortonCodesForNeighbourOctants(const morton_t& current_octant, const uint& currentLevel, const uint& octreeDepth,
const Vector3i& root = Vector3i(0));


OCTREEBUILDER_API uint getMaxLevelOfLLF(const Vector3i& llf, const uint& octreeDepth);


OCTREEBUILDER_API Vector3i getSearchCorner(const morton_t& octant, const uint& level);


OCTREEBUILDER_API bool isMortonCodeDecendant(const morton_t& octant, const uint& levelOfOctant, const morton_t& potentialAncestor, const uint& levelOfAncestor);


OCTREEBUILDER_API ::std::pair<morton_t, uint> nearestCommonAncestor(const morton_t& a, const morton_t& b, const uint& aLevel, const uint& bLevel);
}
