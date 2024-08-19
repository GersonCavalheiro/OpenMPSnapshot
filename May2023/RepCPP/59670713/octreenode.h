#pragma once

#include "octreebuilder_api.h"
#include "mortoncode.h"

#include <iosfwd>

namespace octreebuilder {


class OCTREEBUILDER_API OctreeNode {
public:
OctreeNode();
OctreeNode(morton_t morton_encoded_llf, uint level);
OctreeNode(const Vector3i& coordinate, uint level);


enum Face { LEFT = 0, RIGHT = 1, FRONT = 2, BACK = 3, BOTTOM = 4, TOP = 5 };


static Vector3i getNormalOfFace(const Face& f);


bool isValid() const;


Vector3i getLLF() const;


morton_t getMortonEncodedLLF() const;


coord_t getSize() const;


uint getLevel() const;

bool operator==(const OctreeNode& o) const;

bool operator!=(const OctreeNode& o) const;

private:
morton_t m_morton_llf;
uint m_level;
};

OCTREEBUILDER_API ::std::ostream& operator<<(::std::ostream& s, const OctreeNode::Face& f);
}
