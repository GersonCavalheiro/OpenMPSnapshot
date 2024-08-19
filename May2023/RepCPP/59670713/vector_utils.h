#pragma once

#include "octreebuilder_api.h"

#include "vector3i.h"

#include <iterator>

namespace octreebuilder {

class Box;


class OCTREEBUILDER_API VectorSpace {
public:
class VectorRangeIter;
typedef Vector3i value_type;
typedef VectorRangeIter const_iterator;
typedef coord_t size_type;


VectorSpace(const Vector3i& start, const Vector3i& end);


explicit VectorSpace(const Vector3i& end);


explicit VectorSpace(const Box& box);

class OCTREEBUILDER_API VectorRangeIter : public ::std::iterator<::std::input_iterator_tag, Vector3i> {
public:
VectorRangeIter(const Vector3i& start, const Vector3i& end);
bool operator!=(const VectorRangeIter& other) const;
bool operator==(const VectorRangeIter& other) const;
const value_type& operator*() const;
const VectorRangeIter& operator++();

private:
Vector3i m_current;
const Vector3i m_start;
const Vector3i m_end;
};

VectorRangeIter begin() const;
VectorRangeIter end() const;

coord_t size() const;
bool empty() const;

private:
void init(const Vector3i& start, const Vector3i& end);
Vector3i m_start;
Vector3i m_end;
};


OCTREEBUILDER_API VectorSpace ClosedVectorSpace(const Vector3i& start, const Vector3i& end);


OCTREEBUILDER_API VectorSpace ClosedVectorSpace(const Box& box);


OCTREEBUILDER_API VectorSpace ClosedVectorSpace(const Vector3i& end);
}
