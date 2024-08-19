#pragma once

#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "Point.hpp"

namespace KMeans {

template <typename T>
class Cluster {
static_assert(std::is_floating_point<T>::value, "T must be a floating point");

private:
std::vector<std::size_t> _pointsId;

std::vector<T> _coord;

public:
Cluster() {}

Cluster(const std::vector<T> &coord): _coord(coord) {}

~Cluster() {}

std::size_t GetSize() const { return _pointsId.size();}

std::size_t GetDim() const { return _coord.size();}

const std::vector<T> &GetCoord() const { return _coord;}

void SetCoord(const std::vector<T> &coord) { _coord = coord; }

void Add(const std::size_t &pointId) { _pointsId.push_back(pointId);}

const std::vector<std::size_t> &GetPointsId() const { return _pointsId;}

void Clear() { _pointsId.clear();}

void Update(const std::vector<Point<T>> &);
};

} 
