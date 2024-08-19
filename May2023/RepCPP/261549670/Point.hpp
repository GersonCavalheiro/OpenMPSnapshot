#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>


namespace KMeans {

template <typename T>
class Point {
static_assert(std::is_floating_point<T>::value, "T must be a floating point");

private:
std::vector<T> _coord;
std::size_t _clusterId;

bool _assigned;

public:
Point(): _assigned(false) {}

Point(const std::vector<T> &coord): _coord(coord), _assigned(false) {}

~Point() {}

std::size_t GetDim() const { return _coord.size();}

const std::vector<T> &GetCoord() const { return _coord;}

void SetCoord(const std::vector<T> &coord) { _coord = coord;}

const std::size_t &GetClusterId() const { return _clusterId;}

void SetClusterId(const std::size_t &id)
{
_clusterId = id;
_assigned = true;
}

bool IsAssigned() const { return _assigned;}
};

} 
