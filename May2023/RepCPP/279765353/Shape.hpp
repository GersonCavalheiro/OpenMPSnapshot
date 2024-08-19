
#pragma once
#ifndef SHAPE_HPP
#define SHAPE_HPP
#include <vector>
#include "Coord.hpp"

class GridType;
class Shape {
public:
std::vector<Coord> coords;
int morph;
int orient;

Shape(): morph(0), orient(0) {}

void normalize(int maxTile);

void sortCoords();

std::vector<Coord> getValidTranslations(int maxTile) const;

bool validateCoords(GridType *grid, Coord &wrong) const;

bool validateNoDup(Coord &wrong) const;
};

inline bool operator==(const Shape &a, const Shape &b) {
return a.coords == b.coords;
}

inline bool operator<(const Shape &a, const Shape &b) {
return a.coords < b.coords;
}

#endif
