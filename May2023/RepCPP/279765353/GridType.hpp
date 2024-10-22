
#pragma once
#ifndef GRID_TYPE_HPP
#define GRID_TYPE_HPP
#include <string>
#include <vector>
#include "Coord.hpp"

class GridType {
private:
std::string m_name;
int m_dimension;
int m_rotateCount;
int m_canReflect;
std::vector<int> m_orbit;

public:
static GridType *fromName(std::string name);

virtual Coord rotate(Coord coord, int i) const = 0;

inline std::string name() const { return m_name; }
inline int dimension() const { return m_dimension; }
inline int rotateCount() const { return m_rotateCount; }
inline int canReflect() const { return m_canReflect; }
inline const std::vector<int>& orbit() const { return m_orbit; }

bool validateCoord(Coord coord) const ;

protected:
GridType(std::string name, int dimension, int rotateCount, bool canReflect,
std::vector<int> orbit):
m_name(name),
m_dimension(dimension),
m_rotateCount(rotateCount),
m_canReflect(canReflect),
m_orbit(orbit) {}
};

class GridSquare : public GridType {
public:
GridSquare(): GridType("square", 2, 4, true, {0}) {}

Coord rotate(Coord coord, int orient) const override;
};

class GridCube : public GridType {
public:
GridCube(): GridType("cube", 3, 24, true, {0}) {}

Coord rotate(Coord coord, int orient) const override;
};

class GridTriangle : public GridType {
public:
GridTriangle(): GridType("triangle", 2, 6, true, {0, 0}) {}

Coord rotate(Coord coord, int orient) const override;
};

class GridHexagon : public GridType {
public:
GridHexagon(): GridType("hexagon", 2, 6, true, {0}) {}

Coord rotate(Coord coord, int orient) const override;
};

class GridTan : public GridType {
public:
GridTan(): GridType("tan", 2, 4, true, {0, 0, 0, 0}) {}

Coord rotate(Coord coord, int orient) const override;
};

class GridOctagon : public GridType {
public:
GridOctagon(): GridType("octagon", 2, 4, true, {0, 1}) {}

Coord rotate(Coord coord, int orient) const override;
};

#endif
