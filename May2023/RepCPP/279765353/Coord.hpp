
#pragma once
#ifndef COORD_HPP
#define COORD_HPP

struct Coord {
int x;
int y;
int z;
int w;
};

inline bool operator==(Coord a, Coord b) {
return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline bool operator<(Coord a, Coord b) {
return a.x < b.x || (a.x == b.x && (
a.y < b.y || (a.y == b.y && (
a.z < b.z || (a.z == b.z && a.w < b.w)
))
));
}

inline Coord operator+(Coord a, Coord b) {
return Coord{a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w};
}

inline Coord operator-(Coord a, Coord b) {
return Coord{a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w};
}

#endif
