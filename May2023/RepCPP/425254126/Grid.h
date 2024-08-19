#pragma once

#include <ostream>

class Grid {
public:
int n1, n2, m;

Grid() = default;

Grid(int n1, int n2, int m)
: n1(n1 + 1), n2(n2 + 1), m(m + 1) {}

vec3<int> getData() const {
vec3<int> res;

res.assign(3, 0);

size_t i = 0;
for (const auto& elem : {n1, n2, m}) {
res[i++] = elem;
}

return res;
}

friend std::ostream &operator<<(std::ostream &os, const Grid &grid) {
return os << "(n1, n2, m): (" << grid.n1 << ", " << grid.n2 << ", " << grid.m << ")\n";
}

friend std::istream& operator>>(std::istream& in, Grid& grid) {
return in >> grid.n1 >> grid.n2 >> grid.m;
}
};