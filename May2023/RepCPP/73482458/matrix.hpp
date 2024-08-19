#pragma once
#include <vector>
#include <cassert>


template<typename T>
class Matrix
{

private:

std::vector<T> _data;

int indx(int i, int j) const {
assert(i >= 0 && i <  nx);
assert(j >= 0 && j <  ny);

return i * ny + j;
}

public:

int nx, ny;

Matrix() = default;
Matrix(int nx, int ny) : nx(nx), ny(ny) {
_data.resize(nx * ny);
};

void allocate(int nx_in, int ny_in) {
nx = nx_in;
ny = ny_in;
_data.resize(nx * ny);
};

T& operator()(int i, int j) {
return _data[ indx(i, j) ];
}

const T& operator()(int i, int j) const {
return _data[ indx(i, j) ];
}

T* data(int i=0, int j=0) {return _data.data() + i * ny + j;}

};
