#include "grid.hpp"

#include <iostream>
#include <random>

extern "C" {
#include <omp.h>
}

using namespace std;

Grid::Grid(const Size &size)
: size{size}, field(size.area()), buffer(size.area())
{
cout << "OMP max threads = " << omp_get_max_threads() << endl;
}

void Grid::randomize()
{
random_device rd;
default_random_engine gen{rd()};
uniform_int_distribution dist{0, 1};
for (auto &it : field)
it = dist(gen);
clear_border();
}

int Grid::idx_at(int x, int y) const
{
return y * size.w + x;
}

void Grid::clear_border()
{
for (int i = 0; i < size.w; ++i) {
field[idx_at(i, 0         )] = 0;
field[idx_at(i, size.h - 1)] = 0;
}
for (int i = 0; i < size.h; ++i) {
field[idx_at(0, i         )] = 0;
field[idx_at(size.w - 1, i)] = 0;
}
}

void Grid::do_tick()
{
fill(buffer.begin(), buffer.end(), 0);

#pragma omp parallel for
for (int y = 1; y < size.h - 1; ++y) {
for (int x = 1; x < size.w - 1; ++x) {
int pos = idx_at(x, y);
int alive = field[pos - size.w - 1]
+ field[pos - size.w    ]
+ field[pos - size.w + 1]
+ field[pos - 1         ]
+ field[pos + 1         ]
+ field[pos + size.w - 1]
+ field[pos + size.w    ]
+ field[pos + size.w + 1];
if (alive == 3 || (alive == 2 && field[pos] == 1)) {
buffer[pos] = 1;
}
}
}

swap(field, buffer);
}

void Grid::draw(Render &gfx) const
{
gfx.update_color({1, 0.33f, 0});
gfx.in_points_mode([this, &gfx]() {
int pos = 0;
for (int y = 0; y < size.h; ++y) {
for (int x = 0; x < size.w; ++x) {
if (field[pos] == 1) {
gfx.place({x, y});
}
++pos;
}
}
});
}
