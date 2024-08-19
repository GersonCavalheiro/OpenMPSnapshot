#pragma once

#include "glfw_w.hpp"

#include <vector>

class Grid {

const Size size;
std::vector<char> field;
std::vector<char> buffer;

int idx_at(int x, int y) const;
void clear_border();

public:

explicit Grid(const Size &size);

void randomize();
void do_tick();
void draw(Render &gfx) const;

};
