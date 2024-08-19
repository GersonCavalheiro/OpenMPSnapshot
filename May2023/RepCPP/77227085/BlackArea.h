

#pragma once

#include <cstdint> 

namespace rawspeed {

class BlackArea final {
public:
BlackArea(int offset_, int size_, bool isVertical_)
: offset(offset_), size(size_), isVertical(isVertical_) {}
uint32_t offset; 
uint32_t size;   
bool isVertical; 
};

} 
