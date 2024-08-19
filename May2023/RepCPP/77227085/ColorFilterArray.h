

#pragma once

#include "rawspeedconfig.h" 
#include "adt/Point.h"      
#include <algorithm>        
#include <cstdint>          
#include <map>              
#include <string>           
#include <vector>           

namespace rawspeed {

enum class CFAColor : uint8_t {
RED = 0,
GREEN = 1,
BLUE = 2,
CYAN = 3,
MAGENTA = 4,
YELLOW = 5,
WHITE = 6,
FUJI_GREEN = 7,
END, 
UNKNOWN = 255,

};

class ColorFilterArray {
std::vector<CFAColor> cfa;
iPoint2D size;

public:
ColorFilterArray() = default;
explicit ColorFilterArray(const iPoint2D& size);

void setSize(const iPoint2D& size);
void setColorAt(iPoint2D pos, CFAColor c);
void setCFA(iPoint2D size, ...);

void shiftRight(int n = 1);
void shiftDown(int n = 1);

[[nodiscard]] CFAColor getColorAt(int x, int y) const;
[[nodiscard]] uint32_t getDcrawFilter() const;
[[nodiscard]] std::string asString() const;
[[nodiscard]] iPoint2D getSize() const { return size; }

static std::string colorToString(CFAColor c);
static uint32_t RAWSPEED_READNONE shiftDcrawFilter(uint32_t filter, int x,
int y);
};


} 
