

#pragma once

#include <cstdint> 
#include <vector>  

namespace rawspeed {

class TableLookUp {
public:
TableLookUp(int ntables, bool dither);

void setTable(int ntable, const std::vector<uint16_t>& table);
uint16_t* getTable(int n);
const int ntables;
std::vector<uint16_t> tables;
const bool dither;
};

} 
