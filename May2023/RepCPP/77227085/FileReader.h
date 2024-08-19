

#pragma once

#include "adt/AlignedAllocator.h"
#include "adt/DefaultInitAllocatorAdaptor.h" 
#include "io/Buffer.h"                       
#include <cstdint>                           
#include <memory>                            
#include <utility>                           
#include <vector>                            

namespace rawspeed {

class Buffer;
template <class T, int alignment> class AlignedAllocator;

class FileReader {
const char* fileName;

public:
explicit FileReader(const char* fileName_) : fileName(fileName_) {}

std::pair<std::unique_ptr<std::vector<
uint8_t, DefaultInitAllocatorAdaptor<
uint8_t, AlignedAllocator<uint8_t, 16>>>>,
Buffer>
readFile();
};

} 
