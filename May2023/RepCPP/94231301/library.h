
#pragma once

#include "platform.h"

namespace embree
{

typedef struct opaque_lib_t* lib_t;


lib_t openLibrary(const std::string& file);


void* getSymbol(lib_t lib, const std::string& sym);


void closeLibrary(lib_t lib);
}
