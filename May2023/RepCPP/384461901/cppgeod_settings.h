#pragma once

#include <iostream>

#ifndef APPTAG
#define APPTAG "[cpp_geod] "
#endif

#ifndef CPP_GEOD_DEBUG_LEVEL
#define CPP_GEOD_DEBUG_LEVEL 3
#endif

#define CPP_GEOD_DEBUG_LVL_CRITICAL 0
#define CPP_GEOD_DEBUG_LVL_ERROR 1
#define CPP_GEOD_DEBUG_LVL_WARN 2
#define CPP_GEOD_DEBUG_LVL_IMPORTANT 3
#define CPP_GEOD_DEBUG_LVL_LOG 4
#define CPP_GEOD_DEBUG_LVL_INFO 5
#define CPP_GEOD_DEBUG_LVL_VERBOSE 6
#define CPP_GEOD_DEBUG_LVL_EXCESSIVE 7

void debug_print(const int level, const std::string& message) {
if (level <= CPP_GEOD_DEBUG_LEVEL) {
std::cout << APPTAG << "[" << CPP_GEOD_DEBUG_LEVEL << "] " << message << std::endl;
}
}
