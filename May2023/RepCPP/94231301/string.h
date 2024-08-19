
#pragma once

#include "platform.h"
#include "../math/vec3.h"

namespace embree
{
class IOStreamStateRestorer 
{
public:
IOStreamStateRestorer(std::ostream& iostream)
: iostream(iostream), flags(iostream.flags()), precision(iostream.precision()) {
}

~IOStreamStateRestorer() {
iostream.flags(flags);
iostream.precision(precision);
}

private:
std::ostream& iostream;
std::ios::fmtflags flags;
std::streamsize precision;
};

std::string toLowerCase(const std::string& s);
std::string toUpperCase(const std::string& s);

Vec3f string_to_Vec3f ( std::string str );
}
