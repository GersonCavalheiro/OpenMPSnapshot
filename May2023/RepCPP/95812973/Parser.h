#pragma once

#include <fstream>

#include "Event.h"

class Parser {
public:
static Event parse(unsigned int depth, std::ifstream&, std::ifstream&);
};