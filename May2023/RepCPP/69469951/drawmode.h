


#pragma once


#include <vector>
#include "paraverkerneltypes.h"


enum class DrawModeMethod
{
DRAW_LAST = 0,
DRAW_MAXIMUM,
DRAW_MINNOTZERO,
DRAW_RANDOM,
DRAW_RANDNOTZERO,
DRAW_AVERAGE,
DRAW_AVERAGENOTZERO,
DRAW_MODE,
DRAW_ABSOLUTE_MAXIMUM,
DRAW_ABSOLUTE_MINNOTZERO,
DRAW_NUMMETHODS
};


class DrawMode
{
public:
static TSemanticValue selectValue( std::vector<TSemanticValue>& v,
DrawModeMethod method );
};


