


#pragma once


#include <vector>
#include "paraverkerneltypes.h"
#include "selectionmanagement.h"

class Trace;

class SelectionRowsUtils
{
public:
static void getAllLevelsSelectedRows( const Trace* whichTrace,
const SelectionManagement< TObjectOrder, TTraceLevel > &selectedRow,
TTraceLevel onLevel,
std::vector< TObjectOrder > &selected );
};


