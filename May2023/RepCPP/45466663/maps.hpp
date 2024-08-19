#pragma once

#include "util/template_magic.hpp"

constexpr unsigned char NT_MAP[] = 
{
'-', 
'T', 
'G', 
'K', 
'C', 
'Y', 
'S', 
'B', 
'A', 
'W', 
'R', 
'D', 
'M', 
'H', 
'V', 
'N'  
};
constexpr unsigned char AA_MAP[] = 
{ 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 
'T', 'V', 'W', 'Y', '-', 'X', 'B', 'Z'};

constexpr size_t NT_MAP_SIZE = array_size(NT_MAP);
constexpr size_t AA_MAP_SIZE = array_size(AA_MAP);
