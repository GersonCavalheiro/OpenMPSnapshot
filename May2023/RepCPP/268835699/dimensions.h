



#ifndef _dims_h_
#define _dims_h_

#pragma once
#include <stdlib.h>
#include <stddef.h>

typedef double real;




typedef struct
{

int ncell_x;
int ncell_y;
int ncell_z;


int ncomponents;


int niterations;


int alg_id;
} Dimensions;

#endif
