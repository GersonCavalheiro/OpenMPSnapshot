#pragma once
#ifndef BALANCINGCONTEXT_H	
#define BALANCINGCONTEXT_H

#include "Region.h"

struct BalancingContext {
Region* result;

int resultIndex;
int partsLeft;

double deltaReal;
double deltaImaginary;
};

#endif