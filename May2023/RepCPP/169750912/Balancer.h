#pragma once
#ifndef BALANCER_H	
#define BALANCER_H

#include "Region.h"

class Balancer {
public:
virtual Region* balanceLoad(Region region, int nodeCount) = 0;
virtual ~Balancer();
};

#endif