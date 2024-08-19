#pragma once
#ifndef COLUMNBALANCER_H	
#define COLUMNBALANCER_H

#include "Balancer.h"
#include "Region.h"

#include <string>

class ColumnBalancer : public Balancer {
public:
static const std::string NAME;
Region* balanceLoad(Region region, int nodeCount) override;
};

#endif