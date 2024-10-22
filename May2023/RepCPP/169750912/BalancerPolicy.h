#pragma once
#include "Balancer.h"
#include "Fractal.h"

#include <string>

class BalancerPolicy {
public:
static Balancer* chooseBalancer(std::string balancerName, int predictionAccuracy, Fractal* fractal);
};