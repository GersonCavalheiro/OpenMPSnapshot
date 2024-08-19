#pragma once
#ifndef PREDICTIONBALANCER_H    
#define PREDICTIONBALANCER_H

#include "Balancer.h"
#include "Region.h"
#include "Fractal.h"
#include "Prediction.h"

#include <vector>
#include <string>

class PredictionBalancer : public Balancer {
private:

int predictionAccuracy;
Fractal *f;

Region *splitCol(Region col, int parts, Prediction* prediction);

public:
static const std::string NAME;

Region *balanceLoad(Region region, int nodeCount) override;

~PredictionBalancer() override;

static PredictionBalancer *create(Fractal *f, int predictionAccuracy);
};

#endif