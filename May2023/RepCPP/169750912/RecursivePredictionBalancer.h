#pragma once

#include "Balancer.h"
#include "Region.h"
#include "Fractal.h"
#include "Prediction.h"
#include "BalancingContext.h"

#include <vector>
#include <string>

class RecursivePredictionBalancer : public Balancer {
private:

int predictionAccuracy;
Fractal *f;

int balancingHelper(Region region, Prediction* prediction, BalancingContext context);

Region *halveRegionVertically(Region region, Prediction prediction, Prediction* left, Prediction* right, int nodeCount);

Region *halveRegionHorizontally(Region region, Prediction prediction, Prediction* top, Prediction* bot, int nodeCount);

bool tooFewLeft(int splitPos, bool vertical, int width, int height, int guaranteedDivisor, int nodeCount);

bool enoughAreaForWorkers(int splitPos, bool vertical, int width, int height, int guaranteedDivisor, int nodeCount);

public:
static const std::string NAME;

Region *balanceLoad(Region region, int nodeCount) override;

~RecursivePredictionBalancer() override;

static RecursivePredictionBalancer *create(Fractal *f, int predictionAccuracy);
};