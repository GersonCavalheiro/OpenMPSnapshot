
#include "SpaceSimulator.h"

SpaceSimulator::SpaceSimulator(const int numberOfIterations, const vector<MassPoint *> &massPoints,
GifBuilder &gifBuilder, const char *outputPath, const double maxAbsXY,
const double initialSpeed)
: numberOfIterations(numberOfIterations), massPoints(massPoints), gifBuilder(gifBuilder),
outputPath(outputPath), maxAbsXY(maxAbsXY), initialSpeed(initialSpeed) {}

void SpaceSimulator::execute() {

for (int i = 0; i < massPoints.size(); ++i) {
massPoints[i]->initInitialMove(maxAbsXY, initialSpeed);
}

for (int i = 0; i < numberOfIterations; ++i) {
if (DEBUG && i % (numberOfIterations/10) == 0) {
cout << "Iteration: " << i  << endl;
}
doIteration();
if (i % GIF_STEP == 0) {
gifBuilder.addFrame(massPoints);
}
}
gifBuilder.done();
}

void SpaceSimulator::doIteration() {
#if PARALLEL
#pragma omp parallel for
#endif
for (int i = 0; i < massPoints.size(); ++i) {
MassPoint *&mp = massPoints[i];
double forceX = 0;
double forceY = 0;

#if PARALLEL
#pragma omp simd reduction(+ : forceX, forceY)
#endif
for (int j = 0; j < massPoints.size(); ++j) {
if (i == j) continue;
MassPoint *&mpOther = massPoints[j];
double dist = mp->dist(mpOther);
double distX = mp->distX(mpOther);
double distY = mp->distY(mpOther);
double force = mp->force(mpOther);
forceX += force * distX / dist;
forceY += force * distY / dist;
}

mp->moveX += ((forceX / mp->weight) * TIME_CONSTANT * TIME_CONSTANT / 2);
mp->moveY += ((forceY / mp->weight) * TIME_CONSTANT * TIME_CONSTANT / 2);
}

#if PARALLEL
#pragma omp parallel for
#endif
for (int i = 0; i < massPoints.size(); ++i) {
massPoints[i]->doMove();
}

}


