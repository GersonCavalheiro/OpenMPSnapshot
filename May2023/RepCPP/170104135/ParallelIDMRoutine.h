#ifndef P_IDM_ROUTINE_H
#define P_IDM_ROUTINE_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "AccelerationComputer.h"
#include "LowLevelCar.h"
#include "LowLevelStreet.h"
#include "SimulationData.h"
#include "Timer.h"

template <template <typename Vehicle> typename RfbStructure>
class ParallelIDMRoutine {
private:
using car_iterator = typename LowLevelStreet<RfbStructure>::iterator;
using AccelerationComputerRfb = AccelerationComputer<RfbStructure>;

private:
class LaneChangeValues {
public:
bool valid;          
double acceleration; 
double indicator;    

LaneChangeValues() : valid(false), acceleration(0.0), indicator(0.0) {}
LaneChangeValues(double _acceleration, double _indicator)
: valid(true), acceleration(_acceleration), indicator(_indicator) {}
};

private:
const unsigned long PARALLEL_THRESHOLD = 50;
SimulationData<RfbStructure> &data;
std::vector<unsigned int> carWise;
std::vector<unsigned int> streetWise;

public:
ParallelIDMRoutine(SimulationData<RfbStructure> &_data) : data(_data) {}
void perform() {
#ifdef TIMER
IDMRoutine_thresholdSorting_timer.start();
#endif
for (auto &street : data.getStreets()) {
unsigned int carCount = street.getCarCount();
if (carCount > PARALLEL_THRESHOLD) {
carWise.push_back(street.getId());
} else if (carCount > 0) { 
streetWise.push_back(street.getId());
}
}

#ifdef TIMER
IDMRoutine_thresholdSorting_timer.stop();
IDMRoutine_performStreetWise_timer.start();
performStreetWise(streetWise);
IDMRoutine_performStreetWise_timer.stop();
IDMRoutine_performCarWise_timer.start();
performCarWise(carWise);
IDMRoutine_performCarWise_timer.stop();
#else
performStreetWise(streetWise);
performCarWise(carWise);
#endif

carWise.clear();
streetWise.clear();
}

private:
void performStreetWise(std::vector<unsigned int> &streetIds) {
#pragma omp parallel for shared(data) schedule(static)
for (std::size_t i = 0; i < streetIds.size(); i++) {
auto &street = data.getStreet(streetIds[i]);
AccelerationComputerRfb accelerationComputer(street);
for (car_iterator carIt = street.allIterable().begin(); accelerationComputer.isNotEnd(carIt); ++carIt) {
const double baseAcceleration = accelerationComputer(carIt, 0);
carIt->setNextBaseAcceleration(baseAcceleration);
}
for (car_iterator carIt = street.allIterable().begin(); accelerationComputer.isNotEnd(carIt); ++carIt) {
processLaneDecision(carIt, street);
}
}
}

void performCarWise(std::vector<unsigned int> &streetIds) {
for (auto streetId : streetIds) {
auto &street = data.getStreet(streetId);
AccelerationComputerRfb accelerationComputer(street);
auto streetIterable = street.allIterable();
#pragma omp parallel for shared(street) schedule(static)
for (unsigned i = 0; i < street.getCarCount(); ++i) {
auto carIt                    = streetIterable.begin() + i;
const double baseAcceleration = accelerationComputer(carIt, 0);
carIt->setNextBaseAcceleration(baseAcceleration);
}
#pragma omp parallel for shared(street) schedule(static)
for (unsigned i = 0; i < street.getCarCount(); ++i) {
auto carIt = streetIterable.begin() + i;
processLaneDecision(carIt, street);
}
}
}

void processLaneDecision(car_iterator &carIt, LowLevelStreet<RfbStructure> &street) {
LaneChangeValues leftLaneChange;
LaneChangeValues rightLaneChange;

if (carIt->getLane() > 0) 
leftLaneChange = computeLaneChangeValues(street, carIt, -1);
if (carIt->getLane() < street.getLaneCount() - 1) 
rightLaneChange = computeLaneChangeValues(street, carIt, +1);

double laneOffset       = 0;
double nextAcceleration = carIt->getNextBaseAcceleration();
if (leftLaneChange.valid) {
if (rightLaneChange.valid && rightLaneChange.indicator > leftLaneChange.indicator) {
laneOffset       = +1;
nextAcceleration = rightLaneChange.acceleration;
} else {
laneOffset       = -1;
nextAcceleration = leftLaneChange.acceleration;
}
} else if (rightLaneChange.valid) {
laneOffset       = +1;
nextAcceleration = rightLaneChange.acceleration;
}

computeAndSetDynamics(*carIt, nextAcceleration, carIt->getLane() + laneOffset);
}

LaneChangeValues computeLaneChangeValues(
AccelerationComputerRfb accelerationComputer, car_iterator carIt, const int laneOffset) {
LowLevelStreet<RfbStructure> &street = accelerationComputer.getStreet();

car_iterator laneChangeCarBehindIt = street.getNextCarBehind(carIt, laneOffset);
car_iterator laneChangeCarInFrontIt = street.getNextCarInFront(carIt, laneOffset);

if (!computeIsSpace(accelerationComputer, carIt, laneChangeCarBehindIt.getThisOrNotSpecialCarBehind(),
laneChangeCarInFrontIt.getThisOrNotSpecialCarInFront()))
return LaneChangeValues();

const double acceleration = accelerationComputer(carIt, laneChangeCarInFrontIt);

if (acceleration <= carIt->getNextBaseAcceleration()) return LaneChangeValues();

double carBehindAccelerationDeltas = 0.0;

car_iterator carInFrontIt = street.getNextCarInFront(carIt, 0);
car_iterator carBehindIt = street.getNextCarBehind(carIt, 0);

if (accelerationComputer.isNotEnd(carBehindIt)) {
const double carBehindAcceleration = accelerationComputer(carBehindIt, carInFrontIt);
carBehindAccelerationDeltas += carBehindAcceleration - carBehindIt->getNextBaseAcceleration();
}

if (accelerationComputer.isNotEnd(laneChangeCarBehindIt)) {
const double laneChangeCarBehindAcceleration = accelerationComputer(*laneChangeCarBehindIt, &*carIt);
carBehindAccelerationDeltas += laneChangeCarBehindAcceleration - laneChangeCarBehindIt->getNextBaseAcceleration();
}

const double indicator =
acceleration - carIt->getNextBaseAcceleration() + carIt->getPoliteness() * carBehindAccelerationDeltas;

if (indicator <= 1.0) return LaneChangeValues();

return LaneChangeValues(acceleration, indicator);
}

bool computeIsSpace(AccelerationComputerRfb accelerationComputer, car_iterator carIt, car_iterator carBehindIt,
car_iterator carInFrontIt) const {

if (accelerationComputer.isNotEnd(carBehindIt) &&
carIt->getDistance() - carIt->getLength() < carBehindIt->getDistance() + carIt->getMinDistance())
return false;

if (accelerationComputer.isNotEnd(carInFrontIt) &&
carInFrontIt->getDistance() - carInFrontIt->getLength() < carIt->getDistance() + carIt->getMinDistance())
return false;

return true;
}

void computeAndSetDynamics(LowLevelCar &car, const double nextAcceleration, const unsigned int nextLane) {
const double nextVelocity = std::max(car.getVelocity() + nextAcceleration, 0.0);
const double nextDistance = car.getDistance() + nextVelocity;
car.setNext(nextLane, nextDistance, nextVelocity);
car.updateTravelDistance(nextVelocity); 
}
};

#endif
