#ifndef P_CONSISTENCY_LIGHT_ROUTINE_H
#define P_CONSISTENCY_LIGHT_ROUTINE_H

#include "DomainModel.h"
#include "LowLevelCar.h"
#include "LowLevelStreet.h"
#include "RfbStructure.h"
#include "SimulationData.h"
#include "Timer.h"
#include <algorithm>
#ifdef OMP
#include <omp.h>
#endif

template <template <typename Vehicle> typename RfbStructure>
class ParallelConsistencyRoutine {
public:

ParallelConsistencyRoutine(SimulationData<RfbStructure> &data) : data(data) {}

void perform() {
#ifdef TIMER
consistencyRoutine_restoreConsistency_timer.start();
restoreConsistency(); 
consistencyRoutine_restoreConsistency_timer.stop();
consistencyRoutine_relocateCars_timer.start();
relocateCars(); 
consistencyRoutine_relocateCars_timer.stop();
consistencyRoutine_incorporateCars_timer.start();
incorporateCars(); 
consistencyRoutine_incorporateCars_timer.stop();
#else
restoreConsistency(); 
relocateCars();       
incorporateCars();    
#endif
}


void restoreConsistency() {
#pragma omp parallel for shared(data) schedule(static)
for (std::size_t i = 0; i < data.getStreets().size(); i++) {
auto &street = data.getStreets()[i];
street.updateCarsAndRestoreConsistency();
}
}


void relocateCars() {
DomainModel &model = data.getDomainModel();
for (auto &street : data.getStreets()) {
Street &domStreet                 = model.getStreet(street.getId());
Junction &domJunction             = domStreet.getTargetJunction();
CardinalDirection originDirection = calculateOriginDirection(domJunction, domStreet);
auto beyondsIterable = street.beyondsIterable();
for (auto vehicleIt = beyondsIterable.begin(); vehicleIt != beyondsIterable.end(); ++vehicleIt) {
LowLevelCar &vehicle = *vehicleIt;
relocateCar(vehicle, street, domJunction, originDirection);
}
street.removeBeyonds();
}
}


void incorporateCars() {
#pragma omp parallel for shared(data) schedule(static)
for (std::size_t i = 0; i < data.getStreets().size(); i++) {
auto &street = data.getStreets()[i];
street.incorporateInsertedCars();
}
}


void relocateCar(LowLevelCar &vehicle, LowLevelStreet<RfbStructure> &street, Junction &domJunction,
CardinalDirection originDirection) {
DomainModel &model = data.getDomainModel();
Vehicle &domVehicle                    = model.getVehicle(vehicle.getId());
CardinalDirection destinationDirection = takeTurn(originDirection, domVehicle.getNextDirection());
while (!domJunction.getOutgoingStreet(destinationDirection).isConnected()) {
destinationDirection = CardinalDirection((destinationDirection + 1) % 4);
}
Street *domDestinationStreet = domJunction.getOutgoingStreet(destinationDirection).getStreet();
int newLane = std::min(vehicle.getLane(), domDestinationStreet->getLanes() - 1);
vehicle.setNext(newLane, vehicle.getDistance() - street.getLength(), vehicle.getVelocity());
LowLevelStreet<RfbStructure> &destinationStreet = data.getStreet(domDestinationStreet->getId());
destinationStreet.insertCar(vehicle);
}


CardinalDirection takeTurn(CardinalDirection origin, TurnDirection turn) {
return (CardinalDirection)((origin + turn) % 4);
}


CardinalDirection calculateOriginDirection(Junction &junction, Street &incomingStreet) {
for (const auto &connectedStreet : junction.getIncomingStreets()) {
if (connectedStreet.isConnected() && connectedStreet.getStreet()->getId() == incomingStreet.getId()) {
return connectedStreet.getDirection();
}
}
throw std::invalid_argument("Street is not connected to junction!");
}

SimulationData<RfbStructure> &data;
};

#endif
