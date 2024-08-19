#ifndef P_TRAFFIC_LIGHT_ROUTINE_H
#define P_TRAFFIC_LIGHT_ROUTINE_H

#include "DomainModel.h"
#include "LowLevelCar.h"
#include "LowLevelStreet.h"
#include "RfbStructure.h"
#include "SimulationData.h"
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

template <template <typename Vehicle> typename RfbStructure>
class ParallelTrafficLightRoutine {
public:

const unsigned long PARALLEL_THRESHOLD = 500;


ParallelTrafficLightRoutine(SimulationData<RfbStructure> &data) : data(data) {}


void perform() {
DomainModel &model    = data.getDomainModel();
const auto &junctions = model.getJunctions();
if (junctions.size() > PARALLEL_THRESHOLD) {
performParallel(junctions);
} else {
performSequential(junctions);
}
}

void performParallel(const std::vector<std::unique_ptr<Junction>> &junctions) {
#pragma omp parallel for shared(junctions) schedule(static)
for (std::size_t i = 0; i < junctions.size(); i++) {
const auto &junction = junctions[i];
perform(*junction);
}
}

void performSequential(const std::vector<std::unique_ptr<Junction>> &junctions) {
for (auto const &junction : junctions) { perform(*junction); }
}

void perform(Junction &junction) {
bool lightChanged = junction.nextStep();
if (lightChanged) {
Junction::Signal previous = junction.getPreviousSignal();
toggleStreetForSignal(previous, junction);
Junction::Signal current = junction.getCurrentSignal();
toggleStreetForSignal(current, junction);
}
}

private:

void toggleStreetForSignal(Junction::Signal signal, const Junction &junction) {
auto connectedStreet = junction.getIncomingStreet(signal.getDirection());
if (connectedStreet.isConnected()) {
Street *domainModelStreet            = connectedStreet.getStreet();
LowLevelStreet<RfbStructure> &street = data.getStreet(domainModelStreet->getId());
street.switchSignal();
}
}

SimulationData<RfbStructure> &data;
};

#endif
