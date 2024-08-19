#include "ped_model.h"

#include <nmmintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stack>
#include <thread>

#include "cuda_testkernel.h"
#include "ped_agent_cuda.h"
#include "ped_agent_soa.h"
#include "ped_model.h"
#include "ped_waypoint.h"

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario,
std::vector<Twaypoint*> destinationsInScenario,
IMPLEMENTATION implementation) {
cuda_test();

agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(),
agentsInScenario.end());

destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(),
destinationsInScenario.end());

this->implementation = implementation;

if(implementation == Ped::IMPLEMENTATION::SEQ) 
setupHeatmapSeq();
else { 
setupHeatmapCUDA();
}

}

void Ped::Model::tick() {
auto a1Move = [](Ped::Tagent* agent) {
agent->computeNextDesiredPosition();

int dX = agent->getDesiredX(), dY = agent->getDesiredY();
agent->setX(dX);
agent->setY(dY);
};

auto pFunc = [&](int tId, int start, int end,
std::vector<Ped::Tagent*>& agents) {
for (int i = start; i <= end; i++) {
Ped::Tagent* agent = agents[i];
agent->computeNextDesiredPosition();

int dX = agent->getDesiredX(), dY = agent->getDesiredY();
agent->setX(dX);
agent->setY(dY);
}
};

int agentSize = agents.size();
int threadNum = getThreadNum();
switch (implementation) {
case SEQ: {
for (int i = 0; i < agentSize; i++) {
agents[i]->computeNextDesiredPosition();
move(agents[i]);
}
updateHeatmapSeq();
} break;

case PTHREAD: {
int agentsPerThread = agentSize / threadNum;
int agentLeft = agentSize % threadNum;

std::thread* threads = new std::thread[threadNum];
int start, end;
for (int i = 0; i < threadNum; i++) {
if (i < agentLeft) {
start = i * agentsPerThread + i;
end = start + agentsPerThread;
} else {
start = i * agentsPerThread + agentLeft;
end = start + agentsPerThread - 1;
}

threads[i] =
std::thread(pFunc, i, start, end, std::ref(agents));
}

for (int i = 0; i < threadNum; i++) {
threads[i].join();
}

delete[] threads;

} break;

case OMP: {
int i;
#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
for (i = 0; i < agentSize; i++) {
a1Move(agents[i]);
}

} break;

case VECTOR: {
if (!agentSOA) {
for (int i = 0; i < agents.size(); i++) {
agents[i]->computeNextDesiredPosition();

int dX = agents[i]->getDesiredX(),
dY = agents[i]->getDesiredY();
agents[i]->setX(dX);
agents[i]->setY(dY);
}
agentSOA = new Ped::TagentSOA(agents);
}
agentSOA->setThreads(threadNum);
agentSOA->computeAndMove();
float *xs = agentSOA->xs, *ys = agentSOA->ys;

#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
for (size_t i = 0; i < agents.size(); i++) {
agents[i]->setX(xs[i]);
agents[i]->setY(ys[i]);
}

} break;

case CUDA: {
if (!agentCUDA) {
for (int i = 0; i < agents.size(); i++) {
agents[i]->computeNextDesiredPosition();

int dX = agents[i]->getDesiredX(),
dY = agents[i]->getDesiredY();
agents[i]->setX(dX);
agents[i]->setY(dY);
}
agentCUDA = new Ped::TagentCUDA(agents);
}

agentCUDA->computeAndMove();
h_desiredXs = new float[agents.size()];
h_desiredYs = new float[agents.size()];

for(int i = 0; i < agents.size(); i++) {
h_desiredXs[i] = (*agentCUDA).xs[i];
h_desiredYs[i] = (*agentCUDA).ys[i];
}

updateHeatmapCUDA();

delete[] h_desiredXs;
delete[] h_desiredYs;

float *xs = agentCUDA->xs, *ys = agentCUDA->ys;

#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
for (size_t i = 0; i < agents.size(); i++) {
agents[i]->setX(xs[i]);
agents[i]->setY(ys[i]);
}

} break;

case REGION: {
if (!agentSOA) {
for (int i = 0; i < agents.size(); i++) {
agents[i]->computeNextDesiredPosition();

int dX = agents[i]->getDesiredX(),
dY = agents[i]->getDesiredY();
agents[i]->setX(dX);
agents[i]->setY(dY);
}
agentSOA = new Ped::TagentSOA(agents);
agentSOA->setThreads(threadNum);
sortAgents();

Ped::Tagent* maxAgent = agents[agentsIdx[agents.size() - 1]];

int boardX = ceil((double)maxAgent->getX() / 100 + 2) * 100;
int boardY = ceil((double)maxAgent->getY() / 100 + 2) * 100;


stateBoard = std::vector<std::vector<int>>(
boardX, std::vector<int>(boardY, -1));

for (int i = 0; i < agents.size(); i++) {
int sx = agents[i]->getX(), sy = agents[i]->getY();
stateUnit(sx, sy) = i;
}
}

sortAgentsY();
agentSOA->computeNextDesiredPosition();

h_desiredXs = new float[agents.size()];
h_desiredYs = new float[agents.size()];

for(int i = 0; i < agents.size(); i++) {
h_desiredXs[i] = (*agentSOA).desiredXs[i];
h_desiredYs[i] = (*agentSOA).desiredYs[i];
}

updateHeatmapCUDA();

delete[] h_desiredXs;
delete[] h_desiredYs;


omp_set_num_threads(threadNum);
#pragma omp parallel
{
int agentsInRegion = ceil((double)agents.size() / threadNum);
int threadId = omp_get_thread_num();
int rStart = threadId * agentsInRegion;
int rEnd = rStart + agentsInRegion < agents.size()
? rStart + agentsInRegion
: agents.size();
move(rStart, rEnd);
}

float *xs = agentSOA->xs, *ys = agentSOA->ys;
#pragma omp parallel for shared(agents) num_threads(threadNum) schedule(static)
for (size_t i = 0; i < agents.size(); i++) {
agents[i]->setX(xs[i]);
agents[i]->setY(ys[i]);
}

} break;

default:
break;
}
}


void Ped::Model::sortAgents() {
agentsIdx = vector<int>(agents.size());
std::iota(agentsIdx.begin(), agentsIdx.end(), 0);

sort(agentsIdx.begin(), agentsIdx.end(),
[=](const int& i, const int& j) -> bool {
if (agents[i]->getX() != agents[j]->getX()) 
return agents[i]->getX() < agents[j]->getX(); 
return agents[i]->getY() < agents[i]->getY();
});
}

void Ped::Model::sortAgentsY() {
agentsIdx = vector<int>(agents.size());
std::iota(agentsIdx.begin(), agentsIdx.end(), 0);

sort(agentsIdx.begin(), agentsIdx.end(),
[=](const int& i, const int& j) -> bool {
return agents[i]->getY() < agents[j]->getY();
});
}


void Ped::Model::move(int& rStart, int& rEnd) {
float rangeYStart = agentSOA->ys[agentsIdx[rStart]];
if (rEnd == agentsIdx.size())
rEnd = rEnd - 1;
float rangeYEnd = agentSOA->ys[agentsIdx[rEnd]];


std::srand(std::time(0));
random_shuffle(agentsIdx.begin() + rStart, agentsIdx.begin() + rEnd);

for (int i = rStart; i < rEnd; i++) {
int aId = agentsIdx[i];
int x = agentSOA->xs[aId];
int y = agentSOA->ys[aId];
int desiredX = agentSOA->desiredXs[aId];
int desiredY = agentSOA->desiredYs[aId];

std::pair<int, int> p0, p1, p2;
p0 = std::make_pair(desiredX, desiredY);

auto diffX = desiredX - x;
auto diffY = desiredY - y;

if (diffX == 0 || diffY == 0) {
p1 = std::make_pair(desiredX + diffY, desiredY + diffX);
p2 = std::make_pair(desiredX - diffY, desiredY - diffX);
} else {
p1 = std::make_pair(desiredX, y);
p2 = std::make_pair(x, desiredY);
}
auto pCandidates = std::vector<std::pair<int, int>>({p0, p1, p2});

for (auto position : pCandidates) {
int px, py;
std::tie(px, py) = position;

bool isInRegion = py > rangeYStart && py < rangeYEnd;
if (isInRegion) {
if (stateUnit(px, py) == -1) {
stateUnit(px, py) = aId;
stateUnit(x, y) = -1;
agentSOA->xs[aId] = px;
agentSOA->ys[aId] = py;
break;
}
} else {
if (__sync_bool_compare_and_swap(&stateUnit(px, py), -1, aId)) {
stateUnit(x, y) = -1;
agentSOA->xs[aId] = px;
agentSOA->ys[aId] = py;
break;
}
}
}
}
}

void Ped::Model::move(Ped::Tagent* agent) {
set<const Ped::Tagent*> neighbors =
getNeighbors(agent->getX(), agent->getY(), 2);

std::vector<std::pair<int, int>> takenPositions;
for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin();
neighborIt != neighbors.end(); ++neighborIt) {
std::pair<int, int> position((*neighborIt)->getX(),
(*neighborIt)->getY());
takenPositions.push_back(position);
}

std::vector<std::pair<int, int>> prioritizedAlternatives;
std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
prioritizedAlternatives.push_back(pDesired);

int diffX = pDesired.first - agent->getX();
int diffY = pDesired.second - agent->getY();
std::pair<int, int> p1, p2;
if (diffX == 0 || diffY == 0) {
p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
} else {
p1 = std::make_pair(pDesired.first, agent->getY());
p2 = std::make_pair(agent->getX(), pDesired.second);
}
prioritizedAlternatives.push_back(p1);
prioritizedAlternatives.push_back(p2);

for (std::vector<pair<int, int>>::iterator it =
prioritizedAlternatives.begin();
it != prioritizedAlternatives.end(); ++it) {
if (std::find(takenPositions.begin(), takenPositions.end(), *it) ==
takenPositions.end()) {
agent->setX((*it).first);
agent->setY((*it).second);

break;
}
}
}

set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {
return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
}

Ped::Model::~Model() {
std::for_each(agents.begin(), agents.end(),
[](Ped::Tagent* agent) { delete agent; });
std::for_each(destinations.begin(), destinations.end(),
[](Ped::Twaypoint* destination) { delete destination; });
if (agentSOA != nullptr)
delete agentSOA;
if (agentCUDA != nullptr)
delete agentCUDA;

freeCUDAMem();
}
