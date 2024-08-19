#include <mpi.h>
#include <omp.h>

#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <vector>

#include "ConfigWeight.h"
#include "ConfigWeightTask.h"
#include "Graph.h"
#include "TestData.cpp"

using namespace std;
using namespace std::chrono;

int numberOfProcesses;
int processId;
long minimalSplitWeight = LONG_MAX;      
short* minimalSplitConfig = nullptr;     
int maxPregeneratedLevelFromMaster = 6;  
int maxPregeneratedLevelFromSlave = 9;   
int smallerSetSize;                      
int configLength;
vector<short*> taskPool = {};
Graph graph;


void printConfig(short* config, ostream& os = cout) {
os << "[";
for (int i = 0; i < configLength; i++) {
os << config[i];
if (i == configLength - 1) {
os << "]" << endl;
} else {
os << ", ";
}
}
}


[[nodiscard]] inline bool isMaster() { return processId == MASTER; }


[[nodiscard]] pair<int, int> computeSizeOfXAndY(short* config) {
int countX = 0;
int countY = 0;

for (int i = 0; i < configLength; i++) {
if (config[i] == IN_X) {
countX++;
} else if (config[i] == IN_Y) {
countY++;
} else {
return make_pair(countX, countY);  
}
}

return make_pair(countX, countY);
}


[[nodiscard]] long computeSplitWeight(short* config) {
long weight = 0;

for (int i = 0; i < graph.edgesSize; i++) {
if (config[graph.edges[i].vertexId1] == IN_X && config[graph.edges[i].vertexId2] == IN_Y) {
weight += graph.edges[i].weight;
}
}

return weight;
}


[[nodiscard]] long lowerBoundOfUndecidedPart(short* config, int indexOfFirstUndecided,
long weightOfDecidedPart) {
long lowerBound = 0;

for (int i = indexOfFirstUndecided; i < configLength; i++) {
config[i] = IN_X;
long weightWhenInX = computeSplitWeight(config);
config[i] = IN_Y;
long weightWhenInY = computeSplitWeight(config);
lowerBound += (min(weightWhenInX, weightWhenInY) - weightOfDecidedPart);
config[i] = NOT_DECIDED;
}

return lowerBound;
}


void searchAux(short* config, int indexOfFirstUndecided, int& targetSizeOfSetX) {
pair<int, int> sizeOfXAndY = computeSizeOfXAndY(config);
if (sizeOfXAndY.first > targetSizeOfSetX ||
sizeOfXAndY.second > configLength - targetSizeOfSetX) {
return;
}

long weightOfDecidedPart = computeSplitWeight(config);

if (weightOfDecidedPart > minimalSplitWeight) {
return;
}

if (weightOfDecidedPart +
lowerBoundOfUndecidedPart(config, indexOfFirstUndecided, weightOfDecidedPart) >
minimalSplitWeight) {
return;
}

if (indexOfFirstUndecided == configLength) {
if (computeSizeOfXAndY(config).first != targetSizeOfSetX) {
return;
}

long weight = computeSplitWeight(config);
if (weight < minimalSplitWeight) {
#pragma omp critical
{
if (weight < minimalSplitWeight) {
minimalSplitWeight = weight;
copy(config, config + configLength, minimalSplitConfig);
}
}
}
return;
}

config[indexOfFirstUndecided] = IN_X;
indexOfFirstUndecided++;
searchAux(config, indexOfFirstUndecided, targetSizeOfSetX);

config[indexOfFirstUndecided - 1] = IN_Y;
for (int i = indexOfFirstUndecided; i < configLength; i++) {
config[i] = NOT_DECIDED;
}
searchAux(config, indexOfFirstUndecided, targetSizeOfSetX);
}


void produceTaskPoolAux(short* config, int indexOfFirstUndecided, int maxPregeneratedLength) {
if (indexOfFirstUndecided >= configLength || indexOfFirstUndecided >= maxPregeneratedLength) {
taskPool.push_back(config);
return;
}

short* secondConfig = new short[configLength];
copy(config, config + configLength, secondConfig);

config[indexOfFirstUndecided] = IN_X;
secondConfig[indexOfFirstUndecided] = IN_Y;

indexOfFirstUndecided++;

produceTaskPoolAux(config, indexOfFirstUndecided, maxPregeneratedLength);
produceTaskPoolAux(secondConfig, indexOfFirstUndecided, maxPregeneratedLength);
}


void produceMasterTaskPool() {
short* config = new short[configLength];
fill_n(config, configLength, NOT_DECIDED);
produceTaskPoolAux(config, 0, maxPregeneratedLevelFromMaster);
}


void produceSlaveTaskPool(short* initConfig) {
int indexOfFirstUndecided = 0;
for (int i = 0; i < configLength; i++) {
if (initConfig[i] == NOT_DECIDED) {
indexOfFirstUndecided = (i - 1 >= 0) ? i - 1 : 0;
break;
}
}

produceTaskPoolAux(initConfig, indexOfFirstUndecided, maxPregeneratedLevelFromSlave);
}


void consumeTaskPool() {
int indexOfFirstUndecided = min(maxPregeneratedLevelFromSlave, configLength);

#pragma omp parallel for schedule(dynamic)
for (auto& task : taskPool) {
searchAux(task, indexOfFirstUndecided, smallerSetSize);
}

while (taskPool.size()) {
taskPool.pop_back();
}
}


void sendTaskToSlave(int destination) {
ConfigWeightTask message =
ConfigWeightTask(configLength, minimalSplitWeight, minimalSplitConfig, taskPool.back());
taskPool.pop_back();
message.send(destination, TAG_WORK);
}


void distributeMasterTaskPool() {
for (int destination = 0; destination < numberOfProcesses; destination++) {
sendTaskToSlave(destination);
}
}


void saveConfigIfBest(ConfigWeight& resultMessage) {
if (resultMessage.getWeight() < minimalSplitWeight) {
minimalSplitWeight = resultMessage.getWeight();
for (int i = 0; i < configLength; i++) {
minimalSplitConfig[i] = (short)resultMessage.getConfig()[i];
}
}
}


void collectResults() {
int receivedResults = 0;
ConfigWeight resultMessage = ConfigWeight(configLength);
while (receivedResults < numberOfProcesses - 1) {
resultMessage.receive();
saveConfigIfBest(resultMessage);
receivedResults++;
}
}


void masterMainLoop() {
int workingSlaves = numberOfProcesses - 1;  
MPI_Status status;
ConfigWeight message = ConfigWeight(configLength);

while (workingSlaves > 0) {
status = message.receive(MPI_ANY_SOURCE, TAG_DONE);
saveConfigIfBest(message);

if (taskPool.size() > 0) {
sendTaskToSlave(status.MPI_SOURCE);
} else {
MPI_Send(nullptr, 0, MPI_SHORT, status.MPI_SOURCE, TAG_TERMINATE, MPI_COMM_WORLD);
workingSlaves--;
}
}

collectResults();
}


void master() {
produceMasterTaskPool();
distributeMasterTaskPool();
masterMainLoop();
}


void slave() {
MPI_Status status;
ConfigWeightTask taskMessage = ConfigWeightTask(configLength);
ConfigWeight resultMessage = ConfigWeight(configLength);

while (true) {
status = taskMessage.receive(MASTER, MPI_ANY_TAG);

if (status.MPI_TAG == TAG_TERMINATE) {
resultMessage.setWeightAndConfig(minimalSplitWeight, minimalSplitConfig);
resultMessage.send(MASTER);
return;
}
else if (status.MPI_TAG == TAG_WORK) {
saveConfigIfBest(taskMessage);
produceSlaveTaskPool(taskMessage.getTask());
consumeTaskPool();
resultMessage.setWeightAndConfig(minimalSplitWeight, minimalSplitConfig);
resultMessage.send(MASTER, TAG_DONE);
} else {
printf("ERROR, BAD MESSAGE");
}
}
}


void initSearch() {
configLength = graph.vertexesCount;
minimalSplitWeight = LONG_MAX;
minimalSplitConfig = new short[configLength];
taskPool = {};
}


void search() {
initSearch();

if (isMaster()) {
master();
} else {
slave();
}
}


void testInput(TestData& testData) {
graph = Graph();
graph.loadFromFile(testData.filePath);
smallerSetSize = testData.sizeOfX;
search();

if (isMaster()) {
cout << testData.filePath << endl;
cout << "Minimal weight: " << minimalSplitWeight << endl;
printConfig(minimalSplitConfig);

assert(minimalSplitWeight == testData.weight);
}

delete[] minimalSplitConfig;
}

int main(int argc, char** argv) {
steady_clock::time_point start = steady_clock::now();  

int provided, required = MPI_THREAD_MULTIPLE;
MPI_Init_thread(&argc, &argv, required, &provided);
if (provided < required) {
throw runtime_error("MPI library does not provide required threading support.");
}

MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

MPI_Comm_rank(MPI_COMM_WORLD, &processId);

if (argc < 4) {
cerr << "Usage: " << argv[0]
<< " <path_to_graph> <size_of_set_X> <number_of_threads> <solution>?" << endl;
return 1;
}

char* pathToGraph = argv[1];
int sizeOfSmallerSet = atoi(argv[2]);
int numberOfThreads = atoi(argv[3]);
int solution = -1;
if (argc == 5) {
solution = atoi(argv[4]);
}
TestData testData = TestData(pathToGraph, sizeOfSmallerSet, solution);

omp_set_dynamic(0);
omp_set_num_threads(numberOfThreads);

testInput(testData);

if (isMaster()) {
steady_clock::time_point end = steady_clock::now();  
auto time = duration<double>(end - start);
cout << "time: " << std::round(time.count() * 1000.0) / 1000.0 << "s" << endl;
cout << "________________________________" << endl;
}

MPI_Finalize();

return 0;
}
