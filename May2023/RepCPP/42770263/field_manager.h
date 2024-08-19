#pragma once
#include <vector>
#include <string>
#include <iostream>

class FieldManager {
std::vector<std::vector<bool> > field;
long numberOfiterations;
long currentIteration;
int numberOfThreads;
std::ostream& out;
bool stopped;
public:
FieldManager(std::ostream& out);

void start(const std::string fileName, int numberOfThreads);

void start(long width, long height, int numberOfThreads);

void status();

void run(long numberOfIterations);

void stop();

void quit(bool useless);
private:
void parseCSV(std::string fileName);

void generateField(long wigth, long height);

void runIteration();

int sumOfNeighbours(long i, long j);
};
