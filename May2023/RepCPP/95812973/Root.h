#pragma once

#include <map>

#include "Node.h"

class Root {
private:
std::vector<Node*> children;
double probability;

public:
Root();
~Root();

Node* getChild(unsigned short key);
void addChild(unsigned short key, double probability);

void updateProbability(double probability);

void log();
void logMonteCarlo();
};
