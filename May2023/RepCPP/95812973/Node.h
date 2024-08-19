#pragma once

#include <vector>

class Node {
private:
unsigned short team;
double probability;
public:
std::vector<Node*> children;
Node(
unsigned short team,
double probability = 0
);
~Node();

unsigned short getTeam() const;

Node* getChild(unsigned short team);
void addChild(unsigned short team, double probability);

double getProbability() const;
void updateProbability(double delta);

void log(
unsigned short level = 0,
double initialprobability = 1
);
};

bool compareNodePointers(Node* a, Node* b);