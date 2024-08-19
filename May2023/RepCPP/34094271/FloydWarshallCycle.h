
#pragma once
#include <vector>
#include <algorithm>
#include <cassert>

class FloydWarshallCycle
{
public:
FloydWarshallCycle(int numberOfNodes);
~FloydWarshallCycle();

void AddEdge(int src, int dest);

std::vector<int> GetShortestCycle(int src);

std::vector< std::vector<int> > GetAllUniqueCycles();

std::vector< std::vector<int> > GetAllCommonCycles();

int GetCentricNode();

private:
int numberOfNodes;

std::vector< std::vector<int> > connections;

std::vector< std::vector<int> > graph;

std::vector< std::vector<int> > next;

void floydWarshall();

std::vector<int> getPath(int src, int dest);
std::vector<int> getPath(int connectionIndex);

void setDefaults();

void setValues(int exceptThisOne);

std::vector<int> getConnectionsFor(int index);

std::vector<int> findMinimumPath(const std::vector< std::vector<int> > &paths);

std::vector< std::vector<int> > getUniqueVectors(std::vector< std::vector<int> > allCycles);

bool isVectorsEqual(const std::vector<int> &first, const std::vector<int> &second);

bool haveCommonElements(const std::vector<int> &first, const std::vector<int> &second);

std::vector<int> returnCombinedSet(const std::vector<int> &first, const std::vector<int> &second);
};

