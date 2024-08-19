#pragma once
#include"AbstractGraph.h"
#include<algorithm>
#include<iterator>

class Incremental : public AbstractGraph {
std::vector<unsigned> findPath(int index, std::vector<unsigned> predecessors);
public:
Incremental(std::string graphFilename, unsigned vertexesNumber);
virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
virtual AbstractGraph::path* getCriticalPath() override;
std::pair<std::vector<long>, std::vector<unsigned>>* bellmanFord(unsigned row);

virtual bool linearMatrix() override { return false; };
};