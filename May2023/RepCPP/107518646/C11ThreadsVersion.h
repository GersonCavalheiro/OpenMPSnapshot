#pragma once
#include"AbstractGraph.h"
#include<thread>
#include<mutex>
#include<algorithm>

class C11ThreadsVersion : public AbstractGraph {
protected:
std::vector<std::mutex*>* mutexes;

long* distance;

public:
C11ThreadsVersion(std::string graphFilename, unsigned vertexesNumber);
virtual AbstractGraph::path* getCriticalPath(unsigned vertexStart) override;
virtual AbstractGraph::path* getCriticalPath() override;
void bellmanFord(const int threadIndex, unsigned row);

virtual bool linearMatrix() override { return false; };
};

