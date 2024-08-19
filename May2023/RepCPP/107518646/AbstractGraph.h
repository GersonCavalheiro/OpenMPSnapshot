#pragma once
#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include<ctime>

class AbstractGraph {
protected:
bool GENERATE_GRAPH = false;
bool usingLinearMatrix;
int** matrix;
int* linear_matrix;

std::string graphFilename;
unsigned vertexesNumber;

inline int** getMatrix();
inline int getValueFromMatrix(unsigned x, unsigned y);

int rowCount(int y);
int colCount(int x);


public:
struct path {
int pathLength;
std::vector<unsigned> vertexes;
};

void init(std::string graphFilename, unsigned vertexesNumber);
virtual ~AbstractGraph();

void loadGraphFromFile(std::string filename);
void generateGraph();
inline int* getVertexEdges(unsigned vertexNumber);

void printMatrix();
void saveMatrix();

virtual AbstractGraph::path* getCriticalPath(unsigned vertexNumber) = 0;
virtual AbstractGraph::path* getCriticalPath() = 0;

virtual bool linearMatrix() = 0;
};