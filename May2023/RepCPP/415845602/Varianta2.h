#pragma once

#include "mpi.h"
#include <fstream>
#include <vector>
#include <string>

using namespace std;

class Varianta2 {
private:
vector<int> nr1;
vector<int> nr2;
vector<int> nr3;
vector<int> readFromFile(string);
void writeToFile(vector<int>);
public:
Varianta2();
void run();
};
