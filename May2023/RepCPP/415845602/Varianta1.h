#pragma once

#include "mpi.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

class Varianta1 {
private:
vector<int> nr1;
vector<int> nr2;
vector<int> nr3;
void writeToFile(vector<int>);
public:
Varianta1();
void run();
};