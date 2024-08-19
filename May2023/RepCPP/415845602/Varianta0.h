#pragma once
#include <fstream>
#include <vector>
#include <string>

using namespace std;

class Varianta0 {
private:
vector<int> nr1;
vector<int> nr2;
vector<int> nr3;
int startTime, endTime;
vector<int> readFromFile(string);
void writeToFile(vector<int>);
public:
Varianta0();
void run();
};