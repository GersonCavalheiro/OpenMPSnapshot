#pragma once
#include <random>
#include <ctime>
#include <string>
using namespace std;

class Data {
private:
int N;

public:
Data(int N);

vector<vector<int>> FillMatrixWithNumber(int number);
vector<int> FillVectorWithNumber(int number);

vector<vector<int>> MatrixInput(string name);
vector<int> VectorInput(char name);
int NumInput(char name);

void MatrixOutput(vector<vector<int>> MA, string name);
void VectorOutput(vector<int> A, char name);
void NumOutput(int a, char name);


vector<vector<int>> MatrixTransp(vector<vector<int>> MA);

vector<vector<int>> MatrixMult(vector<vector<int>> MA, vector<vector<int>> MB);


vector<int> VectorMatrixMult(vector<int> A, vector<vector<int>> MA);

vector<int> SumVectors(vector<int> A, vector<int> B);

vector<int> IntVectorMult(int a, vector<int> A);

vector<int> SortVector(vector<int> A);


vector<int> Func1(vector<int> A, vector<int> B, vector<int> C, vector<vector<int>> MA, vector<vector<int>> ME);

vector<vector<int>> Func2(vector<vector<int>> MG, vector<vector<int>> MH, vector<vector<int>> MK);

vector<int> Func3(int t, vector<int> V, vector<int> O, vector<int> P, vector<vector<int>> MO, vector<vector<int>> MP, vector<vector<int>> MR);

};

