
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <list>
#include <cmath>
#include <omp.h>
using namespace std;
using namespace std::chrono;

vector<string> InputReader() {
ifstream file("Text2.txt");
string str;
vector<string> stringVector;
while (file >> str)
{
stringVector.emplace_back(str);
}
return stringVector;
}
pair<string, int> Map(const string& s) {
pair<string, int> key_val_pair;
key_val_pair.first = s;
key_val_pair.second = 1;
return key_val_pair;
}
pair<string, int> Reduce(vector <pair<string, int>> keyValuePairVector) {
pair<string, int> key_val_pair;
string firstKey = keyValuePairVector[0].first;
int count = 0;
for (auto & i : keyValuePairVector) {
if (i.first == firstKey) {
count++;
}
}
key_val_pair.first = firstKey;
key_val_pair.second = count;
return key_val_pair;
}
void printPair(const pair<string, int>& reducePair) {
cout << reducePair.first << " => " << reducePair.second << endl;
}
void firstStage(int thread, vector<vector<pair<string, int>>>& keyValuePairVector, vector<string> wordVector) {
int divWork = (int) ceil(wordVector.size()/3) + 1;

for (int i = thread * divWork; i < (thread + 1) * divWork; i++) {
if (i < wordVector.size()) {

pair<string, int> wordPair = Map(wordVector[i]);
vector<pair<string, int>> column;
column.emplace_back(wordPair);
bool existsInVector = true;
for (auto & x : keyValuePairVector) {
if (x[0].first == wordVector[i]) {
x.emplace_back(wordPair);
existsInVector = false;
break;
}
}
if (existsInVector) {
keyValuePairVector.emplace_back(column);
}
}

}
}
void secondStage(int thread, vector<vector<pair<string, int>>> keyValuePairVector, vector<pair<string, int>> &countedKeyValuePairsVector) {
int divWork = (int) ceil(keyValuePairVector.size()/3) + 1;
for (int i = thread * divWork; i < (thread + 1) * divWork; i++) {
if (i >= keyValuePairVector.size()) {
break;
}
if (keyValuePairVector[i][0].first.empty()) {
continue;
} 
countedKeyValuePairsVector.emplace_back(Reduce(keyValuePairVector[i]));
}
}
void thirdStage(const vector<pair<string, int>>& countedKeyValuePairsVector) {
for (auto & i : countedKeyValuePairsVector) {
printPair(i);
}
cout << "Total unique words: " << countedKeyValuePairsVector.size() << endl;
}

void wordCount() {
vector<vector<pair<string, int>>> keyValuePairVector;
vector<string> wordVector;
wordVector = InputReader();
vector<thread> threadVector;
vector<pair<string, int>> countedKeyValuePairsVector;

vector<pair<string, int>> dummy;
pair<string, int> dummyPair;
dummy.emplace_back(dummyPair);
keyValuePairVector.emplace_back(dummy);
#pragma omp parallel num_threads(3)
{

int id = omp_get_thread_num();
int total = omp_get_num_threads();
#pragma omp critical
{
int data = id;
firstStage(id, keyValuePairVector, wordVector);
}

}

#pragma omp parallel num_threads(3)
{
int id = omp_get_thread_num();
int total = omp_get_num_threads();

#pragma omp critical
{
int data = id;
secondStage(id, keyValuePairVector, countedKeyValuePairsVector);
}
}

sort(countedKeyValuePairsVector.begin(), countedKeyValuePairsVector.end(), [](auto &left, auto &right) {
return left.second > right.second;
});

thirdStage(countedKeyValuePairsVector);
}



int main() {

wordCount();


}

