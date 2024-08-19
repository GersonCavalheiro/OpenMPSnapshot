#include "Bench.h"
#include "rapidcsv.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>

using namespace std;

#ifndef _OPENMP

void omp_set_num_threads(int _x) {}

#endif

class Knn {
public:
Knn(shared_ptr <Bench> bencher,
rapidcsv::Document doc,
vector <string> cvnames
) {
this->bencher = bencher;
this->cvnames = cvnames;
this->doc = doc;
}

vector <vector<double>> process() {
int id = bencher->add_op("DS->KnnDS");

auto v = vector < vector < double >> {
doc.GetColumn<double>("R"),
doc.GetColumn<double>("G"),
doc.GetColumn<double>("B"),
doc.GetColumn<double>("SKIN")
};

vector <vector<double>> test, learn;

for (int i = 0; i < v[0].size(); ++i) {
if(i < v[0].size() * 0.8){
learn.push_back({v[0][i], v[1][i], v[2][i], v[3][i]});
} else {
test.push_back({v[0][i], v[1][i], v[2][i], v[3][i]});
}
}

int success = 0;
#pragma omp parallel for default(shared)
for (int i = 0; i < test.size(); ++i) {
test[i].push_back(search(learn, test[i]));
if(test[i][4] == test[i][3])
++success;
}

bencher->end_op(id);

return test;
}

void spit_csv(
string filename,
vector <vector<double>> ds,
vector <string> cnames
) {
ofstream out;
out.open(filename);

for (auto name : cnames) {
out << name << ",";
}

out << "\n";

for (int i = 0; i < ds.size(); ++i) {
for (int j = 0; j < cnames.size() + 1; ++j) {
out << ds[i][j] << ((j == cnames.size()) ? "\n" : ",");
}
}
out.close();
}

void process_all_and_spit_output() {
spit_csv("knn-skin.csv", process(), cvnames);
}

private:
vector <string> cvnames;
rapidcsv::Document doc;
shared_ptr <Bench> bencher;

double calculateDistance(vector<double> newPoint, vector<double> pointOuter) {
return sqrt(pow(newPoint[0] - pointOuter[0], 2) + pow(newPoint[1] - pointOuter[1], 2) + pow(newPoint[2] - pointOuter[2], 2));
}

int search(vector<vector<double>> learn, vector<double> test) {
const int id = bencher->add_op("KNN one record");
vector<vector<double>> nearestPoints;
const short maxElements = 5;

nearestPoints.push_back({calculateDistance(test, learn[0]), learn[0][3]});

for(int i = 1 ; i < maxElements; ++i) {
double temp = calculateDistance(test, learn[i]);
bool flgFound = false;

for(int j = 0; j < nearestPoints.size(); ++j){
if(nearestPoints[j][0] > temp) {
nearestPoints.insert(nearestPoints.begin() + j, vector<double>{temp, learn[i][3]});
flgFound = true;
break;
}
}
if(!flgFound) {
nearestPoints.push_back(vector<double>{temp, learn[i][3]});
}
}

#pragma omp parallel for default(shared)
for(int i = maxElements ; i < learn.size(); ++i) {
double temp = calculateDistance(test, learn[i]);
bool flgFound = false;
for(int j = 0; j < maxElements  && !flgFound; ++j){
if(nearestPoints[j][0] > temp) {
#pragma omp critical(updateNearestPoints)
{
nearestPoints.insert(nearestPoints.begin() + j, vector<double>{temp, learn[i][3]});
nearestPoints.pop_back();
flgFound = true;
}
}
}
}
bencher->end_op(id);

return (nearestPoints[0][1] + nearestPoints[1][1] + nearestPoints[2][1] + nearestPoints[3][1] + nearestPoints[4][1]) > 2;
}
};

int main(int argc, char *argv[]) {
int expected_threads = stoi(string(argv[1]));
string file_path = string(argv[2]);
ifstream f(file_path.c_str());

if(f.bad()) {
exit(1);
}

rapidcsv::Document doc(file_path);

omp_set_num_threads(expected_threads);

bool with_bench = false;

if (argc == 4) {
with_bench = true;
}

cout << "\n";

if (with_bench) {
vector <shared_ptr<Bench>> benchers;

for (int i = 0; i < 30; ++i) {
auto bencher = make_shared<Bench>(Bench());
auto algo = make_unique<Knn>(Knn(bencher, doc, vector < string > {"R", "G", "B", "SKIN"}));
algo->process();
benchers.push_back(bencher);
}

Bench::print_benches(benchers);
} else {
auto bencher = make_shared<Bench>(Bench());
auto algo = make_unique<Knn>(Knn(bencher, doc, vector < string > {"R", "G", "B", "SKIN"}));

algo->process_all_and_spit_output();

bencher->print_bench();
}

return 0;
}
