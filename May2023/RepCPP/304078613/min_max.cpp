#include "Bench.h"
#include "rapidcsv.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>

#ifndef _OPENMP

void omp_set_num_threads(int _x) {}

#endif

class MinMax {
public:
MinMax(std::shared_ptr<Bench> bencher,
rapidcsv::Document doc,
std::vector<std::string> cvnames
) {
this->bencher = bencher;
this->cvnames = cvnames;
this->doc = doc;
}

std::vector<double> min_max(std::vector<double> xv) {
auto max_t = max(xv);
auto min_t = min(xv);

std::vector<double>nxv = std::vector<double>();

#pragma omp parallel
{
std::vector<double> nxv_private;

#pragma omp for nowait
for (auto x : xv) {
nxv_private.push_back((x - min_t) / (max_t - min_t));
}

#pragma omp critical
nxv.insert(nxv.end(), nxv_private.begin(), nxv_private.end());
}

return nxv;
}

std::vector<std::vector<double>> process() {
int id = bencher->add_op("DS->MinMaxDS");

auto v = std::vector<std::vector<double>>{
min_max(doc.GetColumn<double>("R")),
min_max(doc.GetColumn<double>("G")),
min_max(doc.GetColumn<double>("B")),
doc.GetColumn<double>("SKIN")
};

bencher->end_op(id);

return v;
}

void spit_csv(
std::string filename,
std::vector<std::vector<double>> ds,
std::vector<std::string>cnames
) {
std::ofstream out;
out.open(filename);

for (auto name : cnames) {
out << name << ",";
}

out << "\n";

for (int i = 0; i < ds[0].size(); ++i) {
for (int j = 0; j < cnames.size(); ++j) {
out << ds[j][i] << ((j == cnames.size() - 1) ? "\n" : ",");
}
}

out.close();
}

void process_all_and_spit_output() {
spit_csv("min_max-skin.csv", process(), cvnames);
}

private:
std::vector<std::string> cvnames;
rapidcsv::Document doc;
std::shared_ptr<Bench> bencher;

double max(std::vector<double> xv) {
int id = bencher->add_op("MAX op");

double max = std::numeric_limits<double>::min();

#pragma omp parallel for
for (auto x : xv) {
if (x > max) {
max = x;
}
}

bencher->end_op(id);

return max;
}

double min(std::vector<double> xv) {
int id = bencher->add_op("MIN op");

double min = std::numeric_limits<double>::max();

#pragma omp parallel for
for (auto x : xv) {
if (x < min) {
min = x;
}
}

bencher->end_op(id);

return min;
}
};

int main(int argc, char *argv[]) {
int expected_threads = std::stoi(std::string(argv[1]));
std::string file_path = std::string(argv[2]);
rapidcsv::Document doc(file_path);

omp_set_num_threads(expected_threads);

bool with_bench = false;

if (argc == 4) {
with_bench = true;
}

std::cout << "\n";

if (with_bench) {
std::vector<std::shared_ptr<Bench>> benchers;

for (int i = 0; i < 30; ++i) {
auto bencher = std::make_shared<Bench>(Bench());
auto algo = std::make_unique<MinMax>(MinMax(bencher, doc, std::vector<std::string>{"R", "G", "B", "SKIN"}));
algo->process();
benchers.push_back(bencher);
}

Bench::print_benches(benchers);
} else {
auto bencher = std::make_shared<Bench>(Bench());
auto algo = std::make_unique<MinMax>(MinMax(bencher, doc, std::vector<std::string>{"R", "G", "B", "SKIN"}));

algo->process_all_and_spit_output();

bencher->print_bench();
}

return 0;
}
