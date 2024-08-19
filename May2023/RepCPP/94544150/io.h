
#pragma once

#include <Eigen/Core>
#include <string>
#include <fstream>
#include <sstream>

namespace LR {

namespace IO {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::ifstream;
using std::stringstream;


std::pair<MatrixXd, VectorXd> load_txt(const char *filename, const int row, const int col) {
MatrixXd data(row, col + 1);
VectorXd tags(row);

ifstream f(filename);
assert(f.good());

string line;
for (size_t i = 0; i < row; i++) {
std::getline(f, line);
stringstream ss(line);

string tag;
ss >> tag;
tags(i) = std::stoi(tag);

string idx, val;
while (ss >> idx >> val) {
data(i, std::stoi(idx)) = std::stod(val);
}

data(i, 0) = 1;
}

return std::make_pair(data, tags);
}

std::pair<MatrixXd, VectorXd> load_csv(const char *filename, const int row, const int col) {
MatrixXd data(row, col + 1);
VectorXd tags(row);

ifstream f(filename);
assert(f.good());

string line, tmp;
for (size_t i = 0; i < row; i++) {
std::getline(f, line);
stringstream ss(line);

data(i, 0) = 1;
for (size_t j = 1; j <= col; j++) {
std::getline(ss, tmp, ',');
data(i, j) = std::stod(tmp);
}

std::getline(ss, tmp, ',');
tags(i) = std::stoi(tmp);
}

return std::make_pair(data, tags);
}
}

}
