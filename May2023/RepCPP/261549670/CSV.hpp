#pragma once

#include <fstream>
#include <sstream>
#include <cstddef>
#include <vector>
#include <utility>

#include "Cluster.hpp"
#include "Point.hpp"

namespace CSV {

template <typename T>
KMeans::Point<T> getNextLine(std::istream &str, char sep)
{
std::vector<T> rowData;
std::string line;
std::getline(str, line);
std::stringstream lineStr(line);
std::string cell;
while (std::getline(lineStr, cell, sep))
rowData.push_back(::atof(cell.c_str()));
return KMeans::Point<T>(rowData);
}

template <typename T>
std::vector<KMeans::Point<T>> readCSV(const std::string &filename, char sep = ',')
{
std::ifstream file(filename);
std::vector<KMeans::Point<T>> data;
while (!file.eof())
data.push_back(getNextLine<T>(file, sep));
file.close();
data.pop_back();
return data;
}

template <typename T>
void writeCSV(const std::string &filename, const std::vector<KMeans::Point<T>> &data)
{
std::ofstream file(filename);
for (auto &P : data) {
for(auto &x : P.GetCoord())
file << x << ' ';
file << P.GetClusterId() << '\n';
}
file.close();
}


template<typename T>
void writeCSV(const std::string &filename, const std::vector<std::pair<T, T>> &data)
{
std::ofstream file(filename);
std::size_t N = data.size();
for(std::size_t i = 0; i < N; i++) {
file << data[i].first << ' ' << data[i].second;
if (i < N - 1)
file << '\n';
}
file.close();
}

} 
