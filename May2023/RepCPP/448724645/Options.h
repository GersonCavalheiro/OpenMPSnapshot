#pragma once

#include <string>
#include <vector>



class Options {
public:
Options();
~Options() { delete m_fileconfig; }

Options(const Options& other)
: nrThreads(other.nrThreads),
materialName(other.materialName),
nrPoints(other.nrPoints),
nearestNeighbors(other.nearestNeighbors),
nrLevels(other.nrLevels),
pathNo(other.pathNo),
paths(other.paths),
m_fileconfig(nullptr) {}

Options& operator=(const Options& other) {
nrThreads        = other.nrThreads;
materialName     = other.materialName;
nrPoints         = other.nrPoints;
nearestNeighbors = other.nearestNeighbors;
nrLevels         = other.nrLevels;
pathNo           = other.pathNo;
paths            = other.paths;
m_fileconfig     = nullptr;

return *this;
}

void Load();
void Save();

void print_options();

int         nrThreads;
std::string materialName;
int         nrPoints;
int         nearestNeighbors;
int         nrLevels;

int pathNo;

std::vector<std::vector<std::string>> paths;

protected:
void Open();
void Close();

std::string* m_fileconfig;
};
