

#pragma once

#include "header.h"
#include "timer.h"

namespace trinity {

class RMAT {

friend class Partit;

public:

RMAT();
RMAT(const RMAT& other) = delete;
RMAT& operator=(RMAT other) = delete;
RMAT(RMAT&& other) noexcept = delete;
RMAT& operator=(RMAT&& other) noexcept = delete;
~RMAT();

void reset();
void load(std::string path);
void info(std::string name);

private:

void saveChrono();
int elapsed();

Graph graph;

struct {
int nodes;
int edges;
int rounds;
int error;
int color;
} nb;

struct {
int max;
int avg;
} deg;

struct { double ratio; } stat;
struct { Time start; }   time;
};

} 
