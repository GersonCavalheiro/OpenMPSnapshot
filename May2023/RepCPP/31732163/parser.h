

#pragma once

#include "tools.h"

namespace trinity {

class Parser {

public:
struct {
int cores    = 0;
int hw_cores = 0;
int threads  = 0;
int bucket   = 0;
int rounds   = 0;
int depth    = 0;
int norm     = 0;
int verb     = 0;
int makespan = 0;
int size[2]  = {0,0};

double target = 0.;
double h_min  = 0.;
double h_max  = 0.;

std::string arch   = "";
std::string name   = "";
std::string input  = "";
std::string result = "";
std::string solut  = "";
} param;

Parser() = default;
Parser(int argc, char* argv[]);
~Parser() = default;

void recap(Stats* stat);
void dump(Stats* stat);

private:
void showDesc();
};

}