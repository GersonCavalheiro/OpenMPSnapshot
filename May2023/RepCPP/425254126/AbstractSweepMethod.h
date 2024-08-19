#pragma once

#include <iostream>
#include <vector>


class AbstractSweepMethod {
public:
virtual std::vector<double> run() = 0;
};
