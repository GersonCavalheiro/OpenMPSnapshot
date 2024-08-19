#pragma once

#include "algo/interfaces/Instrumental.h"

#include <utility>


class SerialInstrumental : public Instrumental {
protected:
double h;
vec x;
vec A, C, B;

public:
SerialInstrumental() :
Instrumental(),
h(1 / static_cast<double>(N)),
x(std::move(this->getGridNodes())) {}

explicit SerialInstrumental(size_t n) :
SerialInstrumental(n, vec(n, 0), vec(n, 0), vec(n, 0)) {}

SerialInstrumental(const vec& a, vec c, vec b) :
SerialInstrumental(a.size() + 1, a, std::move(c), std::move(b)) {}

SerialInstrumental(size_t n, vec a, vec c, vec b) :
Instrumental(n),
h(1 / static_cast<double>(N)),
x(this->getGridNodes()),
A(std::move(a)), C(std::move(c)), B(std::move(b)) {}

void prepareData() override;

bool checkData() const override;

vec getGridNodes();

std::tuple<double, vec, vec, vec, vec> getAllFields();


matr createMatr();
};