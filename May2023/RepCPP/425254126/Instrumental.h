#pragma once

#include <omp.h>

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>

#define print() printf("---\n");
#define printd(a) printf("%s = %f\n", #a, a);
#define printi(a) printf("%s = %d\n", #a, a);

#define EPS 0.0001;

using matr = std::vector<std::vector<double>>;
using vec = std::vector<double>;
using pairs = std::pair<double, double>;
using str = std::string;


class Instrumental {
protected:
size_t N, node;
vec u, v;

public:
Instrumental() : Instrumental(5) {}

explicit Instrumental(size_t n) : N(n), node(n + 1), u(node), v(node) {}

void setN(size_t n);

void setUV(vec& u_, vec& v_);

virtual void prepareData();

virtual bool checkData() const;

static void printVec(const vec& a, const str& name);

static void printMatr(const matr& a, const str& name);

double calcR(const vec& x, const vec& b) const;

double calcZ() const;

static vec calcMatrVecMult(const matr& A, const vec& b);

std::tuple<size_t, size_t, vec, vec> getAllFields() const;

static bool compareDouble(double a, double b);

static bool compareMatr(const matr& a, const matr& b);

static bool compareVec(const vec& a, const vec& b);
};