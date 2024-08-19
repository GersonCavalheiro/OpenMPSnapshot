#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>

#include "algo/interfaces/Instrumental.h"

class BaseComponentTest {
public:
static void execute(const std::vector<std::function<void()>>& tests, const std::string& name) {
TestRunner testRunner;

printf("------------ %s ------------\n", name.c_str());
for (const auto& test : tests) {
RUN_TEST(testRunner, test);
print()
}
}
};