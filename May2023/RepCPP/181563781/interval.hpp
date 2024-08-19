#pragma once
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace grid2grid {
struct interval {
int start = 0;
int end = 0;

interval() = default;

interval(int start, int end);

int length() const;

bool contains(interval other) const;

bool non_empty() const;
bool empty() const;

interval intersection(const interval &other) const;


std::pair<int, int> overlapping_intervals(const std::vector<int> &v) const;

bool operator==(const interval &other) const;
bool operator!=(const interval &other) const;
bool operator<(const interval &other) const;
};

std::ostream &operator<<(std::ostream &os, const interval &other);
} 
