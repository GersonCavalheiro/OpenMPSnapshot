#include <bits/stdc++.h>
#pragma once
using namespace std;

bool is_ordered(vector<int>& vec)
{
return is_sorted(vec.begin(), vec.end());
}

void next(vector<int>& vec, mt19937& r)
{
int n = vec.size();
for (int i = 0; i < n - 1; i++)
{
int range = n-1 - i + 1;
unsigned int j = r() % range + i;
int t = vec[j];
vec[j] = vec[i];
vec[i] = t;
}
}
