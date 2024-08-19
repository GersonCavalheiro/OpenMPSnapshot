#pragma once
#include "_main.hxx"




struct SumOptions {
int repeat;

SumOptions(int repeat=1) :
repeat(repeat) {}
};


template <class T>
struct SumResult {
T     result;
float time;

SumResult(T result, float time) :
result(result), time(time) {}
};
