

#pragma once

#include "header.h"
#include "timer.h"
#include "optparse.h"

namespace trinity { namespace tools {

uint32_t hash(const uint32_t id);

int format(int num);

void separator();

void ltrim(std::string& line);

std::string basename(const std::string& s);

bool exists(const std::string& path);

void abort(char option, const char* msg, const optparse::Parser& parser);

bool equals(const char* s1, const char* s2);

std::string getExt(const char* path);

std::ifstream& seekToLine(int nb, std::ifstream& file);

bool isDigit(const char* arg);

std::string rootOf(const std::string& path);

std::string testcase(const std::string& path);

std::string replaceExt(const std::string& fname, const std::string& ext);

void showElapsed(Time& tic, const char* msg, int step);


template<typename type_t>
void display(const std::vector<type_t>& list) {

std::stringstream buffer;
buffer << "[";
for (auto it = list.begin(); it != list.end(); ++it) {
buffer << *it;
if (it + 1 != list.end()) { buffer << ","; }
}
buffer << "]";
std::printf("%s\n", buffer.str().data());
}


template<typename type_t>
void erase(type_t needle, std::vector<type_t>& list) {
auto found = std::find(list.begin(), list.end(), needle);
assert(found != list.end());
std::swap(*found, list.back());
list.pop_back();
}


}} 
