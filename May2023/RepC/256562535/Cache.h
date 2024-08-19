#pragma once
#include <list>
#include <unordered_map>
class Cache {
std::list<int32_t> dq;
std::unordered_map<int32_t*, std::list<int32_t>::iterator> ma;
int csize; 
int cacheMiss = 0;
int cacheHit = 0;
public:
Cache(int);
void refer(int32_t*);
void display();
void displayMisses();
};
