#pragma once


#include <vector>
#include <iostream>
#include <algorithm>

template<class T>
class SortIndex
{
public:
SortIndex( const std::vector<T>& whichV ):v( whichV )
{}

bool operator()( int i1, int i2 )
{
return v[i1] < v[i2];
}

std::vector<int>& sort()
{
pos.clear();
for( unsigned int i = 0; i < v.size(); i++ )
pos.push_back(i);
stable_sort( pos.begin(), pos.end(), *this );
return pos;
}

std::vector<int>& getSorted()
{
return pos;
}

private:
const std::vector<T>& v;
std::vector<int> pos;

};



