

#pragma once

#include <iostream>

namespace advscicomp {

using LargeComplexT = int;

class Yasreet 
{
LargeComplexT *large_thing_;

public:

Yasreet() : large_thing_(new LargeComplexT)
{}

Yasreet(LargeComplexT i) : large_thing_(new LargeComplexT(i))
{}

void SetField(LargeComplexT i)
{
*large_thing_ = i;
}

const LargeComplexT& GetField() const
{
return *large_thing_;
}

};


inline
std::ostream& operator<<(std::ostream & out, Yasreet const& y)
{
out << y.GetField();
return out;
}

}

