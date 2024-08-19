#pragma once

#include <string>
#include <cassert>

class Range {
public:
Range (const unsigned int begin, const unsigned int span)
: begin(begin), span(span) {};
Range() = default;
~Range () = default;

operator bool() const { return span > 0;}

size_t begin;
size_t span;

private:

};

inline std::ostream& operator << (std::ostream& out, Range const& rhs)
{
out << " begin " + std::to_string(rhs.begin);
out << " span " + std::to_string(rhs.span);
return out;
}


inline Range get_valid_range(const std::string& sequence)
{
size_t lower = 0;
size_t upper = sequence.length();

assert(upper);

while(lower < upper and sequence.c_str()[lower] == '-') {
lower++;
}

while(upper > lower and sequence.c_str()[upper - 1u] == '-') {
upper--;
}

return Range(lower, upper - lower);
}


