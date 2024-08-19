#pragma once

#include <utility>

class Area {
public:
pairs x;
pairs y;
pairs t;

Area() = default;

Area(pairs x, pairs y, pairs t) : x(std::move(x)), y(std::move(y)), t(std::move(t)) {}

vec3<pairs> getData() const {
vec3<pairs> res;

res.assign(3, std::make_pair(0, 0));

size_t i = 0;
for (const auto& elem : {x, y, t}) {
res[i++] = elem;
}

return res;
}

friend std::ostream &operator<<(std::ostream &os, const Area &parameters) {
return os << "x: (" << parameters.x.first << ", " << parameters.x.second << "), "
<< "y: (" << parameters.y.first << ", " << parameters.y.second << "), "
<< "t: (" << parameters.t.first << ", " << parameters.t.second << ")\n";
}

friend std::istream& operator>>(std::istream& in, Area& parameters) {
return in >> parameters.x.first >> parameters.x.second
>> parameters.y.first >> parameters.y.second
>> parameters.t.first >> parameters.t.second;
}
};