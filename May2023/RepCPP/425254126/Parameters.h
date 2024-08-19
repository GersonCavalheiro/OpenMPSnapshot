#pragma once

#include <algo/interfaces/Instrumental.h>

#include <ostream>

class Parameters {
public:
pairs alpha;
pairs omega;
pairs beta;
pairs gamma;

Parameters() = default;

Parameters(pairs alpha, pairs omega, pairs beta, pairs gamma)
: alpha(std::move(alpha)), omega(std::move(omega)), beta(std::move(beta)), gamma(std::move(gamma)) {}

vec3<pairs> getData() const {
vec3<pairs> res;

res.assign(4, std::make_pair(0, 0));

size_t i = 0;
for (const auto& elem : {alpha, omega, beta, gamma}) {
res[i++] = elem;
}

return res;
}

friend std::ostream &operator<<(std::ostream &os, const Parameters &parameters) {
return os << "alpha: (" << parameters.alpha.first << ", " << parameters.alpha.second << "), "
<< "beta: (" << parameters.beta.first << ", " << parameters.beta.second << "), "
<< "omega: (" << parameters.omega.first << ", " << parameters.omega.second << "), "
<< "gamma: (" << parameters.gamma.first << ", " << parameters.gamma.second << ")\n";
}

friend std::istream& operator>>(std::istream& in, Parameters& parameters) {
return in >> parameters.alpha.first >> parameters.alpha.second
>> parameters.beta.first >> parameters.beta.second
>> parameters.gamma.first >> parameters.gamma.second;
}
};