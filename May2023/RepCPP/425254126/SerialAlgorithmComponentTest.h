#pragma once

#include <functional>
#include <iostream>
#include <vector>
#include <test/common/BaseComponentTest.h>


class SerialAlgorithmComponentTest final : public SerialSweepMethod, public BaseComponentTest {
private:
void prepareSerialDataForTest(const SerialSweepMethod& sweepMethod) {
std::tie(v, u,
N, node, h, x,
A, C, B,
Phi, kappa, mu, gamma) = sweepMethod.getFields();
}

void setSerialFields(SerialSweepMethod& sweepMethod) {
sweepMethod.setAllFields(v, u,
N, node, h, x,
A, C, B,
Phi, kappa, mu, gamma);
}

public:
SerialAlgorithmComponentTest() : SerialSweepMethod() {}


std::tuple<vec, vec, vec, vec, vec, vec> testTask7(int n) {
SerialSweepMethod ssm(n);
this->prepareSerialDataForTest(ssm);

for (size_t i = 0; i < node; i++) {
u[i] = 10 + 90 * x[i] * x[i];
}

double total = 12. / (h * h);
A.assign(N - 1, total);
C.assign(N - 1, 2. * total + 5.);
B.assign(N - 1, total);

mu = std::make_pair(10., 100.);
kappa = std::make_pair(0., 0.);
gamma = std::make_pair(1., 1.);

for (size_t i = 0; i < node - 2; i++) {
Phi[i] = 450. * x[i + 1] * x[i + 1] - 2110.;
}

vec phi1 = Phi;

this->setSerialFields(ssm);

v = ssm.run();

Instrumental::compareVec(u, v);

Phi.assign(node, 1);
Phi[0] = mu.first; Phi[node - 1] = mu.second;
for (size_t i = 1; i < node - 1; i++) {
Phi[i] = 2110. - 450. * x[i] * x[i];
}

this->setSerialFields(ssm);
vec res = Instrumental::calcMatrVecMult(ssm.createMatr(), v);


Instrumental::compareDouble(ssm.calcR(res, Phi), 0);
Instrumental::compareDouble(ssm.calcZ(), 0);

return std::make_tuple(A, C, B, phi1, v, Phi);
}

void execute() {
std::vector<std::function<void()>> tests = {
[this]() { this->testTask7(5); }
};

BaseComponentTest::execute(tests, "Serial Component Test");
}
};