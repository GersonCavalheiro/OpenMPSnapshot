#include "algo/interfaces/serial/SerialInstrumental.h"


void SerialInstrumental::prepareData() {
this->getGridNodes();
}

bool SerialInstrumental::checkData() const {
return true;
}

std::tuple<double, vec, vec, vec, vec> SerialInstrumental::getAllFields() {
return std::make_tuple(h, x, A, C, B);
}


vec SerialInstrumental::getGridNodes() {
vec res(node);
for (int i = 0; i < node; i++) {
res[i] = (double)i * h;
}

return res;
}

matr SerialInstrumental::createMatr() {
matr res(node, vec(node));

#pragma omp parallel shared(res, node, C, A, B) default(none) if (node > 500)
for (size_t i = 1; i < node; i++) {
for (size_t j = 0; j < node; j++) {
res[i][i]     = -C[0];
res[i][i - 1] = A[0];
res[i - 1][i] = B[0];
}
}

res[0][0] = 1.; res[node - 1][node - 1] = 1.;
res[0][1] = 0.; res[node - 1][node - 2] = 0.;

return res;
}