#include "util.h"


void atomicAdd(QVector3D& vecA, const QVector3D& vecB) {
for (int k = 0; k < 3; ++k) {
float& a = vecA[k];
const float b = vecB[k];
#pragma omp atomic
a += b;
}
}
