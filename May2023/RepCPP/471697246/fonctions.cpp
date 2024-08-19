#include "fonctions.hpp"

void generate_vector(size_t size, int *vector, size_t zeros) {
for (size_t i = 0; i < zeros; i++) vector[i] = 0;
for (size_t i = zeros; i < size; i++) vector[i] = rand() % 20;
}

void matrix_vector(size_t size, const int* const matrix, const int* const vector, int* const output) {
int index = 0;
while (vector[index] == 0) ++index;
for (size_t i = 0; i < size; i++) {
output[i] = 0;
for (size_t j = index; j < size; j++) output[i] += vector[j] * matrix[i * size + j];
}
}

void matrix_vector_omp(size_t size, const int* const matrix, const int* const vector, int* const output) {
int index = 0;
while (vector[index] == 0) ++index;

#pragma omp parallel for
for (size_t i = 0; i < size; i++) {
output[i] = 0;
for (size_t j = index; j < size; j++) output[i] += vector[j] * matrix[i * size + j];
}
}