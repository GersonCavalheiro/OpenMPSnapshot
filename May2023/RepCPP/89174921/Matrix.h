#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

typedef struct matrix_t {
int rows;
int cols;
int **table;
} matrix_t;

matrix_t* create_matrix(int rows, int cols);
int destroy_matrix(matrix_t *matrix);
void print_matrix(matrix_t *matrix);
void rand_matrix(matrix_t *matrix, int percent);
bool solve(matrix_t *matrix, int index);
