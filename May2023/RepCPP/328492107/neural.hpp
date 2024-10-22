

#pragma once

#include "common.hpp"
#include "dataset.hpp"
#include "activation.hpp"


class nn
{
public:
double** z, ** a, ** delta, *** weights;

std::vector<int> layers;

void set_layers(const std::vector<int>& l);
void set_z(const std::vector<int>& l);
void set_a(const std::vector<int>& l);
void set_delta(const std::vector<int>& l);
void set_weights(const std::vector<int>& l, const double min, const double max);
void compile(const std::vector<int>& l, const double min, const double max);
void zero_grad(double* (&X));
void forward(void);
void back_propagation(double* (&Y));
void optimize(void);
int get_label(double* (&y_pred));
int predict(double* (&X));
double mse_loss(double* (&Y), int dim);
int accuracy(double* (&Y), int dim);
void fit(dataset(&TRAIN));
void evaluate(dataset(&TEST));
void export_weights(std::string filename);
void summary(void);

nn()
{

}

~nn()
{
for (int i = 0; i < layers.size(); i += 1)
{
delete[] z[i];
delete[] a[i];
}
delete[] z;
delete[] a;
for (int i = 0; i < layers.size() - 1; i += 1)
{
delete[] delta[i];
}
delete[] delta;
for (int i = 1; i < layers.size(); i += 1)
{
for (int j = 0; j < layers[i] - 1; j += 1)
{
delete[] weights[i - 1][j];
}
delete[] weights[i - 1];
}
delete[] weights;

layers.clear();
layers.shrink_to_fit();
}
};