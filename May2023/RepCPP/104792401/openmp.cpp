

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdexcept>

#include <omp.h>

using namespace std;


class Timer{
private:
struct timespec start_time;
struct timespec end_time;
long msec;
void start(){
clock_gettime(CLOCK_MONOTONIC,&start_time);
}
public:
Timer(){
this->start();
}
void stop(string arg){
clock_gettime(CLOCK_MONOTONIC,&end_time);
msec = (end_time.tv_sec - start_time.tv_sec)*1000 + (end_time.tv_nsec - start_time.tv_nsec)/1000000;
cout << "Took: " << msec << " msec" << arg << endl;
}
};


class Mat{
private:
double random_double(){
return ((float)rand()/RAND_MAX)*2-1;
}
void random_mat(){
srand( (unsigned)time( NULL ) );
int length = this->n_rows * this->n_cols;
#pragma omp parallel for
for(int i=0; i<length; ++i) this->data[i] = random_double();
}
public:
int n_rows;
int n_cols;
double* data;
Mat(int _n_rows, int _n_cols, bool random = false){
n_rows = _n_rows;
n_cols = _n_cols;
data = new double[this->n_rows * this->n_cols];
if(random) this->random_mat();
}
Mat* matmult(Mat *other_mat){
if(this->n_cols != other_mat->n_rows){
throw invalid_argument("\nMatrices dimensions do NOT match for dot product\n");
}
Mat *new_mat = new Mat(this->n_rows, other_mat->n_cols);
int i,j,k;
double* a = this->data;
double* b = other_mat->data;
double* c = new_mat->data;
#pragma omp parallel shared(a,b,c) private(i,j,k)
{
#pragma omp for schedule(static)
for(i=0; i<this->n_rows; ++i){
for(j=0; j<other_mat->n_cols; ++j){
double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for(k=0; k<other_mat->n_rows; ++k){
sum += a[i*(this->n_cols)+k] * b[k*(other_mat->n_cols)+j];
}
c[i*(new_mat->n_cols)+j] = sum;
}
}
}
return new_mat;
}
int get_length(){
return this->n_rows * this->n_cols;
}
void print(){
for(int i=0; i<this->get_length(); ++i){
cout << this->data[i] << " ";
}
cout << endl;
}
Mat* operator-(Mat *other_mat){
if(this->n_cols != other_mat->n_cols || this->n_rows != other_mat->n_rows){
throw invalid_argument("\nMatrices dimensions do NOT match for - operation\n");
}
Mat *new_mat = new Mat(this->n_rows,this->n_cols);
int length = new_mat->get_length();
#pragma omp parallel for
for(int i=0; i<length; ++i){
new_mat->data[i] = this->data[i] - other_mat->data[i];
}
return new_mat;
}
void add(Mat *other_mat){
if(this->n_cols != other_mat->n_cols || this->n_rows != other_mat->n_rows){
throw invalid_argument("\nMatrices dimensions do NOT match for += operation\n");
}
int length = this->get_length();
#pragma omp parallel for
for(int i=0; i<length; ++i){
this->data[i] = this->data[i] + other_mat->data[i];
}
}
Mat* operator*(Mat *other_mat){
if(this->n_cols != other_mat->n_cols || this->n_rows != other_mat->n_rows){
throw invalid_argument("\nMatrices dimensions do NOT match for * operation\n");
}
Mat *new_mat = new Mat(this->n_rows,this->n_cols);
int length = new_mat->get_length();
#pragma omp parallel for
for(int i=0; i<length; ++i){
new_mat->data[i] = this->data[i] * other_mat->data[i];
}
return new_mat;
}
Mat* transpose(){
Mat *new_mat = new Mat(this->n_cols,this->n_rows);
int length = new_mat->get_length();
#pragma omp parallel for collapse(2)
for(int i=0; i<this->n_rows; ++i){
for(int j=0; j<this->n_cols; ++j){
new_mat->data[j*this->n_rows + i] = this->data[i*this->n_cols + j];
}
}
return new_mat;
}
};


class Dataset{
public:
vector<Mat*> *x;
vector<Mat*> *y;
Dataset(){
x = new vector<Mat*>();
y = new vector<Mat*>();
}
void load_x(string path, int n_rows, int n_cols){
ifstream infile(path);
string line;
while(getline(infile,line)){
Mat *mat = new Mat(1,n_cols);
stringstream stream(line);
for(int i=0;i<n_cols;++i){
stream >> mat->data[i];
if (stream.peek() == ',')stream.ignore();
}
this->x->push_back(mat);
}
}
void load_y(string path, int n_rows, int n_cols){
ifstream infile(path);
string line;
while(getline(infile,line)){
stringstream stream(line);
Mat *mat = new Mat(1,1);
double val;
stream >> val;
mat->data[0] = val;
this->y->push_back(mat);
}
}
};


class NeuralNetwork{
private:
int n_input;
int n_output;
int n_hidden;
Mat *syn0;
Mat *syn1;
Dataset *data_train;
Dataset *data_test;

void sigmoid(Mat *mat){
int length = mat->get_length();
for(int i=0; i<length; ++i){
mat->data[i] = 1.0/(1.0+exp(-1*mat->data[i]));
}
}
Mat* sigmoid_derivative(Mat *mat){
Mat *new_mat = new Mat(mat->n_rows, mat->n_cols);
int length = mat->get_length();
#pragma omp parallel for
for(int i=0; i<length; ++i){
new_mat->data[i] = (mat->data[i])*(1-mat->data[i]);
}
return new_mat;
}
void update_weights(Mat *syn,Mat *layer,Mat *delta){
Mat *layer_T = layer->transpose();
Mat *updates = layer_T->matmult(delta);
int length = updates->get_length();
syn->add(updates);
}
public:
NeuralNetwork(int _n_input, int _n_hidden, int _n_output){
n_input = _n_input;
n_hidden = _n_hidden;
n_output = _n_output;
syn0 = new Mat(n_input,n_hidden,true);
syn1 = new Mat(n_hidden,n_output,true);
}
void train(string path_x, string path_y, int n_epoch, int n_sample){
data_train = new Dataset();
data_train->load_x("dataset/data_train_x.csv", n_sample, n_input);
data_train->load_y("dataset/data_train_y.csv", n_sample, 1);
cout << "Succesfully loaded train dataset. \nTraining ... \n";

Timer *train_timer = new Timer();
while(n_epoch--){
for(int i=0; i<data_train->x->size(); ++i){
Mat *layer0 = data_train->x->at(i);
Mat *layer1 = layer0->matmult(syn0);
this->sigmoid(layer1);
Mat *layer2 = layer1->matmult(syn1);
this->sigmoid(layer2);
Mat *y = data_train->y->at(i);
Mat *layer2_error = y->operator-(layer2);
Mat *layer2_delta = layer2_error->operator*(this->sigmoid_derivative(layer2));

Mat *layer1_error = layer2_delta->matmult(syn1->transpose());
Mat *layer1_delta = layer1_error->operator*(this->sigmoid_derivative(layer1));

this->update_weights(syn1,layer1, layer2_delta);
this->update_weights(syn0,layer0, layer1_delta);
}
}
train_timer->stop(" to TRAIN");
}
void test(string path_x, string path_y, int n_sample){
data_test = new Dataset();
data_test->load_x(path_x, n_sample, n_input);
data_test->load_y(path_y, n_sample, n_output);
cout << "Succesfully loaded train dataset. \nTesting ... \n";

int error_counter = 0;
Timer *test_timer = new Timer();
#pragma omp parallel for reduction(+:error_counter)
for(int i=0; i<data_test->x->size(); ++i){
Mat *layer0 = data_test->x->at(i);
Mat *layer1 = layer0->matmult(syn0);
this->sigmoid(layer1);
Mat *layer2 = layer1->matmult(syn1);
this->sigmoid(layer2);
Mat *y = data_test->y->at(i);
Mat *layer2_error = y->operator-(layer2);
int err = nearbyint(y->data[0]) - nearbyint(layer2->data[0]);
if(err != 0) ++error_counter;
}
test_timer->stop(" to TEST");
cout << "Number of wrong predictions: " << error_counter << " out of " << n_sample << " samples." <<endl;
int accuracy = 100 - (error_counter * 100 / 634);
cout << "Accuracy: " << accuracy << "\%\n";
}
};

int main(int argc, char** argv){
int n_train = 2534;
int n_test = 634;
int n_input = 20;
int n_input_aug = 100;
int n_output = 1;
int n_hidden_neurons = 100;

string train_x = "dataset/data_train_x.csv";
string train_x_aug = "dataset/data_train_x_aug.csv";
string train_y = "dataset/data_train_y.csv";
string test_x = "dataset/data_test_x.csv";
string test_x_aug = "dataset/data_test_x_aug.csv";
string test_y = "dataset/data_test_y.csv";

NeuralNetwork *NN = new NeuralNetwork(n_input_aug,n_hidden_neurons,n_output);
NN->train(train_x_aug,train_y,4,n_train);
NN->test(test_x_aug,test_y,n_test);

}
