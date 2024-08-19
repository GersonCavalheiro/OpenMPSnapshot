#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <omp.h>

#define NUM_THREADS 8

void sortrows(std::vector<std::pair<std::vector<float>, int>>& matrix, int col) {    
std::stable_sort(matrix.begin(),
matrix.end(),
[col](const std::pair<std::vector<float>, int>& lhs, const std::pair<std::vector<float>, int>& rhs) {
return lhs.first[col] > rhs.first[col];
});
}

std::vector<std::pair<std::vector<float>, int>> get_copy_with_samples(std::vector<std::vector<float>> &x, std::vector<int> &y, std::vector<int> &samples) {
std::vector<std::pair<std::vector<float>, int>> data;

for (int sample : samples) {
data.push_back(std::make_pair(x[sample], y[sample]));
}

return data;
}

class Data {
public:
Data();
void loadTrainSet(std::string dataFilename, std::string labelsFilename);
void loadTestSet(std::string dataFilename, std::string labelsFilename);
std::vector<float> getColumn(int column, std::vector<std::vector<float>> &data);
std::vector<float> getSample(int row, std::vector<std::vector<float>> &data);
float getElement(int row, int column, std::vector<std::vector<float>> &data);
std::vector<std::vector<float>> trainData;
std::vector<std::vector<float>> testData;
std::vector<int> trainLabels;
std::vector<int> testLabels;

private:
void loadData(std::string filename, std::vector<std::vector<float>> &data);
void loadLabels(std::string filename, std::vector<int> &labels);
};

Data::Data() {
}

void Data::loadData(std::string filename, std::vector<std::vector<float>> &data) {
std::ifstream infile(filename);

if (!infile.is_open()) {
std::cout << "Unable to open file" << std::endl;
return;
}

std::string line = "";
data.clear();

while(getline(infile, line)) {
std::stringstream str_stream(line);
std::vector<float> dataRow;

while(str_stream.good()) {
std::string temp;
getline(str_stream, temp, ',');
dataRow.push_back(stof(temp));
}

data.push_back(dataRow);
}
}

void Data::loadLabels(std::string filename, std::vector<int> &labels) {
std::ifstream infile(filename);

if (!infile.is_open()) {
std::cout << "Unable to open file" << std::endl;
return;
}

std::string line = "";
labels.clear();

while(getline(infile, line)) {
std::stringstream str_stream(line);

while(str_stream.good()) {
std::string temp;
getline(str_stream, temp, ',');
labels.push_back(stoi(temp));
}
}
}

void Data::loadTrainSet(std::string dataFilename, std::string labelsFilename) {
loadData(dataFilename, trainData);
loadLabels(labelsFilename, trainLabels);
}

void Data::loadTestSet(std::string dataFilename, std::string labelsFilename) {
loadData(dataFilename, testData);
loadLabels(labelsFilename, testLabels);
}

float Data::getElement(int row, int column, std::vector<std::vector<float>> &data) {
if (row < 0 || row >= data.size()) {
std::cout << "Error!" << std::endl;
return -1;
}
if (column < 0 || column >= data[0].size()) {
std::cout << "Error!" << std::endl;
return -1;
}
return data[row][column];
}

std::vector<float> Data::getColumn(int column, std::vector<std::vector<float>> &data) {
std::vector<float> result;
if (column < 0 || column >= data[0].size()) {
std::cout << "Error!" << std::endl;
return result;
}
for (int row = 0; row < data.size(); ++row) {
result.push_back(data[row][column]);
}

return result;
}

std::vector<float> Data::getSample(int row, std::vector<std::vector<float>> &data) {
if (row < 0 || row >= data.size()) {
std::cout << "Error!" << std::endl;
std::vector<float> result;
return result;
}
return data[row];
}

class DecisionTree {
public:
DecisionTree(std::vector<std::pair<std::vector<float>, int>> &data1,
int num_all_features1, int num_features1, int num_classes,
int max_depth1, int depth1, int min_split);
std::vector<int> predict(std::vector<std::vector<float>> x);

private:
std::vector<std::pair<std::vector<float>, int>> data;
std::vector<int> features;
int num_of_classes;
int max_depth;
int depth;
int min_samples_split;
bool is_leaf;
int c;
int split_feature;
float split_point;
float old_gini_impurity;
DecisionTree *left;
DecisionTree *right;
int num_all_features;
int num_features;

void fit();
void findBestSplit(float &best_gini);
void findBestSplitForFeature(int feature, float &feature_split_point, float &feature_best_gini);
float giniGain(int feature, float split_val);
float giniImpurity();
int predictSample(std::vector<float> sample);
};

DecisionTree::DecisionTree(std::vector<std::pair<std::vector<float>, int>> &data1,
int num_all_features1, int num_features1, int num_classes,
int max_depth1 = std::numeric_limits<int>::max(), int depth1 = 0,
int min_split = 2) {

data = data1;
num_features = num_features1;
num_all_features = num_all_features1;
for (int i = 0; i < num_features; ++i)
features.push_back(rand() % num_all_features);
num_of_classes = num_classes;
max_depth = max_depth1;
depth = depth1;
min_samples_split = min_split;
is_leaf = false;
left = nullptr;
right = nullptr;
old_gini_impurity = giniImpurity();
fit();
}

void DecisionTree::fit() {
if (depth == max_depth - 1) 
is_leaf = true;

float best_gini = -1;
findBestSplit(best_gini);

if (best_gini < 0) { 
is_leaf = true;
}

std::vector<std::pair<std::vector<float>, int>> data_left;
std::vector<std::pair<std::vector<float>, int>> data_right;

if (!is_leaf) {
for (int sample = 0; sample < data.size(); ++sample) {
if (data[sample].first[split_feature] < split_point)
data_left.push_back(data[sample]);
else
data_right.push_back(data[sample]);
}
}

if (data_left.size() < min_samples_split || data_right.size() < min_samples_split)
is_leaf = true;

if (!is_leaf) {
left = new DecisionTree(data_left, num_all_features, num_features, num_of_classes, max_depth, depth + 1, min_samples_split);
right = new DecisionTree(data_right, num_all_features, num_features, num_of_classes, max_depth, depth + 1, min_samples_split);
} else {
std::vector<int> class_count(num_of_classes, 0);

for (int sample = 0; sample < data.size(); ++sample) 
++class_count[data[sample].second];

c = std::distance(class_count.begin(), 
std::max_element(class_count.begin(),
class_count.end()));
}
}

void DecisionTree::findBestSplit(float &best_gini) {
for (int feature : features) {
float feature_split_point, feature_best_gini = 0;
findBestSplitForFeature(feature, feature_split_point, feature_best_gini);

if (feature_best_gini > best_gini) {
best_gini = feature_best_gini;
split_point = feature_split_point;
split_feature = feature;
}
}
}

void DecisionTree::findBestSplitForFeature(int feature, float &feature_split_point, float &feature_best_gini)  {
std::vector<int> k_tiles;
sortrows(data, feature);
int num_tiles = std::sqrt(data.size());

for (int i = 0; i < std::min(num_tiles, (int) data.size()); ++i) {
k_tiles.push_back(data.size() * i / num_tiles);
}
k_tiles.push_back(data.size() - 1);

for (int sample : k_tiles) {
float gini = giniGain(feature, data[sample].first[feature]);

if (gini > feature_best_gini) {
feature_best_gini = gini;
feature_split_point = data[sample].first[feature];
}
}
}

float DecisionTree::giniGain(int feature, float split_val) {
std::vector<int> class_count_left(num_of_classes, 0);
std::vector<int> class_count_right(num_of_classes, 0);

int count_left = 0, count_right = 0;
for (int sample = 0; sample < data.size(); ++sample) {
if (data[sample].first[feature] < split_val) {
++class_count_left[data[sample].second];
++count_left;
} else {
++class_count_right[data[sample].second];
++count_right;
}
}

float gini_impurity_left = 1;
float gini_impurity_right = 1;

for (int i = 0; i < class_count_left.size(); ++i) {
float prob_left = (float) class_count_left[i] / count_left;
float prob_right = (float) class_count_right[i] / count_right;

gini_impurity_left -= prob_left * prob_left;
gini_impurity_right -= prob_right * prob_right;
}

float gini_impurity = (gini_impurity_left * count_left + gini_impurity_right * count_right) / (count_left + count_right);
float gini_gain = old_gini_impurity - gini_impurity;

return gini_gain;
}

float DecisionTree::giniImpurity() {
std::vector<int> class_count(num_of_classes, 0);

for (int sample = 0; sample < data.size(); ++sample) 
++class_count[data[sample].second];

float gini_impurity = 1;

for (int count : class_count) {
float p = (float) count / data.size();
gini_impurity -= p * p;
}

return gini_impurity;
}

std::vector<int> DecisionTree::predict(std::vector<std::vector<float>> data) {
std::vector<int> predictions;

for (int sample_index = 0; sample_index < data.size(); ++sample_index) {
predictions.push_back(predictSample(data[sample_index]));
}
return predictions;
}

int DecisionTree::predictSample(std::vector<float> sample) {
if (is_leaf)
return c;

if (sample[split_feature] < split_point) 
return left->predictSample(sample);
else 
return right->predictSample(sample);
}

class RandomForest {
public:
RandomForest(std::vector<std::vector<float>> &data, std::vector<int> &labels,
int num_trees, std::string num_features,
int max_depth, int min_samples_split);

std::vector<int> predict(std::vector<std::vector<float>> x);

private:
std::vector<std::vector<float>> x;
std::vector<int> y;
int n_trees;
int n_features;
int max_depth;
int min_samples_split;
int num_of_classes;
std::vector<DecisionTree*> trees;

DecisionTree* createTree();
};

RandomForest::RandomForest(std::vector<std::vector<float>> &data, std::vector<int> &labels,
int num_trees, std::string num_features = "sqrt",
int depth = std::numeric_limits<int>::max(), int min_samples = 2) {
x = data;
y = labels;
n_trees = num_trees;
max_depth = depth;
min_samples_split = min_samples;
num_of_classes = 0;
std::unordered_set<int> seenClasses;

for (int sample : y) {
if (seenClasses.find(sample) == seenClasses.end()) {
++num_of_classes;
seenClasses.insert(sample);
}
}

if (num_features == "sqrt") {
n_features = (int) std::sqrt(x[0].size());
} else if (num_features == "log2") {
n_features = 3 * (int) std::log2(x[0].size() + 1);
} else {
n_features = (int) x[0].size();
}

for (int i = 0; i < num_trees; ++i) {
trees.push_back(nullptr);
}

#pragma omp parallel for default(none)\
firstprivate(num_trees)\
schedule(dynamic)\
num_threads(NUM_THREADS)
for (int i = 0; i < num_trees; ++i) 
{    
trees[i] = createTree();
}
}

DecisionTree* RandomForest::createTree() {

std::vector<int> samples;
for (int i = 0; i < x.size(); ++i) 
samples.push_back(std::rand() % x.size());

std::vector<std::pair<std::vector<float>, int>> data1 = get_copy_with_samples(x, y, samples);
DecisionTree *tree = new DecisionTree(data1, x[0].size(), n_features, num_of_classes, max_depth, 0, min_samples_split);
return tree;
}

std::vector<int> RandomForest::predict(std::vector<std::vector<float>> data) {
std::vector<int> predictions;
std::vector<std::vector<int>> tree_predictions;

for (int i = 0; i < n_trees; ++i) {
std::vector<int> temp;
tree_predictions.push_back(temp);
}

#pragma omp parallel for default(none)\
shared(data, tree_predictions) \
schedule(dynamic)\
num_threads(NUM_THREADS)
for (int i = 0; i < n_trees; ++i) 
{
tree_predictions[i] = trees[i]->predict(data);
}

int num_samples = data.size();

for (int i = 0; i < num_samples; ++i) {
predictions.push_back(0);
}

#pragma omp parallel for default(none)\
firstprivate(num_samples) \
shared(predictions, tree_predictions) \
schedule(static)\
num_threads(NUM_THREADS)
for (int i = 0; i < num_samples; ++i) 
{
std::vector<int> predictions_count(num_of_classes, 0);
for (int j = 0; j < n_trees; ++j) {
++predictions_count[tree_predictions[j][i]];
}
int pred = std::distance(predictions_count.begin(), 
std::max_element(predictions_count.begin(),
predictions_count.end()));
predictions[i] = pred;
}

return predictions;
}

float accuracy(std::vector<int> predicted, std::vector<int> labels) {
int correct = 0;

for (int i = 0; i < predicted.size(); ++i) {
if (predicted[i] == labels[i])
++correct;
}

return (float) correct / predicted.size();
}

int main() {
std::cout << "Execution started" << std::endl << std::endl;

std::cout << "Loading training data" << std::endl;
Data data = Data();
data.loadTrainSet("data/train_x_iris.csv", "data/train_y_iris.csv");
data.loadTestSet("data/test_x_iris.csv", "data/test_y_iris.csv");

std::cout << "Number of samples in training data: " <<  data.trainData.size() << std::endl \
<< "Number of features: " << data.trainData[0].size() << std::endl << std::endl;

std::cout << "Training started" << std::endl << std::endl;

double time_start_parallel = omp_get_wtime();
RandomForest rf = RandomForest(data.trainData, data.trainLabels, 10, "sqrt");
double time_end_parallel = omp_get_wtime();

std::cout << "Random Forest created" << std::endl;
std::cout << "Elapsed time: " << time_end_parallel - time_start_parallel << std::endl << std::endl;

double time_start_parallel_pred = omp_get_wtime();
std::vector<int> predictions = rf.predict(data.testData);
double time_end_parallel_pred = omp_get_wtime();

std::cout << "Accuracy: " << accuracy(predictions, data.testLabels) << std::endl;
std::cout << "Elapsed time: " << time_end_parallel_pred - time_start_parallel_pred << std::endl << std::endl;
return 0;
}
