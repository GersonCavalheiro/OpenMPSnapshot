#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include "../model/model.hpp"
using namespace std;

vector<string> split(const string& str, const string& delim){
vector<string> tokens;
size_t prev = 0, pos = 0;
do
{
pos = str.find(delim, prev);
if (pos == string::npos) pos = str.length();
string token = str.substr(prev, pos-prev);
if (!token.empty()) tokens.push_back(token);
prev = pos + delim.length();
}
while (pos < str.length() && prev < str.length());
return tokens;
}

int main(){

string filename;
string encode_wanted;

cout << "Enter train data as a  .CSV file: ";
cin >> filename;
filename="../train_images.csv";
ifstream file_train_features;

file_train_features.open(filename);

vector<vector<double> > dataframe;

if(file_train_features.fail()){
cout << "Program can't find your file that you have entered please enter a valid path that exists" << endl;
}
else{
cout << "File is opened successfully!" << endl;
while(!file_train_features.eof()){
string line;
getline(file_train_features,line);
vector<string> s = split(line,",");
vector<double> s_(s.size(),0);

#pragma omp parallel for 
for(int i=0;i<s.size();i++){
s_[i]=stod(s[i]);
}
dataframe.push_back(s_);
}
file_train_features.close();
}

cout << "Enter train labels as a  .CSV file: ";
cin >> filename;
filename = "../train_labels.csv";
ifstream file_train_labels;

file_train_labels.open(filename);

vector<double> labels;

if(file_train_labels.fail()){
cout << "Program can't find your file that you have entered please enter a valid path that exists" << endl;
}
else{
cout << "File is opened successfully!" << endl;
while(!file_train_labels.eof()){
string line;
getline(file_train_labels,line);
labels.push_back(stod(line));
}
file_train_labels.close();
}


kNN model;
model.train(dataframe,labels);

cout << "Enter test data as a  .CSV file: ";
cin >> filename;

filename="../test_images.csv";

ifstream file_test_features;

file_test_features.open(filename);

vector<vector<double> > test_dataframe;

if(file_test_features.fail()){
cout << "Program can't find your file that you have entered please enter a valid path that exists" << endl;
}
else{
cout << "File is opened successfully!" << endl;
while(!file_test_features.eof()){
string line;
getline(file_test_features,line);
vector<string> s = split(line,",");
vector<double> s_(s.size(),0);
#pragma omp parallel for
for(int i=0;i<s.size();i++){
s_[i] = stold(s[i]); 
}
test_dataframe.push_back(s_);
}
file_test_features.close();
}

cout << "Enter test labels as a  .CSV file: ";
cin >> filename;

filename="../test_labels.csv";
ifstream file_test_labels;

file_test_labels.open(filename);

vector<double> test_labels;

if(file_test_labels.fail()){
cout << "Program can't find your file that you have entered please enter a valid path that exists" << endl;
}
else{
cout << "File is opened successfully!" << endl;
while(!file_test_labels.eof()){
string line;
getline(file_test_labels,line);
test_labels.push_back(stod(line));
}
file_test_labels.close();
}

int k=5; 

cout << "Choose the k value:";
cin >> k;
double res;
int match = 0,i;
double start,end;
start = omp_get_wtime();
#pragma omp parallel for reduction(+:match)
for(i = 0; i<test_dataframe.size()-1;i++){
res = model.predict(test_dataframe[i],k);
if(test_labels[i]==res)
match++;
}
end = omp_get_wtime();
cout << "Prediction for " << test_dataframe.size()-1 << " took " << end - start << " seconds." << endl;
cout << "Accuracy Score: ";
cout << match << "/" << test_dataframe.size()-1 <<endl;;

}
