#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#pragma once

using namespace std;

struct distance_with_index{
int index;
double distance;
};

class kNN{
public:
vector<vector<double> >  train_d; 

vector<double> train_l;

kNN();

~kNN();

void train(vector<vector<double> > &data,vector<double> &labels);

double predict(vector<double>&input,int k);

private:

double vote(vector<distance_with_index> closest_k);

vector<distance_with_index> closest_k(vector<double> unclassified,int k);

double distance(char option,vector<double> p1, vector<double> p2);
};


kNN::kNN(){
}
kNN::~kNN(){
}

double kNN::vote(vector<distance_with_index> closest_k){
map<double, int> votes;
for(int i=0;i<closest_k.size();i++){
if(votes.count(this->train_l[closest_k[i].index]) == 0){
votes[this->train_l[closest_k[i].index]]=1;
}else{
votes[this->train_l[closest_k[i].index]]+=1;
}	
}
double res_label = 0;
int count = 0;
for ( const auto &[key, value]: votes ) {
if(count<value){
res_label = key;
count = value;
}
}
return res_label;
}

double kNN::distance(char option,vector<double> p1, vector<double> p2){ 
double distance = 0;
if(option=='e'){
for(int i=0; i<p1.size();i++){
distance+=pow((p1[i]-p2[i]),2);
}
}else{
for(int i=0; i<p1.size();i++){
distance+=abs(p1[i]-p2[i]);
}	
}
return option=='e' ? sqrt(distance) : distance;
}

bool compareByDistance(const distance_with_index &a, const distance_with_index &b){ 
return a.distance < b.distance;
}

vector<distance_with_index> kNN::closest_k(vector<double> unclassified,int k){
vector<distance_with_index> result;
for(int i=0;i<this->train_d.size();i++){
distance_with_index temp;
temp.distance = distance('e',this->train_d[i],unclassified);
temp.index = i;
result.push_back(temp);
}
sort(result.begin(), result.end(),compareByDistance);
vector<distance_with_index> returner;

for(int i = 0;i<k;i++){
returner.push_back(result[i]);
}

return returner;
}

void kNN::train(vector<vector<double> > &data,vector<double> &labels){
this->train_d = data;
this->train_l = labels;
}

double kNN::predict(vector<double>&input,int k){
return vote(closest_k(input,k));
}