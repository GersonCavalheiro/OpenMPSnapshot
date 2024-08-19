#pragma once

#include "headers/Matrix.h"
#include "ClosestCentroids.h"

template<typename T>
class KMeans{
public:
KMeans(const Matrix<T>& dataset, int n_clusters, bool stop_criterion=true, int n_threads=1);

Matrix<T> getCentroid();
Matrix<int> getDataToCentroid();
int getNIters();

void mapSampleToCentroid();
void updateCentroids();
void run(int max_iter, float threashold=-1);

void print();


private:
bool _stop_crit;
int _n_threads;
int _n_iters = 0;

int _dims;

int _samples;

int _n_clusters; 

std::unique_ptr<Matrix<T>> _centroids;

Matrix<T> _training_set;

std::unique_ptr<ClosestCentroids<T>> _dataset_to_centroids;
};

template<typename T>
KMeans<T>::KMeans(const Matrix<T>& dataset, int n_clusters, bool stop_criterion, int n_threads) : 
_training_set{ dataset },
_n_clusters{ n_clusters },
_stop_crit{ stop_criterion },
_n_threads{ n_threads } {

_training_set = dataset;
_dims = dataset.getRows();
_samples = dataset.getCols();
_training_set.setThreads(_n_threads);

Matrix<T> vMinValues = _training_set.vMin();
Matrix<T> vMaxValues = _training_set.vMax();

_centroids = std::make_unique<Matrix<T>>(_dims, n_clusters, UNIFORM, vMinValues, vMaxValues);
_centroids->setThreads(_n_threads);

_dataset_to_centroids = std::make_unique<ClosestCentroids<T>>(_samples, 0, stop_criterion, _n_threads);
}

template<typename T>
inline Matrix<T> KMeans<T>::getCentroid(){ return *_centroids; }

template<typename T>
inline Matrix<int> KMeans<T>::getDataToCentroid(){ return *static_cast<Matrix<int>* >(_dataset_to_centroids.get()); }

template<typename T>
inline int KMeans<T>::getNIters(){ return _n_iters; }

template<typename T>
void KMeans<T>::mapSampleToCentroid(){ _dataset_to_centroids->getClosest(_training_set, *_centroids); }

template<typename T>
void KMeans<T>::updateCentroids(){
int occurences[_n_clusters] = {0};
T sample_buff[_n_clusters*_dims] = {0};

for(int i = 0; i < _samples; ++i){
const int& k_index = (*_dataset_to_centroids)(i);
for(int d = 0; d < _dims; ++d){
sample_buff[k_index+d*_n_clusters] += _training_set(d, i);
}
++occurences[k_index];
}
for(int c = 0; c < _n_clusters; ++c){
if(!occurences[c]) continue;
for(int d = 0; d < _dims; ++d){
(*_centroids)(d, c) = sample_buff[c+d*_n_clusters] / occurences[c];
}
}
}

template<typename T>
void KMeans<T>::run(int max_iter, float threashold){

mapSampleToCentroid();
updateCentroids();
_n_iters = 1;
if(max_iter == 1) return;
int epoch = 1;
float modif_rate_prev = 0;
float modif_rate_curr;
float inertia;
do {
mapSampleToCentroid();
updateCentroids();
modif_rate_curr = _dataset_to_centroids->getModifRate();
inertia = modif_rate_curr - modif_rate_prev;
modif_rate_prev = modif_rate_curr;
++epoch;
} while(epoch < max_iter && modif_rate_curr >= threashold  && std::abs(inertia) >= 1e-2);
_n_iters = epoch;
}

template<typename T>
void KMeans<T>::print() {
for(int d = 0; d < _dims; ++d){
std::cout << "[";
std::cout << _centroids->row(d) << "]," << std::endl;
}
}
