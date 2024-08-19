#pragma once

#include "Matrix.h"
#include "Functions.h"
#include "Optimizers.h"
#include <algorithm>
#include <initializer_list>

enum Activation { LOGIT, RELU, SWISH, SIGMOID, TANH, SOFTMAX };


template<typename T>
class FCLayer{
public:
FCLayer(int n_batch, int n_nodes, int n_prev_nodes, T value = 0, int num_threads = 1);

void dropout(float drop_rate);

inline const Matrix<T>& getWeights() const;

void optimizer(OptimizerName op, std::initializer_list<double> args);
void reallocBatch(int batchSize);

const Matrix<T>& logit(const Matrix<T>& prev_layer);
const Matrix<T>& activate(Activation activation);
const Matrix<T>& delta(const Matrix<T>& delta_next, const Matrix<T>& weights_next);
void gradients(const Matrix<T>& prev_layer);
void weights_update();

private:
int _batch;
int _nodes;
int _w_rows, _w_cols;   
int _n_prev_nodes;      
float _dropout = 0;     
Activation _activation = LOGIT;
int _n_threads;

Matrix<T> _layer;          
Matrix<T> _logit;          
Matrix<T> _weights;        
Matrix<T> _biases;         

Matrix<T> _delta;          
Matrix<T> _weights_grad;   
Matrix<T> _biases_grad;    
std::unique_ptr<Optimizer<T>> _optimizer_w;
std::unique_ptr<Optimizer<T>> _optimizer_b;
};


template<typename T>
FCLayer<T>::FCLayer(int n_batch, int n_nodes, int n_prev_nodes, T value, int num_threads) : 
_batch{ n_batch }, _nodes{ n_nodes },
_w_rows{ n_prev_nodes }, _w_cols{ n_nodes },
_n_prev_nodes{ n_prev_nodes },
_n_threads{ num_threads },
_layer{ Matrix<T>(n_batch, n_nodes, 0, num_threads) },
_logit{ Matrix<T>(n_batch, n_nodes, 0, num_threads) },
_weights{ Matrix<T>(n_prev_nodes, n_nodes, XAVIER, n_prev_nodes, 0, num_threads) },
_biases{ Matrix<T>(1, n_nodes, XAVIER, n_prev_nodes, 0, num_threads) },
_delta{ Matrix<T>(_batch, _nodes, 0, _n_threads) },
_weights_grad{ Matrix<T>(_w_rows, _w_cols, 0, _n_threads) },
_biases_grad{ Matrix<T>(1, _w_cols, 0, _n_threads) }{
}



template<typename T>
const Matrix<T>& FCLayer<T>::getWeights() const {
return _weights;
}


template<typename T>
void FCLayer<T>::optimizer(OptimizerName op, std::initializer_list<double> args){
int n_args = args.size();
std::initializer_list<double>::iterator it = args.begin();
switch(op){
case sgd:{
assert(n_args == 1);
_optimizer_w = std::make_unique<SGD<T>>(*it);
_optimizer_b = std::make_unique<SGD<T>>(*it);
break;
}
case momentum:{
assert(n_args == 2);
_optimizer_w = std::make_unique<Momentum<T>>(*it, *(it+1), _w_rows, _w_cols);
_optimizer_b = std::make_unique<Momentum<T>>(*it, *(it+1), 1, _w_cols);
break;
}
case nag:{
assert(n_args == 2);
_optimizer_w = std::make_unique<NAG<T>>(*it, *(it+1), _w_rows, _w_cols);
_optimizer_b = std::make_unique<NAG<T>>(*it, *(it+1), 1, _w_cols);
break;
}
case adagrad:{
assert(n_args == 1);
_optimizer_w = std::make_unique<Adagrad<T>>(*it, _w_rows, _w_cols);
_optimizer_b = std::make_unique<Adagrad<T>>(*it, 1, _w_cols);
break;
}
case rmsprop:{
assert(n_args == 2);
_optimizer_w = std::make_unique<RMSProp<T>>(*it, *(it+1), _w_rows, _w_cols);
_optimizer_b = std::make_unique<RMSProp<T>>(*it, *(it+1), 1, _w_cols);
break;
}
case adam:{
assert(n_args == 3);
_optimizer_w = std::make_unique<Adam<T>>(*it, *(it+1), *(it+2), _w_rows, _w_cols);
_optimizer_b = std::make_unique<Adam<T>>(*it, *(it+1), *(it+2), 1, _w_cols);
break;
}
default:{
std::cout << "Default op used (SGD), lr = 0.001" << std::endl;
_optimizer_w = std::make_unique<SGD<T>>(0.001);
_optimizer_b = std::make_unique<SGD<T>>(0.001);
break;
}
};
}

template<typename T>
void FCLayer<T>::reallocBatch(int batch_size){
if(batch_size == _batch) return;
_logit = Matrix<T>(batch_size, _nodes, 0, _n_threads);
_layer = Matrix<T>(batch_size, _nodes, 0, _n_threads);
_batch = batch_size;
}

template<typename T>
const Matrix<T>& FCLayer<T>::logit(const Matrix<T>& prev_layer) {
_logit.dot(prev_layer, _weights);
_logit.vBroadcast(_biases, SUM);
return _logit;
}

template<typename T>
const Matrix<T>& FCLayer<T>::activate(Activation activation){
_activation = activation;
switch(activation){
case LOGIT:{
break;
}
case RELU:{
func2D::relu(_layer, _logit);
break;
}
case SWISH:{
func2D::swish(_layer, _logit);
break;
}
case SIGMOID:{
func2D::sigmoid(_layer, _logit);
break;
}
case TANH:{
func2D::tanh(_layer, _logit);
break;
}
default:{
break;
}
};
return _layer;
}

template<typename T>
const Matrix<T>& FCLayer<T>::delta(const Matrix<T>& delta_next, const Matrix<T>& weights_next){
_delta.dotTranspose(delta_next, weights_next);

switch(_activation){
case LOGIT:{
break;
}
case RELU:{
auto it_src = _logit.begin();
auto it_dest = _delta.begin();
for(; it_src != _logit.end(); ++it_src, ++it_dest){
if(*it_src<=0) *it_dest = 0;
}
break;
}
case SWISH:{
Matrix<T> swish_der(_logit.getRows(), _logit.getCols(), 0);
deriv2D::swish(swish_der, _logit, _layer);
_delta *= swish_der;
break;
}
case SIGMOID:{
Matrix<T> sig_der(_batch, _nodes);
deriv2D::sigmoid(sig_der, _layer);
_delta *= sig_der;
break;
}
case TANH:{
_delta *= static_cast<T>(1) - _layer * _layer;
break;
}
default:{
break;
}
};
return _delta;
}

template<typename T>
void FCLayer<T>::gradients(const Matrix<T>& prev_layer){
const Matrix<T> prev_layer_T = prev_layer.transpose();
_weights_grad.dot(prev_layer_T, _delta);
_biases_grad = _delta.vSum();
}

template<typename T>
void FCLayer<T>::weights_update(){
(*_optimizer_w)(_weights, _weights_grad);
(*_optimizer_b)(_biases, _biases_grad);
}
