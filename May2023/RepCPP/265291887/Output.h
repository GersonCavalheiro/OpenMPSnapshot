#pragma once

#include "FCLayer.h"

enum CostFunc { CE, bCE, MSE }; 

template<typename T>
class Output{
public:
Output(int n_batch, int n_nodes, int n_prev_nodes, T value = 0, int num_threads = 1);

void dropout(float drop_rate);
void clip(T inf=0.1, T sup=0.9);

float getCost(CostFunc costFunc);
float getAccuracy(const Matrix<T>& target, const std::string& accuracy="cathegorical");
inline const Matrix<T>& getWeights() const;
const Matrix<T> getThreasholdClip() const;


void optimizer(OptimizerName op, std::initializer_list<double> args);
void reallocBatch(int batchSize);

const Matrix<T>& logit(const Matrix<T>& prev_layer);
const Matrix<T>& activate(Activation activation);
const Matrix<T>& delta(CostFunc costFunc, const Matrix<T>& target);
void gradients(const Matrix<T>& prev_layer);
void weights_update();

const Matrix<T>& updateThreashold(const Matrix<T>& target);

private:
int _batch;
int _nodes;
int _w_rows, _w_cols;   
int _n_prev_nodes;      
float _dropout = 0;     
Activation _activation = LOGIT;
CostFunc _costFunc;
int _n_threads;

Matrix<T> _layer;          
Matrix<T> _logit;          
Matrix<T> _weights;        
Matrix<T> _biases;         

Matrix<T> _delta;          
Matrix<T> _weights_grad;   
Matrix<T> _biases_grad;    
Matrix<T> _target;         
std::unique_ptr<Optimizer<T>> _optimizer_w;
std::unique_ptr<Optimizer<T>> _optimizer_b;

Matrix<T> _threashold_buffer; 
};


template<typename T>
Output<T>::Output(int n_batch, int n_nodes, int n_prev_nodes, T value, int num_threads) : 
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
_biases_grad{ Matrix<T>(1, _w_cols, 0, _n_threads) },
_target{ Matrix<T>(_batch, _nodes, 0, _n_threads) },
_threashold_buffer{ Matrix<T>(_batch, _nodes, 0, _n_threads) }{
}

template<typename T>
void Output<T>::clip(T inf, T sup){
for(int i = 0; i < _batch; ++i){
for(int j = 0; j < _nodes; ++j){
if(_target(i, j) && _layer(i, j)<inf) _layer(i, j) = inf;
else if (!_target(i, j) && _layer(i, j)>sup) _layer(i, j) = sup;
}
}
}

template<typename T>
float Output<T>::getCost(CostFunc costFunc){
float cost{ 0 };
switch (costFunc) {
case CE:{
T type_min = std::numeric_limits<T>::min();
for(int i = 0; i < _batch; ++i){
for(int j = 0; j < _nodes; ++j){
const T& y_hat = _layer(i, j);
const T& y = _target(i, j);
cost -= y*log(std::max(type_min, y_hat));
}
}
cost /= (_batch * _nodes);
break;
}
case bCE:{
Matrix<T> relu_tmp(_logit.getRows(), _logit.getCols(), 0);
func2D::relu(relu_tmp, _logit);

Matrix<T> log_tmp(_logit.getRows(), _logit.getCols(), 0);
func2D::abs(log_tmp, _logit);
log_tmp *= (-1);
func2D::exp(log_tmp);
log_tmp += static_cast<T>(1);
func2D::log(log_tmp);

cost = (relu_tmp - _target * _logit + log_tmp).hSum().vSum()(0, 0);
cost /= (_batch * _nodes);
break;
}
case MSE:{
for(int i = 0; i < _batch; ++i){
for(int j = 0; j < _nodes; ++j){
cost += std::pow(_layer(i, j) - _target(i, j), 2);
}
}
cost /= (2 * _batch * _nodes);
break;
}
};
return cost;
}

template<typename T>
float Output<T>::getAccuracy(const Matrix<T>& target, const std::string& accuracy){
Matrix<int> outputMaxIdx = _layer.vMaxIndex();
Matrix<int> targetMaxIdx = target.vMaxIndex();
Matrix<int> comparison = outputMaxIdx.compare(targetMaxIdx);
return static_cast<float>(comparison.vSum()(0, 0)) / comparison.getRows();
}

template<typename T>
const Matrix<T>& Output<T>::getWeights() const {
return _weights;
}

template<typename T>
const Matrix<T> Output<T>::getThreasholdClip() const {
Matrix<T> h_vect = _threashold_buffer.vSum();
T sum = h_vect.hSum()(0, 0);
h_vect /= sum;
return h_vect;
}


template<typename T>
void Output<T>::reallocBatch(int batch_size){
if(batch_size == _batch) return;
_logit = Matrix<T>(batch_size, _nodes, 0, _n_threads);
_layer = Matrix<T>(batch_size, _nodes, 0, _n_threads);
_threashold_buffer = Matrix<T>(batch_size, _nodes, 0, _n_threads);
_batch = batch_size;
}

template<typename T>
void Output<T>::optimizer(OptimizerName op, std::initializer_list<double> args){
int n_args = args.size();
std::initializer_list<double>::iterator it = args.begin();
switch (op) {
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
const Matrix<T>& Output<T>::logit(const Matrix<T>& prev_layer) {
_logit.dot(prev_layer, _weights);
_logit.vBroadcast(_biases, SUM);
return _logit;
}

template<typename T>
const Matrix<T>& Output<T>::activate(Activation activation){
_activation = activation;
switch (activation) {
case LOGIT:{
break;
}
case RELU:{
func2D::relu(_layer, _logit);
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
case SOFTMAX:{
Matrix<T> batchMax = _logit.vMax();     
_logit.hBroadcast(batchMax, SUB);       
func2D::exp(_layer, _logit);            
Matrix<T> summed = _layer.hSum();       
_layer.hBroadcast(summed, DIV);         
break;
}
default:{
break;
}
};
return _layer;
}

template<typename T>
const Matrix<T>& Output<T>::delta(CostFunc costFunc, const Matrix<T>& target){
_costFunc = costFunc;
_target.copy(target);
switch (costFunc) {
case CE:{
_delta = _layer - target;
break;
}
case bCE:{
_delta = _layer - target;
break;
}
case MSE:{
_delta = _layer - target;
switch (_activation) {
case LOGIT:{
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
default:{ break; }
};
break;
}
default:{ break; }
};
return _delta;
}

template<typename T>
void Output<T>::gradients(const Matrix<T>& prev_layer){
const Matrix<T> prev_layer_T = prev_layer.transpose();
_weights_grad.dot(prev_layer_T, _delta);
_biases_grad = _delta.vSum();
}

template<typename T>
void Output<T>::weights_update(){
(*_optimizer_w)(_weights, _weights_grad);
(*_optimizer_b)(_biases, _biases_grad);
}

template<typename T>
const Matrix<T>& Output<T>::updateThreashold(const Matrix<T>& target){
_threashold_buffer += target;
return _threashold_buffer;
}