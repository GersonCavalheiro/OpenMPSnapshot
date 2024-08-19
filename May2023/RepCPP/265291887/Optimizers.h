#pragma once

enum OptimizerName { sgd, momentum, nag, adagrad, rmsprop, adam };

template<typename T>
class Optimizer{
public:
Optimizer(T lr) : _lr{ lr }{}
virtual ~Optimizer(){}

virtual void operator()(Matrix<T>& weights, const Matrix<T>& gradients) = 0;
protected:
T _lr;
};

template<typename T>
class SGD : public Optimizer<T>{
public:
SGD(T lr) : Optimizer<T>(lr) {}

void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
weights -= gradients * this->_lr;
}
};

template<typename T>
class Momentum : public Optimizer<T>{
public:
Momentum(T lr, T damp, int rows, int cols) : 
Optimizer<T>(lr),
_damp{damp},
_velocity{ Matrix<T>(rows, cols, 0) }{}

void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
_velocity = _velocity * _damp + gradients * this->_lr;
weights -= _velocity;
}
private:
T _damp;
Matrix<T> _velocity; 
};

template<typename T>
class NAG : public Optimizer<T>{
public:
NAG(T lr, T damp, int rows, int cols) : 
Optimizer<T>(lr),
_damp{damp},
_velocity{ Matrix<T>(rows, cols, 0) }{}

void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
weights -= _velocity; 
_velocity = _velocity * _damp + gradients * this->_lr;
}
private:
T _damp;
Matrix<T> _velocity; 
};

template<typename T>
class Adagrad : public Optimizer<T>{
public:
Adagrad(T lr, int rows, int cols) : 
Optimizer<T>(lr),
_running_sum{ Matrix<T>(rows, cols, 0) }{}

void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
_running_sum = _running_sum + (gradients*gradients);
Matrix<T> sqrt_tmp(_running_sum.getRows(), _running_sum.getCols(), 0);
func2D::sqrt(sqrt_tmp, _running_sum);
weights -= gradients / sqrt_tmp * this->_lr;
}
protected:
Matrix<T> _running_sum; 
};

template<typename T>
class RMSProp : public Optimizer<T>{
public:
RMSProp(T lr, T decay, int rows, int cols) : 
Optimizer<T>(lr), _decay{ decay },
_running_sum{ Matrix<T>(rows, cols, 0) }{}

void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
_running_sum = _running_sum * _decay;
T coeff = static_cast<T>(1) * _decay;
_running_sum += (gradients*gradients) * coeff;
Matrix<T> sqrt_tmp(_running_sum.getRows(), _running_sum.getCols(), 0);
func2D::sqrt(sqrt_tmp, _running_sum);
weights -=  gradients / sqrt_tmp * this->_lr;
}
protected:
T _decay;
Matrix<T> _running_sum; 
};

template<typename T>
class Adam : public Optimizer<T>{
public:
Adam(T lr, T beta1, T beta2, int rows, int cols) : 
Optimizer<T>(lr),
_beta1{ beta1 }, _beta2{ beta2 },
_beta1_t{ beta1 }, _beta2_t{ beta2 },
_moment1{ Matrix<T>(rows, cols, 0) },
_moment2{ Matrix<T>(rows, cols, 0) }{}

void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
_moment1 = _moment1 * _beta1 + gradients * (1-_beta1);
_moment2 = _moment2 * _beta2 + gradients * gradients * (1-_beta2);
Matrix<T> sqrt_tmp(_moment2.getRows(), _moment2.getCols(), 0);
func2D::sqrt(sqrt_tmp, _moment2);
T alpha_t = this->_lr * std::sqrt(1-_beta2_t) / (1-_beta1_t);
weights -= _moment1 / sqrt_tmp * alpha_t;

_beta1_t *= _beta1;
_beta2_t *= _beta2;
}
private:
T _beta1, _beta2;
T _beta1_t, _beta2_t;
Matrix<T> _moment1;
Matrix<T> _moment2;
};
