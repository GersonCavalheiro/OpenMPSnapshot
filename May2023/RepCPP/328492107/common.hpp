


#pragma once

#include "cmath"                            
#include <array>                            
#include <cerrno>                           
#include <random>                           
#include <vector>                           
#include <limits>                           
#include <cstdio>                           
#include <cstring>                          
#include <iomanip>                          
#include <fstream>                          
#include <cassert>                          
#include <cstdlib>                          
#include <iostream>                         
#include <stdexcept>                        
#include <algorithm>                        
#include <inttypes.h>                       

#include <omp.h>                            

#define array_sizeof(type) ((char *)(&type+1)-(char*)(&type))
typedef intptr_t ssize_t;                   

constexpr int EPOCHS = 10;                  
constexpr int N_THREADS = 12;               
constexpr int N_ACTIVATIONS = 2;            
constexpr int CLI_WINDOW_WIDTH = 50;        
constexpr int MNIST_CLASSES = 10;           
constexpr double LEARNING_RATE = 0.1;       
constexpr double MNIST_TRAIN = 60000.0;     
constexpr double MNIST_TEST = 10000.0;      
constexpr double EXP = 2.718282;            
constexpr double MNIST_MAX_VAL = 255.0;     
constexpr char TRAINING_DATA_FILEPATH[] = "./data/fashion-mnist_train.csv";
constexpr char EVALUATION_DATA_FILEPATH[] = "./data/fashion-mnist_test.csv";
