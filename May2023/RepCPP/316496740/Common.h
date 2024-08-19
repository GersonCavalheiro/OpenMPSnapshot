

#pragma once

#include <iostream>                                   
#include <vector>                                     
#include <iomanip>                                    
#include <limits>                                     
#include <chrono>                                     
#include <array>                                      
#include <algorithm>                                  
#include <utility>                                    
#include <string>                                     
#include <cstdlib>                                    
#include <random>                                     
#include <fstream>                                    
#include <omp.h>                                      

constexpr int TEST_MODE = 0;                          
constexpr int NUM_THR = 12;                           
constexpr int CHUNK_SIZE = (int)(1000 / NUM_THR) + 1; 
constexpr int VERBOSITY = 3;                          
constexpr int MAX_LIMIT = TEST_MODE == 1 ? 1023 : 63; 
constexpr int N = TEST_MODE == 1 ? 50 : 100000;       
constexpr int Nv = TEST_MODE == 1 ? 2 : 1000;         
constexpr int Nc = TEST_MODE == 1 ? 5 : 100;          
constexpr double THR_KMEANS = 0.000001;               