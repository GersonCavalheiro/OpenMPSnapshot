
#pragma once

#include <vector>
#include "mga.hpp"

struct mgadsmproblem
{
int type;                  
std::vector<int> sequence; 
double e;                  
double rp;                 
customobject asteroid;     
double AUdist;             
double DVtotal;            
double DVonboard;          

std::vector<double *> r; 
std::vector<double *> v; 
std::vector<double> DV;  
};

int MGA_DSM(

std::vector<double> x, 
mgadsmproblem &mgadsm, 


double &J 
);
