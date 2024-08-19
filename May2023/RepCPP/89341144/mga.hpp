
#pragma once

#include <vector>
#include "pl_eph_an.hpp"

using namespace std;

const int orbit_insertion = 0;          
const int total_DV_orbit_insertion = 1; 
const int rndv = 2;                     
const int total_DV_rndv = 3;            
const int asteroid_impact = 4;          
const int time2AUs = 5;                 

struct customobject
{
double keplerian[6];
double epoch;
double mu;
};

struct mgaproblem
{
int type;              
vector<int> sequence;  
vector<int> rev_flag;  
double e;              
double rp;             
customobject asteroid; 
double Isp;
double mass;
double DVlaunch;
};

int MGA(
vector<double>,
mgaproblem,

vector<double> &, vector<double> &, double &);
