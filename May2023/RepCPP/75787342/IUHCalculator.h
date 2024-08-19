
#pragma once

#include <string>
#include <vector>

using namespace std;

#define IUHZERO 0.000000001


class IUHCalculator {
public:
IUHCalculator(string inputfile, string watershedfile,
string stream_networkfile, string t0file, string deltafile);

virtual ~IUHCalculator(void);


virtual int calIUH(void) = 0;

void setDt(int t) { dt = t; }

protected:
string inputFile;
string watershedFile;    
string strnetworkFile;
string t0File;
string deltaFile;

int mt;          
int nRows, nCols;    
int dt;              
int nSubs;           
int nCells;          
int nLinks;          

vector <vector<int>> watershed;  
vector <vector<int>> strnetwork;  
vector <vector<int>> link;   
vector <vector<double>> t0;          
vector <vector<double>> delta;       


virtual void readData();


double IUHti(double delta0, double t00, double ti);  
};
