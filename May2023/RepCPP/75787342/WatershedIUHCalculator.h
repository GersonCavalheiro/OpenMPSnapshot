
#pragma once

#include "IUHCalculator.h"

class WatershedIUHCalculator : public IUHCalculator {
public:
WatershedIUHCalculator(string inputfile, string watershedfile,
string stream_networkfile, string t0file, string deltafile,
string runoffcofile,
string uhcellfile, string uhsubfile, string uhwatershedfile = "");

WatershedIUHCalculator(string inputfile, string watershedfile,
string stream_networkfile, string t0file, string deltafile,
string runoffcofile,
string uhwatershedfile = "");

virtual ~WatershedIUHCalculator(void);

private:
string runoffCoFile;
string uhCellFile, uhSubFile, uhWatershedFile;

vector<double> runoffSub;             
vector <vector<double>> runoffCo;    

vector <vector<double>> uhCell, uh1;      
vector <vector<double>> uhSub, uh2;       
vector<double> uhWatershed, uh3; 

int maxtSub;                     
double sumRunCo;

protected:
virtual void readData();

private:
int calCell(void);

int calSub(void);

int calWatershed(void);

public:
virtual int calIUH(void);

int calIUHSubwatershed(void);

int calIUHCell(void);
};
