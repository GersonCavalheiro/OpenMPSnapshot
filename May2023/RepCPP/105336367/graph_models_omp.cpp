#include <cstdio>
#include <getopt.h>
#include <cstring>
#include <iostream>
#include <omp.h>

#include "adf.h"


using namespace std;

#define OPTIMAL

#define BUFFER_SIZE 1024

typedef char mychar;

bool  print_results;
char *input_file_name;
int   num_threads;





class Settings
{
public:
char* dwarfName;
char* inputFile;
char* resultFile;

Settings(char *input_file);
~Settings();
void PrintStringSettings();

private:
static const int maxArraySize = 5000;
};


Settings::Settings(char *input_file)
{
dwarfName = new char[maxArraySize];
inputFile = new char[maxArraySize];
resultFile = new char[maxArraySize];

dwarfName[0] = inputFile[0] = resultFile[0] = '\0';

sprintf(dwarfName, "%s", "GraphModels");
sprintf(inputFile, "%s", input_file);
sprintf(resultFile, "result_omp.txt");

}

Settings::~Settings()
{
delete[] dwarfName;
delete[] inputFile;
delete[] resultFile;
}

void Settings::PrintStringSettings()
{
char* stringSettings = new char[maxArraySize];
stringSettings[0] = '\0';

sprintf(stringSettings, "%s%s", stringSettings, "Kernel settings summary: ");
sprintf(stringSettings, "%s%s", stringSettings, "\nDwarf name        : ");
sprintf(stringSettings, "%s%s", stringSettings, dwarfName);
sprintf(stringSettings, "%s%s", stringSettings, "\nInputFile         : ");
sprintf(stringSettings, "%s%s", stringSettings, inputFile);
sprintf(stringSettings, "%s%s", stringSettings, "\nResultFile        : ");
sprintf(stringSettings, "%s%s", stringSettings, resultFile);

printf("%s", stringSettings);

delete[] stringSettings;
}





class HMM
{
private:
int numberOfStates;                         
int numberOfObservationSymbols;             
public:
double **stateTransitionMatrix;             

double  **observationProbabilityMatrix;     

double *initialStates;                      

HMM(int n, int m);
~HMM();
int getNumberOfStates() { return numberOfStates; }
int getNumberOfObservation() { return numberOfObservationSymbols; }
};


HMM::HMM(int n, int m)
{
numberOfStates = n;
numberOfObservationSymbols = m;

stateTransitionMatrix = new double*[numberOfStates];
observationProbabilityMatrix = new double*[numberOfStates];
for(int i = 0; i < numberOfStates; i++)
{
stateTransitionMatrix[i] = new double[numberOfStates];
observationProbabilityMatrix[i] = new double[numberOfObservationSymbols];
}

initialStates = new double[numberOfStates];
for(int i=0; i < numberOfStates; i++ ) {
for(int j=0; j < numberOfStates; j++) {
stateTransitionMatrix[i][j] = 0;
}
for(int j=0; j < numberOfObservationSymbols; j++) {
observationProbabilityMatrix[i][j] = 0;
}
initialStates[i] = 0;
}
};

HMM::~HMM()
{
for(int i = 0;i < numberOfStates; i++) {
delete[]stateTransitionMatrix[i];
delete[]observationProbabilityMatrix[i];
}
delete []stateTransitionMatrix;
delete []observationProbabilityMatrix;
delete []initialStates;
};




class ViterbiModel
{
private:
int lengthOfObservationSequence;            
double probability;                         
public:
double **delta;                             
int **psi;                                  
int *observationSequence;                   
int *sequenceOfState;                       

ViterbiModel(int t,HMM *hmm);
~ViterbiModel();
int getLengthOfObservationSequence() { return lengthOfObservationSequence; }
double getProbability() { return probability; }
void setProbability(double p) { probability = p; }
};


ViterbiModel::ViterbiModel(int t,HMM *hmm)
{
lengthOfObservationSequence = t;
delta = new double*[lengthOfObservationSequence];
psi = new int*[lengthOfObservationSequence];
int numberOfState = hmm->getNumberOfStates();
for(int i = 0; i < lengthOfObservationSequence; i++) {
delta[i] = new double[numberOfState];
psi[i] = new int[numberOfState];
}

observationSequence = new int[lengthOfObservationSequence];
sequenceOfState = new int[lengthOfObservationSequence];

for(int i = 0; i < lengthOfObservationSequence; i ++) {
for(int j = 0; j < numberOfState; j++) {
delta[i][j] = 0;
psi[i][j] = 0;
}
observationSequence[i] = 0;
sequenceOfState[i] = 0;
}
};


ViterbiModel::~ViterbiModel()
{
for(int i = 0;i < lengthOfObservationSequence; i++) {
delete[]psi[i];
delete[]delta[i];
}
delete []psi;
delete []delta;
delete []observationSequence;
delete []sequenceOfState;
};





class Configurator
{
public:
Configurator(char *input_file) { settings = new Settings(input_file); }
~Configurator() { delete settings; }
void GetContent(HMM **hmm, ViterbiModel **vit);
void WriteSettings() { settings -> PrintStringSettings(); }
void Close(ViterbiModel *vit);

private:
Settings* settings;
};


void Configurator :: GetContent(HMM **hmm, ViterbiModel **vit)
{
int m,n,t, ret;
FILE *file = fopen(settings -> inputFile, "rb");		

ret = fscanf(file, "M:%d\n", &m);		
ret = fscanf(file, "N:%d\n", &n);		

*hmm = new HMM(n,m);

ret = fscanf(file, "A:\n");
for (int i = 0; i < n; i++) {
for (int j = 0; j < n; j++) {
ret = fscanf(file, "%lf", &((*hmm)->stateTransitionMatrix[i][j]));
}
ret = fscanf(file,"\n");
}

ret = fscanf(file, "B:\n");
for (int i = 0; i < n; i++) {
for (int j = 0; j < m; j++) {
ret = fscanf(file, "%lf", &((*hmm)->observationProbabilityMatrix[i][j]));
}
ret = fscanf(file,"\n");
}

ret = fscanf(file, "pi:\n");
for (int i = 0; i < n; i++) {
ret = fscanf(file, "%lf", &((*hmm)->initialStates[i]));
}

ret = fscanf(file,"\n");
ret = fscanf(file, "T:%d\n", &t);
*vit = new ViterbiModel(t, *hmm);
ret = fscanf(file, "O:\n");
for(int i = 0; i < t; i++) {
ret = fscanf(file, "%d", &((*vit)->observationSequence[i]));
}

fclose(file);
printf("number of states = %d\tnumber of observations = %d\n", n, m);
}

void Configurator :: Close(ViterbiModel *vit)
{
if (print_results == true) {
FILE *result = fopen(settings -> resultFile, "wb");

fprintf(result,  "#Result domain:\r\n");
int length = vit->getLengthOfObservationSequence();
for(int i = 0; i < length; i++) {
fprintf(result,"%d ", vit->sequenceOfState[i]);
}
fprintf(result, "\r\n");
fprintf(result,"Probability: %g \r\n", vit->getProbability() );
fprintf(result, "#eof");

fclose(result);
}
}





class Solver
{
public:
HMM          *hmm;          
ViterbiModel *vit;          

Solver(Configurator* configurator);
~Solver();
void Solve();
void PrintSummery() {};
void Finish() { dwarfConfigurator->Close(vit); }

private:
Configurator	 *dwarfConfigurator;

};

Solver :: Solver(Configurator* configurator)
{
hmm = NULL;
vit = NULL;
dwarfConfigurator = configurator;					
dwarfConfigurator -> WriteSettings();
dwarfConfigurator -> GetContent(&hmm, &vit);
}

Solver :: ~Solver()
{
delete hmm;
delete vit;
}


void Solver::Solve()
{

int numberOfStates = hmm->getNumberOfStates();
for (int i = 0; i < numberOfStates; i++) {
vit->delta[0][i] = hmm->initialStates[i] * (hmm->observationProbabilityMatrix[i][vit->observationSequence[0]]);
vit->psi[0][i] = 0;
}


int lengthOfObsSeq = vit->getLengthOfObservationSequence();
printf("\nlengthOfObsSeq = %d\n", lengthOfObsSeq);
printf("numberOfStates = %d\n",numberOfStates);

int maxvalind = 0;                        	
double maxval = 0.0, val = 0.0;             

for (int t = 1; t < lengthOfObsSeq; t++) {

#pragma omp parallel for private(val, maxval, maxvalind)
for (int j = 0; j < numberOfStates; j++) {
maxval = 0.0;
maxvalind = 0;
for (int i = 0; i < numberOfStates; i++) {
val = vit->delta[t-1][i]*(hmm->stateTransitionMatrix[i][j]);
if (val > maxval) {
maxval = val;
maxvalind = i;
}
}
vit->delta[t][j] = maxval*(hmm->observationProbabilityMatrix[j][vit->observationSequence[t]]);
vit->psi[t][j] = maxvalind;
}
}



vit->setProbability(0.0);

vit->sequenceOfState[lengthOfObsSeq - 1] = 0;
for (int i = 0; i < numberOfStates; i++) {
if (vit->delta[lengthOfObsSeq - 1][i] > vit->getProbability()) {
vit->setProbability(vit->delta[lengthOfObsSeq - 1][i]);
vit->sequenceOfState[lengthOfObsSeq - 1] = i;
}
}


for (int t = lengthOfObsSeq - 2; t >= 0; t--) {
vit->sequenceOfState[t] = vit->psi[t+1][vit->sequenceOfState[t+1]];
}
}





void ParseCommandLine(int argc, char **argv)
{
char c;

input_file_name = (char*) "smallcontent.txt";
num_threads = 1;
print_results = false;

while ((c = getopt (argc, argv, "hf:n:r")) != -1)
switch (c) {
case 'h':
printf("\nGraph Models - ADF benchmark application\n"
"\n"
"Usage:\n"
"   graph_models_omp [options ...]\n"
"\n"
"Options:\n"
"   -h\n"
"        Print this help message.\n"
"   -f <filename>\n"
"        Input file name.\n"
"   -n <long>\n"
"        Number of worker threads. (default 1)\n"
"   -r \n"
"        print results to a result file\n"
"\n"
);
exit (0);
case 'f':
input_file_name = optarg;
break;
case 'n':
num_threads = strtol(optarg, NULL, 10);
break;
case 'r':
print_results = true;
break;
case '?':
if (optopt == 'c')
fprintf (stderr, "Option -%c requires an argument.\n", optopt);
else if (isprint (optopt))
fprintf (stderr, "Unknown option `-%c'.\n", optopt);
else
fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
exit(1);
default:
exit(1);
}

if(input_file_name == NULL) {
printf("missing input file!\n");
exit(1);
}

printf ("\nStarting openmp Graph Models.\n");
printf ("Running with %d threads. Input file %s\n", num_threads, input_file_name);
printf ("=====================================================\n\n");
}






int main(int argc, char** argv)
{
ParseCommandLine(argc, argv);

Configurator dwarfConfigurator(input_file_name);
Solver solver(&dwarfConfigurator);					

NonADF_init(num_threads);

omp_set_num_threads(num_threads);
omp_set_nested(1);

solver.Solve();										

NonADF_terminate();

solver.Finish();									

return 0;
}



