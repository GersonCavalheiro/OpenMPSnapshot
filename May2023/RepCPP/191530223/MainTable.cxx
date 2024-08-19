
#include "MainTable.h"

const std::pair<double, double> MainTable::fThreshold = {0, 1e20}; 

MainTable::MainTable()
{
fNComponents = 10000;
fNRealizations = 10000;

fData = new correlated[fNRealizations * fNComponents];
}

MainTable::~MainTable()
{
delete[] fData;
printf("\n\n");
}

void MainTable::read(const char *tableFilename, bool saveAbundancesOccurrences)
{
cout << "Reading " << tableFilename << endl;
fstream file(tableFilename, std::ios::in);
if (!file.is_open())
{
cerr << "file not found" << endl;
}
else
{
string line;
uint64_t idata = 0;

double A[fNComponents];
double O[fNComponents];
double VocabularySize[fNRealizations];
if (saveAbundancesOccurrences)
{
for (uint64_t i = 0; i < fNComponents; i++)
A[i] = 0.;
for (uint64_t i = 0; i < fNComponents; i++)
O[i] = 0.;
for (uint64_t i = 0; i < fNRealizations; i++)
VocabularySize[i] = 0.;
}
getline(file, line);

bool firstRead = false;
uint64_t actualWords = 0;
while (getline(file, line).good())
{
auto tokenizedLine = tokenize(line);

if (!firstRead)
{
fNRealizations = tokenizedLine.size() - 1;
firstRead = true;
}
if (idata % fNRealizations == 0)
printf("\r%lu/%lu", idata, fNRealizations * fNComponents);

for (auto token = tokenizedLine.begin() + 1; token != tokenizedLine.end(); token++)
{
double value = std::stod(*token);
bool binaryValue = (value > MainTable::fThreshold.first) && (value < MainTable::fThreshold.second);
fData[idata] = binaryValue;

if (saveAbundancesOccurrences)
{
A[idata / fNRealizations] += (binaryValue ? value : 0.); 
O[idata / fNRealizations] += (binaryValue ? 1. : 0.) / fNRealizations;
}
VocabularySize[idata % fNRealizations] += value;
idata++;
}
actualWords++;
}

fNComponents = actualWords;
file.close();
cout << endl;

#pragma omp parallel sections
{
#pragma omp section
{
if (saveAbundancesOccurrences)
{
SaveTotalArray("A.dat", A, fNComponents);
SaveTotalArray("O.dat", O, fNComponents);
}
cout << endl;
}
#pragma omp section
{
if (saveAbundancesOccurrences)
{
SaveTotalArray("vocabulary_size.dat", VocabularySize, fNRealizations);
SaveHeapsData(VocabularySize);
}
cout << endl;
}
}
}
}

void MainTable::readNull(const char *tableFilename, bool saveAbundancesOccurrences, bool saveMeansVariances, bool considerZeros, uint8_t maxStatistics)
{

double A[fNComponents];
double O[fNComponents];
long double means[fNComponents];
long double variances[fNComponents];
long double cv2[fNComponents];
double VocabularySize[fNRealizations];
if (saveAbundancesOccurrences)
{
for (uint64_t i = 0; i < fNComponents; i++)
A[i] = 0.;
for (uint64_t i = 0; i < fNComponents; i++)
O[i] = 0.;
for (uint64_t i = 0; i < fNRealizations; i++)
VocabularySize[i] = 0.;
}

if (saveMeansVariances)
{
for (uint64_t i = 0; i < fNComponents; i++)
means[i] = 0.;
for (uint64_t i = 0; i < fNComponents; i++)
variances[i] = 0.;
for (uint64_t i = 0; i < fNComponents; i++)
cv2[i] = 0.;
}

for (uint8_t iStatistics = 0; iStatistics < maxStatistics; iStatistics++)
{
stringstream filename;
auto filenameC = string(tableFilename);
filenameC = filenameC.substr(0, filenameC.size() - 4); 
filename << filenameC << "_" << to_string(iStatistics)<<".csv";

cout << "Reading " << filename.str() << endl;
fstream file(filename.str(), std::ios::in);
if (!file.is_open())
{
cerr << "file not found" << endl;
}
else
{
string line;
uint64_t idata = 0;
double sum = 0.;
double sumsquare = 0.;
int n = 0;

getline(file, line);

bool firstRead = false;
uint64_t actualWords = 0;
while (getline(file, line).good())
{
auto tokenizedLine = tokenize(line);

if (!firstRead)
{
fNRealizations = tokenizedLine.size() - 1;
firstRead = true;
}
if (idata % fNRealizations == 0)
printf("\r%lu/%lu", idata, fNRealizations * fNComponents);

for (auto token = tokenizedLine.begin() + 1; token != tokenizedLine.end(); token++)
{
double value = std::stod(*token);
bool binaryValue = (value > MainTable::fThreshold.first) && (value < MainTable::fThreshold.second);
fData[idata] = binaryValue;

if (saveAbundancesOccurrences)
{
A[idata / fNRealizations] += (binaryValue ? value : 0.); 
O[idata / fNRealizations] += (binaryValue ? 1. : 0.) / fNRealizations;
}

if (value > fThreshold.first && value < fThreshold.second)
{
sum += value;
sumsquare += value * value;
n += 1;
}
else if (considerZeros)
{
if (value < fThreshold.first)
value = 0;
sum += value;
sumsquare += value * value;
n += 1;
}

VocabularySize[idata % fNRealizations] += value;
idata++;
}

if (saveMeansVariances)
{
if (n > 1)
{
long double average = sum / n;
long double variance = (sumsquare - (sum * sum) / n) / (n); 
means[idata / fNRealizations] += average;
variances[idata / fNRealizations] += variance;
cv2[idata / fNRealizations] += average>0.?variance/average/average:0.;
}
}

actualWords++;
}

fNComponents = actualWords;
file.close();
cout << endl;



if (saveAbundancesOccurrences)
{
for (uint64_t i = 0; i < fNComponents; i++)
A[i] /= maxStatistics;
for (uint64_t i = 0; i < fNComponents; i++)
O[i] /= maxStatistics;
for (uint64_t i = 0; i < fNRealizations; i++)
VocabularySize[i] /= maxStatistics;
}

if (saveMeansVariances)
{
for (uint64_t i = 0; i < fNComponents; i++)
means[i] /= maxStatistics;
for (uint64_t i = 0; i < fNComponents; i++)
variances[i] /= maxStatistics;
for (uint64_t i = 0; i < fNComponents; i++)
cv2[i] /= maxStatistics;
}
#pragma omp parallel sections
{
#pragma omp section
{
if (saveAbundancesOccurrences)
{
SaveTotalArray("A.dat", A, fNComponents);
SaveTotalArray("O.dat", O, fNComponents);
}
cout << endl;
}
#pragma omp section
{
if (saveAbundancesOccurrences)
{
SaveTotalArray("vocabulary_size.dat", VocabularySize, fNRealizations);
SaveHeapsData(VocabularySize);
}
cout << endl;
}
}
}
}
}

void MainTable::readBinary()
{
cout << "Reading maintable.csv" << endl;
fstream file("binaryTable.csv", std::ios::in);

if (!file.is_open())
{
cerr << "file not found" << endl;
}
else
{

string line;
uint64_t idata = 0;

bool firstRead = false;
while (getline(file, line).good())
{
auto tokenizedLine = tokenize(line);

if (!firstRead)
{
fNRealizations = tokenizedLine.size() - 1;
firstRead = true;
}
if (idata % 1000 == 0)
printf("\r%lu/%lu", idata, fNRealizations * fNComponents);

for (auto token = tokenizedLine.begin() + 1; token != tokenizedLine.end(); token++)
{
bool value = std::stoi(*token) == 1;
fData[idata++] = value;
}
}

file.close();
cout << endl;
}
}

std::vector<std::string> MainTable::tokenize(const std::string &line)
{
boost::escaped_list_separator<char> sep('\\', ',', '\0');
boost::tokenizer<boost::escaped_list_separator<char>> tokenizer(line, sep);
return std::vector<std::string>(tokenizer.begin(), tokenizer.end());
}

void MainTable::SaveBinary(const char *filename)
{
cout << "Saving binary matrix" << endl;
fstream file(filename, std::ios::out);
for (uint64_t idata = 0; idata < fNComponents * fNRealizations; idata++)
{
if (idata % 100000 == 0)
printf("\r%lu/%lu", idata, fNRealizations * fNComponents);
file << (fData[idata] ? 1 : 0);
if ((idata + 1) % fNRealizations == 0)
file << "\n";
else
file << ",";
}
file.close();
cout << endl;
}

void MainTable::SaveTotalArray(const char *filename, const double *X, uint64_t length)
{
printf("Saving total %s\n", filename);

fstream file(filename, std::ios::out);
file << X[0];
for (uint64_t i = 1; i < length; i++)
{
file << "\n"
<< X[i];
}

file.close();
}

void MainTable::SaveHeapsData(const double *VocabularySize)
{
cout << "Saving Heaps data" << endl;
fstream file("heaps.dat", std::ios::out);

for (uint64_t realisation = 0; realisation < fNRealizations; ++realisation)
{
printf("\r%lu/%lu", realisation + 1, fNRealizations);

uint64_t cNumberOfDifferentWords = 0;
long double cVocabularySize = VocabularySize[realisation];

#pragma omp parallel for reduction(+ \
: cNumberOfDifferentWords)
for (uint64_t component = 0; component < fNComponents; component++)
{
if (get(component, realisation) != 0)
cNumberOfDifferentWords++;
}
file << cVocabularySize << "," << cNumberOfDifferentWords << endl;
file.flush();
}

file.close();
cout << endl;
}

void MainTable::ExtimateCorrelations(const char *filename)
{
cout << "Extimating correlations" << endl;

fNComponents = 1000;
ExtimateHXY(filename);
}

void MainTable::ExtimateHXY(const char *filename)
{
cout << "Extimating H(X,Y)" << endl;
cout << "words: " << fNComponents << "\t documents: " << fNRealizations << endl;

fstream file(filename, ios_base::out);

double norm = 1. / fNRealizations;
double H, h;
double hx[2];
double hy[2];

#pragma omp parallel for shared(file)
for (uint64_t firstComponent = 0; firstComponent < fNComponents; firstComponent++)
{
for (uint64_t secondComponent = firstComponent + 1; secondComponent < fNComponents; secondComponent++)
{
printf("\r%lu/%lu", firstComponent, secondComponent);
double P[4] = {0.};
for (uint64_t realization = 0; realization < fNRealizations; ++realization)
{
auto x = get(firstComponent, realization);
auto y = get(secondComponent, realization);

if (x == y)
{
if (x == 0)
{ 
P[0] += norm;
}
else
{ 
P[3] += norm;
}
}
else
{
if (x == 0)
{ 
P[1] += norm;
}
else
{ 
P[2] += norm;
}
}
}

#pragma omp critical
{
h = GetEntropy(4, P);
hx[0] = P[0] + P[1]; 
hx[1] = 1. - hx[0];  

hy[0] = P[0] + P[2]; 
hy[1] = 1. - hy[0];  

H = GetEntropy(2, hx, firstComponent) + GetEntropy(2, hy, secondComponent) - h;
file  << H << endl;
}
}

file.flush();
}

file.close();
cout << endl;
}

double MainTable::GetEntropy(uint64_t l, double *X, const uint64_t component)
{
static std::map<uint64_t, double> cache;

if (component <= fNComponents)
{
auto it = cache.find(component);
if (it != cache.end())
{
return it->second;
}
else
{
double H = SumEntropy(l, X);
cache.insert(std::pair<uint64_t, double>(component, H));
return H;
}
}
else
{
return SumEntropy(l, X);
}
}

double MainTable::SumEntropy(uint64_t l, double *X)
{
double H = 0.;
for (uint64_t i = 0; i < l; i++)
{
double x = X[i];
if (x > 1e-5)
H += x * log2(x);
}
return -H;
}

void MainTable::SaveMeansVariances(const char *filename, bool considerZeros)
{
cout << "Reading " << filename << endl;
fstream file(filename, std::ios::in);
fstream meanVariances("meanVariances.csv", std::ios::out);
meanVariances << ",mean,variance,type_of_gene" << endl;
if (!file.is_open())
{
cerr << "file not found" << endl;
}
else
{
string line;

std::string header;
getline(file, header).good();

uint64_t nwords = 0;
while (getline(file, line).good())
{
auto tokenizedLine = tokenize(line);

printf("\rnwords: %lu", ++nwords);

auto gene = (*(tokenizedLine.begin())).substr(0, BioParameters::getENSLenght());

long double sum = 0.;
long double sumsquare = 0.;
uint64_t n = 0;

for (auto token = tokenizedLine.begin() + 1; token != tokenizedLine.end(); token++)
{
double value = std::stod(*token);

if (value > fThreshold.first && value < fThreshold.second)
{
sum += value;
sumsquare += value * value;
n += 1;
}
else if (considerZeros)
{
if (value < fThreshold.first)
value = 0;
sum += value;
sumsquare += value * value;
n += 1;
}
}

if (n > 1)
{
long double average = sum / n;
long double variance = (sumsquare - (sum * sum) / n) / (n); 
meanVariances << gene << "," << average << "," << variance << ","
<< " " << endl;
}
else
{
meanVariances << gene << "," << 0 << "," << 0 << ","
<< " " << endl;
}
}
}
cout << endl;
}
