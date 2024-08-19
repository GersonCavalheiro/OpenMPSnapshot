#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
#pragma omp declare reduction (merge : std::vector<long long int> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))  



void load_CSV(string file_name, float** points, long long int size) {
ifstream in(file_name);
if (!in) {
cerr << "Couldn't read file: " << file_name << "\n";
}
long long int point_number = 0; 
while (!in.eof() && (point_number < size)) {
char* line = new char[12];
streamsize row_size = 12;
in.read(line, row_size);
string row = line;
points[point_number] = new float[3];
points[point_number][0] = stof(row.substr(0, 5));
points[point_number][1] = stof(row.substr(6, 5));
points[point_number][2] = -1;
point_number++;
}
}

void save_to_CSV(string file_name, float** points, long long int size) {
fstream fout;
fout.open(file_name, ios::out);
for (long long int i = 0; i < size; i++) {
fout << points[i][0] << ","
<< points[i][1] << ","
<< points[i][2] << "\n";
}
}

float euclideanDistance(float* points1, float* points2) {
return sqrt(pow(points1[0] - points2[0], 2) + pow(points1[1] - points2[1], 2));
}

void initializeVector(vector<long long int> &checkedPoints, int size){
for (int i = 0; i<size ; i++){
checkedPoints.push_back(i);
}
}

long long int popAndReturn (vector<long long int> &checkedPoints){
if (checkedPoints.size() > 0) {
int ret = checkedPoints.back();
checkedPoints.pop_back();

return ret;
} else {
return -1;
}
}

vector<long long int> findNeighbors(int index,float** points, long long int size, float epsilon, int min_samples){
vector<long long int> neighbors;

for (int i = 0; i < size; i++){
if ( euclideanDistance(points[index], points[i]) < epsilon){
neighbors.push_back(i);
}
}

return neighbors;
}

void formClusters (vector<long long int> &checkedPoints, float** points, 
long long int size, float epsilon, int min_samples, bool debug = false){
long long int index = popAndReturn(checkedPoints);
long long int debugCounter = 0;

while (index != -1){
if (debug && debugCounter % 1000 == 0) {
cout << "Working on Element: " << debugCounter << "\n";
}
if(points[index][2] == -1){
vector<long long int> neighbors = findNeighbors(index, points, size, epsilon, min_samples);

if (neighbors.size() >= min_samples){
points[index][2] = 1;
}else{
points[index][2] = 0;
}
}
index = popAndReturn(checkedPoints);

if (debug) {
debugCounter++;
}
}
}

vector<long long int> parallel_findNeighbors(int index,float** points, 
long long int size, float epsilon, int min_samples, long long int CHUNK_SIZE){

vector<long long int> neighbors;
neighbors.reserve(size);

#pragma omp parallel for reduction(merge: neighbors)
for(int i=0; i<size; i++) 
if(euclideanDistance(points[index], points[i]) <=epsilon)
neighbors.push_back(i);

return neighbors;
}

void parallel_formClusters (vector<long long int> &checkedPoints, float** points, 
long long int size, float epsilon, int min_samples, long long int CHUNK_SIZE, bool debug = false){
long long int index = popAndReturn(checkedPoints);
long long int debugCounter = 0;

while (index != -1){
if (debug && debugCounter % 1000 == 0) {
cout << "Working on Element: " << debugCounter << "\n";
}
if(points[index][2] == -1){
vector<long long int> neighbors = parallel_findNeighbors(index, points, size, epsilon, min_samples, CHUNK_SIZE);

if (neighbors.size() >= min_samples){
points[index][2] = 1;
}else{
points[index][2] = 0;
}
}
index = popAndReturn(checkedPoints);

if (debug) {
debugCounter++;
}
}
}

void parallel_initializeVector(vector<long long int> &checkedPoints, int size){

#pragma omp parallel for reduction(merge: checkedPoints)
for(int i=0; i<size; i++) 
checkedPoints.push_back(i);
}


int main(int argc, char** argv) {

const float epsilon = 0.03;
const int min_samples = 10;
long long int size = 20000; 
int ompThreads = 4;

if (argc == 2) {
size = stoll(argv[1]);
} else if (argc == 3) {
size = stoll(argv[1]);
ompThreads = stoi(argv[2]);
}

const string input_file_name = to_string(size)+"_data.csv";
const string output_file_name = to_string(size)+"_results.csv";    
float** points = new float*[size];
vector<long long int> checkedPoints;
const long long int CHUNK_SIZE = size / ompThreads;

checkedPoints.reserve(size);

omp_set_num_threads(ompThreads);

cout<<"Loading CSV..."<<'\n';
load_CSV(input_file_name, points, size);
cout<< "CSV Loaded!"<<'\n';

cout<< "Initializing Paralell DBSCAN with "<< ompThreads << " threads..."<< '\n';
double start = omp_get_wtime();

parallel_initializeVector(checkedPoints,size);
random_shuffle(checkedPoints.begin(), checkedPoints.end());
parallel_formClusters(checkedPoints, points, size, epsilon, min_samples, CHUNK_SIZE);

double end = omp_get_wtime();
double parallelTime = (end - start) * 1000;

cout << "Finished Paralell DBSCAN!"<< "\n";
cout << "   Elapsed time: " << parallelTime<< " Milliseconds" << "\n";

cout << "Saving CSV..." << '\n';
save_to_CSV(output_file_name, points, size);
cout << "CSV saved!" << '\n';


cout << "Freeing up memory..." << '\n';
for (long long int i = size-1; i >=0; i--) {
delete[] points[i];
}
delete[] points;
cout << "Memory freed!" << '\n';

float** serial_points = new float*[size];
vector<long long int> serial_checkedPoints;

load_CSV(input_file_name,serial_points, size); 

cout << "Initializing Serial DBSCAN..."<< '\n';
auto serialStart = chrono::high_resolution_clock::now();
initializeVector(serial_checkedPoints,size);
random_shuffle(serial_checkedPoints.begin(),serial_checkedPoints.end());
formClusters(serial_checkedPoints,serial_points,size,epsilon,min_samples); 

auto serialEnd = chrono::high_resolution_clock::now();
auto duration = chrono::duration_cast<chrono::milliseconds>(serialEnd - serialStart);

cout << "Finished Serial DBSCAN!"<< '\n';
cout << "   Elapsed time: " << duration.count() << " Milliseconds"<< '\n';

cout << "Parallel version is " << duration.count()/parallelTime <<" times faster!" << endl;

cout << "Freeing up memory..." << '\n';
for (long long int i = 0; i < size; i++) {
delete[] serial_points[i];
}
delete[] serial_points;
cout << "Memory freed!" << '\n';

cout << "Done!" << '\n';

return 0;
}