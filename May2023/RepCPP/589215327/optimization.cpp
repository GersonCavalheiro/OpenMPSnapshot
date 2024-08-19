#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <omp.h>
#include <cmath>
#include <tuple>
#include <algorithm>

using namespace std;

const tuple<double, double> x_limit{ -10, 10 };
const tuple<double, double> y_limit{ -10, 10 };

class Shop {
public:
double x;
double y;
Shop(double xx, double yy)
{
x = xx;
y = yy;
}
};

tuple<double, double> nearest_point_to_rectangle(double r_top, double r_bottom, double r_left, double r_right, double x, double y)
{
double d_top = abs(r_top - y);
double d_bottom = abs(r_bottom - y);
double corner_y = (d_top < d_bottom) ? r_top : r_bottom;

double d_left = abs(r_left - x);
double d_right = abs(r_right - x);
double corner_x = (d_left < d_right) ? r_left : r_right;

double d_cx = corner_x - x;
double d_cy = corner_y - y;
double d_corner = sqrt(d_cx * d_cx + d_cy * d_cy);

vector<tuple<double, string>> edges{
tuple<double, string>{d_top, "top"}, tuple<double, string>{d_bottom, "bottom"},
tuple<double, string>{d_left, "left"}, tuple<double, string>{d_right, "right"},
tuple<double, string>{d_corner, "corner"}
};
sort(edges.begin(), edges.end());

string nearest_edge = get<1>(edges.front());
if (nearest_edge == "top")
return tuple<double, double>{x, r_top};
else if (nearest_edge == "bottom")
return tuple<double, double>{x, r_bottom};
else if (nearest_edge == "left")
return tuple<double, double>{r_left, y};
else if (nearest_edge == "right")
return tuple<double, double>{r_right, y};
else if (nearest_edge == "corner")
{
if (x > 0 && y > 0)
return tuple<double, double>{r_right, r_top};
else if (x < 0 && y > 0)
return tuple<double, double>{r_left, r_top};
else if (x > 0 && y < 0)
return tuple<double, double>{r_right, r_bottom};
else if (x < 0 && y < 0)
return tuple<double, double>{r_left, r_bottom};
}
}

double C(char type, double x1, double y1, double x2, double y2)
{
if (type == 'x')
{
return -(((x1 - x2) * exp(-((pow(x1 - x2, 2) + pow(y1 - y2, 2)) / 10))) / 5);
}
else if (type == 'y')
{
return -(((y1 - y2) * exp(-((pow(y1 - y2, 2) + pow(x1 - x2, 2)) / 10))) / 5);
}
else
{
return exp(-0.1 * (pow(x1 - x2, 2) + pow(y1 - y2, 2)));
}
}

double Cr(char type, double x, double y)
{
if (x > get<0>(x_limit) and x < get<1>(x_limit) and y > get<0>(y_limit) and y < get<1>(y_limit))
return 0;
else
{
tuple<double, double> nearest_point = nearest_point_to_rectangle(get<1>(y_limit), get<0>(y_limit), get<0>(x_limit), get<1>(x_limit), x, y);
if (type == 'x')
return x - get<0>(nearest_point);
else if (type == 'y')
return y - get<1>(nearest_point);
else
return 0.5 * (pow(x - get<0>(nearest_point), 2) + pow(y - get<1>(nearest_point), 2));
}
}

double suitability_factor(char type, double x, double y, vector<Shop> shops)
{
double sum = 0;

for (int i = 0; i < shops.size(); i++)
{
if (x == shops[i].x && y == shops[i].y)
{
continue;
}

sum += C(type, x, y, shops[i].x, shops[i].y);
}

sum += Cr(type, x, y);

return sum;
}

tuple<double, double> gradients(double x, double y, vector<Shop> shops)
{
return tuple<double, double> {suitability_factor('x', x, y, shops), suitability_factor('y', x, y, shops)};
}

double norm(tuple<double, double> vector)
{
return sqrt(pow(get<0>(vector), 2) + pow(get<1>(vector), 2));
}

void split(string s, string* res)
{
stringstream ss(s);
string word;
int n = 0;
while (ss >> word) {
res[n] = word;
n++;
}
}

void readData(string filename, vector<Shop>& shops, vector<Shop>& shopsToAdd)
{
ifstream file;
file.open(filename);

string line;
int dest = 0;
while (getline(file, line))
{
if (line[0] == '#')
{
dest = 1;
continue;
}

string values[3];
split(line, values);

if (dest == 0)
{
shops.push_back(Shop(stod(values[0]), stod(values[1])));
}
else
{
shopsToAdd.push_back(Shop(stod(values[0]), stod(values[1])));
}
}
}

void printOriginalData(string filename, vector<Shop> shops, vector<Shop> shopsToAdd)
{
ofstream file;
file.open(filename, ios_base::app);

file << "Pradiniai duomenys:" << endl;
file << "\n";
file << "Jau pastatytu parduotuviu koordinates:" << endl;
file << setw(3) << "x" << setw(10) << "y" << endl;
for (int i = 0; i < shops.size(); i++)
{
file << setw(3) << shops[i].x << setw(10) << shops[i].y << endl;
}
file << "\n";

file << "Parduotuviu koordinates kurias reikia pastatyti:" << endl;
file << setw(3) << "x" << setw(10) << "y" << endl;
for (int i = 0; i < shopsToAdd.size(); i++)
{
file << setw(3) << shopsToAdd[i].x << setw(10) << shopsToAdd[i].y << endl;
}
file << "\n";
}

void printResults(string filename, vector<Shop> shops, int size)
{
ofstream file;
file.open(filename, ios_base::app);

file << "Gauti rezultatai:" << endl;
file << setw(10) << "x" << setw(10) << "y" << endl;
for (int i = 0; i < size; i++)
{
file << setw(10) << shops[i].x << setw(10) << shops[i].y << endl;
}
file << "\n";
}

void optimize(vector<Shop>& shops, vector<Shop>& starting_pos, vector<Shop>& all_shops, vector<Shop>& moved, int threads)
{
omp_set_dynamic(0);
omp_set_num_threads(threads);
double factor = 0;
double next_factor = 0;
int counter = 0;
double alfa = 0.1;

#pragma omp parallel
{
while (counter < 500 && alfa >= 1e-3)
{
double curr_factor = 0;
double new_factor = 0;

#pragma omp for
for (int i = 0; i < starting_pos.size(); i++)
{
curr_factor += suitability_factor(NULL, starting_pos[i].x, starting_pos[i].y, all_shops);

tuple<double, double> gradient = gradients(starting_pos[i].x, starting_pos[i].y, shops);
double nrm = norm(gradient);
double gradient_norm[] = { get<0>(gradient) / nrm, get<1>(gradient) / nrm };
moved[i].x = all_shops[i].x - alfa * gradient_norm[0];
moved[i].y = all_shops[i].y - alfa * gradient_norm[1];
}

#pragma omp barrier

#pragma omp for
for (int i = 0; i < starting_pos.size(); i++)
{
new_factor += suitability_factor(NULL, moved[i].x, moved[i].y, moved);
}

#pragma omp critical
{
factor += curr_factor;
next_factor += new_factor;
}

#pragma omp barrier

#pragma omp single
{
#pragma omp critical
{
counter++;
if (next_factor < factor)
{
all_shops = moved;
factor = next_factor;
next_factor = 0;
}
else
{
alfa /= 2;
moved = all_shops;
}
}
}

#pragma omp barrier
}
}
}

int main() 
{
const string dataFile = "50_50.txt";
const int nThreads = 5;

vector<Shop> shops;
vector<Shop> starting_pos;

readData(dataFile, shops, starting_pos);

string filename = "res.txt";
remove(filename.c_str());

printOriginalData(filename, shops, starting_pos);

vector<Shop> all_shops = shops;
all_shops.insert(all_shops.begin(), starting_pos.begin(), starting_pos.end());

vector<Shop> moved = all_shops;

optimize(shops, starting_pos, all_shops, moved, nThreads);

printResults(filename, all_shops, starting_pos.size());
}