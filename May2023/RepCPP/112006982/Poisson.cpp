#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <mpi.h>
#include <string>

#include "Poisson.h"
using namespace std;

#ifdef _OPENMP
#include <omp.h>
#endif

Poisson::Poisson(){
grid = NULL;
grid_x = 0; grid_y = 0;
}

Poisson::Poisson(const int x, const int y){
grid_x = x; grid_y = y;
grid = new double[x * y];
}

Poisson::~Poisson(){
if (grid != NULL){
delete [] grid;
}
}

double& Poisson::operator ()(int x, int y){
return grid[x * grid_y + y];
}

Poisson& Poisson::operator=(const Poisson& tmp) {
this->grid_x = tmp.grid_x;
this->grid_y = tmp.grid_y;
int i = 0;
#pragma omp parallel
#pragma omp for schedule (static)
for (i = 0; i < grid_x * grid_y; i++) {
this->grid[i] = tmp.grid[i];
}
return *this;
}

int Poisson::size_x() const{
return grid_x;
}

int Poisson::size_y() const{
return grid_y;
}

double Poisson::max() const{
double maximum = grid[0];
for (int i = 1; i < grid_x * grid_y; i++) {
maximum = maximum < grid[i] ? grid[i] : maximum;
}
return maximum;
}

double Solver::F(const double x, const double y) const{
double num = 8.0 * (1.0 - x * x -  y * y);
double denum = (1 + x * x + y * y);
return num / (denum * denum * denum);
}

double Solver::phi(const double x, const double y) const{
return 2.0 / (1 + x * x + y * y);
}

double Solver::x(const int index, const int shift) const{
return lx + (index + shift) * delta;
}

double Solver::y(const int index, const int shift) const{
return ly + (index + shift) * delta;
}

double Solver::DiffScheme(Poisson& p, int i, int j) const{
return ((2 * p(i, j) - p(i - 1, j) - p(i + 1, j))  +
(2 * p(i, j) - p(i, j - 1) - p(i, j + 1))) / delta2;
}

double Solver::ScalarDot(Poisson& p, Poisson& q) const{
double scalar_dot = 0;
int size_x = p.size_x();
int size_y = p.size_y();
#pragma omp parallel
#pragma omp for schedule (static) reduction(+:scalar_dot)
for (int i = 1; i < size_x - 1; i++) {
for (int j = 1; j < size_y - 1; j++) {
scalar_dot += delta2 * p(i, j) * q(i, j);
}
}
return scalar_dot;
}

Solver::Solver() :
dimension(300),
lx(0.0), rx(2.0),
ly(0.0), ry(2.0),
delta((rx - lx) / dimension), delta2(delta * delta),
eps(1e-4),
fname("result.txt")
{ }

Solver::Solver(const int dimension,
const double eps,
string fname) :
dimension(dimension),
lx(0.0), rx(2.0),
ly(0.0), ry(2.0),
delta((rx - lx) / dimension), delta2(delta * delta),
eps(eps),
fname(fname)
{ }

float Solver::ProcessDot(float var, const int rank, const int size) const{
float *processes_sum;
if (rank == 0){
processes_sum = new float[size];
}

MPI_Gather(&var, 1, MPI_FLOAT, processes_sum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

float sum = 0.0f;
if (rank == 0) {
#pragma omp parallel
#pragma omp for schedule (static) reduction(+:sum)
for (int i = 0; i < size; i++)
sum += processes_sum[i];
delete [] processes_sum;
}

MPI_Bcast(&sum, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
return sum;
}

float Solver::ProcessMax(float var, const int rank, const int size) const{
float *processes_max;
if (rank == 0){
processes_max = new float[size];
}

MPI_Gather(&var, 1, MPI_FLOAT, processes_max, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

float max;
if (rank == 0) {
max = processes_max[0];
for (int i = 1; i < size; i++)
max = max < processes_max[i] ? processes_max[i] : max;
delete [] processes_max;
}

MPI_Bcast(&max, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
return max;
}

void Solver::ProcessConform(Poisson& p, const int rank, const int blocks_x,
const int blocks_y){
const int block_pos_x = rank / blocks_y;
const int block_pos_y = rank % blocks_y;

bool left   = true;
bool right  = true;
bool up     = true;
bool bottom = true;

float *send_left, *send_right;
float *send_up,   *send_bottom;

const int width  = p.size_y() - 2;
const int height = p.size_x() - 2;

const int l_neighbor = blocks_y * (block_pos_x + 0) + (block_pos_y - 1);
const int r_neighbor = blocks_y * (block_pos_x + 0) + (block_pos_y + 1);
const int u_neighbor = blocks_y * (block_pos_x - 1) + (block_pos_y + 0);
const int b_neighbor = blocks_y * (block_pos_x + 1) + (block_pos_y + 0);

MPI_Status status;
MPI_Request send_request_left, send_request_right;
MPI_Request send_request_up, send_request_bottom;

up     = block_pos_x == 0            ? false : true;
bottom = block_pos_x == blocks_x - 1 ? false : true;
left   = block_pos_y == 0            ? false : true;
right  = block_pos_y == blocks_y - 1 ? false : true;

int i = 0;
#pragma omp parallel
#pragma omp sections private(i)
{
#pragma omp section
if (left) {
send_left = new float[height];
for (i = 0; i < height; ++i) {
send_left[i] = p(i + 1, 1);
}
MPI_Isend(send_left, height, MPI_FLOAT, l_neighbor, 0,
MPI_COMM_WORLD, &send_request_left);
}
#pragma omp section
if (right) {
send_right = new float[height];
for (i = 0; i < height; ++i) {
send_right[i] = p(i + 1, width);
}
MPI_Isend(send_right, height, MPI_FLOAT, r_neighbor, 0,
MPI_COMM_WORLD, &send_request_right);
}
#pragma omp section
if (up) {
send_up = new float[width];
for (i = 0; i < width; ++i) {
send_up[i] = p(1, i + 1);
}
MPI_Isend(send_up, width, MPI_FLOAT, u_neighbor, 0,
MPI_COMM_WORLD, &send_request_up);
}
#pragma omp section
if (bottom) {
send_bottom = new float[width];
for (i = 0; i < width; ++i) {
send_bottom[i] = p(height, i + 1);
}

MPI_Isend(send_bottom, width, MPI_FLOAT, b_neighbor, 0,
MPI_COMM_WORLD, &send_request_bottom);
}
}
#pragma omp parallel
#pragma omp sections private(i)
{
#pragma omp section
if (left) {
float *recv_left = new float[height];
MPI_Recv(recv_left, height, MPI_FLOAT, l_neighbor, 0, MPI_COMM_WORLD,
MPI_STATUS_IGNORE);
for (i = 0; i < height; ++i) {
p(i + 1, 0) = recv_left[i];
}

delete[] recv_left;
}
#pragma omp section
if (right) {
float *recv_right = new float[height];
MPI_Recv(recv_right, height, MPI_FLOAT, r_neighbor, 0, MPI_COMM_WORLD,
MPI_STATUS_IGNORE);
for (i = 0; i < height; ++i) {
p(i + 1, width + 1) = recv_right[i];
}
delete[] recv_right;
}
#pragma omp section
if (up) {
float *recv_up = new float[width];
MPI_Recv(recv_up, width, MPI_FLOAT, u_neighbor, 0, MPI_COMM_WORLD,
MPI_STATUS_IGNORE);
for (i = 0; i < width; ++i) {
p(0, i + 1) = recv_up[i];
}
delete[] recv_up;
}
#pragma omp section
if (bottom) {
float *recv_bottom = new float[width];
MPI_Recv(recv_bottom, width, MPI_FLOAT, b_neighbor, 0, MPI_COMM_WORLD,
MPI_STATUS_IGNORE);
for (i = 0; i < width; ++i) {
p(height + 1, i + 1) = recv_bottom[i];
}
delete[] recv_bottom;
}
}

if (left) {
MPI_Wait(&send_request_left, &status);
delete[] send_left;
}

if (right) {
MPI_Wait(&send_request_right, &status);
delete[] send_right;
}

if (up) {
MPI_Wait(&send_request_up, &status);
delete[] send_up;
}

if (bottom) {
MPI_Wait(&send_request_bottom, &status);
delete[] send_bottom;
}
}

void Solver::Solve(int argc, char** argv){
double tau, alpha;

int size;
int rank;

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);


int blocks_y = 1;
while (size / blocks_y > 2 * blocks_y) {
blocks_y *= 2;
}
const int blocks_x = size / blocks_y;

const int block_pos_x  = rank / blocks_y;
const int block_pos_y  = rank % blocks_y;

const int block_size_x = (dimension - 1) / blocks_x;
const int block_size_y = (dimension - 1) / blocks_y;

const int start_i = std::max(0, block_size_x * block_pos_x - 1);
const int end_i   = block_pos_x + 1 < blocks_x ? start_i + block_size_x : dimension;

const int start_j = std::max(0, block_size_y * block_pos_y - 1);
const int end_j   = block_pos_y + 1 < blocks_y ? start_j + block_size_y : dimension;

const int block_height = end_i - start_i + 1;
const int block_width  = end_j - start_j + 1;

Poisson p(block_height, block_width);
Poisson pk(block_height, block_width);
Poisson rk(block_height, block_width);
Poisson r_laplass(block_height, block_width);
Poisson gk(block_height, block_width);
Poisson g_laplass(block_height, block_width);
Poisson check(block_height, block_width);
Poisson error(block_height, block_width);

int i = 0;
int j = 0;

#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i + 1 < block_height; i++) {
for (j = 1; j + 1 < block_width; j++) {
check(i, j) = phi(x(i, start_i), y(j, start_j));
}
}

#pragma omp section
if (start_i == 0) {
for (j = 0; j < block_width; j++) {
p(0, j) = phi(x(0, start_i), y(j, start_j));
}
}

#pragma omp section
if (end_i == dimension) {
for (j = 0; j < block_width; j++) {
p(block_height - 1, j) = phi(x(block_height - 1, start_i), y(j, start_j));
}
}

#pragma omp section
if (start_j == 0) {
for (i = 0; i < block_height; i++) {
p(i, 0) = phi(x(i, start_i), y(0, start_j));
}
}

#pragma omp section
if (end_j == dimension) {
for (i = 0; i < block_height; i++) {
p(i, block_width - 1) = phi(x(i, start_i), y(block_width - 1, start_j));
}

}
}
pk = p;

#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
pk(i, j) = 0;
}
}

#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
rk(i, j) = DiffScheme(pk, i, j) - F(x(i, start_i), y(j, start_j));
}
}
}
ProcessConform(rk, rank, blocks_x, blocks_y);

#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
r_laplass(i, j) = DiffScheme(rk, i, j);
}
}
}

float tau_s1 = ScalarDot(rk, rk);
float tau_s2 = ScalarDot(r_laplass, rk);
tau_s1 = ProcessDot(tau_s1, rank, size);
tau_s2 = ProcessDot(tau_s2, rank, size);
tau = tau_s1 / tau_s2;

#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
pk(i, j) -= tau * rk(i, j);
}
}
}

gk = rk;
Poisson p_pred(block_height, block_width);
Poisson term(block_height, block_width);


int count = 0;
while (true) {
count++;
ProcessConform(pk, rank, blocks_x, blocks_y);
#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 0; i < block_height; i++) {
for (j = 0; j < block_width; j++) {
term(i, j) = pk(i, j) - p_pred(i, j);
}
}
}

float maximum = term.max();  
maximum = ProcessMax(maximum, rank, size);
if (maximum < eps) {
break;
}

p_pred = pk;
#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
rk(i, j) = DiffScheme(pk, i, j) - F(x(i, start_i), y(j, start_j));
}
}
}

ProcessConform(rk, rank, blocks_x, blocks_y);

#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
g_laplass(i, j) = DiffScheme(gk, i, j);
}
}
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
r_laplass(i, j) = DiffScheme(rk, i, j);
}
}
}

float alpha_s1 = ScalarDot(r_laplass, gk);
float alpha_s2 = ScalarDot(g_laplass, gk);
alpha_s1 = ProcessDot(alpha_s1, rank, size);
alpha_s2 = ProcessDot(alpha_s2, rank, size);

alpha = alpha_s1 / alpha_s2;


#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
gk(i, j) = rk(i, j) - alpha * gk(i, j);
}
}
}
ProcessConform(gk, rank, blocks_x, blocks_y);

#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
g_laplass(i, j) = DiffScheme(gk, i, j);
}

}
}

float tau_s1 = ScalarDot(rk, gk);
float tau_s2 = ScalarDot(g_laplass, gk);
tau_s1 = ProcessDot(tau_s1, rank, size);
tau_s2 = ProcessDot(tau_s2, rank, size);

tau = tau_s1/tau_s2;

#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i < block_height - 1; i++) {
for (j = 1; j < block_width - 1; j++) {
pk(i, j) -= tau * gk(i, j);
}
}
}
}
#pragma omp parallel
#pragma omp sections private(i, j)
{
#pragma omp section
for (i = 1; i + 1 < block_height; i++) {
for (j = 1; j + 1 < block_width; j++) {
error(i, j) = (check(i, j) - pk(i, j)) * (check(i, j) - pk(i, j));
}
}
}
cout << error.max() << endl;
cout << count << endl;
ofstream res, correct, err;

MPI_File file;
MPI_Status status;
MPI_File_open(MPI_COMM_WORLD, "test.bin", MPI_MODE_CREATE|MPI_MODE_WRONLY,
MPI_INFO_NULL, &file);
float *write_buffer = new float[dimension];

for (int i = start_i + 1; i < end_i; ++i) {
MPI_Offset offset = sizeof(float) * ((dimension - 1) * (i - 1) + start_j);
for (int j = 1; j + 1 < p.size_y(); ++j) {
write_buffer[j - 1] = pk(i - start_i, j);
}
MPI_File_seek(file, offset, MPI_SEEK_SET);
MPI_File_write(file, write_buffer, (end_j - start_j - 1), MPI_FLOAT, &status);
}
delete [] write_buffer;

MPI_File_close(&file);

MPI_Barrier(MPI_COMM_WORLD);
if (rank == 0){
ifstream in("test.bin");
ofstream out(fname.c_str());
float result;
int i = 0;
while (in.read(reinterpret_cast<char *>(&result), sizeof(float))) {
if ((i % (dimension - 1) == 0) && (i != 0)){
out << endl;
}
out << result << " ";
++i;
}
}
MPI_Finalize();
}
