#pragma once


#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

constexpr double divbyzero{0.000000001};

class Point {
public:
int label;
int id;
int cluster_label;
std::vector<double> coords;
static size_t dimensionality;
Point(int _label, std::vector<double> const& vd, int _id = -1): label{_label}, id{_id}, cluster_label{-1}, coords{vd} {
if (dimensionality == 0) [[unlikely]] {
dimensionality = coords.size();
} else if (dimensionality != coords.size()) [[likely]] {
throw std::invalid_argument("Invalid dimensionality.");
}
};
Point() {};

static void resetDimension() noexcept {dimensionality = 0;}
static void setDimension(size_t d) noexcept {dimensionality = d;}

auto begin() const noexcept {
return coords.begin();
}
auto end() const noexcept {
return coords.end();
}
auto cbegin() const noexcept {
return coords.cbegin();
}
auto cend() const noexcept {
return coords.cend();
}
size_t size() const noexcept {
return coords.size();
}
bool empty() const noexcept {
return coords.empty();
}
operator std::vector<double>() const noexcept {
return coords;
}
void resize(size_t d) {coords.resize(d);}
Point& operator+=(std::vector<double> const& rhs) {
for (int i = 0; i < dimensionality; i++) {
coords[i] += rhs[i];
}
return *this;
}
Point& operator-=(std::vector<double> const& rhs) {
for (int i = 0; i < dimensionality; i++) {
coords[i] -= rhs[i];
}
return *this;
}
Point& operator/=(std::vector<double> const& rhs) {
for (int i = 0; i < dimensionality; i++) {
double div = rhs[i];
if (div == 0) [[unlikely]] {
div = divbyzero;
}
coords[i] /= div;
}
return *this;
}
Point& operator*=(std::vector<double> const& rhs) {
for (int i = 0; i < dimensionality; i++) {
coords[i] *= rhs[i];
}
return *this;
}
Point& operator+=(double rhs) {
for (int i = 0; i < dimensionality; i++) {
coords[i] += rhs;
}
return *this;
}
Point& operator-=(double rhs) {
for (int i = 0; i < dimensionality; i++) {
coords[i] -= rhs;
}
return *this;
}
Point& operator/=(double rhs) {
for (int i = 0; i < dimensionality; i++) {
double div = rhs;
if (div == 0) [[unlikely]] {
div = divbyzero;
}
coords[i] /= div;
}
return *this;
}
Point& operator*=(double rhs) {
for (int i = 0; i < dimensionality; i++) {
coords[i] *= rhs;
}
return *this;
}
double operator[](int i) const{
return coords[i];
}
Point& operator=(Point rhs) noexcept {
std::swap(coords, rhs.coords);
label = rhs.label;
id = rhs.id;
cluster_label = rhs.cluster_label;
return *this;
}
void operator=(std::vector<double> rhs) noexcept {
std::swap(coords, rhs);
}
bool operator==(std::vector<double> const& rhs) const{
for (int i=0; i<dimensionality; i++)
if (coords[i] != rhs[i]) [[likely]]
return false;
return true;
}
bool operator!=(std::vector<double> const& rhs) const{
return !(*this == rhs);
}
bool operator<(std::vector<double> const& rhs) const {
if (dimensionality != rhs.size()) [[unlikely]] {
throw std::invalid_argument("Vectors must be of same size");
}
for (int i = 0; i < dimensionality; i++) {
if (coords[i] < rhs[i]) {
return true;
} else if (coords[i] > rhs[i]) {
return false;
}
}
return false;
}
friend std::ostream& operator<<(std::ostream& os, Point const& obj) {
for (auto &p : obj.coords)
os << p << " ";
return os;
}
double dist2(std::vector<double> const& rhs) const {
double res{0.0f};
for (int i=0; i<dimensionality; i++) res += pow(coords[i] - rhs[i], 2);
return res;
}
double dist(std::vector<double> const& rhs) const{
return sqrt(dist2(rhs));
}
};

