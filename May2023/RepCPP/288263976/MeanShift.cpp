
#include <vector>
#include <iostream>
#include <functional>   
#include <algorithm>
#include "Point.h"
#include "MeanShift.h"
#include "MeanShiftUtils.h"


Point MeanShift::updatePoint(Point &point, const std::vector<Point> &original_points, const std::string kernel_type="gaussian"){
std::vector<float> numerator;
for(int i=0; i<point.getDim(); i++){
numerator.push_back(0);
}

float denominator = .0;
for(auto& orig_point: original_points){
float distance = computeDistance(point, orig_point);
float w = computeKernel(distance, this->bandwidth, kernel_type);

for(int d=0; d<point.getDim(); d++){ 
numerator[d] += orig_point.getValues()[d] * w;
}
denominator+= w;
}
std::transform(numerator.begin(), numerator.end(), numerator.begin(),
bind2nd(std::divides<float>(), denominator)); 
return Point(numerator);
};

std::vector<Point> MeanShift::doMeanShift(const std::vector<Point> &points, const int num_threads){
std::vector<Point> copied_points = points;

float bandwidth = this->bandwidth;

for(int i=0; i<this->max_iter; i++){
#pragma omp parallel for default(none) shared(points, bandwidth, copied_points) num_threads(num_threads)
for(int c=0; c< points.size(); c++){
Point newPoint = updatePoint(copied_points[c], points);
copied_points[c] = newPoint;
}
}
return copied_points;
}

std::vector<Point> MeanShift::doSeqMeanShift(const std::vector<Point> &points){
std::vector<Point> copied_points = points;

float bandwidth = this->bandwidth;

for(int i=0; i<this->max_iter; i++){
for(int c=0; c< points.size(); c++){
Point newPoint = updatePoint(copied_points[c], points);
copied_points[c] = newPoint;
}
}
return copied_points;
}
