
#pragma once
#include "Point.h"
#include <random>

bool contatins(std::vector<Point>& points, Point&  p){
for(Point el: points){
if(el == p){
return true;
}
}
return false;
}

std::vector<Point> random_initializer (const std::vector<Point>& points, const int& k){
std::vector<Point> centroids;
std::random_device rand_dev;       
std::mt19937 gen(rand_dev());   
std::uniform_int_distribution<> distrib(0, points.size());
int i = 0;

while(centroids.size()<k){
Point new_centroid = points[distrib(gen)];
if(!contatins(centroids, new_centroid)){
new_centroid.cluster=i;
i++;
centroids.push_back(new_centroid);

}
}
return centroids;
}








