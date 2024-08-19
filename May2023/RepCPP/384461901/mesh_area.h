#pragma once

#include "typedef_vcg.h"


double mesh_area_total(MyMesh& m) {
FaceIterator face;
double area = 0.0;
for(face=m.face.begin(); face != m.face.end(); face++) {
if(!(*face).IsD()) {
area += DoubleArea(*face);
}
}
return(area/2.0);
}


std::vector<double> mesh_area_per_face(MyMesh& m) {
FaceIterator face;
std::vector<double> faceareas;
faceareas.resize(m.fn);
int faceind = 0;
for(face=m.face.begin(); face != m.face.end(); face++) {
if(!(*face).IsD()) {
faceareas[faceind] = DoubleArea(*face) / 2.0;
faceind++;
}
}
return(faceareas);
}