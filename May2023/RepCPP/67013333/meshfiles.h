#pragma once
#include "dmr.h"

void write_mesh(std::string infile, Mesh &mesh) {
FORD *nodex, *nodey;

nodex = mesh.nodex;
nodey = mesh.nodey;

unsigned slash = infile.rfind("/");
std::cout << "  -- " << infile.substr(slash + 1) + ".out.node (" << mesh.nnodes << " nodes)" << std::endl;
std::ofstream outfilenode((infile.substr(slash + 1) + ".out.node").c_str());
outfilenode.precision(17);
outfilenode << mesh.nnodes << " 2 0 0\n";
for (size_t ii = 0; ii < mesh.nnodes; ++ii) {
outfilenode << ii << " " << nodex[ii] << " " << nodey[ii] << "\n";
}
outfilenode.close();

uint3 *elements = mesh.elements;
volatile bool *isdel = mesh.isdel;

unsigned ntriangles2 = mesh.nelements;
unsigned segmentcnt = 0;
for (size_t ii = 0; ii < mesh.nelements; ++ii) {
if(IS_SEGMENT(elements[ii]) || isdel[ii])
ntriangles2--;
if(IS_SEGMENT(elements[ii]) && !isdel[ii])
segmentcnt++;
}

std::cout << "  -- " << infile.substr(slash + 1) + ".out.ele (" << ntriangles2 << " triangles)" << std::endl;
std::ofstream outfileele((infile.substr(slash + 1) + ".out.ele").c_str());

outfileele << ntriangles2 << " 3 0\n";
size_t kk = 0;
for (size_t ii = 0; ii < mesh.nelements; ++ii) {
if(!IS_SEGMENT(elements[ii]) && !isdel[ii])
outfileele << kk++ << " " << elements[ii].x << " " << elements[ii].y << " " << elements[ii].z << "\n";
}
outfileele.close();

std::cout << "  -- " << infile.substr(slash + 1) + ".out.poly (" << segmentcnt << " segments)" <<std::endl;
std::ofstream outfilepoly((infile.substr(slash + 1) + ".out.poly").c_str());
outfilepoly << "0 2 0 1\n";
outfilepoly << segmentcnt << " 0\n";
kk = 0;
for (size_t ii = 0; ii < mesh.nelements; ++ii) {
if(IS_SEGMENT(elements[ii]) && !isdel[ii])
outfilepoly << kk++ << " " << elements[ii].x << " " << elements[ii].y << "\n";
}
outfilepoly << "0\n";
outfilepoly.close();

std::cout << (ntriangles2 + segmentcnt) << " active elements of " << mesh.nelements << " total elements (" << mesh.nelements / (ntriangles2 + segmentcnt) << "x) " << std::endl;
std::cout << 1.0 * mesh.maxnelements / mesh.nelements << " ratio of used to free elements." << std::endl;
}

void next_line(std::ifstream& scanner) { 
scanner.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
} 

void readNodes(std::string filename, Mesh &mesh, int maxfactor = 2) {
size_t index;
FORD x, y;
bool firstindex = true;

std::ifstream scanner(filename.append(".node").c_str());
scanner >> mesh.nnodes;

mesh.maxnnodes = (maxfactor / MAX_NNODES_TO_NELEMENTS) * mesh.nnodes;
printf("memory for (%d) nodes: %d MB\n", 
mesh.maxnnodes, mesh.maxnnodes * sizeof(FORD) * 2 / 1048576);
mesh.nodex = (FORD *)malloc(mesh.maxnnodes*sizeof(FORD));
mesh.nodey = (FORD *)malloc(mesh.maxnnodes*sizeof(FORD));

FORD *nodex = mesh.nodex;
FORD *nodey = mesh.nodey;

for (size_t i = 0; i < mesh.nnodes; i++) {
next_line(scanner);
scanner >> index >> x >> y;
if(firstindex) { assert(index == 0); firstindex = false;}

nodex[index] = x;
nodey[index] = y;
} 
}   

void readTriangles(std::string basename, Mesh &mesh, int maxfactor = 2) {
unsigned ntriangles, nsegments;
unsigned i, index, n1, n2, n3;
bool firstindex = true;
std::string filename;

filename = basename;
std::ifstream scanner(filename.append(".ele").c_str());
scanner >> ntriangles;

filename = basename;
std::ifstream scannerperimeter(filename.append(".poly").c_str());
scannerperimeter >> nsegments; 
assert(nsegments == 0);  
next_line(scannerperimeter);
scannerperimeter >> nsegments;  

mesh.ntriangles = ntriangles;
mesh.nsegments = nsegments;
mesh.nelements = ntriangles + nsegments;
mesh.maxnelements = maxfactor * mesh.nelements;

printf("memory for elements: %d MB\n", mesh.maxnelements * (sizeof(uint3) * 2 + sizeof(bool) * 2) / 1048576);

mesh.alloc();

uint3 *elements = mesh.elements;
volatile bool *isdel = mesh.isdel;
bool *isbad = mesh.isbad;
for (i = 0; i < ntriangles; i++) {
next_line(scanner);
scanner >> index >> n1 >> n2 >> n3;
if(firstindex) { assert(index == 0); firstindex = false;}

elements[index].x = n1;
elements[index].y = n2;
elements[index].z = n3;
isdel[index] = isbad[index] = false;

}

firstindex = true;
for (i = 0; i < nsegments; i++) {
next_line(scannerperimeter);
scannerperimeter >> index >> n1 >> n2;
if(firstindex) { assert(index == 0); firstindex = false;}

elements[index + ntriangles].x = n1;
elements[index + ntriangles].y = n2;
elements[index + ntriangles].z = INVALIDID;
isdel[index] = isbad[index] = false;
}
}
