#pragma once

#include <iostream>
#include <climits>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <cassert>
#include <sstream>
#include <stdexcept>




#ifndef LIBFS_APPTAG
#define LIBFS_APPTAG "[libfs] "
#endif

#define LIBFS_DBG_WARNING

#ifdef LIBFS_DBG_NONE
#undef LIBFS_DBG_WARNING
#endif

#ifdef LIBFS_DBG_CRITICAL
#undef LIBFS_DBG_WARNING
#endif

#ifdef LIBFS_DBG_ERROR
#undef LIBFS_DBG_WARNING
#endif

#ifdef LIBFS_DBG_EXCESSIVE
#define LIBFS_DBG_VERBOSE
#endif

#ifdef LIBFS_DBG_VERBOSE
#define LIBFS_DBG_INFO
#endif

#ifdef LIBFS_DBG_INFO
#define LIBFS_DBG_IMPORTANT
#endif

#ifdef LIBFS_DBG_IMPORTANT
#define LIBFS_DBG_WARNING
#endif

#ifdef LIBFS_DBG_WARNING
#define LIBFS_DBG_ERROR
#endif

#ifdef LIBFS_DBG_ERROR
#define LIBFS_DBG_CRITICAL
#endif


namespace fs {

namespace util {
inline bool ends_with(std::string const & value, std::string const & suffix) {
if (suffix.size() > value.size()) return false;
return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

inline bool ends_with(std::string const & value, std::initializer_list<std::string> suffixes) {
for (auto suffix : suffixes) {
if (ends_with(value, suffix)) {
return true;
}
}
return false;
}

inline bool starts_with(std::string const & value, std::string const & prefix) {
if (prefix.length() > value.length()) return false;
return value.rfind(prefix, 0) == 0;
}

inline bool starts_with(std::string const & value, std::initializer_list<std::string> prefixes) {
for (auto prefix : prefixes) {
if (starts_with(value, prefix)) {
return true;
}
}
return false;
}

inline bool file_exists(const std::string& name) {
if (FILE *file = fopen(name.c_str(), "r")) {
fclose(file);
return true;
} else {
return false;
}
}


std::string fullpath( std::initializer_list<std::string> path_components, std::string path_sep = std::string("/") ) {
std::string fp;
if(path_components.size() == 0) {
throw std::invalid_argument("The 'path_components' must not be empty.");
}

std::string comp;
std::string comp_mod;
size_t idx = 0;
for(auto comp : path_components) {
comp_mod = comp;
if(idx != 0) { 
if (starts_with(comp, path_sep)) {
comp_mod = comp.substr(1, comp.size()-1);
}
}

if(ends_with(comp_mod, path_sep)) {
comp_mod = comp_mod.substr(0, comp_mod.size()-1);
}

fp += comp_mod;
if(idx < path_components.size()-1) {
fp += path_sep;
}
idx++;
}
return fp;
}

void str_to_file(const std::string& filename, const std::string rep) {
std::ofstream ofs;
ofs.open(filename, std::ofstream::out);
#ifdef LIBFS_DBG_VERBOSE
std::cout << LIBFS_APPTAG << "Opening file '" << filename << "' for writing.\n";
#endif
if(ofs.is_open()) {
ofs << rep;
ofs.close();
} else {
throw std::runtime_error("Unable to open file '" + filename + "' for writing.\n");
}
}
}



const int MRI_UCHAR = 0;

const int MRI_INT = 1;

const int MRI_FLOAT = 3;

const int MRI_SHORT = 4;

int _fread3(std::istream&);
template <typename T> T _freadt(std::istream&);
std::string _freadstringnewline(std::istream&);
std::string _freadfixedlengthstring(std::istream&, int32_t, bool);
bool _ends_with(std::string const &fullString, std::string const &ending);
size_t _vidx_2d(size_t, size_t, size_t);
struct MghHeader;

struct Mesh {

Mesh(std::vector<float> cvertices, std::vector<int32_t> cfaces) {
vertices = cvertices; faces = cfaces;
}

Mesh() {}

std::vector<float> vertices;  
std::vector<int32_t> faces;  

static fs::Mesh construct_cube() {
fs::Mesh mesh;
mesh.vertices = { 1.0, 1.0, 1.0,
1.0, 1.0, -1.0,
1.0, -1.0, 1.0,
1.0, -1.0, -1.0,
-1.0, 1.0, 1.0,
-1.0, 1.0, -1.0,
-1.0, -1.0, 1.0,
-1.0, -1.0, -1.0 };
mesh.faces = { 0, 2, 3,
3 ,1, 0,
4, 6, 7,
7, 5, 4,
0, 4, 5,
5, 1, 0,
2, 6, 7,
7, 3, 2,
0, 4, 6,
6, 2, 0,
1, 5, 7,
7, 3, 1 };
return mesh;
}

static fs::Mesh construct_pyramid() {
fs::Mesh mesh;
mesh.vertices = { 0.0, 0.0, 0.0, 
0.0, 1.0, 0.0,
1.0, 1.0, 0.0,
1.0, 0.0, 0.0,
0.5, 0.5, 1.0 }; 
mesh.faces = { 0, 2, 1, 
0, 3, 2,
0, 4, 1, 
1, 4, 2,
3, 2, 4,
0, 3, 4 };
return mesh;
}

static fs::Mesh construct_grid(const size_t nx = 4, const size_t ny = 5, const float distx = 1.0, const float disty = 1.0) {
if(nx < 2 || ny < 2) {
throw std::runtime_error("Parameters nx and ny must be at least 2.");
}
fs::Mesh mesh;
size_t num_vertices = nx * ny;
size_t num_faces = ((nx - 1) * (ny - 1)) * 2;
std::vector<float> vertices;
vertices.reserve(num_vertices * 3);
std::vector<int> faces;
faces.reserve(num_faces * 3);

float cur_x, cur_y, cur_z;
cur_x = cur_y = cur_z = 0.0;
for(size_t i = 0; i < nx; i++) {
for(size_t j = 0; j < ny; j++) {
vertices.push_back(cur_x);
vertices.push_back(cur_y);
vertices.push_back(cur_z);
cur_y += disty;
}
cur_x += distx;
}

for(size_t i = 0; i < num_vertices; i++) {
if((i+1) % ny == 0 || i >= num_vertices - ny) {
continue;
}
faces.push_back(i);
faces.push_back(i + ny + 1);
faces.push_back(i + 1);
faces.push_back(i);
faces.push_back(i + ny + 1);
faces.push_back(i + ny);
}

mesh.vertices = vertices;
mesh.faces = faces;
return mesh;
}


std::string to_obj() const {
std::stringstream objs;
for(size_t vidx=0; vidx<this->vertices.size(); vidx+=3) { 
objs << "v " << vertices[vidx] << " " << vertices[vidx+1] << " " << vertices[vidx+2] << "\n";
}
for(size_t fidx=0; fidx<this->faces.size(); fidx+=3) { 
objs << "f " << faces[fidx]+1 << " " << faces[fidx+1]+1 << " " << faces[fidx+2]+1 << "\n";
}
return(objs.str());
}


void to_obj_file(const std::string& filename) const {
fs::util::str_to_file(filename, this->to_obj());
}


static void from_obj(Mesh* mesh, std::istream* is) {
std::string line;
int line_idx = -1;

std::vector<float> vertices;
std::vector<int> faces;
size_t num_lines_ignored = 0; 

while (std::getline(*is, line)) {
line_idx += 1;
std::istringstream iss(line);
if(fs::util::starts_with(line, "#")) {
continue; 
} else {
if(fs::util::starts_with(line, "v ")) {
std::string elem_type_identifier; float x, y, z;
if (!(iss >> elem_type_identifier >> x >> y >> z)) {
throw std::domain_error("Could not parse vertex line " + std::to_string(line_idx+1) + " of OBJ data, invalid format.\n");
}
assert(elem_type_identifier == "v");
vertices.push_back(x);
vertices.push_back(y);
vertices.push_back(z);
} else if(fs::util::starts_with(line, "f ")) {
std::string elem_type_identifier, v0raw, v1raw, v2raw; int v0, v1, v2;
if (!(iss >> elem_type_identifier >> v0raw >> v1raw >> v2raw)) {
throw std::domain_error("Could not parse face line " + std::to_string(line_idx+1) + " of OBJ data, invalid format.\n");
}
assert(elem_type_identifier == "f");

std::size_t found_v0 = v0raw.find("/");
std::size_t found_v1 = v1raw.find("/");
std::size_t found_v2 = v2raw.find("/");
if (found_v0 != std::string::npos) {
v0raw = v0raw.substr(0, found_v0);
}
if (found_v1 != std::string::npos) {
v1raw = v1raw.substr(0, found_v1);
}
if (found_v2 != std::string::npos) {
v2raw = v0raw.substr(0, found_v2);
}
v0 = std::stoi(v0raw);
v1 = std::stoi(v1raw);
v2 = std::stoi(v2raw);

faces.push_back(v0 - 1);
faces.push_back(v1 - 1);
faces.push_back(v2 - 1);

} else {
num_lines_ignored++;
continue;
}

}
}
#ifdef LIBFS_DBG_INFO
if(num_lines_ignored > 0) {
std::cout << LIBFS_APPTAG << "Ignored " << num_lines_ignored << " lines in Wavefront OBJ format mesh file.\n";
}
#endif
mesh->vertices = vertices;
mesh->faces = faces;
}


static void from_obj(Mesh* mesh, const std::string& filename) {
std::ifstream input(filename);
if(input.is_open()) {
Mesh::from_obj(mesh, &input);
input.close();
} else {
throw std::runtime_error("Could not open Wavefront object format mesh file '" + filename + "' for reading.\n");
}
}


static void from_off(Mesh* mesh, std::istream* is, const std::string& source_filename="") {

std::string msg_source_file_part = source_filename.empty() ? "" : "'" + source_filename + "'";

std::string line;
int line_idx = -1;
int noncomment_line_idx = -1;

std::vector<float> vertices;
std::vector<int> faces;
size_t num_vertices, num_faces, num_edges;
size_t num_verts_parsed = 0;
size_t num_faces_parsed = 0;
float x, y, z;    
int num_verts_this_face, v0, v1, v2;   

while (std::getline(*is, line)) {
line_idx++;
std::istringstream iss(line);
if(fs::util::starts_with(line, "#")) {
continue; 
} else {
noncomment_line_idx++;
if(noncomment_line_idx == 0) {
std::string off_header_magic;
if (!(iss >> off_header_magic)) {
throw std::domain_error("Could not parse first header line " + std::to_string(line_idx+1) + " of OFF data, invalid format.\n");
}
if(!(off_header_magic == "OFF" || off_header_magic == "COFF")) {
throw std::domain_error("OFF magic string invalid, file " + msg_source_file_part + " not in OFF format.\n");
}
} else if (noncomment_line_idx == 1) {
if (!(iss >> num_vertices >> num_faces >> num_edges)) {
throw std::domain_error("Could not parse element count header line " + std::to_string(line_idx+1) + " of OFF data " + msg_source_file_part + ", invalid format.\n");
}
} else {

if(num_verts_parsed < num_vertices) {
if (!(iss >> x >> y >> z)) {
throw std::domain_error("Could not parse vertex coordinate line " + std::to_string(line_idx+1) + " of OFF data " + msg_source_file_part + ", invalid format.\n");
}
vertices.push_back(x);
vertices.push_back(y);
vertices.push_back(z);
num_verts_parsed++;
} else {
if(num_faces_parsed < num_faces) {
if (!(iss >> num_verts_this_face >> v0 >> v1 >> v2)) {
throw std::domain_error("Could not parse face line " + std::to_string(line_idx+1) + " of OFF data " + msg_source_file_part + ", invalid format.\n");
}
if(num_verts_this_face != 3) {
throw std::domain_error("At OFF data " + msg_source_file_part + " line " + std::to_string(line_idx+1) + ": only triangular meshes supported.\n");
}
faces.push_back(v0);
faces.push_back(v1);
faces.push_back(v2);
num_faces_parsed++;
}
}
}
}
}
if(num_verts_parsed < num_vertices) {
throw std::domain_error("Vertex count mismatch between OFF data " + msg_source_file_part + " header (" + std::to_string(num_vertices) + ") and data (" + std::to_string(num_verts_parsed) + ").\n");
}
if(num_faces_parsed < num_faces) {
throw std::domain_error("Face count mismatch between OFF data " + msg_source_file_part + " header  (" + std::to_string(num_faces) + ") and data (" + std::to_string(num_faces_parsed) + ").\n");
}
mesh->vertices = vertices;
mesh->faces = faces;
}


static void from_off(Mesh* mesh, const std::string& filename) {
std::ifstream input(filename);
if(input.is_open()) {
Mesh::from_off(mesh, &input);
input.close();
} else {
throw std::runtime_error("Could not open Object file format (OFF) mesh file '" + filename + "' for reading.\n");
}
}


static void from_ply(Mesh* mesh, std::istream* is) {
std::string line;
int line_idx = -1;
int noncomment_line_idx = -1;

std::vector<float> vertices;
std::vector<int> faces;

bool in_header = true; 
int num_verts = -1;
int num_faces = -1;
while (std::getline(*is, line)) {
line_idx += 1;
std::istringstream iss(line);
if(fs::util::starts_with(line, "comment")) {
continue; 
} else {
noncomment_line_idx++;
if(in_header) {
if(noncomment_line_idx == 0) {
if(line != "ply") throw std::domain_error("Invalid PLY file");
} else if(noncomment_line_idx == 1) {
if(line != "format ascii 1.0") throw std::domain_error("Unsupported PLY file format, only format 'format ascii 1.0' is supported.");
}

if(line == "end_header") {
in_header = false;
} else if(fs::util::starts_with(line, "element vertex")) {
std::string elem, elem_type_identifier;
if (!(iss >> elem >> elem_type_identifier >> num_verts)) {
throw std::domain_error("Could not parse element vertex line of PLY header, invalid format.\n");
}
} else if(fs::util::starts_with(line, "element face")) {
std::string elem, elem_type_identifier;
if (!(iss >> elem >> elem_type_identifier >> num_faces)) {
throw std::domain_error("Could not parse element face line of PLY header, invalid format.\n");
}
} 

} else {  
if(num_verts < 1 || num_faces < 1) {
throw std::domain_error("Invalid PLY file: missing element count lines of header.");
}
if(vertices.size() < (size_t)num_verts * 3) {
float x,y,z;
if (!(iss >> x >> y >> z)) {
throw std::domain_error("Could not parse vertex line of PLY data, invalid format.\n");
}
vertices.push_back(x);
vertices.push_back(y);
vertices.push_back(z);
} else {
if(faces.size() < (size_t)num_faces * 3) {
int verts_per_face, v0, v1, v2;
if (!(iss >> verts_per_face >> v0 >> v1 >> v2)) {
throw std::domain_error("Could not parse face line of PLY data, invalid format.\n");
}
if(verts_per_face != 3) {
throw std::domain_error("Only triangular meshes are supported: PLY faces lines must contain exactly 3 vertex indices.\n");
}
faces.push_back(v0);
faces.push_back(v1);
faces.push_back(v2);
}
}
}
}
}
if(vertices.size() != (size_t)num_verts * 3) {
std::cerr << "PLY header mentions " << num_verts << " vertices, but found " << vertices.size() / 3 << ".\n";
}
if(faces.size() != (size_t)num_faces * 3) {
std::cerr << "PLY header mentions " << num_faces << " faces, but found " << faces.size() / 3 << ".\n";
}
mesh->vertices = vertices;
mesh->faces = faces;
}

static void from_ply(Mesh* mesh, const std::string& filename) {
std::ifstream input(filename);
if(input.is_open()) {
Mesh::from_ply(mesh, &input);
input.close();
} else {
throw std::runtime_error("Could not open Stanford PLY format mesh file '" + filename + "' for reading.\n");
}
}


size_t num_vertices() const {
return(this->vertices.size() / 3);
}

size_t num_faces() const {
return(this->faces.size() / 3);
}

const int32_t& fm_at(const size_t i, const size_t j) const {
size_t idx = _vidx_2d(i, j, 3);
if(idx > this->faces.size()-1) {
throw std::range_error("Indices (" + std::to_string(i) + "," + std::to_string(j) + ") into Mesh.faces out of bounds. Hit " + std::to_string(idx) + " with max valid index " + std::to_string(this->faces.size()-1) + ".\n");
}
return(this->faces[idx]);
}


std::vector<int32_t> face_vertices(const size_t face) const {
if(face > this->num_faces()-1) {
throw std::range_error("Index " + std::to_string(face) + " into Mesh.faces out of bounds, max valid index is " + std::to_string(this->num_faces()-1) + ".\n");
}
std::vector<int32_t> fv(3);
fv[0] = this->fm_at(face, 0);
fv[1] = this->fm_at(face, 1);
fv[2] = this->fm_at(face, 2);
return(fv);
}

std::vector<float> vertex_coords(const size_t vertex) const {
if(vertex > this->num_vertices()-1) {
throw std::range_error("Index " + std::to_string(vertex) + " into Mesh.vertices out of bounds, max valid index is " + std::to_string(this->num_vertices()-1) + ".\n");
}
std::vector<float> vc(3);
vc[0] = this->vm_at(vertex, 0);
vc[1] = this->vm_at(vertex, 1);
vc[2] = this->vm_at(vertex, 2);
return(vc);
}

const float& vm_at(const size_t i, const size_t j) const {
size_t idx = _vidx_2d(i, j, 3);
if(idx > this->vertices.size()-1) {
throw std::range_error("Indices (" + std::to_string(i) + "," + std::to_string(j) + ") into Mesh.vertices out of bounds. Hit " + std::to_string(idx) + " with max valid index " + std::to_string(this->vertices.size()-1) + ".\n");
}
return(this->vertices[idx]);
}

std::string to_ply() const {
std::vector<uint8_t> empty_col;
return(this->to_ply(empty_col));
}

std::string to_ply(const std::vector<uint8_t> col) const {
bool use_vertex_colors = col.size() != 0;
std::stringstream plys;
plys << "ply\nformat ascii 1.0\n";
plys << "element vertex " << this->num_vertices() << "\n";
plys << "property float x\nproperty float y\nproperty float z\n";
if(use_vertex_colors) {
if(col.size() != this->vertices.size()) {
throw std::invalid_argument("Number of vertex coordinates and vertex colors must match when writing PLY file.");
}
plys << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
}
plys << "element face " << this->num_faces() << "\n";
plys << "property list uchar int vertex_index\n";
plys << "end_header\n";

for(size_t vidx=0; vidx<this->vertices.size();vidx+=3) {  
plys << vertices[vidx] << " " << vertices[vidx+1] << " " << vertices[vidx+2];
if(use_vertex_colors) {
plys << " " << (int)col[vidx] << " " << (int)col[vidx+1] << " " << (int)col[vidx+2];
}
plys << "\n";
}

const int num_vertices_per_face = 3;
for(size_t fidx=0; fidx<this->faces.size();fidx+=3) { 
plys << num_vertices_per_face << " " << faces[fidx] << " " << faces[fidx+1] << " " << faces[fidx+2] << "\n";
}
return(plys.str());
}

void to_ply_file(const std::string& filename) const {
fs::util::str_to_file(filename, this->to_ply());
}

void to_ply_file(const std::string& filename, const std::vector<uint8_t> col) const {
fs::util::str_to_file(filename, this->to_ply(col));
}

std::string to_off() const {
std::vector<uint8_t> empty_col;
return(this->to_off(empty_col));
}

std::string to_off(const std::vector<uint8_t> col) const {
bool use_vertex_colors = col.size() != 0;
std::stringstream offs;
if(use_vertex_colors) {
if(col.size() != this->vertices.size()) {
throw std::invalid_argument("Number of vertex coordinates and vertex colors must match when writing OFF file.");
}
offs << "COFF\n";
} else {
offs << "OFF\n";
}
offs << this->num_vertices() << " " << this->num_faces() << " 0\n";

for(size_t vidx=0; vidx<this->vertices.size();vidx+=3) {  
offs << vertices[vidx] << " " << vertices[vidx+1] << " " << vertices[vidx+2];
if(use_vertex_colors) {
offs << " " << (int)col[vidx] << " " << (int)col[vidx+1] << " " << (int)col[vidx+2] << " 255";
}
offs << "\n";
}

const int num_vertices_per_face = 3;
for(size_t fidx=0; fidx<this->faces.size();fidx+=3) { 
offs << num_vertices_per_face << " " << faces[fidx] << " " << faces[fidx+1] << " " << faces[fidx+2] << "\n";
}
return(offs.str());
}

void to_off_file(const std::string& filename) const {
fs::util::str_to_file(filename, this->to_off());
}

void to_off_file(const std::string& filename, const std::vector<uint8_t> col) const {
fs::util::str_to_file(filename, this->to_off(col));
}
};


struct Curv {

Curv(std::vector<float> curv_data) :
num_faces(100000), num_vertices(0), num_values_per_vertex(1) { data = curv_data; num_vertices = data.size(); }

Curv() :
num_faces(100000), num_vertices(0), num_values_per_vertex(1) {}

int32_t num_faces;

std::vector<float> data;

int32_t num_vertices;

int32_t num_values_per_vertex;
};

struct Colortable {
std::vector<int32_t> id;  
std::vector<std::string> name;   
std::vector<int32_t> r;   
std::vector<int32_t> g;   
std::vector<int32_t> b;   
std::vector<int32_t> a;   
std::vector<int32_t> label;   

size_t num_entries() const {
size_t num_ids = this->id.size();
if(this->name.size() != num_ids || this->r.size() != num_ids || this->g.size() != num_ids || this->b.size() != num_ids || this->a.size() != num_ids || this->label.size() != num_ids) {
std::cerr << "Inconsistent Colortable, vector sizes do not match.\n";
}
return num_ids;
}

int32_t get_region_idx(const std::string& query_name) const {
for(size_t i = 0; i<this->num_entries(); i++) {
if(this->name[i] == query_name) {
return (int32_t)i;
}
}
return(-1);
}

int32_t get_region_idx(int32_t query_label) const {
for(size_t i = 0; i<this->num_entries(); i++) {
if(this->label[i] == query_label) {
return (int32_t)i;
}
}
return(-1);
}

};


struct Annot {
std::vector<int32_t> vertex_indices;  
std::vector<int32_t> vertex_labels;   
Colortable colortable;  

std::vector<int32_t> region_vertices(const std::string& region_name) const {
int32_t region_idx = this->colortable.get_region_idx(region_name);
if(region_idx >= 0) {
return(this->region_vertices(this->colortable.label[region_idx]));
} else {
std::cerr << "No such region in annot, returning empty vector.\n";
std::vector<int32_t> empty;
return(empty);
}
}

std::vector<int32_t> region_vertices(int32_t region_label) const {
std::vector<int32_t> reg_verts;
for(size_t i=0; i<this->vertex_labels.size(); i++) {
if(this->vertex_labels[i] == region_label) {
reg_verts.push_back(i);
}
}
return(reg_verts);
}

std::vector<uint8_t> vertex_colors(bool alpha = false) const {
int num_channels = alpha ? 4 : 3;
std::vector<uint8_t> col;
col.reserve(this->num_vertices() * num_channels);
std::vector<size_t> vertex_region_indices = this->vertex_regions();
for(size_t i=0; i<this->num_vertices(); i++) {
col.push_back(this->colortable.r[vertex_region_indices[i]]);
col.push_back(this->colortable.g[vertex_region_indices[i]]);
col.push_back(this->colortable.b[vertex_region_indices[i]]);
if(alpha) {
col.push_back(this->colortable.a[vertex_region_indices[i]]);
}
}
return(col);
}

size_t num_vertices() const {
size_t nv = this->vertex_indices.size();
if(this->vertex_labels.size() != nv) {
throw std::runtime_error("Inconsistent annot, number of vertex indices and labels does not match.\n");
}
return nv;
}

std::vector<size_t> vertex_regions() const {
std::vector<size_t> vert_reg;
for(size_t i=0; i<this->num_vertices(); i++) {
vert_reg.push_back(0);  
}
for(size_t region_idx=0; region_idx<this->colortable.num_entries(); region_idx++) {
std::vector<int32_t> reg_vertices = this->region_vertices(this->colortable.label[region_idx]);
for(size_t region_vert_local_idx=0;  region_vert_local_idx<reg_vertices.size(); region_vert_local_idx++) {
int32_t region_vert_idx = reg_vertices[region_vert_local_idx];
vert_reg[region_vert_idx] = region_idx;
}
}
return vert_reg;
}

std::vector<std::string> vertex_region_names() const {
std::vector<std::string> region_names;
std::vector<size_t> vertex_region_indices = this->vertex_regions();
for(size_t i=0; i<this->num_vertices(); i++) {
region_names.push_back(this->colortable.name[vertex_region_indices[i]]);
}
return(region_names);
}
};


struct MghHeader {
int32_t dim1length;  
int32_t dim2length;  
int32_t dim3length;  
int32_t dim4length;  

int32_t dtype;  
int32_t dof;  
int16_t ras_good_flag; 

size_t num_values() const {
return((size_t) dim1length * dim2length * dim3length * dim4length);
}

float xsize;  
float ysize;  
float zsize;  
std::vector<float> Mdc;  
std::vector<float> Pxyz_c;  
};

struct MghData {
MghData() {}
MghData(std::vector<int32_t> curv_data) { data_mri_int = curv_data; }  
explicit MghData(std::vector<uint8_t> curv_data) { data_mri_uchar = curv_data; }  
explicit MghData(std::vector<short> curv_data) { data_mri_short = curv_data; }  
MghData(std::vector<float> curv_data) { data_mri_float = curv_data; }  
std::vector<int32_t> data_mri_int;  
std::vector<uint8_t> data_mri_uchar;  
std::vector<float> data_mri_float;  
std::vector<short> data_mri_short;  
};

struct Mgh {
MghHeader header;  
MghData data;  
};

template<class T>
struct Array4D {
Array4D(unsigned int d1, unsigned int d2, unsigned int d3, unsigned int d4) :
d1(d1), d2(d2), d3(d3), d4(d4), data(d1*d2*d3*d4) {}

Array4D(MghHeader *mgh_header) :
d1(mgh_header->dim1length), d2(mgh_header->dim2length), d3(mgh_header->dim3length), d4(mgh_header->dim4length), data(d1*d2*d3*d4) {}

Array4D(Mgh *mgh) : 
d1(mgh->header.dim1length), d2(mgh->header.dim2length), d3(mgh->header.dim3length), d4(mgh->header.dim4length), data(d1*d2*d3*d4) {}

const T& at(const unsigned int i1, const unsigned int i2, const unsigned int i3, const unsigned int i4) const {
return data[get_index(i1, i2, i3, i4)];
}

unsigned int get_index(const unsigned int i1, const unsigned int i2, const unsigned int i3, const unsigned int i4) const {
assert(i1 >= 0 && i1 < d1);
assert(i2 >= 0 && i2 < d2);
assert(i3 >= 0 && i3 < d3);
assert(i4 >= 0 && i4 < d4);
return (((i1*d2 + i2)*d3 + i3)*d4 + i4);
}

unsigned int num_values() const {
return(d1*d2*d3*d4);
}

unsigned int d1;  
unsigned int d2;  
unsigned int d3;  
unsigned int d4;  
std::vector<T> data;  
};

void read_mgh_header(MghHeader*, const std::string&);
void read_mgh_header(MghHeader*, std::istream*);
template <typename T> std::vector<T> _read_mgh_data(MghHeader*, const std::string&);
template <typename T> std::vector<T> _read_mgh_data(MghHeader*, std::istream*);
std::vector<int32_t> _read_mgh_data_int(MghHeader*, const std::string&);
std::vector<int32_t> _read_mgh_data_int(MghHeader*, std::istream*);
std::vector<uint8_t> _read_mgh_data_uchar(MghHeader*, const std::string&);
std::vector<uint8_t> _read_mgh_data_uchar(MghHeader*, std::istream*);
std::vector<short> _read_mgh_data_short(MghHeader*, const std::string&);
std::vector<short> _read_mgh_data_short(MghHeader*, std::istream*);
std::vector<float> _read_mgh_data_float(MghHeader*, const std::string&);
std::vector<float> _read_mgh_data_float(MghHeader*, std::istream*);


void read_mgh(Mgh* mgh, const std::string& filename) {
MghHeader mgh_header;
read_mgh_header(&mgh_header, filename);
mgh->header = mgh_header;
if(mgh->header.dtype == MRI_INT) {
std::vector<int32_t> data = _read_mgh_data_int(&mgh_header, filename);
mgh->data.data_mri_int = data;
} else if(mgh->header.dtype == MRI_UCHAR) {
std::vector<uint8_t> data = _read_mgh_data_uchar(&mgh_header, filename);
mgh->data.data_mri_uchar = data;
} else if(mgh->header.dtype == MRI_FLOAT) {
std::vector<float> data = _read_mgh_data_float(&mgh_header, filename);
mgh->data.data_mri_float = data;
} else if(mgh->header.dtype == MRI_SHORT) {
std::vector<short> data = _read_mgh_data_short(&mgh_header, filename);
mgh->data.data_mri_short = data;
} else {
#ifdef LIBFS_DBG_INFO
if(fs::util::ends_with(filename, ".mgz")) {
std::cout << LIBFS_APPTAG << "Note: your MGH filename ends with '.mgz'. Keep in mind that MGZ format is not supported directly. You can ignore this message if you wrapped a gz stream.\n";
}
#endif
throw std::runtime_error("Not reading MGH data from file '" + filename + "', data type " + std::to_string(mgh->header.dtype) + " not supported yet.\n");
}
}

std::vector<std::string> read_subjectsfile(const std::string& filename) {
std::vector<std::string> subjects;
std::ifstream input(filename);
std::string line;

if(! input.is_open()) {
throw std::runtime_error("Could not open subjects file '" + filename + "'.\n");
}

while( std::getline( input, line ) ) {
subjects.push_back(line);
}
return(subjects);
}

void read_mgh(Mgh* mgh, std::istream* is) {
MghHeader mgh_header;
read_mgh_header(&mgh_header, is);
mgh->header = mgh_header;
if(mgh->header.dtype == MRI_INT) {
std::vector<int32_t> data = _read_mgh_data_int(&mgh_header, is);
mgh->data.data_mri_int = data;
} else if(mgh->header.dtype == MRI_UCHAR) {
std::vector<uint8_t> data = _read_mgh_data_uchar(&mgh_header, is);
mgh->data.data_mri_uchar = data;
} else if(mgh->header.dtype == MRI_FLOAT) {
std::vector<float> data = _read_mgh_data_float(&mgh_header, is);
mgh->data.data_mri_float = data;
} else if(mgh->header.dtype == MRI_SHORT) {
std::vector<short> data = _read_mgh_data_short(&mgh_header, is);
mgh->data.data_mri_short = data;
} else {
throw std::runtime_error("Not reading data from MGH stream, data type " + std::to_string(mgh->header.dtype) + " not supported yet.\n");
}
}

void read_mgh_header(MghHeader* mgh_header, std::istream* is) {
const int MGH_VERSION = 1;

int format_version = _freadt<int32_t>(*is);
if(format_version != MGH_VERSION) {
throw std::runtime_error("Invalid MGH file or unsupported file format version: expected version " + std::to_string(MGH_VERSION) + ", found " + std::to_string(format_version) + ".\n");
}
mgh_header->dim1length =  _freadt<int32_t>(*is);
mgh_header->dim2length =  _freadt<int32_t>(*is);
mgh_header->dim3length =  _freadt<int32_t>(*is);
mgh_header->dim4length =  _freadt<int32_t>(*is);

mgh_header->dtype =  _freadt<int32_t>(*is);
mgh_header->dof =  _freadt<int32_t>(*is);

int unused_header_space_size_left = 256;  
mgh_header->ras_good_flag =  _freadt<int16_t>(*is);
unused_header_space_size_left -= 2; 

if(mgh_header->ras_good_flag == 1) {
mgh_header->xsize =  _freadt<float>(*is);
mgh_header->ysize =  _freadt<float>(*is);
mgh_header->zsize =  _freadt<float>(*is);

for(int i=0; i<9; i++) {
mgh_header->Mdc.push_back( _freadt<float>(*is));
}
for(int i=0; i<3; i++) {
mgh_header->Pxyz_c.push_back( _freadt<float>(*is));
}
unused_header_space_size_left -= 60;
}

uint8_t discarded;
while(unused_header_space_size_left > 0) {
discarded = _freadt<uint8_t>(*is);
unused_header_space_size_left -= 1;
}
(void)discarded; 
}

std::vector<int32_t> _read_mgh_data_int(MghHeader* mgh_header, const std::string& filename) {
if(mgh_header->dtype != MRI_INT) {
std::cerr << "Expected MRI data type " << MRI_INT << ", but found " << mgh_header->dtype << ".\n";
}
return(_read_mgh_data<int32_t>(mgh_header, filename));
}

std::vector<int32_t> _read_mgh_data_int(MghHeader* mgh_header, std::istream* is) {
if(mgh_header->dtype != MRI_INT) {
std::cerr << "Expected MRI data type " << MRI_INT << ", but found " << mgh_header->dtype << ".\n";
}
return(_read_mgh_data<int32_t>(mgh_header, is));
}

std::vector<short> _read_mgh_data_short(MghHeader* mgh_header, const std::string& filename) {
if(mgh_header->dtype != MRI_SHORT) {
std::cerr << "Expected MRI data type " << MRI_SHORT << ", but found " << mgh_header->dtype << ".\n";
}
return(_read_mgh_data<short>(mgh_header, filename));
}

std::vector<short> _read_mgh_data_short(MghHeader* mgh_header, std::istream* is) {
if(mgh_header->dtype != MRI_SHORT) {
std::cerr << "Expected MRI data type " << MRI_SHORT << ", but found " << mgh_header->dtype << ".\n";
}
return(_read_mgh_data<short>(mgh_header, is));
}


void read_mgh_header(MghHeader* mgh_header, const std::string& filename) {
std::ifstream ifs;
ifs.open(filename, std::ios_base::in | std::ios::binary);
if(ifs.is_open()) {
read_mgh_header(mgh_header, &ifs);
ifs.close();
} else {
throw std::runtime_error("Unable to open MGH file '" + filename + "'.\n");
}
}


template <typename T>
std::vector<T> _read_mgh_data(MghHeader* mgh_header, const std::string& filename) {
std::ifstream ifs;
ifs.open(filename, std::ios_base::in | std::ios::binary);
if(ifs.is_open()) {
ifs.seekg(284, ifs.beg); 

int num_values = mgh_header->num_values();
std::vector<T> data;
for(int i=0; i<num_values; i++) {
data.push_back( _freadt<T>(ifs));
}
ifs.close();
return(data);
} else {
throw std::runtime_error("Unable to open MGH file '" + filename + "'.\n");
}
}


template <typename T>
std::vector<T> _read_mgh_data(MghHeader* mgh_header, std::istream* is) {
int num_values = mgh_header->num_values();
std::vector<T> data;
for(int i=0; i<num_values; i++) {
data.push_back( _freadt<T>(*is));
}
return(data);
}


std::vector<float> _read_mgh_data_float(MghHeader* mgh_header, const std::string& filename) {
if(mgh_header->dtype != MRI_FLOAT) {
std::cerr << "Expected MRI data type " << MRI_FLOAT << ", but found " << mgh_header->dtype << ".\n";
}
return(_read_mgh_data<float>(mgh_header, filename));
}

std::vector<float> _read_mgh_data_float(MghHeader* mgh_header, std::istream* is) {
if(mgh_header->dtype != MRI_FLOAT) {
std::cerr << "Expected MRI data type " << MRI_FLOAT << ", but found " << mgh_header->dtype << ".\n";
}
return(_read_mgh_data<float>(mgh_header, is));
}

std::vector<uint8_t> _read_mgh_data_uchar(MghHeader* mgh_header, const std::string& filename) {
if(mgh_header->dtype != MRI_UCHAR) {
std::cerr << "Expected MRI data type " << MRI_UCHAR << ", but found " << mgh_header->dtype << ".\n";
}
return(_read_mgh_data<uint8_t>(mgh_header, filename));
}

std::vector<uint8_t> _read_mgh_data_uchar(MghHeader* mgh_header, std::istream* is) {
if(mgh_header->dtype != MRI_UCHAR) {
std::cerr << "Expected MRI data type " << MRI_UCHAR << ", but found " << mgh_header->dtype << ".\n";
}
return(_read_mgh_data<uint8_t>(mgh_header, is));
}

void read_surf(Mesh* surface, const std::string& filename) {
const int SURF_TRIS_MAGIC = 16777214;
std::ifstream is;
is.open(filename, std::ios_base::in | std::ios::binary);
if(is.is_open()) {
int magic = _fread3(is);
if(magic != SURF_TRIS_MAGIC) {
throw std::domain_error("Surf file '" + filename + "' magic code in header did not match: expected " + std::to_string(SURF_TRIS_MAGIC) + ", found " + std::to_string(magic) + ".\n");
}
std::string created_line = _freadstringnewline(is);
std::string comment_line = _freadstringnewline(is);
int num_verts =  _freadt<int32_t>(is);
int num_faces =  _freadt<int32_t>(is);
#ifdef LIBFS_DBG_INFO
std::cout << LIBFS_APPTAG << "Read surface file with " << num_verts << " vertices, " << num_faces << " faces.\n";
#endif
std::vector<float> vdata;
for(int i=0; i<(num_verts*3); i++) {
vdata.push_back( _freadt<float>(is));
}
std::vector<int> fdata;
for(int i=0; i<(num_faces*3); i++) {
fdata.push_back( _freadt<int32_t>(is));
}
is.close();
surface->vertices = vdata;
surface->faces = fdata;
} else {
throw std::runtime_error("Unable to open surface file '" + filename + "'.\n");
}
}


void read_mesh(Mesh* surface, const std::string& filename) {
if(fs::util::ends_with(filename, ".obj")) {
fs::Mesh::from_obj(surface, filename);
} else if(fs::util::ends_with(filename, ".ply")) {
fs::Mesh::from_ply(surface, filename);
} else if(fs::util::ends_with(filename, ".off")) {
fs::Mesh::from_off(surface, filename);
} else {
read_surf(surface, filename);
}
}


bool _is_bigendian() {
short int number = 0x1;
char *numPtr = (char*)&number;
return (numPtr[0] != 1);
}

void read_curv(Curv* curv, std::istream *is, const std::string& source_filename="") {
const std::string msg_source_file_part = source_filename.empty() ? "" : "'" + source_filename + "' ";
const int CURV_MAGIC = 16777215;
int magic = _fread3(*is);
if(magic != CURV_MAGIC) {
throw std::domain_error("Curv file " + msg_source_file_part + "header magic did not match: expected " + std::to_string(CURV_MAGIC) + ", found " + std::to_string(magic) + ".\n");
}
curv->num_vertices = _freadt<int32_t>(*is);
curv->num_faces =  _freadt<int32_t>(*is);
curv->num_values_per_vertex = _freadt<int32_t>(*is);
#ifdef LIBFS_DBG_INFO
std::cout << LIBFS_APPTAG << "Read curv file with " << curv->num_vertices << " vertices, " << curv->num_faces << " faces and " << curv->num_values_per_vertex << " values per vertex.\n";
#endif
if(curv->num_values_per_vertex != 1) { 
throw std::domain_error("Curv file " + msg_source_file_part + "must contain exactly 1 value per vertex, found " + std::to_string(curv->num_values_per_vertex) + ".\n");
}
std::vector<float> data;
for(int i=0; i<curv->num_vertices; i++) {
data.push_back( _freadt<float>(*is));
}
curv->data = data;
}


void read_curv(Curv* curv, const std::string& filename) {
std::ifstream is(filename);
if(is.is_open()) {
read_curv(curv, &is, filename);
is.close();
} else {
throw std::runtime_error("Could not open curv file '" + filename + "' for reading.\n");
}
}

void _read_annot_colortable(Colortable* colortable, std::istream *is, int32_t num_entries) {
int32_t num_chars_orig_filename = _freadt<int32_t>(*is);  

uint8_t discarded;
for(int32_t i=0; i<num_chars_orig_filename; i++) {
discarded = _freadt<uint8_t>(*is);
}
(void)discarded; 

int32_t num_entries_duplicated = _freadt<int32_t>(*is); 
if(num_entries != num_entries_duplicated) {
std::cerr << "Warning: the two num_entries header fields of this annotation do not match. Use with care.\n";
}

int32_t entry_num_chars;
for(int32_t i=0; i<num_entries; i++) {
colortable->id.push_back(_freadt<int32_t>(*is));
entry_num_chars = _freadt<int32_t>(*is);
colortable->name.push_back(_freadfixedlengthstring(*is, entry_num_chars, true));
colortable->r.push_back(_freadt<int32_t>(*is));
colortable->g.push_back(_freadt<int32_t>(*is));
colortable->b.push_back(_freadt<int32_t>(*is));
colortable->a.push_back(_freadt<int32_t>(*is));
colortable->label.push_back(colortable->r[i] + colortable->g[i]*256 + colortable->b[i]*65536 + colortable->a[i]*16777216);
}

}

size_t _vidx_2d(size_t row, size_t column, size_t row_length=3) {
return (row+1)*row_length -row_length + column;
}

void read_annot(Annot* annot, std::istream *is) {

int32_t num_vertices = _freadt<int32_t>(*is);
std::vector<int32_t> vertices;
std::vector<int32_t> labels;
for(int32_t i=0; i<(num_vertices*2); i++) { 
if(i % 2 == 0) {
vertices.push_back(_freadt<int32_t>(*is));
} else {
labels.push_back(_freadt<int32_t>(*is));
}
}
annot->vertex_indices = vertices;
annot->vertex_labels = labels;
int32_t has_colortable = _freadt<int32_t>(*is);
if(has_colortable == 1) {
int32_t num_colortable_entries_old_format = _freadt<int32_t>(*is);
if(num_colortable_entries_old_format > 0) {
throw std::domain_error("Reading annotation in old format not supported. Please open an issue and supply an example file if you need this.\n");
} else {
int32_t colortable_format_version = -num_colortable_entries_old_format; 
if(colortable_format_version == 2) {
int32_t num_colortable_entries = _freadt<int32_t>(*is); 
_read_annot_colortable(&annot->colortable, is, num_colortable_entries);
} else {
throw std::domain_error("Reading annotation in new format version !=2 not supported. Please open an issue and supply an example file if you need this.\n");
}

}

} else {
throw std::domain_error("Reading annotation without colortable not supported. Maybe invalid annotation file?\n");
}
}


void read_annot(Annot* annot, const std::string& filename) {
std::ifstream is(filename);
if(is.is_open()) {
read_annot(annot, &is);
is.close();
} else {
throw std::runtime_error("Could not open annot file '" + filename + "' for reading.\n");
}
}


std::vector<float> read_curv_data(const std::string& filename) {
Curv curv;
read_curv(&curv, filename);
return(curv.data);
}

template <typename T>
T _swap_endian(T u) {
static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

union
{
T u;
unsigned char u8[sizeof(T)];
} source, dest;

source.u = u;

for (size_t k = 0; k < sizeof(T); k++)
dest.u8[k] = source.u8[sizeof(T) - k - 1];

return(dest.u);
}

template <typename T>
T _freadt(std::istream& is) {
T t;
is.read(reinterpret_cast<char*>(&t), sizeof(t));
if(! _is_bigendian()) {
t = _swap_endian<T>(t);
}
return(t);
}

int _fread3(std::istream& is) {
uint32_t i;
is.read(reinterpret_cast<char*>(&i), 3);
if(! _is_bigendian()) {
i = _swap_endian<std::uint32_t>(i);
}
i = ((i >> 8) & 0xffffff);
return(i);
}

template <typename T>
void _fwritet(std::ostream& os, T t) {
if(! _is_bigendian()) {
t = _swap_endian<T>(t);
}
os.write( reinterpret_cast<const char*>( &t ), sizeof(t));
}


void _fwritei3(std::ostream& os, uint32_t i) {
unsigned char b1 = ( i >> 16) & 255;
unsigned char b2 = ( i >> 8) & 255;
unsigned char b3 =  i & 255;

if(!_is_bigendian()) {
b1 = _swap_endian<unsigned char>(b1);
b2 = _swap_endian<unsigned char>(b2);
b3 = _swap_endian<unsigned char>(b3);
}

os.write( reinterpret_cast<const char*>( &b1 ), sizeof(b1));
os.write( reinterpret_cast<const char*>( &b2 ), sizeof(b2));
os.write( reinterpret_cast<const char*>( &b3 ), sizeof(b3));
}

std::string _freadstringnewline(std::istream &is) {
std::string s;
std::getline(is, s, '\n');
return s;
}

std::string _freadfixedlengthstring(std::istream &is, int32_t length, bool strip_last_char=true) {
if(length <= 0) {
throw std::out_of_range("Parameter 'length' must be a positive integer.\n");
}
std::string str;
str.resize(length);
is.read(&str[0], length);
if(strip_last_char) {
str = str.substr(0, length-1);
}
return str;
}


void write_curv(std::ostream& os, std::vector<float> curv_data, int32_t num_faces = 100000) {
const uint32_t CURV_MAGIC = 16777215;
_fwritei3(os, CURV_MAGIC);
_fwritet<int32_t>(os, curv_data.size());
_fwritet<int32_t>(os, num_faces);
_fwritet<int32_t>(os, 1); 
for(size_t i=0; i<curv_data.size(); i++) {
_fwritet<float>(os, curv_data[i]);
}
}


void write_curv(const std::string& filename, std::vector<float> curv_data, const int32_t num_faces = 100000) {
std::ofstream ofs;
ofs.open(filename, std::ofstream::out | std::ofstream::binary);
if(ofs.is_open()) {
write_curv(ofs, curv_data, num_faces);
ofs.close();
} else {
throw std::runtime_error("Unable to open curvature file '" + filename + "' for writing.\n");
}
}

void write_mgh(const Mgh& mgh, std::ostream& os) {
_fwritet<int32_t>(os, 1); 
_fwritet<int32_t>(os, mgh.header.dim1length);
_fwritet<int32_t>(os, mgh.header.dim2length);
_fwritet<int32_t>(os, mgh.header.dim3length);
_fwritet<int32_t>(os, mgh.header.dim4length);

_fwritet<int32_t>(os, mgh.header.dtype);
_fwritet<int32_t>(os, mgh.header.dof);

size_t unused_header_space_size_left = 256;  
_fwritet<int16_t>(os, mgh.header.ras_good_flag);
unused_header_space_size_left -= 2; 

if(mgh.header.ras_good_flag == 1) {
_fwritet<float>(os, mgh.header.xsize);
_fwritet<float>(os, mgh.header.ysize);
_fwritet<float>(os, mgh.header.zsize);

for(int i=0; i<9; i++) {
_fwritet<float>(os, mgh.header.Mdc[i]);
}
for(int i=0; i<3; i++) {
_fwritet<float>(os, mgh.header.Pxyz_c[i]);
}

unused_header_space_size_left -= 60;
}

for(size_t i=0; i<unused_header_space_size_left; i++) {  
_fwritet<uint8_t>(os, 0);
}

size_t num_values = mgh.header.num_values();
if(mgh.header.dtype == MRI_INT) {
if(mgh.data.data_mri_int.size() != num_values) {
throw std::logic_error("Detected mismatch of MRI_INT data size and MGH header dim length values.\n");
}
for(size_t i=0; i<num_values; i++) {
_fwritet<int32_t>(os, mgh.data.data_mri_int[i]);
}
} else if(mgh.header.dtype == MRI_FLOAT) {
if(mgh.data.data_mri_float.size() != num_values) {
throw std::logic_error("Detected mismatch of MRI_FLOAT data size and MGH header dim length values.\n");
}
for(size_t i=0; i<num_values; i++) {
_fwritet<float>(os, mgh.data.data_mri_float[i]);
}
} else if(mgh.header.dtype == MRI_UCHAR) {
if(mgh.data.data_mri_uchar.size() != num_values) {
throw std::logic_error("Detected mismatch of MRI_UCHAR data size and MGH header dim length values.\n");
}
for(size_t i=0; i<num_values; i++) {
_fwritet<uint8_t>(os, mgh.data.data_mri_uchar[i]);
}
} else if(mgh.header.dtype == MRI_SHORT) {
if(mgh.data.data_mri_short.size() != num_values) {
throw std::logic_error("Detected mismatch of MRI_SHORT data size and MGH header dim length values.\n");
}
for(size_t i=0; i<num_values; i++) {
_fwritet<short>(os, mgh.data.data_mri_short[i]);
}
} else {
throw std::domain_error("Unsupported MRI data type " + std::to_string(mgh.header.dtype) + ", cannot write MGH data.\n");
}

}

void write_mgh(const Mgh& mgh, const std::string& filename) {
std::ofstream ofs;
ofs.open(filename, std::ofstream::out | std::ofstream::binary);
if(ofs.is_open()) {
write_mgh(mgh, ofs);
ofs.close();
} else {
throw std::runtime_error("Unable to open MGH file '" + filename + "' for writing.\n");
}
}

struct Label {
std::vector<int> vertex;  
std::vector<float> coord_x;  
std::vector<float> coord_y;  
std::vector<float> coord_z;  
std::vector<float> value;  

std::vector<bool> vert_in_label(size_t surface_num_verts) const {
if(surface_num_verts < this->vertex.size()) { 
std::cerr << "Invalid number of vertices for surface, must be at least " << this->vertex.size() << "\n";
}
std::vector<bool> is_in = std::vector<bool>(surface_num_verts, false);

for(size_t i=0; i < this->vertex.size(); i++) {
is_in[this->vertex[i]] = true;
}
return(is_in);
}

size_t num_entries() const {
size_t num_ent = this->vertex.size();
if(this->coord_x.size() != num_ent || this->coord_y.size() != num_ent || this->coord_z.size() != num_ent || this->value.size() != num_ent || this->value.size() != num_ent) {
std::cerr << "Inconsistent label: sizes of property vectors do not match.\n";
}
return(num_ent);
}
};

void write_surf(std::vector<float> vertices, std::vector<int32_t> faces, std::ostream& os) {
const uint32_t SURF_TRIS_MAGIC = 16777214;
_fwritei3(os, SURF_TRIS_MAGIC);
std::string created_and_comment_lines = "Created by fslib\n\n";
os << created_and_comment_lines;
_fwritet<int32_t>(os, vertices.size() / 3);  
_fwritet<int32_t>(os, faces.size() / 3);  
for(size_t i=0; i < vertices.size(); i++) {
_fwritet<float>(os, vertices[i]);
}
for(size_t i=0; i < faces.size(); i++) {
_fwritet<int32_t>(os, faces[i]);
}
}

void write_surf(std::vector<float> vertices, std::vector<int32_t> faces, const std::string& filename) {
std::ofstream ofs;
ofs.open(filename, std::ofstream::out | std::ofstream::binary);
if(ofs.is_open()) {
write_surf(vertices, faces, ofs);
ofs.close();
} else {
throw std::runtime_error("Unable to open surf file '" + filename + "' for writing.\n");
}
}


void write_surf(const Mesh& mesh, const std::string& filename ) {
std::ofstream ofs;
ofs.open(filename, std::ofstream::out | std::ofstream::binary);
if(ofs.is_open()) {
write_surf(mesh.vertices, mesh.faces, ofs);
ofs.close();
} else {
throw std::runtime_error("Unable to open surf file '" + filename + "' for writing.\n");
}
}

void read_label(Label* label, std::istream* is) {
std::string line;
int line_idx = -1;
size_t num_entries_header = 0;  
size_t num_entries = 0;  
while (std::getline(*is, line)) {
line_idx += 1;
std::istringstream iss(line);
if(line_idx == 0) {
continue; 
} else {
if(line_idx == 1) {
if (!(iss >> num_entries_header)) {
throw std::domain_error("Could not parse entry count from label file, invalid format.\n");
}
} else {
int vertex; float x, y, z, value;
if (!(iss >> vertex >> x >> y >> z >> value)) {
throw std::domain_error("Could not parse line " + std::to_string(line_idx+1) + " of label file, invalid format.\n");
}
label->vertex.push_back(vertex);
label->coord_x.push_back(x);
label->coord_y.push_back(y);
label->coord_z.push_back(z);
label->value.push_back(value);
num_entries++;
}
}
}
if(num_entries != num_entries_header) {
throw std::domain_error("Expected " + std::to_string(num_entries_header) + " entries from label file header, but found " + std::to_string(num_entries) + " in file, invalid label file.\n");
}
if(label->vertex.size() != num_entries || label->coord_x.size() != num_entries || label->coord_y.size() != num_entries || label->coord_z.size() != num_entries || label->value.size() != num_entries) {
throw std::domain_error("Expected " + std::to_string(num_entries) + " entries in all Label vectors, but some did not match.\n");
}
}


void read_label(Label* label, const std::string& filename) {
std::ifstream infile(filename);
if(infile.is_open()) {
read_label(label, &infile);
infile.close();
} else {
throw std::runtime_error("Could not open label file '" + filename + "' for reading.\n");
}
}


void write_label(const Label& label, std::ostream& os) {
const size_t num_entries = label.num_entries();
os << "#!ascii label from subject anonymous\n" << num_entries << "\n";
for(size_t i=0; i<num_entries; i++) {
os << label.vertex[i] << " " << label.coord_x[i] << " " << label.coord_y[i] << " " << label.coord_z[i] << " " << label.value[i] << "\n";
}
}


void write_label(const Label& label, const std::string& filename) {
std::ofstream ofs;
ofs.open(filename, std::ofstream::out);
if(ofs.is_open()) {
write_label(label, ofs);
ofs.close();
} else {
throw std::runtime_error("Unable to open label file '" + filename + "' for writing.\n");
}
}

void write_mesh(const Mesh& mesh, const std::string& filename ) {
if (fs::util::ends_with(filename, {".ply", ".PLY"})) {
mesh.to_ply_file(filename);
} else if (fs::util::ends_with(filename, {".obj", ".OBJ"})) {
mesh.to_obj_file(filename);
} else if (fs::util::ends_with(filename, {".off", ".OFF"})) {
mesh.to_off_file(filename);
} else {
fs::write_surf(mesh, filename);
}
}

} 


