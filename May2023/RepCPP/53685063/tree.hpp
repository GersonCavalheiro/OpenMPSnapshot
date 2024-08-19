


#ifndef tree_hpp
#define tree_hpp

#include "global.hpp"


template <int D>
class Planetesimal;                 

template <class T, int D>
class DataSet;              









template<typename T, typename std::enable_if<(sizeof(T)==sizeof(uint16_t)), int>::type = 0>
T endian_reverse(T &x) {
uint16_t tmp;
std::memcpy(&tmp, &x, sizeof(T));
#ifdef __GNUC__
tmp = __builtin_bswap16(tmp);
#else 
tmp = boost::endian::endian_reverse(tmp);
#endif 
std::memcpy(&x, &tmp, sizeof(T));
return x;
}

template<typename T, typename std::enable_if<(sizeof(T)==sizeof(uint32_t)), int>::type = 0>
T endian_reverse(T &x) {
uint32_t tmp;
std::memcpy(&tmp, &x, sizeof(T));
#ifdef __GNUC__
tmp = __builtin_bswap32(tmp);
#else 
tmp = boost::endian::endian_reverse(tmp);
#endif 
std::memcpy(&x, &tmp, sizeof(T));
return x;
}


template<typename T, typename std::enable_if<(sizeof(T)==sizeof(uint64_t)), int>::type = 0>
T endian_reverse(T &x) {
uint64_t tmp;
std::memcpy(&tmp, &x, sizeof(T));
#ifdef __GNUC__
tmp = __builtin_bswap64(tmp);
#else 
tmp = boost::endian::endian_reverse(tmp);
#endif 
std::memcpy(&x, &tmp, sizeof(T));
return x;
}


template <class T, int D>
class VtkDataScalar {
private:

public:

std::string data_name;


std::string data_type;


int num_components;


std::string table_name;


std::streampos pos;


SmallVec<int, D> num_cells;


using array_type = boost::multi_array<T, D>;


using shape_type = typename boost::array<typename array_type::index, D>;


using view_type = typename array_type::template array_view<D>::type;


template <typename std::enable_if<(D > 1), int>::type = 0>
using view_r1d_type = typename array_type::template array_view<D-1>::type;


template <typename std::enable_if<(D > 2), int>::type = 0>
using view_r2d_type = typename array_type::template array_view<D-2>::type;


array_type data;


shape_type shape;

};


template <class T, int D>
class VtkDataVector {
private:

public:

std::string data_name;


std::string data_type;


std::streampos pos;


SmallVec<int, D> num_cells;


using array_type = boost::multi_array<T, D+1>;


using shape_type = typename boost::array<typename array_type::index, D+1>;


using view_type = typename array_type::template array_view<D+1>::type;


template <typename std::enable_if<(D > 1), int>::type = 0>
using view_r1d_type = typename array_type::template array_view<D>::type;


template <typename std::enable_if<(D > 2), int>::type = 0>
using view_r2d_type = typename array_type::template array_view<D-1>::type;


array_type data;


shape_type shape;

};



template <class T, int D>
class VtkData {
private:

public:

std::string version;


std::string header;


std::string file_format;


std::string dataset_structure;


double time;


SmallVec<int, D> num_cells {SmallVec<int, D>(0)};


SmallVec<double, D> origin {SmallVec<double, D>(-0.1)};


SmallVec<double, D> spacing {SmallVec<double, D>(0.003125)};


long num_cell_data;


std::map<std::string, VtkDataScalar<T, D>> scalar_data;


typename std::map<std::string, VtkDataScalar<T, D>>::iterator sca_it;


std::map<std::string, VtkDataVector<T, D>> vector_data;


typename std::map<std::string, VtkDataVector<T, D>>::iterator vec_it;


int shape_changed_flag;


typename VtkDataVector<T, D>::array_type cell_center;


VtkData() {
;
}


~VtkData() {
;
}




template<class U, class F>
typename std::enable_if<(U::dimensionality==1), void>::type IterateBoostMultiArrayConcept(U& array, F f) {
for (auto& element : array) {
f(element);
}
}


template<class U, class F>
typename std::enable_if<(U::dimensionality>1), void>::type IterateBoostMultiArrayConcept(U& array, F f) {
for (auto element : array) {
IterateBoostMultiArrayConcept<decltype(element), F>(element, f);
}
}


template<class U, class F, class... Args>
typename std::enable_if<(U::dimensionality==1), void>::type IterateBoostMultiArrayConcept(U& array, F f, Args& ...args) {
for (auto& element : array) {
f(element, args...);
}
}


template<class U, class F, class... Args>
typename std::enable_if<(U::dimensionality>1), void>::type IterateBoostMultiArrayConcept(U& array, F f, Args& ...args) {
for (auto element : array) {
IterateBoostMultiArrayConcept<decltype(element), F, Args...>(element, f, args...);
}
}





template<typename RangeArrayType, size_t Dimension>
struct IndicesBuilder {
static auto Build(const RangeArrayType& range) -> decltype(IndicesBuilder<RangeArrayType, Dimension - 1>::Build(range)[range[Dimension - 1]]) {
return IndicesBuilder<RangeArrayType, Dimension - 1>::Build(range)[range[Dimension - 1]];
}
};


template<typename RangeArrayType>
struct IndicesBuilder<RangeArrayType, 1> {

static auto Build(const RangeArrayType& range) -> decltype(boost::indices[range[0]]) {
return boost::indices[range[0]];
}
};


template <typename U, size_t Dimension>
typename boost::multi_array<U, Dimension>::template array_view<Dimension>::type ExtractSubArrayView(boost::multi_array<U, Dimension>& array, const boost::array<size_t, Dimension>& corner, const boost::array<size_t, Dimension>& subarray_size) {

using array_type = boost::multi_array<U, Dimension>;
using range_type = typename array_type::index_range;

std::vector<range_type> range;
for (size_t i = 0; i != Dimension; ++i) {
range.push_back(range_type(corner[i], corner[i]+subarray_size[i]));
}

auto index = IndicesBuilder<decltype(range), Dimension>::Build(range);

typename array_type::template array_view<Dimension>::type view = array[index];
return view;
}



void ReadSingleVtkFile(std::vector<std::string>::iterator it) {
std::ifstream vtk_file;
std::string tmp_string;
vtk_file.open(it->c_str(), std::ios::binary);

const size_t D_T_size = D * sizeof(T);
const size_t one_T_size = sizeof(T);

if (vtk_file.is_open()) {
std::getline(vtk_file, version);
if (version.compare("# vtk DataFile Version 3.0") != 0 && version.compare("# vtk DataFile Version 2.0") != 0) {
progIO->log_info << "Warning: First line of " << *it << " is " << version << std::endl;
}
std::getline(vtk_file, header);
if (header.find("CONSERVED") != std::string::npos) {
size_t time_pos = header.find("time= ");
time = std::stod(header.substr(time_pos+6));
} else {
progIO->error_message << "Error: Expect CONSERVED, but read: " << header << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, file_format);
if (file_format.compare("BINARY") != 0) {
progIO->error_message << "Error: Unsupported file format: " << file_format << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, dataset_structure);
if (dataset_structure.compare("DATASET STRUCTURED_POINTS") != 0) {
progIO->error_message << "Error: Unsupported dataset structure: " << dataset_structure << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("DIMENSIONS") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
int tmp_index = 0;
decltype(num_cells) tmp_num_cells = num_cells;
while (!iss.eof()) {
iss >> num_cells[tmp_index++]; 
}

while (tmp_index > 0) {
num_cells[--tmp_index]--; 
}
if (num_cells != tmp_num_cells) {
shape_changed_flag = 1;
}
} else {
progIO->error_message << "Error: No dimensions info" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("ORIGIN") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
int tmp_index = 0;
while (!iss.eof()) {
iss >> origin[tmp_index++] >> std::ws; 
}
} else {
progIO->error_message << "Error: No origin info" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("SPACING") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
int tmp_index = 0;
while (!iss.eof()) {
iss >> spacing[tmp_index++] >> std::ws; 
}
} else {
progIO->error_message << "Error: No spacing info" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("CELL_DATA") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
iss >> num_cell_data;
SmallVec<long, D> tmp_num_cells = num_cells; 
long tmp_num_cell_data = std::accumulate(tmp_num_cells.data, tmp_num_cells.data+D, 1, std::multiplies<long>());
if (num_cell_data != tmp_num_cell_data) {
progIO->error_message << "Nx*Ny*Nz = " << tmp_num_cell_data << "!= Cell_Data = " << num_cell_data << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
} else {
progIO->error_message << "Error: No info about the number of CELL_DATA" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

progIO->log_info << *it << ", time = " << time << ", " << file_format << ", " << dataset_structure << ", num_cells = " << num_cells
<< ", origin = " << origin << ", spacing = " << spacing << ", CELL_DATA = " << num_cell_data << std::endl;
progIO->log_info << "data names:";

while (!vtk_file.eof()) {
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string[0] == '\n') {
tmp_string.erase(0, 1); 
}
if (tmp_string.compare("SCALARS") == 0) {
std::getline(vtk_file, tmp_string, ' ');

sca_it = scalar_data.find(tmp_string);
if (sca_it == scalar_data.end()) {
scalar_data[tmp_string] = VtkDataScalar<T, D>();
sca_it = scalar_data.find(tmp_string);
sca_it->second.data_name = tmp_string;
sca_it->second.num_cells = num_cells;
for (int i = 0; i != D; i++) {
sca_it->second.shape[i] = num_cells[D-1-i];
}
sca_it->second.data.resize(sca_it->second.shape);
} else {
if (num_cells != sca_it->second.num_cells) {
for (int i = 0; i != D; i++) {
sca_it->second.shape[i] = num_cells[D-1-i];
}
sca_it->second.num_cells = num_cells;
sca_it->second.data.resize(sca_it->second.shape);
}
}

progIO->log_info << " | " << sca_it->second.data_name;

std::getline(vtk_file, tmp_string);
size_t ws_pos = tmp_string.find_first_of(' '); 
if (ws_pos != std::string::npos) { 
std::istringstream iss;
iss.str(tmp_string);
iss >> sca_it->second.data_type >> sca_it->second.num_components;
} else {
sca_it->second.data_type = tmp_string;
sca_it->second.num_components = 1;
}
if (sca_it->second.data_type.compare("float") != 0) {
progIO->error_message << "Error: Expected float format, found " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

std::getline(vtk_file, tmp_string);
if (tmp_string.compare("LOOKUP_TABLE default") != 0) {
progIO->error_message << "Error: Expected \"LOOKUP_TABLE default\", unsupportted file" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
sca_it->second.table_name = "default";
sca_it->second.pos = vtk_file.tellg();

std::streampos length = one_T_size * num_cell_data;
vtk_file.read(reinterpret_cast<char*>(sca_it->second.data.data()), length);

for (auto it = sca_it->second.data.data(); it != sca_it->second.data.data()+sca_it->second.data.num_elements(); it++) {
*it = endian_reverse<T>(*it);
}

} else if (tmp_string.compare("VECTORS") == 0) { 
std::getline(vtk_file, tmp_string, ' ');

vec_it = vector_data.find(tmp_string);
if (vec_it == vector_data.end()) {
vector_data[tmp_string] = VtkDataVector<T, D>();
vec_it = vector_data.find(tmp_string);
vec_it->second.data_name = tmp_string;
vec_it->second.num_cells = num_cells;
for (int i = 0; i != D; i++) {
vec_it->second.shape[i] = num_cells[D-1-i];
}
vec_it->second.shape[D] = D; 
vec_it->second.data.resize(vec_it->second.shape);
} else {
if (num_cells != vec_it->second.num_cells) {
for (int i = 0; i != D; i++) {
sca_it->second.shape[i] = num_cells[D-1-i];
}
vec_it->second.shape[D] = D; 
vec_it->second.num_cells = num_cells;
vec_it->second.data.resize(vec_it->second.shape);
}
}
progIO->log_info << " | " << vec_it->second.data_name;

std::getline(vtk_file, tmp_string);
vec_it->second.data_type = tmp_string;
if (vec_it->second.data_type.compare("float") != 0) {
progIO->error_message << "Error: Expected float format, found " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

vec_it->second.pos = vtk_file.tellg();

std::streampos length = D_T_size * num_cell_data;
vtk_file.read(reinterpret_cast<char*>(vec_it->second.data.data()), length);
for (auto it = vec_it->second.data.data(); it != vec_it->second.data.data()+vec_it->second.data.num_elements(); it++) {
*it = endian_reverse<T>(*it);
}

} else { 
if (tmp_string.length() != 0) {
progIO->error_message << "Error: Expected SCALARS or VECTORS, found " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
}
}
progIO->log_info << " |" << std::endl;
progIO->Output(std::clog, progIO->log_info, __even_more_output, __all_processors);
vtk_file.close();
} else { 
progIO->error_message << "Error: Failed to open file " << it->c_str() << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

}


void ReadMultipleVtkFile(std::vector<std::string>::iterator begin, std::vector<std::string>::iterator end) {

std::ifstream vtk_file;
std::string tmp_string;
std::vector<SmallVec<double, D>> origins;   
std::vector<SmallVec<int, D>> dimensions;   
std::vector<SmallVec<double, D>> endings;   
std::vector<long> grid_cells;               
SmallVec<double, D> ending;                 
num_cell_data = 0;

const size_t D_T_size = D * sizeof(T);
const size_t one_T_size = sizeof(T);

vtk_file.open(begin->c_str(), std::ios::binary);
if (vtk_file.is_open()) {
std::getline(vtk_file, version);
if (version.compare("# vtk DataFile Version 3.0") != 0 &&
version.compare("# vtk DataFile Version 2.0") != 0) {
progIO->log_info << "Warning: First line of " << *begin << " is " << version << std::endl;
}
std::getline(vtk_file, header);
if (header.find("CONSERVED") != std::string::npos) {
size_t time_pos = header.find("time= ");
time = std::stod(header.substr(time_pos + 6));
} else {
progIO->error_message << "Error in " << *begin << ": Expect CONSERVED, but read: " << header << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, file_format);
if (file_format.compare("BINARY") != 0) {
progIO->error_message << "Error in " << *begin << ": Unsupported file format: " << file_format << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, dataset_structure);
if (dataset_structure.compare("DATASET STRUCTURED_POINTS") != 0) {
progIO->error_message << "Error in " << *begin << ": Unsupported dataset structure: " << dataset_structure << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("DIMENSIONS") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
int tmp_index = 0;
SmallVec<int, D> tmp_dimensions;
while (!iss.eof()) {
iss >> tmp_dimensions[tmp_index++]; 
}
while (tmp_index > 0) {
tmp_dimensions[--tmp_index]--; 
}
dimensions.push_back(tmp_dimensions);
} else {
progIO->error_message << "Error in " << *begin << ": No dimensions info" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("ORIGIN") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
int tmp_index = 0;
SmallVec<double, D> tmp_origin;
while (!iss.eof()) {
iss >> tmp_origin[tmp_index++] >> std::ws; 
}
origins.push_back(tmp_origin);
} else {
progIO->error_message << "Error in " << *begin << ": No origin info" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("SPACING") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
int tmp_index = 0;
while (!iss.eof()) {
iss >> spacing[tmp_index++] >> std::ws; 
}
} else {
progIO->error_message << "Error in " << *begin << ": No spacing info" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
endings.push_back(origins[0] + dimensions[0] * spacing);
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("CELL_DATA") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
long tmp_num_cell_data;
iss >> tmp_num_cell_data;
grid_cells.push_back(tmp_num_cell_data);
num_cell_data += tmp_num_cell_data;
} else {
progIO->error_message << "Error in " << *begin << ": No info about the number of CELL_DATA" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

progIO->log_info << *begin << ", time = " << time << ", " << file_format << ", " << dataset_structure << ", num_cells = " << dimensions[0]
<< ", origin = " << origins[0] << ", spacing = " << spacing << ", CELL_DATA = " << num_cell_data << std::endl;
progIO->log_info << "data names:";

while (!vtk_file.eof()) {
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string[0] == '\n') {
tmp_string.erase(0, 1); 
}
if (tmp_string.compare("SCALARS") == 0) {
std::getline(vtk_file, tmp_string, ' ');

sca_it = scalar_data.find(tmp_string);
if (sca_it == scalar_data.end()) {
scalar_data[tmp_string] = VtkDataScalar<T, D>();
sca_it = scalar_data.find(tmp_string);
sca_it->second.data_name = tmp_string;
}
progIO->log_info << " | " << sca_it->second.data_name;

std::getline(vtk_file, tmp_string);
size_t ws_pos = tmp_string.find_first_of(' '); 
if (ws_pos != std::string::npos) { 
std::istringstream iss;
iss.str(tmp_string);
iss >> sca_it->second.data_type >> sca_it->second.num_components;
} else {
sca_it->second.data_type = tmp_string;
sca_it->second.num_components = 1;
}
if (sca_it->second.data_type.compare("float") != 0) {
progIO->error_message << "Error in " << *begin << ": Expected float format, found " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

std::getline(vtk_file, tmp_string);
if (tmp_string.compare("LOOKUP_TABLE default") != 0) {
progIO->error_message << "Error in " << *begin << ": Expected \"LOOKUP_TABLE default\", unsupportted file" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
sca_it->second.table_name = "default";
sca_it->second.pos = vtk_file.tellg();
std::streampos length = one_T_size * num_cell_data;
vtk_file.seekg(length, vtk_file.cur);

} else if (tmp_string.compare("VECTORS") == 0) { 
std::getline(vtk_file, tmp_string, ' ');

vec_it = vector_data.find(tmp_string);
if (vec_it == vector_data.end()) {
vector_data[tmp_string] = VtkDataVector<T, D>();
vec_it = vector_data.find(tmp_string);
vec_it->second.data_name = tmp_string;
}
progIO->log_info << " | " << vec_it->second.data_name;

std::getline(vtk_file, tmp_string);
vec_it->second.data_type = tmp_string;
if (vec_it->second.data_type.compare("float") != 0) {
progIO->error_message << "Error in " << *begin << ": Expected float format, found " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

vec_it->second.pos = vtk_file.tellg();
std::streampos length = D_T_size * num_cell_data;
vtk_file.seekg(length, vtk_file.cur);

} else { 
if (tmp_string.length() != 0) {
progIO->error_message << "Error: Expected SCALARS or VECTORS, found " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
}
}
progIO->log_info << " |" << std::endl;
progIO->Output(std::clog, progIO->log_info, __even_more_output, __all_processors);
vtk_file.close();
} else { 
progIO->error_message << "Error: Failed to open file " << begin->c_str() << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

for (auto it = (begin+1); it != end; it++) {
vtk_file.open(it->c_str(), std::ios::binary);
if (vtk_file.is_open()) {
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("DIMENSIONS") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
int tmp_index = 0;
SmallVec<int, D> tmp_dimensions;
while (!iss.eof()) {
iss >> tmp_dimensions[tmp_index++];
}
while (tmp_index > 0) {
tmp_dimensions[--tmp_index]--; 
}
dimensions.push_back(tmp_dimensions);
} else {
progIO->error_message << "Error in " << *it << ": No dimensions info" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("ORIGIN") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
int tmp_index = 0;
SmallVec<double, D> tmp_origin;
while (!iss.eof()) {
iss >> tmp_origin[tmp_index++] >> std::ws;
}
origins.push_back(tmp_origin);
} else {
progIO->error_message << "Error in " << *it << ": No origin info" << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
endings.push_back(*(--origins.end()) + *(--dimensions.end()) * spacing);
std::getline(vtk_file, tmp_string);

std::getline(vtk_file, tmp_string, ' ');
if (tmp_string.compare("CELL_DATA") == 0) {
std::istringstream iss;
std::getline(vtk_file, tmp_string);
iss.str(tmp_string);
long tmp_num_cell_data;
iss >> tmp_num_cell_data;
grid_cells.push_back(tmp_num_cell_data);
num_cell_data += tmp_num_cell_data;
}
vtk_file.close();
} else { 
progIO->error_message << "Error: Failed to open file " << it->c_str() << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
}

origin = *std::min_element(origins.begin(), origins.end(), SmallVecLessEq<double, double, D>);
ending = *std::max_element(endings.begin(), endings.end(), SmallVecLessEq<double, double, D>);
SmallVec<double, D> tmp_double_num_cells = (ending - origin) / spacing;

decltype(num_cells) tmp_num_cells = num_cells;
for (int i = 0; i != D; i++) {
num_cells[i] = static_cast<int>(std::lrint(tmp_double_num_cells[i]));
}
if (num_cells != tmp_num_cells) {
shape_changed_flag = 1;
}

SmallVec<long, D> tmp_long_num_cells = num_cells; 
long tmp_num_cell_data = std::accumulate(tmp_long_num_cells.data, tmp_long_num_cells.data+D, 1, std::multiplies<long>());
if (num_cell_data != tmp_num_cell_data) {
progIO->error_message << "Nx*Ny*Nz = " << tmp_num_cell_data << "!= Cell_Data = " << num_cell_data << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

for (auto it = scalar_data.begin(); it != scalar_data.end(); it++) {
if (it->second.num_cells != num_cells) {
it->second.num_cells = num_cells;
for (int i = 0; i != D; i++) {
it->second.shape[i] = num_cells[D-1-i];
}
it->second.data.resize(it->second.shape);
}
}

for (auto it = vector_data.begin(); it != vector_data.end(); it++) {
if (it->second.num_cells != num_cells) {
it->second.num_cells = num_cells;
for (int i = 0; i != D; i++) {
it->second.shape[i] = num_cells[D-1-i];
}
it->second.shape[D] = D; 
it->second.data.resize(it->second.shape);
}
}
int file_count = 0;
for (auto it = begin; it != end; it++) {
vtk_file.open(it->c_str(), std::ios::binary);
if (vtk_file.is_open()) {
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 
std::getline(vtk_file, tmp_string); 

SmallVec<double, D> tmp_double_corner = (origins[file_count] - origin) / spacing;

boost::array<size_t, D> scalar_corner, scalar_subarray_size;
boost::array<size_t, D+1> vector_corner, vector_subarray_size;
boost::multi_array<T, 1> tmp_scalar_data (boost::extents[grid_cells[file_count]]);
boost::multi_array<T, 1> tmp_vector_data (boost::extents[grid_cells[file_count]*D]);
for (int i = 0; i != D; i++) {
scalar_corner[i] = static_cast<size_t>(std::lrint(tmp_double_corner[D-1-i]));
scalar_subarray_size[i] = dimensions[file_count][D-1-i];
vector_corner[i] = static_cast<size_t>(std::lrint(tmp_double_corner[D-1-i]));
vector_subarray_size[i] = dimensions[file_count][D-1-i];
}
vector_corner[D] = 0;
vector_subarray_size[D] = D;

while (!vtk_file.eof()) {
std::getline(vtk_file, tmp_string, ' ');
if (tmp_string[0] == '\n') {
tmp_string.erase(0, 1); 
}
if (tmp_string.compare("SCALARS") == 0) {
std::getline(vtk_file, tmp_string, ' ');

sca_it = scalar_data.find(tmp_string);
if (sca_it == scalar_data.end()) {
progIO->error_message << "Error in " << *it << ": find unknown data named " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string);
std::getline(vtk_file, tmp_string);

std::streamsize length = one_T_size * grid_cells[file_count];
vtk_file.read(reinterpret_cast<char*>(tmp_scalar_data.data()), length);

auto read_item =  tmp_scalar_data.data();
typename VtkDataScalar<T, D>::view_type tmp_view = ExtractSubArrayView(sca_it->second.data, scalar_corner, scalar_subarray_size);

IterateBoostMultiArrayConcept(tmp_view, [](T& element, decltype(read_item)& tmp_data)->void {
element = endian_reverse<T>(*tmp_data);
tmp_data++;
}, read_item);
} else if (tmp_string.compare("VECTORS") == 0) { 
std::getline(vtk_file, tmp_string, ' ');

vec_it = vector_data.find(tmp_string);
if (vec_it == vector_data.end()) {
progIO->error_message << "Error in " << *it << ": find unknown data named " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
std::getline(vtk_file, tmp_string);

std::streamsize length = D_T_size * grid_cells[file_count];
vtk_file.read(reinterpret_cast<char*>(tmp_vector_data.data()), length);

auto read_item = tmp_vector_data.data();
typename VtkDataVector<T, D>::view_type tmp_view = ExtractSubArrayView(vec_it->second.data, vector_corner, vector_subarray_size);

IterateBoostMultiArrayConcept(tmp_view, [](T& element, decltype(read_item)& tmp_data)->void {
element = endian_reverse<T>(*tmp_data);
tmp_data++;
}, read_item);
} else { 
if (tmp_string.length() != 0) {
progIO->error_message << "Error: Expected SCALARS or VECTORS, found " << tmp_string << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
}
}
vtk_file.close();

} else { 
progIO->error_message << "Error: Failed to open file " << it->c_str() << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);

}
file_count++;
}

}


void ReadVtkFile(int loop_count) {

if (progIO->flags.combined_flag) {
ReadSingleVtkFile(progIO->file_name.vtk_data_file_name.begin()+loop_count);
} else {
std::vector<std::string>::iterator file_head = progIO->file_name.vtk_data_file_name.begin();
ReadMultipleVtkFile(file_head + loop_count * progIO->num_cpus,
file_head + loop_count * progIO->num_cpus + progIO->num_cpus);
}


if (shape_changed_flag) {
typename VtkDataVector<T, D>::shape_type tmp_shape;
for (int i = 0; i != D; i++) {
tmp_shape[i] = num_cells[D-1-i];
}
tmp_shape[D] = D; 
cell_center.resize(tmp_shape);

long element_count = 0;
SmallVec<long, D> index_limits = num_cells;
SmallVec<long, D> limit_products;
SmallVec<long, D> tmp_indices;
SmallVec<T, D> tmp_cell_center;
SmallVec<T, D> tmp_origin = origin + spacing * SmallVec<T, D>(0.5);
size_t tmp_size = D * sizeof(T);
std::partial_sum(index_limits.data, index_limits.data+D, limit_products.data, std::multiplies<long>());
for (auto item = cell_center.data(); item != cell_center.data()+cell_center.num_elements(); ) {
tmp_indices[0] = element_count % limit_products[0];
for (int dim = D-1; dim != 0; dim--) {
long tmp_element_count = element_count; 
for (int d = D-1; d != dim ; d--) {
tmp_element_count -= tmp_indices[d] * limit_products[d-1]; 
}
tmp_indices[dim] = tmp_element_count / limit_products[dim-1]; 
}
tmp_cell_center = tmp_origin + spacing * tmp_indices;
std::memcpy(item, tmp_cell_center.data, tmp_size);
std::advance(item, D);
element_count++;
}
progIO->log_info << "cell_center constructed, [0][0][0] = (";
std::copy(cell_center.data(), cell_center.data()+D, std::ostream_iterator<T>(progIO->log_info, ", "));
progIO->log_info << ")" << std::endl;
progIO->Output(std::clog, progIO->log_info, __even_more_output, __master_only);

shape_changed_flag = 0;
} 

progIO->numerical_parameters.cell_length = spacing;
progIO->numerical_parameters.cell_volume = std::accumulate(spacing.data, spacing.data+D, 1.0, std::multiplies<double>());

}


};







template <int D>
class Particle {
private:

public:


SmallVec<double, D> pos, vel;


double density;


uint32_t id_in_run;


uint32_t id;


int property_index;


uint16_t cpu_id;


};


template <int D>
class ParticleSet {
private:

public:

using ivec = SmallVec<int, D>;


using fvec = SmallVec<float, D>;


using dvec = SmallVec<double, D>;


uint32_t num_particles;


uint32_t num_ghost_particles;


uint32_t num_total_particles;


unsigned int num_types;


double coor_lim[12];


std::vector<double> type_info;


double time;


double dt;


Particle<D> *particles;


double *new_densities;


ParticleSet() : particles(nullptr) {}


~ParticleSet() {
delete [] particles;
particles = nullptr;
}


Particle<D> operator[] (const size_t i) const {
assert(i < num_total_particles);
return *(particles+i);
}


Particle<D>& operator[] (const size_t i) {
assert(i < num_total_particles);
return *(particles+i);
}


void Reset() {
delete [] particles;
particles = nullptr;
}


void AllocateSpace(int N) {
Reset();
particles = new Particle<D>[N];
}


void ReadMultipleLisFile(std::vector<std::string>::iterator begin, std::vector<std::string>::iterator end) {
std::ifstream lis_file;
long tmp_num_particles;
float tmp_coor_lim[12], tmp_float_value, tmp_float_vector[D];
lis_file.open(begin->c_str(), std::ios::binary);
if (lis_file.is_open()) {
lis_file.read(reinterpret_cast<char*>(tmp_coor_lim), 12*sizeof(float));
for (int i = 0; i != 12; i++) {
coor_lim[i] = static_cast<double>(tmp_coor_lim[i]);
}
progIO->log_info << *begin << ", x1l = " << coor_lim[0] << ", x1u = " << coor_lim[1]
<< ", x2l = " << coor_lim[2] << ", x2u = " << coor_lim[3]
<< ", x3l = " << coor_lim[4] << ", x3u = " << coor_lim[5]
<< ", x1dl = " << coor_lim[6] << ", x1du = " << coor_lim[7]
<< ", x2dl = " << coor_lim[8] << ", x2du = " << coor_lim[9]
<< ", x3dl = " << coor_lim[10] << ", x3du = " << coor_lim[11] << "\n";
lis_file.read(reinterpret_cast<char*>(&num_types), sizeof(int));
progIO->log_info << "num_types = " << num_types;
type_info.resize(num_types);

for (unsigned int i = 0; i != num_types; i++) {
lis_file.read(reinterpret_cast<char*>(&tmp_float_value), sizeof(float));
type_info[i] = static_cast<double>(tmp_float_value);
progIO->log_info << ": type_info[" << i << "] = " << type_info[i];
}
progIO->log_info << "; || ";
lis_file.read(reinterpret_cast<char*>(&tmp_float_value), sizeof(float));
time = static_cast<double>(tmp_float_value);
lis_file.read(reinterpret_cast<char*>(&tmp_float_value), sizeof(float));
dt = static_cast<double>(tmp_float_value);
progIO->log_info << "time = " << time << ", dt = " << dt;
lis_file.read(reinterpret_cast<char*>(&tmp_num_particles), sizeof(long));
num_particles = static_cast<uint32_t>(tmp_num_particles);
lis_file.close();
} else { 
progIO->error_message << "Error: Failed to open file " << begin->c_str() << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

for (auto it = (begin+1); it != end; it++) {
lis_file.open(it->c_str(), std::ios::binary);
if (lis_file.is_open()) {
lis_file.seekg((14+num_types)*sizeof(float)+sizeof(int), std::ios::beg);
lis_file.read(reinterpret_cast<char*>(&tmp_num_particles), sizeof(long));
num_particles += static_cast<uint32_t>(tmp_num_particles);
lis_file.close();
} else {
progIO->error_message << "Error: Failed to open file " << it->c_str() << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
}

num_total_particles = num_particles;
progIO->log_info << ", num_particles = " << num_particles << "; || ";

AllocateSpace(num_particles);

uint32_t tmp_id = 0;
unsigned long tmp_long;
unsigned int tmp_int;
Particle<D> *p;
size_t D_float = D * sizeof(float);
size_t one_float = sizeof(float);
size_t one_int = sizeof(int);
size_t one_long = sizeof(long);

for (auto it = begin; it != end; it++) {
lis_file.open(it->c_str(), std::ios::binary);
if (lis_file.is_open()) {
lis_file.seekg((14+num_types)*sizeof(float)+sizeof(int), std::ios::beg);
lis_file.read(reinterpret_cast<char*>(&tmp_num_particles), sizeof(long));
std::stringstream content;
content << lis_file.rdbuf();
std::string tmp_str = content.str();
const char *tmp_char = tmp_str.data();
for (uint32_t i = 0; i != tmp_num_particles; i++) {
p = &particles[tmp_id];
std::memcpy((char*)tmp_float_vector, tmp_char, D_float);
for (int d = 0; d != D; d++) {
p->pos[d] = static_cast<double>(tmp_float_vector[d]);
}
std::advance(tmp_char, D_float);
std::memcpy((char*)tmp_float_vector, tmp_char, D_float);
for (int d = 0; d != D; d++) {
p->vel[d] = static_cast<double>(tmp_float_vector[d]);
}
std::advance(tmp_char, D_float);
std::memcpy((char*)&tmp_float_value, tmp_char, one_float);
p->density = static_cast<double>(tmp_float_value);
std::advance(tmp_char, one_float);
std::memcpy((char*)&p->property_index, tmp_char, one_int);

std::advance(tmp_char, one_int);
std::memcpy((char*)&tmp_long, tmp_char, one_long);
p->id_in_run = static_cast<uint32_t>(tmp_long);
std::advance(tmp_char, one_long);
std::memcpy((char*)&tmp_int, tmp_char, one_int);
p->cpu_id = static_cast<uint16_t>(tmp_int);
std::advance(tmp_char, one_int);


tmp_id++;
}
lis_file.close();
} else { 
progIO->error_message << "Error: Failed to open file " << it->c_str() << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}
}

uint32_t tmp_index = num_particles - 1;
progIO->log_info << "Last particle's info: id = " << particles[tmp_index].id << ", property_index = " << particles[tmp_index].property_index << ", rad = " << particles[tmp_index].density << ", pos = " << particles[tmp_index].pos << ", v = " << particles[tmp_index].vel << std::endl;
progIO->Output(std::clog, progIO->log_info, __even_more_output, __all_processors);

}


void ReadSingleLisFile(std::vector<std::string>::iterator it) {
std::ifstream lis_file;
long tmp_num_particles;
float tmp_coor_lim[12], tmp_float_value, tmp_float_vector[D];

lis_file.open(it->c_str(), std::ios::binary);
if (lis_file.is_open()) {
lis_file.read(reinterpret_cast<char*>(tmp_coor_lim), 12*sizeof(float));
for (int i = 0; i != 12; i++) {
coor_lim[i] = static_cast<double>(tmp_coor_lim[i]);
}
progIO->log_info << *it << ", x1l = " << coor_lim[0] << ", x1u = " << coor_lim[1]
<< ", x2l = " << coor_lim[2] << ", x2u = " << coor_lim[3]
<< ", x3l = " << coor_lim[4] << ", x3u = " << coor_lim[5]
<< ", x1dl = " << coor_lim[6] << ", x1du = " << coor_lim[7]
<< ", x2dl = " << coor_lim[8] << ", x2du = " << coor_lim[9]
<< ", x3dl = " << coor_lim[10] << ", x3du = " << coor_lim[11] << "\n";
lis_file.read(reinterpret_cast<char*>(&num_types), sizeof(int));
progIO->log_info << "num_types = " << num_types;
type_info.resize(num_types);
for (unsigned int i = 0; i != num_types; i++) {
lis_file.read(reinterpret_cast<char*>(&tmp_float_value), sizeof(float));
type_info[i] = static_cast<double>(tmp_float_value);
progIO->log_info << ": type_info[" << i << "] = " << type_info[i];
}
progIO->log_info << "; || ";
lis_file.read(reinterpret_cast<char*>(&tmp_float_value), sizeof(float));
time = static_cast<double>(tmp_float_value);
lis_file.read(reinterpret_cast<char*>(&tmp_float_value), sizeof(float));
dt = static_cast<double>(tmp_float_value);
progIO->log_info << "time = " << time << ", dt = " << dt;
lis_file.read(reinterpret_cast<char*>(&tmp_num_particles), sizeof(long));
num_particles = static_cast<uint32_t>(tmp_num_particles);
num_total_particles = num_particles;
progIO->log_info << ", num_particles = " << num_particles << "; || ";

AllocateSpace(num_particles);

uint32_t tmp_id = 0; unsigned long tmp_long; unsigned int tmp_int;
Particle<D> *p;
size_t D_float = D * sizeof(float);
size_t one_float = sizeof(float);
size_t one_int = sizeof(int);
size_t one_long = sizeof(long);

std::stringstream content;
content << lis_file.rdbuf();
std::string tmp_str = content.str();
const char *tmp_char = tmp_str.data();
for (uint32_t i = 0; i != tmp_num_particles; i++) {
p = &particles[tmp_id];
std::memcpy((char*)tmp_float_vector, tmp_char, D_float);
for (int d = 0; d != D; d++) {
p->pos[d] = static_cast<double>(tmp_float_vector[d]);
}
std::advance(tmp_char, D_float);
std::memcpy((char*)tmp_float_vector, tmp_char, D_float);
for (int d = 0; d != D; d++) {
p->vel[d] = static_cast<double>(tmp_float_vector[d]);
}
std::advance(tmp_char, D_float);
std::memcpy((char*)&tmp_float_value, tmp_char, one_float);
p->density = static_cast<double>(tmp_float_value);
std::advance(tmp_char, one_float);
std::memcpy((char*)&p->property_index, tmp_char, one_int);

std::advance(tmp_char, one_int);
std::memcpy((char*)&tmp_long, tmp_char, one_long);
p->id_in_run = static_cast<uint32_t>(tmp_long);
std::advance(tmp_char, one_long);
std::memcpy((char*)&tmp_int, tmp_char, one_int);
p->cpu_id = static_cast<uint16_t>(tmp_int);
std::advance(tmp_char, one_int);



tmp_id++;
}
lis_file.close();
} else { 
progIO->error_message << "Error: Failed to open file " << it->c_str() << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
exit(3); 
}

uint32_t tmp_index = num_particles - 1;
progIO->log_info << "Last particle's info: id = " << particles[tmp_index].id << ", property_index = " << particles[tmp_index].property_index << ", rad = " << particles[tmp_index].density << ", pos = " << particles[tmp_index].pos << ", v = " << particles[tmp_index].vel << std::endl;
progIO->Output(std::clog, progIO->log_info, __even_more_output, __all_processors);
}



void ReadLisFile(int loop_count) {
auto &paras = progIO->numerical_parameters;

if (progIO->flags.combined_flag) {
ReadSingleLisFile(progIO->file_name.lis_data_file_name.begin()+loop_count);
} else {
auto file_head = progIO->file_name.lis_data_file_name.begin();
ReadMultipleLisFile(file_head + loop_count * progIO->num_cpus,
file_head + loop_count * progIO->num_cpus + progIO->num_cpus);
}

std::sort(particles, particles+num_particles, [](const Particle<D> &a, const Particle<D> &b) {
if (a.cpu_id == b.cpu_id) {
return a.id_in_run < b.id_in_run;
}
return a.cpu_id < b.cpu_id;
});
for (uint32_t i = 0; i != num_particles; i++) {
particles[i].id = i;
}
progIO->physical_quantities[loop_count].time = time;
progIO->physical_quantities[loop_count].dt = dt;

if (loop_count == mpi->loop_begin) {
paras.box_min = SmallVec<double, D>(coor_lim[6], coor_lim[8], coor_lim[10]);
paras.box_max = SmallVec<double, D>(coor_lim[7], coor_lim[9], coor_lim[11]);
paras.box_center = (paras.box_min + paras.box_max) / 2.0;
paras.box_length = paras.box_max - paras.box_min;
paras.CalculateNewParameters();

paras.mass_per_particle.resize(num_types);
paras.mass_fraction_per_species.resize(num_types);
double tmp_sum = 0;
for (unsigned int i = 0; i != num_types; i++) {
paras.mass_fraction_per_species[i] = type_info[i];
tmp_sum += type_info[i];
}
for (unsigned int i = 0; i != num_types; i++) {
paras.mass_fraction_per_species[i] /= tmp_sum;
paras.mass_total_code_units = paras.solid_to_gas_ratio * paras.box_length[0] * paras.box_length[1] * std::sqrt(2.*paras.PI);
paras.mass_per_particle[i] = paras.mass_fraction_per_species[i] * paras.mass_total_code_units / num_particles;
}
}

if (progIO->flags.user_defined_box_flag) {
for (int i = 0; i != dim; i++) {
if (progIO->user_box_min[i] == 0 && progIO->user_box_max[i] == 0) {
progIO->user_box_min[i] = paras.box_min[i];
progIO->user_box_max[i] = paras.box_max[i];
}
}
if (paras.box_min.InRange(progIO->user_box_min, progIO->user_box_max) && paras.box_max.InRange(progIO->user_box_min, progIO->user_box_max)) {
progIO->log_info << "User-defined coordinate limits are beyond the original box. Nothing to do." << std::endl;
} else {
progIO->log_info << "User-defined coordinate limits are in effect: min = " << progIO->user_box_min << "; max = " << progIO->user_box_max << ". Turning on No_Ghost flag is recommended. ";

Particle<D> *user_selected_particles;
user_selected_particles = new Particle<D>[num_particles];
uint32_t num_user_selected_particles = 0;
for (uint32_t i = 0; i != num_particles; i++) {
if (particles[i].pos.InRange(progIO->user_box_min, progIO->user_box_max)) {
user_selected_particles[num_user_selected_particles] = particles[i];
num_user_selected_particles++;
}
}

progIO->log_info << num_user_selected_particles << " particles are picked out. ";

Reset();
AllocateSpace(num_user_selected_particles);
std::memcpy(particles, user_selected_particles, sizeof(Particle<D>)*num_user_selected_particles);

num_particles = num_user_selected_particles;
num_total_particles = num_particles;

delete [] user_selected_particles;
user_selected_particles = nullptr;
}
progIO->Output(std::clog, progIO->log_info, __more_output, __all_processors);
}
}


void MakeGhostParticles(NumericalParameters &paras) {
if (progIO->flags.no_ghost_particle_flag) {
return;
}

Particle<D> *ghost_particles_x, *ghost_particles_y;
uint32_t tmp_id = num_particles, ghost_id = 0;
double new_y = 0;

dvec non_ghost_width = paras.box_half_width - paras.ghost_zone_width;
for (int i = 0; i != D; i++) {
if (non_ghost_width[i] < 0) {
non_ghost_width[i] = 0;
}
}

dvec non_ghost_min = paras.box_center - non_ghost_width;
dvec non_ghost_max = paras.box_center + non_ghost_width;

ghost_id = 0;
for (uint32_t i = 0; i != num_particles; i++) {
if (particles[i].pos[0] > non_ghost_max[0] || particles[i].pos[0] < non_ghost_min[0]) {
ghost_id++;
}
}
ghost_particles_x = new Particle<D>[ghost_id];
ghost_id = 0;
for (uint32_t i = 0; i != num_particles; i++) {
if (particles[i].pos[0] > non_ghost_max[0]) {
ghost_particles_x[ghost_id] = particles[i];
ghost_particles_x[ghost_id].id = tmp_id++;
ghost_particles_x[ghost_id].pos[0] -= paras.box_length[0];
new_y = ghost_particles_x[ghost_id].pos[1] + paras.shear_speed * time;
ghost_particles_x[ghost_id].pos[1] = new_y - static_cast<int>((new_y - paras.box_min[1]) / paras.box_length[1]) * paras.box_length[1];
ghost_id++;
}
if (particles[i].pos[0] < non_ghost_min[0]) {
ghost_particles_x[ghost_id] = particles[i];
ghost_particles_x[ghost_id].id = tmp_id++;
ghost_particles_x[ghost_id].pos[0] += paras.box_length[0];
new_y = ghost_particles_x[ghost_id].pos[1] - paras.shear_speed * time;
ghost_particles_x[ghost_id].pos[1] = new_y + static_cast<int>((paras.box_max[1] - new_y) / paras.box_length[1]) * paras.box_length[1];
ghost_id++;
}
}

uint32_t tmp_num_ghost_particles = ghost_id;
ghost_id = 0;
for (uint32_t i = 0; i != tmp_num_ghost_particles; i++) {
if (ghost_particles_x[i].pos[1] < non_ghost_min[1] || ghost_particles_x[i].pos[1] > non_ghost_max[1]) {
ghost_id++;
}
}
for (uint32_t i = 0; i != num_particles; i++) {
if (particles[i].pos[1] > non_ghost_max[1] || particles[i].pos[1] < non_ghost_min[1]) {
ghost_id++;
}
}
ghost_particles_y = new Particle<D>[ghost_id];
ghost_id = 0;
for (uint32_t i = 0; i != tmp_num_ghost_particles; i++) {
if (ghost_particles_x[i].pos[1] < non_ghost_min[1]) {
ghost_particles_y[ghost_id] = ghost_particles_x[i];
ghost_particles_y[ghost_id].id = tmp_id++;
ghost_particles_y[ghost_id].pos[1] += paras.box_length[1];
ghost_id++;
}
if (ghost_particles_x[i].pos[1] > non_ghost_max[1]) {
ghost_particles_y[ghost_id] = ghost_particles_x[i];
ghost_particles_y[ghost_id].id = tmp_id++;
ghost_particles_y[ghost_id].pos[1] -= paras.box_length[1];
ghost_id++;
}
}
for (uint32_t i = 0; i != num_particles; i++) {
if (particles[i].pos[1] < non_ghost_min[1]) {
ghost_particles_y[ghost_id] = particles[i];
ghost_particles_y[ghost_id].id = tmp_id++;
ghost_particles_y[ghost_id].pos[1] += paras.box_length[1];
ghost_id++;
}
if (particles[i].pos[1] > non_ghost_max[1]) {
ghost_particles_y[ghost_id] = particles[i];
ghost_particles_y[ghost_id].id = tmp_id++;
ghost_particles_y[ghost_id].pos[1] -= paras.box_length[1];
ghost_id++;
}
}

num_total_particles = tmp_id;
num_ghost_particles = ghost_id + tmp_num_ghost_particles;
assert(num_total_particles == num_particles + num_ghost_particles);

progIO->log_info << "Finish making ghost particles: num_ghost_particles = " << num_ghost_particles << ", and now num_total_particles = " << num_total_particles << std::endl;
progIO->Output(std::clog, progIO->log_info, __more_output, __all_processors);

Particle<D> *tmp_particles = new Particle<D>[num_particles];
std::memcpy(tmp_particles, particles, sizeof(Particle<D>)*num_particles);
AllocateSpace(num_total_particles);
std::memcpy(particles, tmp_particles, sizeof(Particle<D>)*num_particles);
delete [] tmp_particles;
tmp_particles = nullptr;

std::memcpy(particles+num_particles, ghost_particles_x, sizeof(Particle<D>)*tmp_num_ghost_particles);
std::memcpy(particles+num_particles+tmp_num_ghost_particles, ghost_particles_y, sizeof(Particle<D>)*ghost_id);



delete [] ghost_particles_x;
ghost_particles_x = nullptr;
delete [] ghost_particles_y;
ghost_particles_y = nullptr;
}


double** MakeFinerSurfaceDensityMap(const unsigned int Nx, const unsigned int Ny) {
double **Sigma_ghost = nullptr;
double ccx[Nx+4], ccy[Ny+4], tmp, idx_origin, idy_origin; 
double dx, dy, inv_dx, inv_dy, dx2, dy2, half_dx, half_dy, three_half_dx, three_half_dy;
std::vector<double> sigma_per_particle;

double **tmp_Sigma = new double *[Ny+4];
tmp_Sigma[0] = new double[(Ny+4) * (Nx+4)];
std::fill(tmp_Sigma[0], tmp_Sigma[0] + (Ny+4) * (Nx+4), 0.0);
for (size_t i = 1; i != Ny+4; i++) {
tmp_Sigma[i] = tmp_Sigma[i - 1] + Nx+4;
}

if (progIO->flags.user_defined_box_flag) {
dx = (progIO->user_box_max[0] - progIO->user_box_min[0]) / Nx;
dy = (progIO->user_box_max[1] - progIO->user_box_min[1]) / Ny;
} else {
dx = progIO->numerical_parameters.box_length[0] / Nx;
dy = progIO->numerical_parameters.box_length[1] / Ny;
}

inv_dx = 1./dx; dx2 = dx*dx; half_dx = dx/2.; three_half_dx = 1.5*dx;
inv_dy = 1./dy; dy2 = dy*dy; half_dy = dy/2.; three_half_dy = 1.5*dy;

progIO->numerical_parameters.ghost_zone_width = sn::dvec(dx, dy, 0);
progIO->numerical_parameters.max_ghost_zone_width = std::max(dx, dy);
MakeGhostParticles(progIO->numerical_parameters);

if (progIO->flags.user_defined_box_flag) {
tmp = progIO->user_box_min[0] - 2.5 * dx;
idx_origin = progIO->user_box_min[0] - dx; 
} else {
tmp = progIO->numerical_parameters.box_min[0] - 2.5 * dx; 
idx_origin = progIO->numerical_parameters.box_min[0] - dx; 
}
std::generate(ccx, ccx+Nx+4, [&tmp, &dx]() {
tmp += dx;
return tmp;
});

if (progIO->flags.user_defined_box_flag) {
tmp = progIO->user_box_min[1] - 2.5 * dy;
idy_origin = progIO->user_box_min[1] - dy; 
} else {
tmp = progIO->numerical_parameters.box_min[1] - 2.5 * dy; 
idy_origin = progIO->numerical_parameters.box_min[1] - dy; 
}
std::generate(ccy, ccy+Ny+4, [&tmp, &dy]() {
tmp += dy;
return tmp;
});

sigma_per_particle.resize(num_types);
for (unsigned int i = 0; i != num_types; i++) {
sigma_per_particle[i] = progIO->numerical_parameters.mass_per_particle[i] / dx / dy;
}

#ifndef OpenMP_ON
boost::multi_array<double, 2> Sigma_ghost_private;
Sigma_ghost_private.resize(boost::extents[Ny+4][Nx+4]);
Sigma_ghost = new double *[Ny+4];
Sigma_ghost[0] = Sigma_ghost_private.data();
for (int i = 1; i != Ny+4; i++) {
Sigma_ghost[i] = Sigma_ghost[i-1] + Nx+4;
}
Particle<D> *p;
int idx, idy;
double dist, weightx[3], weighty[3];
#else
boost::multi_array<double, 3> Sigma_ghost_private;
Sigma_ghost_private.resize(boost::extents[progIO->numerical_parameters.num_avail_threads][Ny+4][Nx+4]);

omp_set_num_threads(progIO->numerical_parameters.num_avail_threads);
#pragma omp parallel private(Sigma_ghost)
{
int omp_myid = omp_get_thread_num();
Sigma_ghost = new double *[Ny+4];
Sigma_ghost[0] = Sigma_ghost_private.data() + omp_myid * (Ny+4) * (Nx+4);
for (int i = 1; i != Ny + 4; i++) {
Sigma_ghost[i] = Sigma_ghost[i - 1] + Nx + 4;
}
Particle<D> *p;
int idx, idy;
double dist, weightx[3], weighty[3];

#pragma omp for
#endif
for (uint32_t i = 0; i < num_total_particles; i++) {
p = &particles[i];
idx = static_cast<int>(std::floor((p->pos[0] - idx_origin) * inv_dx));
idy = static_cast<int>(std::floor((p->pos[1] - idy_origin) * inv_dy));

if (progIO->flags.user_defined_box_flag) {
if (idx > Nx+1 || idx < 0) {
continue;
}
if (idy > Ny+1 || idy < 0) {
continue;
}
} else {
if (idx == Nx+2) {
idx -= 1; 
}
if (idy == Ny+2) {
idy -= 1; 
}
if (idx == -1) {
idx = 0; 
}
if (idy == -1) {
idy = 0; 
}
}

for (int j = 0; j != 3; j++) {
dist = std::abs(p->pos[0] - ccx[idx + j]);
if (dist <= half_dx) {
weightx[j] = 0.75 - dist * dist / dx2;
} else if (dist < three_half_dx) {
weightx[j] = 0.5 * std::pow(1.5 - dist / dx, 2.);
} else {
weightx[j] = 0.;
}
dist = std::abs(p->pos[1] - ccy[idy + j]);
if (dist <= half_dy) {
weighty[j] = 0.75 - dist * dist / dy2;
} else if (dist < three_half_dy) {
weighty[j] = 0.5 * std::pow(1.5 - dist / dy, 2.);
} else {
weighty[j] = 0.;
}
}


for (int j = 0; j != 3; j++) {
for (int k = 0; k != 3; k++) {
Sigma_ghost[idy + j][idx + k] +=
sigma_per_particle[p->property_index] * weighty[j] * weightx[k];
}
}
}
#ifdef OpenMP_ON
}
#endif
std::memcpy(tmp_Sigma[0], Sigma_ghost_private.data(), sizeof(double)*(Ny+4)*(Nx+4));
#ifdef OpenMP_ON
for (unsigned int i = 1; i != progIO->numerical_parameters.num_avail_threads; i++) {
std::transform(tmp_Sigma[0], &tmp_Sigma[Ny+3][Nx+4], Sigma_ghost_private.data()+i*(Ny+4)*(Nx+4), tmp_Sigma[0], std::plus<double>());
}
Sigma_ghost_private.resize(boost::extents[0][0][0]);
#else
Sigma_ghost_private.resize(boost::extents[0][0]);
#endif
double **Sigma_p = new double *[Ny];
Sigma_p[0] = new double[Ny * Nx];
std::fill(Sigma_p[0], Sigma_p[0] + Ny * Nx, 0.0);
std::memcpy(Sigma_p[0], &tmp_Sigma[2][2], sizeof(double)*Nx);
for (int i = 1; i != Ny; i++) {
Sigma_p[i] = Sigma_p[i - 1] + Nx;
std::memcpy(Sigma_p[i], &tmp_Sigma[i+2][2], sizeof(double)*Nx);
}



if (Sigma_ghost != nullptr){
delete [] Sigma_ghost;
Sigma_ghost = nullptr;
}

return Sigma_p;
}


template <class T>
void RebuildVtk(const unsigned int &Nx, const unsigned int &Ny, const unsigned int &Nz, std::string &filename) {
auto &paras = progIO->numerical_parameters;
VtkDataScalar<T, D> rhop;
VtkDataVector<T, D> w;
rhop.data.resize(boost::extents[Nz][Ny][Nx]);
w.data.resize(boost::extents[Nz][Ny][Nx][3]);

boost::multi_array<T, 3> rhop_ghost;
boost::multi_array<T, 4> w_ghost;
rhop_ghost.resize(boost::extents[Nz+4][Ny+4][Nx+4]);
w_ghost.resize(boost::extents[Nz+4][Ny+4][Nx+4][3]);

double ccx[Nx+4], ccy[Ny+4], ccz[Nz+4], tmp, idx_origin, idy_origin, idz_origin; 
double dx, dy, dz, inv_dx, inv_dy, inv_dz, dx2, dy2, dz2;
double half_dx, half_dy, half_dz, three_half_dx, three_half_dy, three_half_dz;
std::vector<double> rhop_per_particle;

dx = paras.box_length[0] / Nx;
dy = paras.box_length[1] / Ny;
dz = paras.box_length[2] / Nz;

inv_dx = 1./dx; dx2 = dx*dx; half_dx = dx/2.; three_half_dx = 1.5*dx;
inv_dy = 1./dy; dy2 = dy*dy; half_dy = dy/2.; three_half_dy = 1.5*dy;
inv_dz = 1./dz; dz2 = dz*dz; half_dz = dz/2.; three_half_dz = 1.5*dz;

paras.ghost_zone_width = sn::dvec(dx, dy, dz);
paras.max_ghost_zone_width = MaxOf(dx, dy, dz);
MakeGhostParticles(paras);

tmp = paras.box_min[0] - 2.5 * dx; 
idx_origin = paras.box_min[0] - dx; 
std::generate(ccx, ccx+Nx+4, [&tmp, &dx]() {
tmp += dx;
return tmp;
});

tmp = paras.box_min[1] - 2.5 * dy; 
idy_origin = paras.box_min[1] - dy; 
std::generate(ccy, ccy+Ny+4, [&tmp, &dy]() {
tmp += dy;
return tmp;
});

tmp = paras.box_min[2] - 2.5 * dz; 
idz_origin = paras.box_min[2] - dz; 
std::generate(ccz, ccz+Nz+4, [&tmp, &dz]() {
tmp += dz;
return tmp;
});

rhop_per_particle.resize(num_types);
for (unsigned int i = 0; i != num_types; i++) {
rhop_per_particle[i] = paras.mass_per_particle[i] / dx / dy / dz;
}

Particle<D> *p;
int idx, idy, idz;
double dist, weightx[3], weighty[3], weightz[3];
for (uint32_t i = 0; i < num_total_particles; i++) {
p = &particles[i];
idx = static_cast<int>(std::floor((p->pos[0] - idx_origin) * inv_dx));
idy = static_cast<int>(std::floor((p->pos[1] - idy_origin) * inv_dy));
idz = static_cast<int>(std::floor((p->pos[2] - idz_origin) * inv_dz));

if (idx == Nx+2) {
idx -= 1; 
}
if (idy == Ny+2) {
idy -= 1; 
}
if (idz == Nz+2) {
idz -= 1; 
}
if (idx == -1) {
idx = 0; 
}
if (idy == -1) {
idy = 0; 
}
if (idz == -1) {
idz = 0; 
}

for (int j = 0; j != 3; j++) {
dist = std::abs(p->pos[0] - ccx[idx + j]);
if (dist <= half_dx) {
weightx[j] = 0.75 - dist * dist / dx2;
} else if (dist < three_half_dx) {
weightx[j] = 0.5 * std::pow(1.5 - dist / dx, 2.);
} else {
weightx[j] = 0.;
}
dist = std::abs(p->pos[1] - ccy[idy + j]);
if (dist <= half_dy) {
weighty[j] = 0.75 - dist * dist / dy2;
} else if (dist < three_half_dy) {
weighty[j] = 0.5 * std::pow(1.5 - dist / dy, 2.);
} else {
weighty[j] = 0.;
}
dist = std::abs(p->pos[2] - ccz[idz + j]);
if (dist <= half_dz) {
weightz[j] = 0.75 - dist * dist / dz2;
} else if (dist < three_half_dz) {
weightz[j] = 0.5 * std::pow(1.5 - dist / dz, 2.);
} else {
weightz[j] = 0.;
}
}

for (int j = 0; j != 3; j++) {
for (int k = 0; k != 3; k++) {
for (int l = 0; l != 3; l++) {
double tmp_weight = weightz[j] * weighty[k] * weightx[l];
rhop_ghost[idz + j][idy + k][idx + l] += rhop_per_particle[p->property_index] * tmp_weight;
for (int d = 0; d != 3; d++) {
w_ghost[idz + j][idy + k][idx + l][d] += rhop_per_particle[p->property_index] * tmp_weight * p->vel[d];
}
}
}
}
}

for (int iz = 0; iz != Nz; iz++) {
for (int iy = 0; iy != Ny; iy++) {
std::memcpy(&(rhop.data[iz][iy][0]), &(rhop_ghost[iz+2][iy+2][2]), sizeof(T)*Nx);
std::memcpy(&(w.data[iz][iy][0][0]), &(w_ghost[iz+2][iy+2][2][0]), sizeof(T)*Nx*3);
}
}

std::ofstream file_vtk;
file_vtk.open(filename, std::ios::binary);
if (file_vtk.is_open()) {
progIO->out_content << "Writing to " << filename << std::endl;
progIO->Output(std::cout, progIO->out_content, __normal_output, __all_processors);
file_vtk << "# vtk DataFile Version 3.0" << std::endl;
file_vtk << "CONSERVED vars at time= " << std::scientific << std::setprecision(6) << time
<< ", level= 0, domain= 0" << std::endl;
file_vtk << "BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS " << std::fixed << Nx + 1
<< " " << Ny + 1 << " " << Nz + 1 << std::endl;
file_vtk << "ORIGIN" << std::scientific << std::setprecision(6) << std::setw(14) << paras.box_min[0]
<< std::setw(14) << paras.box_min[1] << std::setw(14) << paras.box_min[2] << std::endl;
file_vtk << "SPACING" << std::scientific << std::setprecision(6) << std::setw(13) << paras.cell_length[0]
<< std::setw(13) << paras.cell_length[1] << std::setw(13) << paras.cell_length[2] << std::endl;
file_vtk << "CELL_DATA " << std::fixed << Nx * Ny * Nz << std::endl;
file_vtk << "SCALARS particle_density float\nLOOKUP_TABLE default" << std::endl;
for (auto it = rhop.data.data(); it != rhop.data.data() + rhop.data.num_elements(); it++) {
*it = endian_reverse<T>(*it);
}
file_vtk.write(reinterpret_cast<char *>(rhop.data.data()), sizeof(T) * Nx * Ny * Nz);
file_vtk << std::endl;
file_vtk << "VECTORS particle_momentum float" << std::endl;
for (auto it = w.data.data(); it != w.data.data() + w.data.num_elements(); it++) {
*it = endian_reverse<T>(*it);
}
file_vtk.write(reinterpret_cast<char *>(w.data.data()), D * sizeof(T) * Nx * Ny * Nz);
file_vtk.close();
} else {
progIO->error_message << "Failed to open " << filename << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}

rhop.data.resize(boost::extents[0][0][0]);
w.data.resize(boost::extents[0][0][0][0]);
rhop_ghost.resize(boost::extents[0][0][0]);
w_ghost.resize(boost::extents[0][0][0][0]);

}

};










template <int D>
struct Orthant {
static const SmallVec<int, D> orthants[1U<<D];
};




template <typename T>
void OutBinary(std::ostream &stream, T x) {
std::bitset<sizeof(T)*8> bits(x);
stream << bits;
}


class BaseMortonKey {
public:

#ifdef __GNUC__
using uint128_t = __uint128_t;
#endif 

private:



uint128_t m1;


uint128_t m2;


uint128_t c1;


uint128_t c2;


uint128_t c3;


uint128_t c4;


uint128_t c5;


uint128_t c6;


uint128_t upper32mask0;




static constexpr double MAGIC = 6755399441055744.0;

static constexpr double MAXIMUMINTEGER = 4294967294.0;

public:

using morton_key = uint128_t;


BaseMortonKey();


~BaseMortonKey();


uint32_t Double2Int(double d);


void InitializeMortonConstants();


inline int Key8Level(morton_key m_key, int level) {
int shr = 93 - 3 * (level - 1);
return (m_key>>shr) & 7UL; 
}


void OutKey(std::ostream &stream, morton_key m_key);


inline int ParticleIndex(morton_key m_key) {
return (m_key>>96);
}


morton_key Dilate3_Int32(int pos);

};




struct AscendingMorton {
bool operator() (BaseMortonKey::morton_key x, BaseMortonKey::morton_key y) {
return ( (x<<32) < (y<<32) );
}
};


template <int D>
class MortonKey : public BaseMortonKey {
private:

public:

using ivec = SmallVec<int, D>;


using fvec = SmallVec<float, D>;


using dvec = SmallVec<double, D>;


dvec scale;


dvec boxmin, boxmax;


void InitMortonKey(dvec __boxmin, dvec __boxmax) {
boxmin = __boxmin;
boxmax = __boxmax;
for (int d = 0; d != D; d++) {
scale[d] = 1.0 / (boxmax[d] - boxmin[d]);
}
}


template <class U>
morton_key Morton(const SmallVec<U, D> &pos, uint32_t index) {
dvec pos_scaled = pos - boxmin;
for (int d = 0; d != D; d++) {
pos_scaled[d] *= scale[d];
}

SmallVec<uint32_t, D> int_pos;
for (int d = 0; d != D; d++) {
int_pos[d] = Double2Int(pos_scaled[d]);
}
return Morton(int_pos, index);
}


morton_key Morton(const SmallVec<uint32_t, D> &pos, uint32_t index) {
morton_key result = (static_cast<uint128_t>(index))<<96;
for (int d = 0; d != D; d++) {
result |= (Dilate3_Int32(pos[d])<<d);
}
return result;
}

};







template <int D>
class BHtree : public MortonKey<D> {
private:

public:

using ivec = SmallVec<int, D>;


using fvec = SmallVec<float, D>;


using dvec = SmallVec<double, D>;


struct InternalParticle {

dvec pos;


dvec vel;


double mass;


double ath_density {0.0};


double new_density {0.0};


uint32_t original_id;






bool sink_flag {false};


bool in_clump_flag {false};
};


std::unordered_map<uint32_t, std::vector<InternalParticle>> sink_particle_indices;


struct TreeNode {

dvec center;


double half_width;


uint32_t begin;


uint32_t end;


uint32_t parent;


uint32_t first_daughter;


uint16_t orthant;


uint8_t num_daughters;


uint8_t level;
};


static const int max_level = 32;


typename MortonKey<D>::morton_key *morton;


uint32_t num_particles;


int enough_particle_resolution_flag {0};


InternalParticle *particle_list;


TreeNode *tree;


int num_leaf_nodes, *leaf_nodes;


int num_nodes, *node2leaf;


int max_leaf_size {1U<<D};


int max_daughters;


dvec root_center;


int root;


int root_level;


int node_ptr;


double half_width {0};


double epsilon {1. / 4294967296.};


int level_count[max_level];


int level_ptr[max_level];


std::vector<std::pair<int, double>> heaps;


const double to_diagonal = std::sqrt(D);

dvec cell_length;
dvec half_cell_length;
dvec cell_length_squared;
dvec three_half_cell_length;


BHtree() : morton(nullptr), particle_list(nullptr), tree(nullptr), leaf_nodes(nullptr), node2leaf(nullptr) {
max_daughters = (1U<<D);
root = 0;
root_level = 1;
}


void Reset() {
num_particles = 0;



delete [] particle_list;
particle_list = nullptr;

delete [] tree;
tree = nullptr;

delete [] morton;
morton = nullptr;

delete [] leaf_nodes;
leaf_nodes = nullptr;

delete [] node2leaf;
node2leaf = nullptr;

sink_particle_indices.clear();
assert(sink_particle_indices.size() == 0);
}


~BHtree() {
Reset();
ClearHeaps();
}


void  MakeSinkParticle(ParticleSet<D> &particle_set) {
for (auto it : sink_particle_indices) {
double mass = 0.0;
dvec vel(0);
for (auto it_par : it.second) {
vel += it_par.mass * it_par.vel;
mass += it_par.mass;
}
vel /= mass;
particle_list[it.first].mass = mass;
particle_list[it.first].vel = vel;
}
}


void SortPoints() {
InternalParticle *tmp = new InternalParticle[num_particles];
typename MortonKey<D>::morton_key *tmp_morton = new typename MortonKey<D>::morton_key[num_particles];

uint32_t new_index = 0; 
for (uint32_t i = 0; i < num_particles; ) {
tmp_morton[new_index] = morton[i];
if (i < num_particles && (morton[i]<<32) == (morton[i+1]<<32)) {
tmp[new_index] = particle_list[this->ParticleIndex(morton[i])];
tmp[new_index].sink_flag = true;
auto it = sink_particle_indices.emplace(new_index, std::vector<InternalParticle>());
assert(it.second);
it.first->second.push_back(particle_list[this->ParticleIndex(morton[i])]);
it.first->second.push_back(particle_list[this->ParticleIndex(morton[i+1])]);
uint32_t tmp_i = i + 2;
while (tmp_i < num_particles && (morton[i]<<32) == (morton[tmp_i]<<32)) {
it.first->second.push_back(particle_list[this->ParticleIndex(morton[tmp_i])]);
tmp_i++;
}
i = tmp_i;
} else {
tmp[new_index] = particle_list[this->ParticleIndex(morton[i])];
i++;
}
new_index++;
}
delete [] particle_list;
particle_list = nullptr;
delete [] morton;
morton = nullptr;

num_particles = new_index;
particle_list = new InternalParticle[num_particles]; 
morton = new typename MortonKey<D>::morton_key[num_particles];
std::memcpy(particle_list, tmp, sizeof(InternalParticle)*num_particles);
std::memcpy(morton, tmp_morton, sizeof(typename MortonKey<D>::morton_key)*num_particles);
delete [] tmp;
tmp = nullptr;
delete [] tmp_morton;
tmp_morton = nullptr;
}


void CountNodesLeaves(const int __level, int __begin, const int __end) { 
int orthant = this->Key8Level(morton[__begin], __level);
while ( (orthant < max_daughters) && (__begin < __end)) {
int count = 0;
while (__begin < __end) {
if (this->Key8Level(morton[__begin], __level) == orthant ) {
__begin++;
count++;
} else {
break;
}
}
assert(count > 0); 

level_count[__level]++;
num_nodes++;

if (count <= max_leaf_size || __level == max_level - 1) { 
num_leaf_nodes++; 
} else {
CountNodesLeaves(__level+1, __begin-count, __begin);
}

if (__begin < __end) {
orthant = this->Key8Level(morton[__begin], __level); 
}
}
}


void FillTree(const int __level, int __begin, const int __end, const int __parent, const dvec &__center, const double __half_width) { 
assert(__level < max_level);
assert(__end > __begin); 
assert(tree[__parent].first_daughter == 0);
assert(tree[__parent].num_daughters == 0); 

int orthant = this->Key8Level(morton[__begin], __level);
while (__begin < __end) {
assert( orthant < max_daughters);

int count = 0;
while (__begin < __end) {
if (this->Key8Level(morton[__begin], __level) == orthant) {
__begin++;
count++;
} else {
break;
}
}
assert(count > 0);

int daughter = level_ptr[__level];
level_ptr[__level]++;

if (tree[__parent].first_daughter == 0) {
assert(tree[__parent].num_daughters == 0);
tree[__parent].first_daughter = daughter;
tree[__parent].num_daughters = 1;
} else {
tree[__parent].num_daughters++;
assert(tree[__parent].num_daughters <= max_daughters);
}

TreeNode *p = &tree[daughter];
p->level = __level + 1;
p->parent = __parent;
p->begin = __begin - count;
p->end = __begin;
p->half_width = __half_width;
for (int d = 0; d != D; d++) {
p->center[d] = __center[d] + __half_width * Orthant<D>::orthants[orthant][d];
}
p->orthant = orthant;
p->first_daughter = 0;
p->num_daughters = 0;
node_ptr++;
assert(node_ptr < num_nodes);

if (count <= max_leaf_size || __level == max_level - 1) { 
leaf_nodes[num_leaf_nodes] = daughter;
node2leaf[daughter] = num_leaf_nodes;
num_leaf_nodes++;
} else {
FillTree(p->level, __begin-count, __begin, daughter, p->center, 0.5*__half_width);
}

if (__begin < __end) {
orthant = this->Key8Level(morton[__begin], __level);
}
}
}


void BuildTree(NumericalParameters &paras, ParticleSet<D> &particle_set, bool quiet=false, bool check_pos=true) { 
Reset();

if (check_pos) {
paras.particle_max = particle_set[0].pos;
paras.particle_min = particle_set[0].pos;
for (uint32_t i = 1; i != particle_set.num_total_particles; i++) {
if (!SmallVecLessEq(paras.particle_min, particle_set[i].pos)) {
for (int d = 0; d != D; d++) {
paras.particle_min[d] = std::min(particle_set[i].pos[d], paras.particle_min[d]);
}
}
if (!SmallVecGreatEq(paras.particle_max, particle_set[i].pos)) {
for (int d = 0; d != D; d++) {
paras.particle_max[d] = std::max(particle_set[i].pos[d], paras.particle_max[d]);
}
}
}
paras.max_particle_extent = 0.0;
for (int d = 0; d != D; d++) {
paras.max_particle_extent = std::max(paras.max_particle_extent, paras.particle_max[d] - paras.particle_min[d]);
}

half_width = paras.max_half_width + paras.max_ghost_zone_width;
root_center = paras.box_center;
if (paras.max_particle_extent < half_width) {
half_width = (std::ceil(paras.max_particle_extent * 100.) + 1) / 100.;
for (int d = 0; d != D; d++) {
if (paras.particle_max[d] - paras.particle_min[d] < 0.5 * paras.box_length[d]) {
root_center[d] = (paras.particle_max[d] + paras.particle_min[d]) / 2.0;
root_center[d] = std::round(root_center[d] * 1000.) / 1000.;
} else {
root_center[d] = paras.box_center[d];
}
}
progIO->log_info << "According to the spatial distribution of particles, the root node of BH tree is now centered at " << root_center << ", and with a half_width of " << half_width << std::endl;
}
} else {
if (half_width < 1e-16) {
half_width = paras.max_half_width + paras.max_ghost_zone_width;
root_center = paras.box_center;
}
}

assert(particle_set.num_total_particles < (uint32_t)0xffffffff);
num_particles = particle_set.num_total_particles;
particle_list = new InternalParticle[num_particles];

for (uint32_t i = 0; i != num_particles; i++) {
particle_list[i].pos = particle_set[i].pos;
particle_list[i].vel = particle_set[i].vel;
particle_list[i].mass = progIO->numerical_parameters.mass_per_particle[particle_set[i].property_index];
particle_list[i].ath_density = particle_set[i].density;
particle_list[i].original_id = particle_set[i].id;
}
particle_set.Reset(); 

this->InitMortonKey(root_center-dvec(half_width), root_center+dvec(half_width));
epsilon = 2 * half_width / std::pow(2, 31); 
morton = new typename MortonKey<D>::morton_key[num_particles];

for (uint32_t i = 0; i != num_particles; i++) {
morton[i] = this->Morton(particle_list[i].pos, i);
}
std::sort(&(morton[0]), &(morton[num_particles]), AscendingMorton());


SortPoints();
MakeSinkParticle(particle_set);

for (uint32_t i = 0; i != num_particles-1; i++) {
assert((morton[i]<<32) < (morton[i+1]<<32));
}

num_leaf_nodes = 0;
level_count[0] = 1;
for (int level = 1; level != max_level; level++) {
level_count[level] = 0;
}

num_nodes = 1; 
CountNodesLeaves(root_level, 0, num_particles);
assert(num_nodes == std::accumulate(level_count, level_count+max_level, 0));

node2leaf = new int[num_nodes];
for (int i = 0; i != num_nodes; i++) {
node2leaf[i] = -1;
}
leaf_nodes = new int[num_leaf_nodes];
tree = new TreeNode[num_nodes];

level_ptr[0] = 0;
for (int level = 1; level != max_level; level++) {
level_ptr[level] = level_ptr[level-1] + level_count[level-1];
}
node_ptr = 0;
TreeNode *p = &tree[root];
p->first_daughter = 0;
p->orthant = 0;
p->num_daughters = 0;
p->level = root_level;
p->center = root_center;
p->half_width = half_width;
p->begin = 0;
p->end = num_particles;
p->parent = std::numeric_limits<uint32_t>::max(); 

num_leaf_nodes = 0;
FillTree(root_level, 0, num_particles, root, root_center, 0.5*half_width);
assert(node_ptr + 1 == num_nodes);
delete [] morton;
morton = nullptr;

CheckTree(root, root_level, root_center, half_width);

cell_length = progIO->numerical_parameters.cell_length;
half_cell_length = cell_length / 2.;
cell_length_squared = cell_length * cell_length;
three_half_cell_length = half_cell_length * 3.;

if (!quiet) {
progIO->log_info << "Finish building a tree: num_nodes = " << num_nodes << ", num_sink_particles = " << sink_particle_indices.size();
if (sink_particle_indices.size() > 0) {
progIO->log_info << ", the largest sink particle contains " << std::max_element(sink_particle_indices.begin(), sink_particle_indices.end(), [](const typename decltype(sink_particle_indices)::value_type &a, const typename decltype(sink_particle_indices)::value_type &b) {
return a.second.size() < b.second.size();
})->second.size() << " super particles. ";
}
progIO->log_info << std::endl;
progIO->Output(std::clog, progIO->log_info, __more_output, __all_processors);
}
}


bool Within(const dvec &__pos, const dvec &node_center, const double __half_width) {
for (int d = 0; d != D; d++) {
if ( !(__pos[d] >= node_center[d] - __half_width - epsilon && __pos[d] <= node_center[d] + __half_width + epsilon)) {
return false;
}
}
return true;
}


void CheckTree(const int node, const int __level, const dvec &node_center, const double __half_width) {
assert(tree[node].level == __level);
assert(tree[node].half_width == __half_width);

for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
if (!Within(particle_list[p].pos, node_center, __half_width)) {
progIO->error_message << "Particle " << particle_list[p].pos << " outside node " << node_center << " with width " << 2*tree[node].half_width << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}
}

for (uint32_t daughter = tree[node].first_daughter; daughter != tree[node].first_daughter + tree[node].num_daughters; daughter++) {
dvec tmp_center = node_center;
for (int d = 0; d != D; d++) {
tmp_center[d] += 0.5 * __half_width * Orthant<D>::orthants[tree[daughter].orthant][d];
}
dvec daughter_center = tree[daughter].center;
assert(tmp_center == daughter_center);
CheckTree(daughter, __level+1, daughter_center, 0.5*__half_width);
}
}


inline bool IsLeaf(const int node) {
assert(node < num_nodes);
return (tree[node].num_daughters == 0); 
}


inline unsigned int NodeSize(const int node) {
assert(node < num_nodes);
return tree[node].end - tree[node].begin; 
}


inline bool InNode(const dvec &__pos, const int node) {
return Within(__pos, tree[node].center, tree[node].half_width);
}


uint32_t Key2Leaf(typename MortonKey<D>::morton_key const __morton, const int node, const int __level) {
if (IsLeaf(node)) {
return node;
}

int orthant = this->Key8Level(__morton, __level);
int daughter = -1;
for (uint32_t d = tree[node].first_daughter; d != tree[node].first_daughter + tree[node].num_daughters; d++) {
if (tree[d].orthant == orthant) {
daughter = Key2Leaf(__morton, d, __level+1);
break;
}
}
if (daughter == -1) {
progIO->error_message << "Key2Leaf: leaf cell doesn't exist in tree." << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
assert(daughter >= 0);
}
return daughter;
}


uint32_t Pos2Node(const dvec &__pos) {
assert( Within(__pos, root_center, half_width));
typename MortonKey<D>::morton_key __morton = this->Morton(__pos, 0); 
return Key2Leaf(__morton, root, root_level);
}


bool SphereContainNode(const dvec &__center, const double r, const int node) {
assert(node < num_nodes);
SmallVec<double, D> max_distance = (tree[node].center - __center);
double tmp_distance = tree[node].half_width * to_diagonal; 

for (int i = 0; i != D; i++) {
max_distance[i] = std::fabs(max_distance[i]) + tmp_distance;
}

if (max_distance.Norm2() < r * r) {
return true;
} else {
return false;
}
}


bool SphereNodeIntersect(const dvec &__center, const double r, const int node) {
assert(node < num_nodes);
double c2c = (tree[node].center - __center).Norm2();
double tmp_distance = tree[node].half_width * to_diagonal + r;

if (c2c > tmp_distance * tmp_distance) {
return false;
}

if (c2c < r || c2c < tree[node].half_width) {
return true;
}

dvec pos_min = tree[node].center - dvec(tree[node].half_width);
dvec pos_max = tree[node].center + dvec(tree[node].half_width);

double mindist2 = 0;
for (int d = 0; d != D; d++) {
if (__center[d] < pos_min[d]) {
mindist2 += (__center[d] - pos_min[d]) * (__center[d] - pos_min[d]);
} else if (__center[d] > pos_max[d]) {
mindist2 += (__center[d] - pos_max[d]) * (__center[d] - pos_max[d]);
}
}

return mindist2 <= r*r;
}


template <typename T1, typename T2>
struct less_second {
typedef std::pair<T1, T2> type;
bool operator ()(type const& a, type const& b) const {
return a.second < b.second;
}
};


void Add2Heaps(const unsigned int knn, const int i, const double dr2) {
if (heaps.size() < knn) {
heaps.push_back(std::pair<int, double>(i, dr2));
std::push_heap(heaps.begin(), heaps.end(), less_second<int, double>());
} else {
if (dr2 < heaps.front().second) {
std::pop_heap(heaps.begin(), heaps.end(), less_second<int, double>());
heaps.pop_back();
heaps.push_back(std::pair<int, double>(i, dr2));
std::push_heap(heaps.begin(), heaps.end(), less_second<int, double>());
}
}
}


inline void ClearHeaps() {
std::vector<std::pair<int, double>>().swap(heaps);
}


void RecursiveKNN(const dvec &__pos, const int node, const double dist, const int knn) {
if (SphereContainNode(__pos, dist, node)) {
for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
Add2Heaps(knn, p, (__pos-particle_list[p].pos).Norm2());
}
} else if (SphereNodeIntersect(__pos, dist, node)) {
if (IsLeaf(node)) {
for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
Add2Heaps(knn, p, (__pos-particle_list[p].pos).Norm2());
}
} else {
for (uint32_t d = tree[node].first_daughter; d != tree[node].first_daughter + tree[node].num_daughters; d++) {
RecursiveKNN(__pos, d, dist, knn);
}
}
}
}


void KNN_Search(const dvec &__pos, const unsigned int knn, double &radius_knn, uint32_t *indices, bool in_order = false) {
assert(knn <= num_particles);

if (heaps.size() != 0) {
ClearHeaps(); 
}
heaps.reserve(knn);

double max_dr2 = 0;
if (Within(__pos, root_center, half_width)) {
int node = Pos2Node(__pos);
while (NodeSize(node) < knn/4) {
node = tree[node].parent;
}
max_dr2 = tree[node].half_width*tree[node].half_width;
} else {
for (int d = 0; d < D; d++) {
double dx = MaxOf(root_center[d]-half_width - __pos[d], 0.0, __pos[d] - root_center[d] - half_width);
max_dr2 += dx * dx;
}
}

double max_dr = std::sqrt(max_dr2);
do {
ClearHeaps();
heaps.reserve(knn);
RecursiveKNN(__pos, root, max_dr, knn);
max_dr *= 2;
} while (heaps.size() < knn);


radius_knn = std::sqrt(heaps.front().second);
if (in_order) {
std::sort_heap(heaps.begin(), heaps.end(), less_second<int, double>());
}

for (unsigned int i = 0; i != heaps.size(); i++) {
indices[i] = heaps[i].first; 
}
}



void Add2Heaps_OpenMP(const unsigned int knn, const int i, const double dr2, std::vector<std::pair<int, double>> &local_heaps) {
if (local_heaps.size() < knn) {
local_heaps.push_back(std::pair<int, double>(i, dr2));
std::push_heap(local_heaps.begin(), local_heaps.end(), less_second<int, double>());
} else {
if (dr2 < local_heaps.front().second) {
std::pop_heap(local_heaps.begin(), local_heaps.end(), less_second<int, double>());
local_heaps.pop_back();
local_heaps.push_back(std::pair<int, double>(i, dr2));
std::push_heap(local_heaps.begin(), local_heaps.end(), less_second<int, double>());
}
}
}


void RecursiveKNN_OpenMP(const dvec &__pos, const int node, const double dist, const int knn, std::vector<std::pair<int, double>> &local_heaps) {
if (SphereContainNode(__pos, dist, node)) {
for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
Add2Heaps_OpenMP(knn, p, (__pos-particle_list[p].pos).Norm2(), local_heaps);
}
} else if (SphereNodeIntersect(__pos, dist, node)) {
if (IsLeaf(node)) {
for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
Add2Heaps_OpenMP(knn, p, (__pos-particle_list[p].pos).Norm2(), local_heaps);
}
} else {
for (uint32_t d = tree[node].first_daughter; d != tree[node].first_daughter + tree[node].num_daughters; d++) {
RecursiveKNN_OpenMP(__pos, d, dist, knn, local_heaps);
}
}
}
}


void KNN_Search_OpenMP(const dvec &__pos, const unsigned int knn, double &radius_knn, uint32_t *indices,  std::vector<std::pair<int, double>> &local_heaps, double estimated_distance=0) {
assert(knn <= num_particles);

if (local_heaps.size() != 0) {
std::vector<std::pair<int, double>>().swap(local_heaps);
}
local_heaps.reserve(knn);

double max_dr;
if (estimated_distance == 0) {
double max_dr2 = 0;
if (Within(__pos, root_center, half_width)) {
int node = Pos2Node(__pos);
while (NodeSize(node) < knn/4) {
node = tree[node].parent;
}
max_dr2 = tree[node].half_width*tree[node].half_width;
} else {
for (int d = 0; d < D; d++) {
double dx = MaxOf(root_center[d]-half_width - __pos[d], 0.0, __pos[d] - root_center[d] - half_width);
max_dr2 += dx * dx;
}
}

max_dr = std::sqrt(max_dr2);
} else {
max_dr = estimated_distance;
}

do {
std::vector<std::pair<int, double>>().swap(local_heaps);
local_heaps.reserve(knn);
RecursiveKNN_OpenMP(__pos, root, max_dr, knn, local_heaps);
max_dr *= 2;
} while (local_heaps.size() < knn);


radius_knn = std::sqrt(local_heaps.front().second);
std::sort_heap(local_heaps.begin(), local_heaps.end(), less_second<int, double>());

for (unsigned int i = 0; i != local_heaps.size(); i++) {
indices[i] = local_heaps[i].first; 
}
}


void RecursiveBallSearchCount(const dvec &__pos, int node, const double radius, uint32_t &count) {
if (SphereContainNode(__pos, radius, node)) {
count += (tree[node].end - tree[node].begin);
} else if (SphereNodeIntersect(__pos, radius, node)) {
if (IsLeaf(node)) {
for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
if ((__pos - particle_list[p].pos).Norm2() <= radius * radius) {
count++;
}
}
} else {
for (uint32_t d = tree[node].first_daughter; d != tree[node].first_daughter + tree[node].num_daughters; d++) {
RecursiveBallSearchCount(__pos, d, radius, count);
}
}
}
}


void RecursiveBallSearch(const dvec &__pos, int node, const double radius, uint32_t *indices, uint32_t &count) {
if (SphereContainNode(__pos, radius, node)) {
for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
indices[count++] = p;
}
} else if (SphereNodeIntersect(__pos, radius, node)) {
if (IsLeaf(node)) {
for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
if ((__pos - particle_list[p].pos).Norm2() <= radius * radius) {
indices[count++] = p;
}
}
} else {
for (uint32_t d = tree[node].first_daughter; d != tree[node].first_daughter + tree[node].num_daughters; d++) {
RecursiveBallSearch(__pos, d, radius, indices, count);
}
}
}
}


void BallSearch (const dvec &__center, const double radius, uint32_t *indices, uint32_t &count) {
count = 0;
RecursiveBallSearch(__center, root, radius, indices, count);
}



dvec ShearedCenter2Center(const dvec &__center, const int node, dvec &max_distance, double shear_distance) {
assert(node < num_nodes);
dvec tmp_node_center = tree[node].center;
dvec c2c;
for (int i = 0; i != D; i++) {
c2c[i] = __center[i] - tmp_node_center[i];
if (std::fabs(c2c[i]) > max_distance[i]) {
if (c2c[i] > 0) {
c2c[i] -= progIO->numerical_parameters.box_length[i];
} else {
c2c[i] += progIO->numerical_parameters.box_length[i];
}
if (i == 0 && D > 1) {
if (__center[0] < 0) { 
tmp_node_center[1] -= shear_distance;
tmp_node_center[1] += static_cast<int>((progIO->numerical_parameters.box_max[1]-tmp_node_center[1]) / progIO->numerical_parameters.box_length[1]) * progIO->numerical_parameters.box_length[1];
} else {
tmp_node_center[1] += shear_distance;
tmp_node_center[1] -= static_cast<int>((tmp_node_center[1]-progIO->numerical_parameters.box_min[1]) / progIO->numerical_parameters.box_length[1]) * progIO->numerical_parameters.box_length[1];
}
}
}
}
return c2c;
}


bool SphereContainNodeWithShear(const dvec &__center, const double r, const int node, const dvec &c2c) {
double tmp_distance = tree[node].half_width * to_diagonal; 
dvec tmp_dvec;
for (int i = 0; i != D; i++) {
tmp_dvec[i] = std::fabs(c2c[i])+tmp_distance;
}
if (tmp_dvec.Norm2() < r * r) {
return true;
} else {
return false;
}
}


bool SphereNodeIntersectWithShear(const dvec __center, const double r, const int node, const dvec &c2c) {
double c2c_distance = c2c.Norm2();
double tmp_distance = tree[node].half_width * to_diagonal + r;

if (c2c_distance > tmp_distance * tmp_distance) {
return false;
}

if (c2c_distance < r || c2c_distance < tree[node].half_width) {
return true;
}

dvec pos_min = __center - c2c - dvec(tree[node].half_width);
dvec pos_max = __center - c2c + dvec(tree[node].half_width);

double mindist2 = 0;
for (int d = 0; d != D; d++) {
if (__center[d] < pos_min[d]) {
mindist2 += (__center[d] - pos_min[d]) * (__center[d] - pos_min[d]);
} else if (__center[d] > pos_max[d]) {
mindist2 += (__center[d] - pos_max[d]) * (__center[d] - pos_max[d]);
}
}

return mindist2 <= r*r;
}


void RecursiveBallSearchCountWithShear(const dvec __pos, int node, const double radius, uint32_t &count, dvec &max_distance, double shear_distance) {
dvec c2c = ShearedCenter2Center(__pos, node, max_distance, shear_distance);
if (SphereContainNodeWithShear(__pos, radius, node, c2c)) {
count += (tree[node].end - tree[node].begin);
} else if (SphereNodeIntersectWithShear(__pos, radius, node, c2c)) {
if (IsLeaf(node)) {
for (uint32_t p = tree[node].begin; p != tree[node].end; p++) {
if ((__pos - particle_list[p].pos).Norm2() <= radius * radius) {
count++;
}
}
} else {
for (uint32_t d = tree[node].first_daughter; d != tree[node].first_daughter + tree[node].num_daughters; d++) {
RecursiveBallSearchCountWithShear(__pos, d, radius, count, max_distance, shear_distance);
}
}
}
}


double QuadraticSpline(dvec dist) const {
dist.AbsSelf();
double weight = 1;
for (int i = 0; i != D; i++) {
if (dist[i] <= half_cell_length[i]) {
weight *= (0.75 - dist[i]*dist[i]/cell_length_squared[i]);
} else if (dist[i] < three_half_cell_length[i]) {
weight *= 0.5 * std::pow(1.5 - dist[i]/cell_length[i], 2.);
} else {
return 0.;
}
}
return weight;
}


template <class T>
static double QseudoQuadraticSplinesKernel(const DataSet<T, D> &ds, const double radius, uint32_t self_id, uint32_t *indices, std::vector<std::pair<int, double>> &local_heaps) {
double density = 0;
for (unsigned int i = 0; i != progIO->numerical_parameters.num_neighbors_in_knn_search; i++) {
density += ds.tree.particle_list[indices[i]].mass / progIO->numerical_parameters.cell_volume * ds.tree.QuadraticSpline(ds.tree.particle_list[indices[i]].pos-ds.tree.particle_list[self_id].pos);
}
return density;
}


template <class T>
static double VerboseQseudoQuadraticSplinesKernel(const DataSet<T, D> &ds, const double radius, uint32_t self_id, uint32_t *indices, std::vector<std::pair<int, double>> &local_heaps) {
double density = 0, tmp_density = 0, QSdist = 0;
SmallVec<double, D> dist;
for (unsigned int i = 0; i != progIO->numerical_parameters.num_neighbors_in_knn_search; i++) {
dist = ds.tree.particle_list[indices[i]].pos-ds.tree.particle_list[self_id].pos;
QSdist = ds.tree.QuadraticSpline(dist);
tmp_density = ds.tree.particle_list[indices[i]].mass / progIO->numerical_parameters.cell_volume * QSdist;

std::cout << "i=" << i << "dist=" << dist.Norm() << ", QSdist=" << QSdist << ", tmp_d=" << tmp_density << std::endl;

density += tmp_density;
}
return density;
}


template <class T>
static double PureSphericalDensityKernel(const DataSet<T, D> &ds, const double radius, uint32_t self_id, uint32_t *indices, std::vector<std::pair<int, double>> &local_heaps) {
double density = 0;
for (unsigned int i = 0; i != progIO->numerical_parameters.num_neighbors_in_knn_search; i++) {
density += ds.tree.particle_list[indices[i]].mass;
}
density /= progIO->numerical_parameters.four_PI_over_three * std::pow(radius, 3);
return density;
}


template <class T>
static double MedianSphericalDensityKernel(const DataSet<T, D> &ds, const double radius, uint32_t self_id, uint32_t *indices, std::vector<std::pair<int, double>> &local_heaps) {
int i = progIO->numerical_parameters.num_neighbors_in_knn_search;
double density = 0;
double twice_median_radii_squared = 4 * local_heaps[i/2].second; 

for (i = i-1; i >= 0; i--) {
if (local_heaps[i].second > twice_median_radii_squared) {
continue;
} else {
break;
}
}
twice_median_radii_squared = local_heaps[i].second;
for ( ; i >= 0; i--) {
density += ds.tree.particle_list[indices[i]].mass;
}
density /= progIO->numerical_parameters.four_PI_over_three * std::pow(twice_median_radii_squared, 1.5); 
return density;
}


template <class T>
static double InverseDistanceWeightingDensityKernel(const DataSet<T, D> &ds, const double radius, uint32_t self_id, uint32_t *indices, std::vector<std::pair<int, double>> &local_heaps) {
int i = progIO->numerical_parameters.num_neighbors_in_knn_search;
double cut_radius = 0, mass = 0;
double twice_median_radii_squared = 4 * local_heaps[i/2].second;

for (i = i-1; i >= 0; i--) {
if (local_heaps[i].second > twice_median_radii_squared) {
continue;
} else {
break;
}
}
cut_radius = local_heaps[i].second;

for ( ; i >= 0; i--) {
mass += ds.tree.particle_list[indices[i]].mass * (twice_median_radii_squared - local_heaps[i].second) / twice_median_radii_squared;
}

return mass /= progIO->numerical_parameters.four_PI_over_three * std::pow(cut_radius, 1.5);
}


template <class T>
void RemoveSmallMassAndLowPeak(DataSet<T, D> &ds) {
std::vector<uint32_t> peaks_to_be_deleted;

for (auto &it : ds.planetesimal_list.planetesimals) {
if (it.second.total_mass < ds.planetesimal_list.clump_mass_threshold || particle_list[it.first].new_density < ds.planetesimal_list.peak_density_threshold) {
peaks_to_be_deleted.push_back(it.first);
continue;
}
uint32_t idx_limit = it.second.indices.size() < progIO->numerical_parameters.num_neighbors_in_knn_search ? it.second.indices.size() : progIO->numerical_parameters.num_neighbors_in_knn_search;
double peak_ori_density = 0;
for (uint32_t idx = 0; idx < idx_limit; idx++) {
peak_ori_density = std::max(peak_ori_density, ds.tree.particle_list[it.second.indices[idx]].ath_density);
}
if (peak_ori_density < ds.planetesimal_list.peak_density_threshold) {
peaks_to_be_deleted.push_back(it.first);
}
}
for (auto &it : peaks_to_be_deleted) {
ds.planetesimal_list.planetesimals.erase(it);
}
peaks_to_be_deleted.resize(0);
}


template <class T>
void OutputNaivePeakList(DataSet<T, D> &ds, const std::string &filename, boost::dynamic_bitset<> &mask) {
std::ofstream tmp_file(filename, std::ofstream::out);
if (!tmp_file.is_open()) {
std::cout << "Fail to open "+filename << std::endl;
}
tmp_file << "#" << std::setw(23) << "x" << std::setw(24) << "y" << std::setw(24) << "z" << std::setw(24) << "dis_max" << std::setw(24) << "Npar" << std::setw(24) << "R_1/10" << std::setw(24) << "R_HalfM" << std::setw(24) << "R_moreM" << std::endl;
tmp_file << std::scientific;
int idx = 0;
for (auto &it : ds.planetesimal_list.planetesimals) {
if (mask[idx]) {
for (int i_dim = 0; i_dim != 3; i_dim++) {
tmp_file << std::setprecision(16) << std::setw(24) << it.second.center_of_mass[i_dim];
}
tmp_file << std::setprecision(16) << std::setw(24) << it.second.particles.back().second << std::setw(24) << it.second.particles.size() << std::setw(24) << it.second.one10th_radius << std::setw(24) << it.second.inner_one10th_radius << std::setw(24) << it.second.outer_one10th_radius << std::endl;
}
idx++;
}
tmp_file.close();
}


template <class T>
void FindPlanetesimals(DataSet<T, D> &ds, int loop_count) {
auto &paras = progIO->numerical_parameters;
uint32_t horizontal_resolution = paras.box_resolution[0] * paras.box_resolution[1];
if (ds.tree.half_width < paras.max_half_width + paras.max_ghost_zone_width - 1e-16) {
horizontal_resolution = static_cast<uint32_t>(std::pow(ds.tree.half_width / paras.cell_length[0] * ((ds.tree.half_width - paras.max_ghost_zone_width) / ds.tree.half_width), 2.0));
}
if (ds.particle_set.num_particles >= 4 * horizontal_resolution) {
if (!paras.fixed_num_neighbors_to_hop) {
if (ds.particle_set.num_particles >= 1.67e7) { 
paras.num_neighbors_to_hop = 64;
} else if (ds.particle_set.num_particles >= 2.68e8) { 
paras.num_neighbors_to_hop = 128;
}
} else {
if ( (ds.particle_set.num_particles >= 1.67e7
&& paras.num_neighbors_to_hop < 64)
|| (ds.particle_set.num_particles >= 2.68e8
&& paras.num_neighbors_to_hop < 128) ) {
progIO->log_info << "The number of particles retrieved from data is quite a lot, " << ds.particle_set.num_particles << ", you may want to set a larger num_neighbors_to_hop by specifying \"hop\" in the parameter input file (current value is " << paras.num_neighbors_to_hop << ")." << std::endl;
progIO->Output(std::cout, progIO->log_info, __normal_output, __master_only);
}
}
enough_particle_resolution_flag = 1;
FindPlanetesimals(ds, BHtree<dim>::MedianSphericalDensityKernel<float>, loop_count);
} else if (ds.particle_set.num_particles >= horizontal_resolution) {
enough_particle_resolution_flag = 0;
progIO->log_info << "The number of particles retrieved from data files is merely " << ds.particle_set.num_particles / horizontal_resolution << " times the horizontal grid resolution. Consider using more particles in simulations or output more particle data for better results." << std::endl;
progIO->Output(std::cout, progIO->log_info, __more_output, __master_only);
FindPlanetesimals(ds, BHtree<dim>::QseudoQuadraticSplinesKernel<float>, loop_count);
} else {
enough_particle_resolution_flag = 0;
progIO->log_info << "The number of particles retrieved from data files is only " << ds.particle_set.num_particles << ", a fraction of the horizontal grid resolution " << horizontal_resolution << ". Using more particles in simulations or output more particle data is strongly recommended for more reliable results." << std::endl;
progIO->Output(std::cout, progIO->log_info, __normal_output, __all_processors);
FindPlanetesimals(ds, BHtree<dim>::QseudoQuadraticSplinesKernel<float>, loop_count);
}
}


template <class T, class F>
void FindPlanetesimals(DataSet<T, D> &ds, F DensityKernel, int loop_count) {

ds.planetesimal_list.planetesimals.clear();
ds.planetesimal_list.num_planetesimals = 0;
ds.planetesimal_list.peaks_and_masses.resize(0);
assert(ds.planetesimal_list.planetesimals.size() == 0);
assert(ds.planetesimal_list.planetesimals.size() == 0);
std::vector<uint32_t> peaks_to_be_deleted;


ds.planetesimal_list.density_threshold = std::pow(progIO->numerical_parameters.Omega, 2.) / progIO->numerical_parameters.PI / progIO->numerical_parameters.grav_constant * 2.; 
if (progIO->numerical_parameters.mass_total_code_units / ds.particle_set.num_particles / progIO->numerical_parameters.cell_volume >  ds.planetesimal_list.density_threshold / 4) {
ds.planetesimal_list.density_threshold *= 9. / 8.;
}
auto hydro_res_per_H = static_cast<unsigned int>((progIO->numerical_parameters.box_resolution / progIO->numerical_parameters.box_length).MinElement());
if (hydro_res_per_H <= 1024) {
ds.planetesimal_list.clump_diffuse_threshold = 0.35;
ds.planetesimal_list.Hill_fraction_for_merge = 0.75;
} else if (hydro_res_per_H <= 1536) {
ds.planetesimal_list.clump_diffuse_threshold = 0.4;
ds.planetesimal_list.Hill_fraction_for_merge = 0.5;
} else if (hydro_res_per_H <= 2048) {
ds.planetesimal_list.clump_diffuse_threshold = 0.5;
ds.planetesimal_list.Hill_fraction_for_merge = 0.35;
}
if (progIO->numerical_parameters.clump_diffuse_threshold > 0) {
ds.planetesimal_list.clump_diffuse_threshold = progIO->numerical_parameters.clump_diffuse_threshold;
}
if (progIO->numerical_parameters.Hill_fraction_for_merge > 0) {
ds.planetesimal_list.Hill_fraction_for_merge = progIO->numerical_parameters.Hill_fraction_for_merge;
}
ds.planetesimal_list.clump_mass_threshold = progIO->numerical_parameters.min_trusted_mass_code_unit;
ds.planetesimal_list.peak_density_threshold = 3 * ds.planetesimal_list.density_threshold; 

double radius_Kth_NN = 0.;
uint32_t *indices;
#ifdef OpenMP_ON
omp_set_num_threads(progIO->numerical_parameters.num_avail_threads);
#pragma omp parallel private(radius_Kth_NN, indices)
{
indices = new uint32_t[progIO->numerical_parameters.num_neighbors_in_knn_search];
std::vector<std::pair<int, double>> local_heaps;
#pragma omp for
for (uint32_t i = 0; i < num_particles; i++) {
KNN_Search_OpenMP(particle_list[i].pos, progIO->numerical_parameters.num_neighbors_in_knn_search, radius_Kth_NN, indices, local_heaps);
particle_list[i].new_density = DensityKernel(ds, radius_Kth_NN, i, indices, local_heaps);
}
delete[] indices;
indices = nullptr;
}
#else 
indices = new uint32_t[progIO->numerical_parameters.num_neighbors_in_knn_search];
for (uint32_t i = 0; i != num_particles; i++) {
KNN_Search(particle_list[i].pos, progIO->numerical_parameters.num_neighbors_in_knn_search, radius_Kth_NN, indices, true);
particle_list[i].new_density = DensityKernel(ds, radius_Kth_NN, i, indices, heaps);
}
delete [] indices;
indices = nullptr;
#endif 
progIO->log_info << "Density calculation done, max_dpar = "
<< std::max_element(particle_list, particle_list + num_particles,
[](const InternalParticle &p1, const InternalParticle &p2) {
return p1.new_density < p2.new_density;
}
)->new_density << ", dpar_threshold=" << ds.planetesimal_list.density_threshold << "; ";



auto *densest_neighbor_id_list = new uint32_t[num_particles]();
#ifdef OpenMP_ON
omp_set_num_threads(progIO->numerical_parameters.num_avail_threads);
#pragma omp parallel private(radius_Kth_NN, indices)
{
indices = new uint32_t[progIO->numerical_parameters.num_neighbors_to_hop];
std::vector<std::pair<int, double>> local_heaps;
#pragma omp for schedule(auto)
for (uint32_t i = 0; i < num_particles; i++) {
densest_neighbor_id_list[i] = i;
if (particle_list[i].new_density < ds.planetesimal_list.density_threshold) {
continue;
}
KNN_Search_OpenMP(particle_list[i].pos, progIO->numerical_parameters.num_neighbors_to_hop, radius_Kth_NN, indices, local_heaps);
double tmp_density = particle_list[i].new_density;
for (unsigned int j = 0; j != progIO->numerical_parameters.num_neighbors_to_hop; j++) {
double density_j = particle_list[indices[j]].new_density;
if (tmp_density < density_j) {
densest_neighbor_id_list[i] = indices[j];
tmp_density = density_j;
} else if (tmp_density == density_j && densest_neighbor_id_list[i] > indices[j]) {
densest_neighbor_id_list[i] = indices[j];
}
}
}
delete [] indices;
indices = nullptr;
}
#else 
indices = new uint32_t[progIO->numerical_parameters.num_neighbors_to_hop];
for (uint32_t i = 0; i < num_particles; i++) {
densest_neighbor_id_list[i] = i;
if (particle_list[i].new_density < ds.planetesimal_list.density_threshold) {
continue;
}
KNN_Search(particle_list[i].pos, progIO->numerical_parameters.num_neighbors_to_hop, radius_Kth_NN, indices);
double tmp_density = particle_list[i].new_density;
for (unsigned int j = 0; j != progIO->numerical_parameters.num_neighbors_to_hop; j++) {
double density_j = particle_list[indices[j]].new_density;
if (tmp_density < density_j) {
densest_neighbor_id_list[i] = indices[j];
tmp_density = density_j;
} else if (tmp_density == density_j && densest_neighbor_id_list[i] > indices[j]) {
densest_neighbor_id_list[i] = indices[j];
}
}
}
delete [] indices;
indices = nullptr;
#endif 

progIO->log_info << "Densest neighbor found, particle_list[0]'s densest_neighbor_id = " << densest_neighbor_id_list[0] << "; ";

boost::dynamic_bitset<> mask(num_particles);
mask.set(); 
auto  *tmp_peak_id_list = new uint32_t[num_particles]();
for (uint32_t i = 0; i != num_particles; i++) {
if (!mask[i]) {
continue;
}
if (particle_list[i].new_density < ds.planetesimal_list.density_threshold) {
mask.flip(i);
tmp_peak_id_list[i] = num_particles; 
continue;
}
std::vector<uint32_t> tmp_indices;
tmp_peak_id_list[i] = i;
uint32_t chain = i;
tmp_indices.push_back(chain);
mask.flip(chain);
chain = densest_neighbor_id_list[i];
while (tmp_indices.back() != chain) {
if (mask[chain]) {
tmp_indices.push_back(chain);
mask.flip(chain);
chain = densest_neighbor_id_list[chain];
} else {
break;
}
}
if (tmp_indices.back() == chain) { 
for (auto it : tmp_indices) {
tmp_peak_id_list[it] = chain;
}
} else { 
for (auto it : tmp_indices) {
tmp_peak_id_list[it] = tmp_peak_id_list[chain];
}
}
auto it = ds.planetesimal_list.planetesimals.emplace(tmp_peak_id_list[i], Planetesimal<D>());
if (it.second) { 
it.first->second.peak_index = tmp_peak_id_list[i];
}
it.first->second.indices.insert(it.first->second.indices.end(), tmp_indices.begin(), tmp_indices.end());
}
assert(mask.none()); 
mask.clear();
delete [] tmp_peak_id_list;
tmp_peak_id_list = nullptr;
delete [] densest_neighbor_id_list;
densest_neighbor_id_list = nullptr;
for (auto &it : ds.planetesimal_list.planetesimals) {
it.second.SortParticles(particle_list);
if (ds.planetesimal_list.clump_diffuse_threshold < 0.525) {
if (it.second.IsPositionDispersion2Large(ds.planetesimal_list.clump_diffuse_threshold)) {
peaks_to_be_deleted.push_back(it.first);
}
}
}
if (ds.planetesimal_list.clump_diffuse_threshold < 0.525) {
for (auto it : peaks_to_be_deleted) {
ds.planetesimal_list.planetesimals.erase(it);
}
peaks_to_be_deleted.resize(0);
}
ds.planetesimal_list.num_planetesimals = static_cast<uint32_t>(ds.planetesimal_list.planetesimals.size());


if (enough_particle_resolution_flag == 0) {
RemoveSmallMassAndLowPeak(ds);
}
ds.planetesimal_list.num_planetesimals = static_cast<uint32_t>(ds.planetesimal_list.planetesimals.size());
progIO->log_info << "Hopping to peaks done, naive peak amount = " << ds.planetesimal_list.planetesimals.size() << "; ";





int merge_happened_flag = 1;
double max_radius = 0.;
uint32_t *nearby_mask;
unsigned int merging_count = 0, delete_count = 0, predator_count = 0;
using pl_iterator = decltype(ds.planetesimal_list.planetesimals.begin());
std::vector<pl_iterator> nearby_pi;
std::vector<std::pair<pl_iterator, pl_iterator>> merging_pairs;
std::vector<std::pair<pl_iterator, std::vector<pl_iterator>>> combined_merging_pairs;
nearby_pi.resize(ds.planetesimal_list.planetesimals.size());
merging_pairs.resize(ds.planetesimal_list.num_planetesimals);
combined_merging_pairs.resize(ds.planetesimal_list.num_planetesimals);
peaks_to_be_deleted.resize(ds.planetesimal_list.num_planetesimals);

while (merge_happened_flag) {
ds.planetesimal_list.BuildClumpTree(root_center, half_width, max_radius);
merge_happened_flag = 0;
merging_count = 0;
predator_count = 0;
nearby_mask = new uint32_t[ds.planetesimal_list.num_planetesimals]();
#ifdef OpenMP_ON
omp_set_num_threads(progIO->numerical_parameters.num_avail_threads);
#pragma omp parallel
#endif
{
auto tmp_p1 = ds.planetesimal_list.planetesimals.begin();
auto tmp_p2 = tmp_p1;
double r_p1 = 0, r_p2 = 0, center_dist = 0;
uint32_t nearby_count = 0;
auto *nearby_indices = new uint32_t[ds.planetesimal_list.planetesimals.size()];

for (tmp_p1 = ds.planetesimal_list.planetesimals.begin(); tmp_p1 != ds.planetesimal_list.planetesimals.end(); tmp_p1++) {
r_p1 = tmp_p1->second.one10th_radius;
if (tmp_p1->second.mask) {
ds.planetesimal_list.clump_tree.BallSearch(tmp_p1->second.center_of_mass, tmp_p1->second.one10th_radius+max_radius, nearby_indices, nearby_count);
#ifdef OpenMP_ON
#pragma omp for
#endif
for (size_t idx = 0; idx < nearby_count; idx++) {
uint32_t tmp_p2_id = ds.planetesimal_list.clump_tree.particle_list[nearby_indices[idx]].ath_density; 
if (tmp_p2_id == tmp_p1->first) {
continue;
}
tmp_p2 = ds.planetesimal_list.planetesimals.find(tmp_p2_id);
if (tmp_p2 != ds.planetesimal_list.planetesimals.end()) {

if (tmp_p2->second.mask) {
r_p2 = tmp_p2->second.one10th_radius;
center_dist = (tmp_p1->second.center_of_mass - tmp_p2->second.center_of_mass).Norm();
if (center_dist < std::max(r_p1, r_p2)) {
nearby_mask[idx] = 1;
nearby_pi[idx] = tmp_p2;
tmp_p2->second.mask = false;
} else if (center_dist < (r_p1 + r_p2) && ds.planetesimal_list.IsGravitationallyBound(tmp_p1->second, tmp_p2->second)) {
if (!ds.planetesimal_list.IsSaddlePointDeepEnough(ds, DensityKernel, tmp_p1->second, tmp_p2->second)) {
nearby_mask[idx] = 1;
nearby_pi[idx] = tmp_p2;
tmp_p2->second.mask = false;
}
}
}
} else {
progIO->error_message << "Error: Cannot find a clump with peak_id=" << tmp_p2_id << ". This should not happen. Please report a bug. Proceed for now." << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}
}
}
#ifdef OpenMP_ON
#pragma omp barrier
#pragma omp single
#endif
{
for (size_t idx = 0; idx < nearby_count; idx++) {
if (nearby_mask[idx]) {
merging_pairs[merging_count].first = tmp_p1;
merging_pairs[merging_count].second = nearby_pi[idx];
merging_count++;
peaks_to_be_deleted[delete_count] = nearby_pi[idx]->first;
delete_count++;
nearby_mask[idx] = 0;
}
}
if (merging_count > 0) {
merge_happened_flag = 1;
}
}
}
delete [] nearby_indices;
nearby_indices = nullptr;
#ifdef OpenMP_ON
#pragma omp single
#endif
{
std::vector<decltype(ds.planetesimal_list.planetesimals.begin())> preys;
auto idx_limit = merging_count-1;
for (unsigned int idx = 0; idx != merging_count; ) {
preys.resize(0);
preys.push_back(merging_pairs[idx].second);
while (idx < idx_limit && merging_pairs[idx].first->first == merging_pairs[idx+1].first->first) {
preys.push_back(merging_pairs[idx+1].second);
idx++;
}
combined_merging_pairs[predator_count] = std::make_pair(merging_pairs[idx].first, preys);
predator_count++;
idx++;
}
}
#ifdef OpenMP_ON
#pragma omp for
#endif
for (uint32_t i = 0; i < predator_count; i++) {
for (auto &it : combined_merging_pairs[i].second) {
combined_merging_pairs[i].first->second.MergeAnotherPlanetesimal(it->second, particle_list);
}
}
}
delete [] nearby_mask;
nearby_mask = nullptr;
for (auto &it : ds.planetesimal_list.planetesimals) {
if (it.second.mask) {
if (it.second.IsPositionDispersion2Large()) {
it.second.mask = false;
peaks_to_be_deleted[delete_count] = it.first;
delete_count++;
}
}
}
ds.planetesimal_list.num_planetesimals = ds.planetesimal_list.planetesimals.size() - delete_count;

}

for (auto it : peaks_to_be_deleted) {
ds.planetesimal_list.planetesimals.erase(it);
}
merging_pairs.resize(0);
combined_merging_pairs.resize(0);
peaks_to_be_deleted.resize(0);
nearby_pi.resize(0);


merge_happened_flag = 1;
merging_count = 0; delete_count = 0; predator_count = 0; max_radius = 0;
for (auto &it : ds.planetesimal_list.planetesimals) {
it.second.CalculateHillRadius();
}
nearby_pi.resize(ds.planetesimal_list.planetesimals.size());
merging_pairs.resize(ds.planetesimal_list.num_planetesimals);
combined_merging_pairs.resize(ds.planetesimal_list.num_planetesimals);
peaks_to_be_deleted.resize(ds.planetesimal_list.num_planetesimals);

while (merge_happened_flag) {
ds.planetesimal_list.BuildClumpTree(root_center, half_width, max_radius, true);
merge_happened_flag = 0;
merging_count = 0;
predator_count = 0;
nearby_mask = new uint32_t[ds.planetesimal_list.num_planetesimals]();
#ifdef OpenMP_ON
omp_set_num_threads(progIO->numerical_parameters.num_avail_threads);
#pragma omp parallel
#endif
{
auto tmp_p1 = ds.planetesimal_list.planetesimals.begin();
auto tmp_p2 = tmp_p1;
double r_p1 = 0, r_p2 = 0, center_dist = 0;
uint32_t nearby_count = 0;
auto *nearby_indices = new uint32_t[ds.planetesimal_list.planetesimals.size()];

for (tmp_p1 = ds.planetesimal_list.planetesimals.begin(); tmp_p1 != ds.planetesimal_list.planetesimals.end(); tmp_p1++) {
r_p1 = tmp_p1->second.Hill_radius * ds.planetesimal_list.Hill_fraction_for_merge;
if (tmp_p1->second.mask) {
ds.planetesimal_list.clump_tree.BallSearch(tmp_p1->second.center_of_mass, tmp_p1->second.Hill_radius * ds.planetesimal_list.Hill_fraction_for_merge + max_radius, nearby_indices, nearby_count);
#ifdef OpenMP_ON
#pragma omp for
#endif
for (size_t idx = 0; idx < nearby_count; idx++) {
uint32_t tmp_p2_id = ds.planetesimal_list.clump_tree.particle_list[nearby_indices[idx]].ath_density; 
if (tmp_p2_id == tmp_p1->first) {
continue;
}
tmp_p2 = ds.planetesimal_list.planetesimals.find(tmp_p2_id);
if (tmp_p2 != ds.planetesimal_list.planetesimals.end()) {

if (tmp_p2->second.mask) {
r_p2 = tmp_p2->second.Hill_radius * ds.planetesimal_list.Hill_fraction_for_merge;
center_dist = (tmp_p1->second.center_of_mass - tmp_p2->second.center_of_mass).Norm();
if (center_dist < std::max(r_p1, r_p2)) {
nearby_mask[idx] = 1;
nearby_pi[idx] = tmp_p2;
tmp_p2->second.mask = false;
} else if (center_dist < (r_p1 + r_p2) && ds.planetesimal_list.IsGravitationallyBound(tmp_p1->second, tmp_p2->second)) {
if (!ds.planetesimal_list.IsHillSaddlePointDeepEnough(ds, DensityKernel, tmp_p1->second, tmp_p2->second)) {
nearby_mask[idx] = 1;
nearby_pi[idx] = tmp_p2;
tmp_p2->second.mask = false;
}
}
}
} else {
progIO->error_message << "Error: Cannot find a clump with peak_id=" << tmp_p2_id << ". This should not happen. Please report a bug. Proceed for now." << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}
}
}
#ifdef OpenMP_ON
#pragma omp barrier
#pragma omp single
#endif
{
for (size_t idx = 0; idx < nearby_count; idx++) {
if (nearby_mask[idx]) {
merging_pairs[merging_count].first = tmp_p1;
merging_pairs[merging_count].second = nearby_pi[idx];
merging_count++;
peaks_to_be_deleted[delete_count] = nearby_pi[idx]->first;
delete_count++;
nearby_mask[idx] = 0;
}
}
if (merging_count > 0) {
merge_happened_flag = 1;
}
}
}
delete [] nearby_indices;
nearby_indices = nullptr;
#ifdef OpenMP_ON
#pragma omp single
#endif
{
std::vector<decltype(ds.planetesimal_list.planetesimals.begin())> preys;
auto idx_limit = merging_count-1;
for (unsigned int idx = 0; idx != merging_count; ) {
preys.resize(0);
preys.push_back(merging_pairs[idx].second);
while (idx < idx_limit && merging_pairs[idx].first->first == merging_pairs[idx+1].first->first) {
preys.push_back(merging_pairs[idx+1].second);
idx++;
}
combined_merging_pairs[predator_count] = std::make_pair(merging_pairs[idx].first, preys);
predator_count++;
idx++;
}
}
#ifdef OpenMP_ON
#pragma omp for
#endif
for (uint32_t i = 0; i < predator_count; i++) {
for (auto &it : combined_merging_pairs[i].second) {
combined_merging_pairs[i].first->second.MergeAnotherPlanetesimal(it->second, particle_list);
combined_merging_pairs[i].first->second.CalculateHillRadius();
}
}
}
delete [] nearby_mask;
nearby_mask = nullptr;
for (auto &it : ds.planetesimal_list.planetesimals) {
if (it.second.mask) {
if (it.second.IsPositionDispersion2Large()) {
it.second.mask = false;
peaks_to_be_deleted[delete_count] = it.first;
delete_count++;
}
}
}
ds.planetesimal_list.num_planetesimals = ds.planetesimal_list.planetesimals.size() - delete_count;

}

for (auto it : peaks_to_be_deleted) {
ds.planetesimal_list.planetesimals.erase(it);
}
merging_pairs.resize(0);
combined_merging_pairs.resize(0);
peaks_to_be_deleted.resize(0);
nearby_pi.resize(0);

std::vector<uint32_t> real_peak_indices;
for (auto &it : ds.planetesimal_list.planetesimals) {
uint32_t tmp_peak_id = it.first;
for (auto it_other_peak : it.second.potential_subpeak_indices) {
if (particle_list[tmp_peak_id].new_density < particle_list[it_other_peak].new_density) {
tmp_peak_id = it_other_peak;
}
}
if (tmp_peak_id != it.first) {
peaks_to_be_deleted.push_back(it.first);
real_peak_indices.push_back(tmp_peak_id);
}
}
for (unsigned int i = 0; i != peaks_to_be_deleted.size(); i++) {
auto it = ds.planetesimal_list.planetesimals.emplace(real_peak_indices[i], Planetesimal<D>());
if (it.second) { 
auto original_item = ds.planetesimal_list.planetesimals.find(peaks_to_be_deleted[i]);
std::swap(original_item->second, it.first->second);
it.first->second.potential_subpeak_indices.push_back(original_item->first);
ds.planetesimal_list.planetesimals.erase(original_item);
} else {
progIO->error_message << "Real peak index replacing failed. " << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}
}
peaks_to_be_deleted.resize(0);
real_peak_indices.resize(0); /

if (ds.planetesimal_list.num_planetesimals > 0) {
for (auto &it : ds.planetesimal_list.planetesimals) {
ds.planetesimal_list.peaks_and_masses.push_back(std::pair<uint32_t, double>(it.first, it.second.total_mass));
}
std::sort(ds.planetesimal_list.peaks_and_masses.begin(), ds.planetesimal_list.peaks_and_masses.end(), [](const std::pair<uint32_t, double> &a, const std::pair<uint32_t, double> &b) {
if (a.second == b.second) {
return a.first < b.first; 
}
return a.second < b.second;
});

if (progIO->numerical_parameters.num_peaks > 0 && ds.planetesimal_list.num_planetesimals > progIO->numerical_parameters.num_peaks) {
size_t num_to_be_deleted = ds.planetesimal_list.num_planetesimals - progIO->numerical_parameters.num_peaks;
for (auto it = ds.planetesimal_list.peaks_and_masses.begin(); it != ds.planetesimal_list.peaks_and_masses.begin() + num_to_be_deleted; it++) {
peaks_to_be_deleted.push_back(it->first);
}
std::vector<typename decltype(ds.planetesimal_list.peaks_and_masses)::value_type>(
ds.planetesimal_list.peaks_and_masses.begin() + num_to_be_deleted,
ds.planetesimal_list.peaks_and_masses.end()).swap(ds.planetesimal_list.peaks_and_masses);
for (auto it : peaks_to_be_deleted) {
ds.planetesimal_list.planetesimals.erase(it);
}
peaks_to_be_deleted.resize(0);
ds.planetesimal_list.num_planetesimals = static_cast<uint32_t >(ds.planetesimal_list.planetesimals.size());

progIO->log_info << "Remove low&small peaks once more due to the input max num_peaks limit, now " << ds.planetesimal_list.planetesimals.size() << " left; ";
}

double Mp_tot = std::accumulate(ds.planetesimal_list.peaks_and_masses.begin(), ds.planetesimal_list.peaks_and_masses.end(), 0., [](const double &a, const std::pair<uint32_t, double> &b) {
return a + b.second;
});
progIO->out_content << "found " << ds.planetesimal_list.num_planetesimals << " clumps; " << " Mp_max = " << ds.planetesimal_list.peaks_and_masses.back().second << ", Mp_tot = " << Mp_tot << "(" << std::fixed << Mp_tot/progIO->numerical_parameters.mass_total_code_units*100 << "%) in code units.";
} else {
progIO->out_content << "found zero clumps";
}
if (loop_count >= 0) { 
ds.planetesimal_list.OutputPlanetesimalsInfo(loop_count, ds);
}


progIO->log_info << std::endl;
progIO->Output(std::clog, progIO->log_info, __more_output, __all_processors);
progIO->out_content << std::endl;
progIO->Output(std::cout, progIO->out_content, __normal_output, __all_processors);
}

};







template <int D>
class Planetesimal {
private:

public:

uint32_t peak_index {0};


std::vector<uint32_t> potential_subpeak_indices;


std::vector<uint32_t> indices;


bool mask {true};


double total_mass {0};


SmallVec<double, D> center_of_mass {0};


SmallVec<double, D> vel_com {0};


std::vector<std::pair<uint32_t, double>> particles;




double outer_one10th_radius {0};


double inner_one10th_radius {0};


double one10th_radius {0};






double two_sigma_mass_radius {0};

std::vector<uint32_t> preys;


double Hill_radius {0};


SmallVec<double, D> J {0};


SmallVec<double, D> accumulated_J_in_quarter_Hill_radius {0};


void CalculateKinematicProperties(typename BHtree<D>::InternalParticle *particle_list) {
total_mass = 0;
center_of_mass = SmallVec<double, D>(0);
vel_com = SmallVec<double, D>(0);
for (auto it : indices) {
total_mass += particle_list[it].mass;
center_of_mass += particle_list[it].pos * particle_list[it].mass;
vel_com += particle_list[it].vel * particle_list[it].mass;
}
center_of_mass /= total_mass;
vel_com /= total_mass;
}


void SortParticles(typename BHtree<D>::InternalParticle *particle_list) {
CalculateKinematicProperties(particle_list);
for (auto it : indices) {
particles.push_back(std::pair<uint32_t, double>(it, (particle_list[it].pos-center_of_mass).Norm()));
}
std::sort(particles.begin(), particles.end(), [](const std::pair<uint32_t, double> &a, const std::pair<uint32_t, double> &b) {
return a.second < b.second;
});

auto tmp_it = indices.begin();
for (auto it : particles) {
*tmp_it = it.first;
tmp_it++;
}

auto par_it = particles.begin();
auto par_rit = particles.rbegin();
auto tmp_one10th_peak_density = particle_list[indices[indices.size()/2]].new_density / 10.;
for (; par_it != particles.end(); ++par_it) {
if (particle_list[par_it->first].new_density < tmp_one10th_peak_density) {
break;
}
}
if (par_it != particles.end()) {
inner_one10th_radius = par_it->second;
} else {
inner_one10th_radius = particles.back().second;
}
for (; par_rit != particles.rend(); ++par_rit) {
if (particle_list[par_rit->first].new_density > tmp_one10th_peak_density) {
break;
}
}
outer_one10th_radius = par_rit->second;
one10th_radius = (inner_one10th_radius + outer_one10th_radius) / 2.; /

if ((inner_one10th_radius > 0 && outer_one10th_radius / inner_one10th_radius > 2.5)) {
one10th_radius = std::min(outer_one10th_radius, two_sigma_mass_radius);
} else {
one10th_radius = std::min(one10th_radius, two_sigma_mass_radius);
}
}


bool IsPositionDispersion2Large(double tolerance=0.55, double fraction=1.0) {
size_t num_poi = particles.size(); 

return std::accumulate(particles.begin(), particles.begin()+num_poi, 0.0, [](const double &a, const std::pair<uint32_t, double> &b) { return a + b.second*b.second; }) / num_poi > std::pow(tolerance * particles[num_poi-1].second, 2);
}


void MergeAnotherPlanetesimal(Planetesimal<D> &carnivore, typename BHtree<D>::InternalParticle *particle_list) {
if (particle_list[peak_index].new_density < particle_list[carnivore.peak_index].new_density) {
potential_subpeak_indices.push_back(peak_index);
peak_index = carnivore.peak_index;
} else {
potential_subpeak_indices.push_back(carnivore.peak_index);
}
double tmp_total_mass = total_mass + carnivore.total_mass;
center_of_mass = (center_of_mass * total_mass + carnivore.center_of_mass * carnivore.total_mass) / tmp_total_mass;
vel_com = (vel_com * total_mass + carnivore.vel_com * carnivore.total_mass) / tmp_total_mass;
total_mass = tmp_total_mass;

particles.insert(particles.end(), carnivore.particles.begin(), carnivore.particles.end());
for (auto &it : particles) {
it.second = (particle_list[it.first].pos - center_of_mass).Norm();
}
std::sort(particles.begin(), particles.end(), [](const std::pair<uint32_t, double> &a, const std::pair<uint32_t, double> &b) {
return a.second < b.second;
});

indices.resize(particles.size());
auto tmp_it = indices.begin();
for (auto it : particles) {
*tmp_it = it.first;
tmp_it++;
}

auto par_it = particles.begin();
auto par_rit = particles.rbegin();
auto tmp_one10th_peak_density = particle_list[indices[indices.size()/2]].new_density / 10.;
for (; par_it != particles.end(); ++par_it) {
if (particle_list[par_it->first].new_density < tmp_one10th_peak_density) {
break;
}
}
if (par_it != particles.end()) {
inner_one10th_radius = par_it->second;
} else {
inner_one10th_radius = particles.back().second;
}
for (; par_rit != particles.rend(); ++par_rit) {
if (particle_list[par_rit->first].new_density > tmp_one10th_peak_density) {
break;
}
}
outer_one10th_radius = par_rit->second;
one10th_radius = (inner_one10th_radius + outer_one10th_radius) / 2.; /

if ((inner_one10th_radius > 0 && outer_one10th_radius / inner_one10th_radius > 2.5)) {
one10th_radius = std::min(outer_one10th_radius, two_sigma_mass_radius);
} else {
one10th_radius = std::min(one10th_radius, two_sigma_mass_radius);
}
}


void RemoveUnboundParticles(typename BHtree<D>::InternalParticle *particle_list) {
auto i = particles.size()-1;
double tmp_total_mass;
SmallVec<double, D> tmp_center_of_mass, tmp_vel_com;
double total_energy_over_mass_product;



while (i != 0) {
tmp_total_mass = total_mass - particle_list[particles[i].first].mass;
tmp_center_of_mass = (center_of_mass * total_mass - particle_list[particles[i].first].pos * particle_list[particles[i].first].mass) / tmp_total_mass;
tmp_vel_com = (vel_com * total_mass - particle_list[particles[i].first].vel * particle_list[particles[i].first].mass) / tmp_total_mass;
/

total_energy_over_mass_product =
+ 0.5 / total_mass * (tmp_vel_com - progIO->numerical_parameters.shear_vector*(tmp_center_of_mass[0]-particle_list[particles[i].first].pos[0]) - particle_list[particles[i].first].vel).Norm2()
- progIO->numerical_parameters.grav_constant / (tmp_center_of_mass - particle_list[particles[i].first].pos).Norm();
if (total_energy_over_mass_product > 0) { 
std::swap(indices[i], indices.back());
indices.pop_back();
std::swap(particles[i], particles.back());
particles.pop_back();
vel_com = tmp_vel_com;
center_of_mass = tmp_center_of_mass;
total_mass = tmp_total_mass;
}
i--;
}

std::vector<std::pair<uint32_t, double>> tmp;
particles.swap(tmp);
SortParticles(particle_list);
}


void SearchBoundParticlesWithinHillRadius(BHtree<D> &tree, double density_threshold) {
uint32_t nearby_count = 0, idx = 0;
tree.RecursiveBallSearchCount(center_of_mass, tree.root, Hill_radius, nearby_count);
auto *nearby_indices = new uint32_t[nearby_count];
tree.BallSearch(center_of_mass, Hill_radius, nearby_indices, nearby_count);

double total_energy_over_mass_product;

for (uint32_t i = 0; i != nearby_count; i++) {
idx = nearby_indices[i];
if (tree.particle_list[idx].in_clump_flag || tree.particle_list[idx].new_density < density_threshold) {
continue;
} else {
auto tmp_total_mass = total_mass + tree.particle_list[idx].mass;
total_energy_over_mass_product = + 0.5 / tmp_total_mass * (vel_com - progIO->numerical_parameters.shear_vector*(center_of_mass[0]-tree.particle_list[idx].pos[0]) - tree.particle_list[idx].vel).Norm2()
- progIO->numerical_parameters.grav_constant / (center_of_mass - tree.particle_list[idx].pos).Norm();
if (total_energy_over_mass_product < 0) { 
indices.push_back(idx);
total_mass = tmp_total_mass;
center_of_mass = (center_of_mass * total_mass + tree.particle_list[idx].mass * tree.particle_list[idx].pos) / total_mass;
vel_com = (total_mass * vel_com + tree.particle_list[idx].mass * tree.particle_list[idx].vel) / total_mass;
tree.particle_list[idx].in_clump_flag = true;
}
}
}
if (indices.size() > particles.size()) {
std::vector<std::pair<uint32_t, double>> tmp;
particles.swap(tmp);
SortParticles(tree.particle_list);
}
delete[] nearby_indices;
}


void CalculateHillRadius() {
Hill_radius = pow(total_mass * progIO->numerical_parameters.grav_constant / 3. / progIO->numerical_parameters.Omega / progIO->numerical_parameters.Omega, 1./3.);
}


void CalculateAngularMomentum(BHtree<D> &tree) {
SmallVec<double, D> tmp_j {0};
SmallVec<double, D> tmp_dr {0};
SmallVec<double, D> tmp_dv {0};
double quarter_Hill_radius = 0.25 * Hill_radius;
double Hill_units {total_mass * Hill_radius * Hill_radius * progIO->numerical_parameters.Omega};
SmallVec<double, D> shear_vector (0., progIO->numerical_parameters.q * progIO->numerical_parameters.Omega, 0.);

for (auto it : particles) {
tmp_dr = tree.particle_list[it.first].pos - center_of_mass;
tmp_dv = tree.particle_list[it.first].vel - shear_vector * tree.particle_list[it.first].pos[0]
- (                vel_com - shear_vector * center_of_mass[0]);
tmp_j = tmp_dr.Cross(tmp_dv);
tmp_j += progIO->numerical_parameters.Omega
* SmallVec<double, 3>(-tmp_dr[0]*tmp_dr[2],
-tmp_dr[1]*tmp_dr[2],
(tmp_dr[0]*tmp_dr[0] + tmp_dr[1]*tmp_dr[1]));
J += tree.particle_list[it.first].mass * tmp_j;
if (it.second < quarter_Hill_radius) {
accumulated_J_in_quarter_Hill_radius += tree.particle_list[it.first].mass * tmp_j;
}
}

J /= Hill_units;
accumulated_J_in_quarter_Hill_radius /= Hill_units;
}


void CalculateCumulativeAngularMomentum(BHtree<D> &tree, uint32_t id_peak, std::ofstream &f) {
SmallVec<double, D> tmp_j {0};
SmallVec<double, D> tmp_dr {0};
SmallVec<double, D> tmp_dv {0};
double quarter_Hill_radius = 0.25 * Hill_radius;
double Hill_units {total_mass * Hill_radius * Hill_radius * progIO->numerical_parameters.Omega};
SmallVec<double, D> shear_vector (0., progIO->numerical_parameters.q * progIO->numerical_parameters.Omega, 0.);
std::vector<double> accumulated_m (indices.size(), 0);
std::vector<double> accumulated_Jz (indices.size(), 0);
std::vector<std::pair<double, double>> xyJz (indices.size(), std::pair<double, double>(0, 0));

size_t idx = 0;
for (auto it : indices) {
tmp_dr = tree.particle_list[it].pos - center_of_mass;
tmp_dv = tree.particle_list[it].vel - shear_vector * tree.particle_list[it].pos[0]
- (vel_com - shear_vector * center_of_mass[0]);
tmp_j = tmp_dr.Cross(tmp_dv);

tmp_j[0] -= progIO->numerical_parameters.Omega * tmp_dr[0] * tmp_dr[2];
tmp_j[1] -= progIO->numerical_parameters.Omega * tmp_dr[1] * tmp_dr[2];
tmp_j[2] += progIO->numerical_parameters.Omega * (tmp_dr[0]*tmp_dr[0] + tmp_dr[1]*tmp_dr[1]);

if (particles[idx].second < quarter_Hill_radius) {
accumulated_J_in_quarter_Hill_radius += tree.particle_list[it].mass * tmp_j;
}

J += tree.particle_list[it].mass * tmp_j;
accumulated_Jz[idx] = tree.particle_list[it].mass * tmp_j[2];
accumulated_m[idx] = tree.particle_list[it].mass;
if (idx > 0) {
accumulated_Jz[idx] += accumulated_Jz[idx-1];
accumulated_m[idx] += accumulated_m[idx-1];
}
xyJz[idx].first = std::sqrt(tmp_dr[0]*tmp_dr[0]+tmp_dr[1]*tmp_dr[1]);
xyJz[idx].second = tree.particle_list[it].mass * tmp_j[2];
idx++;
}
std::sort(xyJz.begin(), xyJz.end(), [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
return a.first < b.first;
});

f.unsetf(std::ios_base::floatfield);
f << id_peak << ' ' << indices.size() << std::endl;
idx = 0;
for (auto it : particles) {
if (idx > 0) {
xyJz[idx].second += xyJz[idx-1].second;
}
f << std::scientific << std::setprecision(12) << std::setw(20) << it.second/Hill_radius << std::setw(20) << accumulated_m[idx]/total_mass << std::setw(20) << accumulated_Jz[idx]/Hill_units << std::setw(20) << xyJz[idx].first/Hill_radius  << std::setw(20) << xyJz[idx].second/Hill_units << std::endl;
idx++;
}

J /= Hill_units;
accumulated_J_in_quarter_Hill_radius /= Hill_units;
}
};


template <int D>
class PlanetesimalList {
private:

public:

uint32_t num_planetesimals {0};


double density_threshold {0};


double clump_mass_threshold {0};




double clump_diffuse_threshold {0.55};


double Hill_fraction_for_merge {0.25};


double peak_density_threshold {0};


std::vector<std::pair<uint32_t, double>> peaks_and_masses;


std::map<uint32_t, Planetesimal<D>> planetesimals;


ParticleSet<D> clump_set;


BHtree<D> clump_tree;


bool IsGravitationallyBound(const Planetesimal<D> &p1, const Planetesimal<D> &p2) {
double total_mass = p1.total_mass + p2.total_mass;
double P_grav = - progIO->numerical_parameters.grav_constant / (p1.center_of_mass - p2.center_of_mass).Norm();
double E_k = 0.5 / total_mass * (p1.vel_com - progIO->numerical_parameters.shear_vector * (p1.center_of_mass[0] - p2.center_of_mass[0]) - p2.vel_com).Norm2();
return P_grav + E_k < 0.;
}


template <class T, class F>
bool IsSaddlePointDeepEnough(DataSet<T, D> &ds, F DensityKernel, const Planetesimal<D> &p1, const Planetesimal<D> &p2, double saddle_threshold=2.5) {
auto r12 = p2.center_of_mass - p1.center_of_mass;
r12 *= p1.one10th_radius / (p1.one10th_radius + p2.one10th_radius);
auto possible_saddle_point = p1.center_of_mass + r12;

double radius_Kth_NN = 0;
auto *indices = new uint32_t[progIO->numerical_parameters.num_neighbors_in_knn_search];
std::vector<std::pair<int, double>> local_heaps;
ds.tree.KNN_Search_OpenMP(possible_saddle_point, progIO->numerical_parameters.num_neighbors_in_knn_search, radius_Kth_NN, indices, local_heaps, std::min(p1.one10th_radius, p2.one10th_radius));
double saddle_density = DensityKernel(ds, radius_Kth_NN, 0, indices, local_heaps);
delete [] indices;
indices = nullptr;
return saddle_density < saddle_threshold * ds.planetesimal_list.density_threshold;
}

template <class T, class F>
bool IsHillSaddlePointDeepEnough(DataSet<T, D> &ds, F DensityKernel, const Planetesimal<D> &p1, const Planetesimal<D> &p2, double saddle_threshold=2.5) {
auto r12 = p2.center_of_mass - p1.center_of_mass;
r12 *= p1.Hill_radius / (p1.Hill_radius + p2.Hill_radius);
auto possible_saddle_point = p1.center_of_mass + r12;

double radius_Kth_NN = 0;
auto *indices = new uint32_t[progIO->numerical_parameters.num_neighbors_in_knn_search];
std::vector<std::pair<int, double>> local_heaps;
ds.tree.KNN_Search_OpenMP(possible_saddle_point, progIO->numerical_parameters.num_neighbors_in_knn_search, radius_Kth_NN, indices, local_heaps, std::min(p1.Hill_radius, p2.Hill_radius) / Hill_fraction_for_merge);
double saddle_density = DensityKernel(ds, radius_Kth_NN, 0, indices, local_heaps);
delete [] indices;
indices = nullptr;
return saddle_density < saddle_threshold * ds.planetesimal_list.density_threshold;
}

template <class T>
bool IsPhaseSpaceDistanceWithinTenSigma(DataSet<T, D> &ds, const Planetesimal<D> &p1, const Planetesimal<D> &p2) {
const Planetesimal<D> &small_p = (p1.total_mass > p2.total_mass)? p2 : p1;
const Planetesimal<D> &large_p = (p1.total_mass > p2.total_mass)? p1 : p2;
double sigma2_pos = 0, sigma2_vel = 0, small_n = small_p.particles.size();
double phase_dist2 = 0;
typename BHtree<D>::InternalParticle *p;
for (auto &par: small_p.particles) {
sigma2_pos += par.second*par.second;
p = &ds.tree.particle_list[par.first];
sigma2_vel += (p->vel - small_p.vel_com - progIO->numerical_parameters.shear_vector * (p->pos[0] - small_p.center_of_mass[0]) ).Norm2();
}
sigma2_pos /= small_n;
sigma2_vel /= small_n;

phase_dist2 = (large_p.center_of_mass - small_p.center_of_mass).Norm2() / (sigma2_pos / small_n)
+(large_p.vel_com - small_p.vel_com - progIO->numerical_parameters.shear_vector
* (large_p.center_of_mass[0] - small_p.center_of_mass[0])).Norm2() / (sigma2_vel / small_n);
return phase_dist2 < 200.; 
}


void WriteBasicResults(int loop_count) {
if (loop_count == mpi->loop_begin) {
progIO->out_content << std::setw(progIO->width) << std::setfill(' ') << "#time" << std::setw(progIO->width) << "N_peak" << std::setw(progIO->width) << std::setfill(' ') << "M_p,max,code" << std::setw(progIO->width) << std::setfill(' ') << "M_p,tot,code" << std::setw(progIO->width) << std::setfill(' ') << "M_p,max,frac" << std::setw(progIO->width) << std::setfill(' ') << "M_p,tot,frac" << std::setw(progIO->width) << std::setfill(' ') << "M_p,max,Ceres" << std::setw(progIO->width) << std::setfill(' ') << "M_p,tot,Ceres" << std::endl;
mpi->WriteSingleFile(mpi->result_files[mpi->file_pos[progIO->file_name.planetesimals_file]], progIO->out_content, __master_only);
}

progIO->out_content <<  std::setw(progIO->width) << std::setfill(' ') << std::scientific << progIO->physical_quantities[loop_count].time;
progIO->out_content << std::setw(progIO->width) << std::setfill(' ') << num_planetesimals;
for (auto &it : planetesimals) {
progIO->physical_quantities[loop_count].max_planetesimal_mass = std::max(progIO->physical_quantities[loop_count].max_planetesimal_mass, it.second.total_mass);
progIO->physical_quantities[loop_count].total_planetesimal_mass += it.second.total_mass;
}
progIO->out_content << std::setw(progIO->width) << std::setfill(' ') << progIO->physical_quantities[loop_count].max_planetesimal_mass << std::setw(progIO->width) << std::setfill(' ') << progIO->physical_quantities[loop_count].total_planetesimal_mass;
progIO->out_content << std::setw(progIO->width) << std::setfill(' ') << progIO->physical_quantities[loop_count].max_planetesimal_mass / progIO->numerical_parameters.mass_total_code_units << std::setw(progIO->width) << std::setfill(' ') << progIO->physical_quantities[loop_count].total_planetesimal_mass / progIO->numerical_parameters.mass_total_code_units;
progIO->out_content << std::setw(progIO->width) << std::setfill(' ') << progIO->physical_quantities[loop_count].max_planetesimal_mass * progIO->numerical_parameters.mass_physical / progIO->numerical_parameters.mass_ceres << std::setw(progIO->width) << std::setfill(' ') << progIO->physical_quantities[loop_count].total_planetesimal_mass * progIO->numerical_parameters.mass_physical / progIO->numerical_parameters.mass_ceres;
progIO->out_content << std::endl;
mpi->WriteSingleFile(mpi->result_files[mpi->file_pos[progIO->file_name.planetesimals_file]], progIO->out_content, __all_processors);
}


template <class T>
void OutputPlanetesimalsInfo(int loop_count, DataSet<T, D> &ds) {
std::ofstream file_planetesimals;
std::ostringstream tmp_ss;
tmp_ss << std::setprecision(3) << std::fixed << std::setw(7) << std::setfill('0') << progIO->physical_quantities[loop_count].time;
std::string tmp_file_name;
if (progIO->file_name.output_file_path.find_last_of('/') != std::string::npos) {
tmp_file_name = progIO->file_name.output_file_path.substr(0, progIO->file_name.output_file_path.find_last_of('/')) + std::string("/peaks_at_") + tmp_ss.str() + std::string(".txt");
} else {
tmp_file_name = std::string("peaks_at_") + tmp_ss.str() + std::string(".txt");
}

file_planetesimals.open(tmp_file_name, std::ofstream::out);
if (!(file_planetesimals.is_open())) {
progIO->error_message << "Error: Failed to open file " << tmp_file_name << " due to " << std::strerror(errno) << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}

std::stringstream tmp_ss_id;
if (progIO->flags.save_clumps_flag) {
char mkdir_cmd[500] = "mkdir -p ParList.";
tmp_ss_id << std::setw(4) << std::setfill('0') << loop_count * progIO->interval + progIO->start_num;
std::strcat(mkdir_cmd, tmp_ss_id.str().c_str());
if (std::system(mkdir_cmd) == -1) {
progIO->error_message << "Error: Failed to execute: " << mkdir_cmd << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}
}



file_planetesimals << std::setw(12) << "#peak_id"
<< std::setw(12) << "Npar"
<< std::setw(24) << "total_mass"
<< std::setw(24) << "Hill_radius"
<< std::setw(24) << "center_of_mass[x]"
<< std::setw(24) << "center_of_mass[y]"
<< std::setw(24) << "center_of_mass[z]"
<< std::setw(24) << "vel_COM[x]"
<< std::setw(24) << "vel_COM[y]"
<< std::setw(24) << "vel_COM[z]"
<< std::setw(24) << "J[x]"
<< std::setw(24) << "J[y]"
<< std::setw(24) << "J[z]/(M R_H^2 Omega)"
<< std::setw(24) << "geo_mean_offset[x]"
<< std::setw(24) << "geo_mean_offset[y]"
<< std::setw(24) << "geo_mean_offset[z]"
<< std::setw(24) << "median_offset[x]"
<< std::setw(24) << "median_offset[y]"
<< std::setw(24) << "median_offset[z]"
<< std::endl;
for (auto peak : peaks_and_masses) {
auto it = planetesimals.find(peak.first);
auto tmp_num_particles = it->second.indices.size();
sn::dvec geo_mean_offset, median_offset;
for (auto item : it->second.indices) {
auto tmp_it = ds.tree.sink_particle_indices.find(item);
if (tmp_it != ds.tree.sink_particle_indices.end()) {
tmp_num_particles += tmp_it->second.size()-1;
}
}
std::vector<std::vector<double>> offset;
offset.resize(3);
for (auto &item : offset) {
item.resize(tmp_num_particles);
}
uint32_t idx = 0;
for (auto item : it->second.indices) {
auto tmp_it = ds.tree.sink_particle_indices.find(item);
if (tmp_it != ds.tree.sink_particle_indices.end()) {
for (auto &tmp_sink_it : tmp_it->second) {
for (size_t d = 0; d < 3; d++) {
offset[d][idx] = tmp_sink_it.pos[d] - it->second.center_of_mass[d];
}
idx++;
}
} else {
for (size_t d = 0; d < 3; d++) {
offset[d][idx] = ds.tree.particle_list[item].pos[d] - it->second.center_of_mass[d];
}
idx++;
}
}

auto tmp_J = sn::dvec(it->second.J[0], it->second.J[1], it->second.J[2]);
double tmp_theta = std::acos(it->second.J[2] / tmp_J.Norm());
double tmp_phi = std::atan2(it->second.J[1], it->second.J[0]);

double rot_z[3][3] = {{std::cos(tmp_phi), -std::sin(tmp_phi), 0},
{std::sin(tmp_phi),  std::cos(tmp_phi), 0},
{                0,                  0, 1}};
double rot_y[3][3] = {{ std::cos(tmp_theta), 0, std::sin(tmp_theta)},
{                   0, 1,                   0},
{-std::sin(tmp_theta), 0, std::cos(tmp_theta)}};
double rot[3][3];
for (size_t d1 = 0; d1 < 3; d1++) {
for (size_t d2 = 0; d2 < 3; d2++) {
rot[d1][d2] = 0;
for (size_t d = 0; d < 3; d++) {
rot[d1][d2] += rot_z[d1][d] * rot_y[d][d2];
}
}
}

for (idx = 0; idx < tmp_num_particles; idx++) {
double tmp_offset[3] = {offset[0][idx], offset[1][idx], offset[2][idx]};
for (size_t d = 0; d < 3; d++) {
offset[d][idx] = 0;
for (size_t d1 = 0; d1 < 3; d1++) {
offset[d][idx] += tmp_offset[d1] * rot[d1][d];
}
}
}

auto half_size_offset = tmp_num_particles / 2;
bool is_even = !(tmp_num_particles & 1);
for (size_t d = 0; d < 3; d++) {
for (auto &item : offset[d]) item = std::abs(item);
geo_mean_offset[d] = std::pow(10.0, std::accumulate(offset[d].begin(), offset[d].end(), 0., [](const double &a, const double &b) {
if (b < 1e-32) {
return a - 32;
} else {
return a + std::log10(b);
}
}) / tmp_num_particles);
std::nth_element(offset[d].begin(), offset[d].begin() + half_size_offset, offset[d].end());
median_offset[d] = offset[d][half_size_offset];
if (is_even) {
median_offset[d] = (median_offset[d] + *std::max_element(offset[d].begin(), offset[d].begin() + half_size_offset)) / 2.0;
}
}


if (progIO->flags.save_clumps_flag) {
OutputSinglePlanetesimal(std::string("ParList.")+tmp_ss_id.str()+std::string("/")+std::to_string(it->first)+std::string(".txt"), it->first, ds);
}
file_planetesimals << std::setw(12) << it->first << std::setw(12) << tmp_num_particles
<< std::setprecision(16)
<< std::setw(24) << it->second.total_mass
<< std::setw(24) << it->second.Hill_radius
<< std::setw(24) << it->second.center_of_mass[0]
<< std::setw(24) << it->second.center_of_mass[1]
<< std::setw(24) << it->second.center_of_mass[2]
<< std::setw(24) << it->second.vel_com[0]
<< std::setw(24) << it->second.vel_com[1]
<< std::setw(24) << it->second.vel_com[2]
<< std::setw(24) << it->second.J[0]
<< std::setw(24) << it->second.J[1]
<< std::setw(24) << it->second.J[2]
<< std::setw(24) << geo_mean_offset[0]
<< std::setw(24) << geo_mean_offset[1]
<< std::setw(24) << geo_mean_offset[2]
<< std::setw(24) << median_offset[0]
<< std::setw(24) << median_offset[1]
<< std::setw(24) << median_offset[2]
<< std::endl;
}
file_planetesimals.close(); /
}


template<class T>
void OutputSinglePlanetesimal(const std::string &file_name, uint32_t peak_id, DataSet<T, D> &ds, size_t precision=16) {
auto width = precision + 8;
std::ofstream file_single_clump;
auto search_it = planetesimals.find(peak_id);
if (search_it != planetesimals.end()) {
file_single_clump.open(file_name);
if (!(file_single_clump.is_open())) {
progIO->error_message << "Error: Failed to open file: " << file_name << ". But we proceed." << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}
uint32_t skip_for_sub_sampling = 0;
for (auto it : search_it->second.particles) {
if (skip_for_sub_sampling > 0) {
skip_for_sub_sampling--;
continue;
} else {
skip_for_sub_sampling = progIO->save_clump_sampling_rate - 1;
}
file_single_clump.unsetf(std::ios_base::floatfield);
file_single_clump << std::setw(precision) << ds.tree.particle_list[it.first].original_id;
file_single_clump << std::scientific;
for (int i = 0; i != D; i++) {
file_single_clump << std::setprecision(precision) << std::setw(width) << ds.tree.particle_list[it.first].pos[i];
}
for (int i = 0; i != D; i++) {
file_single_clump << std::setprecision(precision) << std::setw(width) << ds.tree.particle_list[it.first].vel[i];
}
file_single_clump << std::setprecision(precision) << std::setw(width) << ds.tree.particle_list[it.first].mass << std::endl;
}
file_single_clump.close();
}
}


void OutputParticlesByIndices(const std::string &file_name, const std::vector<uint32_t> &indices, const ParticleSet<D> &particle_set) {
std::ofstream file_particles;
file_particles.open(file_name);
if (!(file_particles.is_open())) {
progIO->error_message << "Error: Failed to open file: " << file_name << ". But we proceed. " << std::endl;
progIO->Output(std::cerr, progIO->error_message, __normal_output, __all_processors);
}

for (auto it : indices) {
file_particles.unsetf(std::ios_base::floatfield);
file_particles << std::setw(16) << particle_set[it].id;
file_particles << std::scientific;
for (int i = 0; i != D; i++) {
file_particles << std::setprecision(16) << std::setw(24) << particle_set[it].pos[i];
}
for (int i = 0; i != D; i++) {
file_particles << std::setprecision(16) << std::setw(24) << particle_set[it].vel[i];
}
file_particles <<  std::setprecision(16) << std::setw(24) << progIO->numerical_parameters.mass_per_particle[particle_set[it].property_index] << std::endl;
}
file_particles.close();
}


void BuildClumpTree(sn::dvec &root_center, double half_width, double &max_radius, bool Hill=false) {
clump_set.Reset();
clump_set.num_total_particles = static_cast<uint32_t>(num_planetesimals);
clump_set.num_particles = static_cast<uint32_t>(num_planetesimals);
clump_set.AllocateSpace(clump_set.num_total_particles);

uint32_t tmp_id = 0;
Particle<D> *p;
for (auto &it : planetesimals) {
if (it.second.mask) {
p = &clump_set[tmp_id];
p->pos = it.second.center_of_mass;
p->vel = it.second.vel_com;
p->id = tmp_id;
p->density = it.first; 
p->property_index = 0; 
tmp_id++;
if (Hill) {
max_radius = MaxOf(it.second.Hill_radius * Hill_fraction_for_merge, max_radius);
} else {
max_radius = MaxOf(it.second.one10th_radius, max_radius);
}
}
}

clump_tree.root_center = root_center;
clump_tree.half_width = half_width;
clump_tree.BuildTree(progIO->numerical_parameters, clump_set, true, false);
if (clump_tree.sink_particle_indices.size() > 0) {
progIO->log_info << "(Warning: got " << clump_tree.sink_particle_indices.size() << " sink clumps while building clump tree. Proceed for now.) ";
}
}

};







template <class T, int D>
class DataSet {
private:

public:


VtkData<T, D> vtk_data;


ParticleSet<3> particle_set;


BHtree<D> tree;


PlanetesimalList<D> planetesimal_list;

};


#endif 
