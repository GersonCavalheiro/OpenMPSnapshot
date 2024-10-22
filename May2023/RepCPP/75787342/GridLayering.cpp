#include "GridLayering.h"
#ifdef SUPPORT_OMP
#include "omp.h"
#endif
#include <ogrsf_frmts.h>
#include <queue>
#include <set>

using std::queue;
using std::set;

int find_flow_direction_index_ccw(const int fd) {
for (int i = 1; i <= 8; i++) {
if (fdccw[i] == fd) return i;
}
return -1;
}

int find_flow_direction_index_ccw(const int delta_row, const int delta_col) {
for (int i = 1; i <= 8; i++) {
if (drow[i] == delta_row && dcol[i] == delta_col) return i;
}
return -1;
}

int get_reversed_fdir(const int fd) {
int fd_idx = find_flow_direction_index_ccw(fd);
if (fd_idx < 0) return -1;
int reversed_fd_idx = fd_idx > 4 ? fd_idx - 4 : fd_idx + 4;
return fdccw[reversed_fd_idx];
}

vector<int> uncompress_flow_directions(const int compressed_fd) {
vector<int> flow_dirs; 
if (compressed_fd <= 0) return flow_dirs;
if (compressed_fd & 1) {
flow_dirs.emplace_back(1);
}
if (compressed_fd & 128) {
flow_dirs.emplace_back(128);
}
if (compressed_fd & 64) {
flow_dirs.emplace_back(64);
}
if (compressed_fd & 32) {
flow_dirs.emplace_back(32);
}
if (compressed_fd & 16) {
flow_dirs.emplace_back(16);
}
if (compressed_fd & 8) {
flow_dirs.emplace_back(8);
}
if (compressed_fd & 4) {
flow_dirs.emplace_back(4);
}
if (compressed_fd & 2) {
flow_dirs.emplace_back(2);
}
if (flow_dirs.size() >= 2 && flow_dirs[0] == 1 && flow_dirs.back() == 2) {
flow_dirs[0] = 2;
flow_dirs.back() = 1;
}
return flow_dirs;
}

bool read_stream_vertexes_as_rowcol(string stream_file, FloatRaster* mask, vector<vector<ROW_COL> >& stream_rc) {
#if GDAL_VERSION_MAJOR >= 2
GDALDataset *stream_ds = nullptr;
stream_ds = static_cast<GDALDataset*>(GDALOpenEx(stream_file.c_str(),
GA_ReadOnly | GDAL_OF_VECTOR,
nullptr, nullptr, nullptr));
#else
OGRRegisterAll();
OGRDataSource* stream_ds = OGRSFDriverRegistrar::Open(stream_file.c_str(), FALSE);
#endif
if (nullptr == stream_ds) {
cout << "Read stream shapefile failed!" << endl;
return false;
}
OGRLayer *stream_lyr = stream_ds->GetLayer(0);
if (wkbFlatten(stream_lyr->GetGeomType() != wkbLineString)) {
cout << "The stream shapefile MUST be LINE!" << endl;
return false;
}
vint ft_count = stream_lyr->GetFeatureCount();
stream_lyr->ResetReading();
for (vint i = 0; i < ft_count; i++) {
OGRFeature *ft = stream_lyr->GetNextFeature();
OGRGeometry *poGeometry = nullptr;
OGRPoint ptTemp;
poGeometry = ft->GetGeometryRef();
if (nullptr == poGeometry) { continue; }
if (wkbFlatten(poGeometry->getGeometryType()) != wkbLineString) { continue; }
#if GDAL_VERSION_MAJOR >= 2 && GDAL_VERSION_MINOR >= 3
OGRLineString *poLineString = poGeometry->toLineString();
#else
OGRLineString *poLineString = dynamic_cast<OGRLineString*>(poGeometry);
#endif
int NumberOfVertices = poLineString->getNumPoints();
vector<ROW_COL> tmp_rc;
for (int k = 0; k < NumberOfVertices; k++) {
poLineString->getPoint(k, &ptTemp);
ROW_COL rc = mask->GetPositionByCoordinate(ptTemp.getX(), ptTemp.getY());
if (rc.first == -1 && rc.second == -1) { continue; } 
if (mask->GetPosition(rc.first, rc.second) < 0) { continue; } 
tmp_rc.emplace_back(rc);
}
if (!tmp_rc.empty()) { stream_rc.emplace_back(tmp_rc); }
OGRFeature::DestroyFeature(ft);
}
if (stream_rc.empty()) { return false; }
#if GDAL_VERSION_MAJOR >= 2
GDALClose(stream_ds);
#else
OGRDataSource::DestroyDataSource(stream_ds);
#endif
return true;
}

void print_flow_fractions_mfdmd(FloatRaster* ffrac, int row, int col) {
for (int i = 1; i <= 8; i++) {
cout << ffrac->GetValue(row, col, i) << ",";
}
}


#ifdef USE_MONGODB
GridLayering::GridLayering(const int id, MongoGridFs* gfs, const char* out_dir) :
gfs_(gfs), use_mongo_(true), has_mask_(false), fdtype_(FD_D8), output_dir_(out_dir), subbasin_id_(id),
n_rows_(-1), n_cols_(-1), out_nodata_(-9999.f),
n_valid_cells_(-1), n_layer_count_(-1), pos_index_(nullptr), pos_rowcol_(nullptr),
mask_(nullptr), flowdir_(nullptr), flowdir_matrix_(nullptr), reverse_dir_(nullptr),
flow_in_num_(nullptr), flow_in_acc_(nullptr), flow_in_count_(0), flow_in_cells_(nullptr),
flow_out_num_(nullptr), flow_out_acc_(nullptr), flow_out_count_(0), flow_out_cells_(nullptr),
layers_updown_(nullptr), layers_downup_(nullptr), layers_evenly_(nullptr),
layer_cells_updown_(nullptr), layer_cells_downup_(nullptr), layer_cells_evenly_(nullptr) {
}
#endif

GridLayering::GridLayering(const int id, const char* out_dir):
gfs_(nullptr), use_mongo_(false), has_mask_(false), fdtype_(FD_D8), output_dir_(out_dir), subbasin_id_(id),
n_rows_(-1), n_cols_(-1), out_nodata_(-9999.f),
n_valid_cells_(-1), n_layer_count_(-1), pos_index_(nullptr), pos_rowcol_(nullptr),
mask_(nullptr), flowdir_(nullptr), flowdir_matrix_(nullptr), reverse_dir_(nullptr),
flow_in_num_(nullptr), flow_in_acc_(nullptr), flow_in_count_(0), flow_in_cells_(nullptr),
flow_out_num_(nullptr), flow_out_acc_(nullptr), flow_out_count_(0), flow_out_cells_(nullptr),
layers_updown_(nullptr), layers_downup_(nullptr), layers_evenly_(nullptr),
layer_cells_updown_(nullptr), layer_cells_downup_(nullptr), layer_cells_evenly_(nullptr) {
}

GridLayering::~GridLayering() {
delete flowdir_; 
if (has_mask_) delete mask_;
if (nullptr != pos_index_) Release1DArray(pos_index_);
if (nullptr != reverse_dir_) Release1DArray(reverse_dir_);
if (nullptr != flow_in_num_) Release1DArray(flow_in_num_);
if (nullptr != flow_in_acc_) Release1DArray(flow_in_acc_);
if (nullptr != flow_in_cells_) Release1DArray(flow_in_cells_);
if (nullptr != flow_out_num_) Release1DArray(flow_out_num_);
if (nullptr != flow_out_acc_) Release1DArray(flow_out_acc_);
if (nullptr != flow_out_cells_) Release1DArray(flow_out_cells_);
if (nullptr != layers_updown_) Release1DArray(layers_updown_);
if (nullptr != layers_downup_) Release1DArray(layers_downup_);
if (nullptr != layers_evenly_) Release1DArray(layers_evenly_);
if (nullptr != layer_cells_updown_) Release1DArray(layer_cells_updown_);
if (nullptr != layer_cells_downup_) Release1DArray(layer_cells_downup_);
if (nullptr != layer_cells_evenly_) Release1DArray(layer_cells_evenly_);
}

bool GridLayering::Execute() {
if (!LoadData()) return false;
CalPositionIndex();
return OutputFlowOut() && OutputFlowIn() &&
GridLayeringFromSource() && GridLayeringFromOutlet(); 
}

void GridLayering::CalPositionIndex() {
if (n_valid_cells_ > 0 && nullptr != pos_index_) return;
Initialize1DArray(n_rows_ * n_cols_, pos_index_, -1);
for (int i = 0; i < n_valid_cells_; i++) {
pos_index_[pos_rowcol_[i][0] * n_cols_ + pos_rowcol_[i][1]] = i;
}
}

void GridLayering::GetReverseDirMatrix() {
if (nullptr == reverse_dir_) Initialize1DArray(n_valid_cells_, reverse_dir_, 0.f);
if (nullptr == flow_in_num_) Initialize1DArray(n_valid_cells_, flow_in_num_, 0);
if (nullptr == flow_in_acc_) Initialize1DArray(n_valid_cells_, flow_in_acc_, 0);
for (int valid_idx = 0; valid_idx < n_valid_cells_; valid_idx++) {
int i = pos_rowcol_[valid_idx][0]; 
int j = pos_rowcol_[valid_idx][1]; 
int flow_dir = CVT_INT(flowdir_matrix_[valid_idx]);
if (flowdir_->IsNoData(i, j) || flow_dir <= 0) {
if (FloatEqual(reverse_dir_[valid_idx], 0.f)) {
reverse_dir_[valid_idx] = out_nodata_;
}
continue;
}
vector<int> flow_dirs = uncompress_flow_directions(flow_dir);
for (vector<int>::iterator it = flow_dirs.begin(); it != flow_dirs.end(); ++it) {
int fd_idx = find_flow_direction_index_ccw(*it);
int dst_row = i + drow[fd_idx];
int dst_col = j + dcol[fd_idx];
if (!mask_->ValidateRowCol(dst_row, dst_col) ||
mask_->IsNoData(dst_row, dst_col)) {
continue;
}
int dst_idx = pos_index_[dst_row * n_cols_ + dst_col];
if (FloatEqual(reverse_dir_[dst_idx], out_nodata_) || reverse_dir_[dst_idx] < 0) {
reverse_dir_[dst_idx] = 0;
}
reverse_dir_[dst_idx] += CVT_FLT(get_reversed_fdir(fdccw[fd_idx]));
flow_in_num_[dst_idx] += 1;
}
}
flow_in_acc_[0] = flow_in_num_[0];
flow_in_count_ = flow_in_num_[0];
for (int idx = 1; idx < n_valid_cells_; idx++) {
flow_in_count_ += flow_in_num_[idx];
flow_in_acc_[idx] = flow_in_num_[idx] + flow_in_acc_[idx - 1];
}
}

int GridLayering::BuildMultiFlowOutArray(float*& compressed_dir,
int*& connect_count, float*& p_output) {
p_output[0] = CVT_FLT(n_valid_cells_);
int counter = 1;
for (int valid_idx = 0; valid_idx < n_valid_cells_; valid_idx++) {
int i = pos_rowcol_[valid_idx][0]; 
int j = pos_rowcol_[valid_idx][1]; 
p_output[counter++] = CVT_FLT(connect_count[valid_idx]); 
if (connect_count[valid_idx] == 0) continue;
vector<int> flow_dirs = uncompress_flow_directions(CVT_INT(compressed_dir[valid_idx]));
for (vector<int>::iterator it = flow_dirs.begin(); it != flow_dirs.end(); ++it) {
int fd_idx = find_flow_direction_index_ccw(*it);
if (!mask_->ValidateRowCol(i + drow[fd_idx], j + dcol[fd_idx]) ||
mask_->IsNoData(i + drow[fd_idx], j + dcol[fd_idx])) {
continue;
}
p_output[counter++] = CVT_FLT(pos_index_[(i + drow[fd_idx]) * n_cols_ + j + dcol[fd_idx]]);
}
}
return counter;
}

bool GridLayering::BuildFlowInCellsArray() {
int n_output = flow_in_count_ + n_valid_cells_ + 1;
if (nullptr == flow_in_cells_) Initialize1DArray(n_output, flow_in_cells_, 0.f);
int n_output2 = BuildMultiFlowOutArray(reverse_dir_, flow_in_num_, flow_in_cells_);
if (n_output2 != n_output) {
cout << "BuildFlowInCellsArray failed!" << endl;
return false;
}
return true;
}

void GridLayering::CountFlowOutCells() {
if (nullptr == flow_out_num_) Initialize1DArray(n_valid_cells_, flow_out_num_, 0);
if (nullptr == flow_out_acc_) Initialize1DArray(n_valid_cells_, flow_out_acc_, 0);
#pragma omp parallel for
for (int index = 0; index < n_valid_cells_; index++) {
int i = pos_rowcol_[index][0]; 
int j = pos_rowcol_[index][1]; 
if (flowdir_->IsNoData(i, j) || flowdir_matrix_[index] <= 0) continue;

int flow_dir = CVT_INT(flowdir_matrix_[index]);
vector<int> flow_dirs = uncompress_flow_directions(flow_dir);
for (vector<int>::iterator it = flow_dirs.begin(); it != flow_dirs.end(); ++it) {
int fd_idx = find_flow_direction_index_ccw(*it);
if (mask_->ValidateRowCol(i + drow[fd_idx], j + dcol[fd_idx]) &&
!mask_->IsNoData(i + drow[fd_idx], j + dcol[fd_idx])) {
flow_out_num_[index]++;
}
}
}
flow_out_acc_[0] = flow_out_num_[0];
flow_out_count_ = flow_out_num_[0];
for (int idx = 1; idx < n_valid_cells_; idx++) {
flow_out_count_ += flow_out_num_[idx];
flow_out_acc_[idx] = flow_out_num_[idx] + flow_out_acc_[idx - 1];
}
}

bool GridLayering::BuildFlowOutCellsArray() {
int n_output = flow_out_count_ + n_valid_cells_ + 1;
if (nullptr == flow_out_cells_) Initialize1DArray(n_output, flow_out_cells_, 0.f);
int n_output2 = BuildMultiFlowOutArray(flowdir_matrix_, flow_out_num_,
flow_out_cells_);
if (n_output2 != n_output) {
cout << "BuildFlowOutCellsArray failed!" << endl;
return false;
}
return true;
}

bool GridLayering::Output2DimensionArrayTxt(const string& name, string& header,
float* const matrix, float* matrix2) {
string outpath = string(output_dir_) + SEP + name + ".txt";
std::ofstream ofs(outpath.c_str());
ofs << matrix[0] << endl;
ofs << header << endl;
int tmp_count = 1;
int tmp_count2 = 1;
for (int i = 0; i < CVT_INT(matrix[0]); i++) {
int count = CVT_INT(matrix[tmp_count++]);
ofs << i << "\t" << count << "\t";
for (int j = 0; j < count; j++) {
if (j == count - 1)
ofs << matrix[tmp_count++];
else
ofs << matrix[tmp_count++] << ",";
}
if (nullptr != matrix2) {
ofs << "\t";
tmp_count2 = tmp_count - count;
for (int k = 0; k < count; k++) {
if (k == count - 1)
ofs << matrix2[tmp_count2++];
else
ofs << matrix2[tmp_count2++] << ",";
}
}
ofs << "\n"; 
}
ofs.close();
return true;
}

#ifdef USE_MONGODB
bool GridLayering::OutputArrayAsGfs(const string& name, const vint length, float* const matrix) {
bool flag = false;
int max_loop = 3;
int cur_loop = 1;
while (cur_loop < max_loop) {
if (!OutputToMongodb(name.c_str(), length, reinterpret_cast<char*>(matrix))) {
cur_loop++;
} else {
cout << "Output " << name << " done!" << endl;
flag = true;
break;
}
}
return flag;
}
#endif

bool GridLayering::OutputFlowIn() {
GetReverseDirMatrix();
if (!BuildFlowInCellsArray()) return false;
string header = "ID\tUpstreamCount\tUpstreamID";
bool done = Output2DimensionArrayTxt(flowin_index_name_, header, flow_in_cells_);
if (use_mongo_) {
#ifdef USE_MONGODB
done = done && OutputArrayAsGfs(flowin_index_name_,
n_valid_cells_ + flow_in_count_ + 1,
flow_in_cells_);
#endif
}
return done;
}


bool GridLayering::OutputFlowOut() {
CountFlowOutCells();
if (!BuildFlowOutCellsArray()) return false;
string header = "ID\tDownstreamCount\tDownstreamID";
bool done = Output2DimensionArrayTxt(flowout_index_name_, header, flow_out_cells_);
if (use_mongo_) {
#ifdef USE_MONGODB
done = OutputArrayAsGfs(flowout_index_name_, flow_out_count_ + n_valid_cells_ + 1,
flow_out_cells_);
#endif
}
return done;
}

bool GridLayering::GridLayeringFromSource() {
Initialize1DArray(n_valid_cells_, layers_updown_, out_nodata_);
int* flow_in_num_copy = nullptr;
Initialize1DArray(n_valid_cells_, flow_in_num_copy, flow_in_num_);
int* last_layer = nullptr;
Initialize1DArray(n_valid_cells_, last_layer, out_nodata_);
int num_last_layer = 0; 

for (int i = 0; i < n_valid_cells_; i++) {
if (flow_in_num_copy[i] == 0 && flowdir_matrix_[i] > 0) {
last_layer[num_last_layer++] = i;
}
}
int num_next_layer = 0;
int cur_num = 0;
int* next_layer = nullptr;
Initialize1DArray(n_valid_cells_, next_layer, out_nodata_);
int* tmp = nullptr; 
int valid_idx = 0;
vector<int> lyr_cells;
while (num_last_layer > 0) {
cur_num++;
num_next_layer = 0;
for (int i_in_layer = 0; i_in_layer < num_last_layer; i_in_layer++) {
valid_idx = last_layer[i_in_layer];
lyr_cells.emplace_back(valid_idx);
layers_updown_[valid_idx] = CVT_FLT(cur_num);
int dir = CVT_INT(flowdir_matrix_[valid_idx]);
if (dir <= 0) {
continue;
}
for (int out_idx = 0; out_idx < flow_out_num_[valid_idx]; out_idx++) {
int out_cellidx = 1 + valid_idx + 1 + out_idx;
if (valid_idx > 0) out_cellidx += flow_out_acc_[valid_idx - 1];
int dst_posidx = CVT_INT(flow_out_cells_[out_cellidx]);
if (--flow_in_num_copy[dst_posidx] == 0) {
next_layer[num_next_layer++] = dst_posidx;
}
}
}
vector<int>(lyr_cells).swap(lyr_cells);
n_layer_cells_updown_.emplace_back(vector<int>(lyr_cells));
lyr_cells.clear();

num_last_layer = num_next_layer;
tmp = last_layer;
last_layer = next_layer;
next_layer = tmp;
}
Release1DArray(last_layer);
Release1DArray(next_layer);

n_layer_count_ = CVT_INT(n_layer_cells_updown_.size());
int length = n_valid_cells_ + n_layer_count_ + 1;
Initialize1DArray(length, layer_cells_updown_, 0.f);
layer_cells_updown_[0] = CVT_FLT(n_layer_count_);
valid_idx = 1;
for (auto it = n_layer_cells_updown_.begin();
it != n_layer_cells_updown_.end(); ++it) {
layer_cells_updown_[valid_idx++] = CVT_FLT((*it).size());
for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
layer_cells_updown_[valid_idx++] = CVT_FLT(*it2);
}
}
Release1DArray(flow_in_num_copy);
return OutputGridLayering(layering_updown_name_, length,
layers_updown_, layer_cells_updown_);
}

bool GridLayering::GridLayeringFromOutlet() {
Initialize1DArray(n_valid_cells_, layers_downup_, out_nodata_);
int* flow_out_num_copy = nullptr;
Initialize1DArray(n_valid_cells_, flow_out_num_copy, flow_out_num_);
int* last_layer = nullptr;
Initialize1DArray(n_valid_cells_, last_layer, out_nodata_);
int num_last_layer = 0; 
for (int i = 0; i < n_valid_cells_; i++) {
if (flow_out_num_[i] == 0) {
last_layer[num_last_layer++] = i;
}
}
int num_next_layer = 0;
int* next_layer = nullptr;
Initialize1DArray(n_valid_cells_, next_layer, out_nodata_);
int* tmp = nullptr;
int cur_num = 0;
vector<int> lyr_cells;
while (num_last_layer > 0) {
cur_num++;
num_next_layer = 0;
for (int i_in_layer = 0; i_in_layer < num_last_layer; i_in_layer++) {
int valid_idx = last_layer[i_in_layer];
lyr_cells.emplace_back(valid_idx);
layers_downup_[valid_idx] = CVT_FLT(cur_num);
for (int in_idx = 0; in_idx < flow_in_num_[valid_idx]; in_idx++) {
int in_cellidx = 1 + valid_idx + 1 + in_idx;
if (valid_idx > 0) in_cellidx += flow_in_acc_[valid_idx - 1];
int src_posidx = CVT_INT(flow_in_cells_[in_cellidx]);
if (--flow_out_num_copy[src_posidx] == 0) {
next_layer[num_next_layer++] = src_posidx;
}
}
}
vector<int>(lyr_cells).swap(lyr_cells);
n_layer_cells_downup_.emplace_back(vector<int>(lyr_cells));
lyr_cells.clear();

num_last_layer = num_next_layer;
tmp = last_layer;
last_layer = next_layer;
next_layer = tmp;
}

#pragma omp parallel for
for (int i = 0; i < n_valid_cells_; i++) {
if (!FloatEqual(layers_downup_[i], out_nodata_)) {
layers_downup_[i] = CVT_FLT(cur_num) - layers_downup_[i] + 1.f;
}
}
Release1DArray(last_layer);
Release1DArray(next_layer);

n_layer_count_ = CVT_INT(n_layer_cells_downup_.size());
int length = n_valid_cells_ + n_layer_count_ + 1;
Initialize1DArray(length, layer_cells_downup_, 0.f);
layer_cells_downup_[0] = CVT_FLT(n_layer_count_);
int index = 1;
for (auto it = n_layer_cells_downup_.rbegin();
it != n_layer_cells_downup_.rend(); ++it) {
layer_cells_downup_[index++] = CVT_FLT((*it).size());
for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
layer_cells_downup_[index++] = CVT_FLT(*it2);
}
}
Release1DArray(flow_out_num_copy);
return OutputGridLayering(layering_downup_name_, length,
layers_downup_, layer_cells_downup_);
}

bool GridLayering::GridLayeringEvenly() {
Initialize1DArray(n_valid_cells_, layers_evenly_, out_nodata_);
assert(n_layer_cells_updown_.size() == n_layer_cells_downup_.size());

int lyr_95 = -1;
int lyr_5 = -1;
int count_95 = CVT_INT(ceil(n_valid_cells_ * 0.95f));
int count_5 = CVT_INT(ceil(n_valid_cells_ * 0.05f));
int act_count_95 = 0;
int act_count_5 = 0;
int acc_count = 0;
for (auto it = n_layer_cells_downup_.rbegin();
it != n_layer_cells_downup_.rend(); ++it) {
acc_count += it->size();
if (lyr_5 < 0 && acc_count >= count_5) {
lyr_5 = it - n_layer_cells_downup_.rbegin() + 1;
act_count_5 = acc_count;
}
if (lyr_95 < 0 && acc_count >= count_95) {
lyr_95 = it - n_layer_cells_downup_.rbegin();
act_count_95 = acc_count;
break;
}
}
int lyr_n_ave = 1 + (act_count_95 - act_count_5 - 2) / (lyr_95 - lyr_5 + 1);
int max_loop = n_layer_count_ * 10; 
int cur_loop = 1;
bool has_changes = true;
int* layer_diff = nullptr;
Initialize1DArray(n_valid_cells_, layer_diff, out_nodata_);
for (auto itcopy = n_layer_cells_updown_.begin();
itcopy != n_layer_cells_updown_.end(); ++itcopy) {
n_layer_cells_evenly_.emplace_back(vector<int>(itcopy->size()));
for (auto itvalue = itcopy->begin();
itvalue != itcopy->end(); ++itvalue) {
n_layer_cells_evenly_[itcopy - n_layer_cells_updown_.begin()][itvalue - itcopy->begin()] = *itvalue;
}
}

map<int, vector<int> > lyrdiff_cells;
for (int il = 0; il < n_layer_count_; il++) {
#ifdef HAS_VARIADIC_TEMPLATES
lyrdiff_cells.emplace(il, vector<int>());
#else
lyrdiff_cells.insert(make_pair(il, vector<int>()));
#endif
}

while (cur_loop <= max_loop && has_changes) {
has_changes = false;
#pragma omp parallel for
for (int i = 0; i < n_valid_cells_; i++) {
layer_diff[i] = CVT_INT(layers_downup_[i]) - CVT_INT(layers_updown_[i]) + 1;
}

for (auto it_ilyr = n_layer_cells_evenly_.begin();
it_ilyr != n_layer_cells_evenly_.end(); ++it_ilyr) {
int ilyr = it_ilyr - n_layer_cells_evenly_.begin();
for (auto it_lyrdiff = lyrdiff_cells.begin();
it_lyrdiff != lyrdiff_cells.end(); ++it_lyrdiff) {
it_lyrdiff->second.clear();
}
int ilyr_cell_count = CVT_INT(n_layer_cells_evenly_[ilyr].size());
if (ilyr_cell_count <= lyr_n_ave) {
continue;
}
int reserve_count = 0;
int max_lyrdiff = -1;
for (auto it = it_ilyr->begin(); it != it_ilyr->end(); ++it) {
int cur_cell = CVT_INT(*it);
int cur_lyrdiff = layer_diff[cur_cell];
if (cur_lyrdiff != 1 && CVT_INT(layers_downup_[cur_cell]) != ilyr) {
lyrdiff_cells[cur_lyrdiff].emplace_back(cur_cell);
if (cur_lyrdiff > max_lyrdiff) max_lyrdiff = cur_lyrdiff;
} else {
reserve_count++;
}
}
if (ilyr_cell_count == reserve_count) {
continue;
}
int max_change = ilyr_cell_count - lyr_n_ave;
if (max_change < 0) { 
max_change = 0; 
continue;
}
int maxdiff_count = CVT_INT(lyrdiff_cells[max_lyrdiff].size());
set<int> first_tobechanged;
if (max_change >= maxdiff_count) {
max_change = maxdiff_count;
for (int i = 0; i < max_change; i++) {
first_tobechanged.insert(i);
}
} else { 
srand((unsigned int)time(NULL)); 
while (first_tobechanged.size() < max_change) {
first_tobechanged.insert(rand() % maxdiff_count);
}
}
queue<int> tobechanged;
vector<int> changed;
vector<int>& max_lyrdiff_cells = lyrdiff_cells[max_lyrdiff];
for (set<int>::iterator it_change = first_tobechanged.begin();
it_change != first_tobechanged.end(); ++it_change) {
int ichange_cell = max_lyrdiff_cells[*it_change];
if (find(changed.begin(), changed.end(), ichange_cell) == changed.end()) {
tobechanged.push(ichange_cell);
changed.emplace_back(ichange_cell);
}
}
while (!tobechanged.empty()) {
int cur_cell = tobechanged.front();
tobechanged.pop();
int cur_cell_lyr = CVT_INT(layers_updown_[cur_cell]);
if (layer_diff[cur_cell] <= 1 || CVT_INT(layers_downup_[cur_cell]) == cur_cell_lyr) {
continue; 
}
cur_cell_lyr -= 1;                             
layers_updown_[cur_cell] = cur_cell_lyr + 2.f; 
vector<int>& cur_lyr_cells = n_layer_cells_evenly_[cur_cell_lyr];
vector<int>::iterator it_erase = find(cur_lyr_cells.begin(),
cur_lyr_cells.end(), cur_cell);
if (it_erase != cur_lyr_cells.end()) {
*it_erase = std::move(cur_lyr_cells.back());
cur_lyr_cells.pop_back();
}
n_layer_cells_evenly_[cur_cell_lyr + 1].emplace_back(cur_cell);
layer_diff[cur_cell] -= 1;
has_changes = true;

for (int idown = 0; idown < flow_out_num_[cur_cell]; idown++) {
int down_cell_idx = 1 + cur_cell + 1 + idown;
if (cur_cell > 0) down_cell_idx += flow_out_acc_[cur_cell - 1];
int down_cell = CVT_INT(flow_out_cells_[down_cell_idx]);
if (std::find(changed.begin(), changed.end(), down_cell) != changed.end() ||
layer_diff[down_cell] == 1) {
continue;
}
tobechanged.push(down_cell);
changed.emplace_back(down_cell);
}
}
}
if (!has_changes) continue;
cur_loop++;
}
Release1DArray(layer_diff);
Initialize1DArray(n_valid_cells_ + n_layer_count_ + 1, layer_cells_evenly_, 0.f);
layer_cells_evenly_[0] = CVT_FLT(n_layer_count_);
int valid_idx = 1;
for (auto it = n_layer_cells_evenly_.begin();
it != n_layer_cells_evenly_.end(); ++it) {
layer_cells_evenly_[valid_idx++] = CVT_FLT((*it).size());
for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
layer_cells_evenly_[valid_idx++] = CVT_FLT(*it2);
layers_evenly_[*it2] = it - n_layer_cells_evenly_.begin() + 1;
}
}
return OutputGridLayering(layering_evenly_name_, n_valid_cells_ + n_layer_count_ + 1,
layers_evenly_, layer_cells_evenly_);
}

#ifdef USE_MONGODB
bool GridLayering::OutputToMongodb(const char* name, const vint number, char* s) {
bson_t p = BSON_INITIALIZER;
BSON_APPEND_INT32(&p, "SUBBASIN", subbasin_id_);
BSON_APPEND_UTF8(&p, "TYPE", name);
BSON_APPEND_UTF8(&p, "ID", name);
BSON_APPEND_UTF8(&p, "DESCRIPTION", name);
BSON_APPEND_DOUBLE(&p, "NUMBER", CVT_DBL(number));
BSON_APPEND_UTF8(&p, HEADER_INC_NODATA, "FALSE");

gfs_->RemoveFile(string(name));
vint n = number * sizeof(float);
gfs_->WriteStreamData(string(name), s, n, &p);
bson_destroy(&p);
if (nullptr == gfs_->GetFile(name)) {
return false;
}
return true;
}
#endif

bool GridLayering::OutputGridLayering(const string& name, const int datalength,
float* const layer_grid, float* const layer_cells) {
string outpath = string(output_dir_) + "/" + name + ".tif";
FloatRaster(mask_, layer_grid, n_valid_cells_).OutputFileByGdal(outpath);

string header = "LayerID\tCellCount\tCellIDs";
bool done = Output2DimensionArrayTxt(name, header, layer_cells);
if (use_mongo_) {
#ifdef USE_MONGODB
done = done && OutputArrayAsGfs(name, datalength, layer_cells);
}
#endif
return done;
}
