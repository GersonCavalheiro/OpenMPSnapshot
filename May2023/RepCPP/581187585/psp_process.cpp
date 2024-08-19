#include <fcntl.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <unordered_map>

#include "logging.h"
#include "plot3d.h"
#include "upsp.h"
#include "utils/file_writers.h"
#include "utils/file_readers.h"
#include "PSPVideo.h"
#include "CineReader.h"
#include "MrawReader.h"


#include "utils/pspKdtree.h"
#include "utils/pspRT.h"
#include "utils/cv_extras.h"

union UserData {
void* ptr;
uint64_t val;
};


template <typename M>
std::shared_ptr<rt::BVH> createBVH(const M& model,
std::vector<int>& triNodes) {
std::vector<float> tris;
model.extract_tris(tris, triNodes);
std::vector<std::shared_ptr<rt::Primitive>> triPrims =
rt::CreateTriangleMesh(tris, 3);
return std::make_shared<rt::BVH>(triPrims, 4);
}  

template <typename M>
void getTargets(const M& model, const upsp::CameraCal cal,
const std::vector<upsp::Target>& orig_targs,
std::shared_ptr<rt::BVH> scene, float obliqueThresh,
std::vector<upsp::Target>& targs) {
const kdtree* kdroot = model.kdRoot();
const std::vector<cv::Point3_<float>>& nrm =
model.get_n();  
cv::Point3_<float> cam_center = cal.get_cam_center();
Imath::V3f orig(cam_center.x, cam_center.y, cam_center.z);
int nvisible = 0;
cv::Size sz = cal.size();

for (auto targ : orig_targs) {
cv::Point_<float> img_pt = cal.map_point_to_image(targ.xyz);
if (img_pt.x < 0 or img_pt.y < 0 or img_pt.x >= sz.width or
img_pt.y >= sz.height)
continue;

Imath::V3f pos(targ.xyz.x, targ.xyz.y, targ.xyz.z);
Imath::V3f dir = (pos - orig);
float distFromEye = dir.length();

dir.normalize();
rt::Ray ray(orig, dir);
rt::Hit hitrec;
bool hit = scene->intersect(ray, &hitrec);
if (not hit) continue;

bool occluded = hitrec.t < (distFromEye - 1e-3);  
if (not occluded) {
double hitpos[3] = {hitrec.pos[0], hitrec.pos[1], hitrec.pos[2]};

UserData udata;
struct kdres* res = kd_nearest(const_cast<kdtree*>(kdroot), hitpos);
udata.ptr = kd_res_item_data(res);
int nearestNidx = static_cast<int>(udata.val);
kd_res_free(res);

cv::Point3_<float> tunnelNormal = nrm[nearestNidx];
Imath::V3f nrm(tunnelNormal.x, tunnelNormal.y, tunnelNormal.z);
float cos_theta = nrm.dot(dir);
float ang = acos(cos_theta);
bool forward = ang > obliqueThresh;

if (forward) {
++nvisible;
targs.push_back(targ);
}

}  

}  

}  

template <typename M>
void get_target_diameters(const M& model, const upsp::CameraCal cal,
const std::vector<upsp::Target>& targs,
std::shared_ptr<rt::BVH> scene,
std::vector<float>& diams) {
const kdtree* kdroot = model.kdRoot();
diams.resize(targs.size());

for (unsigned int i = 0; i < targs.size(); ++i) {
if ((targs[i].diameter == 0.0) ||
(not upsp::contains(cal.size(), targs[i].uv))) {
diams[i] = 0;
continue;
}

double pos[3] = {targs[i].xyz.x, targs[i].xyz.y, targs[i].xyz.z};
UserData udata;
struct kdres* res = kd_nearest(const_cast<kdtree*>(kdroot), pos);
udata.ptr = kd_res_item_data(res);
int nearestNidx = static_cast<int>(udata.val);
kd_res_free(res);

auto node = model.node(nearestNidx);
cv::Point3_<float> normal = node.get_normal();

assert(cv::norm(normal) != 0.0);

cv::Point3_<float> a, b;
a = upsp::get_perpendicular(normal);
b = a.cross(normal);

float theta = 0.0;
float out_diameter = 0.0;
for (unsigned int j = 0; j < 4; ++j) {
cv::Point3_<float> est_pt =
targs[i].xyz + 0.5 * targs[i].diameter * std::cos(theta) * a +
0.5 * targs[i].diameter * std::sin(theta) * b;
cv::Point_<float> proj_est_pt = cal.map_point_to_image(est_pt);
out_diameter += 2.0 * cv::norm(proj_est_pt - targs[i].uv);
theta += 2 * PI / 4;
}
diams[i] = out_diameter / 4.0;
}
}

template <typename M>
void create_projection_mat(const M& model, const upsp::CameraCal cal,
std::shared_ptr<rt::BVH> scene,
const std::vector<int>& triNodes,
const float obliqueThresh,
Eigen::SparseMatrix<float, Eigen::RowMajor>& smat,
std::vector<float>& uv,
cv::Mat_<uint8_t>& number_projected_model_nodes) {
cv::Size f_sz = cal.size();
smat = Eigen::SparseMatrix<float, Eigen::RowMajor>(model.size(),
f_sz.width * f_sz.height);
std::vector<Eigen::Triplet<float>> triplets;

uv.resize(model.size() * 2, 0.);

auto idx = [f_sz](cv::Point2i pix) { return pix.y * f_sz.width + pix.x; };

const std::vector<cv::Point3_<float>>& nrm = model.get_n();

cv::Point3_<float> cam_center = cal.get_cam_center();
Imath::V3f orig(cam_center.x, cam_center.y, cam_center.z);


unsigned int blockSize = 500;  

unsigned int count = 0;
typedef typename M::NodeIterator NodeIterator_t;
std::vector<NodeIterator_t> blockBreaks;

for (auto it = model.cnode_begin(); it != model.cnode_end(); ++it) {
if ((count % blockSize) == 0) blockBreaks.push_back(it);
++count;
}
blockBreaks.push_back(model.cnode_end());
unsigned int nBlocks = blockBreaks.size() - 1;

volatile unsigned int nCompleted = 0;
volatile unsigned int curBlock = 0;
std::vector<Eigen::Triplet<float>> localTriplets[nBlocks];

#pragma omp parallel for schedule(dynamic, 1)
for (unsigned int blockIndex = 0; blockIndex < nBlocks; ++blockIndex) {
auto blockBegin = blockBreaks[blockIndex];
auto blockEnd = blockBreaks[blockIndex + 1];
unsigned int localTotalCount = 0;


bool lastWasVisible = false;
for (auto it = blockBegin; it != blockEnd; ++it) {
typename M::Node n = *it;
++localTotalCount;

if (!n.is_datanode()) {
continue;
}

cv::Point3_<float> ipos = n.get_position();

cv::Point_<float> pt = cal.map_point_to_image(ipos);

if (not upsp::contains(f_sz, pt)) continue;

Imath::V3f pos(ipos.x, ipos.y, ipos.z);
Imath::V3f dir = (pos - orig).normalize();
rt::Ray ray(orig, dir);

rt::Hit hitrec;
bool hit = scene->intersect(ray, &hitrec);
if (not hit) continue;

int nidx = n.get_nidx();
int primID = hitrec.primID;
bool visible = triNodes[primID * 3 + 0] == nidx or
triNodes[primID * 3 + 1] == nidx or
triNodes[primID * 3 + 2] == nidx;

if (not visible) {
const int NTESTS = 6;
const float L = 1e-4;
Imath::V3f spos[NTESTS] = {
Imath::V3f(-L, 0, 0), Imath::V3f(L, 0, 0),  Imath::V3f(0, -L, 0),
Imath::V3f(0, L, 0),  Imath::V3f(0, 0, -L), Imath::V3f(0, 0, L),
};

for (int tidx = 0; not visible and tidx < NTESTS; ++tidx) {
Imath::V3f pos2(ipos.x + spos[tidx][0], ipos.y + spos[tidx][1],
ipos.z + spos[tidx][2]);
Imath::V3f dir2 = (pos2 - orig);
rt::Ray ray2(orig, dir2);
dir2.normalize();

rt::Hit hitrec2;
bool hit2 = scene->intersect(ray2, &hitrec2);

if (not hit2) continue;

int primID2 = hitrec2.primID;

visible = triNodes[primID2 * 3 + 0] == nidx or
triNodes[primID2 * 3 + 1] == nidx or
triNodes[primID2 * 3 + 2] == nidx;
}
}  

if (not visible) continue;

lastWasVisible = visible;

cv::Point3_<float> tunnelNormal = nrm[nidx];
Imath::V3f nnrm(tunnelNormal.x, tunnelNormal.y, tunnelNormal.z);

float cos_theta = nnrm.dot(dir);
float theta = acos(cos_theta);
bool forward = theta > obliqueThresh;

if (not forward) continue;

const float u = pt.x / f_sz.width;
const float v = pt.y / f_sz.height;
uv[nidx * 2] = u;
uv[nidx * 2 + 1] = v;

cv::Point2i rpt(round(pt.x), round(pt.y));

localTriplets[blockIndex].push_back({nidx, idx(rpt), 1.0});

}  

}  

for (unsigned int idx = 0; idx < nBlocks; ++idx) {
triplets.insert(triplets.end(), localTriplets[idx].begin(),
localTriplets[idx].end());
}

for (const auto& t : triplets) {
const auto idx = t.col();
const auto y = idx / f_sz.width;
const auto x = idx - (y * f_sz.width);
const uint32_t count =
static_cast<int>(number_projected_model_nodes.at<uint8_t>(y, x));
const uint32_t new_count = count < UINT8_MAX ? count + 1 : count;
number_projected_model_nodes.at<uint8_t>(y, x) =
static_cast<uint8_t>(new_count);
}


smat.setFromTriplets(triplets.begin(), triplets.end());

}  

std::string FilenameWithCameraPrefix(int cam, const std::string& name) {
std::ostringstream oss;
oss.str("");
oss << "cam" << std::setfill('0') << std::setw(2) << cam << "-" << name;
return std::move(oss.str());
}

void write_calibration_check_file(
const cv::Mat& frame, const std::vector<upsp::Target>& input_targ_list,
const std::string& out_dir, const std::string& out_name, int cam,
int count) {
std::vector<upsp::Target> targ_list(input_targ_list);
cv::Mat img_to_show;
img_to_show =
upsp::add_targets(targ_list, frame, cv::Scalar(0, 255, 0), false);

std::ostringstream oss;
oss << out_name << std::setfill('0') << std::setw(2) << cam
<< "  #m=" << count;
cv::putText(img_to_show, oss.str(),
cv::Point(img_to_show.cols / 8, 7 * img_to_show.rows / 8),
cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 255, 255), 1);

cv::imwrite(out_dir + "/" + FilenameWithCameraPrefix(cam, "final_cal.png"),
img_to_show);
}

bool InitializeVideoStreams(const upsp::FileInputs& input_deck,
std::vector<upsp::PSPVideo*>& camera_streams,
unsigned int& number_frames_to_process,
bool printf) {
enum NumberFramesToProcessEvaluationMethod {
NUMBER_FRAMES_MAX_AVAILABLE_FOR_ALL_CAMERAS,
NUMBER_FRAMES_USER_PROVIDED
} method;

method = input_deck.number_frames < 0
? NUMBER_FRAMES_MAX_AVAILABLE_FOR_ALL_CAMERAS
: NUMBER_FRAMES_USER_PROVIDED;

switch (method) {
case NUMBER_FRAMES_MAX_AVAILABLE_FOR_ALL_CAMERAS:
number_frames_to_process = std::numeric_limits<unsigned int>::max();
break;
case NUMBER_FRAMES_USER_PROVIDED:
number_frames_to_process = input_deck.number_frames;
break;
}

for (auto stream : camera_streams) {
delete stream;
}
camera_streams.resize(input_deck.cameras);

for (int ii = 0; ii < input_deck.cameras; ii++) {
const std::string filename(input_deck.camera_filenames[ii]);
const auto ext = filename.substr(filename.rfind('.'));
std::unique_ptr<upsp::VideoReader> reader;
if (ext.compare(".cine") == 0) {
reader = std::unique_ptr<upsp::VideoReader>(new upsp::CineReader(filename));
} else if (ext.compare(".mraw") == 0) {
reader = std::unique_ptr<upsp::VideoReader>(new upsp::MrawReader(filename));
} else {
if (printf) {
std::cerr << "Unknown video file extension '" << ext << "'"
<< " for '" << filename << "'."
<< " Valid extensions: {'.cine', '.mraw'}" << std::endl;
}
return false;
}
camera_streams[ii] = new upsp::PSPVideo(std::move(reader));

switch (method) {
case NUMBER_FRAMES_MAX_AVAILABLE_FOR_ALL_CAMERAS:
number_frames_to_process =
std::min(number_frames_to_process, camera_streams[ii]->get_number_frames());
break;
case NUMBER_FRAMES_USER_PROVIDED:
const auto number_frames_available =
camera_streams[ii]->get_number_frames();
if (number_frames_available < number_frames_to_process) {
if (printf) {
std::cerr << "(" << number_frames_to_process << ") frames"
<< " requested but only (" << number_frames_available
<< ") frames available in '" << filename << "'"
<< std::endl;
}
return false;
}
break;
}
if (printf) {
const auto& s = camera_streams[ii];
std::cout << "Initialized video stream ['" << filename << "']" << std::endl;
std::cout << "  Frames per second : " << s->get_frame_rate() << std::endl;
std::cout << "  Frame size        : " << s->get_frame_size() << std::endl;
std::cout << "  Bit depth         : " << s->get_bit_depth() << std::endl;
const auto typ = s->get_frame(1).type();
std::cout << "  cv::Mat type      : " << upsp::convert_type_string(typ) << std::endl;
}
}
if (printf) {
std::cout << "Will process (" << number_frames_to_process << ") frames"
<< std::endl;
}
return true;
}


unsigned long int msize;          
unsigned long int number_frames;  
const float qNAN = std::numeric_limits<float>::quiet_NaN();

#ifndef HAS__MY_MPI_RANK
#define HAS__MY_MPI_RANK
int my_mpi_rank = 0;  
#endif
int num_mpi_ranks = 1;  
std::vector<int> rank_start_frame;
std::vector<int> rank_num_frames;
std::vector<int> rank_start_node;
std::vector<int> rank_num_nodes;

std::vector<cv::Mat> first_frames_raw;
std::vector<cv::Mat> first_frames_32f;
volatile bool first_frames_ready = false;

std::vector<std::vector<cv::Mat>> input_frames;
volatile long int input_frame_offset_ready = -1;

std::vector<float> coverage;
std::vector<float> sol_avg_final;
std::vector<float> sol_rms_final;
std::vector<double> rms;
std::vector<double> avg;
std::vector<double> gain;
pthread_t async_thread_id;

struct FileInfo {
int fd;
std::string path;
};
std::map<std::string, FileInfo> output_files;

const char* const files_to_create[]{
"intensity",
"intensity_transpose",
"pressure_transpose",
"intensity_avg", 
"intensity_rms",
"intensity_ratio_0",
"avg",
"rms",
"coverage",
"steady_state",
"model_temp",
"X",
"Y",
"Z",
"gain",
};
const int num_filenames = sizeof(files_to_create) / sizeof(char*);

int num_openmp_threads = -1;

long int large_buf_size;
volatile void* ptr_intensity_data = NULL;
volatile void* ptr_intensity_transpose_data = NULL;
volatile void* ptr_pressure_transpose_data = NULL;
volatile void* ptr_temp_workspace = NULL;

volatile bool intensity_transpose_ready = false;
volatile bool intensity_written = false;
volatile bool intensity_transpose_written = false;
volatile bool pressure_written = false;

volatile bool* intensity_row_ready;
volatile bool* pressure_row_ready;


#include <chrono>

void timedBarrierPoint(bool do_output, const char* label) {
static double base = -1.0;
static double previous = -1.0;

double before_barrier = MPI_Wtime();

if (base < 0.0) {
base = before_barrier;
previous = before_barrier;
}

MPI_Barrier(MPI_COMM_WORLD);
double after_barrier = MPI_Wtime();

if (do_output) {
std::cout << "+++ " << label << "   [total elapsed:" << after_barrier - base
<< ",  this thread since previous:" << before_barrier - previous
<< "  barrier (load imbalance): "
<< after_barrier - before_barrier << "]" << std::endl;
}
previous = after_barrier;
}

void apportion(int value, int nBins, int* start, int* extent) {
assert((value >= 0) && (nBins > 0));
unsigned long int blockSize = value / nBins;
unsigned long int remainder = value - (blockSize * nBins);

unsigned long int nextStart = 0;
unsigned long int curBin;
for (curBin = 0; curBin < nBins; ++curBin) {
start[curBin] = nextStart;
extent[curBin] = blockSize + (curBin < remainder);
nextStart += extent[curBin];
}
assert(nextStart == value);
}

void pwrite_full(int fd, const void* buf, size_t nbytes, off_t file_offset) {
unsigned char* src = (unsigned char*)buf;
errno = 0;
size_t amt_remaining = nbytes;
while (amt_remaining > 0) {
long int amt_written = pwrite(fd, src, amt_remaining, file_offset);
if (amt_written < 0) perror("pwrite_full");
assert(amt_written > 0);
amt_remaining -= amt_written;
file_offset += amt_written;
src += amt_written;
}
}

void local_transpose(float* __restrict__ src_ptr, int x_extent, int y_extent,
float* __restrict__ dst_ptr) {

int block_size = 100;

float(*__restrict__ src)[x_extent] = (float(*)[x_extent])src_ptr;
float(*__restrict__ dst)[y_extent] = (float(*)[y_extent])dst_ptr;

int full_x_blocks = x_extent / block_size;
int full_y_blocks = y_extent / block_size;

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
for (int y_block = 0; y_block < full_y_blocks + 1; ++y_block) {
for (int x_block = 0; x_block < full_x_blocks + 1; ++x_block) {
int y_block_start_index = y_block * block_size;
int y_block_extent =
((y_block < full_y_blocks) ? block_size
: y_extent - (full_y_blocks * block_size));

int x_block_start_index = x_block * block_size;
int x_block_extent =
((x_block < full_x_blocks) ? block_size
: x_extent - (full_x_blocks * block_size));

for (int jj = 0; jj < y_block_extent; ++jj) {
for (int ii = 0; ii < x_block_extent; ++ii) {
dst[x_block_start_index + ii][y_block_start_index + jj] =
src[y_block_start_index + jj][x_block_start_index + ii];
}
}
}
}
}


void global_transpose(float* ptr_src, float* ptr_dst) {
int my_num_frames = rank_num_frames[my_mpi_rank];
int my_num_nodes = rank_num_nodes[my_mpi_rank];
MPI_Request requests[num_mpi_ranks];

float(*src)[msize] = (float(*)[msize])ptr_src;
float(*dst)[number_frames] = (float(*)[number_frames])ptr_dst;
float(*temp)[my_num_frames] = (float(*)[my_num_frames])ptr_temp_workspace;

if (0 == my_mpi_rank) std::cout << "  Local transpose ..." << std::endl;
local_transpose(&(src[0][0]), msize, my_num_frames, &(temp[0][0]));

if (0 == my_mpi_rank)
std::cout << "  Global transpose exchange ..." << std::endl;
for (int rank = 0; rank < num_mpi_ranks; ++rank) {
long int start_row = rank_start_node[rank];
long int num_rows = rank_num_nodes[rank];
assert((num_rows * my_num_frames) < INT_MAX);
MPI_Isend(&(temp[start_row][0]), num_rows * my_num_frames, MPI_FLOAT, rank,
0, MPI_COMM_WORLD, &(requests[rank]));
}

long int max_count = rank_num_frames[0] * rank_num_nodes[0];
float* recv_buf = (float*)malloc(max_count * sizeof(float));
assert(recv_buf != NULL);
for (int count = 0; count < num_mpi_ranks; ++count) {
MPI_Status status;
MPI_Recv(recv_buf, max_count, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
&status);

int sender_rank = status.MPI_SOURCE;
int sender_num_frames = rank_num_frames[sender_rank];
int expected_num_items = sender_num_frames * my_num_nodes;

int actual_num_items;
MPI_Get_count(&status, MPI_FLOAT, &actual_num_items);
assert(actual_num_items == expected_num_items);

float(*recv_trans)[sender_num_frames] =
(float(*)[sender_num_frames])recv_buf;
long int start_col = rank_start_frame[sender_rank];
for (long int node_offset = 0; node_offset < my_num_nodes; ++node_offset) {
for (long int frame_offset = 0; frame_offset < sender_num_frames;
++frame_offset) {
dst[node_offset][start_col + frame_offset] =
recv_trans[node_offset][frame_offset];
}
}
}
free(recv_buf);

MPI_Waitall(num_mpi_ranks, requests, MPI_STATUSES_IGNORE);
}



#include <sched.h>
void wait_for(void* ptr_flag, int usec_interval) {
volatile bool* p = (volatile bool*)ptr_flag;
__sync_synchronize();
while (not(*p)) {
if (usec_interval <= 0)
sched_yield();
else
usleep(usec_interval);
__sync_synchronize();
}
}

void allocate_global_data(void) {
int max_num_frames = rank_num_frames[0];
int max_num_nodes = rank_num_frames[0];

long int buf_size1 = rank_num_frames[0] * msize * sizeof(float);
long int buf_size2 = rank_num_nodes[0] * number_frames * sizeof(float);
large_buf_size = std::max(buf_size1, buf_size2);

ptr_intensity_data = malloc(large_buf_size);
assert(ptr_intensity_data != NULL);
ptr_intensity_transpose_data = malloc(large_buf_size);
assert(ptr_intensity_transpose_data != NULL);
ptr_pressure_transpose_data = malloc(large_buf_size);
assert(ptr_pressure_transpose_data != NULL);
ptr_temp_workspace = malloc(large_buf_size);
assert(ptr_temp_workspace != NULL);

intensity_row_ready =
(volatile bool*)malloc(max_num_frames * sizeof(volatile bool));
for (int ii = 0; ii < max_num_frames; ++ii) intensity_row_ready[ii] = false;

pressure_row_ready =
(volatile bool*)malloc(max_num_frames * sizeof(volatile bool));
for (int ii = 0; ii < max_num_frames; ++ii) pressure_row_ready[ii] = false;
}

void open_output_files(const std::string& dir) {
for (int ii = 0; ii < num_filenames; ++ii) {
std::string filename = dir + "/" + files_to_create[ii];
output_files[files_to_create[ii]].path = filename;
}
if (0 == my_mpi_rank) {
for (auto& info : output_files) {
const auto& filename = info.second.path;
std::string cmd = "/bin/rm -f " + filename;
cmd += " ; lfs setstripe -c 60 " + filename + " >& /dev/null";
system(cmd.c_str());
int fd = open(filename.c_str(), O_WRONLY | O_CREAT, 0644);
assert(fd >= 0);
close(fd);
}
}
MPI_Barrier(MPI_COMM_WORLD);
for (auto& info : output_files) {
info.second.fd = open(info.second.path.c_str(), O_WRONLY, 0644);
assert(info.second.fd >= 0);
if (0 == my_mpi_rank) {
std::cout << "Opened '" << info.second.path << "'" << std::endl;
}
}
}

void close_output_files() {
for (auto& info : output_files) {
close(info.second.fd);
if (0 == my_mpi_rank) {
std::cout << "Closed '" << info.second.path << "'" << std::endl;
}
}
}

void* __async_read_ahead(void* arg) {
std::vector<upsp::PSPVideo*>* ptr_cams = (std::vector<upsp::PSPVideo*>*)arg;

unsigned int n_cams = ptr_cams->size();

unsigned long int first_frame_to_read = rank_start_frame[my_mpi_rank];
unsigned long int num_frames_to_read = rank_num_frames[my_mpi_rank];

first_frames_raw.resize(n_cams);
first_frames_32f.resize(n_cams);
for (unsigned int c = 0; c < n_cams; ++c) {
first_frames_raw[c] = ((*ptr_cams)[c])->get_frame(1);
upsp::fix_hot_pixels(first_frames_raw[c], my_mpi_rank == 0);
first_frames_raw[c].convertTo(first_frames_32f[c], CV_32F);
}
__sync_synchronize();
first_frames_ready = true;

input_frames.resize(n_cams);
for (unsigned int c = 0; c < n_cams; ++c) {
input_frames[c].resize(num_frames_to_read);
}
for (unsigned int f = 0; f < num_frames_to_read; ++f) {
for (unsigned int c = 0; c < n_cams; ++c) {
input_frames[c][f] =
((*ptr_cams)[c])->get_frame(first_frame_to_read + f + 1);
}
__sync_synchronize();
input_frame_offset_ready = f;
}

return NULL;
}

void write_behind(void* data, volatile bool* ready_flags, int fd) {
int start_write_row = 0;
int num_write_rows = 0;
int num_remaining = rank_num_frames[my_mpi_rank];
long int file_offset = rank_start_frame[my_mpi_rank] * msize * sizeof(float);
unsigned char* start_addr = (unsigned char*)data;

while (num_remaining > 0) {
__sync_synchronize();
wait_for((void*)&(ready_flags[start_write_row]), 1000000);

do {
if (++num_write_rows >= num_remaining) break;
__sync_synchronize();
} while (ready_flags[start_write_row + num_write_rows]);
assert(num_write_rows <= num_remaining);

long int nbytes = num_write_rows * msize * sizeof(float);
pwrite_full(fd, start_addr, nbytes, file_offset);

file_offset += nbytes;
start_addr += nbytes;
start_write_row += num_write_rows;
num_remaining -= num_write_rows;
num_write_rows = 0;
}
}

void* __async_write_behind(void* arg) {
write_behind((void*)ptr_intensity_data, intensity_row_ready,
output_files["intensity"].fd);
if (0 == my_mpi_rank)
std::cerr << "## Async I/O :  'intensity' written" << std::endl;

return NULL;
}

void write_block(void* data, int fd) {
long int nbytes = number_frames * rank_num_nodes[my_mpi_rank] * sizeof(float);
long int file_offset =
number_frames * rank_start_node[my_mpi_rank] * sizeof(float);
pwrite_full(fd, data, nbytes, file_offset);
}

typedef struct {
std::vector<upsp::PSPVideo*>* ptr_cams;
std::string add_out_dir;
} _asynch_info_t;
_asynch_info_t _asynch_info;

void* __asynch_thread(void* arg) {
int status;
pthread_t tid;
std::vector<pthread_t> thread_ids;

_asynch_info_t* p = (_asynch_info_t*)arg;

status = pthread_create(&tid, NULL, __async_read_ahead, p->ptr_cams);
assert(0 == status);
thread_ids.push_back(tid);


wait_for((void*)&intensity_transpose_ready, 100000);
write_block((void*)ptr_intensity_transpose_data,
output_files["intensity_transpose"].fd);
if (0 == my_mpi_rank)
std::cerr << "## Async I/O :  'intensity_transpose' written" << std::endl;

for (int thrd = 0; thrd < thread_ids.size(); ++thrd) {
int status = pthread_join(thread_ids[thrd], NULL);
assert(0 == status);
}

return NULL;
}


struct Phase1Settings {
Phase1Settings(upsp::FileInputs i_ifile)
: code_version(""), add_out_dir(""), ifile(i_ifile) {
int cal_options = cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 |
cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6 |
cv::CALIB_FIX_FOCAL_LENGTH;
}

bool checkout;

upsp::FileInputs ifile;

unsigned int bound_thickness;
unsigned int buffer_thickness;
float target_diam_sf;

int cal_options;

unsigned int trans_nodes;

std::string code_version;

std::string add_out_dir;

bool has_x_max;
float x_max;
};


template <typename M, typename Patcher>
struct Phase1Elements {
typedef M Model;

Phase1Elements() : model(nullptr) {}

~Phase1Elements() {
delete model;
for (unsigned int i = 0; i < cams.size(); ++i) {
delete cams[i];
}
for (unsigned int i = 0; i < patchers.size(); ++i) {
delete patchers[i];
}
}

Model* model;

std::shared_ptr<rt::BVH> scene;
std::vector<int> triNodes;

std::vector<upsp::PSPVideo*> cams;

std::vector<upsp::CameraCal> cals;

upsp::CameraSettings camera_settings;

std::vector<Patcher*> patchers;

std::vector<cv::Mat> first_frames;

std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> projs;
};

struct Phase2Settings {
Phase2Settings(upsp::FileInputs i_ifile)
: code_version(""),
r(0.896),
gamma(1.4),
F_to_R(459.67),
degree(6),
grid_tol(1e-3),
grid_units("-"),
k(10),
p(2.0),
ifile(i_ifile) {}

unsigned int trans_nodes;

upsp::FileInputs ifile;

bool wind_off;

bool read_model_temp;

std::string code_version;

const float r;       
const float gamma;   
const float F_to_R;  

unsigned int degree;

float grid_tol;

std::string grid_units;

unsigned int k;  
unsigned int p;  
};

struct Phase2Files {
std::string add_out_dir;
std::string h5_out;
std::string h5_out_extra;
std::string steady_p3d;
std::string steady_grid;
std::string paint_cal;
std::string model_temp_p3d;
};


template <typename M>
struct Phase2Elements {
typedef M Model;
Phase2Elements() : model(nullptr) {}
~Phase2Elements() {}
Model* model;
upsp::CameraSettings camera_settings;
upsp::TunnelConditions tunnel_conditions;
};

struct Settings {
Settings(upsp::FileInputs i_ifile):
phase1(i_ifile), phase2(i_ifile) {}
Phase1Settings phase1;
Phase2Settings phase2;
Phase2Files phase2files;
};

void imwrite(
const cv::Mat& img, const Phase1Settings& sett,
int cam, const std::string& name
) {
const auto fn = sett.add_out_dir + "/" + FilenameWithCameraPrefix(cam + 1, name);
cv::imwrite(fn, img);
};


template <typename P1Elems>
int phase0(Phase1Settings& sett, P1Elems& elems);

template <typename P1Elems>
int phase1(Phase1Settings& sett, P1Elems& elems);

template <typename P2Elems>
int phase2(Phase2Settings& sett, Phase2Files& p2_files, P2Elems& elems);

std::shared_ptr<Settings> ParseOpts(int argc, char **argv) {
const cv::String keys =
"{help h usage ? |     | print this message   }"
"{input_file     |     | full input deck }"
"{frames         |     | override input_file number of frames }"
"{code_version   |  XYZ | version of the repo (deprecated)}"
"{trans_nodes    |  250   | number of nodes per chunk in transposed solution }"
"{add_out_dir    |     | output directory for any additional debugging files "
"(default=input deck-specified output directory) }"
"{checkout       |  F  | perform calibration update checkout }"
"{bound_pts      |  2  | thickness of cluster boundary}"
"{buffer_pts     |  1  | thickness of buffer between targets and cluster boundary}"
"{target_diam_sf |  1.2  | scale factor to apply onto target diameter}"
"{cutoff_x_max   |     | ignore nodes beyond this value in projection }"
"{h5_out             |     | output hdf5 file }"
"{steady_p3d         |     | steady state p3d function file (for wind-on); units of Cp }"
"{steady_grid        |     | steady state p3d grid (needed for wind-on unstructured) }"
"{paint_cal          |     | unsteady gain paint calibration file }"
"{model_temp_p3d     |     | temperature p3d function file; units of Temperature degrees F }"
;
cv::CommandLineParser parser(argc, argv, keys);
parser.about(
"Process Unsteady Pressure-Sensitive Paint (uPSP) video files "
"into pressure-time history on model surface grid"
);

if (parser.has("help")) {
if (0 == my_mpi_rank) parser.printMessage();
return nullptr;
}

if (!parser.has("input_file")) {
LOG_ERROR("Must specify -input_file");
return nullptr;
}

if (!parser.has("h5_out")) {
LOG_ERROR("Must specify -h5_out");
return nullptr;
}

if (!parser.has("paint_cal")) {
LOG_ERROR("Must specify -paint_cal");
return nullptr;
}

auto input_filename = parser.get<std::string>("input_file");
upsp::FileInputs input_file;
if (!input_file.Load(input_filename)) {
return nullptr;
}
auto settings = std::make_shared<Settings>(input_file);

auto& p1s = settings->phase1;
p1s.bound_thickness = parser.get<unsigned int>("bound_pts");
p1s.buffer_thickness = parser.get<unsigned int>("buffer_pts");
p1s.target_diam_sf = parser.get<float>("target_diam_sf");
p1s.trans_nodes = parser.get<unsigned int>("trans_nodes");
p1s.checkout = parser.get<bool>("checkout");
p1s.code_version = parser.get<std::string>("code_version");
p1s.add_out_dir = parser.has("add_out_dir") ? parser.get<std::string>("add_out_dir"): p1s.ifile.out_dir;
p1s.has_x_max = parser.has("cutoff_x_max");
p1s.x_max = p1s.has_x_max ? parser.get<float>("cutoff_x_max"): 0.;
p1s.ifile.number_frames = parser.has("frames") ? parser.get<int>("frames") : p1s.ifile.number_frames;

auto& p2s = settings->phase2;
auto& p2f = settings->phase2files;
p2f.h5_out = parser.get<std::string>("h5_out");
p2f.paint_cal = parser.get<std::string>("paint_cal");
p2f.steady_p3d = parser.get<std::string>("steady_p3d");
p2f.model_temp_p3d = parser.get<std::string>("model_temp_p3d");
p2f.steady_grid = parser.get<std::string>("steady_grid");
p2f.add_out_dir = parser.has("add_out_dir") ? parser.get<std::string>("add_out_dir"): p2s.ifile.out_dir;
p2f.h5_out_extra = p2f.add_out_dir + "/" + "extras.h5";
p2s.trans_nodes = parser.get<unsigned int>("trans_nodes");
p2s.code_version = parser.get<std::string>("code_version");
p2s.wind_off = p2f.steady_p3d.empty() ? true: false;
p2s.grid_units = p2s.ifile.grid_units;
p2s.read_model_temp = p2f.model_temp_p3d.empty() ? false: true;

if (!parser.check()) {
parser.printErrors();
return nullptr;
}

if (!p1s.ifile.check_all()) {
return nullptr;
}
if (p1s.ifile.tunnel != "ames_unitary") {
LOG_ERROR("Unrecognized tunnel name '%s'", p1s.ifile.tunnel);
return nullptr;
}
if ((p1s.ifile.registration != upsp::RegistrationType::None) &&
(p1s.ifile.registration != upsp::RegistrationType::Pixel)) {
LOG_ERROR("Unsupported registration type");
return nullptr;
}
if (p1s.ifile.filter_size % 2 == 0) {
LOG_ERROR("Filter size must be odd (currently '%d')", p1s.ifile.filter_size);
return nullptr;
}

bool grid_is_structured;
switch(p1s.ifile.grid_type) {
case upsp::GridType::None:
LOG_ERROR("Unsupported grid type");
return nullptr;
case upsp::GridType::P3D:
grid_is_structured = true;
break;
case upsp::GridType::Tri:
grid_is_structured = false;
break;
}

if (!grid_is_structured && p2f.steady_grid.empty()) {
LOG_ERROR("Must specify steady psp grid for wind-on unstructured grid data");
return nullptr;
}

if (0 == my_mpi_rank) std::cout << p1s.ifile << std::endl;
timedBarrierPoint(0 == my_mpi_rank, "Load and validate input file and command line options");

return settings;
}

template <typename Model>
int RunAllPhases(std::shared_ptr<Settings>, Model*);

int main(int argc, char* argv[]) {
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);
MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_ranks);
if (my_mpi_rank == 0) LogSetLevel(LOG_LEVEL_DEBUG);
else LogSetLevel(LOG_LEVEL_SUPPRESS);

if (0 == my_mpi_rank) system("date --rfc-3339=ns");

timedBarrierPoint(0 == my_mpi_rank, "Begin");

Eigen::initParallel();
cv::setNumThreads(4);
omp_set_nested(false);
num_openmp_threads = omp_get_max_threads();

if (0 == my_mpi_rank)
std::cout << num_mpi_ranks << " MPI ranks, each with " << num_openmp_threads
<< " OpenMP threads" << std::endl;

rank_start_frame.resize(num_mpi_ranks);
rank_num_frames.resize(num_mpi_ranks);
rank_start_node.resize(num_mpi_ranks);
rank_num_nodes.resize(num_mpi_ranks);

auto psett = ParseOpts(argc, argv);
if (!psett) {
LOG_ERROR("Failed to validate command line options; aborting");
MPI_Finalize();
return 1;
}

int res;
const auto& grid_type = psett->phase1.ifile.grid_type;
const auto& grid_filename = psett->phase1.ifile.grid;
switch(grid_type) {
case upsp::GridType::P3D:
{
auto model = new upsp::P3DModel_<float>(grid_filename, 1e-3);
timedBarrierPoint(0 == my_mpi_rank, "set up P3DModel");
return RunAllPhases(psett, model);
}
case upsp::GridType::Tri:
{
auto model = new upsp::TriModel_<float>(grid_filename);
timedBarrierPoint(0 == my_mpi_rank, "set up TriModel");
return RunAllPhases(psett, model);
}
}
}

template <typename Model>
int RunAllPhases(std::shared_ptr<Settings> psett, Model* model) {
typedef upsp::PatchClusters<float> Patcher;
Phase1Elements<Model, Patcher> p1elems;
p1elems.model = model;
int res = phase1(psett->phase1, p1elems);
if (res) {
std::cerr << "Error during phase 1 processing, aborting" << std::endl;
MPI_Finalize();
return res;
}
if (psett->phase1.checkout) {
MPI_Finalize();
return 0;
}
MPI_Barrier(MPI_COMM_WORLD);
Phase2Elements<Model> p2elems;
p2elems.model = model;
p2elems.camera_settings = p1elems.camera_settings;
res = phase2(psett->phase2, psett->phase2files, p2elems);
if (res) {
std::cerr << "Error during phase 2 processing, aborting" << std::endl;
MPI_Finalize();
return res;
}

pthread_join(async_thread_id, NULL);

close_output_files();

free((void*)ptr_intensity_transpose_data);
free((void*)ptr_pressure_transpose_data);
free((void*)pressure_row_ready);
free((void*)intensity_row_ready);
free((void*)ptr_temp_workspace);

if (0 == my_mpi_rank) {
std::cout << "Wait for all ranks to complete" << std::endl;
}
timedBarrierPoint(0 == my_mpi_rank, "All done");

MPI_Finalize();
return 0;
}


template <typename P1Elems>
int phase1(Phase1Settings& sett, P1Elems& elems) {
typedef typename P1Elems::Model Model;

upsp::FileInputs& ifile = sett.ifile;
Model& model = *elems.model;
auto& cams = elems.cams;

if (!sett.checkout) {
if (sett.has_x_max) {
std::vector<float> xs = model.get_x();
for (unsigned int i = 0; i < xs.size(); ++i) {
if (xs[i] > sett.x_max) {
model.set_node_nondata(i);
}
}
}
}

if (!ifile.active_comps.empty()) {
std::unordered_map<int, bool> active_comps =
upsp_files::read_active_comp_file(ifile.active_comps);

if (active_comps.size() > model.number_of_components()) {
std::cerr << "Error: Number of components in active component file";
std::cerr << " cannot be greater than the number of components in";
std::cerr << " the grid" << std::endl;
return 1;
}

for (auto n_it = model.cnode_begin(); n_it != model.cnode_end(); ++n_it) {
auto n = *n_it;
if (n.has_primary_component()) {
int comp = n.get_primary_component();

if (active_comps.find(comp) != active_comps.end()) {
if (!active_comps[comp]) {
model.set_node_nondata(n.get_nidx());
}
}
}
}
}

unsigned int number_frames_to_process;
if (!InitializeVideoStreams(ifile, cams, number_frames_to_process, my_mpi_rank == 0)) {
std::cerr << "Failed to initialize video streams, exiting" << std::endl;
return 1;
}
number_frames = number_frames_to_process;
timedBarrierPoint(0 == my_mpi_rank, "Load cameras");


msize = model.size();  

apportion(number_frames, num_mpi_ranks, &(rank_start_frame[0]),
&(rank_num_frames[0]));
int my_first_frame = rank_start_frame[my_mpi_rank];
int my_num_frames = rank_num_frames[my_mpi_rank];

apportion(msize, num_mpi_ranks, &(rank_start_node[0]), &(rank_num_nodes[0]));
int my_first_node = rank_start_node[my_mpi_rank];
int my_num_nodes = rank_num_nodes[my_mpi_rank];

allocate_global_data();
open_output_files(sett.add_out_dir);
timedBarrierPoint(0 == my_mpi_rank, "opened output files");

_asynch_info.ptr_cams = &cams;
_asynch_info.add_out_dir = sett.add_out_dir;
int status =
pthread_create(&async_thread_id, NULL, __asynch_thread, &_asynch_info);
assert(0 == status);

if (!sett.checkout && 0 == my_mpi_rank) {
std::cout << "opening hdf5" << std::endl;

std::cout << "writing X, Y, Z flat files" << std::endl;
const std::vector<float>& xpos = model.get_x();
pwrite_full(output_files["X"].fd, &(xpos[0]), xpos.size() * sizeof(float),
0);
const std::vector<float>& ypos = model.get_y();
pwrite_full(output_files["Y"].fd, &(ypos[0]), ypos.size() * sizeof(float),
0);
const std::vector<float>& zpos = model.get_z();
pwrite_full(output_files["Z"].fd, &(zpos[0]), zpos.size() * sizeof(float),
0);
}

__sync_synchronize();
if (not first_frames_ready) {
std::cout << "Rank " << my_mpi_rank
<< " waiting for first_frames_* data ..." << std::endl;
while (not first_frames_ready) __sync_synchronize();
std::cout << "Rank " << my_mpi_rank << " GOT first_frames_* data ..."
<< std::endl;
}

if (phase0(sett, elems)) {
std::cerr << "ERROR DURING PHASE 0" << std::endl;
return 1;
}

if (sett.checkout) {
return 0;
}

auto& camsets = elems.camera_settings;
for (unsigned int c = 0; c < ifile.cameras; ++c) {
camsets.focal_lengths.push_back(elems.cals[c].get_focal_length());
camsets.cam_nums.push_back(ifile.cam_nums[c]);
}
camsets.framerate = elems.cams[0]->get_frame_rate();
camsets.fstop = elems.cams[0]->get_aperture();
camsets.exposure = elems.cams[0]->get_exposure();

elems.projs.resize(ifile.cameras);

if (0 == my_mpi_rank) {
std::cout << "Creating projection matricies, " << num_openmp_threads
<< " OpenMP threads" << std::endl;
}
for (unsigned int c = 0; c < ifile.cameras; ++c) {
cv::Size sz = first_frames_raw[c].size();
cv::Mat_<uint8_t> pixel_node_counts(sz, 0);
psp::BlockTimer bt(0 == my_mpi_rank, "Calculating projection matrix");
const float obliqueThresh = deg2_rad(180. - ifile.oblique_angle);
std::vector<float> uv;
create_projection_mat(model, elems.cals[c], elems.scene, elems.triNodes,
obliqueThresh, elems.projs[c], uv, pixel_node_counts);

if (0 == my_mpi_rank) {
{ 
const auto filename = sett.add_out_dir + "/" +
FilenameWithCameraPrefix(c + 1, "nodecount.png");
cv::Mat out_img;
upsp::nodes_per_pixel_colormap(pixel_node_counts, out_img);
cv::imwrite(filename, out_img);
}
{ 
const auto filename = sett.add_out_dir + "/" +
FilenameWithCameraPrefix(c + 1, "uv");
FILE *fp = fopen(filename.c_str(), "wb");
fwrite((void*) &uv[0], sizeof(float), uv.size(), fp);
}
}
}

std::vector<cv::Point3d> centers(ifile.cameras);
for (unsigned int c = 0; c < ifile.cameras; ++c) {
centers[c] = elems.cals[c].get_cam_center();
}

if (ifile.overlap == upsp::OverlapType::BestView) {
upsp::BestView<float> bv;
upsp::adjust_projection_for_weights<Model, float>(model, centers,
elems.projs, bv);
} else {
upsp::AverageViews<float> av;
upsp::adjust_projection_for_weights<Model, float>(model, centers,
elems.projs, av);
}

std::vector<unsigned int> skipped;
upsp::identify_skipped_nodes(elems.projs, skipped);


if (0 == my_mpi_rank) std::cout << "Processing first frame" << std::endl;
std::vector<float> sol1;
{
for (unsigned int c = 0; c < ifile.cameras; ++c) {
cv::Mat img = first_frames_raw[c].clone();

if (ifile.registration == upsp::RegistrationType::Pixel) {
cv::Mat warp_matrix;
img = upsp::register_pixel(first_frames_32f[c], img, warp_matrix);
}

if (ifile.target_patcher == upsp::TargetPatchType::Polynomial) {
img = elems.patchers[c]->operator()(img);
}

if (ifile.filter == upsp::FilterType::Gaussian) {
GaussianBlur(img, img, cv::Size(ifile.filter_size, ifile.filter_size),
0);
} else if (ifile.filter == upsp::FilterType::Box) {
cv::blur(img, img, cv::Size(ifile.filter_size, ifile.filter_size));
}

std::vector<float> c_sols;
upsp::project_frame(elems.projs[c], img, c_sols);

if (c == 0) {
sol1.assign(c_sols.begin(), c_sols.end());
} else {
std::transform(sol1.begin(), sol1.end(), c_sols.begin(), sol1.begin(),
std::plus<float>());
}
}

for (unsigned int i = 0; i < skipped.size(); ++i) {
sol1[skipped[i]] = qNAN;
}

model.adjust_solution(sol1);
}

std::vector<double> sol_rms_partial(msize, 0.0);
std::vector<double> sol_avg_partial(msize, 0.0);

MPI_Barrier(MPI_COMM_WORLD);

if (0 == my_mpi_rank) {
std::cout << "\nNum frames: " << number_frames << ", model size: " << msize
<< std::endl;
std::cout << "\nProcessing frames" << std::endl;
std::cout << "  Rank 0:: first frame: " << my_first_frame
<< ",  num frames: " << my_num_frames << std::endl;
}

float(*intensity_buf)[msize] = (float(*)[msize])ptr_intensity_data;

timedBarrierPoint(0 == my_mpi_rank, "phase1: ready to process frames");

#pragma omp parallel
{
std::vector<double> local_sol_rms(msize, 0.0);
std::vector<double> local_sol_avg(msize, 0.0);
cv::Mat img;

#pragma omp for schedule(dynamic, 1) nowait
for (unsigned int offset = 0; offset < my_num_frames; ++offset) {
unsigned int f = my_first_frame + offset;

__sync_synchronize();
if (input_frame_offset_ready < offset) {
std::cout << " waiting for input frame " << f << std::endl;
while (input_frame_offset_ready < offset) __sync_synchronize();
std::cout << " frame " << f << " ready" << std::endl;
}

if ((0 == my_mpi_rank) && ((offset % 100) == 0)) {
std::cout << "  Rank 0:: processing frame " << f << std::endl;
}

std::vector<float> sol;
for (unsigned int c = 0; c < ifile.cameras; ++c) {
upsp::fix_hot_pixels((input_frames[c])[offset]);
(input_frames[c])[offset].copyTo(img);
assert(!img.empty());

if ((f > 0) && (ifile.registration == upsp::RegistrationType::Pixel)) {
cv::Mat warp_matrix;
img = upsp::register_pixel(elems.first_frames[c], img, warp_matrix);
}

if (ifile.target_patcher == upsp::TargetPatchType::Polynomial) {
img = elems.patchers[c]->operator()(img);
}

if (ifile.filter == upsp::FilterType::Gaussian) {
GaussianBlur(img, img, cv::Size(ifile.filter_size, ifile.filter_size),
0);
} else if (ifile.filter == upsp::FilterType::Box) {
cv::blur(img, img, cv::Size(ifile.filter_size, ifile.filter_size));
}

std::vector<float> c_sols;
upsp::project_frame(elems.projs[c], img, c_sols);

if (c == 0) {
sol.assign(c_sols.begin(), c_sols.end());
} else {
std::transform(sol.begin(), sol.end(), c_sols.begin(), sol.begin(),
std::plus<float>());
}
}

for (unsigned int i = 0; i < skipped.size(); ++i) {
sol[skipped[i]] = std::numeric_limits<float>::quiet_NaN();
}

for (unsigned int i = 0; i < model.size(); ++i) {
local_sol_rms[i] += (sol[i] * sol[i]);
local_sol_avg[i] += sol[i];
}

if (ifile.grid_type == upsp::GridType::P3D) {
model.adjust_solution(sol);
}

for (unsigned int ii = 0; ii < msize; ++ii) {
intensity_buf[offset][ii] = sol[ii];
}

__sync_synchronize();
intensity_row_ready[offset] = true;
}

#pragma omp critical
for (unsigned int i = 0; i < msize; ++i) {
sol_rms_partial[i] += local_sol_rms[i];
sol_avg_partial[i] += local_sol_avg[i];
}
}

if (0 == my_mpi_rank) {
std::cout << "Wait for all ranks to complete frame processing" << std::endl;
}

timedBarrierPoint(0 == my_mpi_rank, "phase1: process frames complete");

MPI_Barrier(MPI_COMM_WORLD);
if (0 == my_mpi_rank) {
std::cout << "Frame processing complete" << std::endl;
}

if (0 == my_mpi_rank)
std::cout << "Global reduction of rms and avg .." << std::endl;
void* sendbuf =
(void*)((0 == my_mpi_rank) ? MPI_IN_PLACE : &(sol_avg_partial[0]));
MPI_Reduce(sendbuf, &(sol_avg_partial[0]), sol_avg_partial.size(), MPI_DOUBLE,
MPI_SUM, 0, MPI_COMM_WORLD);
sendbuf = (void*)((0 == my_mpi_rank) ? MPI_IN_PLACE : &(sol_rms_partial[0]));
MPI_Reduce(sendbuf, &(sol_rms_partial[0]), sol_rms_partial.size(), MPI_DOUBLE,
MPI_SUM, 0, MPI_COMM_WORLD);
if (0 == my_mpi_rank) std::cout << "Global reduction complete" << std::endl;







coverage.resize(msize, 0.);
sol_avg_final.resize(msize, 0.);
sol_rms_final.resize(msize, 0.);

if (0 == my_mpi_rank) {
for (unsigned int i = 0; i < model.size(); ++i) {
sol_avg_final[i] = sol_avg_partial[i] / number_frames;
sol_rms_final[i] = sqrt(sol_rms_partial[i] / number_frames);
}
if (ifile.grid_type == upsp::GridType::P3D) {
model.adjust_solution(sol_rms_final);
model.adjust_solution(sol_avg_final);
}
pwrite_full(output_files["intensity_rms"].fd, &(sol_rms_final[0]),
sol_rms_final.size() * sizeof(float), 0);
pwrite_full(output_files["intensity_avg"].fd, &(sol_avg_final[0]),
sol_avg_final.size() * sizeof(float), 0);

for (unsigned int i = 0; i < sol1.size(); ++i) {
sol1[i] = sol_avg_final[i] / sol1[i] - 1.0;
}
pwrite_full(output_files["intensity_ratio_0"].fd, &(sol1[0]),
sol1.size() * sizeof(float), 0);

std::vector<float> ind_sol;
for (unsigned int c = 0; c < ifile.cameras; ++c) {
cv::Mat one_frame = cv::Mat::ones(elems.cals[c].size().width,
elems.cals[c].size().height, CV_32F);
upsp::project_frame(elems.projs[c], one_frame, ind_sol);
if (c == 0) {
coverage.assign(ind_sol.begin(), ind_sol.end());
} else {
std::transform(coverage.begin(), coverage.end(), ind_sol.begin(),
coverage.begin(), std::plus<float>());
}
}

if (ifile.grid_type == upsp::GridType::P3D) {
model.adjust_solution(coverage);
}
pwrite_full(output_files["coverage"].fd, &(coverage[0]),
coverage.size() * sizeof(float), 0);

{
auto dump = [](const std::string& fname, const std::vector<float> v,
const int maxels) {
if (v.empty()) return false;
float* outp = (float*)malloc(sizeof(float) * maxels);
if (!outp) return false;
int step = v.size() < maxels ? 1 : v.size() / maxels;
int ii = 0, jj = 0;
while (ii < maxels && jj < v.size()) {
outp[ii] = v[jj];
ii = ii + 1;
jj = jj + step;
}
FILE* fp = fopen(fname.c_str(), "wb");
if (!fp) {
free(outp);
return false;
}
fwrite((void*)outp, sizeof(float), ii, fp);
fclose(fp);
free(outp);
return true;
};
dump(ifile.out_dir + "/" + "vv-int-rms.dat", sol_rms_final, 1000);
dump(ifile.out_dir + "/" + "vv-int-avg.dat", sol_avg_final, 1000);
dump(ifile.out_dir + "/" + "vv-int-coverage.dat", coverage, 1000);
dump(ifile.out_dir + "/" + "vv-int-sample1.dat", sol1, 1000);
std::cout << "Wrote regression data" << std::endl;
}
}

MPI_Bcast(&(sol_avg_final[0]), (int)sol_avg_final.size(), MPI_FLOAT, 0,
MPI_COMM_WORLD);
MPI_Bcast(&(sol_rms_final[0]), (int)sol_rms_final.size(), MPI_FLOAT, 0,
MPI_COMM_WORLD);
MPI_Bcast(&(coverage[0]), (int)coverage.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

timedBarrierPoint(0 == my_mpi_rank,
"phase1: begin to construct the transpose");
if (0 == my_mpi_rank) std::cout << "Construct the transpose" << std::endl;

float(*intensity_transpose_buf)[number_frames] =
(float(*)[number_frames])ptr_intensity_transpose_data;
global_transpose(&(intensity_buf[0][0]), &(intensity_transpose_buf[0][0]));

timedBarrierPoint(0 == my_mpi_rank, "phase1: transpose complete");
MPI_Barrier(MPI_COMM_WORLD);
if (0 == my_mpi_rank) std::cout << "transpose complete" << std::endl;

__sync_synchronize();
intensity_transpose_ready = true;

return 0;
}

template <typename P1Elems>
int InitializeCameraCalibration(Phase1Settings& sett, P1Elems& elems) {
typedef typename P1Elems::Model Model;
upsp::FileInputs& ifile = sett.ifile;
Model& model = *elems.model;

elems.cals.resize(ifile.cameras);
elems.first_frames.resize(ifile.cameras);
for (unsigned int c = 0; c < ifile.cameras; ++c) {
cv::Mat img = elems.cams[c]->get_frame(1);
img.convertTo(elems.first_frames[c], CV_32F);
cv::Mat img8U = upsp::convert2_8U(img);

if (0 == my_mpi_rank) imwrite(img8U, sett, c, "8bit-raw.png");
if (0 == my_mpi_rank) imwrite(elems.first_frames[c], sett, c, "raw.exr");

upsp::read_json_camera_calibration(elems.cals[c], ifile.cals[c]);
}

return 0;
}

template <typename P1Elems>
int InitializeImagePatches(Phase1Settings& sett, P1Elems& elems) {
const auto& model = *(elems.model);
elems.patchers.resize(sett.ifile.cameras);
for (unsigned int c = 0; c < sett.ifile.cameras; ++c) {
cv::Mat img = elems.cams[c]->get_frame(1);
cv::Mat img8U = upsp::convert2_8U(img);

std::vector<upsp::Target> targs;
{
const auto fname = sett.ifile.targets[c];
std::vector<upsp::Target> orig_targs;
std::vector<upsp::Target> fiducials;
if (!upsp_files::read_psp_target_file(fname, orig_targs)) return 1;
if (!upsp_files::read_psp_target_file(fname, fiducials, false, "*Fiducials")) return 1;
std::vector<upsp::Target> tmptargs = orig_targs;
tmptargs.insert(tmptargs.end(), fiducials.begin(), fiducials.end());
const auto oblique_angle = sett.ifile.oblique_angle;
const float thresh = deg2_rad(180. - std::min(oblique_angle + 5.0, 90.0));
getTargets(model, elems.cals[c], tmptargs, elems.scene, thresh, targs);
elems.cals[c].map_points_to_image(targs);
}

if (0 == my_mpi_rank) {
const auto img_out = upsp::add_targets(targs, img8U, cv::Scalar(0, 255, 0), true);
imwrite(img_out, sett, c, "8bit-projected-fiducials.png");
}

std::vector<float> diams;
get_target_diameters(model, elems.cals[c], targs, elems.scene, diams);
for (unsigned int i = 0; i < targs.size(); ++i) {
targs[i].diameter = diams[i] * sett.target_diam_sf;
}

std::vector<std::vector<upsp::Target>> clusters;
const int bound_pts = sett.bound_thickness + sett.buffer_thickness;
upsp::cluster_points(targs, clusters, bound_pts);

if (0 == my_mpi_rank) {
std::cout << "Sorted " << targs.size() << " targets into ";
std::cout << clusters.size() << " clusters" << std::endl;
}

if (0 == my_mpi_rank) {
auto img_out = img8U.clone();
std::vector<cv::Scalar> cluster_colors;
upsp::get_colors(clusters.size(), cluster_colors);
for (unsigned int i = 0; i < clusters.size(); ++i) {
const auto color = cluster_colors[i];
img_out = upsp::add_targets(clusters[i], img_out, color, false);
}
imwrite(img_out, sett, c, "8bit-fiducial-clusters.png");
}


cv::Mat_<uint16_t> img16 = img.clone();
const auto bit_depth = elems.cams[c]->get_bit_depth();
std::vector<int> edges;
std::vector<int> counts;
upsp::intensity_histc(img16, edges, counts, bit_depth, TWO_POW(8));
unsigned int thresh = edges[upsp::first_min_threshold(counts, 5)] + 5;

elems.patchers[c] = new upsp::PatchClusters<float>(
clusters, img.size(), sett.bound_thickness, sett.buffer_thickness
);
elems.patchers[c]->threshold_bounds(img16, thresh, 2);

if (0 == my_mpi_rank) {
auto img_out = img8U.clone();
cv::cvtColor(img_out, img_out, cv::COLOR_GRAY2RGB);
for (int i = 0; i < elems.patchers[c]->bounds_x.size(); ++i) {
for (int j = 0; j < elems.patchers[c]->bounds_x[i].size(); ++j) {
img_out.at<cv::Vec3b>(
elems.patchers[c]->bounds_y[i][j],
elems.patchers[c]->bounds_x[i][j]
) = cv::Vec3b(255, 0, 0);
}
}
imwrite(img_out, sett, c, "8bit-cluster-boundaries.png");
}
}

return 0;
}

template <typename P1Elems>
int InitializeModel(Phase1Settings& sett, P1Elems& elems) {
if (sett.ifile.has_normals()) {
upsp_files::set_surface_normals(sett.ifile.normals, *(elems.model));
timedBarrierPoint(0 == my_mpi_rank, ("Overwrote select model surface "
"normals (using '" + sett.ifile.normals + "')").c_str());
}
{
psp::BlockTimer bt(0 == my_mpi_rank, "Creating BVH");
elems.scene = createBVH(*(elems.model), elems.triNodes);
}
return 0;
}


template <typename P1Elems>
int phase0(Phase1Settings& sett, P1Elems& elems) {
if (InitializeModel(sett, elems)) return 1;
if (InitializeCameraCalibration(sett, elems)) return 1;
if (InitializeImagePatches(sett, elems)) return 1;
timedBarrierPoint(0 == my_mpi_rank, "Initialized camera cals and image patching");
return 0;
}


template <typename Model>
struct OverlapNodeStorage {
typedef typename Model::data_type FP;

OverlapNodeStorage(Model* model, unsigned int n_frames)
: model_(model), n_frames_(n_frames), stored_nodes_(0) {}


void store_data(unsigned int nidx, FP* data) {
if (model_->is_overlapping(nidx) && !model_->is_superceded(nidx)) {
data_.resize(data.size() + n_frames_);
log_[nidx] = &data[stored_nodes_ * n_frames_];

std::copy(data, data + n_frames_, &data_[stored_nodes_ * n_frames_]);

++stored_nodes_;
}
}


FP* access_data(unsigned int nidx) {
assert(model_->is_superceded(nidx));

assert(log_.find(nidx) != log_.end());

return log_[nidx];
}



unsigned int n_frames_;

unsigned int stored_nodes_;

std::vector<FP> data_;
std::unordered_map<unsigned int, FP*> log_;  

Model* model_;
};



template <typename P2Elems>
int phase2(Phase2Settings& sett, Phase2Files& p2_files, P2Elems& elems) {
typedef typename P2Elems::Model Model;
auto& model = *(elems.model);
const auto& ifile = sett.ifile;





upsp::PaintCalibration pcal(p2_files.paint_cal);

if (0 == my_mpi_rank) {
std::cout << "Paint Calibration = " << std::endl;
std::cout << pcal << std::endl;
}

elems.tunnel_conditions = upsp::read_tunnel_conditions(ifile.sds);
auto& tcond = elems.tunnel_conditions;
tcond.test_id = ifile.test_id;
tcond.run = ifile.run;
tcond.seq = ifile.sequence;

tcond.ttot += sett.F_to_R;  
float t_inf =
tcond.ttot / (1.0 + (sett.gamma - 1.0) * 0.5 * tcond.mach * tcond.mach);
tcond.ttot -= sett.F_to_R;  
t_inf -= sett.F_to_R;       

float wall_temp = sett.r * (tcond.ttot - t_inf) + t_inf;  

float model_temp = wall_temp;
if (!std::isnan(tcond.tcavg)) {
model_temp = tcond.tcavg;
LOG_INFO(
"*** Using thermocouple average (%4.1fF) "
"for model temp, supersedes estimated temperature "
"based on boundary layer recovery factor (%4.1fF)",
tcond.tcavg, wall_temp
);
} else {
LOG_INFO(
"*** Using estimated temperature based on"
" boundary layer recovery factor (%4.1fF)",
wall_temp
);
}

MPI_Barrier(MPI_COMM_WORLD);





std::vector<float> model_temp_input;
model_temp_input = std::vector<float>(msize, model_temp);

if (sett.read_model_temp) {

LOG_INFO("*** Using temperature file: %s",p2_files.model_temp_p3d.c_str());

if (upsp::is_structured<Model>()) {
model_temp_input = upsp::read_plot3d_scalar_function_file(p2_files.model_temp_p3d);  
if (model_temp_input.size() != msize) {
LOG_ERROR(
"Mode-temperature function file inconsistent with grid "
"(expect %d values, got %d)",
msize, model_temp_input.size()
);
return 1;
}
} else {
std::vector<float> in_model_temp =
upsp::read_plot3d_scalar_function_file(p2_files.model_temp_p3d);  
upsp::P3DModel_<float> steady_grid(p2_files.steady_grid, sett.grid_tol);

model_temp_input = upsp::interpolate(steady_grid, in_model_temp, model, sett.k, sett.p);
}
}
MPI_Barrier(MPI_COMM_WORLD);





std::vector<float> steady;
if (sett.wind_off) {
steady = std::vector<float>(msize, 0.0);
} else {
if (upsp::is_structured<Model>()) {

LOG_INFO("*** Using steady state file: %s",p2_files.steady_p3d.c_str());

steady = upsp::read_plot3d_scalar_function_file(p2_files.steady_p3d);  
if (steady.size() != msize) {
LOG_ERROR(
"Steady-state function file inconsistent with grid "
"(expect %d values, got %d)",
msize, steady.size()
);
return 1;
}
} else {
std::vector<float> in_steady =
upsp::read_plot3d_scalar_function_file(p2_files.steady_p3d);  
upsp::P3DModel_<float> steady_grid(p2_files.steady_grid, sett.grid_tol);

steady = upsp::interpolate(steady_grid, in_steady, model, sett.k, sett.p);
}
}
MPI_Barrier(MPI_COMM_WORLD);





upsp::TransPolyFitter<float> pfitter;
pfitter = upsp::TransPolyFitter<float>(number_frames, sett.degree, msize);





apportion(msize, num_mpi_ranks, &(rank_start_node[0]), &(rank_num_nodes[0]));

upsp::PSPWriter<Model>* h5t;
upsp::PSPWriter<Model>* h5_extra;
if (0 == my_mpi_rank) {
std::cout << "Initializing new hdf5 files:" << std::endl;
std::cout << "    " << (p2_files.h5_out) << std::endl;
h5t = new upsp::PSPWriter<Model>(p2_files.h5_out, &model, number_frames,
1.0, true, sett.trans_nodes);
std::cout << "    " << (p2_files.h5_out_extra) << std::endl;
h5_extra = new upsp::PSPWriter<Model>(p2_files.h5_out_extra, &model, 1);

std::cout << "Writing header information to hdf5 files" << std::endl;
h5t->write_grid(sett.grid_units);
h5_extra->write_grid(sett.grid_units);

h5t->write_tunnel_conditions(elems.tunnel_conditions);
h5_extra->write_tunnel_conditions(elems.tunnel_conditions);

h5t->write_camera_settings(elems.camera_settings);
h5_extra->write_camera_settings(elems.camera_settings);

h5t->write_string_attribute("code_version", sett.code_version);
h5_extra->write_string_attribute("code_version", sett.code_version);
}
MPI_Barrier(MPI_COMM_WORLD);





rms.resize(msize, 0.);
avg.resize(msize, 0.);
gain.resize(msize, 0.);

float(*intensity_transpose_buf)[number_frames] =
(float(*)[number_frames])ptr_intensity_transpose_data;

MPI_Barrier(MPI_COMM_WORLD);

unsigned int node_start = rank_start_node[my_mpi_rank];
unsigned int my_num_nodes = rank_num_nodes[my_mpi_rank];
unsigned int node_end = node_start + my_num_nodes - 1;

std::cout << "Rank " << my_mpi_rank << " will process nodes "
<< "[" << node_start << "," << node_end << "]" << std::endl;

MPI_Barrier(MPI_COMM_WORLD);
if (0 == my_mpi_rank) std::cout << "Beginning node processing" << std::endl;

float(*pressure_transpose_buf)[number_frames] =
(float(*)[number_frames])ptr_pressure_transpose_data;

#pragma omp parallel
{
std::vector<double> local_rms(msize, 0.);
std::vector<double> local_avg(msize, 0.);
std::vector<double> local_gain(msize, 0.);

#pragma omp for
for (unsigned int idx = 0; idx < my_num_nodes; ++idx) {
unsigned int i = node_start + idx;
std::vector<float> node_sol(number_frames, 0.);

if (coverage[i] == 0) {
pfitter.skip_fit(idx);
local_rms[i] = std::numeric_limits<float>::quiet_NaN();
local_avg[i] = std::numeric_limits<float>::quiet_NaN();
local_gain[i] = std::numeric_limits<float>::quiet_NaN();
continue;
}

float Pss = tcond.qbar * steady[i] + tcond.ps;
local_gain[i] = pcal.get_gain(model_temp_input[i], Pss);

for (unsigned int f = 0; f < number_frames; ++f) {
node_sol[f] = sol_avg_final[i] / intensity_transpose_buf[idx][f];
}

std::vector<float> sol_fit = pfitter.eval_fit(&node_sol[0], 1, idx);

for (unsigned int f = 0; f < number_frames; ++f) {
float pressure = (node_sol[f] - sol_fit[f]) * local_gain[i];

node_sol[f] = pressure * 12.0 * 12.0 / tcond.qbar;
pressure_transpose_buf[idx][f] = node_sol[f];

local_rms[i] += (node_sol[f] * node_sol[f]);
local_avg[i] += node_sol[f];
}
}

#pragma omp critical
for (unsigned int i = 0; i < msize; ++i) {
rms[i] += local_rms[i];
avg[i] += local_avg[i];
gain[i] += local_gain[i];
}
}


std::cout << "Rank " << my_mpi_rank << " finished nodes "
<< "[" << node_start << "," << node_end << "]" << std::endl;
MPI_Barrier(MPI_COMM_WORLD);



if (0 == my_mpi_rank)
std::cout << "Global reduction of avg, rms and gain ..." << std::endl;
void* sendbuf = (void*)((0 == my_mpi_rank) ? MPI_IN_PLACE : &(gain[0]));
MPI_Reduce(sendbuf, &(gain[0]), msize, MPI_DOUBLE, MPI_SUM, 0,
MPI_COMM_WORLD);
sendbuf = (void*)((0 == my_mpi_rank) ? MPI_IN_PLACE : &(avg[0]));
MPI_Reduce(sendbuf, &(avg[0]), msize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
sendbuf = (void*)((0 == my_mpi_rank) ? MPI_IN_PLACE : &(rms[0]));
MPI_Reduce(sendbuf, &(rms[0]), msize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
if (0 == my_mpi_rank) std::cout << "Global reduction complete" << std::endl;

if (0 == my_mpi_rank) {

std::vector<float> avg_final(msize, 0.0);
std::vector<float> rms_final(msize, 0.0);
std::vector<float> gain_final(msize, 0.0);
for (unsigned int i = 0; i < msize; ++i) {
avg_final[i] = avg[i] / number_frames;
rms_final[i] = sqrt(rms[i] / number_frames);
gain_final[i] = gain[i];
}

h5_extra->write_new_dataset("rms", rms_final, "delta Cp");
h5_extra->write_new_dataset("average", avg_final, "delta Cp");

h5t->write_new_dataset("rms", rms_final, "delta Cp");

std::cout << "Writing pressure rms, average, gain flat files" << std::endl;
pwrite_full(output_files["rms"].fd, &(rms_final[0]),
rms_final.size() * sizeof(float), 0);
pwrite_full(output_files["avg"].fd, &(avg_final[0]),
avg_final.size() * sizeof(float), 0);
pwrite_full(output_files["gain"].fd, &(gain_final[0]),
gain_final.size() * sizeof(float), 0);

h5_extra->write_new_dataset("coverage", coverage);
h5t->write_new_dataset("coverage", coverage);

for (unsigned int i = 0; i < msize; ++i) {
if (steady[i] > 3.0) {
steady[i] = std::numeric_limits<float>::quiet_NaN();
}
}

std::cout << "Writing steady state flat file" << std::endl;
h5t->write_new_dataset("steady_state", steady, "Cp");
h5_extra->write_new_dataset("steady_state", steady, "Cp");
pwrite_full(output_files["steady_state"].fd, &(steady[0]),
steady.size() * sizeof(float), 0);

std::cout << "Writing model temperature flat file" << std::endl;
h5t->write_new_dataset("model_temp", model_temp_input, "F");
h5_extra->write_new_dataset("model_temp", model_temp_input, "F");
pwrite_full(output_files["model_temp"].fd, &(model_temp_input[0]),
model_temp_input.size() * sizeof(float), 0);

{
upsp::fwrite(p2_files.add_out_dir + "/" + "vv-cp-rms.dat", rms_final,
1000);
upsp::fwrite(p2_files.add_out_dir + "/" + "vv-cp-avg.dat", avg_final,
1000);
std::cout << "Wrote regression data" << std::endl;
}

delete h5t;
delete h5_extra;
}

if (0 == my_mpi_rank)
std::cout << "Write pressure transpose ..." << std::endl;

write_block((void*)ptr_pressure_transpose_data,
output_files["pressure_transpose"].fd);
if (0 == my_mpi_rank)
std::cerr << "## 'pressure_transpose' written" << std::endl;

if (0 == my_mpi_rank) {
std::cout << "Wait for all ranks to complete" << std::endl;
}
timedBarrierPoint(0 == my_mpi_rank, "Phase 2 done");

return 0;
}
