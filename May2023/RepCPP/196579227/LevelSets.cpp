
#include <chrono>
#include "ImageHistogram.hpp"
#include "LevelSets.hpp"


void LevelSetMainFunction(string test_image_dir, string result_image_dir, int lower_threshold, int upper_threshold,
int background_lower_threshold, double v, int grid_resolution, uint_dist_type band_size, bool mark_all, int contour_age_out,
int min_active_contour_size, int max_number_threads, bool write_on_image, bool write_initial){

bool use_resize = false; 
int resize_factor = 2;
uint_dist_type band_size_sq = band_size*band_size;

uint_dist_type big_number = band_size*band_size + 10;
int write_interval = 200;
float speed = 1.0;
int max_iter = 1500;
bool write_distance = false;
bool write_bw = false; 
bool write_background_threshold = false;
bool write_active_grid = false; 
int connected_component_reset_timing = -1;

if (grid_resolution < band_size){
grid_resolution = band_size;
}


string writefile = result_image_dir + "trace.txt";
std::ofstream outwrite(writefile.c_str());

string writefile2 = result_image_dir + "active_contour.txt";
std::ofstream activecontourwrite(writefile2.c_str());

string writefile3 = result_image_dir + "total_contour.txt";
std::ofstream totalcontourwrite(writefile3.c_str());

string writefile4 = result_image_dir + "time.txt";
std::ofstream timewrite(writefile4.c_str());

string writefile5 = result_image_dir + "number_active_blocks.txt";
std::ofstream gridwrite(writefile5.c_str());

outwrite << "LS 13 " << endl;
outwrite << "testing_dir  " << test_image_dir << endl;
outwrite << "lower_threshold  " << lower_threshold << endl;
outwrite << "upper_threshold " << upper_threshold << endl;
outwrite << "background threshold " << background_lower_threshold << endl;
outwrite << "grid resolution " << grid_resolution << endl;
outwrite<< "band size        " << band_size << endl;
outwrite << "v                " << v << endl;
outwrite << "use_resize " << use_resize << " resize factor " << resize_factor << endl;
outwrite << "contour_age_out " << contour_age_out << endl;
outwrite << "write interval " << write_interval << endl;
outwrite << "Speed " << speed << endl;
outwrite << "Mark all " << mark_all << endl;
outwrite << "number threads " << max_number_threads << endl;

vector<string> test_files;
vector<string> masks;

vector<pair<int, Mat> > index_mask_pairs;

ReadDirectory(test_image_dir + "images", test_files);
ReadDirectory(test_image_dir + "masks", masks);

vector<string> dir_contents;

ReadDirectory(test_image_dir, dir_contents);

int temp_index = IsNameInListLS(dir_contents, "ylimit");

string filename;

Mat prev_im;
Mat current_im;
Mat mask;

int mask_index = 0;

int rows = 0; 
int cols = 0; 

int number_images= test_files.size();

Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y, grad, edge_mask;

int_type_t rs_index = 0;
vector<Mat> images;
int_type_t xsize, ysize, zsize;
int_type_t grid_xsize, grid_ysize, grid_zsize;
int max_row = -1;
int original_rows = 0;
int original_cols = 0;

vector< float* > curvature_layers;

Mat set_mask;

Rect roi;

vector<double> energy_history;

vector<bool*> class_flag;
vector<bool*> grid;
vector<bool*> grid_active;
Vec3b intensity_bgr;
Scalar intensity_gray;

int number_channels_above =0;

Mat color_im;


string iter_write_dir;
string command;

Mat bw_im;

string filename_mask;

int_type_t xi, yi, zi;

Mat cloned_image;

vector<uint_dist_type*> dt_xyz;
vector<uint_dist_type*> g_xy;
vector<int8_t*> contour_age;

int_type_t gx, gy, gz, minx, miny, minz, maxx, maxy, maxz;

int_type_t number_threads = max_number_threads;

cout << "Number threads " << number_threads << endl;
omp_set_num_threads(number_threads);

if (temp_index > -1){
vector<string> ylims;
ReadDirectory(test_image_dir + "ylimit", ylims);

if (ylims.size() > 0){
filename = test_image_dir + "ylimit/" + ylims[0];
cout << "Reading " << filename << endl;

mask = imread(filename.c_str(), cv::IMREAD_COLOR);


max_row = mask.rows - 1;
cout << "Mask size " << mask.rows << ", " << mask.cols << endl;

for (int r = 0, rn = mask.rows; r < rn; r++){
for (int c = 0, cn = mask.cols; c < cn; c++){
intensity_bgr = mask.at<Vec3b>(r, c);
number_channels_above = 0;

for (int i = 0; i < 3; i++){
if (intensity_bgr[i] > 200){
number_channels_above++;
}
}

if (number_channels_above == 1){
r < max_row ? max_row = r : 0;
}
}
}

cout << "max row selected .... " << max_row << endl;
outwrite << "Max row selected " << max_row << endl;

roi.x = 0; roi.y = 0;
roi.width = mask.cols;
roi.height = max_row + 1;

set_mask = imread(filename.c_str(), cv::IMREAD_GRAYSCALE);
set_mask.setTo(255);

for (int r = max_row, rn = mask.rows; r < rn; r++){
for (int c = 0, cn = mask.cols; c < cn; c++){

set_mask.at<uchar>(r, c) = 0;
}
}

filename = result_image_dir + "mask_outline.png";
imwrite(filename.c_str(), set_mask);

}	else {
cout << "ylimit dir provided, but empty.  Error -- check input. " << endl;
exit(1);
}



}



outwrite << "number of input files ... " << test_files.size() << endl;

images.resize(test_files.size());

int n_files = test_files.size();


filename = test_image_dir + "images/" + test_files[0];
images[0] = imread(filename.c_str(), cv::IMREAD_GRAYSCALE);

original_rows = images[0].rows;
original_cols = images[0].cols;


if (max_row != -1){

images[0] = images[0](roi);
}

rows = images[0].rows;
cols = images[0].cols;
cout << "rows, cols " << rows << ", " << cols << endl;

xsize = cols;  ysize = rows; zsize = number_images;


grid_xsize = xsize/grid_resolution;
grid_xsize += xsize % grid_resolution > 0;

grid_ysize = ysize/grid_resolution;
grid_ysize += ysize % grid_resolution > 0;

grid_zsize = zsize/grid_resolution;
grid_zsize += zsize % grid_resolution > 0;


for (int i = 0; i < number_images; i++){
if (i %100 == 0){
cout << "i allocated out of : " << i << " out of " << number_images << endl;
}
class_flag.push_back(new bool[rows*cols]);
#pragma omp parallel for
for (int h = 0; h < rows*cols; h++){
class_flag[i][h] = true;
}
}

for (int_type_t i = 0; i < grid_zsize; i++){

grid.push_back(new bool[grid_xsize*grid_ysize]);
grid_active.push_back(new bool[grid_xsize*grid_ysize]);
#pragma omp parallel for
for (int_type_t h = 0; h < grid_xsize*grid_ysize; h++){
grid[i][h] = false;
grid_active[i][h] = false;
}
}


#pragma omp parallel for private (filename)
for (int i = 1; i <n_files; i++){

filename = test_image_dir + "images/" + test_files[i];
if (i %1 == 0){
cout << "Loading ... " << filename << endl;
}

images[i] = imread(filename.c_str(), cv::IMREAD_GRAYSCALE);

if (max_row != -1){

images[i] = images[i](roi);
}
}


for (int i = 0; i <n_files; i++){
mask_index = IsNameInListLS(masks, test_files[i]);
if (mask_index > -1){
int image_value;

filename_mask = test_image_dir + "masks/" + test_files[i];

outwrite << "regular mask " << test_files[i] << endl;
mask = imread(filename_mask.c_str(), cv::IMREAD_COLOR);

if (use_resize){
Mat temp_im;
resize(mask, temp_im, Size(mask.cols/resize_factor, mask.rows/resize_factor));
mask = temp_im.clone();

}

if (max_row != -1){
mask = mask(roi);
}

for (int r = 0; r < rows; r++){
for (int c = 0; c < cols; c++){

intensity_bgr = mask.at<Vec3b>(r, c);
number_channels_above = 0;

for (int k = 0; k < 3; k++){
if (intensity_bgr[k] > 200){
number_channels_above++;
}
}

if (number_channels_above == 1){

intensity_gray = images[i].at<uchar>(r, c);

if (intensity_gray[0] > lower_threshold && intensity_gray[0] < upper_threshold){
class_flag[i][c*rows + r] = false;
}
}

}
}
}
}


outwrite << "number of voxels       : " << xsize*ysize*zsize << endl;
outwrite << "grid x, y, z " << grid_xsize << ", " << grid_ysize << ", " << grid_zsize << endl;
outwrite << "number of grid elements: " << grid_xsize*grid_ysize*grid_zsize << endl;

auto t0 = std::chrono::high_resolution_clock::now();





vector<vector< int_type_t*> > d_gamma_indices_per_thread(number_threads, vector<int_type_t*>()); 
vector<int_type_t> count_d_gamma_per_thread(number_threads, 0);
int_type_t number_d_gamma = 0;

for (int_type_t i = 0; i < number_threads; i++){
d_gamma_indices_per_thread[i].push_back(new int_type_t[rows*cols]); 
}

cout << "Creating structures " << endl;

vector<uint32_t*> cc_map;

for (int i = 0; i < number_images; i++){
if (i %100 == 0){
cout << "i allocated out of : " << i << " out of " << number_images << endl;
}
dt_xyz.push_back(new uint_dist_type[rows*cols]);
contour_age.push_back(new int8_t[rows*cols]);
cc_map.push_back(new uint32_t[rows*cols]);

}

ImageHistogram V1;
ImageHistogram V2;

double energy_last_iter = 0;
double current_energy = 0;


int thread_id;

vector<ImageHistogram> vs1;
vector<ImageHistogram> vs2;

for (int_type_t i = 0; i < number_threads; i++){
vs1.push_back(ImageHistogram());
vs2.push_back(ImageHistogram());
}

for (int_type_t z = 0; z < zsize; z++){
for (int_type_t x = 0; x < xsize; x++){
for (int_type_t y = 0; y < ysize; y++){
if (class_flag[z][x*ysize + y] == false){
gx = x/grid_resolution;
gy = y/grid_resolution;
gz = z/grid_resolution;

ReturnLimits(gx, gy, gz, grid_xsize, grid_ysize, grid_zsize, minx, miny, minz, maxx, maxy, maxz);

for (int_type_t x0 = minx; x0 <= maxx; x0++){
for (int_type_t y0 = miny; y0 <= maxy; y0++){
for (int_type_t z0 = minz; z0 <= maxz; z0++){
grid_active[z0][x0*grid_ysize + y0] = true;
}
}
}
}
}
}
}

if (!mark_all){
#pragma omp parallel for
for (int_type_t i = 0; i < grid_zsize; i++){
for (int_type_t h = 0; h < grid_xsize*grid_ysize; h++){
grid[i][h] = grid_active[i][h];
}
}
}	else {
#pragma omp parallel for
for (int_type_t i = 0; i < grid_zsize; i++){
for (int_type_t h = 0; h < grid_xsize*grid_ysize; h++){
grid[i][h] = true;
}
}
}

int image_value;
cout << "Creating initial histogram -- parallel" << endl;    

int_type_t xstart, xend, ystart, yend, zstart, zend;

for (int_type_t gz = 0; gz < grid_zsize; gz++){
for (int_type_t gx = 0; gx < grid_xsize; gx++){
for (int_type_t gy = 0; gy < grid_ysize; gy++){
if (grid[gz][gx*grid_ysize + gy] == true){
gx == grid_xsize - 1 ? xend = xsize : xend = (gx + 1)*grid_resolution;
gy == grid_ysize - 1 ? yend = ysize : yend = (gy + 1)*grid_resolution;
gz == grid_zsize - 1 ? zend = zsize : zend = (gz + 1)*grid_resolution;
xstart = (gx)*grid_resolution;
ystart = (gy)*grid_resolution;
zstart = (gz)*grid_resolution;
#pragma omp parallel for private(thread_id, image_value, intensity_gray)
for (int_type_t xi = xstart; xi < xend; xi++){
thread_id = omp_get_thread_num();
for (int_type_t yi = ystart; yi < yend; yi++){
for (int_type_t zi = zstart; zi < zend; zi++){
intensity_gray = images[zi].at<uchar>(yi, xi);
image_value = intensity_gray[0];
if (class_flag[zi][xi*ysize + yi] == false){

vs2[thread_id].AddToHistogram(image_value);
}	else {
if (image_value > background_lower_threshold){
vs1[thread_id].AddToHistogram(image_value);
}

}
}
}
}
}
}
}
}


for (int_type_t i = 0; i < number_threads; i++){
V1.AddTo(vs1[i]);
V2.AddTo(vs2[i]);
}


cout << "Computing statistics, V1" << endl;
V1.computeStatistics();
cout << "Computing statistics, V2" << endl;
V2.computeStatistics();
cout << "Beginning statistics, new function.  Group1 (non-root) : " << V1.mu << ", " << V1.sigma_sq << ", " << V1.n << endl;
cout << "Beginning statistics, new function.  Group2 (    root) : " << V2.mu << ", " << V2.sigma_sq << ", " << V2.n << endl;

outwrite << "Beginning statistics, new function.  Group1 (non-root) : " << V1.mu << ", " << V1.sigma_sq << ", " << V1.n << endl;
outwrite << "Beginning statistics, new function.  Group2 (    root) : " << V2.mu << ", " << V2.sigma_sq << ", " << V2.n << endl;

outwrite << "Total number blocks " << grid_xsize*grid_ysize*grid_zsize << endl;

cout << "Allocating memory .... " << endl;

gridwrite << "iteration     number_blocks_hist     number_blocks_contour  " << endl;


#pragma omp parallel for
for (int i = 0; i < number_images; i++){
for (int_type_t slice = 0; slice < int_type_t(rows*cols); slice++){
contour_age[i][slice] = 0;
}
}

cout << "Calculating edge list." << endl;

int_type_t number_aged_out;
pair<int_type_t, int_type_t> numbers_pair  = UpdateEdgeListDiffStructure(class_flag, dt_xyz, contour_age, d_gamma_indices_per_thread,
count_d_gamma_per_thread, number_threads, xsize, ysize, zsize, false, 0, contour_age_out);

number_d_gamma = numbers_pair.first;
number_aged_out = numbers_pair.second;

outwrite << "Number aged out " << number_aged_out << endl;
cout << "Number aged out " << number_aged_out << endl;
cout << "Number of d_gamma members -- new function " << number_d_gamma << endl;

current_energy = ComputeEnergyWithHistograms(V1, V2, v, number_d_gamma, number_aged_out);
cout << "current energy " << current_energy << endl;
outwrite << "beginning energy " << current_energy << endl;

cout << "Before distance transform " << endl;

DistanceTransformSqdParallelTubeXIV(d_gamma_indices_per_thread, count_d_gamma_per_thread, dt_xyz, big_number, band_size,
xsize, ysize, zsize, grid_active, grid_xsize, grid_ysize, grid_zsize, grid_resolution, number_threads);

cout << "After distance transform " << endl;



bw_im = images[0].clone(); 

vector<Mat> bw_ims;

for (int_type_t i= 0; i < number_threads; i++){
bw_ims.push_back(images[0].clone());
}


activecontourwrite << number_d_gamma << endl;
totalcontourwrite << number_d_gamma << endl;

energy_last_iter = 10000000;

for (int iter = 0; iter < max_iter && number_d_gamma > int_type_t(min_active_contour_size); iter++){

if (iter != 0){
energy_last_iter = current_energy;
}
cout << "iter " << iter << endl;

if ( iter  == 0 ){

bool some_found;

#pragma omp parallel for private(thread_id, filename, some_found)
for (int i = 0; i < number_images; i++){
thread_id = omp_get_thread_num();
bw_ims[thread_id].setTo(0);
some_found = false;
for (int r = 0; r < rows; r++){
for (int c = 0; c < cols; c++){
if (class_flag[i][c*rows + r] == false){
bw_ims[thread_id].at<uchar>(r, c) = 255;
some_found = true;
}
}
}

if (iter == 0 && some_found){
#pragma omp critical
{
index_mask_pairs.push_back(pair<int, Mat>(i, bw_ims[thread_id].clone()));

}
}
}
}

if (iter != 0 && (iter % (write_interval) == 0) && write_on_image == true){
iter_write_dir  = result_image_dir + "color" + ToString<int>(iter);
WriteResultOnImages(iter_write_dir, number_images, images, class_flag);

}

if (iter  == 0 && write_background_threshold == true){

iter_write_dir  = result_image_dir + "background" + ToString<int>(iter);
WriteThresholdsOnImages(iter_write_dir, number_images, images, class_flag, lower_threshold, upper_threshold, background_lower_threshold);

}

if ( (iter % write_interval == 0 && write_bw == true) || (iter == 0 && write_initial == true) ){

iter_write_dir  = result_image_dir + ToString<int>(iter);
WriteBWResultImages(iter_write_dir, number_images, bw_ims, class_flag);

}


if (iter % write_interval == 0 && write_active_grid == true){

iter_write_dir  = result_image_dir + "grid_activity" + ToString<int>(iter);

WriteGridImages(iter_write_dir, number_images, bw_ims, grid_active, grid_resolution,
grid_xsize,  grid_ysize,  grid_zsize);

}

if (iter % write_interval == 0 && write_distance == true){

iter_write_dir  = result_image_dir + "distance" + ToString<int>(iter);

WriteDistanceImages(iter_write_dir, number_images, bw_ims, dt_xyz,
class_flag, contour_age, contour_age_out, band_size_sq);

}

int number_hist = 0;  int number_active = 0;

for (int_type_t i = 0; i < grid_zsize; i++){
for (int xy = 0, xyn = grid_xsize*grid_ysize; xy < xyn; xy++){
number_hist += grid[i][xy];
number_active += grid_active[i][xy];
}

}
gridwrite << iter << "   " << number_hist << "     " << number_active << endl;

cout << "Exiting after writing preliminary information." << endl;


cout << "Before stats " << endl;

ImageHistogram V12;
ImageHistogram V21;

CurvatureAndDTParallelOneAllocLayers(class_flag, dt_xyz, band_size, number_images == 1, xsize, ysize, zsize, images, V1, V2, V12, V21, curvature_layers, v, speed,
lower_threshold, upper_threshold, grid_active, grid_xsize, grid_ysize, grid_resolution);


cout << "Adding and subtracting histograms ... " << endl;
V1.AddTo(V21);  V1.SubtractFrom(V12);
cout << "V2 ... " << endl;
V2.AddTo(V12);  V2.SubtractFrom(V21);


{
if (iter < 10){
cout << "Dealing with the case of adding back initialization information " << index_mask_pairs.size() << endl;

vector<ImageHistogram> vs12_cc;

for (int_type_t i = 0; i < number_threads; i++){
vs12_cc.push_back(ImageHistogram());
}

ImageHistogram V12_cc;
int first_index;
for (int j = 0, jn = index_mask_pairs.size(); j < jn; j++){
first_index = index_mask_pairs[j].first;
#pragma omp parallel for private (thread_id)
for (int_type_t xi = 0; xi < xsize; xi++){
thread_id = omp_get_thread_num();
for (int_type_t yi = 0; yi < ysize; yi++){
if (index_mask_pairs[j].second.at<uchar>(yi, xi) > 200){ 

if (class_flag[first_index][xi*ysize + yi] == true){
class_flag[first_index][xi*ysize + yi] = false;
contour_age[first_index][xi*ysize + yi] = 0;
vs12_cc[thread_id].AddToHistogram(int(images[first_index].at<uchar>(yi, xi)));

}

}
}
}
}

for (int_type_t i = 0; i < number_threads; i++){
V12_cc.AddTo(vs12_cc[i]);
}

V12_cc.Sum();


outwrite << "Flipping labels back to initialization ... this number needed treatment .. " << V12_cc.n << endl;

cout << "Adding and subtracting histograms ... initialization. " << endl;
V1.SubtractFrom(V12_cc);
V2.AddTo(V12_cc);

cout << "Done initialization business." << endl;
}

}

cout << "Running Edge Finding... " << endl;



pair<int_type_t, int_type_t> numbers_pair  = UpdateEdgeListDiffStructure(class_flag, dt_xyz, contour_age, d_gamma_indices_per_thread,
count_d_gamma_per_thread, number_threads, xsize, ysize, zsize, false, 0, contour_age_out);




number_d_gamma = numbers_pair.first;
number_aged_out = numbers_pair.second;

activecontourwrite << number_d_gamma << endl;
totalcontourwrite << number_d_gamma + number_aged_out << endl;


outwrite << "Number active contour                 " << number_d_gamma << endl;
outwrite << "Number aged out w/in active grid      " << number_aged_out << endl;


cout << "Number active contour " << number_d_gamma << endl;
cout << "Number aged out       " << number_aged_out << endl;


UpdateGridWithNewEdges( d_gamma_indices_per_thread,
count_d_gamma_per_thread, grid_active, xsize,  ysize,
grid_xsize, grid_ysize, grid_zsize, grid_resolution, number_threads);



cout << "Updating histogram with new frontier .... " << endl;
number_hist  = 0;
for (int_type_t gz = 0; gz < grid_zsize; gz++){
for (int_type_t gx = 0; gx < grid_xsize; gx++){
for (int_type_t gy = 0; gy < grid_ysize; gy++){
if (grid_active[gz][gx*grid_ysize + gy] == true && grid[gz][gx*grid_ysize + gy] == false){
grid[gz][gx*grid_ysize + gy] = true;
number_hist++;

for (int_type_t i = gz*grid_resolution; i < zsize && i <(gz + 1)*grid_resolution; i++){
for (int_type_t r = gy*grid_resolution; r < ysize && r <(gy + 1)*grid_resolution; r++){
for (int_type_t c = gx*grid_resolution; c < xsize && i <(gx + 1)*grid_resolution; c++){
image_value = int(images[i].at<uchar>(r, c));

if (image_value > background_lower_threshold){
V1.AddToHistogram(image_value);
}
}
}

}

}
}
}
}

cout << "Number hist blocks added ... " << number_hist << endl;


DistanceTransformSqdParallelTubeXIV(d_gamma_indices_per_thread, count_d_gamma_per_thread, dt_xyz, big_number, band_size,
xsize, ysize, zsize, grid_active, grid_xsize, grid_ysize, grid_zsize, grid_resolution, number_threads);


outwrite <<"iteration " << iter << "-------" << endl;
outwrite << "Before switch, number in each group " << V1.n << ", " << V2.n << endl;
outwrite << "V12, V21 numbers " << V12.n << ", " << V21.n << endl;

V1.computeStatistics();
V2.computeStatistics();

cout << "iter statistics(mu, sigma_sq, n).  Group1 (non-root) : " << V1.mu << ", " << V1.sigma_sq << ", " << V1.n << endl;
cout << "iter statistics(mu, sigma_sq, n).  Group2 (    root) : " << V2.mu << ", " << V2.sigma_sq << ", " << V2.n << endl;
outwrite << "iter statistics(mu, sigma_sq, n).  Group1 (non-root) : " << V1.mu << ", " << V1.sigma_sq << ", " << V1.n << endl;
outwrite << "iter statistics(mu, sigma_sq, n).  Group2 (    root) : " << V2.mu << ", " << V2.sigma_sq << ", " << V2.n << endl;

current_energy = ComputeEnergyWithHistograms(V1, V2, v, number_d_gamma, number_aged_out);

energy_history.push_back(current_energy);


if (iter == 0){
energy_last_iter = current_energy + 1;
}

cout << "current energy " << current_energy << endl;
outwrite << "iter " << iter << "  energy " << current_energy << endl;


if (connected_component_reset_timing > 0 && ((iter + 1) % connected_component_reset_timing == 0) && iter != 0){

cout << "iter + 1 " << iter + 1 << endl;
cout << "cc reset timing " << connected_component_reset_timing << endl;

ImageHistogram V21_cc;
vector<int_type_t> counts_per;


for (int i = 0; i < number_images; i++){
#pragma omp parallel for
for (int h = 0; h < rows*cols; h++){
cc_map[i][h] = class_flag[i][h]; 
}

}


cout << "Before connected components " << endl;
ConnectedComponents(cc_map, counts_per, xsize, ysize, zsize);
cout << "After connected components " << counts_per.size() - 2 <<  endl;
bool* cc_we_want = new bool[counts_per.size()];
uint32_t cc_value;

for (int j = 0, jn = counts_per.size(); j < jn; j++){
cc_we_want[j] = false;
}

cout << "Identifying connected components we want from initialization " << endl;
for (int j = 0, jn = index_mask_pairs.size(); j < jn; j++){
for (int_type_t xi = 0; xi < xsize; xi++){
for (int_type_t yi = 0; yi < ysize; yi++){
if (index_mask_pairs[j].second.at<uchar>(yi, xi) > 200){ 
cc_value  = cc_map[index_mask_pairs[j].first][xi*ysize + yi];

if (cc_value >= 2){
cc_we_want[cc_value] = true;
}
}
}
}
}

int number_selected = 0;
for (int j = 0, jn = counts_per.size(); j < jn; j++){
if (cc_we_want[j] == true){
number_selected++;
outwrite << "Component selecteed " << j << " with volume " << counts_per[j] << endl;
}
}

outwrite << "Number of selected connected components " << number_selected << endl;
cout << "Number of selected connected components " << number_selected << endl;

for (int i = 0; i < number_images; i++){
for (int r = 0; r < rows; r++){
for (int c = 0; c < cols; c++){

cc_value  = cc_map[i][c*rows + r];

if (cc_value >= 2 && cc_we_want[cc_value] == false){
class_flag[i][c*rows + r] = true;
V21_cc.AddToHistogram(int(images[i].at<uchar>(r, c)));

}
}

}
}

delete [] cc_we_want;

V1.AddTo(V21_cc);
V2.SubtractFrom(V21_cc);

V1.computeStatistics();
V2.computeStatistics();

cout << "iter statistics (mu, sigma, n).  Group1 (non-root) : " << V1.mu << ", " << V1.sigma_sq << ", " << V1.n << endl;
cout << "iter statistics (mu, sigma, n).  Group2 (    root) : " << V2.mu << ", " << V2.sigma_sq << ", " << V2.n << endl;
outwrite << "iter statistics (mu, sigma, n).  Group1 (non-root) : " << V1.mu << ", " << V1.sigma_sq << ", " << V1.n << endl;
outwrite << "iter statistics (mu, sigma, n).  Group2 (    root) : " << V2.mu << ", " << V2.sigma_sq << ", " << V2.n << endl;

current_energy = ComputeEnergyWithHistograms(V1, V2, v, number_d_gamma, number_aged_out);

energy_history.push_back(current_energy);


if (iter == 0){
energy_last_iter = current_energy + 1;
}

cout << "current energy " << current_energy << endl;
outwrite << "iter " << iter << "  energy " << current_energy << endl;

}
}


{
iter_write_dir  = result_image_dir + "final";

WriteBWResultImages(iter_write_dir, number_images, bw_ims, class_flag);
}


cout << "Deallocating dx, contour age." << endl;
for (int i = 0; i < number_images; i++){
delete [] dt_xyz[i];
delete [] contour_age[i];
if (i %100 == 0){
cout << "i allocated out of : " << i << " out of " << number_images << endl;
}
}

cout << "Allocating cc_map " << endl;

vector<int_type_t> counts_per;

for (int i = 0; i < number_images; i++){
#pragma omp parallel for
for (int h = 0; h < rows*cols; h++){
cc_map[i][h] = class_flag[i][h]; 
}

}

cout << "Before connected components " << endl;
ConnectedComponents(cc_map, counts_per, xsize, ysize, zsize);
cout << "After connected components " << counts_per.size() - 2 <<  endl;

bool* cc_we_want = new bool[counts_per.size()];
uint32_t cc_value;

int index_biggest_cc = -1;
int_type_t max_cc = 0;

for (int j = 0, jn = counts_per.size(); j < jn; j++){
cc_we_want[j] = false;
}

cout << "Identifying connected components we want from initialization " << endl;
for (int j = 0, jn = index_mask_pairs.size(); j < jn; j++){
for (int_type_t xi = 0; xi < xsize; xi++){
for (int_type_t yi = 0; yi < ysize; yi++){
if (index_mask_pairs[j].second.at<uchar>(yi, xi) > 200){ 

cc_value  = cc_map[index_mask_pairs[j].first][xi*ysize + yi];

if (cc_value >= 2){
cc_we_want[cc_value] = true;
}
}
}
}
}

int number_selected = 0;
for (int j = 0, jn = counts_per.size(); j < jn; j++){
if (cc_we_want[j] == true){
number_selected++;
outwrite << "Component selected " << j << " with volume " << counts_per[j] << endl;
if (counts_per[j] > max_cc){
max_cc = counts_per[j];
index_biggest_cc = j;
}

}
}

outwrite << "Number of selected connected components " << number_selected << endl;
cout << "Number of selected connected components " << number_selected << endl;


auto t1 = std::chrono::high_resolution_clock::now();

cout << "Time for segmentation ... " << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()<< " seconds "<< endl;

timewrite << "Time for segmentation ... " << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count()<< " seconds "<< endl;



{
iter_write_dir  = result_image_dir + "CC";
WriteBWResultImagesConnectedComponents(iter_write_dir, number_images, bw_ims, class_flag, cc_we_want,
cc_map, -1 );
}

{

iter_write_dir  = result_image_dir + "CClargest";
WriteBWResultImagesConnectedComponents(iter_write_dir, number_images, bw_ims, class_flag, cc_we_want,
cc_map, index_biggest_cc );

}

if (write_on_image == true){
iter_write_dir  = result_image_dir + "color_final";
WriteResultOnImages(iter_write_dir, number_images, images, class_flag);

}

delete [] cc_we_want;

for (int i = 0; i < number_images; i++){
delete [] cc_map[i];
delete [] class_flag[i];
}

for (int i = 0; i < int(curvature_layers.size()); i++){
delete [] curvature_layers[i];
}

}




void CurvatureAndDTParallelOneAllocLayers(vector<bool*>& class_flag, vector<uint_dist_type*>& dt_xyz, uint_dist_type band_size,
bool two_d_images,  int_type_t xsize, int_type_t ysize, int_type_t zsize, vector<Mat>& images,
ImageHistogram& V1, ImageHistogram& V2, ImageHistogram& V12, ImageHistogram& V21, vector< float* >& current_layers,
float v, float speed, int lower_threshold, int upper_threshold,
vector<bool*>& active_grid, int_type_t grid_xsize, int_type_t grid_ysize, int_type_t grid_resolution ){


if (two_d_images || zsize < 3){
cout << "Estimate curvature in LS1.cpp -- not set up for two d images or for zsize < 3" << zsize << endl;
cout << "Quit " << endl;
exit(1);
}

cout << "In estimate curvature and DT parallel OneAlloc Layers! " << endl;
uint_dist_type band_sq = band_size*band_size;



int number_threads = omp_get_max_threads();
cout << "Number threads " << number_threads << endl;

vector<ImageHistogram> vs12;
vector<ImageHistogram> vs21;

for (int i = 0; i < number_threads; i++){
vs12.push_back(ImageHistogram());vs21.push_back(ImageHistogram());
}

vector<int_type_t> v_number_switches(number_threads, 0);
int_type_t number_switches = 0;

vector<double> data_values_for_pixel_value(256, 0);

for (int i = 0; i < 256; i++){
data_values_for_pixel_value[i] = pow(double(i)/255.0 - V2.mu, 2)/(2*V2.sigma_sq) - pow(double(i)/255.0 - V1.mu, 2)/(2*V1.sigma_sq) + log10(V2.sigma_sq/V1.sigma_sq);
}

bool class_label_before;
bool class_label_after;
int image_value;
double dGamma_dt = 0;
double dt_temp;

int thread_id = 0;

if (current_layers.size() == 0){
for (int i = 0; i < 3; i++){
current_layers.push_back(new float[xsize*ysize]);
}
}

for (int_type_t zc = 0; zc < 3; zc++){
#pragma omp parallel for
for (int_type_t slice = 0; slice < xsize*ysize; slice++){
if (dt_xyz[zc][slice] <= band_sq){
current_layers[zc][slice] = sqrt(float(dt_xyz[zc][slice]));
if (class_flag[zc][slice] == false){
current_layers[zc][slice] *= -1.0;
}
}	else {
current_layers[zc][slice] = 0;
}
}
}


float local_max, local_min, curvature;
int_type_t volume_index;
int_type_t gz;
int_type_t xstart, xend, ystart, yend;

for (int_type_t zi = 1; zi < zsize - 1; zi++){
gz = zi/grid_resolution;

for (int_type_t gx = 0; gx < grid_xsize; gx++){
for (int_type_t gy = 0; gy < grid_ysize; gy++){
if (active_grid[gz][gx*grid_ysize + gy] == true){

gx == 0 ? xstart = 1 : xstart = gx*grid_resolution;
gy == 0 ? ystart = 1 : ystart = gy*grid_resolution;
gx == grid_xsize - 1 ? xend = xsize - 1 : xend = (gx + 1)*grid_resolution;
gy == grid_ysize - 1 ? yend = ysize - 1 : yend = (gy + 1)*grid_resolution;

#pragma omp parallel for private(local_max, local_min, curvature, thread_id, dGamma_dt, class_label_before, class_label_after, dt_temp, image_value, volume_index)
for (int_type_t xi = xstart; xi < xend; xi++){
thread_id = omp_get_thread_num();
for (int_type_t yi = ystart; yi < yend; yi++){

if (dt_xyz[zi][xi*ysize + yi] < band_sq - 1){
image_value = int(images[zi].at<uchar>(yi, xi));
if (image_value > lower_threshold && image_value < upper_threshold){
curvature = computeCurvatureFloat(current_layers, yi, xi, 1, two_d_images, ysize, local_max, local_min);

volume_index = xi*ysize + yi;

dGamma_dt = v*curvature + data_values_for_pixel_value[image_value];

class_label_before = class_flag[zi][volume_index];

dt_temp = current_layers[1][volume_index];

dt_temp += speed*dGamma_dt;

class_label_after = dt_temp > 0; 

if ((class_label_before == true && class_label_after == false) || (class_label_before == false && class_label_after == true)){
v_number_switches[thread_id]++;


if (class_label_before == true && class_label_after == false){
{
vs12[thread_id].AddToHistogram(image_value);
}
class_flag[zi][volume_index] = false; 
}	else {
if (class_label_before == false && class_label_after == true){
{
vs21[thread_id].AddToHistogram(image_value);
}
class_flag[zi][volume_index] = true;  
}
}
}
}
}

}

}
}
} 
} 





for (int_type_t zc = 0; zc < 2; zc++){
#pragma omp parallel for
for (int_type_t slice = 0; slice < xsize*ysize; slice++){
current_layers[zc][slice] = current_layers[zc + 1][slice];
}
}


if (zi != zsize - 2){
#pragma omp parallel for
for (int_type_t slice = 0; slice < xsize*ysize; slice++){
if (dt_xyz[zi + 2][slice] <= band_sq){
current_layers[2][slice] = sqrt(float(dt_xyz[zi + 2][slice]));
if (class_flag[zi + 2][slice] == false){
current_layers[2][slice] *= -1;
}


}	else {
current_layers[2][slice] = 0;

}
}
}
}




for (int i = 0; i < number_threads; i++){
V12.AddTo(vs12[i]);
V21.AddTo(vs21[i]);
number_switches += v_number_switches[i];
}



V12.Sum();  V21.Sum();
cout << "number of switches " << number_switches << " compare histograms " << V12.n + V21.n << endl;

}


float computeCurvatureFloat(vector<float*>& phi, int r, int c, int i, bool two_d_curvature, int rows, float& maxgrad, float& mingrad){


float Dx, Dy, Dz, Dxplus, Dyplus, Dzplus;
float Dxminus, Dyminus, Dzminus, Dxplusy, Dxminusy, Dxplusz, Dxminusz, Dyplusx, Dyminusx, Dyplusz, Dyminusz, Dzminusx, Dzplusx, Dzminusy, Dzplusy;
float nplusx, nplusy, nplusz;
float nminusx, nminusy, nminusz;
float k = 0;

Dx = (phi[i][indexRC(r, c + 1, rows)] - phi[i][indexRC(r, c - 1, rows)])/2.0;
Dy = (phi[i][indexRC(r - 1, c, rows)] - phi[i][indexRC(r + 1, c, rows)])/2.0;

if (two_d_curvature){
Dz = 0;
}	else {
Dz = (phi[i+1][indexRC(r, c, rows)] - phi[i-1][indexRC(r, c, rows)])/2.0;
}

Dxplus = (phi[i][indexRC(r, c + 1, rows)] - phi[i][indexRC(r, c, rows)]);
Dyplus = (phi[i][indexRC(r - 1, c, rows)] - phi[i][indexRC(r, c, rows)]);

if (two_d_curvature){
Dzplus = 0;
} else {
Dzplus = (phi[i+1][indexRC(r, c, rows)] - phi[i][indexRC(r, c, rows)]);
}

Dxminus = (phi[i][indexRC(r, c, rows)] - phi[i][indexRC(r, c - 1, rows)]);
Dyminus = (phi[i][indexRC(r, c, rows)] - phi[i][indexRC(r + 1, c, rows)]);

if (two_d_curvature){
Dzminus = 0;
}	else {
Dzminus = (phi[i][indexRC(r, c, rows)] - phi[i - 1][indexRC(r, c, rows)]);
}

Dxplusy = (phi[i][indexRC(r - 1, c + 1, rows)] - phi[i][indexRC(r - 1, c -1, rows)])/2.0;
Dxminusy = (phi[i][indexRC(r + 1, c + 1, rows)] - phi[i][indexRC(r + 1, c -1, rows)])/2.0;

if (two_d_curvature){
Dxplusz = 0;
Dxminusz = 0;
} else {
Dxplusz = (phi[i + 1][indexRC(r, c + 1, rows)] - phi[i + 1][indexRC(r, c - 1, rows)])/2.0;
Dxminusz = (phi[i - 1][indexRC(r, c + 1, rows)] - phi[i - 1][indexRC(r, c - 1, rows)])/2.0;
}


Dyplusx = (phi[i][indexRC(r - 1, c + 1, rows)] - phi[i][indexRC(r + 1, c + 1, rows)])/2.0;
Dyminusx = (phi[i][indexRC(r - 1, c - 1, rows)] - phi[i][indexRC(r + 1, c + 1, rows)])/2.0;

if (two_d_curvature){
Dyplusz = 0;
Dyminusz = 0;

Dzminusx = 0;
Dzplusx = 0;

Dzminusy = 0;
Dzplusy = 0;

}	else {
Dyplusz = (phi[i+1][indexRC(r - 1, c, rows)] - phi[i+1][indexRC(r + 1, c, rows)])/2.0;
Dyminusz = (phi[i-1][indexRC(r - 1, c, rows)] - phi[i-1][indexRC(r + 1, c, rows)])/2.0;

Dzminusx = (phi[i + 1][indexRC(r, c - 1, rows)] - phi[i - 1][indexRC(r, c - 1, rows)])/2.0;
Dzplusx = (phi[i + 1][indexRC(r, c + 1, rows)] - phi[i - 1][indexRC(r, c + 1, rows)])/2.0;

Dzminusy = (phi[i + 1][indexRC(r + 1, c, rows)] - phi[i - 1][indexRC(r + 1, c, rows)])/2.0;
Dzplusy = (phi[i + 1][indexRC(r - 1, c, rows)] - phi[i - 1][indexRC(r - 1, c, rows)])/2.0;

}



double tolerance = 0.00001;
double denom0 = fabs(pow(Dxplus, 2) + pow((Dyplusx + Dy)/2.0, 2) + pow((Dzplusx + Dz)/2.0, 2));

if (denom0 <= tolerance ){

nplusx = 0;
}		else {
nplusx = Dxplus/sqrt(denom0);
}

double denom1 = fabs(pow(Dyplus, 2) + pow((Dxplusy + Dx)/2.0, 2) + pow((Dzplusy + Dz)/2.0, 2));
if (denom1 <= tolerance ){
nplusy = 0;
} else {
nplusy = Dyplus/sqrt(denom1);
}

double denom2 = fabs(pow(Dxminus, 2) + pow((Dyminusx + Dy)/2.0, 2) + pow((Dzminusx + Dz)/2.0, 2));
if (denom2 <= tolerance ){
nminusx = 0;
} else {
nminusx = Dxminus/sqrt(denom2);
}

double denom3 = fabs(pow(Dyminus, 2) + pow((Dxminusy + Dx)/2.0, 2) + pow((Dzminusy + Dz)/2.0, 2));
if (denom3 <= tolerance ){
nminusy = 0;
} else {
nminusy = Dyminus/sqrt(denom3);
}

if (two_d_curvature){
nplusz = 0;
nminusz = 0;
}	else {
double denom4 = fabs(pow(Dzplus, 2) + pow((Dxplusz + Dx)/2.0, 2) + pow((Dyplusz + Dy)/2.0, 2));
if (denom4 <= tolerance ){
nplusz = 0;
}	else {
nplusz = Dzplus/sqrt(denom4);
}

double denom5 = fabs(pow(Dzminus, 2) + pow((Dxminusz + Dx)/2.0, 2) + pow((Dyminusz + Dy)/2.0, 2));
if (denom5 <= tolerance ){
nminusz = 0;
}	else {
nminusz = Dzminus/sqrt(denom5);
}

}

k = 0.5*((nplusx - nminusx) + (nplusy - nminusy) + (nplusz - nminusz));

return float(k);

}


pair<int_type_t, int_type_t> UpdateEdgeListDiffStructure(vector<bool*>& class_flag, vector<uint_dist_type*>& dt_xyz,
vector<int8_t*>& contour_age, vector<vector< int_type_t*> >& d_gamma_indices_per_thread,
vector<int_type_t>& count_d_gamma_per_thread, int number_threads, int_type_t xsize, int_type_t ysize,
int_type_t zsize,
bool use_band_size, uint_dist_type band_size_sq, int8_t contour_age_threshold){
vector<int_type_t> aged_out_per(number_threads, 0);

vector<int_type_t> panel_number_per_thread(number_threads, 0);
vector<int_type_t> index_per_panel(number_threads, 0);
vector<int_type_t> number_aged_out_per_thread(number_threads, 0);



int_type_t max_per_panel = xsize*ysize; 
int_type_t rs_index = 0;
int_type_t n_index = 0;
int number_images = dt_xyz.size();
bool n_found = false;
bool test_this_item = 0;
int_type_t number_aged_out = 0;
int_type_t total_d_gamma = 0;
int thread_id = 0;

int_type_t x0, y0, z0;

cout << "Use band size??? " << use_band_size << endl;


#pragma omp parallel for private(thread_id, test_this_item, n_found, rs_index, x0, y0, z0)
for (int_type_t zi = 1; zi < zsize - 1; zi++){
thread_id = omp_get_thread_num();

for (int_type_t xi = 1; xi < xsize-1; xi++){
for (int_type_t yi = 1; yi < ysize -1; yi++){

if (use_band_size){
test_this_item = class_flag[zi][xi*ysize + yi] == false && fabs(dt_xyz[zi][xi*ysize + yi]) <= band_size_sq;
}	else {
test_this_item = class_flag[zi][xi*ysize + yi] == false;
}

if (test_this_item){ 
n_found = false;

for (z0 = zi-1; z0 <= zi + 1 && n_found == false; z0++){
for (x0 = xi - 1; x0 <= xi + 1 && n_found == false; x0++){
for (y0 = yi - 1; y0 <= yi + 1 && n_found == false; y0++){
if (class_flag[z0][x0*ysize + y0] == true){
n_found = true;
}
}
}
}


if (n_found == true){
if (contour_age[zi][xi*ysize + yi] < contour_age_threshold){
rs_index = ReturnIndexFromXYZ(xi, yi, zi, xsize, ysize);
if (index_per_panel[thread_id] == max_per_panel){
#pragma omp critical
{
cout << "Creating new panel for thread " << thread_id << endl;
d_gamma_indices_per_thread[thread_id].push_back(new int_type_t[max_per_panel]);
}

index_per_panel[thread_id] = 0;
panel_number_per_thread[thread_id]++;
}

d_gamma_indices_per_thread[thread_id][panel_number_per_thread[thread_id]][index_per_panel[thread_id]] = rs_index;
index_per_panel[thread_id]++;

}	else {
number_aged_out_per_thread[thread_id]++;
}

}

if (contour_age[zi][xi*ysize + yi] < contour_age_threshold + 2){
contour_age[zi][xi*ysize + yi]++; 
}

}	else {


}
}
}

}


for (int i = 0; i < number_threads; i++){
count_d_gamma_per_thread[i] = (panel_number_per_thread[i])*max_per_panel + index_per_panel[i];
number_aged_out += number_aged_out_per_thread[i];
total_d_gamma += count_d_gamma_per_thread[i];
}

return pair<int_type_t, int_type_t>(total_d_gamma, number_aged_out);

}


void ReturnLimits(int_type_t gx, int_type_t gy, int_type_t gz, int_type_t mx, int_type_t my, int_type_t mz,
int_type_t& minx, int_type_t& miny, int_type_t& minz,
int_type_t& maxx, int_type_t& maxy, int_type_t& maxz){

gx > 1 ? minx = gx - 1 : minx = 0;
gy > 1 ? miny = gy - 1 : miny = 0;
gz > 1 ? minz = gz - 1 : minz = 0;

gx < mx - 1 ? maxx = gx + 1 : maxx = gx;
gy < my - 1 ? maxy = gy + 1 : maxy = gy;
gz < mz - 1 ? maxz = gz + 1 : maxz = gz;

}


void UpdateGridWithNewEdges(vector<vector< int_type_t*> >& d_gamma_indices_per_thread,
vector<int_type_t>& count_d_gamma_per_thread, vector<bool*>& grid, int_type_t xsize, int_type_t ysize,
int_type_t grid_xsize, int_type_t grid_ysize, int_type_t grid_zsize, int_type_t grid_resolution, int_type_t number_threads){
int_type_t xi, yi, zi; 

int_type_t number_panels;
int_type_t index_this_panel;
int_type_t max_per_panel = xsize*ysize;  
int_type_t gx, gy, gz;
int_type_t minx, miny, minz, maxx, maxy, maxz;


#pragma omp parallel for
for (int_type_t z = 0; z < grid_zsize; z++){
for (int_type_t xy = 0; xy < grid_xsize*grid_ysize; xy++){
grid[z][xy] = false;
}

}



maxx = grid_xsize; maxy = grid_ysize; maxz = grid_zsize;

for (int_type_t thread_id= 0; thread_id < number_threads; thread_id++){

number_panels = count_d_gamma_per_thread[thread_id]/max_per_panel;
if (count_d_gamma_per_thread[thread_id] > 0){
for (int_type_t j = 0; j <= number_panels; j++){


if (j == number_panels){
index_this_panel = count_d_gamma_per_thread[thread_id] % max_per_panel;

if (index_this_panel == 0){ 
index_this_panel = max_per_panel;
}
}	else {
index_this_panel = max_per_panel;
}

for (int_type_t k = 0; k < index_this_panel; k++){
ReturnXYZIndicesFromIndex(d_gamma_indices_per_thread[thread_id][j][k], xi, yi, zi, xsize, ysize);

gx = xi/grid_resolution;
gy = yi/grid_resolution;
gz = zi/grid_resolution;


ReturnLimits(gx, gy, gz, grid_xsize, grid_ysize, grid_zsize, minx, miny, minz, maxx, maxy, maxz);

{
for (int_type_t x = minx; x <= maxx; x++){
for (int_type_t y = miny; y <= maxy; y++){
for (int_type_t z = minz; z <= maxz; z++){
grid[z][x*grid_ysize + y] = true;
}
}
}
}
}
}

}
}
}







