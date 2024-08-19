
#include "DistanceTransforms.hpp"


double f(int_type_t x, int_type_t i, double* g){

return pow((double(x) - double(i)), 2) + pow(g[i], 2);
}

double f_with_square(int_type_t x, int_type_t i, double* g){

return pow((double(x) - double(i)), 2) + (g[i]); 
}



double Sep_with_square(int_type_t i, int_type_t u, double* g, double big_number){
if (u == i){
cout << "Error!  Divide by zero " << i << ", " << u << endl;
exit(1);
}

double g_value_u = g[u];
double g_value_i = g[i];

double s = int(double(u*u) - double(i*i)
+ g_value_u - g_value_i)/int(2*(double(u - i)));

if (s < 0){

if (fabs(s) < 0.00001){
s = 0;
}	else {
if (fabs(g_value_u - big_number) < 0.01){
return u;
}	else {
cout << "u, i " << u << ", " << i << endl;
cout << "gu, gi " <<  g_value_u << ", " << g_value_i << endl;
double numer = int(double(u*u) - double(i*i)
+ g_value_u - g_value_i);
double denom = int(2*(double(u - i)));
cout << "number, denom " << numer << ", " << denom << endl;


cout << "Error!  Sep_sq less than zero " << s << endl;
exit(1);
}
}
}
return s;
}


double Sep(int_type_t i, int_type_t u,  double* g, double big_number){
if (u == i){
cout << "Error!  Divide by zero " << i << ", " << u << endl;
exit(1);
}

double g_value_u = g[u];

double g_value_i = g[i];

double s = int(double(u*u) - double(i*i) + pow(g_value_u, 2) - pow(g_value_i, 2))/int(2*(double(u - i)));


if (s < 0){
if (fabs(s) < 0.00001){
s = 0;
}	else {
if ((g_value_u - big_number) < 0.01){
return u;
}	else {

cout << "i, u " << i << ", " << u << endl;
cout << "gi, gu " << g_value_i << ", " << g_value_u << endl;
cout << "269 Error!  Sep less than zero " << s << endl;
exit(1);
}
}
}


return s;
}



void ReturnXYZIndicesFromIndex(int_type_t index, int_type_t& x, int_type_t& y, int_type_t& z, int_type_t xsize, int_type_t ysize){
z = index/(xsize*ysize);
int_type_t interim = index % (xsize*ysize);

x = interim/ysize;
y = interim % ysize;
}

void DistanceTransformSqdParallelTubeXIV(vector<vector< int_type_t*> >& d_gamma_indices_per_thread,
vector<int_type_t>& count_d_gamma_per_thread, vector<uint_dist_type*>& dt_xyz, uint_dist_type big_number,
uint_dist_type band_increment, int_type_t xsize, int_type_t ysize, int_type_t zsize, vector<bool*>& grid, int_type_t grid_xsize,
int_type_t grid_ysize, int_type_t grid_zsize, int_type_t grid_resolution, int_type_t number_threads){



uint_dist_type band_sq = band_increment*band_increment;
double double_big_number = xsize*ysize*zsize/2;
double_big_number = xsize*ysize;
int_type_t n = max(max(xsize, ysize), zsize);
int_type_t xi, yi, zi;

cout << "xsize, ysize, zsize " << xsize << ", " << ysize << ", " << zsize << endl;


#pragma omp parallel for
for (int_type_t z_slice = 0; z_slice < zsize; z_slice++){
for (int_type_t slice_i = 0; slice_i < xsize*ysize; slice_i++){
dt_xyz[z_slice][slice_i] = big_number;
}
}

int_type_t number_panels;
int_type_t index_this_panel;
int_type_t max_per_panel = xsize*ysize;  
int_type_t gx, gy, gz;

#pragma omp parallel for private(number_panels, index_this_panel, xi, yi, zi)
for (int_type_t thread_id= 0; thread_id < number_threads; thread_id++){
if (count_d_gamma_per_thread[thread_id] > 0){
number_panels = count_d_gamma_per_thread[thread_id]/max_per_panel;
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
dt_xyz[zi][xi*ysize + yi] = 0;
}
}
}
}




vector< int_type_t* > s(number_threads, 0);
vector< int_type_t* > t(number_threads, 0);
vector< bool* > occupied_line(number_threads, 0);
vector< int_type_t* > starts(number_threads, 0);
vector< int_type_t* > stops(number_threads, 0);
vector< double* > dt_line(number_threads, 0);
vector< double* > g_xy_line(number_threads, 0);

for (int_type_t i = 0; i < number_threads; i++){
s[i] = new int_type_t[max(xsize, zsize)];
t[i] = new int_type_t[max(xsize, zsize)];
dt_line[i] = new double[max(xsize, zsize)];
g_xy_line[i] = new double[max(xsize, zsize)];
starts[i] = new int_type_t[n + 2];
stops[i] = new int_type_t[n + 2];
occupied_line[i] = new bool[max(xsize, zsize)];
}
int q; int w;


int_type_t thread_id = 0;


cout << "Start scan 0" << endl;

int_type_t local_start, local_stop;


int_type_t limits_y;
int_type_t limits_top_y;
#pragma omp parallel for private (gx, gz,limits_y, limits_top_y )  
for (int_type_t z = 0; z < zsize; z++){
gz = z/grid_resolution;
for (int_type_t x= 0; x < xsize; x++){
gx = x/grid_resolution;

for (int_type_t gy0 = 0; gy0 < grid_ysize; gy0++){
if (grid[gz][gx*grid_ysize + gy0] == true){
if (gy0 == grid_ysize - 1){
limits_y = ysize;
}	else {
limits_y = (gy0 + 1)*grid_resolution + 1;
}

for (int_type_t y= gy0*grid_resolution + 1; y < limits_y; y++){
if (dt_xyz[z][x*ysize + y - 1] < band_increment - 1){
dt_xyz[z][x*ysize + y] = min(uint_dist_type(dt_xyz[z][x*ysize + y - 1] + 1), dt_xyz[z][x*ysize + y]);
}
}
}

}

for (int_type_t gy0 = grid_ysize; gy0 > 0; gy0--){
if (grid[gz][gx*grid_ysize + (gy0 - 1)] == true){
if (gy0 - 1 == 0){
limits_y = 1;
}	else {
limits_y = (gy0 -1)*grid_resolution;
}

if (gy0 == grid_ysize){
limits_top_y = ysize;

}	else {
limits_top_y = gy0*grid_resolution; 
}

for (int_type_t y= limits_top_y; y > limits_y; y--){
if (dt_xyz[z][x*ysize + y - 1] < band_increment - 1){
dt_xyz[z][x*ysize + y - 2] = min(uint_dist_type(dt_xyz[z][x*ysize + y - 1] + 1), dt_xyz[z][x*ysize + y - 2]);
}
}
}
}

}
}

cout << "Start scan 2" << endl;

int current_distance; bool walk_condition;

int_type_t line_counter = 0;


int_type_t j_index;
int_type_t stop_index;


cout << "Start scan 3" << endl;

bool some_found = false;
int_type_t number_starts, number_stops;
double temporary_double;

int_type_t relevant_gs = 0;

int_type_t xstart, ystart, zstart, yend, xend, zend;

for (int_type_t gz = 0; gz < grid_zsize; gz++){
for (int_type_t gy = 0; gy < grid_ysize; gy++){
number_starts = 0; number_stops = 0;

if (grid[gz][0*grid_ysize + gy] == true){
starts[0][number_starts] = 0; number_starts++;
}

for (int_type_t gx = 0; gx < grid_xsize; gx++){

if (grid[gz][gx*grid_ysize + gy] == true){
if (gx  > 0){
if (grid[gz][(gx - 1)*grid_ysize + gy] == false){
starts[0][number_starts] = gx*grid_resolution; number_starts++;
}
}

if (gx < (grid_xsize - 1)){
if (grid[gz][(gx + 1)*grid_ysize + gy] == false){
stops[0][number_stops] = (gx + 1)*grid_resolution - 1; number_stops++;
}
}
}
}


if (grid[gz][(grid_xsize - 1)*grid_ysize + gy] == true){
stops[0][number_stops] = xsize - 1; number_stops++;
}

if (number_starts != number_stops){
cout << "gy, gz, " << gy << ", " << gz << endl;
cout << "Start and stop size not the same " << number_starts << ", " << number_stops << endl;
exit(1);
}

gy == grid_ysize - 1 ? yend = ysize : yend = (gy + 1)*grid_resolution;
gz == grid_zsize - 1 ? zend = zsize : zend = (gz + 1)*grid_resolution;
ystart = (gy)*grid_resolution;
zstart = (gz)*grid_resolution;
if (number_starts > 0){
for (int_type_t z = zstart; z < zend; z++){
#pragma omp parallel for private(q, thread_id, w, local_start, local_stop, temporary_double)
for (int_type_t y = ystart; y < yend; y++){
thread_id = omp_get_thread_num();
int_type_t line_counter = 0;
for (int_type_t ss = 0; ss < number_starts; ss++ ){

local_start = starts[0][ss];
local_stop = stops[0][ss];


line_counter = 0;
for (int_type_t abs_u = local_start; abs_u < local_stop + 1; abs_u++, line_counter++){
if (dt_xyz[z][abs_u*ysize + y] > band_sq){
dt_line[thread_id][line_counter] = double_big_number;
}	else {
dt_line[thread_id][line_counter] = double(dt_xyz[z][abs_u*ysize + y]);
}
}

if (line_counter != local_stop - local_start + 1){
cout << "Error line counter ..." << line_counter << endl;
cout << local_stop - local_start + 1 << endl;
exit(1);
}

q = 0; s[thread_id][0]= 0; t[thread_id][0] = 0;

for (int_type_t u = 1; u < line_counter; u++){

while (q >= 0 && f(t[thread_id][q], s[thread_id][q], dt_line[thread_id]) > f(t[thread_id][q], u, dt_line[thread_id])){
q--;
}

if (q < 0){
q = 0;  s[thread_id][0] = u;
} else {

w = 1 + Sep(s[thread_id][q], u, dt_line[thread_id], double_big_number);


if (w < 0){
cout << "EEEO! w less than zero " << endl;
exit(1);
}
if (w < int(line_counter)){
q++; s[thread_id][q] = u; t[thread_id][q] = w;
}
}
}

for (int_type_t u = line_counter; u > 0; u--){
temporary_double = f(u - 1, s[thread_id][q], dt_line[thread_id]);

if (temporary_double <= band_sq){
dt_xyz[z][(local_start + u - 1)*ysize + y] = uint_dist_type(temporary_double); 
}	else {
dt_xyz[z][(local_start + u - 1)*ysize + y] = big_number; 
}

if ((u-1) == t[thread_id][q]){
q--;
}
}
}

}
}
}
}
}



cout << "End scan 4" << endl;
if (zsize ==1){

}	else {


for (int_type_t gx = 0; gx < grid_xsize; gx++){
for (int_type_t gy = 0; gy < grid_ysize; gy++){
number_starts = 0; number_stops = 0;

if (grid[0][gx*grid_ysize + gy] == true){
starts[0][number_starts] = 0; number_starts++;
}

for (int_type_t gz = 0; gz < grid_zsize; gz++){

if (grid[gz][gx*grid_ysize + gy] == true){
if (gz  > 0){
if (grid[gz - 1][(gx)*grid_ysize + gy] == false){
starts[0][number_starts] = gz*grid_resolution; number_starts++;
}
}

if (gz < (grid_zsize - 1)){
if (grid[gz + 1][(gx)*grid_ysize + gy] == false){
stops[0][number_stops] = (gz + 1)*grid_resolution - 1; number_stops++;
}
}
}

}

if (grid[grid_zsize - 1][gx*grid_ysize + gy] == true){
stops[0][number_stops] = zsize - 1; number_stops++;
}

if (number_starts != number_stops){
cout << "gy, gx, " << gy << ", " << gx << endl;
cout << "Start and stop size not the same " << number_starts << ", " << number_stops << endl;
exit(1);
}




gy == grid_ysize - 1 ? yend = ysize : yend = (gy + 1)*grid_resolution;
gx == grid_xsize - 1 ? xend = xsize : xend = (gx + 1)*grid_resolution;
ystart = (gy)*grid_resolution;
xstart = (gx)*grid_resolution;
if (number_starts > 0){

for (int_type_t x = xstart; x < xend; x++){
#pragma omp parallel for private(q, thread_id, w, local_start, local_stop, temporary_double)
for (int_type_t y = ystart; y < yend; y++){
thread_id = omp_get_thread_num();

int_type_t line_counter = 0;

for (int_type_t ss = 0, sn = number_starts; ss < sn; ss++ ){
local_start = starts[0][ss];
local_stop = stops[0][ss];

line_counter = 0;
for (int_type_t abs_u = local_start; abs_u < local_stop + 1; abs_u++, line_counter++){
if (dt_xyz[abs_u][x*ysize + y] > band_sq){
g_xy_line[thread_id][line_counter] = double_big_number;
}	else {
g_xy_line[thread_id][line_counter]  = double(dt_xyz[abs_u][x*ysize + y]);
}
}

q = 0; s[thread_id][0]= 0; t[thread_id][0] = 0;

for (int_type_t u = 1; u < line_counter; u++){

while (q >= 0 && f_with_square(t[thread_id][q], s[thread_id][q], g_xy_line[thread_id])
> f_with_square(t[thread_id][q], u, g_xy_line[thread_id])){
q--;
}

if (q < 0){
q = 0;  s[thread_id][0] = u;
} else {
w = 1 + Sep_with_square(s[thread_id][q], u, g_xy_line[thread_id], double_big_number);

if (w < 0){
cout << "EEEO! w less than zero " << endl;
exit(1);
}
if (w < int(line_counter)){
q++; s[thread_id][q] = u; t[thread_id][q] = w;
}
}
}

for (int_type_t u = line_counter; u > 0; u--){


temporary_double = f_with_square(u - 1, s[thread_id][q], g_xy_line[thread_id]);
if (temporary_double <= band_sq){
dt_xyz[local_start + u - 1][x*ysize + y] = temporary_double;
}	else {
dt_xyz[local_start + u - 1][x*ysize + y] = big_number;
}

if ((u-1) == t[thread_id][q]){
q--;
}
}
}
}
}
}
}
}
}




cout << "Complete scan " << endl;

for (int_type_t i = 0; i < number_threads; i++){
delete [] s[i];
delete [] t[i];
delete [] dt_line[i];
delete [] g_xy_line[i];
delete [] occupied_line[i];
delete [] starts[i];
delete [] stops[i];
}

}



