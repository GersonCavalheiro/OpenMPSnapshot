


#include "grid_decomposition.h"
#include "common_utils.h"
#include "projection.h"
#include "netcdf_utils.h"
#include "opencv_utils.h"
#include "timer.h"
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <tr1/unordered_map>
#include <sys/time.h>
#include <omp.h>

#define PDLN_DEFAULT_EXPANGDING_RATIO (0.2)
#define PDLN_DEFAULT_EXPANGDING_SCALE (2)

#define PDLN_SPOLAR_MAX_LAT (-19.47)
#define PDLN_NPOLAR_MIN_LAT (19.47)

#define PDLN_SPOLAR_MIN_LAT (-91)
#define PDLN_NPOLAR_MAX_LAT (91)

#define PAT_POLE_STRETCHING_THRESHOLD (7);

#define PDLN_MAX_ITER_COUNT (10)

#define PDLN_TOLERABLE_ERROR (0.0001)

#define PDLN_DECOMPOSE_COMMON_MODE (0)
#define PDLN_DECOMPOSE_SPOLAR_MODE (1)
#define PDLN_DECOMPOSE_NPOLAR_MODE (2)

#define PDLN_NODE_TYPE_COMMON PDLN_DECOMPOSE_COMMON_MODE
#define PDLN_NODE_TYPE_SPOLAR PDLN_DECOMPOSE_SPOLAR_MODE
#define PDLN_NODE_TYPE_NPOLAR PDLN_DECOMPOSE_NPOLAR_MODE

#define PDLN_DOUBLE_INVALID_VALUE ((double)0xDEADBEEFDEADBEEF)

#define PDLN_MAX_NUM_PROCESSING_UNITS 512

#define PDLN_POLAR_WORKLOAD_RATE (0.09)


#define PDLN_SEPARATELY_EXPANDING_COUNT (3)
#define PDLN_MIN_EXPANDING_QUOTA (1000.0)


static inline bool is_in_region(double x, double y, Boundry region);

bool operator == (Triangle_ID_Only t1, Triangle_ID_Only t2)
{
if(t2.id[0] != t1.id[0] && t2.id[0] != t1.id[1] && t2.id[0] != t1.id[2])
return false;
if(t2.id[1] != t1.id[0] && t2.id[1] != t1.id[1] && t2.id[1] != t1.id[2])
return false;
if(t2.id[2] != t1.id[0] && t2.id[2] != t1.id[1] && t2.id[2] != t1.id[2])
return false;
return true;
}


bool Boundry::operator== (const Boundry &boundry) const
{
return min_lat == boundry.min_lat && min_lon == boundry.min_lon && max_lat == boundry.max_lat && max_lon == boundry.max_lon;
}


bool Boundry::operator!= (const Boundry &boundry) const
{
return !(min_lat == boundry.min_lat && min_lon == boundry.min_lon && max_lat == boundry.max_lat && max_lon == boundry.max_lon);
}


bool Boundry::operator<= (const Boundry &boundry) const
{
return min_lat >= boundry.min_lat && min_lon >= boundry.min_lon && max_lat <= boundry.max_lat && max_lon <= boundry.max_lon;
}


Boundry& Boundry::operator* (double ratio)
{
min_lat -= (max_lat - min_lat) * ratio * 0.5;
max_lat += (max_lat - min_lat) * ratio * 0.5;
min_lon -= (max_lon - min_lon) * ratio * 0.5;
max_lon += (max_lon - min_lon) * ratio * 0.5;
return *this;
}


void Boundry::legalize(const Boundry *outer_boundry, bool is_cyclic)
{
min_lat = std::max(min_lat, outer_boundry->min_lat);
max_lat = std::min(max_lat, outer_boundry->max_lat);
if(!is_cyclic) {
min_lon = std::max(min_lon, outer_boundry->min_lon);
max_lon = std::min(max_lon, outer_boundry->max_lon);
}
}


void Boundry::max(const Boundry b)
{
double min_lo = b.min_lon;
double max_lo = b.max_lon;
double min_la = b.min_lat;
double max_la = b.max_lat;
min_lon = std::min(min_lon, min_lo);
max_lon = std::max(max_lon, max_lo);
min_lat = std::min(min_lat, min_la);
max_lat = std::max(max_lat, max_la);
}


void Boundry::max(double min_lo, double max_lo, double min_la, double max_la)
{
min_lon = std::min(min_lon, min_lo);
max_lon = std::max(max_lon, max_lo);
min_lat = std::min(min_lat, min_la);
max_lat = std::max(max_lat, max_la);
}

Search_tree_node::Search_tree_node(Search_tree_node *p, double *coord_value[2], int *global_index, bool *mask, int num_points, Boundry boundry, int type)
: parent(p)
, node_type(type)
, region_id(-1)
, is_leaf(false)
, fast_triangulate(false)
, kernel_boundry(NULL)
, expand_boundry(NULL)
, project_boundry(NULL)
, real_boundry(NULL)
, len_expand_coord_buf(0)
, num_kernel_points(num_points)
, num_expand_points(0)
, num_projected_points(0)
, midline(Midline{-1, -361.0})
, group_intervals(NULL)
, triangulation(NULL)
, bind_with(0) 
, is_bind(false)
, polars_local_index(NULL)
, shifted_polar_lat(0)
, virtual_point_local_index(-1)
{
PDASSERT(num_points >= 0);
children[0] = NULL;
children[1] = NULL;
children[2] = NULL;
projected_coord[0] = NULL;
projected_coord[1] = NULL;

kernel_boundry  = new Boundry();
expand_boundry  = new Boundry();
*kernel_boundry = boundry;
*expand_boundry = boundry;

kernel_coord[0] = coord_value[0];
kernel_coord[1] = coord_value[1];
kernel_index    = global_index;
kernel_mask     = mask;

expand_coord[0] = NULL;
expand_coord[1] = NULL;
expand_index    = NULL;
expand_mask     = NULL;

if(type == PDLN_NODE_TYPE_COMMON) {
center[PDLN_LON] = (boundry.min_lon + boundry.max_lon) * 0.5;
center[PDLN_LAT] = (boundry.min_lat + boundry.max_lat) * 0.5;
}
else if(type == PDLN_NODE_TYPE_SPOLAR) {
center[PDLN_LON] = 0.0;
center[PDLN_LAT] = -90.0;
}
else if(type == PDLN_NODE_TYPE_NPOLAR) {
center[PDLN_LON] = 0.0;
center[PDLN_LAT] = 90.0;
}

non_monotonic = kernel_boundry->min_lon > kernel_boundry->max_lon;

expanding_scale[0] = PDLN_DEFAULT_EXPANGDING_SCALE;
expanding_scale[1] = PDLN_DEFAULT_EXPANGDING_SCALE;
expanding_scale[2] = PDLN_DEFAULT_EXPANGDING_SCALE;
expanding_scale[3] = PDLN_DEFAULT_EXPANGDING_SCALE;
num_neighbors_on_boundry[0] = 0;
num_neighbors_on_boundry[1] = 0;
num_neighbors_on_boundry[2] = 0;
num_neighbors_on_boundry[3] = 0;
edge_expanding_count[0] = 0;
edge_expanding_count[1] = 0;
edge_expanding_count[2] = 0;
edge_expanding_count[3] = 0;
}


Search_tree_node::~Search_tree_node()
{
delete kernel_boundry;
delete expand_boundry;
delete real_boundry;
delete project_boundry;
for(int i = 0; i < 3; i ++)
delete children[i];
delete triangulation;

delete[] projected_coord[0];
delete[] projected_coord[1];

delete polars_local_index;
}


const double fixed_view_points[2][4] = {{45, 135, 225, 315},
{0, 0, 0, 0}};
void Search_tree_node::fix_view_point()
{
unsigned current_index = 0;
double current_distence;

while(center[PDLN_LON] <    0) center[PDLN_LON] += 360;
while(center[PDLN_LON] >= 360) center[PDLN_LON] -= 360;
current_distence = (center[PDLN_LON] - fixed_view_points[PDLN_LON][0]) * (center[PDLN_LON] - fixed_view_points[PDLN_LON][0]) +
(center[PDLN_LAT] - fixed_view_points[PDLN_LAT][0]) * (center[PDLN_LAT] - fixed_view_points[PDLN_LAT][0]); 
for(unsigned i = 1; i < sizeof(fixed_view_points)/sizeof(double)/2; i++) {
double dist = (center[PDLN_LON] - fixed_view_points[PDLN_LON][i]) * (center[PDLN_LON] - fixed_view_points[PDLN_LON][i]) +
(center[PDLN_LAT] - fixed_view_points[PDLN_LAT][i]) * (center[PDLN_LAT] - fixed_view_points[PDLN_LAT][i]);
if (dist < current_distence) {
current_distence = dist;
current_index = i;
}
}
center[PDLN_LON] = fixed_view_points[PDLN_LON][current_index];
center[PDLN_LAT] = fixed_view_points[PDLN_LAT][current_index];
}


void Search_tree_node::update_region_ids(int start, int end)
{
ids_start = start;
ids_end = end;
}


inline void calculate_circle_center(double x[3], double y[3], double *center_x, double *center_y)
{
PAT_REAL mid_x[2], mid_y[2];
PAT_REAL k[2];

mid_x[0] = (x[0] + x[1]) * 0.5;
mid_y[0] = (y[0] + y[1]) * 0.5;
mid_x[1] = (x[0] + x[2]) * 0.5;
mid_y[1] = (y[0] + y[2]) * 0.5;
k[0] = - (x[1] - x[0]) / (y[1] - y[0]);
k[1] = - (x[2] - x[0]) / (y[2] - y[0]);

*center_x = (mid_y[1] - mid_y[0] - k[1]*mid_x[1] + k[0]*mid_x[0]) / (k[0] - k[1]);
*center_y = mid_y[0] + k[0]*(mid_y[1] - mid_y[0] - k[1]*mid_x[1] + k[1]*mid_x[0]) / (k[0] - k[1]);
}


void Search_tree_node::calculate_latitude_circle_projection(double lat, Point* circle_center, double* radius)
{
double x[3], y[3];

calculate_stereographic_projection(0,   lat, center[PDLN_LON], center[PDLN_LAT], x[0], y[0]);
calculate_stereographic_projection(90,  lat, center[PDLN_LON], center[PDLN_LAT], x[1], y[1]);
calculate_stereographic_projection(180, lat, center[PDLN_LON], center[PDLN_LAT], x[2], y[2]);

if (float_eq(x[0], x[1]) && float_eq(x[1], x[2]) && float_eq(y[0], y[1]) && float_eq(y[1], y[2])) {
circle_center->x = x[0];
circle_center->y = y[0];
*radius = 0;
} else {
calculate_circle_center(x, y, &circle_center->x, &circle_center->y);
*radius = sqrt((x[2]-circle_center->x)*(x[2]-circle_center->x)+(y[2]-circle_center->y)*(y[2]-circle_center->y));
}
}


void Search_tree_node::calculate_cyclic_boundary_projection(unsigned mode, Point* head, Point* tail)
{
double polar_lat = 0;
double equat_lat = 0;

if (mode == PDLN_NODE_TYPE_NPOLAR) {
equat_lat = -center[PDLN_LAT]+20;
polar_lat = real_boundry->max_lat+0.1;
} else if (mode == PDLN_NODE_TYPE_SPOLAR) {
equat_lat = -center[PDLN_LAT]-20;
polar_lat = real_boundry->min_lat;
}

double lon = (real_boundry->max_lon + real_boundry->min_lon + 360.0) * 0.5;
double head_lon, head_lat, tail_lon, tail_lat;

calculate_stereographic_projection(lon, polar_lat, center[PDLN_LON], center[PDLN_LAT], head_lon, head_lat);
calculate_stereographic_projection(lon, equat_lat, center[PDLN_LON], center[PDLN_LAT], tail_lon, tail_lat);

head->x = head_lon;
head->y = head_lat;
tail->x = tail_lon;
tail->y = tail_lat;
}


extern double global_p_lon[4];
extern double global_p_lat[4];
#define PDLN_INSERT_VIRTUAL_POINT (true)
#define PDLN_REMOVE_UNNECESSARY_TRIANGLES (true)
void Search_tree_node::generate_local_triangulation(bool is_cyclic, int vpoint_begin, int vpoint_num, bool is_fine_grid)
{
log(LOG_DEBUG, "%d region - %d kernel points, %d expanded points\n", region_id, num_kernel_points, num_expand_points);
timeval start, end;
gettimeofday(&start, NULL);

if (triangulation && fast_triangulate)
return;


double* ori_lon  = new double[num_kernel_points + num_expand_points];
double* ori_lat  = new double[num_kernel_points + num_expand_points];
int*    ori_idx  = new int[num_kernel_points + num_expand_points];
bool*   ori_mask = NULL;

if (kernel_mask)
ori_mask = new bool[num_kernel_points + num_expand_points];

memcpy(ori_lon, kernel_coord[PDLN_LON], sizeof(double)*num_kernel_points);
memcpy(ori_lon+num_kernel_points, expand_coord[PDLN_LON], sizeof(double)*num_expand_points);
memcpy(ori_lat, kernel_coord[PDLN_LAT], sizeof(double)*num_kernel_points);
memcpy(ori_lat+num_kernel_points, expand_coord[PDLN_LAT], sizeof(double)*num_expand_points);
memcpy(ori_idx, kernel_index, sizeof(int)*num_kernel_points);
memcpy(ori_idx+num_kernel_points, expand_index, sizeof(int)*num_expand_points);
if (kernel_mask) {
memcpy(ori_mask, kernel_mask, sizeof(bool)*num_kernel_points);
memcpy(ori_mask+num_kernel_points, expand_mask, sizeof(bool)*num_expand_points);
}


#ifdef DEBUG
if (project_boundry)
report_redundent_points(projected_coord[PDLN_LON], projected_coord[PDLN_LAT], ori_idx, num_kernel_points+num_expand_points);
#endif



if (triangulation == NULL) {

triangulation = new Delaunay_Voronoi();
if (node_type == PDLN_NODE_TYPE_COMMON)
triangulation->set_original_center_lon(center[PDLN_LON]);
else
triangulation->set_original_center_lon(180.);

if (project_boundry == NULL && !is_cyclic)
triangulation->set_regional(true);

if (fast_triangulate) {
triangulation->set_origin_coord(ori_lon, ori_lat, num_kernel_points + num_expand_points);
if (triangulation->try_fast_triangulate(expand_boundry->min_lon, expand_boundry->max_lon,
expand_boundry->min_lat, expand_boundry->max_lat)) {
triangulation->set_checksum_bound(kernel_boundry->min_lon, kernel_boundry->max_lon, kernel_boundry->min_lat, kernel_boundry->max_lat, 0);
triangulation->make_bounding_triangle_pack();
return;
} else {
fast_triangulate = false;
}
}

if (project_boundry) {
triangulation->add_points(projected_coord[PDLN_LON], projected_coord[PDLN_LAT], ori_mask, num_kernel_points+num_expand_points);
triangulation->set_origin_coord(ori_lon, ori_lat, num_kernel_points + num_expand_points);
triangulation->set_checksum_bound(kernel_boundry->min_lon, kernel_boundry->max_lon, kernel_boundry->min_lat, kernel_boundry->max_lat, 0);
if (node_type != PDLN_NODE_TYPE_COMMON)
triangulation->set_polar_mode(true);
} else {
triangulation->add_points(ori_lon, ori_lat, ori_mask, num_kernel_points+num_expand_points);
triangulation->set_checksum_bound(kernel_boundry->min_lon, kernel_boundry->max_lon, kernel_boundry->min_lat, kernel_boundry->max_lat, 0);
}

num_old_points = num_expand_points;
} else {

if (project_boundry) {
triangulation->add_points(projected_coord[PDLN_LON]+num_kernel_points+num_old_points,
projected_coord[PDLN_LAT]+num_kernel_points+num_old_points,
ori_mask ? ori_mask+num_kernel_points+num_old_points : NULL,
num_expand_points-num_old_points);
triangulation->set_origin_coord(ori_lon, ori_lat, num_kernel_points + num_expand_points);
} else {
triangulation->add_points(expand_coord[PDLN_LON]+num_old_points,
expand_coord[PDLN_LAT]+num_old_points,
ori_mask ? ori_mask+num_kernel_points+num_old_points : NULL,
num_expand_points-num_old_points);
}

num_old_points = num_expand_points;
}

triangulation->map_global_index(ori_idx);


triangulation->triangulate();


if (project_boundry == NULL) {
}


if (project_boundry && node_type == PDLN_NODE_TYPE_COMMON) {
if (expand_boundry->max_lon - expand_boundry->min_lon > 90) {
double radius;
Point  circle_center;
Point  boundary_head, boundary_tail;

calculate_real_boundary();

if(PDLN_REMOVE_UNNECESSARY_TRIANGLES && real_boundry->min_lat < 0) {
calculate_latitude_circle_projection(real_boundry->min_lat, &circle_center, &radius);
if(radius < 100) {
triangulation->set_avoiding_circle(0, circle_center, radius);
calculate_cyclic_boundary_projection(PDLN_NODE_TYPE_SPOLAR, &boundary_head, &boundary_tail);
triangulation->set_avoiding_line(0, boundary_head, boundary_tail);
}
}

if(PDLN_REMOVE_UNNECESSARY_TRIANGLES && real_boundry->max_lat > 0) {
calculate_latitude_circle_projection(real_boundry->max_lat, &circle_center, &radius);
if(radius < 100) {
triangulation->set_avoiding_circle(1, circle_center, radius);
calculate_cyclic_boundary_projection(PDLN_NODE_TYPE_NPOLAR, &boundary_head, &boundary_tail);
triangulation->set_avoiding_line(1, boundary_head, boundary_tail);
}
}
}
}

if(vpoint_num > 0) {
log(LOG_DEBUG, "Remove virtual %d points, begin with %d\n", vpoint_num, vpoint_begin, vpoint_num);
triangulation->remove_triangles_containing_vertexs(vpoint_begin, vpoint_num);
}

triangulation->make_bounding_triangle_pack();



gettimeofday(&end, NULL);


}


void Search_tree_node::reset_polars(double* lat_buf)
{
#ifdef DEBUG
PDASSERT(node_type != PDLN_NODE_TYPE_COMMON);
#endif
double reset_lat_value = node_type == PDLN_NODE_TYPE_NPOLAR ? 90 : -90;
for(unsigned i = 0; i < polars_local_index->size(); i++)
lat_buf[(*polars_local_index)[i]] = reset_lat_value;
}


void Search_tree_node::sort_by_line(Midline* midline, int* left_num, int* rite_num)
{
if(non_monotonic && midline->type == PDLN_LON)
PDASSERT(false);

sort_by_line_internal(kernel_coord, kernel_index, kernel_mask, midline, 0, num_kernel_points, left_num, rite_num);
}


static inline void swap(double& a, double& b)
{
double tmp = a;
a = b;
b = tmp;
}


static inline void swap(int& a, int& b)
{
int tmp = a;
a = b;
b = tmp;
}


void Search_tree_node::sort_by_line_internal(double* coord[2], int* index, bool* mask, Midline* midline, int start, int num, int* left_num, int* rite_num)
{
int    i, j;
int    type_curt = midline->type;
int    type_opst = (midline->type+1)%2;
double value = midline->value;

PDASSERT(num > 0);

for(i = start, j = start + num - 1; i <= j;) {
if(coord[type_curt][i] < value) {
i++;
} else {
std::swap(coord[type_curt][i], coord[type_curt][j]);
std::swap(coord[type_opst][i], coord[type_opst][j]);
std::swap(index[i], index[j]);
if (mask)
std::swap(mask[i], mask[j]);
j--;
}
while (coord[type_curt][j] >= value && i <= j)
j--;
}

if (left_num)
*left_num = j + 1 - start;
if (rite_num)
*rite_num = start + num - j - 1;
}


void Search_tree_node::divide_at_fix_line(Midline midline, double *c_points_coord[4], int *c_points_idx[2], bool *c_points_mask[2], int c_num_points[2])
{
sort_by_line(&midline, &c_num_points[0], &c_num_points[1]);

c_points_coord[PDLN_LON] = kernel_coord[PDLN_LON];
c_points_coord[PDLN_LAT] = kernel_coord[PDLN_LAT];
c_points_coord[PDLN_LON+2] = &kernel_coord[PDLN_LON][c_num_points[0]];
c_points_coord[PDLN_LAT+2] = &kernel_coord[PDLN_LAT][c_num_points[0]];
c_points_idx[0] = kernel_index;
c_points_idx[1] = &kernel_index[c_num_points[0]];
if (kernel_mask) {
c_points_mask[0] = kernel_mask;
c_points_mask[1] = &kernel_mask[c_num_points[0]];
} else {
c_points_mask[0] = NULL;
c_points_mask[1] = NULL;
}
}


void Search_tree_node::decompose_by_processing_units_number(double *workloads, double *c_points_coord[4], int *c_points_idx[2], 
bool *c_points_mask[2], int c_num_points[2], Boundry c_boundry[2],
int c_ids_start[2], int c_ids_end[2], int mode, int *c_intervals[2],
int c_num_intervals[2], int min_points)
{
PDASSERT(ids_size() > 1);


if(mode == PDLN_DECOMPOSE_COMMON_MODE) {
PDASSERT(c_intervals);
PDASSERT(c_num_intervals);
int mid_off = 0; 
if(num_groups == 1) {
mid_off = ids_size()/2;
c_intervals[0] = c_intervals[1] = NULL;
c_num_intervals[0] = c_num_intervals[1] = 1;
} else {
for(int i = 0; i < num_groups/2; i++)
mid_off += group_intervals[i];
c_intervals[0] = group_intervals;
c_intervals[1] = group_intervals+num_groups/2;
c_num_intervals[0] = num_groups/2;
c_num_intervals[1] = num_groups - c_num_intervals[0];
}

c_ids_start[0] = ids_start;
c_ids_end[0]   = ids_start + mid_off;
c_ids_start[1] = ids_start + mid_off;
c_ids_end[1]   = ids_end;
} else if(mode == PDLN_DECOMPOSE_SPOLAR_MODE) {
c_ids_start[0] = ids_start;
c_ids_end[0]   = ids_start + 1;
c_ids_start[1] = ids_start + 1;
c_ids_end[1]   = ids_end;
} else if(mode == PDLN_DECOMPOSE_NPOLAR_MODE) {
c_ids_start[0] = ids_start;
c_ids_end[0]   = ids_end - 1;
c_ids_start[1] = ids_end - 1;
c_ids_end[1]   = ids_end;
} else
PDASSERT(false);

if (kernel_boundry->min_lon == kernel_boundry->max_lon || kernel_boundry->min_lat == kernel_boundry->max_lat) {
c_num_points[0] = c_num_points[1] = 0;
c_boundry[0] = c_boundry[1] = *kernel_boundry;
return;
}

double length[2], boundry_values[4], c_total_workload[2];
Midline midline;

boundry_values[PDLN_LON] = kernel_boundry->min_lon;
boundry_values[PDLN_LAT] = kernel_boundry->min_lat;
boundry_values[PDLN_LON+2] = kernel_boundry->max_lon;
boundry_values[PDLN_LAT+2] = kernel_boundry->max_lat;

if(non_monotonic) {
PDASSERT(false);
boundry_values[PDLN_LON] -= 360.0;
}

PDASSERT(boundry_values[PDLN_LON] != boundry_values[PDLN_LON+2]);
PDASSERT(boundry_values[PDLN_LAT] < boundry_values[PDLN_LAT+2]);
length[0] = boundry_values[PDLN_LON+2] - boundry_values[PDLN_LON];
length[1] = boundry_values[PDLN_LAT+2] - boundry_values[PDLN_LAT];
PDASSERT(length[0] >= 0 || length[1] >= 0);
PDASSERT(length[0] <= (360.0+PDLN_HIGH_BOUNDRY_SHIFTING) && length[0] >= 0.0 && length[1] <= (180.0+PDLN_HIGH_BOUNDRY_SHIFTING) && length[1] >= 0.0);

if(mode == PDLN_DECOMPOSE_SPOLAR_MODE || mode == PDLN_DECOMPOSE_NPOLAR_MODE)
midline.type = PDLN_LAT;
else if(length[1] > length[0])
midline.type = PDLN_LAT;
else
midline.type = PDLN_LON;

#ifdef DEBUG
PDASSERT(c_ids_start[0] >= 0);
PDASSERT(c_ids_start[1] >= 0);
PDASSERT(c_ids_end[0] >= 0);
PDASSERT(c_ids_end[1] >= 0);
PDASSERT(c_ids_end[0] - c_ids_start[0] + c_ids_end[1] - c_ids_start[1] == ids_size());
#endif

int i;
if(ids_size() > 1) {
for(i = c_ids_start[0], c_total_workload[0] = 0.0; i < c_ids_end[0]; i++)
c_total_workload[0] += workloads[i];
for(i = c_ids_start[1], c_total_workload[1] = 0.0; i < c_ids_end[1]; i++)
c_total_workload[1] += workloads[i];

c_num_points[0] = c_num_points[1] = 0;
reorganize_kernel_points(c_total_workload[0], c_total_workload[1], boundry_values[midline.type], boundry_values[midline.type+2],
0, num_kernel_points, &midline, c_num_points, min_points);
PDASSERT(c_num_points[0] + c_num_points[1] == num_kernel_points);
}
else
midline.value = boundry_values[2+midline.type];

c_points_coord[PDLN_LON] = kernel_coord[PDLN_LON];
c_points_coord[PDLN_LAT] = kernel_coord[PDLN_LAT];
c_points_coord[PDLN_LON+2] = &kernel_coord[PDLN_LON][c_num_points[0]];
c_points_coord[PDLN_LAT+2] = &kernel_coord[PDLN_LAT][c_num_points[0]];
c_points_idx[0] = kernel_index;
c_points_idx[1] = &kernel_index[c_num_points[0]];
if (kernel_mask) {
c_points_mask[0] = kernel_mask;
c_points_mask[1] = &kernel_mask[c_num_points[0]];
} else {
c_points_mask[0] = NULL;
c_points_mask[1] = NULL;
}

if(midline.type == PDLN_LON) {
c_boundry[0].min_lat = c_boundry[1].min_lat = kernel_boundry->min_lat;
c_boundry[0].max_lat = c_boundry[1].max_lat = kernel_boundry->max_lat;
c_boundry[0].min_lon = kernel_boundry->min_lon;
c_boundry[0].max_lon = c_boundry[1].min_lon = midline.value;
c_boundry[1].max_lon = kernel_boundry->max_lon;
}
else if(midline.type == PDLN_LAT) {
c_boundry[0].min_lon = c_boundry[1].min_lon = kernel_boundry->min_lon;
c_boundry[0].max_lon = c_boundry[1].max_lon = kernel_boundry->max_lon;
c_boundry[0].min_lat = kernel_boundry->min_lat;
c_boundry[0].max_lat = c_boundry[1].min_lat = midline.value;
c_boundry[1].max_lat = kernel_boundry->max_lat;
}
else
PDASSERT(false);
}


void Search_tree_node::reorganize_kernel_points(double left_expt, double rite_expt, double left_bound, double rite_bound, 
int offset, int num_points, Midline* midline, int c_num_points[2],
int min_points) {
log(LOG_DEBUG_V, "divide points: ========== A new round ========== \n");
int ret = divide_points(kernel_coord, kernel_index, kernel_mask, left_expt, rite_expt,
left_bound, rite_bound, offset, num_points, 0, midline,
c_num_points, 0, 0, PDLN_DOUBLE_INVALID_VALUE, min_points);

PDASSERT(midline->value != PDLN_DOUBLE_INVALID_VALUE);
if (ret)
sort_by_line_internal(kernel_coord, kernel_index, kernel_mask, midline, offset, num_points, &c_num_points[0], &c_num_points[1]);
}





double value;
double left_solid;
double rite_solid;


int Search_tree_node::divide_points(double *coord[2], int *index, bool *mask, double left_expt, double rite_expt, double left_bound, double rite_bound,
int offset, int num_points, int count, Midline* midline, int c_num_points[2],
double left_solid, double rite_solid, double best_value, int min_points) {
PDASSERT(num_points >= 0);

log(LOG_DEBUG_V, "divide points: l %lf, r %lf\n", left_expt, rite_expt);
if (float_eq(rite_expt, 0)) {
assert(false);
midline->value = rite_bound;
c_num_points[0] = num_points;
c_num_points[1] = 0;
return 0;
}

midline->value = left_bound + (rite_bound - left_bound) * left_expt / (left_expt + rite_expt);
PDASSERT(midline->value >= left_bound);
PDASSERT(midline->value <= rite_bound);

int tmp_num[2];
sort_by_line_internal(coord, index, mask, midline, offset, num_points, &tmp_num[0], &tmp_num[1]);
log(LOG_DEBUG_V, "divide points: midline %lf. %d vs %d\n", midline->value, tmp_num[0], tmp_num[1]);
if (tmp_num[0] > 0 && tmp_num[1] > 0 && left_solid + tmp_num[0] >= min_points && rite_solid + tmp_num[1] >= min_points)
best_value = midline->value;

log(LOG_DEBUG_V, "divide points: best_value %lf\n", best_value);

PDASSERT(tmp_num[0] >= 0);
PDASSERT(tmp_num[0] <= num_points);
PDASSERT(tmp_num[1] >= 0);
PDASSERT(tmp_num[1] <= num_points);

if (count > 20 || (left_expt > rite_expt ?
fabs(left_expt/rite_expt - (double)tmp_num[0]/tmp_num[1]) < 0.1 :
fabs(rite_expt/left_expt - (double)tmp_num[1]/tmp_num[0]) < 0.1) ) {

c_num_points[0] += tmp_num[0];
c_num_points[1] += tmp_num[1];

if (best_value == midline->value || best_value == PDLN_DOUBLE_INVALID_VALUE) {
return 0;
} else {
midline->value = best_value;
return 1;
}
}

if (tmp_num[0] > left_expt) {

c_num_points[1] += tmp_num[1];
PDASSERT(num_points-tmp_num[1] >= 0);
return divide_points(coord, index, mask, left_expt, rite_expt-tmp_num[1], left_bound, midline->value, offset, num_points-tmp_num[1],
count+1, midline, c_num_points, left_solid, rite_solid + tmp_num[1], best_value, min_points);
} else {

c_num_points[0] += tmp_num[0];
PDASSERT(tmp_num[1] >= 0);
return divide_points(coord, index, mask, left_expt-tmp_num[0], rite_expt, midline->value, rite_bound, offset+tmp_num[0], tmp_num[1],
count+1, midline, c_num_points, left_solid + tmp_num[0], rite_solid, best_value, min_points);
}
}


void Search_tree_node::add_expand_points(double *lon_value, double *lat_value, int *global_idx, bool *mask, int num_points)
{
if (num_points)
return;

double *coord_value[2];
coord_value[PDLN_LON] = lon_value;
coord_value[PDLN_LAT] = lat_value;
add_expand_points(coord_value, global_idx, mask, num_points);
}


void Search_tree_node::add_expand_points(double *coord_value[2], int *global_idx, bool *mask, int num_points)
{
double* tmp_coord[2];
int*    tmp_index;
bool*   tmp_mask;

if (num_expand_points + num_points > len_expand_coord_buf) {
len_expand_coord_buf = num_expand_points + num_points * 4 * PDLN_EXPECTED_EXPANDING_TIMES;

tmp_coord[0] = new double[len_expand_coord_buf];
tmp_coord[1] = new double[len_expand_coord_buf];
tmp_index    = new int[len_expand_coord_buf];
if (kernel_mask)
tmp_mask = new bool[len_expand_coord_buf];

memcpy(tmp_coord[0], expand_coord[0], sizeof(double) * num_expand_points);
memcpy(tmp_coord[1], expand_coord[1], sizeof(double) * num_expand_points);
memcpy(tmp_index,    expand_index,    sizeof(int)    * num_expand_points);
if (kernel_mask)
memcpy(tmp_mask, expand_mask,     sizeof(bool)   * num_expand_points);

delete[] expand_coord[0];
delete[] expand_coord[1];
delete[] expand_index;
if (kernel_mask)
delete[] expand_mask;

expand_coord[0] = tmp_coord[0];
expand_coord[1] = tmp_coord[1];
expand_index = tmp_index;
if (kernel_mask)
expand_mask = tmp_mask;

if (projected_coord[0] != NULL) {
tmp_coord[0] = new double[num_kernel_points + len_expand_coord_buf];
tmp_coord[1] = new double[num_kernel_points + len_expand_coord_buf];

memcpy(tmp_coord[0], projected_coord[0], sizeof(double) * (num_kernel_points + num_expand_points));
memcpy(tmp_coord[1], projected_coord[1], sizeof(double) * (num_kernel_points + num_expand_points));

delete[] projected_coord[0];
delete[] projected_coord[1];

projected_coord[0] = tmp_coord[0];
projected_coord[1] = tmp_coord[1];
}
}

memcpy(expand_coord[0] + num_expand_points, coord_value[0], sizeof(double) * num_points);
memcpy(expand_coord[1] + num_expand_points, coord_value[1], sizeof(double) * num_points);
memcpy(expand_index    + num_expand_points, global_idx,     sizeof(int)    * num_points);
if (kernel_mask)
memcpy(expand_mask + num_expand_points, mask,           sizeof(bool)   * num_points);

fix_expand_boundry(num_expand_points, num_points);

num_expand_points += num_points;

#ifdef DEBUG
PDASSERT(num_expand_points >= num_points);
#endif
}


void Search_tree_node::calculate_real_boundary()
{
Boundry boundry;
boundry.min_lat = 1e10;
boundry.max_lat = -1e10;
boundry.min_lon = 1e10;
boundry.max_lon = -1e10;
for(int i = 0; i < num_kernel_points; i++) {
if(kernel_coord[PDLN_LON][i] < boundry.min_lon) boundry.min_lon = kernel_coord[PDLN_LON][i];
if(kernel_coord[PDLN_LON][i] > boundry.max_lon) boundry.max_lon = kernel_coord[PDLN_LON][i];
if(kernel_coord[PDLN_LAT][i] < boundry.min_lat) boundry.min_lat = kernel_coord[PDLN_LAT][i];
if(kernel_coord[PDLN_LAT][i] > boundry.max_lat) boundry.max_lat = kernel_coord[PDLN_LAT][i];
}

for(int i = 0; i < num_expand_points; i++) {
if(expand_coord[PDLN_LON][i] < boundry.min_lon) boundry.min_lon = expand_coord[PDLN_LON][i];
if(expand_coord[PDLN_LON][i] > boundry.max_lon) boundry.max_lon = expand_coord[PDLN_LON][i];
if(expand_coord[PDLN_LAT][i] < boundry.min_lat) boundry.min_lat = expand_coord[PDLN_LAT][i];
if(expand_coord[PDLN_LAT][i] > boundry.max_lat) boundry.max_lat = expand_coord[PDLN_LAT][i];
}


if(real_boundry == NULL)
real_boundry = new Boundry();
*real_boundry = boundry;
}


void Boundry::move_close(double *coord[2], int offset, int count)
{
double min_lon = 1e10;
double max_lon = -1e10;
double min_lat = 1e10;
double max_lat = -1e10;

for(int i = offset; i < offset + count; i++) {
if (min_lon > coord[PDLN_LON][i]) min_lon = coord[PDLN_LON][i];
if (max_lon < coord[PDLN_LON][i]) max_lon = coord[PDLN_LON][i];
if (min_lat > coord[PDLN_LAT][i]) min_lat = coord[PDLN_LAT][i];
if (max_lat < coord[PDLN_LAT][i]) max_lat = coord[PDLN_LAT][i];
}
}


void Search_tree_node::fix_expand_boundry(int index, int count)
{
double min_lon = 1e10;
double max_lon = -1e10;
double min_lat = 1e10;
double max_lat = -1e10;

for(int i = index; i < index + count; i++) {
if (min_lon > expand_coord[PDLN_LON][i]) min_lon = expand_coord[PDLN_LON][i];
if (max_lon < expand_coord[PDLN_LON][i]) max_lon = expand_coord[PDLN_LON][i];
if (min_lat > expand_coord[PDLN_LAT][i]) min_lat = expand_coord[PDLN_LAT][i];
if (max_lat < expand_coord[PDLN_LAT][i]) max_lat = expand_coord[PDLN_LAT][i];
}
if (node_type == PDLN_DECOMPOSE_COMMON_MODE)
expand_boundry->max(min_lon, max_lon + PDLN_HIGH_BOUNDRY_SHIFTING, min_lat, max_lat + PDLN_HIGH_BOUNDRY_SHIFTING);
else if (node_type == PDLN_DECOMPOSE_SPOLAR_MODE)
expand_boundry->max_lat = std::max(expand_boundry->max_lat, max_lat + PDLN_HIGH_BOUNDRY_SHIFTING);
else if (node_type == PDLN_DECOMPOSE_NPOLAR_MODE)
expand_boundry->min_lat = std::min(expand_boundry->min_lat, min_lat);
}


bool operator == (pair<Search_tree_node*, bool> p1, Search_tree_node* p2)
{
return p1.first == p2;
}


void Search_tree_node::add_neighbors(vector<Search_tree_node*> ns)
{
for(unsigned int i = 0; i < ns.size(); i++) {
if(region_id == ns[i]->region_id)
continue;

if(find(neighbors.begin(), neighbors.end(), ns[i]) == neighbors.end())
neighbors.push_back(pair<Search_tree_node*, bool>(ns[i], false));
}
}


static inline void lonlat2xyz(double lon, double lat, PAT_REAL *x, PAT_REAL *y, PAT_REAL *z)
{
*x = cosl(DEGREE_TO_RADIAN(lat)) * sinl(DEGREE_TO_RADIAN(lon));
*y = sinl(DEGREE_TO_RADIAN(lat));
*z = cosl(DEGREE_TO_RADIAN(lat)) * cosl(DEGREE_TO_RADIAN(lon));
}


static inline void normalize_vector(PAT_REAL *x, PAT_REAL *y, PAT_REAL *z)
{
PAT_REAL length = std::sqrt(*x * *x + *y * *y + *z * *z);
*x /= length;
*y /= length;
*z /= length;
}


using std::fabs;
using std::min;
static inline void calculate_unit_vectors(double lon_tan, double lat_tan,
PAT_REAL *v1_x, PAT_REAL *v1_y, PAT_REAL *v1_z,
PAT_REAL *v2_x, PAT_REAL *v2_y, PAT_REAL *v2_z)
{
PAT_REAL t_x, t_y, t_z;

lonlat2xyz(lon_tan, lat_tan, &t_x, &t_y, &t_z);

PAT_REAL min_dir = min(fabs(t_x),min(fabs(t_y),fabs(t_z)));
PAT_REAL axis_x, axis_y, axis_z;

if (min_dir == fabs(t_x)) {
axis_x = 1.0;
axis_y = 0.0;
axis_z = 0.0;
} else if (min_dir == fabs(t_y)) {
axis_x = 0.0;
axis_y = 1.0;
axis_z = 0.0;
} else if (min_dir == fabs(t_z)) {
axis_x = 0.0;
axis_y = 0.0;
axis_z = 1.0;
} else {
assert(false);
}

*v1_x = t_y * axis_z - t_z * axis_y;
*v1_y = t_z * axis_x - t_x * axis_z;
*v1_z = t_x * axis_y - t_y * axis_x;

normalize_vector(v1_x, v1_y, v1_z);

*v2_x = t_y * *v1_z - t_z * *v1_y;
*v2_y = t_z * *v1_x - t_x * *v1_z;
*v2_z = t_x * *v1_y - t_y * *v1_x;

normalize_vector(v2_x, v2_y, v2_z);
}


void Search_tree_node::project_grid()
{


PAT_REAL uv1_x, uv1_y, uv1_z;
PAT_REAL uv2_x, uv2_y, uv2_z;
PAT_REAL center_x, center_y, center_z;
lonlat2xyz(center[PDLN_LON], center[PDLN_LAT], &center_x, &center_y, &center_z);
calculate_unit_vectors(center[PDLN_LON], center[PDLN_LAT], &uv1_x, &uv1_y, &uv1_z, &uv2_x, &uv2_y, &uv2_z);

if(projected_coord[0] == NULL) {
projected_coord[0] = new double[num_kernel_points + len_expand_coord_buf];
projected_coord[1] = new double[num_kernel_points + len_expand_coord_buf];

for(int i = 0; i < num_kernel_points; i++) {
fast_stereographic_projection(kernel_coord[PDLN_LON][i], kernel_coord[PDLN_LAT][i],
center_x, center_y, center_z, uv1_x, uv1_y, uv1_z, uv2_x, uv2_y, uv2_z,
projected_coord[PDLN_LON][i], projected_coord[PDLN_LAT][i]);
}

for(int i = 0; i < num_expand_points; i++) {
fast_stereographic_projection(expand_coord[PDLN_LON][i], expand_coord[PDLN_LAT][i],
center_x, center_y, center_z, uv1_x, uv1_y, uv1_z, uv2_x, uv2_y, uv2_z,
projected_coord[PDLN_LON][i+num_kernel_points], projected_coord[PDLN_LAT][i+num_kernel_points]);
}

num_projected_points = num_kernel_points + num_expand_points;
} else {
for(int i = num_projected_points - num_kernel_points; i < num_expand_points; i++) {
fast_stereographic_projection(expand_coord[PDLN_LON][i], expand_coord[PDLN_LAT][i],
center_x, center_y, center_z, uv1_x, uv1_y, uv1_z, uv2_x, uv2_y, uv2_z,
projected_coord[PDLN_LON][i+num_kernel_points], projected_coord[PDLN_LAT][i+num_kernel_points]);
}

num_projected_points = num_kernel_points + num_expand_points;
}


double top = -1e20, bot = 1e20, left = 1e20, right = -1e20;
for(int i = 0; i < num_projected_points; i++) { 
if (projected_coord[PDLN_LON][i] < left)  left = projected_coord[PDLN_LON][i];
if (projected_coord[PDLN_LON][i] > right) right = projected_coord[PDLN_LON][i];
if (projected_coord[PDLN_LAT][i] < bot) bot = projected_coord[PDLN_LAT][i];
if (projected_coord[PDLN_LAT][i] > top) top = projected_coord[PDLN_LAT][i];
}
if(project_boundry != NULL) {
PDASSERT(project_boundry->min_lon >= left && project_boundry->max_lon <= right && project_boundry->min_lat >= bot && project_boundry->max_lat <= top);
delete project_boundry;
}
project_boundry = new Boundry(left, right, bot, top);
}


Delaunay_grid_decomposition::Delaunay_grid_decomposition(Grid_info grid_info, Processing_resource *proc_info, int min_points_per_chunk)
: search_tree_root(NULL)
, min_points_per_chunk(min_points_per_chunk)
, original_grid(0)
, mask(NULL)
, global_index(NULL)
, processing_info(proc_info)
, active_processing_units_flag(NULL)
, is_local_proc_active(false)
, workloads(NULL)
, average_workload(0)
, regionID_to_unitID(NULL)
, all_group_intervals(NULL)
, buf_int(NULL)
, buf_bool(NULL)
{
buf_double[0] = buf_double[1] = NULL;

timeval start, end;
double **coords;
DISABLING_POINTS_METHOD mask_method;
int num;
void *data;

PDASSERT(processing_info != NULL);

coords     = grid_info.coord_values;
mask       = grid_info.mask;
num_points = grid_info.num_total_points;
is_cyclic  = grid_info.is_cyclic;
num_fence_points = grid_info.num_fence_points;
boundary_from_user = grid_info.boundary;
coord_values[0] = coords[0];
coord_values[1] = coords[1];

bool south_pole = float_eq(boundary_from_user.min_lat, -90.0);
bool north_pole = float_eq(boundary_from_user.max_lat,  90.0);
int regions_id_end = initialze_workloads(south_pole, north_pole);

if (!is_local_proc_active)
return;

global_index = new int[num_points];
for(int i = 0; i < num_points; i++)
global_index[i] = i;

Boundry boundary = boundary_from_user;
if(boundary.max_lon - boundary.min_lon < 360.0)
boundary.max_lon += PDLN_HIGH_BOUNDRY_SHIFTING;
boundary.max_lat += PDLN_HIGH_BOUNDRY_SHIFTING;

PDASSERT(boundary.max_lon - boundary.min_lon <= 360.0);
search_tree_root = new Search_tree_node(NULL, coord_values, global_index, mask, num_points, boundary, PDLN_NODE_TYPE_COMMON);
search_tree_root->calculate_real_boundary();
search_tree_root->update_region_ids(1, regions_id_end);
PDASSERT(search_tree_root->ids_size() > 0);



initialze_buffer();
}


Delaunay_grid_decomposition::~Delaunay_grid_decomposition()
{
delete[] global_index;
delete[] coord_values[0];
delete[] coord_values[1];
delete search_tree_root;
delete[] regionID_to_unitID;
delete[] workloads; 
delete[] active_processing_units_flag;
delete[] all_group_intervals;

int num_local_threads = processing_info->get_num_local_threads();
if (buf_int)
for (int i = 0; i < num_local_threads; i++) {
delete[] buf_double[0][i];
delete[] buf_double[1][i];
delete[] buf_int[i];
if (mask)
delete[] buf_bool[i];
}
delete[] buf_double[0];
delete[] buf_double[1];
delete[] buf_int;
delete[] buf_bool;
}


int Delaunay_grid_decomposition::initialze_workloads(bool south_pole, bool north_pole)
{
int max_punits   = (num_points + min_points_per_chunk - 1) / min_points_per_chunk;
int total_punits = processing_info->get_num_total_processing_units();
num_regions      = std::max(std::min(total_punits, max_punits), 4);
average_workload = (double)num_points / num_regions;

PDASSERT(min_points_per_chunk > 0);

double polar_workload = average_workload * (1 - PDLN_POLAR_WORKLOAD_RATE);

if (south_pole && north_pole)
average_workload += 2 * average_workload * PDLN_POLAR_WORKLOAD_RATE / (num_regions - 2);
else if (south_pole)
average_workload += average_workload * PDLN_POLAR_WORKLOAD_RATE / (num_regions - 1);
else if (north_pole)
average_workload += average_workload * PDLN_POLAR_WORKLOAD_RATE / (num_regions - 1);


regionID_to_unitID = new int[num_regions+2];
workloads          = new double[num_regions+2];


int regions_id_end = num_regions+1;
for(int i = 1; i < num_regions+1; i++)
workloads[i] = average_workload;

if (north_pole)
workloads[num_regions] = polar_workload;
if (south_pole)
workloads[1] = polar_workload;

active_processing_units_flag = new bool[processing_info->get_num_total_processing_units()];
processing_info->pick_out_active_processing_units(num_regions, active_processing_units_flag);

int  num_local_punits = processing_info->get_num_local_threads();
int* local_punits_id = processing_info->get_local_proc_common_id();
for (int i = 0; i < num_local_punits; i++)
if (active_processing_units_flag[local_punits_id[i]]) {
is_local_proc_active = true;
break;
}


const int offset = 1;
if(total_punits > 3) {
for(int i = 0, regionID = 1; i < total_punits; i++)
if(active_processing_units_flag[i])
regionID_to_unitID[regionID++] = i;
}
else if(total_punits == 1)
regionID_to_unitID[0+offset] = regionID_to_unitID[1+offset] = regionID_to_unitID[2+offset] = regionID_to_unitID[3+offset] = 0;
else if(total_punits == 2) {
regionID_to_unitID[0+offset] = regionID_to_unitID[1+offset] = 0;
regionID_to_unitID[2+offset] = regionID_to_unitID[3+offset] = 1;
}
else if(total_punits == 3) {
regionID_to_unitID[0+offset] = 0;
regionID_to_unitID[1+offset] = regionID_to_unitID[2+offset] = 1;
regionID_to_unitID[3+offset] = 2;
}

return regions_id_end;
}


void Delaunay_grid_decomposition::update_workloads(int total_workload, int start, int end, bool kill_tiny_region)
{

int size = end - start;
if (size  == 1) {
workloads[start] = total_workload;
return;
}

double old_total_workload = 0.0;
for (int i = start; i < end; i++)
old_total_workload += workloads[i];

int average_new_load = total_workload / size;
for (int i = start; i < end; i++)
workloads[i] = average_new_load;
for (int i = start; i < start + total_workload % size; i++)
workloads[i] += 1;

if (!kill_tiny_region)
return;

int non_zero_regions = size;
for (int i = start; i < end; i++)
if (workloads[i] == 0)
non_zero_regions--;

for (int i = start; i < end; i++) {
if (non_zero_regions < 2)
break;

if (workloads[i] > 0 && workloads[i] < min_points_per_chunk) {
active_processing_units_flag[i] = false;
non_zero_regions--;
double average_workload = workloads[i] / non_zero_regions;
workloads[i] = 0;
for(int j = start; j < end; j++)
if (workloads[j] > 0)
workloads[j] += average_workload;
}
}
#ifdef DEBUG
PDASSERT(non_zero_regions > 0);
#endif
}


void Delaunay_grid_decomposition::initialze_buffer()
{
int num_local_threads = processing_info->get_num_local_threads();
buf_double[0] = new double*[num_local_threads];
buf_double[1] = new double*[num_local_threads];
buf_int = new int*[num_local_threads];
if (mask)
buf_bool = new bool*[num_local_threads];

for (int i = 0; i < num_local_threads; i++) {
buf_double[0][i] = new double[search_tree_root->num_kernel_points];
buf_double[1][i] = new double[search_tree_root->num_kernel_points];
buf_int[i] = new int[search_tree_root->num_kernel_points];
if (mask)
buf_bool[i] = new bool[search_tree_root->num_kernel_points];
}
}



void decompose_common_node_recursively(Delaunay_grid_decomposition *decomp, Search_tree_node *node, int min_points_per_chunk, bool lazy_mode)
{
double* c_points_coord[4];
int*    c_points_index[2];
bool*   c_points_mask[2];
int     c_num_points[2];
Boundry c_boundry[2];
int     c_ids_start[2];
int     c_ids_end[2];
int*    c_intervals[2];
int     c_num_intervals[2];

PDASSERT(node->ids_size() > 0);
if(node->ids_size() == 1) {
if(node->num_kernel_points > 0 && decomp->have_local_region_ids(node->ids_start, node->ids_end))
#pragma omp critical
decomp->local_leaf_nodes.push_back(node);
#pragma omp critical
decomp->all_leaf_nodes.push_back(node);
return;
}

node->decompose_by_processing_units_number(decomp->workloads, c_points_coord, c_points_index, c_points_mask,
c_num_points, c_boundry, c_ids_start, c_ids_end, 
PDLN_DECOMPOSE_COMMON_MODE, c_intervals, 
c_num_intervals, min_points_per_chunk);

for (int i = 0; i < c_num_points[0]; i++)
PDASSERT(is_in_region(c_points_coord[0][i], c_points_coord[1][i], c_boundry[0]));
for (int i = 0; i < c_num_points[1]; i++)
PDASSERT(is_in_region(c_points_coord[2][i], c_points_coord[3][i], c_boundry[1]));

if(c_num_points[0] != 0 || c_num_points[1] != 0) {
PDASSERT(c_points_coord[0] + c_num_points[0] == c_points_coord[2]);
PDASSERT(c_points_coord[1] + c_num_points[0] == c_points_coord[3]);
PDASSERT(c_points_coord[2] + c_num_points[1] == node->kernel_coord[0] + node->num_kernel_points);
PDASSERT(c_points_coord[3] + c_num_points[1] == node->kernel_coord[1] + node->num_kernel_points);
}
for (int i = 0; i < c_num_points[0]; i++)
PDASSERT(is_in_region(c_points_coord[0][i], c_points_coord[1][i], c_boundry[0]));
for (int i = 0; i < c_num_points[1]; i++)
PDASSERT(is_in_region(c_points_coord[2][i], c_points_coord[3][i], c_boundry[1]));

node->children[0] = decomp->alloc_search_tree_node(node, c_points_coord,   c_points_index[0], c_points_mask[0], c_num_points[0], c_boundry[0], c_ids_start[0], c_ids_end[0], PDLN_NODE_TYPE_COMMON);
node->children[2] = decomp->alloc_search_tree_node(node, c_points_coord+2, c_points_index[1], c_points_mask[1], c_num_points[1], c_boundry[1], c_ids_start[1], c_ids_end[1], PDLN_NODE_TYPE_COMMON);

node->children[0]->set_groups(c_intervals[0], c_num_intervals[0]);
node->children[2]->set_groups(c_intervals[1], c_num_intervals[1]);

if(!lazy_mode || decomp->have_local_region_ids(node->children[0]->ids_start, node->children[0]->ids_end)) {
#pragma omp task
decompose_common_node_recursively(decomp, node->children[0], min_points_per_chunk, lazy_mode);
}
if(!lazy_mode || decomp->have_local_region_ids(node->children[2]->ids_start, node->children[2]->ids_end)) {
#pragma omp task
decompose_common_node_recursively(decomp, node->children[2], min_points_per_chunk, lazy_mode);
}
}


Search_tree_node* Delaunay_grid_decomposition::alloc_search_tree_node(Search_tree_node* parent, double *coord_values[2], int *index, 
bool *mask, int num_points, Boundry boundary, int ids_start,
int ids_end, int type, bool kill_tiny_region)
{
PDASSERT(ids_end - ids_start > 0);
Search_tree_node *new_node = new Search_tree_node(parent, coord_values, index, mask, num_points, boundary, type);

#pragma omp critical
update_workloads(num_points, ids_start, ids_end, kill_tiny_region);

new_node->update_region_ids(ids_start, ids_end);
if(ids_end - ids_start == 1) {
new_node->region_id = ids_start;
new_node->is_leaf = true;
}

return new_node;
}


bool Delaunay_grid_decomposition::do_two_regions_overlap(Boundry region1, Boundry region2)
{
if(region1.max_lat <= region2.min_lat || region1.min_lat >= region2.max_lat)
return false;
if(region1.max_lon <= region2.min_lon || region1.min_lon >= region2.max_lon)
return false;

return true;
}

#define PDLN_BOUNDRY_TYPE_CLEAR     (0x0FFFFFFFFFFFFFFF)
#define PDLN_BOUNDRY_TYPE_NON       (0x0000000000000000)
#define PDLN_BOUNDRY_TYPE_U         (0x1000000000000000)
#define PDLN_BOUNDRY_TYPE_D         (0x2000000000000000)
#define PDLN_BOUNDRY_TYPE_L         (0x4000000000000000)
#define PDLN_BOUNDRY_TYPE_R         (0x8000000000000000)
#define PDLN_BOUNDRY_TYPE_LR        (0xC000000000000000)
#define PDLN_BOUNDRY_TYPE_INVALID   (0xF000000000000000)
#define set_boundry_type(val, type) ((val & PDLN_BOUNDRY_TYPE_CLEAR) | type)
#define get_boundry_type(val)       (val & ~PDLN_BOUNDRY_TYPE_CLEAR)

unsigned Delaunay_grid_decomposition::compute_common_boundry(Search_tree_node *tree_node, Search_tree_node *neighbor_node, Point *boundry_head,
Point *boundry_tail, Point *cyclic_boundry_head, Point *cyclic_boundry_tail)
{
unsigned boundry_type = 0;
double coord_value[2][2];

coord_value[0][PDLN_LAT] = coord_value[0][PDLN_LON] = coord_value[1][PDLN_LAT] = coord_value[1][PDLN_LON] = PDLN_DOUBLE_INVALID_VALUE;
if(tree_node->kernel_boundry->max_lat == neighbor_node->kernel_boundry->min_lat) { 
if(std::max(tree_node->kernel_boundry->min_lon, neighbor_node->kernel_boundry->min_lon) <
std::min(tree_node->kernel_boundry->max_lon, neighbor_node->kernel_boundry->max_lon)) {
tree_node->num_neighbors_on_boundry[PDLN_UP]++;
boundry_type |= PDLN_BOUNDRY_TYPE_U;
coord_value[0][PDLN_LAT] = coord_value[1][PDLN_LAT] = tree_node->kernel_boundry->max_lat;
coord_value[0][PDLN_LON] = std::max(tree_node->kernel_boundry->min_lon, neighbor_node->kernel_boundry->min_lon);
coord_value[1][PDLN_LON] = std::min(tree_node->kernel_boundry->max_lon, neighbor_node->kernel_boundry->max_lon);
}
}
else if(tree_node->kernel_boundry->min_lon == neighbor_node->kernel_boundry->max_lon) { 
if(std::max(tree_node->kernel_boundry->min_lat, neighbor_node->kernel_boundry->min_lat) <
std::min(tree_node->kernel_boundry->max_lat, neighbor_node->kernel_boundry->max_lat)) {
tree_node->num_neighbors_on_boundry[PDLN_LEFT]++;
boundry_type |= PDLN_BOUNDRY_TYPE_L;
coord_value[0][PDLN_LON] = coord_value[1][PDLN_LON] = neighbor_node->kernel_boundry->max_lon;
coord_value[0][PDLN_LAT] = std::max(tree_node->kernel_boundry->min_lat, neighbor_node->kernel_boundry->min_lat);
coord_value[1][PDLN_LAT] = std::min(tree_node->kernel_boundry->max_lat, neighbor_node->kernel_boundry->max_lat);
}
}
else if(tree_node->kernel_boundry->min_lat == neighbor_node->kernel_boundry->max_lat)  { 
if(std::max(tree_node->kernel_boundry->min_lon, neighbor_node->kernel_boundry->min_lon) < 
std::min(tree_node->kernel_boundry->max_lon, neighbor_node->kernel_boundry->max_lon)) {
tree_node->num_neighbors_on_boundry[PDLN_DOWN]++;
boundry_type |= PDLN_BOUNDRY_TYPE_D;
coord_value[0][PDLN_LAT] = coord_value[1][PDLN_LAT] = neighbor_node->kernel_boundry->max_lat;
coord_value[0][PDLN_LON] = std::max(tree_node->kernel_boundry->min_lon, neighbor_node->kernel_boundry->min_lon);
coord_value[1][PDLN_LON] = std::min(tree_node->kernel_boundry->max_lon, neighbor_node->kernel_boundry->max_lon);
}
}
else if(tree_node->kernel_boundry->max_lon == neighbor_node->kernel_boundry->min_lon) { 
if(std::max(tree_node->kernel_boundry->min_lat, neighbor_node->kernel_boundry->min_lat) <
std::min(tree_node->kernel_boundry->max_lat, neighbor_node->kernel_boundry->max_lat)) {
tree_node->num_neighbors_on_boundry[PDLN_RIGHT]++;
boundry_type |= PDLN_BOUNDRY_TYPE_R;
coord_value[0][PDLN_LON] = coord_value[1][PDLN_LON] = tree_node->kernel_boundry->max_lon;
coord_value[0][PDLN_LAT] = std::max(tree_node->kernel_boundry->min_lat, neighbor_node->kernel_boundry->min_lat);
coord_value[1][PDLN_LAT] = std::min(tree_node->kernel_boundry->max_lat, neighbor_node->kernel_boundry->max_lat);
}
}
*boundry_head = Point(coord_value[0][PDLN_LON], coord_value[0][PDLN_LAT], -1);
*boundry_tail = Point(coord_value[1][PDLN_LON], coord_value[1][PDLN_LAT], -1);


coord_value[0][PDLN_LAT] = coord_value[0][PDLN_LON] = coord_value[1][PDLN_LAT] = coord_value[1][PDLN_LON] = PDLN_DOUBLE_INVALID_VALUE;
if(std::max(tree_node->kernel_boundry->min_lat, neighbor_node->kernel_boundry->min_lat) < 
std::min(tree_node->kernel_boundry->max_lat, neighbor_node->kernel_boundry->max_lat)) {
if(float_eq(fabs(tree_node->kernel_boundry->min_lon - neighbor_node->kernel_boundry->max_lon), 360.0)) { 
coord_value[0][PDLN_LON] = coord_value[1][PDLN_LON] = tree_node->kernel_boundry->min_lon;
coord_value[0][PDLN_LAT] = std::max(tree_node->kernel_boundry->min_lat, neighbor_node->kernel_boundry->min_lat);
coord_value[1][PDLN_LAT] = std::min(tree_node->kernel_boundry->max_lat, neighbor_node->kernel_boundry->max_lat);
tree_node->num_neighbors_on_boundry[PDLN_LEFT]++;
boundry_type |= PDLN_BOUNDRY_TYPE_L;
}
if(float_eq(fabs(tree_node->kernel_boundry->max_lon - neighbor_node->kernel_boundry->min_lon), 360.0)) { 
coord_value[0][PDLN_LON] = coord_value[1][PDLN_LON] = tree_node->kernel_boundry->max_lon;
coord_value[0][PDLN_LAT] = std::max(tree_node->kernel_boundry->min_lat, neighbor_node->kernel_boundry->min_lat);
coord_value[1][PDLN_LAT] = std::min(tree_node->kernel_boundry->max_lat, neighbor_node->kernel_boundry->max_lat);
tree_node->num_neighbors_on_boundry[PDLN_RIGHT]++;
boundry_type |= PDLN_BOUNDRY_TYPE_R;
}
}
*cyclic_boundry_head = Point(coord_value[0][PDLN_LON], coord_value[0][PDLN_LAT], -1);
*cyclic_boundry_tail = Point(coord_value[1][PDLN_LON], coord_value[1][PDLN_LAT], -1);
return boundry_type;
}



void Delaunay_grid_decomposition::send_checksum_to_remote(int src_common_id, int dst_common_id, unsigned long* checksum, int tag, MPI_Request** req)
{
if(processing_info->get_local_process_id() == processing_info->get_processing_unit(dst_common_id)->process_id) {
*req = NULL;
processing_info->send_to_local_thread(checksum, 1, sizeof(unsigned long),
processing_info->get_processing_unit(src_common_id)->thread_id,
processing_info->get_processing_unit(dst_common_id)->thread_id, tag);
} else {
*req = new MPI_Request;
#pragma omp critical
{
MPI_Isend(checksum, 1, MPI_UNSIGNED_LONG, processing_info->get_processing_unit(dst_common_id)->process_id, 
tag, processing_info->get_mpi_comm(), *req);
}
}
}


void Delaunay_grid_decomposition::recv_checksum_from_remote(int src_common_id, int dst_common_id, unsigned long* checksum, int tag, MPI_Request** req)
{
if(processing_info->get_local_process_id() == processing_info->get_processing_unit(src_common_id)->process_id) {
*req = NULL;
processing_info->recv_from_local_thread(checksum, 1, sizeof(unsigned long),
processing_info->get_processing_unit(src_common_id)->thread_id,
processing_info->get_processing_unit(dst_common_id)->thread_id, tag);
} else {
*req = new MPI_Request;
#pragma omp critical
{
MPI_Irecv(checksum, 1, MPI_UNSIGNED_LONG, processing_info->get_processing_unit(src_common_id)->process_id, 
tag, processing_info->get_mpi_comm(), *req);
}
}
}


namespace std   
{
namespace tr1
{
template <>
struct hash<Point>
{
std::size_t operator()(const Point &p) const
{
return (hash<double>()(p.x) ^ hash<double>()(p.y) << 1) >> 1;
}
};

template <>  
struct hash<Triangle_inline>  
{  
std::size_t operator()(const Triangle_inline &t) const  
{  
return hash<Point>()(t.v[0]) ^ hash<Point>()(t.v[1]) ^ hash<Point>()(t.v[2]);
}
}; 

template <>  
struct hash<Triangle_ID_Only>  
{  
std::size_t operator()(const Triangle_ID_Only &t) const  
{  
return hash<int>()(t.id[0]+t.id[1]+t.id[2]+t.id[0]*t.id[1]*t.id[2]);
}
}; 
}
} 


struct Triangles_digest {
unsigned int  num;
unsigned long checksum;
};


#define PDLN_SET_TAG_ITER(it)           (it<<24 &0x0F000000)
#define PDLN_SET_TAG_SRC(tag, id)      ((id<<12 &0x00FFF000) | tag)
#define PDLN_SET_TAG_DST(tag, id)      ((id     &0x00000FFF) | tag)
#define PDLN_SET_TAG(src, dst, iter)   (PDLN_SET_TAG_DST(PDLN_SET_TAG_SRC(PDLN_SET_TAG_ITER(iter), src), dst))
void Delaunay_grid_decomposition::send_recv_checksums_with_neighbors(Search_tree_node *leaf_node, unsigned long *local_checksums,
unsigned long *remote_checksums, vector<MPI_Request*> *waiting_list, int iter)
{

Point common_boundary_head, common_boundary_tail, cyclic_common_boundary_head, cyclic_common_boundary_tail;

double threshold = std::min(search_tree_root->kernel_boundry->max_lon - search_tree_root->kernel_boundry->min_lon,
search_tree_root->kernel_boundry->max_lat - search_tree_root->kernel_boundry->min_lat) /
sqrt(processing_info->get_num_total_processing_units()) / 2.0;

leaf_node->init_num_neighbors_on_boundry(0);
for(unsigned i = 0; i < leaf_node->neighbors.size(); i++) {
#ifdef DEBUG
PDASSERT(leaf_node->neighbors[i].first->ids_size() == 1);
PDASSERT(iter <= 0xFF);
PDASSERT(leaf_node->region_id <= 0xFFF);
PDASSERT(leaf_node->neighbors[i].first->region_id <= 0xFFF);
#endif

unsigned boundry_type = compute_common_boundry(leaf_node, leaf_node->neighbors[i].first, &common_boundary_head, &common_boundary_tail,
&cyclic_common_boundary_head, &cyclic_common_boundary_tail);


unsigned long checksum = 0;
if(common_boundary_head.x != PDLN_DOUBLE_INVALID_VALUE)
checksum ^= leaf_node->triangulation->cal_checksum(common_boundary_head, common_boundary_tail, threshold);

if(cyclic_common_boundary_head.x != PDLN_DOUBLE_INVALID_VALUE)
checksum ^= leaf_node->triangulation->cal_checksum(cyclic_common_boundary_head, cyclic_common_boundary_tail, threshold);

checksum = set_boundry_type(checksum, boundry_type);
local_checksums[i] = checksum;


if(common_boundary_head.x != PDLN_DOUBLE_INVALID_VALUE || cyclic_common_boundary_head.x != PDLN_DOUBLE_INVALID_VALUE) {
MPI_Request *req;
send_checksum_to_remote(regionID_to_unitID[leaf_node->region_id], regionID_to_unitID[leaf_node->neighbors[i].first->region_id],
&local_checksums[i], PDLN_SET_TAG(leaf_node->region_id, leaf_node->neighbors[i].first->region_id, iter),
&req);
#ifdef DEBUG
if (req)
waiting_list->push_back(req);
#endif

recv_checksum_from_remote(regionID_to_unitID[leaf_node->neighbors[i].first->region_id], regionID_to_unitID[leaf_node->region_id],
&remote_checksums[i], PDLN_SET_TAG(leaf_node->neighbors[i].first->region_id, leaf_node->region_id, iter),
&req);
if (req)
waiting_list->push_back(req);
}
else
remote_checksums[i] = 0;
}
}


bool Delaunay_grid_decomposition::are_checksums_identical(Search_tree_node *leaf_node, unsigned long *local_checksums, unsigned long*remote_checksums)
{
if(leaf_node->neighbors.size() == 0) {
printf("some region has no neighbor, that's weird\n");
return false;
}

bool ok = true;

for(unsigned i = 0; i < leaf_node->neighbors.size(); i++) {
unsigned long l_checksum = local_checksums[i] & PDLN_BOUNDRY_TYPE_CLEAR;
unsigned long r_checksum = remote_checksums[i] & PDLN_BOUNDRY_TYPE_CLEAR;
if (l_checksum == r_checksum) {
leaf_node->neighbors[i].second = true;
leaf_node->reduce_num_neighbors_on_boundry(get_boundry_type(local_checksums[i]));
leaf_node->clear_expanding_count(get_boundry_type(local_checksums[i]));

if (l_checksum != 0)
log(LOG_DEBUG, "%3d -> %3d same, %016lx vs %016lx\n", leaf_node->region_id, leaf_node->neighbors[i].first->region_id, l_checksum, r_checksum);
} else {
leaf_node->neighbors[i].second = false;
ok = false;

log(LOG_DEBUG, "%3d -> %3d diff, %016lx vs %016lx\n", leaf_node->region_id, leaf_node->neighbors[i].first->region_id, l_checksum, r_checksum);
}
}

return ok;
}


struct Decomposing_Info
{
double* coord[2];
int*    index;
bool*   mask;
Boundry bound;
int     num_points;
int     id_lower_bound;
int     id_upper_bound;
};


int Delaunay_grid_decomposition::assign_polars(bool assign_south_polar, bool assign_north_polar)
{
Midline     midline;
timeval     start, end;
double*     c_points_coord[4];
int*        c_points_index[2];
bool*       c_points_mask[2];
Boundry     c_boundry[2];
int         c_num_points[2];
int         c_ids_start[2];
int         c_ids_end[2];

if(!assign_south_polar && !assign_north_polar)
return 0;

if(assign_south_polar) {
for (;;) {
current_tree_node->decompose_by_processing_units_number(workloads, c_points_coord, c_points_index, c_points_mask,
c_num_points, c_boundry, c_ids_start, c_ids_end,
PDLN_DECOMPOSE_SPOLAR_MODE, NULL, NULL, min_points_per_chunk);
bool valid = is_polar_region_valid(c_num_points[0], &c_boundry[0]);
if (valid)
break;
else {
double old_polar_workload = workloads[1];
int delta = old_polar_workload * 0.5;
update_workloads(old_polar_workload + delta, c_ids_start[0], c_ids_end[0], false);
update_workloads(c_num_points[0] + c_num_points[1] - old_polar_workload - delta, c_ids_start[1], c_ids_end[1], false);
}
}
if(c_boundry[0].max_lat > PDLN_SPOLAR_MAX_LAT || c_boundry[0].max_lat < PDLN_SPOLAR_MIN_LAT) {
midline.type = PDLN_LAT;
midline.value = c_boundry[0].max_lat > PDLN_SPOLAR_MAX_LAT ? PDLN_SPOLAR_MAX_LAT : PDLN_SPOLAR_MIN_LAT;
current_tree_node->divide_at_fix_line(midline, c_points_coord, c_points_index, c_points_mask, c_num_points);;

if(c_num_points[0] < min_points_per_chunk)
goto fail;

c_boundry[0].max_lat = midline.value;
c_boundry[1].min_lat = midline.value;

if (c_boundry[0].max_lat > PDLN_SPOLAR_MAX_LAT) {

c_ids_start[0] = 0;
c_ids_end[0]   = 1;
c_ids_start[1] = 1;

workloads[1] -= c_num_points[0];
} else {
c_ids_start[0] = 1;
c_ids_end[0]   = 2;
c_ids_start[1] = 2;
}
}
PDASSERT(c_points_coord[0] + c_num_points[0] == c_points_coord[2]);
PDASSERT(c_points_coord[1] + c_num_points[0] == c_points_coord[3]);
PDASSERT(c_points_coord[2] + c_num_points[1] == current_tree_node->kernel_coord[0] + current_tree_node->num_kernel_points);
PDASSERT(c_points_coord[3] + c_num_points[1] == current_tree_node->kernel_coord[1] + current_tree_node->num_kernel_points);
for (int i = 0; i < c_num_points[0]; i++)
PDASSERT(is_in_region(c_points_coord[0][i], c_points_coord[1][i], c_boundry[0]));
for (int i = 0; i < c_num_points[1]; i++)
PDASSERT(is_in_region(c_points_coord[2][i], c_points_coord[3][i], c_boundry[1]));
search_tree_root->children[0] = alloc_search_tree_node(search_tree_root, c_points_coord,   c_points_index[0], c_points_mask[0], c_num_points[0],
c_boundry[0], c_ids_start[0], c_ids_end[0], PDLN_NODE_TYPE_SPOLAR);
search_tree_root->children[1] = alloc_search_tree_node(search_tree_root, c_points_coord+2, c_points_index[1], c_points_mask[1], c_num_points[1],
c_boundry[1], c_ids_start[1], c_ids_end[1], PDLN_NODE_TYPE_COMMON, false);

current_tree_node = search_tree_root->children[1];

if(have_local_region_ids(search_tree_root->children[0]->ids_start, search_tree_root->children[0]->ids_end))
local_leaf_nodes.push_back(search_tree_root->children[0]);
all_leaf_nodes.push_back(search_tree_root->children[0]);

if (search_tree_root->children[0]->num_kernel_points > average_workload * 4)
search_tree_root->children[0]->fast_triangulate = true;

}

if(assign_north_polar) {
for (;;) {
current_tree_node->decompose_by_processing_units_number(workloads, c_points_coord, c_points_index, c_points_mask,
c_num_points, c_boundry, c_ids_start, c_ids_end,
PDLN_DECOMPOSE_NPOLAR_MODE, NULL, NULL, min_points_per_chunk);
bool valid = is_polar_region_valid(c_num_points[1], &c_boundry[1]);
if (valid)
break;
else {
double old_polar_workload = workloads[num_regions];
int delta = old_polar_workload * 0.5;
update_workloads(old_polar_workload + delta, c_ids_start[1], c_ids_end[1], false);
update_workloads(c_num_points[0] + c_num_points[1] - old_polar_workload - delta, c_ids_start[0], c_ids_end[0], false);
}
}
if(c_boundry[1].min_lat < PDLN_NPOLAR_MIN_LAT || c_boundry[1].min_lat > PDLN_NPOLAR_MAX_LAT) {
midline.type = PDLN_LAT;
midline.value = c_boundry[1].min_lat < PDLN_NPOLAR_MIN_LAT ? PDLN_NPOLAR_MIN_LAT : PDLN_NPOLAR_MAX_LAT;
current_tree_node->divide_at_fix_line(midline, c_points_coord, c_points_index, c_points_mask, c_num_points);;

if(c_num_points[1] < min_points_per_chunk)
goto fail;

c_boundry[0].max_lat = midline.value;
c_boundry[1].min_lat = midline.value;

if (c_boundry[1].min_lat < PDLN_NPOLAR_MIN_LAT) {

c_ids_start[1] = num_regions+1;
c_ids_end[1]   = num_regions+2;
c_ids_end[0]   = num_regions+1;

workloads[num_regions] -= c_num_points[1];
} else {
c_ids_start[1] = num_regions;
c_ids_end[1]   = num_regions+1;
c_ids_end[0]   = num_regions;
}
}
delete search_tree_root->children[1];

PDASSERT(c_points_coord[0] + c_num_points[0] == c_points_coord[2]);
PDASSERT(c_points_coord[1] + c_num_points[0] == c_points_coord[3]);
PDASSERT(c_points_coord[2] + c_num_points[1] == current_tree_node->kernel_coord[0] + current_tree_node->num_kernel_points);
PDASSERT(c_points_coord[3] + c_num_points[1] == current_tree_node->kernel_coord[1] + current_tree_node->num_kernel_points);
for (int i = 0; i < c_num_points[0]; i++) {
if (!is_in_region(c_points_coord[0][i], c_points_coord[1][i], c_boundry[0]))
printf("(%lf, %lf), [%lf, %lf, %lf, %lf]\n", c_points_coord[0][i], c_points_coord[1][i], c_boundry[0].min_lon, c_boundry[0].max_lon, c_boundry[0].min_lat, c_boundry[0].max_lat);
PDASSERT(is_in_region(c_points_coord[0][i], c_points_coord[1][i], c_boundry[0]));
}
for (int i = 0; i < c_num_points[1]; i++)
PDASSERT(is_in_region(c_points_coord[2][i], c_points_coord[3][i], c_boundry[1]));
search_tree_root->children[2] = alloc_search_tree_node(search_tree_root, c_points_coord+2, c_points_index[1], c_points_mask[1], c_num_points[1], c_boundry[1],
c_ids_start[1], c_ids_end[1], PDLN_NODE_TYPE_NPOLAR);

search_tree_root->children[1] = alloc_search_tree_node(search_tree_root, c_points_coord,   c_points_index[0], c_points_mask[0], c_num_points[0], c_boundry[0],
c_ids_start[0], c_ids_end[0], PDLN_NODE_TYPE_COMMON);

current_tree_node = search_tree_root->children[1];

if(have_local_region_ids(search_tree_root->children[2]->ids_start, search_tree_root->children[2]->ids_end))
local_leaf_nodes.push_back(search_tree_root->children[2]);
all_leaf_nodes.push_back(search_tree_root->children[2]);

if (search_tree_root->children[2]->num_kernel_points > average_workload * 4)
search_tree_root->children[2]->fast_triangulate = true;

}

return 0;

fail:
printf("assign polars fault, %d, %d vs %d\n", c_num_points[0], c_num_points[1], min_points_per_chunk);
return 1;
}


double Delaunay_grid_decomposition::is_polar_region_valid(int num_points, Boundry* boundary)
{




return 360./(4. * sqrt(num_points)) < PAT_POLE_STRETCHING_THRESHOLD;
}


bool Delaunay_grid_decomposition::have_local_region_ids(int start, int end)
{
int* local_proc_ids = processing_info->get_local_proc_common_id();
int  num_local_proc_ids = processing_info->get_num_local_proc_processing_units();

if (num_local_proc_ids < 1)
return false;
if (local_proc_ids[0] > regionID_to_unitID[end-1])
return false;
if (local_proc_ids[num_local_proc_ids-1] < regionID_to_unitID[start])
return false;

for(int j = 0; j < num_local_proc_ids; j++)
for(int i = start; i < end; i++)
if(regionID_to_unitID[i] == local_proc_ids[j])
return true;

return false;
}


int Delaunay_grid_decomposition::generate_grid_decomposition(bool lazy_mode)
{
if (!is_local_proc_active)
return 0;

bool south_pole = float_eq(boundary_from_user.min_lat, -90.0);
bool north_pole = float_eq(boundary_from_user.max_lat,  90.0);

current_tree_node = search_tree_root;

if(assign_polars(south_pole, north_pole))
return 1;

int num_computing_nodes = processing_info->get_num_computing_nodes();
Processing_unit** units = processing_info->get_processing_units();

delete all_group_intervals;
all_group_intervals    = new int[num_computing_nodes]();
unsigned old_checksum  = units[regionID_to_unitID[current_tree_node->ids_start]]->hostname_checksum;
int cur_group          = 0;
all_group_intervals[0] = 1;
for(int i = current_tree_node->ids_start+1; i < current_tree_node->ids_end; i++)
if (old_checksum == units[regionID_to_unitID[i]]->hostname_checksum)
all_group_intervals[cur_group]++;
else {
all_group_intervals[++cur_group]++;
old_checksum = units[regionID_to_unitID[i]]->hostname_checksum;
}
PDASSERT(cur_group+1 <= num_computing_nodes);

current_tree_node->set_groups(all_group_intervals, cur_group+1);
#pragma omp parallel
{
#pragma omp single
{
decompose_common_node_recursively(this, current_tree_node, min_points_per_chunk, lazy_mode);
}
}
return 0;
}


double Search_tree_node::load_polars_info()
{
polars_local_index = new vector<int>();

if(node_type == PDLN_NODE_TYPE_NPOLAR) {
double nearest_point_lat = -1e10;
for(int i = 0; i < num_kernel_points+num_expand_points; i++) {
if(float_eq(kernel_coord[PDLN_LAT][i], 90.0))
polars_local_index->push_back(i);
else if(nearest_point_lat < kernel_coord[PDLN_LAT][i])
nearest_point_lat = kernel_coord[PDLN_LAT][i];
}
if (nearest_point_lat == -1e10)
nearest_point_lat = 89.0;

if(PDLN_INSERT_VIRTUAL_POINT && polars_local_index->size() != 1) {
shifted_polar_lat = (90.0 + nearest_point_lat) * 0.5;


for(unsigned i = 0; i < polars_local_index->size(); i++)
kernel_coord[PDLN_LAT][(*polars_local_index)[i]] = shifted_polar_lat;

double vpoint_lon = 0;
double vpoint_lat = 90;
int vpoint_idx = -1;
virtual_point_local_index = num_kernel_points+num_expand_points;
bool tmp_mask = true;
add_expand_points(&vpoint_lon, &vpoint_lat, &vpoint_idx, &tmp_mask, 1);
#ifdef DEBUG
PDASSERT(expand_coord[PDLN_LON][virtual_point_local_index-num_kernel_points] == vpoint_lon);
PDASSERT(expand_coord[PDLN_LAT][virtual_point_local_index-num_kernel_points] == vpoint_lat);
PDASSERT(expand_index[virtual_point_local_index-num_kernel_points] == vpoint_idx);
#endif
return shifted_polar_lat;
}
return PDLN_DOUBLE_INVALID_VALUE;
}
else if(node_type == PDLN_NODE_TYPE_SPOLAR) {
double nearest_point_lat = 1e10;
for(int i = 0; i < num_kernel_points+num_expand_points; i++) {
if(float_eq(kernel_coord[PDLN_LAT][i], -90.0))
polars_local_index->push_back(i);
else if(nearest_point_lat > kernel_coord[PDLN_LAT][i])
nearest_point_lat = kernel_coord[PDLN_LAT][i];
}
if (nearest_point_lat == 1e10)
nearest_point_lat = -89.0;

if(PDLN_INSERT_VIRTUAL_POINT && polars_local_index->size() != 1) {
shifted_polar_lat = (-90.0 + nearest_point_lat) * 0.5;

for(unsigned i = 0; i < polars_local_index->size(); i++)
kernel_coord[PDLN_LAT][(*polars_local_index)[i]] = shifted_polar_lat;

double vpoint_lon = 0;
double vpoint_lat = -90;
int vpoint_idx = -1;
virtual_point_local_index = num_kernel_points+num_expand_points;
bool tmp_mask = true;
add_expand_points(&vpoint_lon, &vpoint_lat, &vpoint_idx, &tmp_mask, 1);
#ifdef DEBUG
PDASSERT(expand_coord[PDLN_LON][virtual_point_local_index-num_kernel_points] == vpoint_lon);
PDASSERT(expand_coord[PDLN_LAT][virtual_point_local_index-num_kernel_points] == vpoint_lat);
PDASSERT(expand_index[virtual_point_local_index-num_kernel_points] == vpoint_idx);
#endif
return shifted_polar_lat;
}
return PDLN_DOUBLE_INVALID_VALUE;
}
return PDLN_DOUBLE_INVALID_VALUE;
}


Boundry Search_tree_node::expand()
{
Boundry expanded = *expand_boundry;

if(node_type == PDLN_NODE_TYPE_SPOLAR) {
if(num_neighbors_on_boundry[PDLN_UP] > 0) expanded.max_lat += (expanded.max_lat - expanded.min_lat) * expanding_scale[PDLN_UP] / 10.;
return expanded;
}
if(node_type == PDLN_NODE_TYPE_NPOLAR) {
if(num_neighbors_on_boundry[PDLN_DOWN] > 0) expanded.min_lat -= (expanded.max_lat - expanded.min_lat) * expanding_scale[PDLN_DOWN] / 10.;
return expanded;
}
if(node_type == PDLN_NODE_TYPE_COMMON) {
if(num_neighbors_on_boundry[PDLN_UP] > 0) {
expanded.max_lat += (expanded.max_lat - expanded.min_lat) * expanding_scale[PDLN_UP] / 10.;
edge_expanding_count[PDLN_UP]++;
}
if(num_neighbors_on_boundry[PDLN_LEFT] > 0) { 
expanded.min_lon -= (expanded.max_lon - expanded.min_lon) * expanding_scale[PDLN_LEFT] / 10.;
edge_expanding_count[PDLN_LEFT]++;
}
if(num_neighbors_on_boundry[PDLN_DOWN] > 0) {
expanded.min_lat -= (expanded.max_lat - expanded.min_lat) * expanding_scale[PDLN_DOWN] / 10.;
edge_expanding_count[PDLN_DOWN]++;
}
if(num_neighbors_on_boundry[PDLN_RIGHT] > 0) {
expanded.max_lon += (expanded.max_lon - expanded.min_lon) * expanding_scale[PDLN_RIGHT] / 10.;
edge_expanding_count[PDLN_RIGHT]++;
}
}
return expanded;
}


bool Search_tree_node::expanding_success(bool go_on[4])
{
log(LOG_DEBUG, "todo boundary: %d, %d, %d, %d\n", go_on[0], go_on[1], go_on[2], go_on[3]);
return !(go_on[0] || go_on[1] || go_on[2] || go_on[3]);
}


void Search_tree_node::set_groups(int *intervals, int num)
{
group_intervals = intervals;
num_groups = num;
}


#define PDLN_EXPANDING_LAYERS (8)
int Delaunay_grid_decomposition::expand_tree_node_boundry(Search_tree_node* tree_node, double expanding_ratio)
{
int num_found;
vector<Search_tree_node*> leaf_nodes_found;

Boundry* bound = tree_node->kernel_boundry;
double height_length_ratio = (bound->max_lat - bound->min_lat) / (bound->max_lon - bound->min_lon);
double quota[4];
if (is_polar_node(tree_node)) {
quota[PDLN_LEFT] = quota[PDLN_RIGHT] = sqrt(tree_node->num_kernel_points * height_length_ratio) * 2;
quota[PDLN_UP] = quota[PDLN_DOWN] = sqrt(tree_node->num_kernel_points / height_length_ratio) * 2;
} else {
quota[PDLN_LEFT] = quota[PDLN_RIGHT] = sqrt(tree_node->num_kernel_points * height_length_ratio) * PDLN_EXPANDING_LAYERS;
quota[PDLN_UP] = quota[PDLN_DOWN] = sqrt(tree_node->num_kernel_points / height_length_ratio) * PDLN_EXPANDING_LAYERS;
}

for (int i = 0; i < 4; i++)
if (tree_node->edge_expanding_count[i] > PDLN_SEPARATELY_EXPANDING_COUNT) {
tree_node->num_neighbors_on_boundry[(i+1)%4] = 1;
tree_node->num_neighbors_on_boundry[(i+2)%4] = 1;
tree_node->num_neighbors_on_boundry[(i+3)%4] = 1;
break;
}

int    num_edge_needing_expand = 0;
double unassigned_quota = 0;
for (int i = 0; i < 4; i++)
if (tree_node->num_neighbors_on_boundry[i] > 0)
num_edge_needing_expand++;
else {
unassigned_quota += quota[i];
quota[i] = 0;
}
unassigned_quota /= num_edge_needing_expand;
for (int i = 0; i < 4; i++)
if (tree_node->num_neighbors_on_boundry[i] > 0)
quota[i] += unassigned_quota;

PDASSERT(num_edge_needing_expand > 0);

int thread_id = omp_get_thread_num();
double* tmp_coord[2];
int*    tmp_index;
bool*   tmp_mask = NULL;
tmp_coord[0] = buf_double[0][thread_id];
tmp_coord[1] = buf_double[1][thread_id];
tmp_index = buf_int[thread_id];
if (mask)
tmp_mask  = buf_bool[thread_id];

bool go_on[4];
bool tree_need_extension = false;
do {
Boundry* old_boundry = tree_node->expand_boundry;
Boundry  new_boundry = tree_node->expand();
new_boundry.legalize(search_tree_root->kernel_boundry, is_cyclic);

go_on[0] = go_on[1] = go_on[2] = go_on[3] = false;

log(LOG_DEBUG, "kern boundary: %lf, %lf, %lf, %lf\n", tree_node->kernel_boundry->min_lon, tree_node->kernel_boundry->max_lon, tree_node->kernel_boundry->min_lat, tree_node->kernel_boundry->max_lat);
log(LOG_DEBUG, "last boundary: %lf, %lf, %lf, %lf\n", tree_node->expand_boundry->min_lon, tree_node->expand_boundry->max_lon, tree_node->expand_boundry->min_lat, tree_node->expand_boundry->max_lat);
log(LOG_DEBUG, "expd boundary: %lf, %lf, %lf, %lf\n", new_boundry.min_lon, new_boundry.max_lon, new_boundry.min_lat, new_boundry.max_lat);
leaf_nodes_found = adjust_expanding_boundry(old_boundry, &new_boundry, quota, tmp_coord, tmp_index, tmp_mask, go_on, &num_found, &tree_need_extension);
log(LOG_DEBUG, "adjt boundary: %lf, %lf, %lf, %lf\n", new_boundry.min_lon, new_boundry.max_lon, new_boundry.min_lat, new_boundry.max_lat);
log(LOG_DEBUG, "glbl boundary: %lf, %lf, %lf, %lf\n", search_tree_root->kernel_boundry->min_lon, search_tree_root->kernel_boundry->max_lon, search_tree_root->kernel_boundry->min_lat, search_tree_root->kernel_boundry->max_lat);

tree_node->add_neighbors(leaf_nodes_found);

if(tree_need_extension) {
log(LOG_WARNING, "region %d now re-extending search tree\n", tree_node->region_id);
return -1;
}

*tree_node->expand_boundry = new_boundry;
tree_node->add_expand_points(tmp_coord, tmp_index, tmp_mask, num_found);


if(new_boundry.max_lon - new_boundry.min_lon > (search_tree_root->kernel_boundry->max_lon - search_tree_root->kernel_boundry->min_lon) * 0.75 &&
new_boundry.max_lat - new_boundry.min_lat > (search_tree_root->kernel_boundry->max_lat - search_tree_root->kernel_boundry->min_lat) * 0.75) {
log(LOG_ERROR, "region %d too large\n", tree_node->region_id);
return 1;
}

if(new_boundry == *search_tree_root->kernel_boundry || new_boundry.max_lon - new_boundry.min_lon > 360.0) {
log(LOG_ERROR, "region %d expanded to the max\n", tree_node->region_id);
return 1;
}
}while(!tree_node->expanding_success(go_on));

log(LOG_DEBUG, "final expanded boundary: %lf, %lf, %lf, %lf\n", tree_node->expand_boundry->min_lon, tree_node->expand_boundry->max_lon, tree_node->expand_boundry->min_lat, tree_node->expand_boundry->max_lat);

return 0;
}



void Delaunay_grid_decomposition::halo_to_rectangles(Boundry inner_boundry, Boundry outer_boundry, Boundry sub_rectangle[4])
{
sub_rectangle[0].min_lon = sub_rectangle[1].min_lon = outer_boundry.min_lon;
sub_rectangle[1].min_lat = sub_rectangle[2].min_lat = outer_boundry.min_lat;
sub_rectangle[2].max_lon = sub_rectangle[3].max_lon = outer_boundry.max_lon;
sub_rectangle[3].max_lat = sub_rectangle[0].max_lat = outer_boundry.max_lat;

sub_rectangle[0].min_lat = sub_rectangle[1].max_lat = inner_boundry.min_lat;
sub_rectangle[1].max_lon = sub_rectangle[2].min_lon = inner_boundry.max_lon;
sub_rectangle[2].max_lat = sub_rectangle[3].min_lat = inner_boundry.max_lat;
sub_rectangle[3].min_lon = sub_rectangle[0].max_lon = inner_boundry.min_lon;
}


void Delaunay_grid_decomposition::rectangles_to_halo(Boundry sub_rectangle[4], Boundry* outer_boundry)
{
outer_boundry->min_lon = sub_rectangle[0].min_lon;
outer_boundry->min_lat = sub_rectangle[1].min_lat;
outer_boundry->max_lon = sub_rectangle[2].max_lon;
outer_boundry->max_lat = sub_rectangle[3].max_lat;
}


static inline bool is_in_region(double x, double y, Boundry region)
{
return x >= region.min_lon && x < region.max_lon && y >= region.min_lat && y < region.max_lat;
}


int Delaunay_grid_decomposition::classify_points(double *coord[2], int *index, bool *mask, int num, Boundry region, int start)
{
int j = start;
for (int i = start; i < num; i++)
if (is_in_region(coord[PDLN_LON][i], coord[PDLN_LAT][i], region)) {
std::swap(coord[PDLN_LON][i], coord[PDLN_LON][j]);
std::swap(coord[PDLN_LAT][i], coord[PDLN_LAT][j]);
std::swap(index[i], index[j]);
if (mask)
std::swap(mask[i], mask[j]);
j++;
}
return j - start;
}


#define PDLN_SAVE_N (0)
#define PDLN_SAVE_L (1)
#define PDLN_SAVE_R (2)
void Delaunay_grid_decomposition::adjust_subrectangle(double l, double r, double *coord[2], int *idx, bool *mask,
int offset, int num, Boundry *bound, int linetype,
int save_option, int* offset_picked, int* num_picked)
{
int c_num_points[2];
double boundry_values[4];
Midline midline;

midline.value = 0;
midline.type = linetype;

PDASSERT(l > 0);
PDASSERT(r > 0);

if (bound->min_lon == bound->max_lon || bound->min_lat == bound->max_lat) {
*offset_picked = *num_picked = 0;
return;
}

boundry_values[PDLN_LON] = bound->min_lon;
boundry_values[PDLN_LAT] = bound->min_lat;
boundry_values[PDLN_LON+2] = bound->max_lon;
boundry_values[PDLN_LAT+2] = bound->max_lat;

PDASSERT(boundry_values[PDLN_LON] != boundry_values[PDLN_LON+2]);
PDASSERT(boundry_values[PDLN_LAT] < boundry_values[PDLN_LAT+2]);

c_num_points[0] = c_num_points[1] = 0;
int ret = Search_tree_node::divide_points(coord, idx, mask, l, r, boundry_values[linetype], boundry_values[linetype+2], offset, num, 0, &midline, c_num_points, 0, 0, PDLN_DOUBLE_INVALID_VALUE, 0);

if (ret)
Search_tree_node::sort_by_line_internal(coord, idx, mask, &midline, offset, num, &c_num_points[0], &c_num_points[1]);

PDASSERT(c_num_points[0] >= 0);
PDASSERT(c_num_points[1] >= 0);

if (midline.value == PDLN_DOUBLE_INVALID_VALUE) {
*offset_picked = offset;
*num_picked = num;
} else if (save_option == PDLN_SAVE_L) {
*offset_picked = offset;
*num_picked = c_num_points[0];
if (linetype == PDLN_LON)
bound->max_lon = midline.value;
else
bound->max_lat = midline.value;
} else {
*offset_picked = offset + c_num_points[0];
*num_picked = c_num_points[1];
if (linetype == PDLN_LON)
bound->min_lon = midline.value;
else
bound->min_lat = midline.value;
}
return;
}


vector<Search_tree_node*> Delaunay_grid_decomposition::adjust_expanding_boundry(const Boundry* inner, Boundry* outer, double quota[4],
double *output_coord[2], int *output_index, bool* output_mask,
bool go_on[4], int *total_num, bool *tree_need_extension)
{
vector<Search_tree_node*> leaf_nodes_found;
*total_num = 0;
*tree_need_extension = false;

if (*inner != *outer)
*tree_need_extension = search_down_for_points_in_halo(search_tree_root, inner, outer, &leaf_nodes_found, output_coord, output_index, output_mask, total_num);

if (*tree_need_extension)
return leaf_nodes_found;

PDASSERT(!have_redundent_points(output_coord[PDLN_LON], output_coord[PDLN_LAT], *total_num));
Boundry sub_rectangles[4];
int offset_picked[4];
int num_points_picked[4];
halo_to_rectangles(*inner, *outer, sub_rectangles);
int sorted = 0;
for (int i = 0; i < 4; i++) {
int num = classify_points(output_coord, output_index, output_mask, *total_num, sub_rectangles[i], sorted);
offset_picked[i] = sorted;
num_points_picked[i] = num;
if (num <= 0 && sub_rectangles[i].min_lat != sub_rectangles[i].max_lat && sub_rectangles[i].min_lon != sub_rectangles[i].max_lon)
go_on[i] = true;
if (num > quota[i]) {
double l_num = i<2 ? num - quota[i] : quota[i];
double r_num = i<2 ? quota[i] : num - quota[i];
int linetype = i%2 ? PDLN_LAT : PDLN_LON;
int savetype = i<2 ? PDLN_SAVE_R : PDLN_SAVE_L;
adjust_subrectangle(l_num, r_num, output_coord, output_index, output_mask, sorted, num,
&sub_rectangles[i], linetype, savetype, &offset_picked[i], &num_points_picked[i]);
if (num_points_picked[i] <= 0)
go_on[i] = true;
}
sorted += num;
log(LOG_DEBUG, "%d Edge found %d new points\n", i, num_points_picked[i]);
}


rectangles_to_halo(sub_rectangles, outer);
PDASSERT(!have_redundent_points(output_coord[PDLN_LON], output_coord[PDLN_LAT], *total_num));
*total_num = move_together(output_coord, output_index, output_mask, offset_picked, num_points_picked, *outer);
PDASSERT(!have_redundent_points(output_coord[PDLN_LON], output_coord[PDLN_LAT], *total_num));

if (outer->min_lat == search_tree_root->kernel_boundry->min_lat) go_on[1] = false;
if (outer->max_lat == search_tree_root->kernel_boundry->max_lat) go_on[3] = false;
if (!is_cyclic && outer->min_lon == search_tree_root->kernel_boundry->min_lon) go_on[0] = false;
if (!is_cyclic && outer->max_lon == search_tree_root->kernel_boundry->max_lon) go_on[2] = false;
return leaf_nodes_found;
}


int Delaunay_grid_decomposition::move_together(double *coord[2], int *index, bool *mask, int offset[4], int num[4], Boundry bound) 
{
int total_num = 0;

for (int i = 0; i < 4; i++)
for (int j = offset[i]; j < offset[i]+num[i]; j++)
if (is_in_region(coord[PDLN_LON][j], coord[PDLN_LAT][j], bound)) {
coord[0][total_num] = coord[0][j];
coord[1][total_num] = coord[1][j];
index[total_num] = index[j];
if (mask)
mask[total_num] = mask[j];
total_num++;
}
PDASSERT(total_num <= num[0] + num[1] + num[2] + num[3]);

return total_num;
}


void extend_search_tree(Delaunay_grid_decomposition *decomp, Search_tree_node *node, const Boundry* outer_boundarys, int num_boundarys, int min_points_per_chunk)
{
double*     c_points_coord[4];
int*        c_points_index[2];
bool*       c_points_mask[2];
int         c_num_points[2];
Boundry     c_boundry[2];
int         c_ids_start[2];
int         c_ids_end[2];
int*        c_intervals[2];
int         c_num_intervals[2];

PDASSERT(node->ids_size() > 0);

if(node->ids_size() == 1) {
#pragma omp critical
decomp->all_leaf_nodes.push_back(node);
return;
}

if(node->children[0] == NULL && node->children[2] == NULL) {
node->decompose_by_processing_units_number(decomp->workloads, c_points_coord, c_points_index, c_points_mask,
c_num_points, c_boundry, c_ids_start, c_ids_end, PDLN_DECOMPOSE_COMMON_MODE,
c_intervals, c_num_intervals, min_points_per_chunk);
PDASSERT(c_ids_end[0] - c_ids_start[0] > 0);

node->children[0] = decomp->alloc_search_tree_node(node, c_points_coord,   c_points_index[0], c_points_mask[0], c_num_points[0],
c_boundry[0], c_ids_start[0], c_ids_end[0], PDLN_NODE_TYPE_COMMON);

node->children[2] = decomp->alloc_search_tree_node(node, c_points_coord+2, c_points_index[1], c_points_mask[1], c_num_points[1],
c_boundry[1], c_ids_start[1], c_ids_end[1], PDLN_NODE_TYPE_COMMON);
node->children[0]->set_groups(c_intervals[0], c_num_intervals[0]);
node->children[2]->set_groups(c_intervals[1], c_num_intervals[1]);

PDASSERT(node->children[0]->ids_size() > 0);
PDASSERT(node->children[2]->ids_size() > 0);

}

for (int i = 0; i < 3; i ++) {
if(node->children[i] == NULL)
continue;

for (int j = 0; j < num_boundarys; j++) {
const Boundry& region = outer_boundarys[j];

if (region.min_lon == 0 && region.max_lon == 0)
continue;

if(decomp->do_two_regions_overlap(region, *node->children[i]->kernel_boundry) ||
decomp->do_two_regions_overlap(Boundry(region.min_lon + 360.0, region.max_lon + 360.0, region.min_lat, region.max_lat), *node->children[i]->kernel_boundry) ||
decomp->do_two_regions_overlap(Boundry(region.min_lon - 360.0, region.max_lon - 360.0, region.min_lat, region.max_lat), *node->children[i]->kernel_boundry)) {
if (node->children[i]->num_kernel_points < 2000)
extend_search_tree(decomp, node->children[i], outer_boundarys, num_boundarys, min_points_per_chunk);
else {
#pragma omp task
extend_search_tree(decomp, node->children[i], outer_boundarys, num_boundarys, min_points_per_chunk);
}
break;
}
}
}
}


bool Delaunay_grid_decomposition::search_down_for_points_in_halo(Search_tree_node *node, const Boundry *inner_boundary,
const Boundry *outer_boundary, vector<Search_tree_node*> *leaf_nodes_found,
double *output_coord[2], int *output_index, bool *output_mask, int *num_found)
{
PDASSERT(node->ids_size() > 0);

Boundry region = *outer_boundary;
if(node->ids_size() == 1) {
if (node->num_kernel_points == 0)
return false;

if(do_two_regions_overlap(region, *node->kernel_boundry) ||
do_two_regions_overlap(Boundry(region.min_lon + 360.0, region.max_lon + 360.0, region.min_lat, region.max_lat), *node->kernel_boundry) ||
do_two_regions_overlap(Boundry(region.min_lon - 360.0, region.max_lon - 360.0, region.min_lat, region.max_lat), *node->kernel_boundry)) {
node->search_points_in_halo(inner_boundary, outer_boundary, output_coord, output_index, output_mask, num_found);
#pragma omp critical
(*leaf_nodes_found).push_back(node);
}
return false;
}

if(node->children[0] == NULL && node->children[2] == NULL) {
return true;
}

bool tree_need_extension = false;
for(int i = 0; i < 3; i ++)
if(node->children[i] != NULL) {
if(do_two_regions_overlap(region, *node->children[i]->kernel_boundry) ||
do_two_regions_overlap(Boundry(region.min_lon + 360.0, region.max_lon + 360.0, region.min_lat, region.max_lat), *node->children[i]->kernel_boundry) ||
do_two_regions_overlap(Boundry(region.min_lon - 360.0, region.max_lon - 360.0, region.min_lat, region.max_lat), *node->children[i]->kernel_boundry)) {
tree_need_extension = tree_need_extension || search_down_for_points_in_halo(node->children[i], inner_boundary, outer_boundary,
leaf_nodes_found, output_coord, output_index,
output_mask, num_found);
}
}

return tree_need_extension;
}


void Search_tree_node::search_points_in_halo_internal(const Boundry *inner_boundary, const Boundry *outer_boundary,
double *const coord[2], const int *idx, const bool *mask, int num_points,
double *output_coord[2], int *output_index, bool *output_mask, int *num_found)
{
Boundry l_inner = *inner_boundary;
Boundry l_outer = *outer_boundary;
Boundry r_inner = *inner_boundary;
Boundry r_outer = *outer_boundary;
l_inner.min_lon -= 360.0;
l_inner.max_lon -= 360.0;
l_outer.min_lon -= 360.0;
l_outer.max_lon -= 360.0;
r_inner.min_lon += 360.0;
r_inner.max_lon += 360.0;
r_outer.min_lon += 360.0;
r_outer.max_lon += 360.0;

int count = *num_found;

for(int j = 0; j < num_points; j++) {
if (is_coordinate_in_halo(coord[PDLN_LON][j], coord[PDLN_LAT][j], inner_boundary, outer_boundary)) {
output_coord[PDLN_LON][count] = coord[PDLN_LON][j];
output_coord[PDLN_LAT][count] = coord[PDLN_LAT][j];
output_index[count] = idx[j];
if (mask)
output_mask[count] = mask[j];
count++;
continue;
}
if (is_coordinate_in_halo(coord[PDLN_LON][j], coord[PDLN_LAT][j], &l_inner, &l_outer)) {
output_coord[PDLN_LON][count] = coord[PDLN_LON][j] + 360.0;
output_coord[PDLN_LAT][count] = coord[PDLN_LAT][j];
output_index[count] = idx[j];
if (mask)
output_mask[count] = mask[j];
count++;
continue;
}
if (is_coordinate_in_halo(coord[PDLN_LON][j], coord[PDLN_LAT][j], &r_inner, &r_outer)) {
output_coord[PDLN_LON][count] = coord[PDLN_LON][j] - 360.0;
output_coord[PDLN_LAT][count] = coord[PDLN_LAT][j];
output_index[count] = idx[j];
if (mask)
output_mask[count] = mask[j];
count++;
continue;
}
}

*num_found = count;
}


void Search_tree_node::search_points_in_halo(const Boundry *inner_boundary, const Boundry *outer_boundary,
double *output_coord[2], int *output_index, bool *output_mask, int *num_found)
{
if(*kernel_boundry <= *inner_boundary)
return;
search_points_in_halo_internal(inner_boundary, outer_boundary, kernel_coord, kernel_index, kernel_mask, num_kernel_points, output_coord, output_index, output_mask, num_found);
}


inline bool Search_tree_node::is_coordinate_in_halo(double x, double y, const Boundry *inner, const Boundry *outer)
{
return !(x < inner->max_lon && x >= inner->min_lon && y < inner->max_lat && y >= inner->min_lat) &&
(x < outer->max_lon && x >= outer->min_lon && y < outer->max_lat && y >= outer->min_lat);
}


void Search_tree_node::init_num_neighbors_on_boundry(int n)
{
num_neighbors_on_boundry[0] = num_neighbors_on_boundry[1] = num_neighbors_on_boundry[2] = num_neighbors_on_boundry[3] = n;
}


void Search_tree_node::reduce_num_neighbors_on_boundry(unsigned long type)
{
switch(type) {
case PDLN_BOUNDRY_TYPE_U: num_neighbors_on_boundry[PDLN_UP]--; break;
case PDLN_BOUNDRY_TYPE_D: num_neighbors_on_boundry[PDLN_DOWN]--; break;
case PDLN_BOUNDRY_TYPE_L: num_neighbors_on_boundry[PDLN_LEFT]--; break;
case PDLN_BOUNDRY_TYPE_R: num_neighbors_on_boundry[PDLN_RIGHT]--; break;
case PDLN_BOUNDRY_TYPE_LR:
num_neighbors_on_boundry[PDLN_LEFT]--;
num_neighbors_on_boundry[PDLN_RIGHT]--;
break;
case PDLN_BOUNDRY_TYPE_NON: break;
default: PDASSERT(false);
}
}


void Search_tree_node::clear_expanding_count(unsigned long type)
{
switch(type) {
case PDLN_BOUNDRY_TYPE_U: edge_expanding_count[PDLN_UP] = 0; break;
case PDLN_BOUNDRY_TYPE_D: edge_expanding_count[PDLN_DOWN] = 0; break;
case PDLN_BOUNDRY_TYPE_L: edge_expanding_count[PDLN_LEFT] = 0; break;
case PDLN_BOUNDRY_TYPE_R: edge_expanding_count[PDLN_RIGHT] = 0; break;
case PDLN_BOUNDRY_TYPE_LR:
edge_expanding_count[PDLN_LEFT] = 0;
edge_expanding_count[PDLN_RIGHT] = 0;
break;
case PDLN_BOUNDRY_TYPE_NON: break;
default: PDASSERT(false);
}
}


bool Delaunay_grid_decomposition::is_polar_node(Search_tree_node *node) const
{
return node != NULL && node->node_type != PDLN_NODE_TYPE_COMMON;
}


bool node_ptr_comp(Search_tree_node* a, Search_tree_node* b)
{
return a->region_id < b->region_id;
}


void Delaunay_grid_decomposition::set_binding_relationship()
{
if (local_leaf_nodes.size() == 0)
return;

std::sort(local_leaf_nodes.begin(), local_leaf_nodes.end(), node_ptr_comp);

int old_unit_id = regionID_to_unitID[local_leaf_nodes[0]->region_id];
for (unsigned i = 1; i < local_leaf_nodes.size(); i++) {
if (local_leaf_nodes[i]->region_id == old_unit_id) {
local_leaf_nodes[i-1]->bind_with = i;
local_leaf_nodes[i]->is_bind = true;
} else
old_unit_id = local_leaf_nodes[i]->region_id;
}
}


int Delaunay_grid_decomposition::generate_trianglulation_for_local_decomp()
{
timeval start, end;
bool* is_local_leaf_node_finished = new bool[local_leaf_nodes.size()]();
unsigned long** local_leaf_checksums   = new unsigned long*[local_leaf_nodes.size()];
unsigned long** remote_leaf_checksums  = new unsigned long*[local_leaf_nodes.size()];
Boundry* outer_bound = new Boundry[local_leaf_nodes.size()];

#ifdef DEBUG
for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
PDASSERT(is_local_leaf_node_finished[i] == false);
}
#endif


set_binding_relationship();

int max_neighbors = std::max(processing_info->get_num_total_processing_units(), 4); 
for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
local_leaf_checksums[i] = new unsigned long[max_neighbors];
remote_leaf_checksums[i] = new unsigned long[max_neighbors];
}

vector<MPI_Request*> *waiting_lists = new vector<MPI_Request*> [local_leaf_nodes.size()];

int iter = 0;
unsigned global_finish = 0;
double expanding_ratio = PDLN_DEFAULT_EXPANGDING_RATIO;
for(unsigned i = 0; i < local_leaf_nodes.size(); i++)
local_leaf_nodes[i]->init_num_neighbors_on_boundry(1);

while(iter < 50) {
log(LOG_DEBUG, "local triangulation loop %d\n", iter);
MPI_Barrier(processing_info->get_mpi_comm());
gettimeofday(&start, NULL);

log(LOG_DEBUG, "expanding\n");
int expanding_fail = 0;
volatile int goon;
do {

log(LOG_DEBUG, "extending search tree\n");

for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
if(!is_local_leaf_node_finished[i]) {
Search_tree_node* tree_node = local_leaf_nodes[i];

outer_bound[i] = tree_node->expand();
} else {
outer_bound[i].min_lon = 0;
outer_bound[i].max_lon = 0;
}
}


if (local_leaf_nodes.size() > 0) {
#pragma omp parallel
{
#pragma omp single
{
extend_search_tree(this, search_tree_root, outer_bound, local_leaf_nodes.size(), min_points_per_chunk);
}
}
}


goon = 0;
#pragma omp parallel for
for(unsigned i = 0; i < local_leaf_nodes.size(); i++)
if(!is_local_leaf_node_finished[i]) {
int local_ret = expand_tree_node_boundry(local_leaf_nodes[i], expanding_ratio);
#pragma omp critical
{
if (local_ret == 1)
expanding_fail = 1;
else if (local_ret == -1)
goon = 1;
}
}
} while (!expanding_fail && goon);

int global_expanding_fail = 0;
MPI_Allreduce((int*)(&expanding_fail), &global_expanding_fail, 1, MPI_UNSIGNED, MPI_BOR, processing_info->get_mpi_comm());

gettimeofday(&end, NULL);
if (!global_finish) {
time_expand += (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

if(global_expanding_fail) {
log(LOG_ERROR, "Failed in expanding\n");
global_finish = false;
break;
}

log(LOG_DEBUG, "updating projection\n");
MPI_Barrier(processing_info->get_mpi_comm());
gettimeofday(&start, NULL);

if (local_leaf_nodes.size() > 0)
if (is_polar_node(search_tree_root->children[0]) || is_polar_node(search_tree_root->children[2])) {
#pragma omp parallel for
for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
if (local_leaf_nodes[i]->is_bind)
continue;
for(unsigned cur = i;;) {
if (!is_local_leaf_node_finished[cur])
local_leaf_nodes[cur]->project_grid();
cur = local_leaf_nodes[cur]->bind_with;
if (cur == 0) break;
}
}
}

gettimeofday(&end, NULL);
if (!global_finish) {
time_expand += (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

log(LOG_DEBUG, "updating triangulation\n");
MPI_Barrier(processing_info->get_mpi_comm());
gettimeofday(&start, NULL);
#pragma omp parallel for
for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
if (local_leaf_nodes[i]->is_bind)
continue;
for(unsigned cur = i;;) {
if (!is_local_leaf_node_finished[cur])
local_leaf_nodes[cur]->generate_local_triangulation(is_cyclic, num_points - num_fence_points, num_fence_points, num_points > 1e6);
cur = local_leaf_nodes[cur]->bind_with;
if (cur == 0) break;
}
}

gettimeofday(&end, NULL);

if (!global_finish || iter == 0) {
time_local_tri += (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

log(LOG_DEBUG, "verifying consistency\n");
MPI_Barrier(processing_info->get_mpi_comm());
gettimeofday(&start, NULL);
#pragma omp parallel for
for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
#ifdef DEBUG
PDASSERT(local_leaf_nodes[i]->neighbors.size() <= max_neighbors);
#endif
send_recv_checksums_with_neighbors(local_leaf_nodes[i], local_leaf_checksums[i], remote_leaf_checksums[i], &waiting_lists[i], iter);
}

processing_info->do_thread_send_recv();

for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
for(unsigned j = 0; j < waiting_lists[i].size(); j++) {
MPI_Wait(waiting_lists[i][j], MPI_STATUS_IGNORE);
delete waiting_lists[i][j];
}
waiting_lists[i].clear();
}

#pragma omp parallel for
for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
is_local_leaf_node_finished[i] = are_checksums_identical(local_leaf_nodes[i], local_leaf_checksums[i], remote_leaf_checksums[i]);
}
gettimeofday(&end, NULL);
if (!global_finish || iter == 0) {
time_consisty_check += (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

unsigned local_finish = 1;
for(unsigned i = 0; i < local_leaf_nodes.size(); i++)
if(!is_local_leaf_node_finished[i])
local_finish = 0;

MPI_Allreduce(&local_finish, &global_finish, 1, MPI_UNSIGNED, MPI_BAND, processing_info->get_mpi_comm());

if(global_finish)
break;

expanding_ratio += 0.1;
iter++;
}

delete[] is_local_leaf_node_finished;

for(unsigned i = 0; i < local_leaf_nodes.size(); i++) {
delete[] local_leaf_checksums[i];
delete[] remote_leaf_checksums[i];
local_leaf_checksums[i] = NULL;
remote_leaf_checksums[i] = NULL;
}

delete[] local_leaf_checksums;
delete[] remote_leaf_checksums;

delete[] waiting_lists;
delete[] outer_bound;

if(global_finish)
return 0;
else
return 1;
}


void Delaunay_grid_decomposition::print_tree_node_info_recursively(Search_tree_node *node)
{
if(node->ids_size() == 1){
printf("[ID%d]x[ST-Info] LEAF %p\n", local_leaf_nodes[0]->region_id, node);
return;
}

printf("[ID%d]x[ST-Info] %p: %p, %p, %p\n", local_leaf_nodes[0]->region_id, node, node->children[0], node->children[1], node->children[2]);
for(int i = 0; i < 3; i ++)
if(node->children[i])
print_tree_node_info_recursively(node->children[i]);
}


void Delaunay_grid_decomposition::print_whole_search_tree_info()
{
printf("[ID%d]x[ST-Info] ROOT %p\n", local_leaf_nodes[0]->region_id, search_tree_root);
print_tree_node_info_recursively(search_tree_root);
}


#ifdef OPENCV
void Delaunay_grid_decomposition::plot_grid_decomposition(const char *filename)
{
if (processing_info->get_local_process_id() == 0) {
plot_points_into_file(filename, search_tree_root->kernel_coord[PDLN_LON], search_tree_root->kernel_coord[PDLN_LAT], mask, search_tree_root->num_kernel_points, PDLN_PLOT_GLOBAL);
for(unsigned i = 0; i < all_leaf_nodes.size(); i++) {
Boundry b = *all_leaf_nodes[i]->kernel_boundry;
plot_rectangle_into_file(filename, b.min_lon, b.max_lon, b.min_lat, b.max_lat, PDLN_PLOT_COLOR_RED, PDLN_PLOT_FILEMODE_APPEND);
char number[8];
snprintf(number, 8, "%d", all_leaf_nodes[i]->region_id);
plot_text_into_file(filename, number, b.min_lon, b.max_lon, b.min_lat, b.max_lat, PDLN_PLOT_COLOR_RED);
}
}
}


void Delaunay_grid_decomposition::plot_local_triangles(const char *perfix)
{
for(unsigned int i = 0; i < local_leaf_nodes.size(); i++) {
char filename[64];
snprintf(filename, 64, "%s_%d.png", perfix, local_leaf_nodes[i]->region_id);
local_leaf_nodes[i]->triangulation->plot_into_file(filename, local_leaf_nodes[i]->kernel_boundry->min_lon,
local_leaf_nodes[i]->kernel_boundry->max_lon,
local_leaf_nodes[i]->kernel_boundry->min_lat,
local_leaf_nodes[i]->kernel_boundry->max_lat);
}
}
#endif


static inline bool on_a_line(Triangle_inline* t)
{
return (float_eq(t->v[0].y, t->v[1].y) && float_eq(t->v[1].y, t->v[2].y)) ||
(float_eq(t->v[0].x, t->v[1].x) && float_eq(t->v[1].x, t->v[2].x));
}


void delete_redundent_triangles(Triangle_inline *&all_triangles, int &num)
{
std::tr1::unordered_map<Triangle_inline, std::list<int> > hash_table;
std::tr1::unordered_map<Triangle_inline, std::list<int> >::iterator it_hash;

if(num == 0)
return;

Triangle_inline *tmp_triangles = new Triangle_inline[num];

int count = 0;
for(int i = 0; i < num; i++) {
if (on_a_line(&all_triangles[i]))
continue;

it_hash = hash_table.find(all_triangles[i]);
if(it_hash != hash_table.end()) {
bool same = false;
for(std::list<int>::iterator it_list = it_hash->second.begin(); it_list != it_hash->second.end(); it_list ++)
if(all_triangles[*it_list] == all_triangles[i]) {
same = true;
break;
}
if(same)
continue;
else {
it_hash->second.push_back(i);
tmp_triangles[count++] = all_triangles[i];
}
}
else {
hash_table[all_triangles[i]].push_back(i);
tmp_triangles[count++] = all_triangles[i];
}
}

delete[] all_triangles;
all_triangles = tmp_triangles;
num = count;

return;
}


void Delaunay_grid_decomposition::save_unique_triangles_into_file(Triangle_inline *&triangles, int num_triangles, bool sort)
{
int num_different_triangles;
if (sort) {
sort_points_in_triangle(triangles, num_triangles);
sort_triangles(triangles, num_triangles);
int i, j;
for(i = 0, j = 1; j < num_triangles; j++) {
if((triangles[i].v[0].id == triangles[j].v[0].id &&
triangles[i].v[1].id == triangles[j].v[1].id &&
triangles[i].v[2].id == triangles[j].v[2].id) ||
on_a_line(&triangles[j])) {
continue;
}
else
triangles[++i] = triangles[j];
}
num_different_triangles = i + 1;
} else {
delete_redundent_triangles(triangles, num_triangles);
num_different_triangles = num_triangles;
}

#ifndef TIME_PERF 
char file_fmt[] = "log/global_triangles_%d";
char filename[64];
snprintf(filename, 64, file_fmt, processing_info->get_num_total_processing_units());
FILE *fp = fopen(filename, "w");
for(int i = 0; i < num_different_triangles; i++)
fprintf(fp, "%d, %d, %d\n", triangles[i].v[0].id, triangles[i].v[1].id, triangles[i].v[2].id);
fclose(fp);
#endif



#ifdef OPENCV
char file_fmt3[] = "log/image_global_triangles_%d";
char filename3[64];
snprintf(filename3, 64, file_fmt3, processing_info->get_num_total_processing_units());
plot_triangles_into_file(filename3, triangles, num_different_triangles, true);
#endif
}


#define PDLN_MERGE_TAG_MASK 0x0200
void Delaunay_grid_decomposition::merge_all_triangles(bool sort)
{


int local_buf_len = 0;
for(unsigned i = 0; i < local_leaf_nodes.size(); i++)
local_buf_len += (local_leaf_nodes[i]->num_kernel_points + local_leaf_nodes[i]->num_expand_points) * 3 * 2;

Triangle_inline* local_triangles = new Triangle_inline[local_buf_len];
int num_local_triangles = 0;
int num_triangles = 0;
for(unsigned int i = 0; i < local_leaf_nodes.size(); i++) {
local_leaf_nodes[i]->triangulation->get_triangles_in_region(local_leaf_nodes[i]->kernel_boundry->min_lon, local_leaf_nodes[i]->kernel_boundry->max_lon,
local_leaf_nodes[i]->kernel_boundry->min_lat, local_leaf_nodes[i]->kernel_boundry->max_lat,
local_triangles + num_local_triangles, &num_triangles, local_buf_len - num_local_triangles); 
num_local_triangles += num_triangles;
}

MPI_Barrier(processing_info->get_mpi_comm());

if(processing_info->get_local_process_id() == 0) {
int *num_remote_triangles = new int[processing_info->get_num_total_processes()];
int remote_buf_len = 0;
MPI_Status status;

for(int i = 1; i < processing_info->get_num_total_processes(); i++)
MPI_Recv(&num_remote_triangles[i], 1, MPI_INT, i, PDLN_MERGE_TAG_MASK, processing_info->get_mpi_comm(), &status);
for(int i = 1; i < processing_info->get_num_total_processes(); i++) {
remote_buf_len += num_remote_triangles[i];
}
Triangle_inline *remote_triangles = new Triangle_inline[remote_buf_len + num_local_triangles];

int count = 0;
for(int i = 1; i < processing_info->get_num_total_processes(); i++) {
MPI_Recv(remote_triangles + count, num_remote_triangles[i] * sizeof(Triangle_inline), MPI_CHAR, i, PDLN_MERGE_TAG_MASK, processing_info->get_mpi_comm(), &status);
int tmp_count;
MPI_Get_count(&status, MPI_CHAR, &tmp_count);


#ifdef DEBUG
PDASSERT(tmp_count % sizeof(Triangle_inline) == 0);
#endif
count += tmp_count / sizeof(Triangle_inline);
}
PDASSERT(count == remote_buf_len);
memcpy(remote_triangles + remote_buf_len, local_triangles, num_local_triangles * sizeof(Triangle_inline));
save_unique_triangles_into_file(remote_triangles, remote_buf_len + num_local_triangles, sort);
delete[] remote_triangles;
delete[] num_remote_triangles;
}
else {
MPI_Send(&num_local_triangles, 1, MPI_INT, 0, PDLN_MERGE_TAG_MASK, processing_info->get_mpi_comm());
MPI_Send(local_triangles, num_local_triangles * sizeof(Triangle_inline), MPI_CHAR, 0, PDLN_MERGE_TAG_MASK, processing_info->get_mpi_comm());
}

delete[] local_triangles;
}


static double fRand(double fMin, double fMax)
{
double f = (double)rand() / RAND_MAX;
return fMin + f * (fMax - fMin);
}


void Grid_info_manager::gen_basic_grid()
{
int size = 300;

num_points = size * size;
coord_values[0] = new double[num_points]();
coord_values[1] = new double[num_points]();
srand(0);
for(int i = 0; i < size; i++)
for(int j = 0; j < size; j++) {
coord_values[0][i * size + j] =  0.0  + 360.0 * j / size;
coord_values[1][i * size + j] = -89.0 + 178.0 * i / size;
} 

coord_values[0][0] = 0.0;
coord_values[1][0] = -90.0;
coord_values[0][299] = 0.0;
coord_values[1][299] = 90.0;

}


#ifdef NETCDF
void Grid_info_manager::gen_three_polar_grid()
{
int num_dims;
int *dim_size_ptr;
int field_size;
int field_size2;
void *coord_buf0, *coord_buf1;
bool squeeze = true;

read_file_field_as_float("gridfile/three_polars_grid.nc", "nav_lon", &coord_buf0, &num_dims, &dim_size_ptr, &field_size);
delete dim_size_ptr;
read_file_field_as_float("gridfile/three_polars_grid.nc", "nav_lat", &coord_buf1, &num_dims, &dim_size_ptr, &field_size2);
delete dim_size_ptr;
PDASSERT(field_size == field_size2);
num_points = field_size;
coord_values[PDLN_LON] = (double*)coord_buf0;
coord_values[PDLN_LAT] = (double*)coord_buf1;

for(int i = 0; i < num_points; i++)
if(coord_values[PDLN_LON][i] < 0.0)
coord_values[PDLN_LON][i] += 360.0;

for(int i = 0; i < num_points; i++)
if(float_eq(coord_values[PDLN_LON][i], 360.0)) {
coord_values[PDLN_LON][i] = 0.0;
}

delete_redundent_points(coord_values[PDLN_LON], coord_values[PDLN_LAT], num_points);
PDASSERT(have_redundent_points(coord_values[PDLN_LON], coord_values[PDLN_LAT], num_points) == false);

if(squeeze) {
for(int i = 0; i < num_points/100; i++) {
coord_values[PDLN_LON][i] = coord_values[PDLN_LON][i*100];
coord_values[PDLN_LAT][i] = coord_values[PDLN_LAT][i*100];
}
num_points /= 100;
}

min_lon =   0.0;
max_lon = 360.0;
min_lat = -80.0;
max_lat =  90.0;
}

void Grid_info_manager::gen_latlon_grid()
{
int num_dims;
int *dim_size_ptr;
int field_size;
int field_size2;
void *coord_buf0, *coord_buf1;

read_file_field_as_double("gridfile/lonlat.nc", "lon", &coord_buf0, &num_dims, &dim_size_ptr, &field_size);
delete dim_size_ptr;
read_file_field_as_double("gridfile/lonlat.nc", "lat", &coord_buf1, &num_dims, &dim_size_ptr, &field_size2);
delete dim_size_ptr;

num_points = field_size*field_size2;
coord_values[PDLN_LON] = new double [num_points];
coord_values[PDLN_LAT] = new double [num_points];

int count = 0;
for(int i = 0; i < field_size; i ++)
for(int j = 0; j < field_size2; j++) {
coord_values[PDLN_LON][count] = ((double*)coord_buf0)[i];
coord_values[PDLN_LAT][count++] = ((double*)coord_buf1)[j];
}

PDASSERT(count == num_points);
PDASSERT(!have_redundent_points(coord_values[PDLN_LON], coord_values[PDLN_LAT], num_points));

min_lon =   1.0;
max_lon = 360.0;
min_lat = -89.0;
max_lat =  89.0;
}


void Grid_info_manager::gen_latlon_90_grid()
{
int num_dims;
int *dim_size_ptr;
int field_size;
int field_size2;
void *coord_buf0, *coord_buf1;

read_file_field_as_double("gridfile/lonlat_90.nc", "lon", &coord_buf0, &num_dims, &dim_size_ptr, &field_size);
delete dim_size_ptr;
read_file_field_as_double("gridfile/lonlat_90.nc", "lat", &coord_buf1, &num_dims, &dim_size_ptr, &field_size2);
delete dim_size_ptr;

num_points = field_size*field_size2;
coord_values[PDLN_LON] = new double [num_points];
coord_values[PDLN_LAT] = new double [num_points];

int count = 0;
for(int j = field_size2-1; j >= 0; j--)
for(int i = 0; i < field_size; i ++) {
coord_values[PDLN_LON][count] = ((double*)coord_buf0)[i];
coord_values[PDLN_LAT][count++] = ((double*)coord_buf1)[j];
}

PDASSERT(count == num_points);
PDASSERT(!have_redundent_points(coord_values[PDLN_LON], coord_values[PDLN_LAT], num_points));

min_lon =   0.0;
max_lon = 360.0;
min_lat = -90.0;
max_lat =  90.0;
}
#endif


Grid_info_manager::Grid_info_manager()
: num_points(0)
, min_lon(0.)
, max_lon(0.)
, min_lat(0.)
, max_lat(0.)
, is_cyclic(false)
, disabling_method(NO_DISABLED_POINTS)
, disabling_num(0)
, disabling_data(NULL)
{
coord_values[0] = coord_values[1] = NULL;
}


Grid_info_manager::~Grid_info_manager()
{
delete coord_values[0];
delete coord_values[1];
}


bool Grid_info_manager::read_grid_from_text(const char filename[])
{
FILE* fp = fopen(filename, "r");

if (!fp)
return false;

fscanf(fp, "%d", &num_points);

if (num_points < 1) {
fprintf(stderr, "Invalid points number\n");
goto fail;
}

fscanf(fp, "%lf %lf %lf %lf", &min_lon, &max_lon, &min_lat, &max_lat);

if (max_lat < -90 || max_lat > 90 || min_lat < -90 || min_lat > 90 ||
(min_lat >= max_lat || min_lon >= max_lon || max_lon - min_lon > 360)) {
fprintf(stderr, "Invalid boundary value\n");
goto fail;
}

coord_values[PDLN_LON] = new double [num_points];
coord_values[PDLN_LAT] = new double [num_points];

for(int i = 0; i < num_points; i ++)
fscanf(fp, "%lf %lf\n", &coord_values[PDLN_LON][i], &coord_values[PDLN_LAT][i]);

if (have_redundent_points(coord_values[PDLN_LON], coord_values[PDLN_LAT], num_points)) {
fprintf(stderr, "Redundent points found\n");
goto fail;
}

is_cyclic = float_eq(max_lon - min_lon, 360);

char disable_method[64];
if(fread(disable_method, 1, 23, fp)) {
if (strncmp(disable_method, "DISABLE_POINTS_BY_INDEX", 23) == 0) {
fscanf(fp, "%d", &disabling_num);

int* tmp_buf = new int[disabling_num];
for (int i = 0; i < disabling_num; i++)
fscanf(fp, "%d", &tmp_buf[i]);

disabling_data = (void*) tmp_buf;
} else if (strncmp(disable_method, "DISABLE_POINTS_BY_RANGE", 23) == 0) {
fscanf(fp, "%d", &disabling_num);

double* tmp_buf = new double[disabling_num*3];
for (int i = 0; i < disabling_num; i++)
fscanf(fp, "(%lf, %lf, %lf)", &tmp_buf[i*3], &tmp_buf[i*3+1], &tmp_buf[i*3+2]);

disabling_data = (void*) tmp_buf;
} else if (strncmp(disable_method, "DISABLE_POINTS_NONE", 19) != 0) {
goto fail;
}
}

fclose(fp);
return true;

fail:
fclose(fp);
return false;
}


#ifdef NETCDF
void Grid_info_manager::read_grid_from_nc(const char filename[], const char lon_var_name[], const char lat_var_name[])
{
int num_dims;
int *dim_size_ptr;
int field_size;
int field_size2;
void *coord_buf0, *coord_buf1;

read_file_field_as_double(filename, lon_var_name, &coord_buf0, &num_dims, &dim_size_ptr, &field_size);
delete dim_size_ptr;
read_file_field_as_double(filename, lat_var_name, &coord_buf1, &num_dims, &dim_size_ptr, &field_size2);
delete dim_size_ptr;

num_points = field_size*field_size2;
coord_values[PDLN_LON] = new double [num_points];
coord_values[PDLN_LAT] = new double [num_points];

int count = 0;
for(int j = field_size2-1; j >= 0; j--)
for(int i = 0; i < field_size; i ++) {
coord_values[PDLN_LON][count] = ((double*)coord_buf0)[i];
coord_values[PDLN_LAT][count++] = ((double*)coord_buf1)[j];
}

PDASSERT(count == num_points);
PDASSERT(!have_redundent_points(coord_values[PDLN_LON], coord_values[PDLN_LAT], num_points));

min_lon =   0.0;
max_lon = 360.0;
min_lat = -90.0;
max_lat =  90.0;
is_cyclic = true;
}
#endif


double** Grid_info_manager::get_grid_coord_values(int grid_id)
{
return coord_values;
}


bool* Grid_info_manager::get_grid_mask(int grid_id)
{
return NULL;
}


void Grid_info_manager::get_disabled_points_info(int id, DISABLING_POINTS_METHOD *method, int *num, void **data)
{
*method = disabling_method;
*num = disabling_num;
*data = disabling_data;
}


int Grid_info_manager::get_grid_num_points(int grid_id)
{
return num_points;
}


void Grid_info_manager::get_grid_boundry(int grid_id, double* mi_lon, double* ma_lon, double* mi_lat, double* ma_lat)
{
*mi_lon = min_lon;
*ma_lon = max_lon;
*mi_lat = min_lat;
*ma_lat = max_lat;
}


void Grid_info_manager::set_grid_boundry(int grid_id, double mi_lon, double ma_lon, double mi_lat, double ma_lat)
{
min_lon = mi_lon;
max_lon = ma_lon;
min_lat = mi_lat;
max_lat = ma_lat;
}


bool Grid_info_manager::is_grid_cyclic(int grid_id)
{
return is_cyclic;
}
