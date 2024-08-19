#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"
#include <stdbool.h>
#define RED 0xFF0000FF
#define GREEN 0x00FF00FF
#define BLUE 0x0000FFFF
unsigned version = 0;
void first_touch_v1 (void);
void first_touch_v2 (void);
void init_v3(void);
unsigned compute_v0 (unsigned nb_iter);
unsigned compute_v1 (unsigned nb_iter);
unsigned compute_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);
unsigned compute_v4 (unsigned nb_iter);
unsigned compute_v5 (unsigned nb_iter);
unsigned compute_v6 (unsigned nb_iter);
unsigned compute_v7 (unsigned nb_iter);
unsigned compute_v8 (unsigned nb_iter);
void_func_t first_touch [] = {
NULL,
NULL,
NULL,
NULL,
};
void_func_t init[] = {
NULL,
NULL,
NULL,
init_v3,
NULL,
init_v3,
NULL,
NULL,
NULL
};
int_func_t compute [] = {
compute_v0, 
compute_v1, 
compute_v2, 
compute_v3, 
compute_v4, 
compute_v5, 
compute_v6, 
compute_v7, 
compute_v8,
};
char *version_name [] = {
"Séquentielle",
"OpenMP For basique",
"OpenMP For Tuile",
"OpenMP For Optimisee",
"OpenMP Task tuilee",
"OpenMP Task optimisee",
"OpenCL basique",
"OpenCL optimisee"
};
unsigned opencl_used [] = {
0,
0,
0,
0,
0,
0,
1,
1,
1
};
unsigned tranche;
bool **tiles_tracker;
#define GRAIN 32
int verify_life(unsigned i, unsigned j) {
int alive = 0;
int start_x = i == 1 ? 0 : i-1;
int start_y = j == 1 ? 0 : j-1;
int end_x = start_x + 3 >= DIM-1 ? DIM-1 : start_x + 3;
int end_y = start_y + 3 >= DIM-1 ? DIM-1 : start_y + 3;
for(int x = start_x; x < end_x; x++) {
for(int y = start_y; y < end_y; y++) {
if((x != i || y != j) && cur_img(x, y) != 0) {
alive+=1;
}
}
}
if(cur_img(i, j) != 0) {
return (alive == 2 || alive == 3) ? BLUE : 0;
} else {
return (alive == 3) ? GREEN : 0;
}
}
unsigned compute_v0 (unsigned nb_iter)
{
for (unsigned it = 1; it <= nb_iter; it ++) {
for (int i = 1; i < DIM-1; i++) {
for (int j = 1; j < DIM-1; j++) {
next_img (i, j) = verify_life(i, j);
}
}
swap_images();
}
return 0;
}
void first_touch_v1 ()
{
int i,j ;
#pragma omp parallel for collapse(2)
for(i=1; i<DIM-1 ; i++) {
for(j=1; j < DIM-1 ; j ++)
next_img (i, j) = verify_life (i, j);
}
}
unsigned compute_v1(unsigned nb_iter)
{
first_touch_v1();
swap_images();
return 0;
}
bool pixel_handler (int x, int y)
{
int alive = 0;
for (int i = x-1; i <= x+1; i++) {
for (int j = y-1; j <= y+1; j++) {
if ((i != x || j != y) && cur_img (i,j)) {
alive += 1;
}
}
}
if(cur_img(x, y))
next_img(x, y) = (alive == 2 || alive == 3) ? BLUE : 0;
else
next_img(x, y) = (alive == 3) ? GREEN : 0;
return  (0x000000FF & next_img(x, y)) != (0x000000FF & cur_img(x, y));
}
void tile_handler (int i, int j)
{
int i_d = (i == 1) ? 1 : i * GRAIN;
int j_d = (j == 1) ? 1 : j * GRAIN;
int i_f = (i == tranche-1) ? DIM-1 : (i+1) * GRAIN;
int j_f = (j == tranche-1) ? DIM-1 : (j+1) * GRAIN;
for(int x = i_d; x < i_f; ++x) {
for(int y = j_d; y < j_f; ++y) {
pixel_handler(x, y);
}
}
}
int launch_tile_handlers (void)
{
tranche = DIM / GRAIN;
#pragma omp parallel for collapse(2) schedule(static)
for (int i=1; i < tranche; i++)
for (int j=1; j < tranche; j++)
tile_handler (i, j);
return 0;
}
unsigned compute_v2(unsigned nb_iter) {
launch_tile_handlers();
swap_images();
return 0;
}
void tile_handler_optim (int i, int j)
{
if(tiles_tracker[i][j] == true) {
tiles_tracker[i][j] = false;
int i_d = (i == 1) ? 1 : i * GRAIN;
int j_d = (j == 1) ? 1 : j * GRAIN;
int i_f = (i == tranche-1) ? DIM-1 : (i+1) * GRAIN;
int j_f = (j == tranche-1) ? DIM-1 : (j+1) * GRAIN;
for(int x = i_d; x < i_f; ++x) {
for(int y = j_d; y < j_f; ++y) {
if(pixel_handler(x, y)) {
tiles_tracker[i][j] = true;
tiles_tracker[i+1][j] = true;
tiles_tracker[i][j+1] = true;
tiles_tracker[i-1][j] = true;
tiles_tracker[i][j-1] = true;
}
}
}
}
}
int launch_tile_handlers_optim (void)
{
tranche = DIM / GRAIN;
#pragma omp parallel for collapse(2) schedule(static)
for (int i=1; i < tranche; i++)
for (int j=1; j < tranche; j++)
tile_handler_optim (i, j);
return 0;
}
void init_v3() {
tranche = (DIM+GRAIN-1) / GRAIN;
tiles_tracker = malloc(sizeof(bool*)*(tranche+2));
for(int i = 0; i < tranche+1; i++) {
tiles_tracker[i] = malloc(sizeof(bool)*(tranche+2));
for(int j = 0; j < tranche+1; j++) {
tiles_tracker[i][j] = true;
}
}
}
unsigned compute_v3(unsigned nb_iter)
{
launch_tile_handlers_optim();
swap_images();
return 0; 
}
int launch_tile_handlers_task (void)
{
tranche = (DIM+GRAIN-1) / GRAIN;
#pragma omp parallel
for (int i=1; i < tranche; i++) {
for (int j=1; j < tranche; j++) {
#pragma omp single nowait
#pragma omp task
{
tile_handler (i, j);
}
}
}
return 0;
}
unsigned compute_v4 (unsigned nb_iter)
{
launch_tile_handlers_task();
swap_images();
return 0;
}
void tile_handler_optim_task (int i, int j)
{
if(tiles_tracker[i][j] == true) {
#pragma omp single nowait
#pragma omp task
{
tiles_tracker[i][j] = false;
int i_d = (i == 1) ? 1 : i * GRAIN;
int j_d = (j == 1) ? 1 : j * GRAIN;
int i_f = (i == tranche-1) ? DIM-1 : (i+1) * GRAIN;
int j_f = (j == tranche-1) ? DIM-1 : (j+1) * GRAIN;
for(int x = i_d; x < i_f; ++x) {
for(int y = j_d; y < j_f; ++y) {
if(pixel_handler(x, y)) {
tiles_tracker[i][j] = true;
tiles_tracker[i+1][j] = true;
tiles_tracker[i][j+1] = true;
tiles_tracker[i-1][j] = true;
tiles_tracker[i][j-1] = true;
}
}
}
}
}
}
int launch_tile_handlers_optim_task (void)
{
tranche = (DIM+GRAIN-1) / GRAIN;
#pragma omp parallel
for (int i=1; i < tranche; i++) {
for (int j=1; j < tranche; j++) {
tile_handler_optim_task (i, j);
}
}
return 0;
}
unsigned compute_v5(unsigned nb_iter)
{
launch_tile_handlers_optim_task();
swap_images();
return 0; 
}
unsigned compute_v6(unsigned nb_iter)
{
return ocl_compute_naif (nb_iter);
}
unsigned compute_v7(unsigned nb_iter)
{    
return ocl_compute_optimized (nb_iter);
}
unsigned compute_v8(unsigned nb_iter)
{
return ocl_compute_naif (nb_iter);
}
