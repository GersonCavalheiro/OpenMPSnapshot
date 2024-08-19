#ifdef USE_MPI
#include <mpi.h>
#endif
#include <new>
#include <cmath>
#include <random>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>
typedef double real_t;
#define C_R		(-0.74323348754012)
#define C_I		( 0.13121889397412)
#define RADIUS		(1.E-7)
#define WIDTH		1920
#define HEIGHT		1080
#define COLORMAP_CYCLE	(0x01<<9)
#define OUTPUT_FNAME	"output.ppm"
#define D		(2.0*RADIUS/std::min(WIDTH,HEIGHT))
#define MIN_SAMPLING	(0x01<<4)
#define MAX_SAMPLING	(0x01<<16)
#define MAX_ITER	(0x01<<16)
void init_colormap  (uint8_t *);
void init_jitter    (float   *, float   *);
void draw_image     (uint8_t *, uint8_t *, uint8_t *, float   *, float *, int, int);
void rough_sketch   (uint8_t *, uint8_t *, int, int);
bool detect_edge    (uint8_t *, uint8_t *, uint8_t *, uint8_t *, int    , int    );
bool same_color     (uint8_t  , uint8_t  , uint8_t  , uint8_t  , uint8_t, uint8_t);
#pragma omp declare simd notinbranch
inline
int  mandelbrot     (real_t   , real_t  );
void write_out_image(uint8_t *, const char *);
int main(int argc, char **argv)
{
uint8_t *colormap = nullptr, *sketch = nullptr, *image = nullptr;
float   *dx       = nullptr, *dy     = nullptr;
int      nprocs   = 1      ,  myrank = 0      ;
#ifdef USE_MPI
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif
try {
colormap = new uint8_t[3 * (COLORMAP_CYCLE + 1)];
sketch   = new uint8_t[3 * WIDTH * HEIGHT      ];
image    = new uint8_t[3 * WIDTH * HEIGHT      ];
dx       = new float  [MAX_SAMPLING            ];
dy       = new float  [MAX_SAMPLING            ];
} catch (std::bad_alloc e) {
std::cerr << e.what() << std::endl;
return EXIT_FAILURE;
}
std::memset(sketch, 0x00, 3 * WIDTH * HEIGHT * sizeof(uint8_t));
std::memset(image , 0x00, 3 * WIDTH * HEIGHT * sizeof(uint8_t));
if (myrank == 0) {
init_colormap(colormap);
init_jitter  (dx, dy  );
}
#ifdef USE_MPI
MPI_Bcast(colormap, 3 * (COLORMAP_CYCLE + 1) * sizeof(uint8_t), MPI_BYTE, 0, MPI_COMM_WORLD);
MPI_Bcast(dx      , MAX_SAMPLING             * sizeof(float  ), MPI_BYTE, 0, MPI_COMM_WORLD);
MPI_Bcast(dy      , MAX_SAMPLING             * sizeof(float  ), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
draw_image(image, sketch, colormap, dx, dy, nprocs, myrank);
if (myrank == 0) {
write_out_image(image, OUTPUT_FNAME);
}
delete[] dx;
delete[] dy;
delete[] image;
delete[] sketch;
delete[] colormap;
#ifdef USE_MPI
MPI_Finalize();
#endif
return EXIT_SUCCESS;
}
void init_colormap(uint8_t *colormap)
{				
for (int i = 0; i < COLORMAP_CYCLE; i++) {
uint8_t r = ((int) (127.0 * std::cos((2.0 * M_PI * i) / (COLORMAP_CYCLE   )) + 0.5)) + 128,
g = ((int) (127.0 * std::sin((2.0 * M_PI * i) / (COLORMAP_CYCLE   )) + 0.5)) + 128,
b = ((int) (127.0 * std::sin((2.0 * M_PI * i) / (COLORMAP_CYCLE>>1)) + 0.5)) + 128;
colormap[3 * i + 0] = r;
colormap[3 * i + 1] = g;
colormap[3 * i + 2] = b;
}
colormap[3 * COLORMAP_CYCLE + 0] =
colormap[3 * COLORMAP_CYCLE + 1] =
colormap[3 * COLORMAP_CYCLE + 2] = 0;
return;
}
void init_jitter(float *dx, float *dy)
{				
std::mt19937 mt_engine(std::random_device{}());
std::uniform_real_distribution<float> frand(0.0, 1.0);
dx[0] = dy[0] = 0.0f;
for (int i = 1; i < MAX_SAMPLING; i++) {
dx[i] = frand(mt_engine);
dy[i] = frand(mt_engine);
}
return;
}
void draw_image(uint8_t *image, uint8_t *sketch, uint8_t *colormap,
float *dx, float *dy, int nprocs, int myrank)
{				
rough_sketch(sketch, colormap, nprocs, myrank);
#pragma omp parallel for schedule(dynamic,1)
for (size_t l = myrank; l < WIDTH * HEIGHT; l += nprocs) {
int i = l % WIDTH,
j = l / WIDTH;
uint8_t r , g , b ,
rr, gg, bb;
if (detect_edge(sketch, &r, &g, &b, i, j)) {	
int n     = 1, m     = MIN_SAMPLING,
sum_r = r, sum_g = g, sum_b = b;
do {
rr = r;
gg = g;
bb = b;
#pragma omp simd reduction(+:sum_r,sum_g,sum_b)
for (int k = n; k < m; k++) {
real_t p_r = C_R + D * (i + dx[k] - WIDTH  / 2),
p_i = C_I - D * (j + dy[k] - HEIGHT / 2);
int  index = mandelbrot(p_r, p_i);
sum_r += colormap[3 * index + 0];
sum_g += colormap[3 * index + 1];
sum_b += colormap[3 * index + 2]; 
}
r = (uint8_t) ((sum_r + (m>>1)) / m);
g = (uint8_t) ((sum_g + (m>>1)) / m);
b = (uint8_t) ((sum_b + (m>>1)) / m);
} while (!same_color(r, g, b, rr, gg, bb) &&
(m = (n = m) << 1) <= MAX_SAMPLING);
}
image[3 * l + 0] = r;
image[3 * l + 1] = g;
image[3 * l + 2] = b;
}
#ifdef USE_MPI
MPI_Allreduce(MPI_IN_PLACE, image, 3 * WIDTH * HEIGHT * sizeof(uint8_t),
MPI_BYTE, MPI_BOR, MPI_COMM_WORLD);
#endif
return;
}
void rough_sketch(uint8_t *sketch, uint8_t *colormap, int nprocs, int myrank)
{				
#pragma omp parallel for schedule(dynamic,1)
for (size_t k = myrank; k < WIDTH * HEIGHT; k += nprocs) {
int i = k % WIDTH,
j = k / WIDTH;
real_t p_r = C_R + D * (i - WIDTH  / 2),
p_i = C_I - D * (j - HEIGHT / 2);
int  index = mandelbrot(p_r, p_i);
uint8_t  r = colormap[3 * index + 0],
g = colormap[3 * index + 1],
b = colormap[3 * index + 2];
sketch[3 * k + 0] = r;
sketch[3 * k + 1] = g;
sketch[3 * k + 2] = b;
}
#ifdef USE_MPI
MPI_Allreduce(MPI_IN_PLACE, sketch, 3 * WIDTH * HEIGHT * sizeof(uint8_t),
MPI_BYTE, MPI_BOR, MPI_COMM_WORLD);
#endif
return;
}
bool detect_edge(uint8_t *image, uint8_t *r, uint8_t *g, uint8_t *b, int x, int y)
{				
*r = image[3 * (x + y * WIDTH) + 0];
*g = image[3 * (x + y * WIDTH) + 1];
*b = image[3 * (x + y * WIDTH) + 2];
for (int j = std::max(0, y-1); j <= std::min(HEIGHT-1, y+1); j++) {
for (int i = std::max(0, x-1); i <= std::min(WIDTH-1, x+1); i++) {
int rr = image[3 * (i + j * WIDTH) + 0],
gg = image[3 * (i + j * WIDTH) + 1],
bb = image[3 * (i + j * WIDTH) + 2];
if (!same_color(*r, *g, *b, rr, gg, bb)) {
return true;
}
}
}
return false;
}
bool same_color(uint8_t r1, uint8_t g1, uint8_t b1,
uint8_t r2, uint8_t g2, uint8_t b2)
#if 0
{				
return r1 == r2 &&
g1 == g2 &&
b1 == b2;
}
#else				
{				
return 3 * std::abs((int) r1 - r2) +
6 * std::abs((int) g1 - g2) +
1 * std::abs((int) b1 - b2) < 15;
}
#endif
inline
int mandelbrot(real_t p_r, real_t p_i)
{
int i;
real_t z_r, z_i, work;
z_r  = p_r;
z_i  = p_i;
work = 2.0 * z_r * z_i;
for (i = 1; i < MAX_ITER && (z_r *= z_r) +
(z_i *= z_i) < 4.0; i++) {
z_r += p_r - z_i ;
z_i  = p_i + work;
work = 2.0 * z_r * z_i;
}
if (i &= MAX_ITER       - 1) {
i &= COLORMAP_CYCLE - 1;
} else {	
i  = COLORMAP_CYCLE    ;
}
return i;
}
void write_out_image(uint8_t *image, const char *fname)
{				
std::ofstream fout(fname);
if (!fout.good()) {
std::cerr << "Error: cannot open " << fname << "." << std::endl;
std::exit(EXIT_FAILURE);
}
fout << "P6"  << std::endl;
fout << WIDTH << " " << HEIGHT << std::endl;
fout << "255" << std::endl;
fout.write((const char *) image, 3 * WIDTH * HEIGHT * sizeof(uint8_t));
if (fout.fail()) {
std::cerr << "Error: cannot write out " << fname << "." << std::endl;
std::exit(EXIT_FAILURE);
}
fout.close();
return;
}
