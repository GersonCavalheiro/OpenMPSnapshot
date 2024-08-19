#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <tgmath.h>
#include <unistd.h>
#define WIDTH		640
#define HEIGHT		480
#define NSUBSAMPLES	8
#define NAO_SAMPLES	8
#define MIN(x,y)	(((x)<(y))?(x):(y))
#ifdef USE_FLOAT
typedef float  real_t;
#else
typedef double real_t;
#endif
typedef uint8_t   uchar;
typedef uint64_t seed_t;
typedef struct _vec {
real_t x;
real_t y;
real_t z;
} vec;
typedef struct _Isect {
real_t t;
vec p;
vec n;
int hit;
} Isect;
typedef struct _Sphere {
vec center;
real_t radius;
} Sphere;
typedef struct _Plane {
vec p;
vec n;
} Plane;
typedef struct _Ray {
vec org;
vec dir;
} Ray;
void   init_scene          (Sphere *, Plane  *, real_t  );
void   saveppm             (uchar  *, int, int, char   *);
void   render              (uchar  *, int, int, int, int, Sphere *, Plane *);
vec    ambient_occlusion   (Isect  *, Sphere *, Plane  *, seed_t *, int);
void   ortho_basis         (vec    *, vec);
void   ray_sphere_intersect(Isect  *, Ray    *, Sphere *);
void   ray_plane_intersect (Isect  *, Ray    *, Plane  *);
real_t vdot                (vec     , vec);
vec    vcross              (vec     , vec);
vec    vnormalize          (vec     );
uchar  clamp               (real_t  );
real_t drand64_r           (seed_t *);
double wtime               (void);
int main(int argc, char **argv)
{
Plane  plane;
Sphere spheres[3];
uchar *image = NULL;
real_t theta = 0.0;
double ts, te;
if ((image = (uchar *) malloc(3 * WIDTH * HEIGHT * sizeof(*image))) == NULL) {
perror("malloc");
return EXIT_FAILURE;
}
#pragma omp parallel for	
for (int i = 0; i < 3 * WIDTH * HEIGHT; i++)
image[i] = 0;
init_scene(spheres, &plane, theta);
ts = wtime();
#pragma omp parallel
render (image, WIDTH, HEIGHT, NSUBSAMPLES, NAO_SAMPLES, spheres, &plane);
te = wtime();
saveppm(image, WIDTH, HEIGHT, "output.ppm");
printf("Time=%f[msec.]\n", (te - ts) * 1000.0);
free(image);
return EXIT_SUCCESS;
}
void init_scene(Sphere *spheres, Plane *plane, real_t theta)
{
spheres[0].center.x = -2.0;
spheres[0].center.y =  0.0;
spheres[0].center.z = -3.5;
spheres[0].radius   =  0.5;
spheres[1].center.x = -0.5;
spheres[1].center.y = -0.5 * cos(theta) + 0.5;
spheres[1].center.z = -3.0;
spheres[1].radius   =  0.5;
spheres[2].center.x =  1.0;
spheres[2].center.y =  0.0;
spheres[2].center.z = -2.2;
spheres[2].radius   =  0.5;
plane->p.x =  0.0;
plane->p.y = -0.5;
plane->p.z =  0.0;
plane->n.x =  0.0;
plane->n.y =  1.0;
plane->n.z =  0.0;
return;
}
void saveppm(uchar *image, int w, int h, char *fname)
{
FILE *fp;
fp = fopen(fname, "wb");
assert(fp != NULL);
fprintf(fp, "P6\n%d %d\n255\n", w, h);
fwrite (image, 3 * w * h, sizeof(*image), fp);
fclose (fp);
return;
}
void render(uchar *image, int w, int h, int nsubsamples, int nao_samples, Sphere *spheres, Plane *plane)
{
int x, y;
int u, v;
real_t d = 2.0 / MIN(w, h);
seed_t seed = UINT64_MAX - omp_get_thread_num();
#pragma omp for collapse(2) schedule(static,1) nowait
for (y = 0; y < h; y++) {
for (x = 0; x < w; x++) {
real_t rr = 0.0,
gg = 0.0,
bb = 0.0;
for (v = 0; v < nsubsamples; v++) {
for (u = 0; u < nsubsamples; u++) {
real_t px =  (x + (u / (real_t) nsubsamples) - (w / 2.0)) * d;
real_t py = -(y + (v / (real_t) nsubsamples) - (h / 2.0)) * d;
Ray ray;
ray.org.x =  0.0;
ray.org.y =  0.0;
ray.org.z =  0.0;
ray.dir.x =   px;
ray.dir.y =   py;
ray.dir.z = -1.0;
ray.dir   = vnormalize(ray.dir);
Isect isect;
isect.t   = 1.0e+17;
isect.hit = 0;
ray_sphere_intersect(&isect, &ray, &spheres[0]);
ray_sphere_intersect(&isect, &ray, &spheres[1]);
ray_sphere_intersect(&isect, &ray, &spheres[2]);
ray_plane_intersect (&isect, &ray,  plane);
if (isect.hit) {
vec col = ambient_occlusion(&isect, spheres, plane, &seed, nao_samples);
rr += col.x;
gg += col.y;
bb += col.z;
}
}
}
image[3 * (x + y * w) + 0] = clamp(rr / (real_t) (nsubsamples * nsubsamples));
image[3 * (x + y * w) + 1] = clamp(gg / (real_t) (nsubsamples * nsubsamples));
image[3 * (x + y * w) + 2] = clamp(bb / (real_t) (nsubsamples * nsubsamples));
}
}
return;
}
vec ambient_occlusion(Isect *isect, Sphere *spheres, Plane *plane, seed_t *seed, int nao_samples)
{
int i, j;
int ntheta = nao_samples;
int nphi   = nao_samples;
real_t eps = 0.0001;
vec p;
p.x = isect->p.x + eps * isect->n.x;
p.y = isect->p.y + eps * isect->n.y;
p.z = isect->p.z + eps * isect->n.z;
vec basis[3];
ortho_basis(basis, isect->n);
real_t occlusion = 0.0;
for (j = 0; j < ntheta; j++) {
for (i = 0; i < nphi; i++) {
real_t theta =         sqrt(drand64_r(seed));
real_t phi   = 2.0 * M_PI * drand64_r(seed) ;
real_t x = cos(phi) * theta;
real_t y = sin(phi) * theta;
real_t z = sqrt(1.0 - theta * theta);
real_t rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
real_t ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
real_t rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;
Ray ray;
ray.org   = p ;
ray.dir.x = rx;
ray.dir.y = ry;
ray.dir.z = rz;
Isect occIsect;
occIsect.t = 1.0e+17;
occIsect.hit = 0;
ray_sphere_intersect(&occIsect, &ray, &spheres[0]);
ray_sphere_intersect(&occIsect, &ray, &spheres[1]);
ray_sphere_intersect(&occIsect, &ray, &spheres[2]);
ray_plane_intersect (&occIsect, &ray,  plane);
if (occIsect.hit)
occlusion += 1.0;
}
}
occlusion = (ntheta * nphi - occlusion) / (real_t) (ntheta * nphi);
p.x = occlusion;
p.y = occlusion;
p.z = occlusion;
return p;
}
void ortho_basis(vec *basis, vec n)
{
basis[1].x = 0.0;
basis[1].y = 0.0;
basis[1].z = 0.0;
basis[2]   = n  ;
if (n.x < 0.6 && n.x > -0.6) {
basis[1].x = 1.0;
} else if (n.y < 0.6 && n.y > -0.6) {
basis[1].y = 1.0;
} else if (n.z < 0.6 && n.z > -0.6) {
basis[1].z = 1.0;
} else {
basis[1].x = 1.0;
}
basis[0] = vcross    (basis[1], basis[2]);
basis[0] = vnormalize(basis[0]);
basis[1] = vcross    (basis[2], basis[0]);
basis[1] = vnormalize(basis[1]);
return;
}
real_t vdot(vec v0, vec v1)
{
return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
void ray_sphere_intersect(Isect *isect, Ray *ray, Sphere *sphere)
{
vec rs;
rs.x = ray->org.x - sphere->center.x;
rs.y = ray->org.y - sphere->center.y;
rs.z = ray->org.z - sphere->center.z;
real_t B = vdot(rs, ray->dir);
real_t C = vdot(rs, rs) - sphere->radius * sphere->radius;
real_t D = B * B - C;
if (D > 0.0) {
real_t t = -B - sqrt(D);
if (t > 0.0 && t < isect->t) {
isect->t   = t;
isect->hit = 1;
isect->p.x = ray->org.x + ray->dir.x * t;
isect->p.y = ray->org.y + ray->dir.y * t;
isect->p.z = ray->org.z + ray->dir.z * t;
isect->n.x = isect->p.x - sphere->center.x;
isect->n.y = isect->p.y - sphere->center.y;
isect->n.z = isect->p.z - sphere->center.z;
isect->n   = vnormalize(isect->n);
}
}
return;
}
void ray_plane_intersect(Isect *isect, Ray *ray, Plane *plane)
{
real_t d = -vdot(plane->p, plane->n);
real_t v =  vdot(ray->dir, plane->n);
if (fabs(v) < 1.0e-17)
return;
real_t t = -(vdot(ray->org, plane->n) + d) / v;
if (t > 0.0 && t < isect->t) {
isect->t   = t;
isect->hit = 1;
isect->p.x = ray->org.x + ray->dir.x * t;
isect->p.y = ray->org.y + ray->dir.y * t;
isect->p.z = ray->org.z + ray->dir.z * t;
isect->n   = plane->n;
}
return;
}
vec vcross(vec v0, vec v1)
{
vec c;
c.x = v0.y * v1.z - v0.z * v1.y;
c.y = v0.z * v1.x - v0.x * v1.z;
c.z = v0.x * v1.y - v0.y * v1.x;
return c;
}
vec vnormalize(vec c)
{
real_t length = sqrt(vdot(c, c));
if (fabs(length) > 1.0e-17) {
c.x /= length;
c.y /= length;
c.z /= length;
}
return c;
}
uchar clamp(real_t f)
{
int i = (int) (256.0 * f);
if (i <   0) i =   0;
if (i > 255) i = 255;
return (uchar) i;
}
real_t drand64_r(seed_t *x)
{				
*x = *x ^ (*x << 13);
*x = *x ^ (*x >>  7);
*x = *x ^ (*x << 17);
return ((real_t) *x) / ((real_t) UINT64_MAX);
}
double wtime(void)
{				
struct timespec ts;
if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
return -1.0;		
return (double) ts.tv_sec  +
(double) ts.tv_nsec * 1.E-9;
}
