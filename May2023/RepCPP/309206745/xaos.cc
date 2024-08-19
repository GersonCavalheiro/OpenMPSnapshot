#include <math.h>
#include <float.h>

#include "common.h"
#include "xaos.h"
#include "sse.h"

#define ZOOM_LIMIT 1e295*MAXX*DBL_MIN

typedef struct tagPoint {
double distance;
int idx_original;
int idx_best;
} Point;

int compare_points(const void *p1, const void *p2)
{
double d1 = ((Point*)p1)->distance;
double d2 = ((Point*)p2)->distance;
return \
(d2 < d1) ? -1 :
(d2 > d1) ?  1 :
0;
}

#if defined(__x86_64__) && !defined(__WIN64__)
#define AUTO_DISPATCH __attribute__((target_clones("default","sse","avx")))
#else
#define AUTO_DISPATCH
#endif

AUTO_DISPATCH
void mandel(
double xld, double yld, double xru, double yru,
double percentageOfPixelsToRedraw)
{
int i, j;

double xstep, ystep, xcur, ycur;

static int bufIdx = 0;

static double *xcoords[2] = {NULL, NULL}, *ycoords[2] = {NULL, NULL};

static int *xlookup, *ylookup;

static Uint8 *bufferMem[2];

static Point *points;

int bFirstFrameEver = !xcoords[0];

if (bFirstFrameEver) {
for (i=0; i<2; i++) {
xcoords[i] = new double[MAXX];
ycoords[i] = new double[MAXY];
if (!xcoords[i] || !ycoords[i])
panic("Out of memory");
memset(xcoords[i], 0, MAXX*sizeof(double));
memset(ycoords[i], 0, MAXY*sizeof(double));

bufferMem[i] = new Uint8[MAXX*MAXY];
if (!bufferMem[i])
panic("Out of memory");
}

xlookup = new int[MAXX];
ylookup = new int[MAXY];
if (!xlookup || !ylookup)
panic("Out of memory");

points = new Point[MAXX];
if (!points)
panic("Out of memory");
}

bufIdx ^= 1;

xstep = (xru - xld)/MAXX;
ystep = (yru - yld)/MAXY;

xcur = xld;
for (i=0; i<MAXX; i++) {
int idx_best = -1;
double diff = 1e10;
xcoords[bufIdx][i] = xcur;
for (j=i-30; j<i+30; j++) {
if(j<0) continue;
if(j>MAXX-1) continue;
double ndiff = fabs(xcur - xcoords[bufIdx^1][j]);
if (ndiff < diff) {
diff = ndiff;
idx_best = j;
}
}
points[i].distance = diff;
points[i].idx_best = idx_best;
points[i].idx_original = i;
xcur += xstep;
}
qsort(points, MAXX, sizeof(Point), compare_points);
for(i=0; i<MAXX; i++) {
int orig_idx = points[i].idx_original;
int idx_best = points[i].idx_best;
if (bFirstFrameEver || (i<MAXX*percentageOfPixelsToRedraw/100))
xlookup[orig_idx] = -1;
else {
xlookup[orig_idx] = idx_best;
xcoords[bufIdx][orig_idx] = xcoords[bufIdx^1][idx_best];
}
}

ycur = yru;
for (i=0; i<MAXY; i++) {
int idx_best = -1;
double diff = 1e10;
ycoords[bufIdx][i] = ycur;
for (j=i-30; j<i+30; j++) {
if(j<0) continue;
if(j>MAXY-1) continue;
double ndiff = fabs(ycur - ycoords[bufIdx^1][j]);
if (ndiff < diff) {
diff = ndiff;
idx_best = j;
}
}
points[i].distance = diff;
points[i].idx_best = idx_best;
points[i].idx_original = i;
ycur -= ystep;
}
qsort(points, MAXY, sizeof(Point), compare_points);
for(i=0; i<MAXY; i++) {
int orig_idx = points[i].idx_original;
int idx_best = points[i].idx_best;
if (bFirstFrameEver || (i<MAXY*percentageOfPixelsToRedraw/100))
ylookup[orig_idx] = -1;
else {
ylookup[orig_idx] = idx_best;
ycoords[bufIdx][orig_idx] = ycoords[bufIdx^1][idx_best];
}
}

#pragma omp parallel for private(xcur, j) schedule(dynamic,1)
for (int i=0; i<MAXY; i++) {
double ycur = yru - i*ystep;
unsigned char *p = &bufferMem[bufIdx][i*MAXX];
int yclose = ylookup[i];
xcur = xld;
for (j=0; j<MAXX; j+=4) {
int xclose  = xlookup[j];
int xclose2 = xlookup[j+1];
int xclose3 = xlookup[j+2];
int xclose4 = xlookup[j+3];
if (xclose  != -1 && xclose2 != -1 &&
xclose3 != -1 && xclose4 != -1 && yclose != -1)
{
*p++ = bufferMem[bufIdx^1][yclose*MAXX + xclose];
*p++ = bufferMem[bufIdx^1][yclose*MAXX + xclose2];
*p++ = bufferMem[bufIdx^1][yclose*MAXX + xclose3];
*p++ = bufferMem[bufIdx^1][yclose*MAXX + xclose4];
} else {
CoreLoopDouble(xcur, ycur, xstep, &p);
}
xcur += 4*xstep;
}
}
Uint8 *pixels = (Uint8*)surface->pixels;
Uint8 *src = bufferMem[bufIdx];
for (int i=0; i<MAXY; i++) {
memcpy(pixels, src, MAXX);
src += MAXX;
pixels += surface->pitch;
}
SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
SDL_RenderCopy(renderer, texture, NULL, NULL);
SDL_RenderPresent(renderer);
SDL_DestroyTexture(texture);
}

AUTO_DISPATCH
double autopilot(double percent, bool benchmark)
{
static double interesting_points[][2] = {
{-0.72996052273553402312, -0.24047620199671820851},
{-0.73162093699311890000, -0.25655927868100719680},
{-1.03178026025649338671, -0.36035584735925418887},
{-0.73174633145360257203,  0.21761907852168510535},
{-1.25616009010536688884,  0.39944527454476780326},
{-0.03804043691413014350, -0.98541408335385771711}
};
const int total_interesting_points =
sizeof(interesting_points) / sizeof(interesting_points[0]);
int start_idx =
benchmark ? 0 : (rand() % total_interesting_points);

int frames = 0;
unsigned long ticks = 0;
while(1) {
int rand_idx = start_idx % total_interesting_points;
double targetx = interesting_points[rand_idx][0];
double targety = interesting_points[rand_idx][1];
start_idx++;

double xld = -2.2, yld=-1.1, xru=-2+(MAXX/MAXY)*3., yru=1.1;

double percentage_of_pixels = 100.0;

while(1) {
unsigned st = SDL_GetTicks();
mandel(xld, yld, xru, yru, percentage_of_pixels);
unsigned en = SDL_GetTicks();
ticks += en-st;

percentage_of_pixels = percent;

if (en - st < minimum_ms_per_frame)
SDL_Delay(minimum_ms_per_frame - en + st);

int x,y;
int result = kbhit(&x, &y);
if (result == SDL_QUIT)
return ((double)frames)*1000.0/ticks;

double xrange = xru-xld;
if (xrange < ZOOM_LIMIT)
break;
xld += (targetx - xld)/100.;
xru += (targetx - xru)/100.;
yld += (targety - yld)/100.;
yru += (targety - yru)/100.;
frames++;
}
if (benchmark)
break;
}
printf("[-] Rendered  : %d frames\n", frames);
return ((double)frames)*1000.0/ticks;
}

AUTO_DISPATCH
double mousedriven(double percent)
{
int x,y;
double xld = -2.2, yld=-1.1, xru=-2+(MAXX/MAXY)*3., yru=1.1;
unsigned time_since_we_moved = SDL_GetTicks();
bool drawn_full = false, moved = false;
int frames = 0;
unsigned long ticks = 0;

while(1) {
if (!moved && (SDL_GetTicks() - time_since_we_moved > 200)) {
if (!drawn_full) {
drawn_full = true;
unsigned st = SDL_GetTicks();
mandel(xld, yld, xru, yru, 100.0);
unsigned en = SDL_GetTicks();
ticks += en-st;
frames++;
if (en - st < minimum_ms_per_frame)
SDL_Delay(minimum_ms_per_frame - en + st);
} else
SDL_Delay(minimum_ms_per_frame);
} else if (moved) {
drawn_full = false;
unsigned st = SDL_GetTicks();
mandel(xld, yld, xru, yru, percent);
unsigned en = SDL_GetTicks();
ticks += en-st;
frames++;
if (en - st < minimum_ms_per_frame)
SDL_Delay(minimum_ms_per_frame - en + st);
moved = false;
}
int result = kbhit(&x, &y);
if (result == SDL_QUIT)
break;
else if (result == SDL_BUTTON_LEFT || result == SDL_BUTTON_RIGHT) {
moved = true;
time_since_we_moved = SDL_GetTicks();
double ratiox = ((double)x)/window_width;
double ratioy = ((double)y)/window_height;
double xrange = xru-xld;
double yrange = yru-yld;
double direction = result==SDL_BUTTON_LEFT ? 1. : -1.;
if (result == SDL_BUTTON_LEFT && xrange < ZOOM_LIMIT)
continue;
xld += direction*0.01*ratiox*xrange;
xru -= direction*0.01*(1.-ratiox)*xrange;
yld += direction*0.01*(1.-ratioy)*yrange;
yru -= direction*0.01*ratioy*yrange;
} else if (result == SDL_WINDOWEVENT) {
moved = true;
time_since_we_moved = SDL_GetTicks();
SDL_GetWindowSize(window, &window_width, &window_height);
}
}
printf("[-] Rendered  : %d frames\n", frames);
return ((double)frames)*1000.0/ticks;
}

