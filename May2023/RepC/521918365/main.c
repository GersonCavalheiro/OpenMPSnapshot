#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <conio.h>
#include <math.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
typedef struct { unsigned char r, g, b; } rgb;
void set_texture();
void keypress();
void render();
rgb** texture = 0;
int width, height;
double scale = 1. / 256;
double offsetx = -.7, offsety = 0;
int max_iterations = 128;
int color_rotate = 0;
int saturation = 1;
int invert = 0;
const double MaxValueExtent = 2.0;
const double MaxNorm = 4.0; 
int dump = 1;
int screenshot_bmp_count = 1;
void reset()
{
scale = 1. / 256;
offsetx = -.7;
offsety = 0;
max_iterations = 128;
}
void hsv_to_rgb(int hue, int min, int max, rgb* p)
{
if (min == max) max = min + 1;
if (invert) hue = max - (hue - min);
if (!saturation)
{
p->r = p->g = p->b = 255 * (max - hue) / (max - min);
return;
}
double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
#	define VAL 255
double c = VAL * saturation;
double X = c * (1 - fabs(fmod(h, 2) - 1));
p->r = p->g = p->b = 0;
switch ((int)h) {
case 0: p->r = c; p->g = X; return;
case 1:	p->r = X; p->g = c; return;
case 2: p->g = c; p->b = X; return;
case 3: p->g = X; p->b = c; return;
case 4: p->r = X; p->b = c; return;
default:p->r = c; p->b = X;
}
}
void mandelbrot()
{
int i, j;
int shared_min = max_iterations;
int shared_max = 0;
#pragma omp parallel shared(shared_min) shared(shared_max)
{
#pragma omp for nowait private(i,j)
for (i = 0; i < height; i++) {
rgb* px = texture[i];
double Re, Im, Re2, Im2;
double y = (i - height / 2) * scale + offsety;
for (j = 0; j < width; px++, j++) {
double x = (j - width / 2) * scale + offsetx;
Re = Im = Re2 = Im2 = 0;
int iteration = 0;
for (; Re2 + Im2 < MaxNorm && iteration < max_iterations; iteration++) {
Im = 2 * Re * Im + y;
Re = Re2 - Im2 + x;
Re2 = Re * Re;
Im2 = Im * Im;
}
*(unsigned short*)px = iteration;
#pragma omp critical
{
if (iteration < shared_min) shared_min = iteration;
if (iteration > shared_max) shared_max = iteration;
}
}
}
}
{
#pragma omp parallel
{
#pragma omp for private(i,j)		
for (i = 0; i < height; i++)
{
rgb* px;
for (j = 0, px = texture[i]; j < width; j++, px++)
hsv_to_rgb(*(unsigned short*)px, shared_min, shared_max, px);
}
}
}
}
int gwin;
GLuint gltexture;
int tex_w, tex_h;
void allocate_texture()
{
int i, ow = tex_w, oh = tex_h;
for (tex_w = 1; tex_w < width; tex_w <<= 1);
for (tex_h = 1; tex_h < height; tex_h <<= 1);
if (tex_h != oh || tex_w != ow)
texture = realloc(texture, tex_h * tex_w * 3 + tex_h * sizeof(rgb*));
for (texture[0] = (rgb*)(texture + tex_h), i = 1; i < tex_h; i++)
texture[i] = texture[i - 1] + tex_w;
}
void set_texture()
{
allocate_texture();
mandelbrot();
glEnable(GL_TEXTURE_2D);
glBindTexture(GL_TEXTURE_2D, gltexture);
glTexImage2D(GL_TEXTURE_2D, 0, 3, tex_w, tex_h, 0, GL_RGB, GL_UNSIGNED_BYTE, texture[0]);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
render();
}
void mouseclick(int button, int state, int x, int y)
{
if (state != GLUT_UP) return;
offsetx += (x - width / 2) * scale;
offsety -= (y - height / 2) * scale;
switch (button)
{
case GLUT_LEFT_BUTTON: 
if (scale > fabs(x) * 1e-16 && scale > fabs(y) * 1e-16)
{
scale /= 2;
}
break;
case GLUT_RIGHT_BUTTON: 
if (scale < 1. / 256)
{
scale *= 2;
}
break;
}
set_texture();
}
int create_screenshot_directory()
{
int result = mkdir("screenshots", 0777);
return result;
}
void message_screenshot_saved(char filename[255])
{
printf("Screenshot saved in %s\n", filename);
}
void message_unable_to_create_files()
{
printf("Error: Unable to create file.\n");
}
void message_max_iterations()
{
printf("Max iterations: %d\n", max_iterations);
}
void screenshot_ppm()
{
create_screenshot_directory();
char fn[255];
sprintf(fn, "screenshots/screenshot_mandelbrot_set_%03d.ppm", dump++);
FILE* fp = fopen(fn, "w");
if (fp)
{
fprintf(fp, "P6\n%d %d\n255\n", width, height);
for (int i = height - 1; i >= 0; i--)
{
fwrite(texture[i], 1, width * 3, fp);
}			
fclose(fp);
message_screenshot_saved(fn);
}
else
{
message_unable_to_create_files();
}
}
void screenshot_bmp()
{
create_screenshot_directory();
int w = width, h = height;
FILE* f;
unsigned char* img = NULL;
int filesize = 54 + 3 * w * h;  
img = (unsigned char*)malloc(3 * w * h);
memset(img, 0, 3 * w * h);
for (int i = 0; i < w; i++)
{
for (int j = 0; j < h; j++)
{
int x = i; int y = (h - 1) - j;
unsigned char r = texture[j][i].r;
unsigned char g = texture[j][i].g;
unsigned char b = texture[j][i].b;
if (r > 255) r = 255;
if (g > 255) g = 255;
if (b > 255) b = 255;
img[(x + y * w) * 3 + 2] = (unsigned char)(r);
img[(x + y * w) * 3 + 1] = (unsigned char)(g);
img[(x + y * w) * 3 + 0] = (unsigned char)(b);
}
}
unsigned char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
unsigned char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };
unsigned char bmppad[3] = { 0,0,0 };
bmpfileheader[2] = (unsigned char)(filesize);
bmpfileheader[3] = (unsigned char)(filesize >> 8);
bmpfileheader[4] = (unsigned char)(filesize >> 16);
bmpfileheader[5] = (unsigned char)(filesize >> 24);
bmpinfoheader[4] = (unsigned char)(w);
bmpinfoheader[5] = (unsigned char)(w >> 8);
bmpinfoheader[6] = (unsigned char)(w >> 16);
bmpinfoheader[7] = (unsigned char)(w >> 24);
bmpinfoheader[8] = (unsigned char)(h);
bmpinfoheader[9] = (unsigned char)(h >> 8);
bmpinfoheader[10] = (unsigned char)(h >> 16);
bmpinfoheader[11] = (unsigned char)(h >> 24);
char fn[255];
sprintf(fn, "screenshots/screenshot_mandelbrot_set_%03d.bmp", screenshot_bmp_count++);
f = fopen(fn, "wb");
if (f)
{
fwrite(bmpfileheader, 1, 14, f);
fwrite(bmpinfoheader, 1, 40, f);
for (int i = 0; i < h; i++)
{
fwrite(img + (w * (h - i - 1) * 3), 3, w, f);
fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
}
free(img);
fclose(f);
message_screenshot_saved(fn);
}
else
{
message_unable_to_create_files();
}
}
void resize(int w, int h)
{
width = w;
height = h;
glLoadIdentity(); 
glViewport(0, 0, w, h);
glOrtho(0, w, 0, h, -1, 1);
set_texture();
}
void init_gfx(int* c, char** v)
{
glutInit(c, v);
glutInitDisplayMode(GLUT_RGB);
glutInitWindowSize(600, 540);
gwin = glutCreateWindow("Mandelbrot set by Cvtx");
glutDisplayFunc(render);
glutReshapeFunc(resize);
glutKeyboardFunc(keypress);
glutMouseFunc(mouseclick);
glGenTextures(1, &gltexture);
set_texture();
}
void render()
{
double x = (double)width / tex_w;
double y = (double)height / tex_h;
glClear(GL_COLOR_BUFFER_BIT);
glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
glBindTexture(GL_TEXTURE_2D, gltexture);
glBegin(GL_QUADS);
glTexCoord2f(0, 0); glVertex2i(0, 0);
glTexCoord2f(x, 0); glVertex2i(width, 0);
glTexCoord2f(x, y); glVertex2i(width, height);
glTexCoord2f(0, y); glVertex2i(0, height);
glEnd();
glFlush();
glFinish();
}
void keypress(unsigned char key, int x, int y)
{
switch (key) {
case 'q':
glFinish();
glutDestroyWindow(gwin);
return;
case 27: reset(); break;
case '>': case '.':
max_iterations += 128;
if (max_iterations > 1 << 15) max_iterations = 1 << 15;
message_max_iterations();
break;
case '<': case ',':
max_iterations -= 128;
if (max_iterations < 128) max_iterations = 128;
message_max_iterations();
break;
case 'x':
max_iterations = 4096;
message_max_iterations();
break;
case 'z':
max_iterations = 128;
message_max_iterations();
break;
case 'r':	color_rotate = (color_rotate + 1) % 6; break;
case 'c':	saturation = 1 - saturation; break;
case ' ':	invert = !invert; break;
case 's': screenshot_bmp(); return; break;
}
set_texture();
}
int main(int c, char** v)
{
init_gfx(&c, v);
printf("Keybindings:\n\tmouse buttons to zoom/unzoom\n\t< , >: decrease/increase drawing distance (max iterations) \n\t\z , x: min/max drawing distance\n\tr: color rotation\n\tc: monochrome\n\tspace: invert colors\n\ts: screenshot (.bmp) warning: will overwrite existing screenshots\n\tesc: reset\n\tq: quit\n");
printf("More iterations = longer to compute\n");
printf("Bigger window   = longer to compute\n");
glutMainLoop();
return 0;
}