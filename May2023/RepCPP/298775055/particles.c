
#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <tinycthread.h>
#include <getopt.h>
#include <linmath.h>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifndef GL_EXT_separate_specular_color
#define GL_LIGHT_MODEL_COLOR_CONTROL_EXT  0x81F8
#define GL_SINGLE_COLOR_EXT               0x81F9
#define GL_SEPARATE_SPECULAR_COLOR_EXT    0x81FA
#endif 



typedef struct
{
float x, y, z;
} Vec3;

typedef struct
{
GLfloat s, t;         
GLuint  rgba;         
GLfloat x, y, z;      
} Vertex;



float aspect_ratio;

int wireframe;

struct {
double    t;         
float     dt;        
int       p_frame;   
int       d_frame;   
cnd_t     p_done;    
cnd_t     d_done;    
mtx_t     particles_lock; 
} thread_sync;



#define P_TEX_WIDTH  8    
#define P_TEX_HEIGHT 8
#define F_TEX_WIDTH  16   
#define F_TEX_HEIGHT 16

GLuint particle_tex_id, floor_tex_id;

const unsigned char particle_texture[ P_TEX_WIDTH * P_TEX_HEIGHT ] = {
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
0x00, 0x00, 0x11, 0x22, 0x22, 0x11, 0x00, 0x00,
0x00, 0x11, 0x33, 0x88, 0x77, 0x33, 0x11, 0x00,
0x00, 0x22, 0x88, 0xff, 0xee, 0x77, 0x22, 0x00,
0x00, 0x22, 0x77, 0xee, 0xff, 0x88, 0x22, 0x00,
0x00, 0x11, 0x33, 0x77, 0x88, 0x33, 0x11, 0x00,
0x00, 0x00, 0x11, 0x33, 0x22, 0x11, 0x00, 0x00,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

const unsigned char floor_texture[ F_TEX_WIDTH * F_TEX_HEIGHT ] = {
0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
0xff, 0xf0, 0xcc, 0xf0, 0xf0, 0xf0, 0xff, 0xf0, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
0xf0, 0xcc, 0xee, 0xff, 0xf0, 0xf0, 0xf0, 0xf0, 0x30, 0x66, 0x30, 0x30, 0x30, 0x20, 0x30, 0x30,
0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xee, 0xf0, 0xf0, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
0xf0, 0xf0, 0xf0, 0xf0, 0xcc, 0xf0, 0xf0, 0xf0, 0x30, 0x30, 0x55, 0x30, 0x30, 0x44, 0x30, 0x30,
0xf0, 0xdd, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0x33, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xff, 0xf0, 0xf0, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x60, 0x30,
0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0x33, 0x33, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30,
0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x33, 0x30, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
0x30, 0x30, 0x30, 0x30, 0x30, 0x20, 0x30, 0x30, 0xf0, 0xff, 0xf0, 0xf0, 0xdd, 0xf0, 0xf0, 0xff,
0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x55, 0x33, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xff, 0xf0, 0xf0,
0x30, 0x44, 0x66, 0x30, 0x30, 0x30, 0x30, 0x30, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0xf0, 0xf0, 0xf0, 0xaa, 0xf0, 0xf0, 0xcc, 0xf0,
0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0xff, 0xf0, 0xf0, 0xf0, 0xff, 0xf0, 0xdd, 0xf0,
0x30, 0x30, 0x30, 0x77, 0x30, 0x30, 0x30, 0x30, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
};



#define MAX_PARTICLES   3000

#define LIFE_SPAN       8.f

#define BIRTH_INTERVAL (LIFE_SPAN/(float)MAX_PARTICLES)

#define PARTICLE_SIZE   0.7f

#define GRAVITY         9.8f

#define VELOCITY        8.f

#define FRICTION        0.75f

#define FOUNTAIN_HEIGHT 3.f

#define FOUNTAIN_RADIUS 1.6f

#define MIN_DELTA_T     (BIRTH_INTERVAL * 0.5f)



typedef struct {
float x,y,z;     
float vx,vy,vz;  
float r,g,b;     
float life;      
int   active;    
} PARTICLE;

static PARTICLE particles[MAX_PARTICLES];

static float min_age;

static float glow_color[4];

static float glow_pos[4];



const GLfloat fountain_diffuse[4]  = { 0.7f, 1.f,  1.f,  1.f };
const GLfloat fountain_specular[4] = {  1.f, 1.f,  1.f,  1.f };
const GLfloat fountain_shininess   = 12.f;
const GLfloat floor_diffuse[4]     = { 1.f,  0.6f, 0.6f, 1.f };
const GLfloat floor_specular[4]    = { 0.6f, 0.6f, 0.6f, 1.f };
const GLfloat floor_shininess      = 18.f;
const GLfloat fog_color[4]         = { 0.1f, 0.1f, 0.1f, 1.f };



static void usage(void)
{
printf("Usage: particles [-bfhs]\n");
printf("Options:\n");
printf(" -f   Run in full screen\n");
printf(" -h   Display this help\n");
printf(" -s   Run program as single thread (default is to use two threads)\n");
printf("\n");
printf("Program runtime controls:\n");
printf(" W    Toggle wireframe mode\n");
printf(" Esc  Exit program\n");
}



static void init_particle(PARTICLE *p, double t)
{
float xy_angle, velocity;

p->x = 0.f;
p->y = 0.f;
p->z = FOUNTAIN_HEIGHT;

p->vz = 0.7f + (0.3f / 4096.f) * (float) (rand() & 4095);

xy_angle = (2.f * (float) M_PI / 4096.f) * (float) (rand() & 4095);
p->vx = 0.4f * (float) cos(xy_angle);
p->vy = 0.4f * (float) sin(xy_angle);

velocity = VELOCITY * (0.8f + 0.1f * (float) (sin(0.5 * t) + sin(1.31 * t)));
p->vx *= velocity;
p->vy *= velocity;
p->vz *= velocity;

p->r = 0.7f + 0.3f * (float) sin(0.34 * t + 0.1);
p->g = 0.6f + 0.4f * (float) sin(0.63 * t + 1.1);
p->b = 0.6f + 0.4f * (float) sin(0.91 * t + 2.1);

glow_pos[0] = 0.4f * (float) sin(1.34 * t);
glow_pos[1] = 0.4f * (float) sin(3.11 * t);
glow_pos[2] = FOUNTAIN_HEIGHT + 1.f;
glow_pos[3] = 1.f;
glow_color[0] = p->r;
glow_color[1] = p->g;
glow_color[2] = p->b;
glow_color[3] = 1.f;

p->life = 1.f;
p->active = 1;
}



#define FOUNTAIN_R2 (FOUNTAIN_RADIUS+PARTICLE_SIZE/2)*(FOUNTAIN_RADIUS+PARTICLE_SIZE/2)

static void update_particle(PARTICLE *p, float dt)
{
if (!p->active)
return;

p->life -= dt * (1.f / LIFE_SPAN);

if (p->life <= 0.f)
{
p->active = 0;
return;
}

p->vz = p->vz - GRAVITY * dt;

p->x = p->x + p->vx * dt;
p->y = p->y + p->vy * dt;
p->z = p->z + p->vz * dt;

if (p->vz < 0.f)
{
if ((p->x * p->x + p->y * p->y) < FOUNTAIN_R2 &&
p->z < (FOUNTAIN_HEIGHT + PARTICLE_SIZE / 2))
{
p->vz = -FRICTION * p->vz;
p->z  = FOUNTAIN_HEIGHT + PARTICLE_SIZE / 2 +
FRICTION * (FOUNTAIN_HEIGHT +
PARTICLE_SIZE / 2 - p->z);
}

else if (p->z < PARTICLE_SIZE / 2)
{
p->vz = -FRICTION * p->vz;
p->z  = PARTICLE_SIZE / 2 +
FRICTION * (PARTICLE_SIZE / 2 - p->z);
}
}
}



static void particle_engine(double t, float dt)
{
int i;
float dt2;

while (dt > 0.f)
{
dt2 = dt < MIN_DELTA_T ? dt : MIN_DELTA_T;

for (i = 0;  i < MAX_PARTICLES;  i++)
update_particle(&particles[i], dt2);

min_age += dt2;

while (min_age >= BIRTH_INTERVAL)
{
min_age -= BIRTH_INTERVAL;

for (i = 0;  i < MAX_PARTICLES;  i++)
{
if (!particles[i].active)
{
init_particle(&particles[i], t + min_age);
update_particle(&particles[i], min_age);
break;
}
}
}

dt -= dt2;
}
}



#define BATCH_PARTICLES 70  
#define PARTICLE_VERTS  4   

static void draw_particles(GLFWwindow* window, double t, float dt)
{
int i, particle_count;
Vertex vertex_array[BATCH_PARTICLES * PARTICLE_VERTS];
Vertex* vptr;
float alpha;
GLuint rgba;
Vec3 quad_lower_left, quad_lower_right;
GLfloat mat[16];
PARTICLE* pptr;


glGetFloatv(GL_MODELVIEW_MATRIX, mat);

quad_lower_left.x = (-PARTICLE_SIZE / 2) * (mat[0] + mat[1]);
quad_lower_left.y = (-PARTICLE_SIZE / 2) * (mat[4] + mat[5]);
quad_lower_left.z = (-PARTICLE_SIZE / 2) * (mat[8] + mat[9]);
quad_lower_right.x = (PARTICLE_SIZE / 2) * (mat[0] - mat[1]);
quad_lower_right.y = (PARTICLE_SIZE / 2) * (mat[4] - mat[5]);
quad_lower_right.z = (PARTICLE_SIZE / 2) * (mat[8] - mat[9]);

glDepthMask(GL_FALSE);

glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE);

if (!wireframe)
{
glEnable(GL_TEXTURE_2D);
glBindTexture(GL_TEXTURE_2D, particle_tex_id);
}

glInterleavedArrays(GL_T2F_C4UB_V3F, 0, vertex_array);

mtx_lock(&thread_sync.particles_lock);
while (!glfwWindowShouldClose(window) &&
thread_sync.p_frame <= thread_sync.d_frame)
{
struct timespec ts;
clock_gettime(CLOCK_REALTIME, &ts);
ts.tv_nsec += 100 * 1000 * 1000;
ts.tv_sec += ts.tv_nsec / (1000 * 1000 * 1000);
ts.tv_nsec %= 1000 * 1000 * 1000;
cnd_timedwait(&thread_sync.p_done, &thread_sync.particles_lock, &ts);
}

thread_sync.t = t;
thread_sync.dt = dt;

thread_sync.d_frame++;

particle_count = 0;
vptr = vertex_array;
pptr = particles;

for (i = 0;  i < MAX_PARTICLES;  i++)
{
if (pptr->active)
{
alpha =  4.f * pptr->life;
if (alpha > 1.f)
alpha = 1.f;

((GLubyte*) &rgba)[0] = (GLubyte)(pptr->r * 255.f);
((GLubyte*) &rgba)[1] = (GLubyte)(pptr->g * 255.f);
((GLubyte*) &rgba)[2] = (GLubyte)(pptr->b * 255.f);
((GLubyte*) &rgba)[3] = (GLubyte)(alpha * 255.f);


vptr->s    = 0.f;
vptr->t    = 0.f;
vptr->rgba = rgba;
vptr->x    = pptr->x + quad_lower_left.x;
vptr->y    = pptr->y + quad_lower_left.y;
vptr->z    = pptr->z + quad_lower_left.z;
vptr ++;

vptr->s    = 1.f;
vptr->t    = 0.f;
vptr->rgba = rgba;
vptr->x    = pptr->x + quad_lower_right.x;
vptr->y    = pptr->y + quad_lower_right.y;
vptr->z    = pptr->z + quad_lower_right.z;
vptr ++;

vptr->s    = 1.f;
vptr->t    = 1.f;
vptr->rgba = rgba;
vptr->x    = pptr->x - quad_lower_left.x;
vptr->y    = pptr->y - quad_lower_left.y;
vptr->z    = pptr->z - quad_lower_left.z;
vptr ++;

vptr->s    = 0.f;
vptr->t    = 1.f;
vptr->rgba = rgba;
vptr->x    = pptr->x - quad_lower_right.x;
vptr->y    = pptr->y - quad_lower_right.y;
vptr->z    = pptr->z - quad_lower_right.z;
vptr ++;

particle_count ++;
}

if (particle_count >= BATCH_PARTICLES)
{
glDrawArrays(GL_QUADS, 0, PARTICLE_VERTS * particle_count);
particle_count = 0;
vptr = vertex_array;
}

pptr++;
}

mtx_unlock(&thread_sync.particles_lock);
cnd_signal(&thread_sync.d_done);

glDrawArrays(GL_QUADS, 0, PARTICLE_VERTS * particle_count);

glDisableClientState(GL_VERTEX_ARRAY);
glDisableClientState(GL_TEXTURE_COORD_ARRAY);
glDisableClientState(GL_COLOR_ARRAY);

glDisable(GL_TEXTURE_2D);
glDisable(GL_BLEND);

glDepthMask(GL_TRUE);
}



#define FOUNTAIN_SIDE_POINTS 14
#define FOUNTAIN_SWEEP_STEPS 32

static const float fountain_side[FOUNTAIN_SIDE_POINTS * 2] =
{
1.2f, 0.f,  1.f, 0.2f,  0.41f, 0.3f, 0.4f, 0.35f,
0.4f, 1.95f, 0.41f, 2.f, 0.8f, 2.2f,  1.2f, 2.4f,
1.5f, 2.7f,  1.55f,2.95f, 1.6f, 3.f,  1.f, 3.f,
0.5f, 3.f,  0.f, 3.f
};

static const float fountain_normal[FOUNTAIN_SIDE_POINTS * 2] =
{
1.0000f, 0.0000f,  0.6428f, 0.7660f,  0.3420f, 0.9397f,  1.0000f, 0.0000f,
1.0000f, 0.0000f,  0.3420f,-0.9397f,  0.4226f,-0.9063f,  0.5000f,-0.8660f,
0.7660f,-0.6428f,  0.9063f,-0.4226f,  0.0000f,1.00000f,  0.0000f,1.00000f,
0.0000f,1.00000f,  0.0000f,1.00000f
};



static void draw_fountain(void)
{
static GLuint fountain_list = 0;
double angle;
float  x, y;
int m, n;

if (!fountain_list)
{
fountain_list = glGenLists(1);
glNewList(fountain_list, GL_COMPILE_AND_EXECUTE);

glMaterialfv(GL_FRONT, GL_DIFFUSE, fountain_diffuse);
glMaterialfv(GL_FRONT, GL_SPECULAR, fountain_specular);
glMaterialf(GL_FRONT, GL_SHININESS, fountain_shininess);

for (n = 0;  n < FOUNTAIN_SIDE_POINTS - 1;  n++)
{
glBegin(GL_TRIANGLE_STRIP);
for (m = 0;  m <= FOUNTAIN_SWEEP_STEPS;  m++)
{
angle = (double) m * (2.0 * M_PI / (double) FOUNTAIN_SWEEP_STEPS);
x = (float) cos(angle);
y = (float) sin(angle);

glNormal3f(x * fountain_normal[n * 2 + 2],
y * fountain_normal[n * 2 + 2],
fountain_normal[n * 2 + 3]);
glVertex3f(x * fountain_side[n * 2 + 2],
y * fountain_side[n * 2 + 2],
fountain_side[n * 2 +3 ]);
glNormal3f(x * fountain_normal[n * 2],
y * fountain_normal[n * 2],
fountain_normal[n * 2 + 1]);
glVertex3f(x * fountain_side[n * 2],
y * fountain_side[n * 2],
fountain_side[n * 2 + 1]);
}

glEnd();
}

glEndList();
}
else
glCallList(fountain_list);
}



static void tessellate_floor(float x1, float y1, float x2, float y2, int depth)
{
float delta, x, y;

if (depth >= 5)
delta = 999999.f;
else
{
x = (float) (fabs(x1) < fabs(x2) ? fabs(x1) : fabs(x2));
y = (float) (fabs(y1) < fabs(y2) ? fabs(y1) : fabs(y2));
delta = x*x + y*y;
}

if (delta < 0.1f)
{
x = (x1 + x2) * 0.5f;
y = (y1 + y2) * 0.5f;
tessellate_floor(x1, y1,  x,  y, depth + 1);
tessellate_floor(x, y1, x2,  y, depth + 1);
tessellate_floor(x1,  y,  x, y2, depth + 1);
tessellate_floor(x,  y, x2, y2, depth + 1);
}
else
{
glTexCoord2f(x1 * 30.f, y1 * 30.f);
glVertex3f(  x1 * 80.f, y1 * 80.f, 0.f);
glTexCoord2f(x2 * 30.f, y1 * 30.f);
glVertex3f(  x2 * 80.f, y1 * 80.f, 0.f);
glTexCoord2f(x2 * 30.f, y2 * 30.f);
glVertex3f(  x2 * 80.f, y2 * 80.f, 0.f);
glTexCoord2f(x1 * 30.f, y2 * 30.f);
glVertex3f(  x1 * 80.f, y2 * 80.f, 0.f);
}
}



static void draw_floor(void)
{
static GLuint floor_list = 0;

if (!wireframe)
{
glEnable(GL_TEXTURE_2D);
glBindTexture(GL_TEXTURE_2D, floor_tex_id);
}

if (!floor_list)
{
floor_list = glGenLists(1);
glNewList(floor_list, GL_COMPILE_AND_EXECUTE);

glMaterialfv(GL_FRONT, GL_DIFFUSE, floor_diffuse);
glMaterialfv(GL_FRONT, GL_SPECULAR, floor_specular);
glMaterialf(GL_FRONT, GL_SHININESS, floor_shininess);

glNormal3f(0.f, 0.f, 1.f);
glBegin(GL_QUADS);
tessellate_floor(-1.f, -1.f, 0.f, 0.f, 0);
tessellate_floor( 0.f, -1.f, 1.f, 0.f, 0);
tessellate_floor( 0.f,  0.f, 1.f, 1.f, 0);
tessellate_floor(-1.f,  0.f, 0.f, 1.f, 0);
glEnd();

glEndList();
}
else
glCallList(floor_list);

glDisable(GL_TEXTURE_2D);

}



static void setup_lights(void)
{
float l1pos[4], l1amb[4], l1dif[4], l1spec[4];
float l2pos[4], l2amb[4], l2dif[4], l2spec[4];

l1pos[0] =  0.f;  l1pos[1] = -9.f; l1pos[2] =   8.f;  l1pos[3] = 1.f;
l1amb[0] = 0.2f;  l1amb[1] = 0.2f;  l1amb[2] = 0.2f;  l1amb[3] = 1.f;
l1dif[0] = 0.8f;  l1dif[1] = 0.4f;  l1dif[2] = 0.2f;  l1dif[3] = 1.f;
l1spec[0] = 1.f; l1spec[1] = 0.6f; l1spec[2] = 0.2f; l1spec[3] = 0.f;

l2pos[0] =  -15.f; l2pos[1] =  12.f; l2pos[2] = 1.5f; l2pos[3] =  1.f;
l2amb[0] =    0.f; l2amb[1] =   0.f; l2amb[2] =  0.f; l2amb[3] =  1.f;
l2dif[0] =   0.2f; l2dif[1] =  0.4f; l2dif[2] = 0.8f; l2dif[3] =  1.f;
l2spec[0] =  0.2f; l2spec[1] = 0.6f; l2spec[2] = 1.f; l2spec[3] = 0.f;

glLightfv(GL_LIGHT1, GL_POSITION, l1pos);
glLightfv(GL_LIGHT1, GL_AMBIENT, l1amb);
glLightfv(GL_LIGHT1, GL_DIFFUSE, l1dif);
glLightfv(GL_LIGHT1, GL_SPECULAR, l1spec);
glLightfv(GL_LIGHT2, GL_POSITION, l2pos);
glLightfv(GL_LIGHT2, GL_AMBIENT, l2amb);
glLightfv(GL_LIGHT2, GL_DIFFUSE, l2dif);
glLightfv(GL_LIGHT2, GL_SPECULAR, l2spec);
glLightfv(GL_LIGHT3, GL_POSITION, glow_pos);
glLightfv(GL_LIGHT3, GL_DIFFUSE, glow_color);
glLightfv(GL_LIGHT3, GL_SPECULAR, glow_color);

glEnable(GL_LIGHT1);
glEnable(GL_LIGHT2);
glEnable(GL_LIGHT3);
}



static void draw_scene(GLFWwindow* window, double t)
{
double xpos, ypos, zpos, angle_x, angle_y, angle_z;
static double t_old = 0.0;
float dt;
mat4x4 projection;

dt = (float) (t - t_old);
t_old = t;

mat4x4_perspective(projection,
65.f * (float) M_PI / 180.f,
aspect_ratio,
1.0, 60.0);

glClearColor(0.1f, 0.1f, 0.1f, 1.f);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

glMatrixMode(GL_PROJECTION);
glLoadMatrixf((const GLfloat*) projection);

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();

angle_x = 90.0 - 10.0;
angle_y = 10.0 * sin(0.3 * t);
angle_z = 10.0 * t;
glRotated(-angle_x, 1.0, 0.0, 0.0);
glRotated(-angle_y, 0.0, 1.0, 0.0);
glRotated(-angle_z, 0.0, 0.0, 1.0);

xpos =  15.0 * sin((M_PI / 180.0) * angle_z) +
2.0 * sin((M_PI / 180.0) * 3.1 * t);
ypos = -15.0 * cos((M_PI / 180.0) * angle_z) +
2.0 * cos((M_PI / 180.0) * 2.9 * t);
zpos = 4.0 + 2.0 * cos((M_PI / 180.0) * 4.9 * t);
glTranslated(-xpos, -ypos, -zpos);

glFrontFace(GL_CCW);
glCullFace(GL_BACK);
glEnable(GL_CULL_FACE);

setup_lights();
glEnable(GL_LIGHTING);

glEnable(GL_FOG);
glFogi(GL_FOG_MODE, GL_EXP);
glFogf(GL_FOG_DENSITY, 0.05f);
glFogfv(GL_FOG_COLOR, fog_color);

draw_floor();

glEnable(GL_DEPTH_TEST);
glDepthFunc(GL_LEQUAL);
glDepthMask(GL_TRUE);

draw_fountain();

glDisable(GL_LIGHTING);
glDisable(GL_FOG);

draw_particles(window, t, dt);

glDisable(GL_DEPTH_TEST);
}



static void resize_callback(GLFWwindow* window, int width, int height)
{
glViewport(0, 0, width, height);
aspect_ratio = height ? width / (float) height : 1.f;
}



static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
if (action == GLFW_PRESS)
{
switch (key)
{
case GLFW_KEY_ESCAPE:
glfwSetWindowShouldClose(window, GLFW_TRUE);
break;
case GLFW_KEY_W:
wireframe = !wireframe;
glPolygonMode(GL_FRONT_AND_BACK,
wireframe ? GL_LINE : GL_FILL);
break;
default:
break;
}
}
}



static int physics_thread_main(void* arg)
{
GLFWwindow* window = arg;

for (;;)
{
mtx_lock(&thread_sync.particles_lock);

while (!glfwWindowShouldClose(window) &&
thread_sync.p_frame > thread_sync.d_frame)
{
struct timespec ts;
clock_gettime(CLOCK_REALTIME, &ts);
ts.tv_nsec += 100 * 1000 * 1000;
ts.tv_sec += ts.tv_nsec / (1000 * 1000 * 1000);
ts.tv_nsec %= 1000 * 1000 * 1000;
cnd_timedwait(&thread_sync.d_done, &thread_sync.particles_lock, &ts);
}

if (glfwWindowShouldClose(window))
break;

particle_engine(thread_sync.t, thread_sync.dt);

thread_sync.p_frame++;

mtx_unlock(&thread_sync.particles_lock);
cnd_signal(&thread_sync.p_done);
}

return 0;
}



int main(int argc, char** argv)
{
int ch, width, height;
thrd_t physics_thread = 0;
GLFWwindow* window;
GLFWmonitor* monitor = NULL;

if (!glfwInit())
{
fprintf(stderr, "Failed to initialize GLFW\n");
exit(EXIT_FAILURE);
}

while ((ch = getopt(argc, argv, "fh")) != -1)
{
switch (ch)
{
case 'f':
monitor = glfwGetPrimaryMonitor();
break;
case 'h':
usage();
exit(EXIT_SUCCESS);
}
}

if (monitor)
{
const GLFWvidmode* mode = glfwGetVideoMode(monitor);

glfwWindowHint(GLFW_RED_BITS, mode->redBits);
glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

width  = mode->width;
height = mode->height;
}
else
{
width  = 640;
height = 480;
}

window = glfwCreateWindow(width, height, "Particle Engine", monitor, NULL);
if (!window)
{
fprintf(stderr, "Failed to create GLFW window\n");
glfwTerminate();
exit(EXIT_FAILURE);
}

if (monitor)
glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

glfwMakeContextCurrent(window);
gladLoadGL(glfwGetProcAddress);
glfwSwapInterval(1);

glfwSetFramebufferSizeCallback(window, resize_callback);
glfwSetKeyCallback(window, key_callback);

glfwGetFramebufferSize(window, &width, &height);
resize_callback(window, width, height);

glGenTextures(1, &particle_tex_id);
glBindTexture(GL_TEXTURE_2D, particle_tex_id);
glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, P_TEX_WIDTH, P_TEX_HEIGHT,
0, GL_LUMINANCE, GL_UNSIGNED_BYTE, particle_texture);

glGenTextures(1, &floor_tex_id);
glBindTexture(GL_TEXTURE_2D, floor_tex_id);
glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, F_TEX_WIDTH, F_TEX_HEIGHT,
0, GL_LUMINANCE, GL_UNSIGNED_BYTE, floor_texture);

if (glfwExtensionSupported("GL_EXT_separate_specular_color"))
{
glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL_EXT,
GL_SEPARATE_SPECULAR_COLOR_EXT);
}

glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
wireframe = 0;

thread_sync.t  = 0.0;
thread_sync.dt = 0.001f;
thread_sync.p_frame = 0;
thread_sync.d_frame = 0;

mtx_init(&thread_sync.particles_lock, mtx_timed);
cnd_init(&thread_sync.p_done);
cnd_init(&thread_sync.d_done);

if (thrd_create(&physics_thread, physics_thread_main, window) != thrd_success)
{
glfwTerminate();
exit(EXIT_FAILURE);
}

glfwSetTime(0.0);

while (!glfwWindowShouldClose(window))
{
draw_scene(window, glfwGetTime());

glfwSwapBuffers(window);
glfwPollEvents();
}

thrd_join(physics_thread, NULL);

glfwDestroyWindow(window);
glfwTerminate();

exit(EXIT_SUCCESS);
}

