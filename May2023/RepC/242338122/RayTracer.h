#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __GNUC__
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
#if defined(PROFILE) && defined(__GNUC__)
#define PROFILEFUNC __attribute__((noinline))
#define PROFILEorINLINE __attribute__((noinline))
#else
#define PROFILEFUNC 
#define PROFILEorINLINE inline
#endif
#define IMP 0.1
#define TDS(X) typedef struct X X;
#define MAX_FLOAT 9999999999999
TDS(Point)
TDS(Vec3)
TDS(Camera)
TDS(Ray)
TDS(Color)
TDS(Sphere)
TDS(Object)
TDS(Light)
TDS(Scene)
TDS(Image)
int X,Y,HX,HY;
float FX=0;
float fastInvSqrt(float x) {
int i = *(int*)&x;
i = 0x5f3759df - (i >> 1);
float y = *(float*)&i;
return y * (1.5F - 0.5F * x * y * y);
}
struct Point
{
float x,y,z;
};
struct Vec3
{
float x,y,z;
};
Vec3  Vadd(Vec3 a,Vec3 b)
{
return (struct Vec3){a.x+b.x,a.y+b.y,a.z+b.z};
}
Point  Pshift(Point a,Vec3 b)
{
return (struct Point){a.x+b.x,a.y+b.y,a.z+b.z};
}
PROFILEFUNC
Vec3  Psub(Point a,Point b)
{
return (struct Vec3){a.x-b.x,a.y-b.y,a.z-b.z};
}
Point  pPshift(Point *a,Vec3 *b)
{
a->x+=b->x;
a->y+=b->y;
a->z+=b->z;
}
Vec3  Vsub(Vec3 a,Vec3 b)
{
return (struct Vec3){a.x-b.x,a.y-b.y,a.z-b.z};
}
PROFILEFUNC
float VSmul(Vec3 a,Vec3 b)
{
return a.x*b.x+a.y*b.y+a.z*b.z;
}
Vec3  VCross(Vec3 a,Vec3 b)
{
return (struct Vec3){a.y*b.z-a.z*b.y,
a.z*b.x-a.x*b.z,
a.x*b.y-a.y*b.x};
}
void  pNorm(Vec3 *a)
{
#ifdef FFAST
float imag=fastInvSqrt(a->x*a->x+a->y*a->y+a->z*a->z);
#else
float imag=1/sqrt(a->x*a->x+a->y*a->y+a->z*a->z);
#endif
a->x*=imag;
a->y*=imag;
a->z*=imag;
}
PROFILEFUNC
Vec3  Norm(Vec3 a)
{
float imag=fastInvSqrt(a.x*a.x+a.y*a.y+a.z*a.z);
return (struct Vec3){a.x*imag,a.y*imag,a.z*imag};
}
PROFILEFUNC
float VMag(Vec3 a)
{
return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);
}
Vec3  Vscale(Vec3 a,float mag)
{
return (struct Vec3){a.x*mag,a.y*mag,a.z*mag};
}
struct Camera
{
Point pos;
Vec3 dir;
};
struct Ray
{
Point start;
Vec3 dir;
};
struct Color
{
unsigned char r,g,b;
};
Color CScale(float A,Color C)
{
int r=A*C.r,g=A*C.g,b=A*C.b;
Color c;
c.r=r>255 ? 255 : (r<0 ? 0 : r);
c.g=g>255 ? 255 : (g<0 ? 0 : g);
c.b=b>255 ? 255 : (b<0 ? 0 : b);
return c;
}
Color CMul(Color A,Color B)
{
return(struct Color){((int)A.r)*B.r/255,((int)A.g)*B.g/255,((int)A.b*B.b)/255};
}
Color CSCom(Color A,Color B,float b)
{
float a=1-b;
return(struct Color){A.r*a+B.r*b,A.g*a+B.g*b,A.b*a+B.b*b};
}
Color CAdd(Color A,Color B)
{
return(struct Color){A.r+B.r,A.g+B.g,A.b+B.b};
}
struct Image
{
int width,height;
Color *Pixels;
};
Image NewImage(int x,int y)
{
return (struct Image){x,y,(Color*)malloc(x*y*sizeof(Color))};
}
struct Sphere
{
Point C;
float r;
};
#define SPHERE 1
struct Object
{
union
{
Sphere sphere;
}u;
unsigned char type;
Color color;    
float diffuse,reflection;
};
struct Light
{
Point pos;
Color color;
float x;
};
struct Scene
{
int numObj,numLight;
Object *Objects;
Light  *Lights;
};
PROFILEorINLINE
float intersect(Ray ray,Object obj)
{
if(likely(obj.type=SPHERE))
{
Vec3 CamtoCen=Psub(obj.u.sphere.C,ray.start);
float tca =VSmul(CamtoCen,ray.dir);      
if (unlikely(tca < 0))return -0.0;
float dist2 = VSmul(CamtoCen,CamtoCen) - tca * tca;
float r2=obj.u.sphere.r;r2*=r2;
return dist2 > r2? -0.0: tca-sqrt(r2 - dist2);          
}
}
PROFILEorINLINE
Color shade(Point P,int obji,int numObj,Object *Objects,int NL,Light* Lights)
{
Object Obj=Objects[obji];
if(Obj.type=SPHERE)
{
Vec3 SufNorm=Psub(P,Obj.u.sphere.C);
pNorm(&SufNorm);
int i=0;      
float g,b,r=g=b=0,x;       
for(i=0;i<NL;i++)
{
Light light=Lights[i];
Vec3 PtoLight=Psub(light.pos,P);
float dist=VMag(PtoLight);
float InvDist=1/dist;
PtoLight=Vscale(PtoLight,InvDist);
int j;
float min=MAX_FLOAT;
for(j=0;j<numObj;j++)
min=(x=intersect((struct Ray){P,PtoLight},Objects[j]))?(x<min?x:min):min;
if(min!=MAX_FLOAT&&min<dist) continue;
InvDist*=InvDist;
x=VSmul(SufNorm,PtoLight)*light.x*InvDist;
r+=x>0?x*light.color.r:0;
g+=x>0?x*light.color.g:0;
b+=x>0?x*light.color.b:0;
}
unsigned char r2,g2,b2;
r*=Obj.color.r;
g*=Obj.color.g;
b*=Obj.color.b;
r2=r<255?r:255;
g2=g<255?g:255;
b2=b<255?b:255;
return (struct Color){r2,g2,b2};
}
}
#ifdef WIN32
inline
#else
PROFILEFUNC
#endif 
Color RayTrace(const Ray ray,Scene scene,const float imp)
{
const int numObj=scene.numObj;
int i=0,firsti=-1;float min=MAX_FLOAT,dist;
Object *O=scene.Objects;
Object obj;
for(i=0;i<numObj;i++)
{   
dist=intersect(ray,O[i]);
firsti=(dist&&dist<min)?i:firsti;
min=(dist&&dist<min)?dist:min;       
}
if(firsti==-1)return (struct Color){255,255,255};
obj=O[firsti];
if(likely(obj.type==SPHERE))
{
Point intersectionP=Pshift(ray.start,Vscale(ray.dir,min));
Color c=CScale(obj.diffuse,shade(intersectionP,firsti,numObj,O,scene.numLight,scene.Lights));
if(unlikely(obj.reflection*imp>IMP))
c=CSCom(c,RayTrace((struct Ray){intersectionP, Norm(Psub(intersectionP,obj.u.sphere.C))},scene,obj.reflection*imp),obj.reflection);
return c;
}
} 
Ray RayofPix(Point cam_pos,Vec3 cam_dir,Vec3 up,Vec3 right,float x,float y)
{
Vec3 dir=Vadd(cam_dir,Vadd(Vscale(right,(x-HX)*FX),Vscale(up,(y-HY)*FX)));
pNorm(&dir);
return (struct Ray){cam_pos,dir};
}
void RTRender(Image* I,Camera cam,Scene scene)
{
HX=X/2,HY=Y/2;
if(FX==0)
FX=VMag(cam.dir)/X;
Point cam_pos=cam.pos;
Vec3 cam_dir=cam.dir;pNorm(&cam_dir);
Vec3 up,right;
right=VCross(cam_dir,(struct Vec3){0,-1,0});pNorm(&right);
up=VCross(cam_dir,right);
if(up.y<0)up=VCross(right,cam_dir);
Color* P=I->Pixels;
const int numObj=scene.numObj;
#if defined(WIN32) || defined(_WIN32)
#define RTNoInline
#endif
int x,y;
#pragma omp parallel for private(x) 
for(y=0;y<Y;y++)
#pragma omp task
for(x=0;x<X;x++)
#if !defined(DoF) && !defined(AA) && defined(RTNoInline)
P[y*X+x]=RayTrace(RayofPix(cam.pos,cam_dir,up,right,x,y),scene,1.0);
#elif !defined(DoF) && !defined(AA) 
{                 
Ray ray=RayofPix(cam_pos,cam_dir,up,right,x,y);
int i=0,firsti=-1;float min=MAX_FLOAT,dist;
Object *O=scene.Objects;
Object obj;
for(i=0;i<numObj;i++)
{   
dist=intersect(ray,O[i]);
firsti=(dist&&dist<min)?i:firsti;
min=(dist&&dist<min)?dist:min;       
}
if(firsti==-1){P[y*X+x]=(struct Color){255,255,255};continue;}
obj=O[firsti];
if(likely(obj.type==SPHERE))
{
Point intersectionP=Pshift(ray.start,Vscale(ray.dir,min));
Color c=CScale(obj.diffuse,shade(intersectionP,firsti,numObj,O,scene.numLight,scene.Lights));
if(unlikely(obj.reflection>IMP))
c=CSCom(c,RayTrace((struct Ray){intersectionP, Norm(Psub(intersectionP,obj.u.sphere.C))},scene,obj.reflection),obj.reflection);
P[y*X+x]=c;
}
}
#elif defined(DoF) defined(DoFy)
{
Point lcam_pos=cam_pos;
P[y*X+x]=RayTrace(RayofPix(cam_pos,cam_dir,up,right,x,y),scene,1.0);
lcam_pos=Pshift(Pshift(cam.pos,Vscale(up,DoF)),Vscale(right,DoF));
P[y*X+x]=CSCom(P[y*X+x],RayTrace(RayofPix(lcam_pos,cam_dir,up,right,x-DoFy,y-DoFy),scene,1.0),1/2.0);
lcam_pos=Pshift(Pshift(cam.pos,Vscale(up,-DoF)),Vscale(right,DoF));
P[y*X+x]=CSCom(P[y*X+x],RayTrace(RayofPix(lcam_pos,cam_dir,up,right,x-DoFy,y+DoFy),scene,1.0),1/3.0);
lcam_pos=Pshift(Pshift(cam.pos,Vscale(up,DoF)),Vscale(right,-DoF));
P[y*X+x]=CSCom(P[y*X+x],RayTrace(RayofPix(lcam_pos,cam_dir,up,right,x+DoFy,y-DoFy),scene,1.0),1/4.0);
lcam_pos=Pshift(Pshift(cam.pos,Vscale(up,-DoF)),Vscale(right,-DoF));
P[y*X+x]=CSCom(P[y*X+x],RayTrace(RayofPix(lcam_pos,cam_dir,up,right,x+DoFy,y+DoFy),scene,1.0),1/5.0);
}
#else 
{
P[y*X+x]=RayTrace(RayofPix(cam_pos,cam_dir,up,right,x,y),scene,1.0);
P[y*X+x]=CSCom(P[y*X+x],RayTrace(RayofPix(cam_pos,cam_dir,up,right,x+0.25,y+0.25),scene,1.0),1/2.0);
P[y*X+x]=CSCom(P[y*X+x],RayTrace(RayofPix(cam_pos,cam_dir,up,right,x-0.25,y+0.25),scene,1.0),1/3.0);
P[y*X+x]=CSCom(P[y*X+x],RayTrace(RayofPix(cam_pos,cam_dir,up,right,x+0.25,y-0.25),scene,1.0),1/4.0);
P[y*X+x]=CSCom(P[y*X+x],RayTrace(RayofPix(cam_pos,cam_dir,up,right,x-0.25,y-0.25),scene,1.0),1/5.0);
}
#endif
}
