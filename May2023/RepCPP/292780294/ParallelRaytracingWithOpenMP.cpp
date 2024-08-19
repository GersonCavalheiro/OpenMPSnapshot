




#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "geometry.h"
#include <sys/time.h>
#include <time.h>




double timef_()
{


double msec;
struct timeval tv;
gettimeofday(&tv, 0);
msec = tv.tv_sec * 1.0e3 + tv.tv_usec * 1.0e-3;
return msec;
}

float second_()
{
return (float)clock() / CLOCKS_PER_SEC;
}



struct Light {
Light(const Vec3f& p, const float i) : position(p), intensity(i) {}
Vec3f position;
float intensity;
};



struct objecttype {
objecttype(const float r, const Vec4f& a, const Vec3f& color, const float spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
objecttype() : refractive_index(1), albedo(1, 0, 0, 0), diffuse_color(), specular_exponent() {}
float refractive_index;
Vec4f albedo;
Vec3f diffuse_color;
float specular_exponent;
};


struct Sphere {

Vec3f center;
float radius;
objecttype object;

Sphere(const Vec3f& c, const float r, const objecttype& m) : center(c), radius(r), object(m) {}

bool ray_intersect(const Vec3f& orig, const Vec3f& direction, float& t0) const {



Vec3f L = center - orig;  
float tca = L * direction;         
float d2 = L * L - tca * tca;  
if (d2 > radius * radius) return false;  
float thc = sqrtf(radius * radius - d2);  


t0 = tca - thc;
float t1 = tca + thc;
if (t0 < 0) t0 = t1;
if (t0 < 0) return false;
return true;
}
};

Vec3f reflect(const Vec3f& I, const Vec3f& N) {

return I - N * 2.f * (I * N);
}

Vec3f refract(const Vec3f& I, const Vec3f& N, const float eta_t, const float eta_i = 1.f) {



float cosi = -std::max(-1.f, std::min(1.f, I * N));
if (cosi < 0) return refract(I, -N, eta_i, eta_t); 
float eta = eta_i / eta_t;
float k = 1 - eta * eta * (1 - cosi * cosi);
return k < 0 ? Vec3f(1, 0, 0) : I * eta + N * (eta * cosi - sqrtf(k)); 
}


bool scene_intersect(const Vec3f& orig, const Vec3f& direction, const std::vector<Sphere>& spheres, Vec3f& hit, Vec3f& N, objecttype& object) {



float spheres_dist = std::numeric_limits<float>::max();
for (size_t i = 0; i < spheres.size(); i++) {
float dist_i;
if (spheres[i].ray_intersect(orig, direction, dist_i) && dist_i < spheres_dist) { 
spheres_dist = dist_i;
hit = orig + direction * dist_i; 
N = (hit - spheres[i].center).normalize(); 
object = spheres[i].object;
}
}


float plain_surface_dist = std::numeric_limits<float>::max();
if (fabs(direction.y) > 1e-3) {
float d = -(orig.y + 4) / direction.y; 
Vec3f pt = orig + direction * d;
if (d > 0 && fabs(pt.x) < 10 && pt.z<-10 && pt.z>-30 && d < spheres_dist) {
plain_surface_dist = d; 
hit = pt;
N = Vec3f(0, 1, 0);
object.diffuse_color = (int(.5 * hit.x + 1000) + int(.5 * hit.z)) & 1 ? Vec3f(.3, .2, .1) : Vec3f(.3, .2, .1);
}
}
return std::min(spheres_dist, plain_surface_dist) < 1000;
}

Vec3f cast_ray(const Vec3f& orig, const Vec3f& direction, const std::vector<Sphere>& spheres, const std::vector<Light>& lights, size_t depth = 0) {



Vec3f point, N;
objecttype object;
vec<3, float> diffuse, specular, reflection, refraction;

if (depth > 4 || !scene_intersect(orig, direction, spheres, point, N, object)) {
return Vec3f(0.1, 0.4, 0.5); 
}

Vec3f reflect_direction = reflect(direction, N).normalize();
Vec3f refract_direction = refract(direction, N, object.refractive_index).normalize();
Vec3f refract_orig = refract_direction * N < 0 ? point - N * 1e-3 : point + N * 1e-3;


Vec3f reflect_orig = reflect_direction * N < 0 ? point - N * 1e-3 : point + N * 1e-3; 
Vec3f reflect_color = cast_ray(reflect_orig, reflect_direction, spheres, lights, depth + 1);
Vec3f refract_color = cast_ray(refract_orig, refract_direction, spheres, lights, depth + 1);



float diffuse_light_intensity = 0, specular_light_intensity = 0;

for (size_t i = 0; i < lights.size(); i++) {
Vec3f light_direction = (lights[i].position - point).normalize();
float light_distance = (lights[i].position - point).norm();



Vec3f shadow_orig = light_direction * N < 0 ? point - N * 1e-3 : point + N * 1e-3; 
Vec3f shadow_pt, shadow_N;
objecttype tmpmaterial;
if (scene_intersect(shadow_orig, light_direction, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt - shadow_orig).norm() < light_distance)
continue;

diffuse_light_intensity += lights[i].intensity * std::max(0.f, light_direction * N); 
specular_light_intensity += powf(std::max(0.f, -reflect(-light_direction, N) * direction), object.specular_exponent) * lights[i].intensity; 
}



diffuse = object.diffuse_color * diffuse_light_intensity * object.albedo[0];
specular = Vec3f(1., 1., 1.) * specular_light_intensity * object.albedo[1];
reflection = reflect_color * object.albedo[2];
refraction = refract_color * object.albedo[3];

return    diffuse + specular + reflection + refraction;
}



int main() {




objecttype     gray_sphere(1.0, Vec4f(0.6, 0.3, 0.1, 0.0), Vec3f(0.4, 0.4, 0.3), 50.);
objecttype     violet_sphere(1.0, Vec4f(0.9, 0.1, 0.0, 0.0), Vec3f(0.2, 0.1, 0.3), 10.);
objecttype     left_mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);
objecttype     right_mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);
objecttype     transparent_sphere_1(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.3, 0.8), 75.);
objecttype     transparent_sphere_2(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.3, 0.8), 75.);
objecttype     transparent_sphere_3(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.3, 0.8), 75.);
objecttype     transparent_sphere_4(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.3, 0.8), 75.);
objecttype     transparent_sphere_5(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.3, 0.8), 75.);
objecttype     transparent_sphere_6(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.3, 0.8), 75.);
objecttype     transparent_sphere_7(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.2, 0.8), 75.);



std::vector<Sphere> spheres;


spheres.push_back(Sphere(Vec3f(-3, 0, -16), 2, gray_sphere));
spheres.push_back(Sphere(Vec3f(1.5, -0.5, -18), 3, violet_sphere));
spheres.push_back(Sphere(Vec3f(-8, 5, -17), 4, left_mirror));
spheres.push_back(Sphere(Vec3f(7, 5, -18), 4, right_mirror));
spheres.push_back(Sphere(Vec3f(-1.0, -1.5, -12), 2, transparent_sphere_1));
spheres.push_back(Sphere(Vec3f(1.3, -2.5, -9), 0.85, transparent_sphere_2));
spheres.push_back(Sphere(Vec3f(-3.1, -2.5, -9), 0.85, transparent_sphere_3));
spheres.push_back(Sphere(Vec3f(3.0, -2.5, -9), 0.85, transparent_sphere_4));
spheres.push_back(Sphere(Vec3f(-4.30, -2.5, -9), 0.85, transparent_sphere_5));
spheres.push_back(Sphere(Vec3f(4.67, -2.5, -9), 0.85, transparent_sphere_6));
spheres.push_back(Sphere(Vec3f(-5.70, -2.5, -9), 0.85, transparent_sphere_7));


std::vector<Light>  lights;



lights.push_back(Light(Vec3f(-20, 20, 20), 1.5));
lights.push_back(Light(Vec3f(30, 50, -25), 1.8));
lights.push_back(Light(Vec3f(30, 20, 30), 1.7));

std::cout << "Parallel RayTracing is Running...." << std::endl;

double t_start, t_end; 







t_start = timef_() / 1000.0; 

std::cout << "Raytracing Algorithm Started..........."<< std::endl;

const int   rows = 1024; 
const int   cols = 768;  
const float fov = M_PI / 3.; 
std::vector<Vec3f> image(rows * cols); 



#pragma omp parallel for  
for (size_t j = 0; j < cols; j++) { 
for (size_t i = 0; i < rows; i++) {
float directionX = (i + 0.5) - rows / 2.;
float directionY = -(j + 0.5) + cols / 2.;    
float directionZ = -cols / (2. * tan(fov / 2.));
image[i + j * rows] = cast_ray(Vec3f(0, 0, 0), Vec3f(directionX, directionY, directionZ).normalize(), spheres, lights);
}
}

t_end = timef_() / 1000.0; 
std::cout << "Raytracing Algorithm Finished......" << std::endl;

std::cout << "Time required to run ray tracing algorithm in parallel ray tracing program....." << t_end - t_start << " seconds." << std::endl;





std::ofstream file_pointer;
file_pointer.open("./ParallelRaytracingWithOpenMP.ppm", std::ios::binary);  
file_pointer << "P6\n" << rows << " " << cols << "\n255\n"; 



for (size_t i = 0; i < cols * rows; ++i) {
Vec3f& c = image[i];
float max = std::max(c[0], std::max(c[1], c[2]));
if (max > 1) c = c * (1. / max);
for (size_t j = 0; j < 3; j++) {
file_pointer << (char)(255 * std::max(0.f, std::min(1.f, image[i][j])));
}
}

file_pointer.close(); 


return 0;

}