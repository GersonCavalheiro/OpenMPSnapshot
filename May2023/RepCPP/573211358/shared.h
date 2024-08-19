#pragma once

#ifdef GUI
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <cstdlib>

#include <Windows.h>
#endif

#if defined(__APPLE__)

#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif
#endif

#ifdef NDEBUG
#define DEBUG false
#else
#define DEBUG true
#endif

#include <chrono>
#include <cstdio>
#include <cmath>

#include "physics.h"

int size; 
float* data;
float* new_data;
bool* fire_area;


std::chrono::high_resolution_clock::time_point t1;
std::chrono::high_resolution_clock::time_point t2;
std::chrono::high_resolution_clock::time_point t3;
std::chrono::high_resolution_clock::time_point t4;
std::chrono::duration<double> time_span;

bool cont = true;
int count = 1;
double total_time = 0;
#ifdef GUI
GLubyte* pixels = new GLubyte[resolution * resolution * 3];
#endif

inline void initialize(float* data) {
int len = size * size;
for (int i = 0; i < len; i++) {
data[i] = wall_temp;
}
}


inline void generate_fire_area(bool *fire_area){
int len = size * size;
for (int i = 0; i < len; i++) {
fire_area[i] = false;
}

float fire1_r2 = fire_size * fire_size;
for (int i = 0; i < size; i++){
for (int j = 0; j < size; j++){
int a = i - size / 2;
int b = j - size / 2;
int r2 = 0.5 * a * a + 0.8 * b * b - 0.5 * a * b;
if (r2 < fire1_r2) fire_area[i * size + j] = 1;
}
}

float fire2_r2 = (fire_size / 2) * (fire_size / 2);
for (int i = 0; i < size; i++){
for (int j = 0; j < size; j++){
int a = i - 1 * size / 3;
int b = j - 1 * size / 3;
int r2 = a * a + b * b;
if (r2 < fire2_r2) fire_area[i * size + j] = 1;
}
}
}

inline void swap(float* &a, float* &b) {
float* tmp = a;
a = b;
b = tmp;
}


#ifdef GUI
inline void data2pixels(float *data, GLubyte* pixels){
float factor_data_pixel = (float) size / resolution;
float factor_temp_color = (float) 255 / fire_temp;
for (int x = 0; x < resolution; x++){
for (int y = 0; y < resolution; y++){
int idx = x * resolution + y;
int idx_pixel = idx * 3;
int x_raw = x * factor_data_pixel;
int y_raw = y * factor_data_pixel;
int idx_raw = y_raw * size + x_raw;
float temp = data[idx_raw];
int color = ((int) temp / 5 * 5) * factor_temp_color;
pixels[idx_pixel] = color;
pixels[idx_pixel + 1] = 255 - color;
pixels[idx_pixel + 2] = 255 - color;
}
}
}


inline void plot(GLubyte* pixels){
glClear(GL_COLOR_BUFFER_BIT);
glDrawPixels(resolution, resolution, GL_RGB, GL_UNSIGNED_BYTE, pixels);
glutSwapBuffers();
}
#endif


