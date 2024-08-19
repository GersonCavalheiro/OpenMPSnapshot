#pragma once
#include "glad.h"
#include "GLFW/glfw3.h"
#include "bird.h"
#include "Dimension.h"

#define WINDOW_SIZE_X 1280
#define WINDOW_SIZE_Y 720
#define WINDOW_NAME "Flocking Simulation"
#define BACKGROUND_COLOR_R 0.8
#define BACKGROUND_COLOR_G 0.8
#define BACKGROUND_COLOR_B 0.8

class Graphics_manager {
GLFWwindow* window;

GLuint vbo; 
GLuint vao; 
GLuint vs;  
GLuint fs;  
GLuint sp;  

public:
Graphics_manager();
~Graphics_manager();

void draw_birds(Bird**, int);
void draw_bird(Bird*);
bool loop();
void swap_buffers();
Dimension getDimensions();
};

