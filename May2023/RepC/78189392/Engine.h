#pragma once
#include <SDL2\SDL.h>
#include <GLEW\glew.h>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <iostream>
#include <functional>
#include "Shader.h"
#include "Transofrm.h"
#include "Camera.h"
#include "DataToDraw.h"
#define COLOR(r,g,b)(r | g << 8 | b << 16);
class DataBuffers
{
public:
GLuint VertexBufer;
GLuint ColorBuffer;
GLuint IndexBuffer;
GLuint vao;
};
class Engine
{
public:
Engine(int width, int height, int major_version, int minor_version, std::function<void(DataToDraw &, float)> callback);
~Engine();
void Start();
void SetupStaticData(DataToDraw data);
private:
SDL_Window* window;
SDL_GLContext context;
int width, height;
Shader shader;						
Camera camera;
DataBuffers staticBuffers;
DataBuffers pointsBuffers;
GLuint gWorldLocation;
DataToDraw staticData;
DataToDraw pointsData;
std::function<void(DataToDraw &, float)> update_callback;
bool init_sdl(int width, int height, int major_version, int minor_version);
void init_opengl();
void sdl_loop();
void init_static_buffers(DataBuffers & buffers);
void init_points_buffer(DataBuffers & buffers, int bufferSize);
void render(GLfloat* points, unsigned int points_count, unsigned int * indexes, unsigned int indexes_count);
void render(DataToDraw data);
void renderStatic(DataToDraw dataToDraw, DataBuffers & buffers);
void renderPoints(DataToDraw & dataToDraw, DataBuffers & buffers);
};
