#pragma once

#include <functional>
#include <memory>
#include <string>

class Glfw {

public:

Glfw();
~Glfw();

double current_time();
void poll_events();

};

struct Size {

int w;
int h;

Size(int w, int h);

int area() const;

};

struct Color {

float r;
float g;
float b;

Color(float r, float g, float b);

};

struct Point {

int x;
int y;

};

enum class Key {
escape = 256
};

enum class Action {
press = 1
};

struct KeyEvent {

Key key;
Action action;

};

class Scene {

public:

virtual ~Scene() = default;

virtual void loop() = 0;
virtual void on_key_event(const KeyEvent &event) = 0;

};

struct GLFWwindow;

class Window {

Glfw &glfw;
GLFWwindow *handle;
std::unique_ptr<Scene> scene;

friend void key_cb_wrapper(GLFWwindow *, int, int, int, int);

public:

Window(Glfw &glfw, const Size &size, const std::string &title);
~Window();

void swap_buffers();
void update_title(const std::string &title);
bool should_close();
void close();
void run_scene(std::unique_ptr<Scene> scene);

};

class Render {

public:

explicit Render(const Window &win);

void clear();
void configure_camera_ortho(const Size &size);
void update_color(const Color &color);
void in_points_mode(std::function<void()> fn);
void place(const Point &point);

};
