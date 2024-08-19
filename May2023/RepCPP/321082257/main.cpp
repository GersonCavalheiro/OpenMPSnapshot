#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <SDL2/SDL.h>

#include "omp.h"

#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"

#include "cube.hpp"
#include "disk.hpp"
#include "plane.hpp"
#include "sphere.hpp"

#include "scene.hpp"

#include "script.hpp"

#include "ltimer.h"

#include "sdl2.h"

using namespace std::string_literals;

void TestV2()
{
std::cout << std::boolalpha;

Vec2 v2(1.0, 2.0);

std::cout << "--- V2 ---"s << std::endl;
std::cout << "v2 " << v2 << std::endl;

Vec2 a(1.0, 1.1);
Vec2 b(1.3, 1.4);

std::cout << a << " + " << b << " = " << a + b << std::endl; 
}

void TestV3()
{
std::cout << std::boolalpha;

Vec3 v3(1.0, 2.0, 3.0);

std::cout << "--- V3 ---"s << std::endl;
std::cout << "v3 " << v3 << std::endl;

Vec3 a(1.0, 1.1, 1.2);
Vec3 b(1.3, 1.4, 1.5);

std::cout << a << " + " << b << " = " << a + b << std::endl; 
std::cout << a << " - " << b << " = " << a - b << std::endl; 

std::cout << a << " * 3.0 = " << a * 3.0 << std::endl; 
std::cout << a << " / 3.0 = " << a / 3.0 << std::endl; 

std::cout << a << " dot " << b << " = " << a.dot(b) << std::endl; 
std::cout << a << " cross " << b << " = " << a.cross(b) << std::endl; 

std::cout << "normalized version of " << a << " = " << a.normalize() << std::endl; 

std::cout << "len of " << a << " = " << a.len() << std::endl; 
std::cout << "len of " << b << " = " << b.len() << std::endl; 

std::cout << a << " < " << b << "? " << (a < b) << std::endl;
std::cout << a << " > " << b << "? " << (a > b) << std::endl;
}

void TestV4()
{
std::cout << std::boolalpha;

Vec4 v4(1.0, 2.0, 3.0, 4.0);

std::cout << "--- V4 ---"s << std::endl;
std::cout << "v4 " << v4 << std::endl;

Vec4 a(1.0, 1.1, 1.2, 1.3);
Vec4 b(1.3, 1.4, 1.5, 1.6);

std::cout << a << " + " << b << " = " << a + b << std::endl; 
std::cout << a << " - " << b << " = " << a - b << std::endl; 

std::cout << a << " * 3.0 = " << a * 3.0 << std::endl; 
std::cout << a << " / 3.0 = " << a / 3.0 << std::endl; 

std::cout << a << " dot " << b << " = " << a.dot(b) << std::endl; 


std::cout << "normalized version of " << a << " = " << a.normalize() << std::endl; 

std::cout << "len of " << a << " = " << a.len() << std::endl; 
std::cout << "len of " << b << " = " << b.len() << std::endl; 

std::cout << a << " < " << b << "? " << (a < b) << std::endl;
std::cout << a << " > " << b << "? " << (a > b) << std::endl;
}

void TestSphere()
{
std::cout << std::boolalpha;

std::cout << "--- Sphere ---"s << std::endl;

Sphere s1 { 1.0, 2.0, 3.0, 4.0 };
Vec3 pos { 0.1, 0.2, 0.3 };
Sphere s2 { pos, 5.0 };

std::cout << "Sphere 1 "s << s1 << std::endl;
std::cout << "Sphere 2 "s << s2 << std::endl;
}

void TestCube()
{
std::cout << std::boolalpha;

std::cout << "--- Cube ---"s << std::endl;

Cube c1 { 1.0, 2.0, 3.0, 1.0, 1.0, 1.0 };
Vec3 pos { 0.1, 0.2, 0.3 };

std::cout << "Cube 1: "s << c1 << std::endl;

Cube c2 { 0.5, 0.5, 0.5, 1.0 };
std::cout << "Cube 2: "s << c2 << std::endl;
std::cout << "Cube 2 point methods:"s << std::endl;
std::cout << "\t" << c2.p0() << std::endl;
std::cout << "\t" << c2.p1() << std::endl;
std::cout << "\t" << c2.p2() << std::endl;
std::cout << "\t" << c2.p3() << std::endl;
std::cout << "\t" << c2.p4() << std::endl;
std::cout << "\t" << c2.p5() << std::endl;
std::cout << "\t" << c2.p6() << std::endl;
std::cout << "\t" << c2.p7() << std::endl;
std::cout << "Cube 2 points: "s << c2.points() << std::endl;

Vec3 p { 0, 0, 0 };
std::cout << "Cube 2 normal, using point (0,0,0): " << c2.normal(p) << std::endl;
}

void TestDisk()
{
std::cout << std::boolalpha;

std::cout << "--- Disk ---"s << std::endl;

Disk c1 { 1.0, 2.0, 3.0, 5.0 };

std::cout << "Disk 1: "s << c1 << std::endl;
}

void TestPlane()
{
std::cout << std::boolalpha;

std::cout << "--- Plane ---"s << std::endl;

Plane p1 { Point3 { 1.0, 2.0, 3.0 }, Vec3 { 0, 5.0, 0 } };

std::cout << "Plane 1: "s << p1 << std::endl;
}

void TestRay()
{
std::cout << std::boolalpha;

std::cout << "--- Ray ---"s << std::endl;

Vec3 pos0 { 0.1, 0.2, 0.3 };
Vec3 pos1 { 0.1, 0.2, 0.3 };

Ray ray { pos0, pos1 };

std::cout << "Ray: " << ray << std::endl;

Sphere s1 { 1.0, 2.0, 3.0, 4.0 };

std::cout << "Sphere: " << s1 << std::endl;

if (auto maybeIntersectionPoint = ray.intersect(s1)) {
const auto intersectionPointAndNormal = maybeIntersectionPoint.value();
const auto intersectionPoint = intersectionPointAndNormal.first;
const auto intersectionPointNormal = intersectionPointAndNormal.second;
std::cout << "Ray intersects sphere at " << intersectionPoint << std::endl;
std::cout << "Ray has normal at intersection point " << intersectionPointNormal
<< std::endl;
} else {
std::cout << "Ray does not intersect the sphere." << std::endl;
}
}

auto TestSDL2RayTrace(const bool verbose) -> int
{

using std::cerr;
using std::endl;

auto sys = sdl2::make_sdlsystem(SDL_INIT_EVERYTHING);
if (!sys) {
cerr << "Error creating SDL2 system: " << SDL_GetError() << endl;
return 1;
}

bool fullscreen = false;

const int W = 495;
const int H = 270;

int winw;
int winh;
int winf;

if (fullscreen) {
winw = 0;
winh = 0;
winf = SDL_WINDOW_FULLSCREEN_DESKTOP;
} else {
winw = 1980;
winh = 1080;
winf = SDL_WINDOW_SHOWN;
}

auto win = sdl2::make_window(
"Sphere Mover", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, winw, winh, winf);
if (!win) {
cerr << "Error creating window: " << SDL_GetError() << endl;
return 1;
}

SDL_ShowCursor(!fullscreen);

auto ren = sdl2::make_renderer(win.get(), -1, SDL_RENDERER_ACCELERATED);
if (!ren) {
cerr << "Error creating renderer: " << SDL_GetError() << endl;
return 1;
}

auto tex = sdl2::make_buffer_texture(ren.get(), W, H);
if (!tex) {
cerr << "Error creating texture: " << SDL_GetError() << endl;
return 1;
}

uint32_t* textureBuffer = new uint32_t[W * H];

const Sphere light { Vec3 { 0, 0, 50 }, 1 };
const Plane plane { Vec3 { 0, 0, 100 }, (Vec3 { 0, 0, 0.5 }).normalize() };

const Sphere sphere1 { Vec3 { W * .4, H * .5, 50 }, 50 };
const Sphere sphere2 { Vec3 { W * .5, H * .5, 50 }, 50 };
const Sphere sphere3 { Vec3 { W * .6, H * .5, 50 }, 50 };
const Cube cube1 { Vec3 { W * .7, H * .5, 50 }, 50 };

std::vector<Sphere> spheres = { sphere1, sphere2, sphere3 };

Scene scene1 { light, plane, spheres, cube1, Color::darkgray };
std::unique_ptr<Scene> scene_ptr = std::make_unique<Scene>(scene1);

const Point3 fromPoint { 0, 0, -W * 2 };

size_t currentSphere = 0;

SDL_Event event;

SDL_Joystick* joystick = nullptr;
auto found_joysticks = SDL_NumJoysticks();
if (found_joysticks > 0) {
SDL_Joystick* temp_joystick = nullptr;
for (auto i = 0; i < found_joysticks; i++) {
temp_joystick = SDL_JoystickOpen(i);
if (temp_joystick == nullptr) {
std::cerr << "JOYSTICK: " << i << ": " << SDL_GetError() << std::endl;
continue;
}
std::string joystick_name = SDL_JoystickName(temp_joystick);
SDL_JoystickClose(temp_joystick);
temp_joystick = nullptr;
if (verbose) {
std::cout << "JOYSTICK: " << joystick_name << std::endl;
}
}
joystick = SDL_JoystickOpen(0);
}

std::cout << "arrow keys and tab to move spheres around" << std::endl;
std::cout << "esc or q to quit" << std::endl;
std::cout << "f to toggle fullscreen" << std::endl;

bool quit = false;

const int SCREEN_FPS = 60;
const int SCREEN_TICKS_PER_FRAME = 1000 / SCREEN_FPS;
LTimer fpsTimer;
LTimer capTimer;
int countedFrames = 0;
fpsTimer.start();

double joy_left_offset_x = 0;
double joy_left_offset_y = 0;

double joy_right_offset_x = 0;
double joy_right_offset_y = 0;

const int JOYSTICK_DEAD_ZONE = 8000;

while (!quit) {

capTimer.start();

while (SDL_PollEvent(&event)) {
switch (event.type) {
case SDL_QUIT:
quit = true;
break;
case SDL_JOYAXISMOTION:
if (event.jaxis.which == 0) { 
if (event.jaxis.axis == 0) { 
joy_left_offset_x = 0;
if (event.jaxis.value < -JOYSTICK_DEAD_ZONE) {
joy_left_offset_x = event.jaxis.value / 32768.0;
} else if (event.jaxis.value > JOYSTICK_DEAD_ZONE) {
joy_left_offset_x = event.jaxis.value / 32767.0;
}
} else if (event.jaxis.axis == 1) { 
joy_left_offset_y = 0;
if (event.jaxis.value < -JOYSTICK_DEAD_ZONE) {
joy_left_offset_y = event.jaxis.value / 32768.0;
} else if (event.jaxis.value > JOYSTICK_DEAD_ZONE) {
joy_left_offset_y = event.jaxis.value / 32767.0;
}
} else if (event.jaxis.axis == 3) { 
joy_right_offset_x = 0;
if (event.jaxis.value < -JOYSTICK_DEAD_ZONE) {
joy_right_offset_x = -1;
} else if (event.jaxis.value > JOYSTICK_DEAD_ZONE) {
joy_right_offset_x = 1;
}
} else if (event.jaxis.axis == 4) { 
joy_right_offset_y = 0;
if (event.jaxis.value < -JOYSTICK_DEAD_ZONE) {
joy_right_offset_y = -1;
} else if (event.jaxis.value > JOYSTICK_DEAD_ZONE) {
joy_right_offset_y = 1;
}
}
}
break;
case SDL_JOYBUTTONUP: 
currentSphere++;
if (currentSphere >= spheres.size()) {
currentSphere = 0;
}
break;
case SDL_KEYDOWN:
switch (event.key.keysym.sym) {
case SDLK_SPACE:
case SDLK_TAB: {
currentSphere++;
if (currentSphere >= spheres.size()) {
currentSphere = 0;
}
break;
}
case SDLK_d:
case SDLK_RIGHT: {
const auto newScene = scene_ptr->sphere_move(currentSphere, Vec3 { 1, 0, 0 });
scene_ptr = std::make_unique<Scene>(newScene);
break;
}
case SDLK_a:
case SDLK_LEFT: {
const auto newScene = scene_ptr->sphere_move(currentSphere, Vec3 { -1, 0, 0 });
scene_ptr = std::make_unique<Scene>(newScene);
break;
}
case SDLK_w:
case SDLK_UP: {
const auto newScene = scene_ptr->sphere_move(currentSphere, Vec3 { 0, -1, 0 });
scene_ptr = std::make_unique<Scene>(newScene);
break;
}
case SDLK_s:
case SDLK_DOWN: {
const auto newScene = scene_ptr->sphere_move(currentSphere, Vec3 { 0, 1, 0 });
scene_ptr = std::make_unique<Scene>(newScene);
break;
}
case SDLK_f:
case SDLK_F11: {
auto window_flags = SDL_GetWindowFlags(win.get());
fullscreen = (window_flags & SDL_WINDOW_FULLSCREEN_DESKTOP)
|| (window_flags & SDL_WINDOW_FULLSCREEN);
SDL_SetWindowFullscreen(
win.get(), (fullscreen ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP));
SDL_ShowCursor(!fullscreen);
break;
}
case SDLK_q:
case SDLK_ESCAPE:
quit = true;
break;
}
}
}

if (joy_left_offset_x != 0 || joy_left_offset_y != 0) {
const auto newScene = scene_ptr->sphere_move(
currentSphere, Vec3 { joy_left_offset_x, joy_left_offset_y, 0 });
scene_ptr = std::make_unique<Scene>(newScene);
}

if (joy_right_offset_x != 0 || joy_right_offset_y != 0) {
currentSphere++;
if (currentSphere >= spheres.size()) {
currentSphere = 0;
}
const auto newScene = scene_ptr->sphere_move(
currentSphere, Vec3 { joy_right_offset_x, joy_right_offset_y, 0 });
scene_ptr = std::make_unique<Scene>(newScene);

if (currentSphere > 0) {
currentSphere--;
} else {
currentSphere = spheres.size() - 1;
}
}

double avgFPS = countedFrames / (fpsTimer.getTicks() / 1000.0);
if (avgFPS > 2000000) {
avgFPS = 0;
}

#pragma omp parallel for
for (int y = 0; y < H; ++y) {
for (int x = 0; x < W; ++x) {
const RGB c = scene_ptr->color(fromPoint, x, y).clamp255();
textureBuffer[(y * W) + x] = 0xFF000000 | (static_cast<uint8_t>(c.R()) << 16)
| (static_cast<uint8_t>(c.B()) << 8) | static_cast<uint8_t>(c.G());
}
}

SDL_UpdateTexture(tex.get(), nullptr, textureBuffer, W * sizeof(uint32_t));

SDL_RenderClear(ren.get());
SDL_RenderCopy(ren.get(), tex.get(), nullptr, nullptr);
SDL_RenderPresent(ren.get());

++countedFrames;

int frameTicks = capTimer.getTicks();
if (frameTicks < SCREEN_TICKS_PER_FRAME) {
SDL_Delay(SCREEN_TICKS_PER_FRAME - frameTicks);
}
}

if (joystick != nullptr) {
SDL_JoystickClose(joystick);
joystick = nullptr;
}

SDL_Quit();
return 0;
}

void TestRayTrace(const std::string filename)
{
const int W = 320;
const int H = 200;

const Sphere light { Vec3 { 0, 0, 50 }, 1 };
const Plane plane { Vec3 { 0, 0, 100 }, (Vec3 { 0, 0, 0.5 }).normalize() };

const Sphere sphere1 { Vec3 { W * .4, H * .5, 50 }, 50 };
const Sphere sphere2 { Vec3 { W * .5, H * .5, 50 }, 50 };
const Sphere sphere3 { Vec3 { W * .6, H * .5, 50 }, 50 };

const Cube cube1 { Vec3 { W * .7, H * .5, 50 }, 50 };

std::vector<Sphere> spheres = { sphere1, sphere2, sphere3 };

std::ofstream out(filename);
out << "P3\n"s << W << " "s << H << " "s
<< "255\n"s;

Scene scene { light, plane, spheres, cube1, Color::darkgray };

const Point3 fromPoint { 0, 0, -W * 2 };

for (int y = 0; y < H; ++y) {
for (int x = 0; x < W; ++x) {
out << scene.color(fromPoint, x, y).clamp255().ppm() << "\n";
}
}
}

auto mustRead(const std::string filename) -> const std::string
{
std::stringstream buf;
std::ifstream f { filename };
buf << f.rdbuf();
return buf.str();
}

auto TestScript(const std::string filename) -> int
{
std::cout << "--- SCRIPT: " << filename << " ---" << std::endl;
const std::string source = mustRead(filename);
std::cout << "read " << source.length() << " characters" << std::endl;
return interpret(tokenize(source));
}

auto main(int argc, char** argv) -> int
{
if (argc > 1) { 

TestV2();
TestV3();
TestV4();

TestSphere();
TestCube();
TestDisk();
TestPlane();

TestRay();
TestRayTrace("/tmp/out.ppm"s);

TestScript(SCRIPTDIR "hello.pip"s);
TestScript(SCRIPTDIR "hello2.pip"s);

} else { 

TestSDL2RayTrace(true);
}

return EXIT_SUCCESS;
}
