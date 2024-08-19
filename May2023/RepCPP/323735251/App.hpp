#pragma once
#include "Text.hpp"
#include "Color.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "StopWatch.hpp"
#include "FrameBuffer.hpp"
#include "RayTracer.hpp"
#include <iostream>


struct AppOptions {
bool        logInfo{ true };
std::string imageOutputFile{ "./scene.ppm" };
Vec2        imageOutputSize{ CommonResolutions::HD_1080p };
float       imageOutputGamma{ 2.20f };

float  rayTracingBias{ 0.02f };
size_t rayTracingReflectionLimit{ 3 };

Color skyBoxColor{ 0.125f, 0.125f, 0.125f };
Color shadowColor{ 0.500f, 0.500f, 0.500f };

float cameraNearZ{   0.50f };
float cameraFarZ { 1000.0f };
float cameraFieldOfView{ 120.0f };

Vec3 viewTarget{  0,  0,  0  };
Vec3 viewOffset{  0,  0, 10  };
};

std::ostream& operator<<(std::ostream& os, const AppOptions& appOptions);



class App {
public:
App(Scene&& scene, const AppOptions& options);
void run();

private:
AppOptions options_;
StopWatch stopWatch_;

Scene scene_;
Camera camera_;
RayTracer rayTracer_;
FrameBuffer frameBuffer_;

public:
friend std::ostream& operator<<(std::ostream& os, const App& app);
};

std::ostream& operator<<(std::ostream& os, const App& app);
