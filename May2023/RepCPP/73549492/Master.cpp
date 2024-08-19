#include "Master.hpp"

#include <chrono>
#include <iostream>
#include <mpi.h>

#include "Window.hpp"
#include "Input.hpp"
#include "Shaders.hpp"
#include "Texture.hpp"
#include "FullscreenQuad.hpp"
#include "Raytracer.hpp"
#include "MessageType.hpp"


std::shared_ptr<Master> Master::Create() {
auto master = std::make_shared<Master>();
SetSingleton(master);
return master;
}

void Master::Run(unsigned int machineCount) {
auto window = Window::Create(500, 400, "Raytracer - OpenMP / MPI");
Shaders::CreateAndLoad();
Texture::Create();
auto quad = FullscreenQuad::Create();
auto raytracer = Raytracer::Create();

auto start = std::chrono::high_resolution_clock::now();
auto last = std::chrono::high_resolution_clock::now();

int width, height;
Window::GetSingleton()->GetSize(width, height);

int maxI = width * height / machineCount;
BroadcastSlaveParts(machineCount, width * height, width, height);

std::cout << "Time it takes to render a frame:\n";
while (!window->IsClosed()) {
window->PollEvents();
if (Input::IsKeyPressed(Input::Key::Escape)) {
window->Close();
break;
}

auto timestamp = std::chrono::high_resolution_clock::now();
auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp - last).count();
last = timestamp;
std::cout << delta << "ms\n";

float time = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp - start).count() / 1000.f;
BroadcastTime(time, machineCount);

auto &data = Texture::GetSingleton()->GetData();

#pragma omp parallel for default(shared)
for (auto i = 0; i < maxI; ++i) {
glm::vec2 uv((i % width) / (float) width, (i / width) / (float) height);
glm::vec2 uvNorm((uv.x * 2 - 1) * width / height, uv.y * 2 - 1);

glm::vec3 color = raytracer->TracePixel(uvNorm, time);

data[i * 3 + 0] = color.r;
data[i * 3 + 1] = color.g;
data[i * 3 + 2] = color.b;
}

int machinesDone = 0;
while (machinesDone < machineCount - 1) {
int incomingFlag;
MPI_Iprobe(MPI_ANY_SOURCE, (int) MessageType::Color, MPI_COMM_WORLD, &incomingFlag, MPI_STATUS_IGNORE);
if (incomingFlag) {
float incomingData[4];
MPI_Recv(incomingData, 4, MPI_FLOAT, MPI_ANY_SOURCE, (int) MessageType::Color, MPI_COMM_WORLD,
MPI_STATUS_IGNORE);

int i = (int) incomingData[3];
data[i * 3 + 0] = incomingData[0];
data[i * 3 + 1] = incomingData[1];
data[i * 3 + 2] = incomingData[2];
}

else {
int doneFlag;
MPI_Status doneStatus;
MPI_Iprobe(MPI_ANY_SOURCE, (int) MessageType::Done, MPI_COMM_WORLD, &doneFlag, &doneStatus);
if (doneFlag) {
int doneData;
MPI_Recv(&doneData, 1, MPI_INT, doneStatus.MPI_SOURCE, (int) MessageType::Done, MPI_COMM_WORLD,
MPI_STATUS_IGNORE);
++machinesDone;
}
}
}

Texture::GetSingleton()->UploadData();

window->Clear();
quad->Render();
window->SwapBuffers();
}

BroadcastExit(machineCount);
std::cout << "\nMASTER EXITING\n";
}

void Master::BroadcastSlaveParts(unsigned int machineCount, int size, int width, int height) {
std::cout << "Sending the slave data parts and texture size\n";
for (auto i = 1; i < machineCount; ++i) {
int slaveData[] = {size / machineCount * i, size / machineCount * (i + 1), width, height};
MPI_Send(slaveData, 4, MPI_INT, i, (int) MessageType::Init, MPI_COMM_WORLD);
}
}

void Master::BroadcastExit(unsigned int machineCount) {
for (auto i = 1; i < machineCount; ++i) {
int data = 1;
MPI_Send(&data, 1, MPI_INT, i, (int) MessageType::Exit, MPI_COMM_WORLD);
}
}

void Master::BroadcastTime(float time, unsigned int machineCount) {
for (auto i = 1; i < machineCount; ++i) {
MPI_Send(&time, 1, MPI_FLOAT, i, (int) MessageType::Time, MPI_COMM_WORLD);
}
}
