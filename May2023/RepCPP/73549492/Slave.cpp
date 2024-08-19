#include "Slave.hpp"

#include <iostream>
#include <mpi.h>

#include "MessageType.hpp"
#include "Raytracer.hpp"


std::shared_ptr<Slave> Slave::Create() {
auto master = std::make_shared<Slave>();
SetSingleton(master);
return master;
}

void Slave::Run() {
std::cout << "Waiting for the data\n";
int data[4];
MPI_Recv(data, 4, MPI_INT, 0, (int) MessageType::Init, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
std::cout << "This slave will compute the data part " << data[0] << ":" << data[1] << std::endl;
std::cout << "The texture has size " << data[2] << "x" << data[3] << std::endl;

auto raytracer = Raytracer::Create();

while (true) {
int timeFlag;
MPI_Iprobe(0, (int) MessageType::Time, MPI_COMM_WORLD, &timeFlag, MPI_STATUS_IGNORE);

if (timeFlag) {
float time;
MPI_Recv(&time, 1, MPI_FLOAT, 0, (int) MessageType::Time, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
std::cout << "Slave received time " << time << std::endl;

#pragma omp parallel for default(shared)
for (auto i = data[0]; i < data[1]; ++i) {
glm::vec2 uv((i % data[2]) / (float) data[2], (i / data[2]) / (float) data[3]);
glm::vec2 uvNorm((uv.x * 2 - 1) * data[2] / data[3], uv.y * 2 - 1);

glm::vec3 color = raytracer->TracePixel(uvNorm, time);
float colorData[] = {color.r, color.g, color.b, (float) i};

#pragma omp critical
MPI_Send(colorData, 4, MPI_FLOAT, 0, (int) MessageType::Color, MPI_COMM_WORLD);
}
int doneData = 1;
MPI_Send(&doneData, 1, MPI_INT, 0, (int) MessageType::Done, MPI_COMM_WORLD);
}

if (ShouldExit()) {
break;
}
}
std::cout << "\nSLAVE EXITING\n";
}

bool Slave::ShouldExit() const {
int flag;
MPI_Status exitStatus;
MPI_Iprobe(0, (int) MessageType::Exit, MPI_COMM_WORLD, &flag, &exitStatus);
return flag > 0;
}
