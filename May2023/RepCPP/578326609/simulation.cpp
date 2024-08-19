

#include "simulation.h"

#include <utility>
#include <unistd.h>
#include "tester.h"
#include "graphics.h"
#include "omp.h"


double getTime() {
struct timespec s;
clock_gettime(CLOCK_MONOTONIC_RAW, &s);
double seconds = s.tv_sec * 1000.0 * 1000.0; 
double nanoseconds = s.tv_nsec / 1000.0; 
return (seconds + nanoseconds) / 1000.0;
}

string ftos(float in) {
char *c = new char[9];
sprintf(c, "%.2f", in);
string str = string(c);
delete[]c;
return str;
}

Simulation::Simulation() {
testManager = new TestManager(&bodies);
settings->maxThreads = omp_get_max_threads();
settings->realtime = false;
settings->numThreads = settings->maxThreads > 2 ? settings->maxThreads : 1;
graphics = new Graphics(1024 + 256 + 16, 1024 + 48);
graphics->centerFull(1024 - 32 - 256);
}

Simulation::Simulation(Settings *s) {
settings = s;
settings->maxThreads = omp_get_max_threads();
if (settings->numThreads == -1) {
settings->numThreads = omp_get_max_threads();
}
graphics = nullptr;
testManager = new TestManager(&bodies);
}

void Simulation::runHeadless() {

double simulationBegin = getTime();

while ((float) settings->maxTicks >= currentStep) {
currentStep += settings->stepSize;
simulate();
}

for (const auto &item: bodies) {
item->print();
free(item);
}

printf("Completed. (%.4f ms)\n", (getTime() - simulationBegin));

}

void Simulation::run() {

fps = 60;
double memoryCooldown = getTime();
while (running) {
tickBegin = getTime();

int counter = 0;

while ((getTime() - tickBegin) < 16.667) {
double start = getTime();
currentStep += settings->stepSize;

simulate();
counter++;
ups = (float) (1000.0 / (getTime() - start));
}

cycles = counter;


double renderBegin = getTime();

render();

handleEvents();

double renderDelta = getTime() - renderBegin;
fps = (float) (1000.0 / renderDelta);

if (getTime() - memoryCooldown > 5000) {
struct rusage r_usage;
getrusage(RUSAGE_SELF, &r_usage);
maxRSS = r_usage.ru_maxrss;
}

if (settings->realtime) {
recalculateCores();
} else {
settings->numThreads = settings->maxThreads;
}

}
}

void Simulation::recalculateCores() {

if (ups < settings->targetFps * 0.925) {
if (underflow == 0) {
underflow = getTime();
} else if (getTime() - underflow >= 1500) {
settings->numThreads =
settings->maxThreads > settings->numThreads ? settings->numThreads + 1 : settings->numThreads;
underflow = 0;
}
} else if (ups > settings->targetFps) {
if (overflow == 0 && settings->numThreads > 2) {
overflow = getTime();
} else if (getTime() - overflow >= 1500) {
settings->numThreads = 1 < settings->numThreads - 1 ? settings->numThreads - 1 : settings->numThreads;
overflow = 0;
}
}
}

void Simulation::simulate() {

#pragma omp parallel for default(none) shared(bodies, settings) num_threads(settings->numThreads) if(settings->useParallel)
for (auto to: bodies)
for (auto from: bodies)
if (to != from) {
if (settings->useCollisions) to->handleCollision(*from);
to->addForce(*from);
}


#pragma omp parallel for default(none) shared(bodies, currentStep) num_threads(settings->numThreads) if(settings->useParallel)
for (auto body: bodies) {
body->update(currentStep);
}


for (auto body: bodies) body->clearForce();

}


void Simulation::render() {


graphics->clear();
graphics->setRadius(settings->radius);

float maxMass = 0;
for (auto body: bodies) {
maxMass = body->getMass() > maxMass ? body->getMass() : maxMass;
}

graphics->beginSimulationFrame();


for (auto body: bodies) {
body->draw(graphics, maxMass);
}

graphics->setAlphaColor(128, 128, 128, 128);
graphics->drawOrigin(0, 0, 0, settings->radius, settings->radius, settings->radius);
graphics->drawCube(-settings->radius, -settings->radius,
-settings->radius, settings->radius * 2,
settings->radius * 2,
settings->radius * 2);

graphics->endSimulationFrame();




showStats();


graphics->render();

}

bool dragging = false;

void Simulation::handleEvents() {
SDL_Event e;
while (SDL_PollEvent(&e) != 0) {
switch (e.type) {
case SDL_QUIT:
running = false;
break;
case SDL_MOUSEWHEEL:
settings->scale = graphics->getCamera().scale;
settings->scale += e.wheel.y * (float) 0.01;
if (settings->scale <= 0.0001) settings->scale = 0.001;
graphics->scaleCamera(settings->scale);
break;
case SDL_MOUSEBUTTONDOWN:
dragging = true;
break;
case SDL_MOUSEBUTTONUP:
dragging = false;
break;
case SDL_MOUSEMOTION:
if (dragging) {
settings->rotY -= e.motion.xrel % 360;
settings->rotP += e.motion.yrel % 360;
}
break;
case SDL_KEYDOWN:
switch (e.key.keysym.sym) {
case SDLK_UP:
testManager->lastTest();
break;
case SDLK_DOWN:
testManager->nextTest();
break;
case SDLK_RETURN:
testManager->selectTest();
currentStep = 0;
break;
case 'i':
settings->showControls = !settings->showControls;
break;
case ']':
settings->stepSize += 0.25;
break;
case '[':
settings->stepSize -= 0.25;
break;
case 'c':
settings->useCollisions = !settings->useCollisions;
break;
case 'p':
settings->useParallel = !settings->useParallel;
break;
case 'r':
settings->realtime = !settings->realtime;
break;
case 'a':
settings->rotY += 22.5;
break;
case 'd':
settings->rotY -= 22.5;
break;
case 'w':
settings->rotP -= 22.5;
break;
case 's':
settings->rotP += 22.5;
break;
case 'z':
settings->rotY = 0;
settings->rotP = 0;
settings->scale = 0.75;
graphics->centerFull(1024 - 32 - 256);
break;
default:
break;
}

}
graphics->rotate(settings->rotY, settings->rotP, 0);
break;
}
}

void Simulation::showStats() {
auto cam = graphics->getCamera();
float xPad = 0, yPad = 8;
float xOff = 1024 + xPad, yOff = 32;
float lnOff = yOff;
graphics->setAlphaColor(32, 32, 32, 128);
graphics->fillRect(xOff - xPad * 2, yPad * 2, 256 - xPad, 1024 + 32 + 16 -
yPad *
4);
graphics->setColor(200, 200, 200);
xOff += 16;
graphics->drawString("N-Bodies II", 3, xOff, lnOff);
graphics->drawString("Statistics", 2, xOff + 10, lnOff += 40);
graphics->setColor(128, 128, 128);

graphics->drawString("Step Size: " + ftos(settings->stepSize),
1.5, xOff +
20,
lnOff += 25);
graphics->drawString("Relative Step: " + ftos(currentStep), 1.5, xOff +
20,
lnOff += 20);
graphics->drawString(
"Total Steps: " + ftos(currentStep / settings->stepSize), 1.5,
xOff + 20,
lnOff += 20);

graphics->drawString("Remaining Bodies: " + ftos(bodies.size()), 1.5,
xOff + 20,
lnOff += 20);

graphics->setColor(200, 200, 200);
graphics->drawString("Rendering", 2, xOff + 10, lnOff += 25);
graphics->setColor(128, 128, 128);

graphics->drawString("Update Time: " + ftos((float) 1000 / ups) + "ms",
1.5, xOff + 20, lnOff += 25);

graphics->drawString("Frame Time: " + ftos((float) 1000 / fps) + "ms",
1.5, xOff + 20, lnOff += 20);

graphics->drawString("FPS: " + ftos((fps > 0) ? fps : 60.0) + "fps",
1.5, xOff + 20, lnOff += 20);

graphics->drawString("UPS: " + ftos((ups > 0) ? ups : 60.0) + "ups",
1.5, xOff + 20, lnOff += 20);

graphics->drawString("UP/FR: " + ftos(cycles) + "ups",
1.5,
xOff +
20,
lnOff += 20);


graphics->setColor(200, 200, 200);
graphics->drawString("Runtime", 2, xOff + 10, lnOff += 30);
graphics->setColor(128, 128, 128);

graphics->drawString(
"Parallel: " + string((settings->useParallel) ? "Parallel" : "Sequential"),
1.5, xOff + 20, lnOff += 20);

graphics->drawString("Cores: " + string(std::to_string(settings->numThreads)) + "/" +
string(std::to_string(settings->maxThreads)),
1.5, xOff + 20, lnOff += 20);

lnOff += 16;
double a = 200.0 / (settings->maxThreads);
double b = 180.0 / (settings->maxThreads);

for (int i = 0; i < settings->maxThreads; ++i) {
if (settings->numThreads > i) {
graphics->setColor(128, 128, 128);
graphics->fillRect(xOff + 20 + (float) i * a, lnOff, b, 6);
} else {
graphics->setColor(96, 96, 96);
graphics->strokeRect(xOff + 20 + (float) i * a, lnOff, b, 6);
}
}

graphics->setColor(128, 128, 128);
graphics->drawString("MaxRSS: " + ftos(maxRSS / (1024 *
1024)) +
" Mb", 1.5, xOff + 20,
lnOff += 20);



graphics->setColor(200, 200, 200);
graphics->drawString("Parameters", 2, xOff + 10, lnOff += 25);
graphics->setColor(128, 128, 128);
graphics->drawString((settings->useCollisions ? "Collisions: On" : "Collisions: Off"), 1.5, xOff + 20,
lnOff += 25);
graphics->drawString("Outer Container: " + ftos(cam.radius), 1.5, xOff + 20,
lnOff += 25);
graphics->drawString(
settings->realtime ? "Realtime: On" : "Realtime: Off", 1.5,
xOff + 20,
lnOff += 20);
graphics->setColor(200, 200, 200);
graphics->drawString("Resources", 2, xOff + 10, lnOff += 40);
graphics->setColor(128, 128, 128);
graphics->setColor(200, 200, 200);
graphics->drawString("Camera", 2, xOff + 10, lnOff += 30);
graphics->setColor(128, 128, 128);
graphics->drawString("Scale: " + ftos(cam.scale), 1.5, xOff + 20, lnOff
+= 25);
graphics->drawMeter(xOff + 20, lnOff += 16, 200, 6,
fmap(abs(cam.scale), 0, abs(cam.scale) * 2, 0,
1));
graphics->setAlphaColor(128, 128, 128, 128);
graphics->drawString("Pitch: " + ftos(cam.pitch) + " deg",
1.5,
xOff + 20,
lnOff += 20);
graphics->drawMeter(xOff + 20, lnOff += 16, 200, 6,
fmap(abs((int) cam.pitch) % 360, 0,
360, 0,
1));
graphics->setAlphaColor(128, 128, 128, 128);
graphics->drawString("Yaw: " + ftos(cam.yaw) + " deg", 1.5,
xOff + 20,
lnOff += 20);
graphics->drawMeter(xOff + 20, lnOff += 16, 200, 6,
fmap(abs((int) cam.yaw) % 360, 0,
360, 0,
1));
graphics->setColor(200, 200, 200);
graphics->drawString("Tests", 2, xOff + 10, lnOff += 30);
float ptOff = 0;
lnOff += 5;
if (testManager->getTests().size() >= 1) {
for (auto test: testManager->getTests()) {
if (test == testManager->getTest()) {
graphics->setAlphaColor(255, 200, 64, 255);
graphics->drawString(test->name, 1.5, xOff + 20, lnOff += 20);
} else {
graphics->setColor(128, 128, 128);
graphics->drawString(test->name, 1.5, xOff + 20, lnOff += 20);
}
if (test == testManager->getSelected()) {
graphics->setAlphaColor(200, 96, 64, 255);
graphics->drawCircle(xOff + 12, lnOff + 6, 3);

}
ptOff++;
}
} else {
graphics->setAlphaColor(255, 64, 64, 255);
graphics->drawString("'tests'  directory  not  found.", 1.5, xOff + 20,
lnOff +=
20);
}
graphics->setAlphaColor(32, 32, 32, 128);
graphics->fillRect(16, 1024, 1024 - 32, 32);
graphics->setAlphaColor(128, 128, 128, 128);

float hOff = 24;
float vCenter = 1034;

hOff += graphics->drawStringGetLength("Toggle Collisions: C", 1.5,
hOff, vCenter);
hOff += graphics->drawStringGetLength("Realtime: R", 1.5,
hOff, vCenter);
hOff += graphics->drawStringGetLength("Pitch: W / S", 1.5,
hOff, vCenter);
hOff += graphics->drawStringGetLength("Yaw: A / D", 1.5,
hOff, vCenter);
hOff += graphics->drawStringGetLength("Reset View: Z", 1.5,
hOff, vCenter);
hOff += graphics->drawStringGetLength("Zoom: Scroll", 1.5,
hOff, vCenter);
hOff += graphics->drawStringGetLength("Cycle Tests: Arrow Up/Down", 1.5,
hOff, vCenter);
hOff += graphics->drawStringGetLength("Select Test: Enter", 1.5,
hOff, vCenter);


graphics->setColor(128, 128, 128);
graphics->drawString("Made by Braden Nicholson - 2021/2022", 1.5, xOff,
1024 - 8);
graphics->drawString("https:
1024 + 8);

}


