


#include <omp.h>  
#include "platform.hpp"
#include "include/Voxelyze.h"
#include "MeshRender.h"
#include "Creature.h"
#include <windows.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <unordered_map>

#include "tdogl/Program.h"
#include "tdogl/Camera.h"


#define WITH_DISPLAY 

int NB_CREATURE = 100;
int NB_SAVED = 6;
int NB_GENERATION = 1;
double SIMULATION_DURATION = 15.;

int creatureID = 73; 
int WIDTH = 800;
int HEIGHT = 450;

char numstr[4]; 
char name[20];
int iGeneration = 0;

std::vector<Creature*> animCreature;

std::vector<std::string> get_all_files_names_within_folder(std::string folder)
{
std::vector<std::string> names;
std::string search_path = folder + "\\*.*";
WIN32_FIND_DATA fd;
HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
if (hFind != INVALID_HANDLE_VALUE) {
do {
if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
names.push_back(fd.cFileName);
}
} while (::FindNextFile(hFind, &fd));
::FindClose(hFind);
}
return names;
}


const glm::vec2 SCREEN_SIZE(WIDTH, HEIGHT);

GLFWwindow* gWindow = NULL;
double gScrollY = 0.0;
tdogl::Program* gProgram = NULL;
tdogl::Program* gProgram2D = NULL;
tdogl::Camera gCamera;

GLfloat gDegreesRotated = 0.0f;
Creature* pCreatureDisplay = NULL;

std::unordered_map<std::string, Creature*> mapCreature;


static void LoadShaders() {
std::vector<tdogl::Shader> shaders;
shaders.push_back(tdogl::Shader::shaderFromFile(ResourcePath("vertex-shader.txt"), GL_VERTEX_SHADER));
shaders.push_back(tdogl::Shader::shaderFromFile(ResourcePath("fragment-shader.txt"), GL_FRAGMENT_SHADER));
gProgram = new tdogl::Program(shaders);
}

static void LoadShaders2D() {
std::vector<tdogl::Shader> shaders;
shaders.push_back(tdogl::Shader::shaderFromFile(ResourcePath("vertex-shader-2d.txt"), GL_VERTEX_SHADER));
shaders.push_back(tdogl::Shader::shaderFromFile(ResourcePath("fragment-shader-2d.txt"), GL_FRAGMENT_SHADER));
gProgram2D = new tdogl::Program(shaders);
}



GLuint gVAO = 0;
GLuint gVBO = 0;

void addPoint(std::vector<float>* array, float x, float y, float u, float v) {
array->push_back(x);
array->push_back(y);
array->push_back(0);
array->push_back(u);
array->push_back(v);
}


static void LoadTriangle() {
glGenVertexArrays(1, &gVAO);
glGenBuffers(1, &gVBO);

glBindVertexArray(gVAO);
glBindBuffer(GL_ARRAY_BUFFER, gVBO);

std::vector<float> data;
addPoint(&data, -1, -1,  0, 0);
addPoint(&data, -1, 1, 0, 1); 
addPoint(&data, 1, 1,  1, 1);

addPoint(&data, -1, -1, 0, 0);
addPoint(&data, 1, 1,  1, 1);
addPoint(&data, 1,  -1,  1, 0);

glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);

glEnableVertexAttribArray(gProgram2D->attrib("vert"));
glVertexAttribPointer(gProgram2D->attrib("vert"), 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), NULL);

glEnableVertexAttribArray(gProgram2D->attrib("vertTexCoord"));
glVertexAttribPointer(gProgram2D->attrib("vertTexCoord"), 2, GL_FLOAT, GL_TRUE, 5 * sizeof(GLfloat), (const GLvoid*)(3 * sizeof(GLfloat)));

glBindVertexArray(0);
}


void Update(float secondsElapsed) {

if (glfwGetKey(gWindow, 'A') || glfwGetKey(gWindow, 'a')) {
pCreatureDisplay->resetTime();
}
if (glfwGetKey(gWindow, 'S') || glfwGetKey(gWindow, 's')) {
pCreatureDisplay->saveJSON("C:\\Users\\durands\\Desktop\\creature.json");
}
}

void OnScroll(GLFWwindow* window, double deltaX, double deltaY) {
gScrollY += deltaY;
}

void OnError(int errorCode, const char* msg) {
throw std::runtime_error(msg);
}


std::string createName(int gen, int id) {
sprintf_s(name, (NB_CREATURE <= 100)?"%d-%02d": "%d-%03d", gen, id);
std::string sname;
sname.append(name);
return sname;
}


void createMutation(Creature* creature, int nbCreature, std::vector<Creature*>* nextGeneration) {

for (int i = 0; i < nbCreature; i++) {

int nbMutation = creature->voxelEngine->voxelCount() / 10;
if (nbMutation < 2) nbMutation = 2;

Creature* newCreature = creature->mutate(nbMutation);

bool isSameInList = false;
for (int j = 0; j < nextGeneration->size(); j++) {
if (newCreature->isSame((*nextGeneration)[j])) {
isSameInList = true;
break;
}
}
if (!isSameInList) {
nextGeneration->push_back(newCreature);
}
else {
delete newCreature;
}
}
}


bool simulateAll(std::vector<Creature*>* nextGeneration, double simulationTime) {

bool exit = false;
std::vector<Creature*> toEvaluate;
Creature* creature, oldCreature;
for (int i = 0; i < nextGeneration->size(); i++) {
creature = (*nextGeneration)[i];
if (creature->getScore() < 0) {
std::string key = creature->getKey();
Creature* oldCreature = mapCreature[key];
if (oldCreature != NULL) {
creature->setScore(oldCreature->getScore());
} 
}
if (creature->getScore() < 0) {
toEvaluate.push_back(creature);
}
}

std::cout << toEvaluate.size() << "/" << nextGeneration->size() << " New creatures to evaluate\n";

if (toEvaluate.size() > 0) {
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < toEvaluate.size(); i++) {
toEvaluate[i]->simulate(i, simulationTime);
if (GetAsyncKeyState(VK_SPACE) & 0x8000) {
exit = true;
}
}
}

for (int i = 0; i < toEvaluate.size(); i++) {
mapCreature[toEvaluate[i]->getKey()] = toEvaluate[i];
}
}

std::cout << "\n\n";

std::sort(nextGeneration->begin(), nextGeneration->end(), Creature::sorter);

float score;
Creature *parent;

for (int i = 0; i < nextGeneration->size(); i++) {
creature = (*nextGeneration)[i];
parent = creature->getParent();
score = creature->getScore();

if (creature->getName() == "") {
creature->setName(createName(iGeneration, i));
}

if (parent == NULL) {
std::cout << "    ";
}
else {
std::cout << (score > parent->getScore() ? "+++ " :
score < parent->getScore() ? "--- " : "=== ");
}

std::cout << creature->getGenealogie() << " :  ";

if (parent == NULL) {
std::cout << score << "m\n";
}
else {
std::cout << parent->getScore() << " => " << score << "m\n";
}
}

return exit;
}


void initOpenGL() {

glfwSetErrorCallback(OnError);
if (!glfwInit())
throw std::runtime_error("glfwInit failed");

glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
glfwWindowHint(GLFW_SAMPLES, 4); 

gWindow = glfwCreateWindow((int)SCREEN_SIZE.x, (int)SCREEN_SIZE.y, "Go Darwin, go!", NULL, NULL);
if (!gWindow)
throw std::runtime_error("glfwCreateWindow failed. Can your hardware handle OpenGL 3.2?");

glfwMakeContextCurrent(gWindow);

glewExperimental = GL_TRUE; 

if (glewInit() != GLEW_OK)
throw std::runtime_error("glewInit failed");

while (glGetError() != GL_NO_ERROR) {}

std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

if (!GLEW_VERSION_3_2)
throw std::runtime_error("OpenGL 3.2 API is not available.");

glEnable(GL_DEPTH_TEST);
glDepthFunc(GL_LESS);
glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

LoadShaders();
LoadShaders2D();
LoadTriangle();
}


void DisplayCreature(Creature* creature) {

pCreatureDisplay->glInit();
pCreatureDisplay->glUpdate(gProgram->attrib("vert"), gProgram->attrib("vertTexCoord"), gProgram->attrib("vertColor"));

glm::vec3 pos = pCreatureDisplay->getPosition();

float t = .1*glfwGetTime();
float scale = 1.;
float r = scale*(.6 + .2*cos(t*.1));
gCamera.setPosition(glm::vec3(pos.x + r*cos(t), pos.y + r*sin(t),  scale*.5));
gCamera.lookAt(pos);
gCamera.setViewportAspectRatio(SCREEN_SIZE.x / SCREEN_SIZE.y);
gCamera.setFieldOfView(50.);

glClearColor(1., .2, .21, 1); 
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

gProgram->use();

gProgram->setUniform("camera", gCamera.matrix()); 
gProgram->setUniform("model", glm::mat4());

pCreatureDisplay->glRender();
gProgram->stopUsing();
}

void Display2DGUI()
{
gProgram2D->use();
glBindVertexArray(gVAO);
glDrawArrays(GL_TRIANGLES, 0, 2*3);
glBindVertexArray(0);
gProgram2D->stopUsing();
}
int min(int i, int j) {
return i < j ? i : j;
}

void DisplayCreatureGL() {
double lastTime = glfwGetTime();
double creatureTime = lastTime;

while (!glfwWindowShouldClose(gWindow)) {
glfwPollEvents();

#ifdef DISPLAY_ANIM
double thisTime = (double)timeGetTime() / 1000.;
#else
double thisTime = glfwGetTime();
#endif
float dt = (float)(thisTime - lastTime);
lastTime = thisTime;

Update(dt);

#ifdef WITH_SIMULATION
#pragma omp critical(pCreatureDisplay)
#endif
{
int nbTry1 = 0, nbTry3 = 0;

#ifdef DISPLAY_ANIM
if (thisTime - creatureTime > 2*(creatureID/5)+5) {
creatureID++;
if (creatureID < animCreature.size())
{
creatureTime = thisTime;
pCreatureDisplay = animCreature[min(creatureID, animCreature.size() - 1)]->clone();
pCreatureDisplay->animate(&nbTry1, &nbTry3);
pCreatureDisplay->resetTime();
}
}
#endif
double dtc = 0;

for (int i = 0; i < 200; i++) {
dtc += pCreatureDisplay->animate(&nbTry1, &nbTry3);
if (dtc >= dt)
break;
}

DisplayCreature(pCreatureDisplay);
}

glfwSwapBuffers(gWindow);

GLenum error = glGetError();
if (error != GL_NO_ERROR)
std::cerr << "OpenGL Error " << error << std::endl;


if (glfwGetKey(gWindow, GLFW_KEY_ESCAPE))
glfwSetWindowShouldClose(gWindow, GL_TRUE);
}
}


void DoEvolution(Creature* pCreature1) {
bool exit = false;

std::vector<Creature*> sortedGeneration;
std::vector<Creature*> nextGeneration;

sortedGeneration.push_back(pCreature1);


while (!glfwWindowShouldClose(gWindow)) {

nextGeneration.clear();

for (int i = 0; i < NB_SAVED && i<sortedGeneration.size(); i++) {
nextGeneration.push_back(sortedGeneration[i]);
}

size_t remains = NB_CREATURE - nextGeneration.size();
for (int j = 0; j < remains; j++) {
Creature* creatureRandom = sortedGeneration[(rand() % (j + 1)) % sortedGeneration.size()];
createMutation(creatureRandom, 1, &nextGeneration);
}

for (int i = NB_SAVED; i < sortedGeneration.size(); i++) {
sortedGeneration[i]->clearMemory();
}
sortedGeneration.clear();

sortedGeneration.insert(sortedGeneration.end(), nextGeneration.begin(), nextGeneration.end());

if (simulateAll(&sortedGeneration, SIMULATION_DURATION)) {

}

double moy = 0;
for (int i = 0; i < sortedGeneration.size(); i++) {
moy += sortedGeneration[i]->getScore();
}
moy /= (double)sortedGeneration.size();

std::cout << "\n--------";
std::cout << "\nGeneration : " << iGeneration;
std::cout << "\n  Best: " << sortedGeneration[0]->getScore();
std::cout << "\n  Average:" << moy;
std::cout << "\n--------\n";

iGeneration++;


if (sortedGeneration[0]->getScore() > pCreatureDisplay->getScore())
#ifdef WITH_DISPLAY
#pragma omp critical(pCreatureDisplay)
#endif
{
if (pCreatureDisplay != NULL) delete pCreatureDisplay;	
pCreatureDisplay = sortedGeneration[0]->clone();
#ifdef WITH_DISPLAY
int nbTry1 = 0, nbTry3 = 0;
pCreatureDisplay->animate(&nbTry1, &nbTry3);
pCreatureDisplay->resetTime();			
#endif 
sprintf_s(numstr, "%03d", creatureID);
std::string filePath = "C:\\Users\\durands\\Desktop\\Creature\\creatureSave";
filePath += numstr;
filePath += ".json";
sortedGeneration[0]->saveJSON(filePath);
creatureID++;
}


}
}

void AppMain(int argc, char *argv[]) {



Creature* pCreature1;

#ifdef DISPLAY_ANIM

for (int i = 0; i < 33; i++) {
sprintf_s(numstr, "%03d", i);
std::string filePath = "C:\\Users\\durands\\Desktop\\Creature\\creatureSave";
filePath += numstr;
filePath += ".json";
animCreature.push_back(new Creature(filePath));
}
pCreature1 = animCreature[0];

#else

if (argc == 2) {
pCreature1 = new Creature(argv[1]);
std::cout << "Open: " << argv[1];
}
else {

sprintf_s(numstr, "%03d", creatureID);
std::string filePath = "C:\\Users\\durands\\Desktop\\Creature\\creatureSave";
filePath += numstr;
filePath += ".json";
creatureID++;

pCreature1 = new Creature(filePath);

}
#endif

int nbTry1=0, nbTry3=0;
pCreatureDisplay = pCreature1->clone();
pCreatureDisplay->animate(&nbTry1, &nbTry3);
pCreatureDisplay->resetTime();

#if defined(WITH_DISPLAY) && defined(WITH_SIMULATION)
omp_set_num_threads(2);
omp_set_nested(1);

#pragma omp parallel sections
{
#pragma omp section
{
#endif
#ifdef WITH_DISPLAY
initOpenGL();
DisplayCreatureGL();
glfwTerminate();
#endif
#if defined(WITH_DISPLAY) && defined(WITH_SIMULATION)		
}
#pragma omp section
{
#endif
#ifdef WITH_SIMULATION
DoEvolution(pCreature1);
#endif
#if defined(WITH_DISPLAY) && defined(WITH_SIMULATION)
}
}
#endif

}


int main(int argc, char *argv[]) {
try {
AppMain(argc, argv);
} catch (const std::exception& e){
std::cerr << "ERROR: " << e.what() << std::endl;
return EXIT_FAILURE;
}

return EXIT_SUCCESS;
}
