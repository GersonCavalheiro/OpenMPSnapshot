#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include "Scene.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "Triangle.h"
int Scene::readSceneFile(string fileName) {
string line;
ifstream input(fileName);
if (input.fail()) {
cout << "Can't open file '" << fileName << "'" << endl;
return 1;
}
Material material;
output_image = "raytraced.bmp";
maxDepth = 5;
maxNormals = -1;
maxVertices = -1;
samplingMethod = BASIC;
samplingRate = 3;
string command;
while (input >> command) { 
if (command[0] == '#') {
getline(input, line); 
cout << "Skipping comment: " << command << line << endl;
continue;
}
if (command == "material") { 
Colour a{}, d{}, s{}, t{};
float ns, ior;
input >> a.r >> a.g >> a.b >> d.r >> d.g >> d.b >> s.r >> s.g >> s.b >> ns >> t.r >> t.g >> t.b >> ior;
material.setMaterial(a, d, s, t, ns, ior);
printf("Material configuration changed to a(%f, %f, %f), d(%f, %f, %f), s(%f, %f, %f), t(%f, %f, %f), ns %f and ior %f.\n",
a.r, a.g, a.b, d.r, d.g, d.b, s.r, s.g, s.b, t.r, t.g, t.b, ns, ior);
} else if (command == "camera") { 
float ha;
Vector3D p{}, d{}, u{};
input >> p.x >> p.y >> p.z >> d.x >> d.y >> d.z >> u.x >> u.y >> u.z >> ha;
camera.setCamera(p, normalize(d), normalize(u), ha);
printf("Camera at position (%f, %f, %f) with view direction (%f, %f, %f), up (%f, %f, %f) and half angle %f.\n",
p.x, p.y, p.z, d.x, d.y, d.z, u.x, u.y, u.z, ha);
} else if (command == "sphere") { 
float r;
Vector3D p{};
input >> p.x >> p.y >> p.z >> r;
auto sphere = (Surface*)new Sphere(p, r, material);
surfaces.push_back(sphere);
printf("Sphere at position (%f, %f, %f) with radius %f.\n", p.x, p.y, p.z, r);
} else if (command == "triangle") { 
int v1, v2, v3;
input >> v1 >> v2 >> v3;
auto triangle = (Surface*)new Triangle(vertices[v1], vertices[v2], vertices[v3], material);
surfaces.push_back(triangle);
printf("Triangle using vertices (%d, %d, %d).\n", v1, v2, v3);
} else if (command == "normal_triangle") { 
int v1, v2, v3, n1, n2, n3;
input >> v1 >> v2 >> v3 >> n1 >> n2 >> n3;
auto triangle = (Surface*)new Triangle(vertices[v1], vertices[v2], vertices[v3],
normals[n1], normals[n2], normals[n3], material);
surfaces.push_back(triangle);
printf("Triangle using vertices (%d, %d, %d) and normals (%d, %d, %d).\n", v1, v2, v3, n1, n2, n3);
} else if (command == "vertex") { 
if (maxVertices < 0) {
cout << "max_vertices should be specified before vertex." << endl;
return 2;
}
Vector3D p{};
input >> p.x >> p.y >> p.z;
vertices.push_back(p);
printf("Vertex #%ld at position (%f, %f, %f).\n", vertices.size(), p.x, p.y, p.z);
} else if (command == "normal") { 
if (maxNormals < 0) {
cout << "max_normals should be specified before normal." << endl;
return 2;
}
Vector3D d{};
input >> d.x >> d.y >> d.z;
normals.push_back(normalize(d));
printf("Normal #%ld in direction (%f, %f, %f).\n", normals.size(), d.x, d.y, d.z);
} else if (command == "background") { 
float r, g, b;
input >> r >> g >> b;
setRGB(&background, r, g, b);
printf("Background color of (%f, %f, %f).\n", r, g, b);
} else if (command == "ambient_light") { 
Colour a{};
input >> a.r >> a.g >> a.b;
ambientLight.c = a;
printf("Ambient Light color of (%f, %f, %f).\n", a.r, a.g, a.b);
} else if (command == "point_light") { 
Colour c{};
Vector3D p{};
input >> c.r >> c.g >> c.b >> p.x >> p.y >> p.z;
Light pointLight{c, p};
pointLights.push_back(pointLight);
printf("Point Light added at (%f, %f, %f) with color of (%f, %f, %f).\n", p.x, p.y, p.z, c.r, c.g, c.b);
} else if (command == "directional_light") { 
Colour c{};
Vector3D v{};
input >> c.r >> c.g >> c.b >> v.x >> v.y >> v.z;
Light directionalLight{c, normalize(v)};
directionalLights.push_back(directionalLight);
printf("Directional Light added in direction (%f, %f, %f) with color of (%f, %f, %f).\n", v.x, v.y, v.z,
c.r, c.g, c.b);
} else if (command == "spot_light") { 
Colour c{};
Vector3D v{};
Vector3D p{};
float a1, a2;
input >> c.r >> c.g >> c.b >> p.x >> p.y >> p.z >> v.x >> v.y >> v.z >> a1 >> a2;
SpotLight spotLight{c, p, normalize(v), a1, a2};
spotLights.push_back(spotLight);
printf("Spot Light added at (%f, %f, %f) in direction (%f, %f, %f) with color of (%f, %f, %f) and angles (%f, %f).\n",
p.x, p.y, p.z, v.x, v.y, v.z, c.r, c.g, c.b, a1, a2);
} else if (command == "max_vertices"){ 
input >> maxVertices;
printf("Max Vertices set as: %d\n", maxVertices);
} else if (command == "max_normals"){ 
input >> maxNormals;
printf("Max Normals set as: %d\n", maxNormals);
} else if (command == "supersampling"){ 
int supersamplingMethod;
input >> supersamplingMethod >> samplingRate;
samplingMethod = SamplingMethod(supersamplingMethod);
printf("Supersampling method set to %d with sampling rate of: %d\n", samplingMethod, samplingRate);
} else if (command == "output_image") { 
string outFile;
input >> outFile;
output_image = outFile;
printf("Render to file named: %s\n", outFile.c_str());
} else if (command == "film_resolution") { 
int width, height;
input >> width >> height;
camera.setFilm(width, height);
printf("Set film resolution as: %d x %d.\n", width, height);
} else if (command == "max_depth") { 
input >> maxDepth;
printf("Set maximum depth as: %d.\n", maxDepth);
} else {
getline(input, line); 
cout << "WARNING. Do not know command: " << command << endl;
}
}
return 0;
}
void Scene::writeImageResult() {
stbi_write_bmp(output_image.c_str(), camera.getFilmWidth(), camera.getFilmHeight(), 4, data.raw);
}
void Scene::initializeFilm(Component r, Component g, Component b, Component a) {
int num_pixels = camera.getFilmWidth() * camera.getFilmHeight();
data.raw = new uint8_t[num_pixels * 4];
int byte = 0; 
for (int i = 0; i < camera.getFilmWidth(); ++i) {
for (int j = 0; j < camera.getFilmHeight(); ++j) {
data.raw[byte++] = r;
data.raw[byte++] = g;
data.raw[byte++] = b;
data.raw[byte++] = a;
}
}
}
void Scene::rayTrace() {
initializeFilm(0, 0, 0, 255);
#pragma omp parallel for schedule(dynamic, 10) collapse(2)
for (int j = 0; j < camera.getFilmHeight(); ++j) {
for (int i = 0; i < camera.getFilmWidth(); ++i) {
Colour pixelColour{};
switch (samplingMethod) {
case BASIC:
pixelColour = basicSampling(i, j);
break;
case UNIFORM:
pixelColour = regularSampling(i, j);
break;
case RANDOM:
pixelColour = randomSampling(i, j);
break;
case JITTERED:
pixelColour = jitteredSampling(i, j);
break;
}
data.pixels[(camera.getFilmHeight() - 1 - j) * camera.getFilmWidth() + (camera.getFilmWidth() - 1 - i)] = Pixel(pixelColour);
}
}
}
HitInfo *Scene::hit(Ray viewingRay, float tMin, float tMax, bool isShadowRay) {
float tBest = INFINITY;
HitInfo *hitBest = nullptr;
for (const auto &surface : surfaces) {
HitInfo *hitInfo = surface->intersect(viewingRay, tMin, tMax);
if (hitInfo != nullptr) {
if(isShadowRay)
return hitInfo;
if (hitInfo->getT() < tBest) {
hitBest = hitInfo;
tBest = hitInfo->getT();
}
}
}
return hitBest;
}
Colour Scene::getColour(Ray viewingRay, int depth) {
HitInfo *hitBest = hit(viewingRay, 0.01, INFINITY, false);
if (hitBest != nullptr) {
Colour finalColour = ambientLight.c * hitBest->getMaterial().getAmbient();
for (auto light : pointLights) {
Vector3D lambertian = light.v - hitBest->getPoint();
HitInfo *shadowHit = hit(Ray(hitBest->getPoint(), normalize(lambertian)), 0.01, magnitude(lambertian), true);
if (shadowHit == nullptr) {
Colour fallOffIntensity = light.c / pow(magnitude(lambertian), 2);
Vector3D half = lambertian - hitBest->getD();
finalColour =
finalColour + fallOffIntensity * fmaxf(0.0, normalize(lambertian) * hitBest->getNormal())
* hitBest->getMaterial().getDiffuse();
finalColour = finalColour + fallOffIntensity *
pow(fmaxf(0.0, normalize(half) * hitBest->getNormal()),
hitBest->getMaterial().getNs())
* hitBest->getMaterial().getSpecular();
}
}
for (auto light: directionalLights) {
Vector3D lambertian = -light.v;
HitInfo *shadowHit = hit(Ray(hitBest->getPoint(), normalize(lambertian)), 0.01, INFINITY, true);
if (shadowHit == nullptr) {
Vector3D half = lambertian - hitBest->getD();
finalColour =
finalColour + light.c * fmaxf(0.0, normalize(lambertian) * hitBest->getNormal())
* hitBest->getMaterial().getDiffuse();
finalColour = finalColour + light.c *
pow(fmaxf(0.0, normalize(half) * hitBest->getNormal()),
hitBest->getMaterial().getNs())
* hitBest->getMaterial().getSpecular();
}
}
for(auto light: spotLights){
Vector3D lambertian = light.p - hitBest->getPoint();
HitInfo *shadowHit = hit(Ray(hitBest->getPoint(), normalize(lambertian)), 0.01, magnitude(lambertian), true);
if (shadowHit == nullptr) {
float angle = angleBetween(-lambertian, light.d);
Colour fallOffIntensity = light.c;
if(angle>light.angle2)
continue;
else if(angle>=light.angle1){
float factor = 1 - (angle-light.angle1)/(light.angle2-light.angle1);
fallOffIntensity = fallOffIntensity * factor;
}
fallOffIntensity = fallOffIntensity / pow(magnitude(lambertian), 2);
Vector3D half = lambertian - hitBest->getD();
finalColour =
finalColour + fallOffIntensity * fmaxf(0.0, normalize(lambertian) * hitBest->getNormal())
* hitBest->getMaterial().getDiffuse();
finalColour = finalColour + fallOffIntensity *
pow(fmaxf(0.0, normalize(half) * hitBest->getNormal()),
hitBest->getMaterial().getNs())
* hitBest->getMaterial().getSpecular();
}
}
if(depth > 0) {
float R = 1.0f;
if(!isBlack(hitBest->getMaterial().getTransmissive())) {
Vector3D refracted{};
float cosine = INFINITY;
if(hitBest->getD()*hitBest->getNormal() < 0) {
cosine = -hitBest->getD()*hitBest->getNormal();
getRefractedRay(hitBest->getD(),hitBest->getNormal(), hitBest->getMaterial().getIor(), &refracted);
} else if(getRefractedRay(hitBest->getD(), -hitBest->getNormal(), 1/hitBest->getMaterial().getIor(), &refracted)){
cosine = refracted * hitBest->getNormal();
}
if(cosine!=INFINITY) {
float R0 = powf(hitBest->getMaterial().getIor() - 1, 2) / powf(hitBest->getMaterial().getIor() + 1, 2);
R = R0 + (1 - R0) * powf(1 - cosine, 5);
finalColour = finalColour + hitBest->getMaterial().getTransmissive() *
getColour(Ray(hitBest->getPoint(), normalize(refracted)), depth - 1) * (1-R);
}
}
if (!isBlack(hitBest->getMaterial().getSpecular())) {
Vector3D reflected =
hitBest->getD() - hitBest->getNormal() * 2 * (hitBest->getD() * hitBest->getNormal());
finalColour = finalColour + hitBest->getMaterial().getSpecular() *
getColour(Ray(hitBest->getPoint(), normalize(reflected)), depth - 1) * R;
}
}
return finalColour;
}
return background;
}
Colour Scene::basicSampling(int i, int j) {
Ray viewingRay = camera.getRay(i + 0.5f, j + 0.5f);
return getColour(viewingRay, maxDepth);
}
Colour Scene::regularSampling(int i, int j) {
Colour c{0.0f, 0.0f, 0.0f};
for (int p = 0; p < samplingRate; p++) {
for (int q = 0; q < samplingRate; q++) {
Ray viewingRay = camera.getRay(i + (p + 0.5f) / samplingRate, j + (q + 0.5f) / samplingRate);
c = c + getColour(viewingRay, maxDepth);
}
}
return c / pow(samplingRate, 2);
}
Colour Scene::randomSampling(int i, int j) {
Colour c{0.0f, 0.0f, 0.0f};
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<float> dist(0.0, 1.0);
for (int p = 0; p < pow(samplingRate, 2); p++) {
Ray viewingRay = camera.getRay(i + dist(mt), j + dist(mt));
c = c + getColour(viewingRay, maxDepth);
}
return c / pow(samplingRate, 2);
}
Colour Scene::jitteredSampling(int i, int j) {
Colour c{0.0f, 0.0f, 0.0f};
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<float> dist(0.0, 1.0);
for (int p = 0; p < samplingRate; p++) {
for (int q = 0; q < samplingRate; q++) {
Ray viewingRay = camera.getRay(i + (p + dist(mt)) / samplingRate, j + (q + dist(mt)) / samplingRate);
c = c + getColour(viewingRay, maxDepth);
}
}
return c / pow(samplingRate, 2);
}
bool Scene::getRefractedRay(Vector3D d,Vector3D n, float ior, Vector3D *refracted) {
float c = 1-(1-powf(d*n, 2))/powf(ior, 2);
if(c<0)
return false;
*refracted = (d - n*(d*n))*(1/ior) - n*sqrt(c);
*refracted = normalize(*refracted);
return true;
}
