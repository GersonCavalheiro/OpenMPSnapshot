
#include "Renderer.h"


Renderer::Renderer(int frameWidth, int frameHeight, int samplesPerPixel):
frameWidth(frameWidth), frameHeight(frameHeight), samplesPerPixel(samplesPerPixel) {

cameraPosition = Vector3();
cameraDirection = Vector3();
}

void Renderer::setCameraPosition(double x, double y, double z) {
cameraPosition = Vector3(x, y, z);
}

void Renderer::setCameraDirection(double x, double y, double z) {
cameraDirection = Vector3(x, y, z);
}

void Renderer::addObject(Sphere hittableObject) {
hittableObjects.push_back(hittableObject);
}

Vector3 Renderer::radiance(const Ray &ray, int depth, unsigned short *seed) {
double t;
int objectId = 0;
if (!intersect(ray, t, objectId))
return Vector3();
Sphere &hittableObject = hittableObjects[objectId];
Vector3 x = ray.getOrigin() + ray.getDirection() * t;
Vector3 n = (x - hittableObject.getPosition()).normalize();
Vector3 nl = n.dotProduct(ray.getDirection()) < 0 ? n : n * -1;
Vector3 f = hittableObject.getColor();
double p = f.getX() > f.getY() && f.getX() > f.getZ() ? f.getX() : f.getY() > f.getZ() ? f.getY() : f.getZ();
if (++depth > 5) {
if (erand48(seed) < p)
f = f * (1 / p);
else
return hittableObject.getEmission();
}
if (hittableObject.getMaterial() == DIFFUSE) {
double r1 = 2 * M_PI * erand48(seed);
double r2 = erand48(seed);
double r2s = sqrt(r2);
Vector3 w = nl;
Vector3 u = ((fabs(w.getX()) > .1 ? Vector3(0, 1, 0) : Vector3(1, 0, 0)) % w).normalize();
Vector3 v = w % u;
Vector3 d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).normalize();
return hittableObject.getEmission() + f.multiply(radiance(Ray(x, d), depth, seed));
}
else if (hittableObject.getMaterial() == SPECULAR) {
return hittableObject.getEmission() + f.multiply(radiance(Ray(
x, ray.getDirection() - n * 2 * n.dotProduct(ray.getDirection())), depth, seed));
}
Ray reflectedRay(x, ray.getDirection() - n * 2 * n.dotProduct(ray.getDirection()));
bool into = n.dotProduct(nl) > 0;
double nc = 1, nt = 1.5, cos2t;
double  nnt = into ? nc / nt : nt / nc;
double ddn = ray.getDirection().dotProduct(nl);
if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0)
return hittableObject.getEmission() + f.multiply(radiance(reflectedRay, depth, seed));
Vector3 tDirection = (ray.getDirection() * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).normalize();
double a = nt - nc;
double b = nt + nc;
double R0 = a * a / (b * b);
double c = 1 - (into ? -ddn : tDirection.dotProduct(n));
double Re = R0 + (1 - R0) * c * c * c * c * c;
double Tr = 1 - Re;
double P = .25 + .5 * Re;
double RP = Re / P;
double TP = Tr / (1 - P);
return hittableObject.getEmission() + f.multiply(depth > 2 ? (erand48(seed) < P ?
radiance(reflectedRay, depth, seed) * RP : radiance(Ray(x, tDirection), depth, seed) * TP) :
radiance(reflectedRay, depth, seed) * Re + radiance(Ray(x, tDirection), depth, seed) * Tr);
}

void Renderer::render(bool enableProgressBar) {
samplesPerPixel = samplesPerPixel / 4;
Ray camera(cameraPosition, cameraDirection.normalize());
Vector3 cx = Vector3(frameWidth * .5135 / frameHeight, 0, 0);
Vector3 r;
Vector3 cy = (cx % camera.getDirection()).normalize() * .5135;
Vector3 *c=new Vector3[frameWidth * frameHeight];
tqdm progressBar;
printf("Rendering Scene...\n");
#pragma omp parallel for schedule(dynamic, 1) private(r)
for (int y = 0; y < frameHeight; y++) {
if (enableProgressBar)
progressBar.progress(frameHeight, frameHeight - y);
else
fprintf(stderr, "\rProgress -> %5.2f%%", 100. * y / (frameHeight - 1));
unsigned short seedY = (unsigned short) y * (unsigned short) y * (unsigned short) y;
for (unsigned short x = 0, seed[3] = {0, 0, seedY}; x < frameWidth; x++) {
for (int sy = 0, i = (frameHeight - y - 1) * frameWidth + x; sy < 2; sy++) {
for (int sx = 0; sx < 2; sx++, r=Vector3()) {
for (int s = 0; s < samplesPerPixel; s++) {
double r1 = 2 * erand48(seed);
double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
double r2 = 2 * erand48(seed);
double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
Vector3 d = cx * ( ( (sx + .5 + dx) / 2 + x) / frameWidth - .5) +
cy * ( ( (sy + .5 + dy) / 2 + y) / frameHeight - .5) + camera.getDirection();
r = r + radiance(
Ray(camera.getOrigin() + d * 140, d.normalize()),0, seed
) * (1. / samplesPerPixel);
}
c[i] = c[i] + Vector3(clamp(r.getX()), clamp(r.getY()), clamp(r.getZ())) * .25;
}
}
}
}
FILE *f = fopen("image.ppm", "w");
fprintf(f, "P3\n%d %d\n%d\n", frameWidth, frameHeight, 255);
for (int i = 0; i < frameWidth * frameHeight; i++)
fprintf(f, "%d %d %d ", toInt(c[i].getX()), toInt(c[i].getY()), toInt(c[i].getZ()));
fclose(f);
}

void Renderer::exportWorld(char *worldFileName) {
FILE *worldFile = fopen(worldFileName, "w");
fprintf(
worldFile,
"ObjectID, Radius, PositionX, PositionY, PositionZ, EmissionX, EmissionY, EmissionZ, ColorR, ColorG, ColorB, Material\n");
int objectID = 0;
for(Sphere& sphere : hittableObjects) {
fprintf(
worldFile, "%d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %d\n",
objectID, sphere.getRadius(),
sphere.getPosition().getX(), sphere.getPosition().getY(), sphere.getPosition().getZ(),
sphere.getEmission().getX(), sphere.getEmission().getY(), sphere.getEmission().getZ(),
sphere.getColor().getX(), sphere.getColor().getY(), sphere.getColor().getZ(), sphere.getMaterial());
objectID++;
}
fclose(worldFile);
}

void Renderer::exportCamera(char *cameraFileName) {
FILE *cameraFile = fopen(cameraFileName, "w");
fprintf(
cameraFile,
"PositionX, PositionY, PositionZ, DirectionX, DirectionY, DirectionZ\n");
fprintf(
cameraFile, "%lf, %lf, %lf, %lf, %lf, %lf\n",
cameraPosition.getX(), cameraPosition.getY(), cameraPosition.getZ(),
cameraDirection.getX(), cameraDirection.getY(), cameraDirection.getZ());
fclose(cameraFile);
}

void Renderer::importWorld(std::string worldFileName) {
std::ifstream infile(worldFileName);
std::string line;
int count = 0;
while (std::getline(infile, line)) {
std::stringstream ss(line);
std::vector<std::string> temp;
if (count > 0) {
while (ss.good()) {
std::string substr;
std::getline(ss, substr, ',');
temp.push_back(substr);
}
Sphere sphere(
std::stod(temp.at(1)),
Vector3(
std::stod(temp.at(2)),
std::stod(temp.at(3)),
std::stod(temp.at(4))),
Vector3(
std::stod(temp.at(5)),
std::stod(temp.at(6)),
std::stod(temp.at(7))),
Vector3(
std::stod(temp.at(8)),
std::stod(temp.at(9)),
std::stod(temp.at(10))),
static_cast<SurfaceReflectionType>(std::stoi(temp.at(11))));
addObject(sphere);
}
count++;
}
}

void Renderer::importCamera(std::string cameraFileName) {
std::ifstream infile(cameraFileName);
std::string line;
int count = 0;
std::getline(infile, line);
while (std::getline(infile, line)) {
std::stringstream ss(line);
std::vector<std::string> temp;
while (ss.good()) {
std::string substr;
std::getline(ss, substr, ',');
temp.push_back(substr);
}
setCameraPosition(
std::stod(temp.at(0)),
std::stod(temp.at(1)),
std::stod(temp.at(2)));
setCameraDirection(
std::stod(temp.at(3)),
std::stod(temp.at(4)),
std::stod(temp.at(5)));
}
}
