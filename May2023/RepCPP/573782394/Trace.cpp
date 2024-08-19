
#include "../include/Trace.hpp"

#include <omp.h>

#include <algorithm>
#include <fstream>

#include "../include/Material.hpp"
#include "../include/Triangle.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../third-parties/tinyobjloader/tiny_obj_loader.h"

namespace sre {
Tracer::Tracer(size_t _depth, size_t _samples, float _p)
: scenes(nullptr), maxDepth(_depth), samples(_samples), thresholdP(_p) {}

Tracer::~Tracer() {
if (scenes != nullptr) {
delete scenes;
}
scenes = nullptr;
}

void Tracer::loadExampleScene() {
float eyePosZ = 2;
camera.setWidth(500);
camera.setHeight(500);
camera.setFovy(90);
camera.setEye(0, 0, -eyePosZ);
camera.setLookAt(0, 0, -1.5);
camera.setWorld(0, 1, 0);

float w = eyePosZ;
float h = eyePosZ;
float l = eyePosZ * 2;

std::vector<Triangle> triangles;

Material m;
m.setEmissive(true);
m.setEmission(34, 24, 8);
m.setDiffusion(0, 0, 0);
m.setSpecularity(0, 0, 0);
m.setTransmittance(1, 1, 1);
m.setShiness(1);
m.setRefraction(0);
triangles.emplace_back(0, Vec3<float>(-0.5, 2, 2), Vec3<float>(-0.5, 2, 0.5),
Vec3<float>(0.5, 2, 0.5), m);
triangles.emplace_back(1, Vec3<float>(-0.5, 2, 2), Vec3<float>(0.5, 2, 0.5),
Vec3<float>(0.5, 2, 2), m);
triangles.emplace_back(2, Vec3<float>(-0.5, 2, -2),
Vec3<float>(-0.5, 2, -0.5), Vec3<float>(0.5, 2, -0.5),
m);
triangles.emplace_back(3, Vec3<float>(-0.5, 2, -2), Vec3<float>(0.5, 2, -2),
Vec3<float>(0.5, 2, -0.5), m);
light.setLight(triangles[0]);
light.setLight(triangles[1]);
light.setLight(triangles[2]);
light.setLight(triangles[3]);

m.setEmissive(false);
m.setEmission(0, 0, 0);
m.setDiffusion(0.79, 0.76, 0.73);
m.setSpecularity(0, 0, 0);
m.setTransmittance(1, 1, 1);
m.setShiness(1);
m.setRefraction(0);
triangles.emplace_back(4, Vec3<float>(-w * 3, -h, l),
Vec3<float>(w * 3, -h, -1),
Vec3<float>(-w * 3, -h, -1), m);
triangles.emplace_back(5, Vec3<float>(-w * 3, -h, l),
Vec3<float>(w * 3, -h, l), Vec3<float>(w * 3, -h, -1),
m);

m.setEmissive(false);
m.setDiffusion(0.79, 0.76, 0.73);
m.setSpecularity(0, 0, 0);
m.setTransmittance(1, 1, 1);
m.setShiness(1);
m.setRefraction(0);
triangles.emplace_back(6, Vec3<float>(-w, h, l), Vec3<float>(-w, h, -1),
Vec3<float>(w, h, -1), m);
triangles.emplace_back(7, Vec3<float>(-w, h, l), Vec3<float>(w, h, -1),
Vec3<float>(w, h, l), m);

m.setEmissive(false);
m.setDiffusion(0.79, 0.76, 0.73);
m.setSpecularity(0, 0, 0);
m.setTransmittance(1, 1, 1);
m.setShiness(1);
m.setRefraction(0);
triangles.emplace_back(8, Vec3<float>(-w, -h, l), Vec3<float>(-w, h, l),
Vec3<float>(w, h, l), m);
triangles.emplace_back(9, Vec3<float>(-w, -h, l), Vec3<float>(w, h, l),
Vec3<float>(w, -h, l), m);

m.setEmissive(false);
m.setDiffusion(0.2, 0.76, 0);
m.setSpecularity(0, 0, 0);
m.setTransmittance(1, 1, 1);
m.setShiness(1);
m.setRefraction(0);
triangles.emplace_back(10, Vec3<float>(-w, h, 0), Vec3<float>(-w, h, l),
Vec3<float>(-w, -h, 0), m);
triangles.emplace_back(11, Vec3<float>(-w, h, l), Vec3<float>(-w, -h, l),
Vec3<float>(-w, -h, 0), m);

m.setEmissive(false);
m.setDiffusion(0, 0.24, 0.9);
m.setSpecularity(0, 0, 0);
m.setTransmittance(1, 1, 1);
m.setShiness(1);
m.setRefraction(0);
triangles.emplace_back(12, Vec3<float>(w, h, 0), Vec3<float>(w, -h, 0),
Vec3<float>(w, h, l), m);
triangles.emplace_back(13, Vec3<float>(w, h, l), Vec3<float>(w, -h, 0),
Vec3<float>(w, -h, l), m);


m.setEmissive(false);
m.setDiffusion(0, 0, 0);
m.setSpecularity(0, 0, 0);
m.setTransmittance(0.8, 1, 0.95);
m.setShiness(1);
m.setRefraction(1.5);
triangles.emplace_back(14, Vec3<float>(-0.4, 0.4, 0.2),
Vec3<float>(0.4, 0.4, 0.2), Vec3<float>(0, -0.4, 0.2),
Vec3<float>(0, 0, -1), m);
for (const auto &triangle : triangles) {
Hittable *obj = new Triangle(triangle);
objects.push_back(obj);
}

std::vector<Hittable *> objectsCopy(objects);
std::sort(objectsCopy.begin(), objectsCopy.end(), BVHNode::zCmp);
scenes = new BVH(objectsCopy, 0, objects.size());

printStatus();
}

bool Tracer::loadConfiguration(
const std::string &configName,
std::unordered_map<std::string, Vec3<float>> &lightRadiances) {
std::ifstream ifs;
ifs.open(configName, std::ios::in);
if (!ifs.is_open()) {
return false;
}

std::string buf;
int i, j;

int width, height;
float fovy;
float xyzs[3][3];

getline(ifs, buf);
getline(ifs, buf);
i = 0;
while (i < buf.size()) {
switch (buf[i]) {
case 'w':
assert(buf[i + 1] == 'i' && buf[i + 2] == 'd' && buf[i + 3] == 't' &&
buf[i + 4] == 'h' && buf[i + 5] == '=' && buf[i + 6] == '\"');
i += 7;
j = buf.find('\"', i);
width = stoi(buf.substr(i, j - i));
i = j + 1;
break;
case 'h':
assert(buf[i + 1] == 'e' && buf[i + 2] == 'i' && buf[i + 3] == 'g' &&
buf[i + 4] == 'h' && buf[i + 5] == 't' && buf[i + 6] == '=' &&
buf[i + 7] == '\"');
i += 8;
j = buf.find('\"', i);
height = stoi(buf.substr(i, j - i));
i = j + 1;
break;
case 'f':
assert(buf[i + 1] == 'o' && buf[i + 2] == 'v' && buf[i + 3] == 'y' &&
buf[i + 4] == '=' && buf[i + 5] == '\"');
i += 6;
j = buf.find('\"', i);
fovy = stof(buf.substr(i, j - i));
i = j + 1;
break;
default:
i += 1;
break;
}
}

for (int row = 0; row < 3; row++) {
getline(ifs, buf);
i = 0;

int col = 0;
while (i < buf.size()) {
if (i > 0 && buf[i - 1] == ' ' &&
(buf[i] == 'x' || buf[i] == 'y' || buf[i] == 'z')) {
assert(buf[i + 1] == '=' && buf[i + 2] == '\"');
i += 3;
j = buf.find('\"', i);
xyzs[row][col] = stof(buf.substr(i, j - i));
i = j + 1;
col += 1;
} else {
i += 1;
}
}
}

while (getline(ifs, buf)) {
std::string mtlname;
Vec3<float> radiance;
i = 0;
while (i < buf.size()) {
if (i > 0 && buf[i - 1] == ' ' && buf[i] == 'm') {
assert(buf.substr(i, 9) == "mtlname=\"");
i += 9;
j = buf.find('\"', i);
mtlname = buf.substr(i, j - i);
i = j + 1;
} else if (i > 0 && buf[i - 1] == ' ' && buf[i] == 'r') {
assert(buf.substr(i, 10) == "radiance=\"");
i += 10;
j = buf.find(',', i);
radiance.x = stof(buf.substr(i, j - i));
i = j + 1;

j = buf.find(',', i);
radiance.y = stof(buf.substr(i, j - i));
i = j + 1;

j = buf.find('\"', i);
radiance.z = stof(buf.substr(i, j - i));
i = j + 1;
} else {
i += 1;
}
}
if (!mtlname.empty()) {
lightRadiances.emplace(mtlname, radiance);
}
}

camera.setWidth(width);
camera.setHeight(height);
camera.setFovy(fovy);
camera.setEye(xyzs[0][0], xyzs[0][1], xyzs[0][2]);
camera.setLookAt(xyzs[1][0], xyzs[1][1], xyzs[1][2]);
camera.setWorld(xyzs[2][0], xyzs[2][1], xyzs[2][2]);
return true;
}

bool Tracer::loadModel(
const std::string &modelName, const std::string &pathName,
const std::unordered_map<std::string, Vec3<float>> &lightRadiances) {
tinyobj::attrib_t attrib;
std::vector<tinyobj::shape_t> shapes;
std::vector<tinyobj::material_t> materials;
std::string warn;
std::string err;
if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, &warn,
modelName.c_str(), pathName.c_str())) {
return false;
}

std::vector<Material> actualMaterials;
for (const auto &material : materials) {
Material actualMaterial;

auto itr = lightRadiances.find(material.name);
actualMaterial.setName(material.name);

if (itr != lightRadiances.end()) {
actualMaterial.setEmissive(true);
actualMaterial.setEmission(itr->second.x, itr->second.y, itr->second.z);
} else {
actualMaterial.setEmissive(false);
}

actualMaterial.setDiffusion(material.diffuse[0], material.diffuse[1],
material.diffuse[2]);
actualMaterial.setSpecularity(material.specular[0], material.specular[1],
material.specular[2]);
actualMaterial.setTransmittance(material.transmittance[0],
material.transmittance[1],
material.transmittance[2]);
actualMaterial.setShiness(material.shininess);
actualMaterial.setRefraction(material.ior);

if (!material.ambient_texname.empty()) {
actualMaterial.setAmbientTexture(pathName + material.ambient_texname);
}
if (!material.diffuse_texname.empty()) {
actualMaterial.setDiffuseTexture(pathName + material.diffuse_texname);
}
if (!material.specular_texname.empty()) {
actualMaterial.setSpecularTexture(pathName + material.specular_texname);
}

actualMaterials.emplace_back(actualMaterial);
}

size_t id = 0;
for (const auto &shape : shapes) {
assert(shape.mesh.material_ids.size() ==
shape.mesh.num_face_vertices.size());

size_t triagnleNum = shape.mesh.material_ids.size();
for (size_t face_i = 0; face_i < triagnleNum; face_i++) {
Vec3<float> points[3];
for (size_t point_i = 0; point_i < 3; point_i++) {
int vertex_index =
shape.mesh.indices[face_i * 3 + point_i].vertex_index;

points[point_i].x = attrib.vertices[vertex_index * 3 + 0];
points[point_i].y = attrib.vertices[vertex_index * 3 + 1];
points[point_i].z = attrib.vertices[vertex_index * 3 + 2];
}

Vec2<float> point_textures[3];
for (size_t point_i = 0; point_i < 3; point_i++) {
int texcoord_index =
shape.mesh.indices[face_i * 3 + point_i].texcoord_index;

point_textures[point_i].u = attrib.texcoords[texcoord_index * 2 + 0];
point_textures[point_i].v = attrib.texcoords[texcoord_index * 2 + 1];
}

bool normalValid = true;
Vec3<float> point_normals[3];
for (size_t point_i = 0; point_i < 3; point_i++) {
int normal_index =
shape.mesh.indices[face_i * 3 + point_i].normal_index;

if (normal_index * 3 + 0 < 0 ||
normal_index * 3 + 2 >= attrib.normals.size()) {
normalValid = false;
break;
}

point_normals[point_i].x = attrib.normals[normal_index * 3 + 0];
point_normals[point_i].y = attrib.normals[normal_index * 3 + 1];
point_normals[point_i].z = attrib.normals[normal_index * 3 + 2];
}

Vec3<float> normal =
Vec3<float>::cross(points[1] - points[0], points[2] - points[0]);
if (normalValid && (point_normals[0] == point_normals[1] ||
point_normals[0] == point_normals[2] ||
point_normals[1] == point_normals[2])) {
if (point_normals[0] == point_normals[1] ||
point_normals[0] == point_normals[2]) {
normal = point_normals[0];
} else {
normal = point_normals[1];
}
} else {
if (Vec3<float>::dot(point_normals[0], normal) < 0) {
normal = -normal;
}
}

Material material = actualMaterials[shape.mesh.material_ids[face_i]];
Triangle triangle(id, points[0], points[1], points[2], normal, material);
if (material.isEmissive()) {
light.setLight(triangle);
}
Hittable *obj =
new Triangle(id, points[0], points[1], points[2], point_textures[0],
point_textures[1], point_textures[2], normal, material);
objects.push_back(obj);
id += 1;
}
}

std::vector<Hittable *> objectsCopy(objects);
std::sort(objectsCopy.begin(), objectsCopy.end(), BVHNode::zCmp);
scenes = new BVH(objects, 0, objects.size());
return true;
}

void Tracer::load(const std::string &pathName, const std::string &modelName,
const std::string &configName) {
std::unordered_map<std::string, Vec3<float>> lightRadiances;
std::string config = pathName + configName;
if (!loadConfiguration(config, lightRadiances)) {
std::cout << "Camera config loading fails!" << std::endl;
return;
}

std::string model = pathName + modelName;
if (!loadModel(model, pathName, lightRadiances)) {
std::cout << "Model loading fails!" << std::endl;
return;
}

printStatus();
}

cv::Mat Tracer::render() {
int height = camera.getHeight(), width = camera.getWidth();
cv::Mat img(cv::Size(width, height), CV_32FC3, cv::Scalar(0, 0, 0));

#pragma omp parallel for num_threads(1000)
for (int row = 0; row < height; row++) {
#pragma omp parallel for num_threads(1000)
for (int col = 0; col < width; col++) {
Vec3<float> color(0, 0, 0);
#pragma omp parallel for num_threads(10)
for (int k = 0; k < samples; k++) {
Ray ray = camera.getRay(row, col);
color += trace(ray, 0);
}
color /= samples;
img.at<cv::Vec3f>(row, col)[2] = color.x;
img.at<cv::Vec3f>(row, col)[1] = color.y;
img.at<cv::Vec3f>(row, col)[0] = color.z;
}
}
return img;
}

Vec3<float> Tracer::trace(const Ray &ray, size_t depth) {
assert(scenes != nullptr);
if (depth >= maxDepth) {
return Vec3<float>(0, 0, 0);
}

HitResult res;
scenes->hit(ray, res);
if (!res.isHit) {
return Vec3<float>(0, 0, 0);
}
assert(res.id >= 0 && res.id < objects.size());

float cosine =
std::max(0.0f, Vec3<float>::dot(-ray.getDirection(), res.normal));
float dis = res.distance;
Vec2<float> texCoord = objects[res.id]->getTexCoord(res.hitPoint);

float possibility = randFloat(1);
if (possibility > thresholdP) {
return Vec3<float>(0, 0, 0);
}

Vec3<float> directLight(0, 0, 0), indirectLight(0, 0, 0);
Vec3<float> diffusion = res.material.getDiffusion(texCoord);
Vec3<float> specularity = res.material.getSpecularity(texCoord);
Vec3<float> transmittance = res.material.getTransmittance();

if (res.material.isEmissive()) {
directLight = res.material.getEmission();
return directLight / thresholdP;
} else {
size_t id = -1;
float area = 0;
Vec3<float> lightPoint;
Vec3<float> radiance;
light.getRandomPoint(id, lightPoint, radiance, area);
float pdfLight = 1 / area;

Ray tmpRay(res.hitPoint, lightPoint - res.hitPoint);
HitResult tmpRes;
scenes->hit(tmpRay, tmpRes);
if (tmpRes.isHit && tmpRes.id == id) {
dis = tmpRes.distance;
directLight = radiance * diffusion * cosine / (dis * dis * pdfLight);
}
}

dis = res.distance;
if (res.material.isDiffusive()) {
static float pdf = 1 / (2 * PI);
Ray reflectRay(res.hitPoint, Ray::randomReflect(ray, res.normal));
Vec3<float> reflectLight = trace(reflectRay, depth + 1);
indirectLight += reflectLight * diffusion * cosine / (dis * dis * pdf);
}
if (res.material.isSpecular()) {
Ray reflectRay(res.hitPoint, Ray::standardReflect(ray, res.normal));
float shiness = pow(
std::max(Vec3<float>::dot(reflectRay.getDirection(), res.normal), 0.0f),
res.material.getShiness());
Vec3<float> reflectLight = trace(reflectRay, depth + 1);
indirectLight += reflectLight * specularity * shiness / (dis * dis);
}
if (res.material.isTransmissive()) {
float ior = res.material.getRefraction();
Ray refractRay(res.hitPoint, Ray::standardRefract(ray, res.normal, ior));
static int flag = 0;
if (refractRay.getDirection() != ray.getDirection()) {
Vec3<float> refractLight = trace(refractRay, depth + 1);
if (flag == 0) {
flag = 1;
std::cout << "refract success!" << std::endl;
}
indirectLight += refractLight * transmittance / (dis * dis);
}
}

return (directLight * 0.5 + indirectLight) / thresholdP;
}

void Tracer::printStatus() {
std::cout << "sample number: " << samples << '\n'
<< "tracing depth: " << maxDepth << '\n'
<< "threshod probability: " << thresholdP << '\n';
camera.printStatus();
std::cout << "shapes" << '\n'
<< "triange number: "
<< (scenes == nullptr ? 0 : scenes->getNodeNum()) << '\n';
}
}  
