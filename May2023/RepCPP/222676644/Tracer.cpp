#include "Tracer.h"
#include "omp.h"

using namespace glm;

const float PI = 4 * atan(1.0f);

SRay CTracer::MakeRay(glm::uvec2 pixelPos)
{
float aspectRation = m_camera.m_resolution.x / m_camera.m_resolution.y;
float x = (((pixelPos.x + 0.5f) / m_camera.m_resolution.x) * 2.0f - 1.0f) * aspectRation * tan(m_camera.m_FOV.x / 2 * PI / 180);
float y = (((pixelPos.y + 0.5f) / m_camera.m_resolution.y) * 2.0f - 1.0f) * tan(m_camera.m_FOV.y / 2 * PI / 180);
vec3 dir = m_camera.m_forward + m_camera.m_right * x + m_camera.m_up * y;
return SRay(m_camera.m_pos, normalize(dir));
}

glm::vec3 CTracer::TraceRay(SRay ray)
{
vec3 color(0, 0, 0);

bool hit = false;
float minDistance = INFINITY, distance = 0.0f, transparency = 0.005f, step = 25.0f;
int polygonHit = -1;

for (int i = 0; i < m_scene->flat.m_triangles.size(); i++)
{
hit = m_scene->countIntersection(ray, m_scene->flat.m_triangles[i], &distance);

if (hit && distance < minDistance)
{
minDistance = distance;
polygonHit = i;
}
}

if (polygonHit != -1)
{
if (m_scene->flat.m_normals[m_scene->flat.m_triangles_normals[polygonHit]].z)
color = vec3(90.0f / 255.0f);
if (m_scene->flat.m_normals[m_scene->flat.m_triangles_normals[polygonHit]].y)
color = vec3(140.0f / 255.0f);
if (m_scene->flat.m_normals[m_scene->flat.m_triangles_normals[polygonHit]].x)
color = vec3(180.0f / 255.0f);
}
float length = ((hit) ? minDistance : glm::distance(m_scene->highBound, m_scene->lowBound));
vec3 vecStep = ray.m_dir * -step;
vec3 line = ray.m_start + ray.m_dir * length;
for (; length > 0; length -= step, line += vecStep)
{
if (line.x > m_scene->lowBound.x && line.y > m_scene->lowBound.y && line.z > m_scene->lowBound.z &&
line.x < m_scene->highBound.x && line.y < m_scene->highBound.y && line.z < m_scene->highBound.z)
break;
}
int old_x = static_cast<int>(abs(m_scene->lowBound.x - line.x) / m_scene->resizeRatio.x);
int old_y = static_cast<int>(abs(m_scene->lowBound.y - line.y) / m_scene->resizeRatio.y);
int old_z = static_cast<int>(abs(m_scene->lowBound.z - line.z) / m_scene->resizeRatio.z);
color = transparency * pickColor(m_scene->voxel[old_x][old_y][old_z]) + (1.0f - transparency) * color;
for (; length > 0; length -= step, line += vecStep)
{
if (line.x < m_scene->lowBound.x || line.y < m_scene->lowBound.y || line.z < m_scene->lowBound.z ||
line.x > m_scene->highBound.x || line.y > m_scene->highBound.y || line.z > m_scene->highBound.z)
break;
int x = static_cast<int>(abs(m_scene->lowBound.x - line.x) / m_scene->resizeRatio.x);
int y = static_cast<int>(abs(m_scene->lowBound.y - line.y) / m_scene->resizeRatio.y);
int z = static_cast<int>(abs(m_scene->lowBound.z - line.z) / m_scene->resizeRatio.z);
if (old_x != x || old_y != y || old_z != z)
{
old_x = x;
old_y = y;
old_z = z;
color = transparency * pickColor(m_scene->voxel[old_x][old_y][old_z]) + (1.0f - transparency) * color;
}
}
return color;
}

glm::vec3 CTracer::pickColor(float power)
{
float maxPower = 100.0f;
int newColor = static_cast<int>(((power > maxPower) ? maxPower : power) / maxPower * 510);
if (newColor <= 255)
return vec3(0.0f, newColor / 255.0f, 1.0f);
else
return vec3(1.0f, (510.0f - newColor) / 255.0f, 0.0f);
}

void CTracer::RenderImage(int xRes, int yRes, bool openMP)
{
m_camera.m_resolution = uvec2(xRes, yRes);
m_camera.m_pixels.resize(xRes * yRes);

if (!openMP)
{
for (int i = 0; i < yRes; i++)
for (int j = 0; j < xRes; j++)
{
SRay ray = MakeRay(uvec2(j, i));
m_camera.m_pixels[i * xRes + j] = TraceRay(ray);
}
}
else
{
for (int i = 0; i < yRes; i++)
#pragma omp parallel for shared(i)
for (int j = 0; j < xRes; j++)
{
SRay ray = MakeRay(uvec2(j, i));
m_camera.m_pixels[i * xRes + j] = TraceRay(ray);
}
}
}

void CTracer::SaveImageToFile(std::string fileName)
{
CImage image;

int width = m_camera.m_resolution.x;
int height = m_camera.m_resolution.y;

image.Create(width, height, 24);

int pitch = image.GetPitch();
unsigned char* imageBuffer = (unsigned char*)image.GetBits();

if (pitch < 0)
{
imageBuffer += pitch * (height - 1);
pitch = -pitch;
}

int i, j;
int imageDisplacement = 0;
int textureDisplacement = 0;

for (i = 0; i < height; i++)
{
for (j = 0; j < width; j++)
{
vec3 color = m_camera.m_pixels[textureDisplacement + j];

imageBuffer[imageDisplacement + j * 3] = clamp(color.b, 0.0f, 1.0f) * 255.0f;
imageBuffer[imageDisplacement + j * 3 + 1] = clamp(color.g, 0.0f, 1.0f) * 255.0f;
imageBuffer[imageDisplacement + j * 3 + 2] = clamp(color.r, 0.0f, 1.0f) * 255.0f;
}

imageDisplacement += pitch;
textureDisplacement += width;
}

image.Save(fileName.c_str());
image.Destroy();
}

CImage* CTracer::LoadImageFromFile(std::string fileName)
{
CImage* pImage = new CImage;

if(SUCCEEDED(pImage->Load(fileName.c_str())))
return pImage;
else
{
delete pImage;
return NULL;
}
}