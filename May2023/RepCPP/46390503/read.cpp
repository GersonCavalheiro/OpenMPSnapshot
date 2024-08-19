

#include "read.h"
#include <assimp/Importer.hpp>      
#include <assimp/Exporter.hpp>      
#include <assimp/scene.h>           
#include <assimp/postprocess.h>     

#include "../../geometry/structure/Mesh.h"

std::vector<std::string> splitString(
std::string input,
std::string delimiter)
{
std::vector<std::string> output;
char *str = strdup(input.c_str());
char *pch = strtok (str, delimiter.c_str());

while (pch != NULL)
{
output.push_back(pch);
pch = strtok (NULL,  delimiter.c_str());
}

free(str);

return output;
}

void delta::core::io::parseModelGridSchematics(
std::string fileName,
std::vector<std::vector<std::string>> &componentGrid,
std::vector<std::string> &componentSeq)
{
std::string line;
std::ifstream myfile;
myfile.open(fileName.c_str());

if (myfile.is_open())
{
while (std::getline (myfile, line))
{
std::vector<std::string> vstring = splitString(line, ",");

componentSeq.push_back(vstring[2]);

if(std::stoi(vstring[0]) == 46)
{
componentGrid.push_back(componentSeq);
}
}
myfile.close();
}
else std::cout << "Unable to open file";
}

delta::geometry::mesh::Mesh *delta::core::io::readVTKGeometry(
char* fileName)
{
std::vector<iREAL> xCoordinates;
std::vector<iREAL> yCoordinates;
std::vector<iREAL> zCoordinates;


char filename[100];
strncpy(filename, fileName, 100);
FILE *fp1 = fopen(filename, "r+");

if( fp1 == NULL )
{
perror("Error while opening the file.\n");
exit(EXIT_FAILURE);
}

char ch, word[100];
iREAL *point[3];

do
{
ch = fscanf(fp1,"%s",word);
if(strcmp(word, "POINTS")==0)
{
ch = fscanf(fp1,"%s",word);
int n = atol(word);

point[0] = new iREAL[n];
point[1] = new iREAL[n];
point[2] = new iREAL[n];

ch = fscanf(fp1,"%s",word);

for(int i=0;i<n;i++)
{
fscanf(fp1, "%lf", &point[0][i]);
fscanf(fp1, "%lf", &point[1][i]);
fscanf(fp1, "%lf", &point[2][i]);
}
}

if(strcmp(word, "CELLS")==0 || strcmp(word, "POLYGONS") == 0)
{
ch = fscanf(fp1,"%s",word);
int numberOfTriangles = atol(word);
ch = fscanf(fp1,"%s",word);

xCoordinates.resize( numberOfTriangles*3 );
yCoordinates.resize( numberOfTriangles*3 );
zCoordinates.resize( numberOfTriangles*3 );

for(int i=0;i<numberOfTriangles*3;i+=3)
{
ch = fscanf(fp1,"%s",word);
ch = fscanf(fp1,"%s",word);

int index = atol(word);
xCoordinates[i] = ((point[0][index]));
yCoordinates[i] = ((point[1][index]));
zCoordinates[i] = ((point[2][index]));

ch = fscanf(fp1,"%s",word);
index = atol(word);
xCoordinates[i+1] = ((point[0][index]));
yCoordinates[i+1] = ((point[1][index]));
zCoordinates[i+1] = ((point[2][index]));

ch = fscanf(fp1,"%s",word);
index = atol(word);
xCoordinates[i+2] = ((point[0][index]));
yCoordinates[i+2] = ((point[1][index]));
zCoordinates[i+2] = ((point[2][index]));
}
}
} while (ch != EOF);

fclose(fp1);

return new delta::geometry::mesh::Mesh(xCoordinates, yCoordinates, zCoordinates);
}

void delta::core::io::readScenarioSpecification(std::string fileName)
{

}

std::vector<delta::geometry::mesh::Mesh> delta::core::io::readGeometry(std::string fileName)
{
Assimp::Importer importer;

const aiScene* scene = importer.ReadFile( fileName,
aiProcess_CalcTangentSpace       |
aiProcess_Triangulate            |
aiProcess_JoinIdenticalVertices  |
aiProcess_SortByPType);

bool n = scene->HasMeshes();
int nn = scene->mNumMeshes;

printf("Importing %i Meshes.\n", nn);

std::vector<delta::geometry::mesh::Mesh> meshVector;

for(uint m_i = 0; m_i < scene->mNumMeshes; m_i++)
{
std::vector<std::array<int, 3>> 		triangleFaces;
std::vector<std::array<iREAL, 3>> 	uniqueVertices;

const aiMesh* mesh = scene->mMeshes[m_i];

std::vector<iREAL> g_vp;
g_vp.reserve(3 * mesh->mNumVertices);


#pragma omp parallel for
for(uint v_i = 0; v_i < mesh->mNumVertices; v_i++)
{
if(mesh->HasPositions())
{
const aiVector3D* vp = &(mesh->mVertices[v_i]);
g_vp.push_back(vp->x);
g_vp.push_back(vp->y);
g_vp.push_back(vp->z);

std::array<iREAL, 3> vertex = {vp->x, vp->y, vp->z};

#pragma omp critical
uniqueVertices.push_back(vertex);
}
}


for(uint f_i = 0; f_i < mesh->mNumFaces; f_i++)
{
for(uint index = 0; index < mesh->mFaces[f_i].mNumIndices; index+=3)
{
int idxA = mesh->mFaces[f_i].mIndices[index];
int idxB = mesh->mFaces[f_i].mIndices[index+1];
int idxC = mesh->mFaces[f_i].mIndices[index+2];
std::array<int, 3> triangle = {idxA, idxB, idxC};



triangleFaces.push_back(triangle);
}
}

for(int i=0; i<triangleFaces.size(); i++)
{
std::cout << triangleFaces[i][0] << " " << triangleFaces[i][1] << " " << triangleFaces[i][2] << std::endl;
}
delta::geometry::mesh::Mesh *meshgeometry = new delta::geometry::mesh::Mesh(triangleFaces, uniqueVertices);
meshVector.push_back(*meshgeometry);
}

return meshVector;
}

delta::geometry::mesh::Mesh* delta::core::io::readPartGeometry(std::string fileName)
{
Assimp::Importer importer;

const aiScene* scene = importer.ReadFile( fileName,
aiProcess_CalcTangentSpace       |
aiProcess_Triangulate            |
aiProcess_JoinIdenticalVertices  |
aiProcess_SortByPType);

bool n = scene->HasMeshes();
int nn = scene->mNumMeshes;


std::vector<std::array<int, 3>> 	triangleFaces;
std::vector<std::array<iREAL, 3>> 	uniqueVertices;

const aiMesh* mesh = scene->mMeshes[0];

std::vector<iREAL> g_vp;
g_vp.reserve(3 * mesh->mNumVertices);


for(uint v_i = 0; v_i < mesh->mNumVertices; v_i++)
{
if(mesh->HasPositions())
{
const aiVector3D* vp = &(mesh->mVertices[v_i]);
g_vp.push_back(vp->x);
g_vp.push_back(vp->y);
g_vp.push_back(vp->z);

std::array<iREAL, 3> vertex = {vp->x, vp->y, vp->z};

uniqueVertices.push_back(vertex);
}
}


for(uint f_i = 0; f_i < mesh->mNumFaces; f_i++)
{
for(uint index = 0; index < mesh->mFaces[f_i].mNumIndices; index+=3)
{
int idxA = mesh->mFaces[f_i].mIndices[index];
int idxB = mesh->mFaces[f_i].mIndices[index+1];
int idxC = mesh->mFaces[f_i].mIndices[index+2];
std::array<int, 3> triangle = {idxA, idxB, idxC};

triangleFaces.push_back(triangle);
}
}
return new delta::geometry::mesh::Mesh(triangleFaces, uniqueVertices);
}

void delta::core::io::readVTKLegacy() {

vtkSmartPointer<vtkUnstructuredGridReader> reader =
vtkSmartPointer<vtkUnstructuredGridReader>::New();
reader->SetFileName("../output/grid_0.vtk");
reader->Update();

if(reader->IsFileUnstructuredGrid())
{
std::cout << "output is a unstructured grid" << std::endl;
vtkUnstructuredGrid* output = reader->GetOutput();
std::cout << "output has " << output->GetNumberOfPoints() << " points." << std::endl;
}
}



void delta::core::io::readmbfcp(std::string 									filename,
std::vector<delta::world::structure::Object>& 	objects,
iREAL 											epsilon) {
std::ifstream file(filename);
if (file.is_open()) {
std::cout << "Opened: " << filename << std::endl;
}

int idCounter = 0;
int bodies = 0;

for (std::string line; getline(file, line);)
{
if (line.size() == 0)
{
std::cout << std::endl;
continue;
}


if (line.find("SURFACE_MATERIALS:") != std::string::npos) {
int nmaterials = std::stoi(line.substr(18));
getline(file, line);
std::string surf1 = line.substr(7);
getline(file, line);
std::string surf2 = line.substr(7);
getline(file, line);
std::string model = line.substr(7);
getline(file, line);
iREAL friction = std::stof(line.substr(10));
getline(file, line);
iREAL cohesion = std::stoi(line.substr(10));
getline(file, line);
iREAL spring = std::stof(line.substr(8));
getline(file, line);
iREAL dashpot = std::stof(line.substr(8));


getline(file, line);
getline(file, line);
int bulk_materials = std::stoi(line.substr(15));
getline(file, line);
std::string bulk_materials_label = line.substr(7);
getline(file, line);
std::string mat_model = line.substr(7);
getline(file, line);
iREAL young = std::stof(line.substr(7));
getline(file, line);
iREAL poisson = std::stof(line.substr(9));
getline(file, line);
iREAL density = std::stof(line.substr(9));
getline(file, line);
getline(file, line);
getline(file, line);
iREAL gravity[3];
gravity[0] = std::stof(line.substr(10));
getline(file, line);
gravity[1] = std::stof(line.substr(10));
getline(file, line);
gravity[2] = std::stof(line.substr(10));
}

if (line.find("BODIES:") != std::string::npos) {
bodies = std::stoi(line.substr(8));
std::cout << "# bodies: " << bodies << std::endl;
}





if (line.find("ID:") != std::string::npos && line.find("SURFID:") != 0) {


int id = std::stoi(line.substr(3));
getline(file, line);

std::string label = line.substr(6);
label.erase(std::remove(label.begin(), label.end(), ' '), label.end());
label.erase(std::remove(label.begin(), label.end(), '\t'), label.end());

getline(file, line);
std::string kinematics = line.substr(11);
kinematics.erase(std::remove(kinematics.begin(), kinematics.end(), ' '), kinematics.end());
kinematics.erase(std::remove(kinematics.begin(), kinematics.end(), '\t'), kinematics.end());

getline(file, line);
std::string bulk_material = line.substr(14);
bulk_material.erase(std::remove(bulk_material.begin(), bulk_material.end(), ' '), bulk_material.end());
bulk_material.erase(std::remove(bulk_material.begin(), bulk_material.end(), '\t'), bulk_material.end());

getline(file, line);
int shapes = std::stoi(line.substr(7));

getline(file, line);

if (line.find("SPHERES:") != std::string::npos) {
getline(file, line);
std::cout << line << std::endl;

iREAL x = std::stod(line.substr(8, 4));
iREAL y = std::stod(line.substr(14, 4));
iREAL z = std::stod(line.substr(20, 4));

std::cout << x << "|" << y << "|" << z <<"|"<< std::endl;

getline(file, line);
std::string radius = line.substr(8, 4);
iREAL rad = std::stod(radius);


std::array<iREAL, 3> centreArray = {x, y, z};
std::array<iREAL, 3> linear = {0.0, 0.0, 0.0};

delta::world::structure::Object object("sphere", rad, id, centreArray, delta::geometry::material::MaterialType::WOOD, false, false, true, epsilon, linear, {0,0,0});
objects.push_back(object);


} else if(line.find("VERTEXES:") != std::string::npos) {

}



}

std::cout << line << std::endl;
}
}
