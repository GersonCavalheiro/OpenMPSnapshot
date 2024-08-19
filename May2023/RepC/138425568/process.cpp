#include <iostream>
#include <sstream>
#include <math.h>
#include <omp.h>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>

#define gaussianTotal 159
#define upperThreshold 60
#define lowerThreshold 30

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

int width, height, bpp;
int avgSize=64;
int imgWidth, imgHeight, spillWidth, spillHeight;
unsigned char* image;
unsigned char* rgbImage;
unsigned char* avgImage;
unsigned char* rgbTable;
int edgeDir[64*64];
float gradient[64*64];
int GxMask[3][3];		
int GyMask[3][3];		
int gaussianMask[5][5];		
unsigned long i;		

int Gx;				
int Gy;				
float thisAngle;		
int newAngle;			
int newPixel;
unsigned long iOffset;

string getName();
int readFile(string name);
void declareMasks();
void findEdge(int rowShift, int colShift, int row, int col, int dir);
void suppressNonMax(int rowShift, int colShift, int row, int col, int dir);
void readDirectory(const string& name, vector<string>& v);

int main(){
vector<string> v;
readDirectory("./img", v);
cout<<(v.size()-2)<<" files found."<<endl;

cout<<"Loading data table...";
rgbTable= new unsigned char[256*256*256];
ifstream finput("rgbTable",ios::binary);
for(unsigned long x= 0; x<256*256*256; ++x)
finput.read(reinterpret_cast<char*> (&rgbTable[x]), sizeof(unsigned char));
finput.close();
cout<<"done."<<endl;


time_t  timeNow=time(0);
cout<<"Starting..."<<endl;
declareMasks();

for(int file=2; file<v.size(); ++file){
cout<<"Loading "<<v[file]<<" : "<<file-1<<"/"<<v.size()-2<<" images...";
readFile("img/"+v[file]);
if(width<64 || height<64){
cout<<"Image is too small."<<endl;
delete rgbImage;
remove(("img/"+v[file]).c_str());
continue;
}
cout<<"done."<<endl;

cout<<"Resizing & getting grayscale...";
int temp = width;
if (width>height)
temp=height;

avgSize = 64;
imgWidth = imgHeight = avgSize;
avgSize = temp/avgSize;

avgImage = new unsigned char[(imgWidth)*(imgHeight)*3];
image = new unsigned char[imgWidth*imgHeight];
int r,g,b;
unsigned long imgI;

#pragma omp parallel for schedule(static) shared(avgImage,imgHeight,imgWidth,rgbImage,avgSize,image) private(r,g,b,imgI,i)
for(int imgRow = 0; imgRow<imgHeight; ++imgRow){
for(int imgCol = 0; imgCol<imgWidth; ++imgCol){
r=0, g=0, b=0;
imgI=(imgRow*imgWidth*3)+3*imgCol;
for(int row = 0; row<avgSize; ++row){
for(int col = 0; col<avgSize; ++col){
i = (unsigned long) (imgRow*avgSize+row)*width*3+3*(avgSize*imgCol+col);
r+=rgbImage[i];
g+=rgbImage[i+1];
b+=rgbImage[i+2];
}
}
avgImage[imgI]=r/avgSize/avgSize;
avgImage[imgI+1]=g/avgSize/avgSize;
avgImage[imgI+2]=b/avgSize/avgSize;

image[imgI/3]=(avgImage[imgI]*30+avgImage[imgI+1]*59+avgImage[imgI+2]*11)/100;
}
}
cout<<"done."<<endl;

cout<<"Calculating average RGB....";
int totalR=0,totalG=0,totalB=0;
#pragma omp parallel for schedule(static) shared(avgImage,totalR,totalG,totalB) private(imgI)
for(int imgRow = 0; imgRow<imgHeight; ++imgRow){
for(int imgCol = 0; imgCol<imgWidth; ++imgCol){
imgI=(imgRow*imgWidth*3)+3*imgCol;
#pragma omp critical
{
totalR+=avgImage[imgI];
totalG+=avgImage[imgI+1];
totalB+=avgImage[imgI+2];
}
}
}
totalR= totalR/imgHeight/imgWidth;
totalG= totalG/imgHeight/imgWidth;
totalB= totalB/imgHeight/imgWidth;
cout<<"done."<<endl;

cout<<"Writing resized image...";
int rgbVal= totalR*256*256+totalG*256+totalB;
int count = rgbTable[rgbVal];
rgbTable[rgbVal]++;

stringstream strs;
strs << rgbVal <<"-"<<count<<".jpg";
string name = strs.str();

stbi_write_jpg(("data/rgb/"+name).c_str(), imgWidth, imgHeight, 3, avgImage, 100);
delete rgbImage;
delete avgImage;
height = width = 64;
cout<<"done."<<endl;

cout<<"Applying Gaussian mask...";
#pragma omp parallel for schedule(static) shared(gaussianMask,image,height,width) private(newPixel,iOffset)
for(unsigned int row = 2; row<height-2; ++row){
for(unsigned int col = 2; col<width-2; ++col){
newPixel=0;
for (int rowOffset=-2; rowOffset<=2; rowOffset++) {
for (int colOffset=-2; colOffset<=2; colOffset++) {
iOffset = (unsigned long) ((row+rowOffset)*width+ col + colOffset);
newPixel+= image[iOffset] * gaussianMask[2+rowOffset][2+colOffset];
}
}
image[row*width+col]=newPixel/gaussianTotal;
}
}

cout<<"Applying Sobel mask...";
#pragma omp parallel for schedule(static) shared(height,width,gradient,edgeDir) private(i,Gx,Gy,thisAngle,newAngle,iOffset)
for (unsigned int row = 1; row < height-1; ++row) {
for (unsigned int col = 1; col < width-1; ++col) {
i = (unsigned long)(row*width + col);
Gx = 0;
Gy = 0;

for (int rowOffset=-1; rowOffset<=1; rowOffset++) {
for (int colOffset=-1; colOffset<=1; colOffset++) {
iOffset = (unsigned long)((row+rowOffset)*width + col+colOffset);
Gx = Gx + (image[iOffset] * GxMask[rowOffset + 1][colOffset + 1]);
Gy = Gy + (image[iOffset] * GyMask[rowOffset + 1][colOffset + 1]);
}
}

gradient[i] = sqrt(pow(Gx,2.0) + pow(Gy,2.0));	
thisAngle = (atan2(Gx,Gy)/3.14159) * 180.0;		


if ( ( (thisAngle < 22.5) && (thisAngle > -22.5) ) || (thisAngle > 157.5) || (thisAngle < -157.5) )
newAngle = 0;
if ( ( (thisAngle > 22.5) && (thisAngle < 67.5) ) || ( (thisAngle < -112.5) && (thisAngle > -157.5) ) )
newAngle = 45;
if ( ( (thisAngle > 67.5) && (thisAngle < 112.5) ) || ( (thisAngle < -67.5) && (thisAngle > -112.5) ) )
newAngle = 90;
if ( ( (thisAngle > 112.5) && (thisAngle < 157.5) ) || ( (thisAngle < -22.5) && (thisAngle > -67.5) ) )
newAngle = 135;

edgeDir[i] = newAngle;		
}
}

cout<<"Finding edges...";
#pragma omp parallel for schedule(static) shared(height,width,edgeDir,gradient,image) private(i)
for (unsigned int row = 1; row < height - 1; ++row) {
for (unsigned int col = 1; col < width - 1; ++col) {
i = (unsigned long)(row*width + col);
if (gradient[i] > upperThreshold) {

switch (edgeDir[i]){
case 0:
findEdge(0, 1, row, col, 0);
break;
case 45:
findEdge(1, 1, row, col, 45);
break;
case 90:
findEdge(1, 0, row, col, 90);
break;
case 135:
findEdge(1, -1, row, col, 135);
break;
default :
image[i]=0;
break;
}
}
else
image[i]=0;
}
}

cout<<"Cleaning up edges...";
#pragma omp parallel for schedule(static) shared(image,height,width)
for (i=0; i<height*width;++i){
if( image[i] != 255 && image[i] != 0 )
image[i] = 0;
}

cout<<"Applying NonMax supression...";
#pragma omp parallel for schedule(static) shared(width,height,image,edgeDir,gradient) private(i)
for (unsigned int row = 1; row < height - 1; ++row) {
for (unsigned long col = 1; col < width - 1; ++col) {
i = (unsigned long)(row*width + col);
if (image[i] == 255) {		

switch (edgeDir[i]) {
case 0:
suppressNonMax( 1, 0, row, col, 0);
break;
case 45:
suppressNonMax( 1, -1, row, col, 45);
break;
case 90:
suppressNonMax( 0, 1, row, col, 90);
break;
case 135:
suppressNonMax( 1, 1, row, col, 135);
break;
default :
break;
}
}
}
}
cout<<"done."<<endl;

cout<<"Writing image...";
stbi_write_jpg(("data/b/"+name).c_str(), width, height, 1, image, 100);
cout<<"done."<<endl;
delete image;

cout<<"Removing original image...";
remove(("img/"+v[file]).c_str());
cout<<"done."<<endl;
}

cout<<"Updating data table...";
ofstream foutput("rgbTable",ios::binary);
for(unsigned long x= 0; x<256*256*256; ++x)
foutput.write(reinterpret_cast<char*> (&rgbTable[x]), sizeof(unsigned char));
foutput.close();
cout<<"done."<<endl;
delete rgbTable;

timeNow= time(0)-timeNow;
cout<<"Processing End: "<<timeNow<<endl;
return 0;
}
void suppressNonMax(int rowShift, int colShift, int row, int col, int dir){
int newRow = 0;
int newCol = 0;
bool edgeEnd = false;
float* nonMax = new float[width*3];		
int pixelCount = 0;		
int count;
int max[3];			

if (colShift < 0) {
if (col > 0)
newCol = col + colShift;
else
edgeEnd = true;
} else if (col < width - 1) {
newCol = col + colShift;
} else
edgeEnd = true;		
if (rowShift < 0) {
if (row > 0)
newRow = row + rowShift;
else
edgeEnd = true;
} else if (row < height - 1) {
newRow = row + rowShift;
} else
edgeEnd = true;
i = (unsigned long)(newRow*width + newCol);

while (edgeDir[i] == dir && !edgeEnd && image[i] == 255) {
if (colShift < 0) {
if (newCol > 0)
newCol = newCol + colShift;
else
edgeEnd = true;
} else if (newCol < width - 1) {
newCol = newCol + colShift;
} else
edgeEnd = true;
if (rowShift < 0) {
if (newRow > 0)
newRow = newRow + rowShift;
else
edgeEnd = true;
} else if (newRow < height - 1) {
newRow = newRow + rowShift;
} else
edgeEnd = true;
nonMax[pixelCount] = newRow;
nonMax[pixelCount+ 1] = newCol;
nonMax[pixelCount+ 2] = gradient[newRow*width+newCol];
pixelCount+=3;
i = (unsigned long)(newRow*width + newCol);
}


edgeEnd = false;
colShift *= -1;
rowShift *= -1;
if (colShift < 0) {
if (col > 0)
newCol = col + colShift;
else
edgeEnd = true;
} else if (col < width - 1) {
newCol = col + colShift;
} else
edgeEnd = true;
if (rowShift < 0) {
if (row > 0)
newRow = row + rowShift;
else
edgeEnd = true;
} else if (row < height - 1) {
newRow = row + rowShift;
} else
edgeEnd = true;
i = (unsigned long)(newRow*width + newCol);
while ((edgeDir[newRow*width + newCol] == dir) && !edgeEnd && (image[i] == 255)) {
if (colShift < 0) {
if (newCol > 0)
newCol = newCol + colShift;
else
edgeEnd = true;
} else if (newCol < width - 1) {
newCol = newCol + colShift;
} else
edgeEnd = true;
if (rowShift < 0) {
if (newRow > 0)
newRow = newRow + rowShift;
else
edgeEnd = true;
} else if (newRow < height - 1) {
newRow = newRow + rowShift;
} else
edgeEnd = true;
nonMax[pixelCount] = newRow;
nonMax[pixelCount+1] = newCol;
nonMax[pixelCount+2] = gradient[newRow*width+newCol];
pixelCount+=3;
i = (unsigned long)(newRow*width + newCol);
}


max[0] = 0;
max[1] = 0;
max[2] = 0;
for (count = 0; count < pixelCount; count+=3) {
if (nonMax[count+2] > max[2]) {
max[0] = nonMax[count];
max[1] = nonMax[count+1];
max[2] = nonMax[count+2];
}
}
for (count = 0; count < pixelCount; count+=3) {
i = (unsigned long)(nonMax[count]*width + nonMax[count+1]);
image[i] = 0;
}
delete nonMax;
}
void findEdge(int rowShift, int colShift, int row, int col, int dir){
int newRow;
int newCol;
bool edgeEnd = false;


if (colShift < 0) {
if (col > 0)
newCol = col + colShift;
else
edgeEnd = true;
} else if (col < width - 1) {
newCol = col + colShift;
} else
edgeEnd = true;		
if (rowShift < 0) {
if (row > 0)
newRow = row + rowShift;
else
edgeEnd = true;
} else if (row < height - 1) {
newRow = row + rowShift;
} else
edgeEnd = true;


i = (unsigned long)(newRow*width + newCol);
while ( (edgeDir[i]==dir) && !edgeEnd && (gradient[i] > lowerThreshold) ) {

i = (unsigned long)(newRow*width + newCol);
image[i] = 255;
if (colShift < 0) {
if (newCol > 0)
newCol = newCol + colShift;
else
edgeEnd = true;
} else if (newCol < width - 1) {
newCol = newCol + colShift;
} else
edgeEnd = true;
if (rowShift < 0) {
if (newRow > 0)
newRow = newRow + rowShift;
else
edgeEnd = true;
} else if (newRow < height - 1) {
newRow = newRow + rowShift;
} else
edgeEnd = true;
}
}
void declareMasks(){

GxMask[0][0] = -1; GxMask[0][1] = 0; GxMask[0][2] = 1;
GxMask[1][0] = -2; GxMask[1][1] = 0; GxMask[1][2] = 2;
GxMask[2][0] = -1; GxMask[2][1] = 0; GxMask[2][2] = 1;

GyMask[0][0] =  1; GyMask[0][1] =  2; GyMask[0][2] =  1;
GyMask[1][0] =  0; GyMask[1][1] =  0; GyMask[1][2] =  0;
GyMask[2][0] = -1; GyMask[2][1] = -2; GyMask[2][2] = -1;


gaussianMask[0][0] = 2;	 gaussianMask[0][1] = 4;  gaussianMask[0][2] = 5;  gaussianMask[0][3] = 4;  gaussianMask[0][4] = 2;
gaussianMask[1][0] = 4;	 gaussianMask[1][1] = 9;  gaussianMask[1][2] = 12; gaussianMask[1][3] = 9;  gaussianMask[1][4] = 4;
gaussianMask[2][0] = 5;	 gaussianMask[2][1] = 12; gaussianMask[2][2] = 15; gaussianMask[2][3] = 12; gaussianMask[2][4] = 2;
gaussianMask[3][0] = 4;	 gaussianMask[3][1] = 9;  gaussianMask[3][2] = 12; gaussianMask[3][3] = 9;  gaussianMask[3][4] = 4;
gaussianMask[4][0] = 2;	 gaussianMask[4][1] = 4;  gaussianMask[4][2] = 5;  gaussianMask[4][3] = 4;  gaussianMask[4][4] = 2;
}
string getName(){
string temp;
cout << "Enter the file name> ";
getline(cin, temp);
return temp+".jpg";
}
int readFile(string name){
rgbImage = stbi_load(name.c_str(),&width, &height, &bpp,3);
if(rgbImage==NULL){
cout<<"Load unsuccessful."<<endl;
return 1;
}
else{
cout<<"Load successful."<<endl;
for(unsigned long i = 0; i<64*64; ++i)
edgeDir[i]=0;
return 0;
}
}
void readDirectory(const string& name, vector<string>& v){
DIR* dirp = opendir(name.c_str());
struct dirent * dp;
while ((dp = readdir(dirp)) != NULL) {
v.push_back(dp->d_name);
}
closedir(dirp);
}
