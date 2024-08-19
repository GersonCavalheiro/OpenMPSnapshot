#include <iostream>
#include <math.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include <omp.h>

#define gaussianTotal 159
#define upperThreshold 60
#define lowerThreshold 30
#define avgSize 64

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

int width, height, bpp;
int imgWidth, imgHeight, spillWidth, spillHeight;
int r,g,b;
unsigned long imgI;

unsigned char* image;
unsigned char* avgImage;
unsigned char* rgbImage;
unsigned char* rgbTable;
unsigned char* finalImage;
int* edgeDir;
float* gradient;

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

int main(){
time_t  timeNow=time(0);

cout<<"Program Starting..."<<endl;
rgbTable= new unsigned char[256*256*256];
ifstream finput("rgbTable",ios::binary);
for(unsigned long x= 0; x<256*256*256; ++x)
finput.read(reinterpret_cast<char*> (&rgbTable[x]), sizeof(unsigned char));
finput.close();
timeNow=time(0)-timeNow;
cout<<"Read datatable: "<<timeNow<<endl;

string name=getName();
timeNow=time(0);
if(readFile(name)==1)
return 1;
declareMasks();
timeNow=time(0)-timeNow;
cout<<"Image Loaded: "<<timeNow<<endl;


timeNow=time(0);
imgWidth = width/avgSize;
imgHeight = height/avgSize;
spillWidth = width - imgWidth*avgSize;
spillHeight = height - imgHeight*avgSize;
height= imgHeight*64;
width=imgWidth*64;

avgImage = new unsigned char[(imgWidth)*(imgHeight)*3];
image = new unsigned char[height*width];

#pragma omp parallel for schedule(static) shared(avgImage,imgHeight,imgWidth,rgbImage) private(r,g,b,imgI,i)
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
image[i/3]=(rgbImage[i]*30+rgbImage[i+1]*59+rgbImage[i+2]*11)/100;
}
}
avgImage[imgI]=r/avgSize/avgSize;
avgImage[imgI+1]=g/avgSize/avgSize;
avgImage[imgI+2]=b/avgSize/avgSize;
}
}
timeNow=time(0)-timeNow;
cout<<"Average RGB Caclulated: "<<timeNow<<endl;

timeNow=time(0);
delete rgbImage;
edgeDir=new int[width*height];
gradient= new float[width*height];
for(i = 0; i<height*width; ++i)
edgeDir[i]=0;

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
timeNow=time(0)-timeNow;
cout<<"Gaussian Mask Applied: "<<timeNow<<endl;

timeNow=time(0);
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
timeNow=time(0)-timeNow;
cout<<"Sobel Mask Applied: "<<timeNow<<endl;

timeNow=time(0);
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
timeNow=time(0)-timeNow;
cout<<"Edges Calculated: "<<timeNow<<endl;

timeNow=time(0);
#pragma omp parallel for schedule(static) shared(image,height,width)
for (i=0; i<height*width;++i){
if( image[i] != 255 && image[i] != 0 )
image[i] = 0;
}
timeNow=time(0)-timeNow;
cout<<"Edges Cleaned Up: "<<timeNow<<endl;

timeNow=time(0);
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
timeNow=time(0)-timeNow;
cout<<"NonMax Suppressed: "<<timeNow<<endl;

timeNow=time(0);
int x, newX;
int val;
finalImage = new unsigned char[height*width*3];
int pHeight, pWidth, pBpp, pI;

#pragma omp parallel for schedule(static) shared(finalImage,rgbTable) private(x,newX,imgI,r,g,b,val,pHeight,pWidth,pBpp,pI,i)
for(int imgRow=0; imgRow<imgHeight; ++imgRow){
for(int imgCol=0; imgCol<imgWidth; ++imgCol){
imgI=(imgRow*imgWidth*3)+3*imgCol;

r=avgImage[imgI];
g=avgImage[imgI+1];
b=avgImage[imgI+2];
x=r*256*256+g*256+b;
val = rgbTable[x];
newX=x;

if(val==0){
int offset = 16;
int distance = 100000000;
int tempX;
for(int roffset=1;roffset<offset;++roffset){
for(int goffset=1;goffset<offset;++goffset){
for(int boffset=1;boffset<offset;++boffset){
if(r+roffset<256 && g+goffset<256 && boffset<256){
tempX=x+roffset*256*256+goffset*256+boffset+1;
val = rgbTable[tempX];
int tempDis=2*roffset*roffset+4*goffset*goffset+3*boffset*boffset;
if(val>0 && tempDis<distance){
distance=tempDis;
newX=tempX;
}
}
if(r-roffset>-1&&g-goffset>-1&&b-boffset>-1){
tempX=x-roffset*256*256-goffset*256-boffset+1;
val = rgbTable[tempX];
int tempDis=2*roffset*roffset+4*goffset*goffset+3*boffset*boffset;
if(val>0 && tempDis<distance){
distance=tempDis;
newX=tempX;
}
}
}
}
}
val=rgbTable[newX];
}

if(val==0){
for(int row=0; row<avgSize; ++row){
for (int col=0; col<avgSize; ++col){
i = (unsigned long) (imgRow*avgSize+row)*width*3+3*(avgSize*imgCol+col);
finalImage[i]=avgImage[i];
finalImage[i+1]=avgImage[i+1];
finalImage[i+2]=avgImage[i+2];
}
}
}
else if (val==1){
stringstream strs;
strs <<"data/rgb/"<<newX<<"-0.jpg";
string name = strs.str();
unsigned char* newImage = stbi_load(name.c_str(),&pHeight,&pWidth,&pBpp,3);

for(int row=0; row<avgSize; ++row){
for (int col=0; col<avgSize; ++col){
i = (unsigned long) (imgRow*avgSize+row)*width*3+3*(avgSize*imgCol+col);
pI=row*avgSize*3+col*3;
finalImage[i]=newImage[pI];
finalImage[i+1]=newImage[pI+1];
finalImage[i+2]=newImage[pI+2];
}
}

delete newImage;
}
else{
int lowestCount=64*64;
int lowestFile=0;
for(int fileNumber=0; fileNumber<val; ++fileNumber){
int diffCount=0;
stringstream strs;
strs <<"data/b/"<<newX<<"-"<<fileNumber<<".jpg";
string name = strs.str();
unsigned char* newImage = stbi_load(name.c_str(),&pHeight,&pWidth,&pBpp,1);

for(int row=0; row<avgSize; ++row){
for (int col=0; col<avgSize; ++col){
i = (unsigned long) (imgRow*avgSize+row)*width+(avgSize*imgCol+col);
pI=row*avgSize+col;
if(image[i]!=newImage[pI])
diffCount++;
}
}
delete newImage;

if(diffCount<lowestCount){
lowestFile=fileNumber;
lowestCount=diffCount;
}
}

stringstream strs;
strs <<"data/rgb/"<<newX<<"-"<<lowestFile<<".jpg";
string name = strs.str();
unsigned char* newImage = stbi_load(name.c_str(),&pHeight,&pWidth,&pBpp,3);

for(int row=0; row<avgSize; ++row){
for (int col=0; col<avgSize; ++col){
i = (unsigned long) (imgRow*avgSize+row)*width*3+3*(avgSize*imgCol+col);
pI=row*avgSize*3+col*3;
finalImage[i]=newImage[pI];
finalImage[i+1]=newImage[pI+1];
finalImage[i+2]=newImage[pI+2];
}
}
delete newImage;
}
}
}
timeNow=time(0)-timeNow;
cout<<"Mosaic Created: "<<timeNow<<endl;

name=getName();
timeNow=time(0);
stbi_write_jpg(name.c_str(), width, height, 3, finalImage, 100);
timeNow=time(0)-timeNow;
cout<<"Image Written: "<<timeNow<<endl;

delete rgbTable;
delete avgImage;
delete finalImage;
delete image;
delete edgeDir;
delete gradient;
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
return 0;
}
}
