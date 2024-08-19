
#define GL_SILENCE_DEPRECATION
#include <omp.h> 
#include <chrono>

#include <stdio.h> 
#include <stdlib.h>
#include <iostream>  
#include <vector> 
#include <random> 
#include <SFML/Graphics.hpp>
#include <OpenGL/gl.h>

int h = 1000;
int w = 1000;
long int ENTITIES = h * w;
float vertexCoords[2*1000*1000] ; 
unsigned char colors[3*1000*1000] ; 

using namespace std;
int main(int argc, char* argv[]) 
{ 


int arr1[h+2][w+2];
int arr2[h+2][w+2];

printf("Hello\n");

for (int i=0;i<h+2;i++) {
for (int j=0;j<w+2;j++) {
if(i==0 || j==0 || i==h+1 || j==w+1) {
arr1[i][j] = 0;
arr2[i][j] = 0;
}
else{
arr1[i][j] = rand() % 2;
arr2[i][j] = 0;		
}
}
}
printf("Hello\n");

sf::RenderWindow window(sf::VideoMode(2000, 2000, 32), "My window");
glViewport(0,0,2000,2000) ; 
glMatrixMode(GL_PROJECTION) ;
glLoadIdentity() ;
glOrtho(0,2000, 2000,0,-1,1) ;
printf("Hello\n");

printf("Hello\n");

printf("Hello1\n");
for(long int i(0) ; i < ENTITIES ; i++)
{
colors[3*i] = floor(10 + 230.0 * (i/w)/h) ;
colors[3*i+1] = floor(10 + 230 - 230.0 * (i%w)/w) ; 
colors[3*i+2]= floor(10 + 230 - 230.0 * (i/w)/h) ;
}
printf("Hello2\n");
glEnable( GL_POINT_SMOOTH ); 
glPointSize( 1 );

const int distance = 4;
const float offset = distance/2.f;
const float height = 1;
printf("Hello\n");
window.setFramerateLimit(30) ;

while (window.isOpen())
{
sf::Event event;
while (window.pollEvent(event))
{
if (event.type == sf::Event::Closed)
window.close();
}


int iterations = 1000;
int c = 20;
#pragma omp parallel for
for (int ni=0; ni<floor(h/c); ni++){
for (int i=0;i<c;i++) {
for (int j=0;j<w;j++) {
int ci = ni*c + i + 1;
int cj = j+1;
int live = arr1[ci+1][cj] + arr1[ci+1][cj+1] + arr1[ci][cj+1] + arr1[ci-1][cj+1] + arr1[ci-1][cj] + arr1[ci-1][cj-1] + arr1[ci][cj-1] + arr1[ci+1][cj-1];
int curr = arr1[ci][cj];
if (curr == 1){
if(live < 2){
curr = 0;
}
if(live > 3){
curr = 0;
}
}else{
if(live == 3){
curr = 1;
}
}
arr2[ci][cj] = curr;
}
}
}
glClearColor(0,0,0,0); 
glClear(GL_COLOR_BUFFER_BIT); 

glPushMatrix() ; 
glTranslatef(2000/2.f,2000/2.f,0) ; 
glScaled(1, 1, 1);
glTranslated(-1000,-1000,0) ; 
glEnableClientState(GL_VERTEX_ARRAY) ; 
glEnableClientState(GL_COLOR_ARRAY) ;

for (int i=1; i<h+1; ++i){
for (int j=1; j<w+1; ++j){
long int flattenedIndx = (i-1)*w + (j-1);

if(arr2[i][j] == 1){	
vertexCoords[flattenedIndx*2 + 0] = i*2.0f ;
vertexCoords[flattenedIndx*2 + 1] = j*2.0f ;
}
else{
vertexCoords[flattenedIndx*2] = 0.0f ;
vertexCoords[flattenedIndx*2 + 1] = 0.0f ;	
}
}
}
glVertexPointer(2,GL_FLOAT,0,vertexCoords) ;
glColorPointer(3,GL_UNSIGNED_BYTE,0,colors) ;
glDrawArrays(GL_POINTS,0,ENTITIES) ;

glDisableClientState(GL_VERTEX_ARRAY) ;
glDisableClientState(GL_COLOR_ARRAY) ;
glPopMatrix();
glFlush();
window.display();


for (int i=0;i<h+2;i++) {
for (int j=0;j<w+2;j++) {
arr1[i][j] = arr2[i][j];
}
}


}

} 