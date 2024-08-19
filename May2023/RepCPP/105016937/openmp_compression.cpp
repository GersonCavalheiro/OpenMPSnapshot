#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/map.hpp>
#include <omp.h>

#include<bits/stdc++.h>



using namespace cv;
using namespace std;


#define COL 16


Mat image;

vector<vector<int> > img;

vector<vector<int> > colors;

map<int,vector<int> > cluster; 





void write_compressed(map<int,vector<int> > a,vector < vector <int> > b,int row,int col, string filepath)

{
std::ofstream ofs(filepath.c_str());
boost::archive::text_oarchive oa(ofs);
oa<<a<<b<<row<<col;
}

void write_vector(vector < vector <int> > a, string filepath)

{
std::ofstream ofs(filepath.c_str());
boost::archive::text_oarchive oa(ofs);
oa & a;
}


std::string remove_extension(const std::string& filename) {
size_t lastdot = filename.find_last_of(".");
if (lastdot == std::string::npos) return filename;
return filename.substr(0, lastdot); 
}

void initCentroids()
{
for(int i=0;i<COL;i++)
{






int rand_index = rand()%img.size();

colors.push_back(img[rand_index]);



}



} 


int dis(vector<int> a,vector<int> b)
{
return sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2)+pow(a[2]-b[2],2));
}



void kmeans()
{

double start=omp_get_wtime();

bool flag=true;
int count=0;

while(flag&&count<100)
{
count++;

cluster.clear();


int min_rgb,min_rgb_index;

#pragma omp parallel for private(min_rgb,min_rgb_index)
for(int i=0;i<img.size();i++)
{
min_rgb=INT_MAX;
min_rgb_index=-1;

for(int j=0;j<colors.size();j++)
{
if(j==i)
continue;
int distance=dis(img[i],colors[j]);



if(min_rgb>distance)
{
min_rgb=distance;
min_rgb_index=j;
}

}
#pragma omp critical
{
if(cluster.find(min_rgb_index)==cluster.end())
{    vector<int> v;
v.push_back(i);
cluster.insert(make_pair(min_rgb_index,v));
}
else
cluster[min_rgb_index].push_back(i);
}
}

map<int,vector<int> >::iterator it;




for(int i=0;i<COL;i++)
{
int blue=0,green=0,red=0,j=0;
#pragma omp parallel for reduction(+:red,green,blue) private(j)
for(j=0;j<cluster[i].size();j++)
{
blue+=img[cluster[i][j]][0];
green+=img[cluster[i][j]][1];
red+=img[cluster[i][j]][2];
}

vector<int> new_val;
if(cluster[i].size())
{
new_val.push_back(blue/cluster[i].size());
new_val.push_back(green/cluster[i].size());
new_val.push_back(red/cluster[i].size());
}
else
{
int rand_index = rand()%img.size();
new_val = img[rand_index];
}


for(int k=0;k<3;k++)
{
if(new_val[k]!=colors[i][k])
{
colors[i][k]=new_val[k];
flag=flag && false;
}
else
flag=flag && true;
}
}
flag=!flag;


}




double end=omp_get_wtime();

cout<<"Time taken:"<<end-start<<endl;

}


int main( int argc, char** argv )
{
if( argc != 2)
{
cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
return -1;
}


image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   

if(! image.data )                              
{
cout <<  "Could not open or find the image" << std::endl ;
return -1;
}


namedWindow( "Initial Image", WINDOW_AUTOSIZE );

imshow( "Initial Image", image );                




vector<int> temp;
for(int i=0;i<image.rows;i++)
{

for(int j=0;j<image.cols;j++)
{
temp.clear();
Vec3b intensity = image.at<Vec3b>(i,j);
temp.push_back((int)intensity.val[0]);
temp.push_back((int)intensity.val[1]);
temp.push_back((int)intensity.val[2]);
img.push_back(temp);
}

}



write_vector(img,remove_extension(argv[1]).append("_initial.bin"));

initCentroids();
kmeans();

write_compressed(cluster,colors,image.rows,image.cols,remove_extension(argv[1]).append("_compressed.bin"));





return 0;
}

