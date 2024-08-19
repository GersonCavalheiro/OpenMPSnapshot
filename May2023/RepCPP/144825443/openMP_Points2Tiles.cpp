

#include "stdafx.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <direct.h>
#include <io.h>
#include <algorithm>
using namespace std;

static double LatMin = -90.0;
static double LngMin = -180.0;
static int nResolution = 256;

class Point {
public:
double lat;
double lng;
};
class PointsTile {
public:
int nLevel;
int nRow;
int nCol;
vector <Point> Points;
};
class PointCount {
public:
int nPosition;
long nCount;
};
class PointCountTile {
public:
int nLevel;
int nRow;
int nCol;
vector <PointCount> pointCount;
};

class LevelCountMinMax {
public:
int nLevel;
int nMin;
int nMax;
};
vector <Point> vcPoints; 
vector <PointsTile> vcPointsTiles; 
vector <LevelCountMinMax> vcLevelCountMinMax; 

int nTotalLevel = 0;
string sOutputFilePath = "";

void readPointsFromCSV(string pathInput, int nLatIndex, int nLngIndex, vector<Point> &vcPoints);
void cutTile(int nTotalLevel, vector<Point> points, vector <PointsTile> &vcPointsTiles);
void outputPointsTiles(string pathFile, vector <PointsTile> vcPointsTiles);
void transPointCountTiles(string pathFile, int nTotalLevel, vector <LevelCountMinMax> &vcLevelCountMinMax,int nResolution);

int main()
{
clock_t start, end;
start = clock();

nTotalLevel = 10;
sOutputFilePath = "D:\\VSProjects\\openMP_Points2Tiles\\dpTile";
readPointsFromCSV("pointsExample.csv", 11, 10, vcPoints);
transPointCountTiles(sOutputFilePath, nTotalLevel, vcLevelCountMinMax, nResolution);
end = clock();
cout << "Total time: " << (end - start) << " ms" << endl;
getchar();
return 0;
}

void SplitString(const string& s, vector<string>& v, const string& c)
{
string::size_type pos1, pos2;
pos2 = s.find(c);
pos1 = 0;
while (string::npos != pos2)
{
v.push_back(s.substr(pos1, pos2 - pos1));

pos1 = pos2 + c.size();
pos2 = s.find(c, pos1);
}
if (pos1 != s.length())
v.push_back(s.substr(pos1));
}

void readPointsFromCSV(string pathInput, int nLatIndex, int nLngIndex, vector<Point> &vcPoints) {
ifstream inPointsCSV(pathInput, ios::in);
string sLine = "";
int nCount = 1;
vector <string> vcFile;
while (getline(inPointsCSV, sLine))
{
vcFile.push_back(sLine);
if (vcFile.size() == 100000) {
#pragma omp parallel for
for (int i = 0; i < vcFile.size(); i++)
{
vector <string> vcLine;
SplitString(vcFile[i], vcLine, ",");
Point tPoint;
tPoint.lat = stod(vcLine[nLatIndex]); tPoint.lng = stod(vcLine[nLngIndex]);
if (tPoint.lat < 90.0 && tPoint.lat > -90.0 && tPoint.lng < 180.0 && tPoint.lng > -180.0) {
#pragma omp critical
vcPoints.push_back(tPoint);
}
}
vcFile.clear();
}
}
#pragma omp parallel for
for (int i = 0; i < vcFile.size(); i++)
{
vector <string> vcLine;
SplitString(vcFile[i], vcLine, ",");
Point tPoint;
tPoint.lat = stod(vcLine[nLatIndex]); tPoint.lng = stod(vcLine[nLngIndex]);
if (tPoint.lat < 90.0 && tPoint.lat > -90.0 && tPoint.lng < 180.0 && tPoint.lng > -180.0) {
#pragma omp critical
vcPoints.push_back(tPoint);
}
}
vcFile.clear();

cout << "Points num " << vcPoints.size() << "\n";
cutTile(nTotalLevel, vcPoints, vcPointsTiles);
cout << "Cut tiles finished.\n";
}

void initTiles(int nTotalLevel, vector <PointsTile> &vcPointsTiles) {
int nTileTotalNum = 0;
for (int k = 0; k < nTotalLevel; k++)
{
nTileTotalNum += int(pow(4.0, k + 1));
}
vcPointsTiles.resize(nTileTotalNum);
for (int i = 0; i < nTotalLevel; i++) {
int nTileBaseNum = 0;
for (int m = 1; m < i + 1; m++)
{
nTileBaseNum += int(pow(4.0, m));
}
#pragma omp parallel for
for (int m = 0; m < int(pow(2.0, i + 1)); m++) {
#pragma omp parallel for
for (int n = 0; n < int(pow(2.0, i + 1)); n++) {
int nTileNum = nTileBaseNum + m * int(pow(2.0, i + 1)) + n;
vcPointsTiles[nTileNum].nLevel = i; vcPointsTiles[nTileNum].nRow = m; vcPointsTiles[nTileNum].nCol = n;
}
}
}
}
void cutTile(int nTotalLevel, vector<Point> points, vector <PointsTile> &vcPointsTiles) {
clock_t start, end;
start = clock();

initTiles(nTotalLevel, vcPointsTiles);
cout << "Init Finish.\n";
for (int i = 0; i < nTotalLevel; i++) {
int nTileBaseNum = 0;
for (int m = 1; m < i + 1; m++)
{
nTileBaseNum += int(pow(4.0, m));
}
# pragma omp parallel for 
for (int j = 0; j < points.size(); j++) {
int nRow = int((points[j].lat - LatMin) / 180.0 * pow(2.0, i + 1));
int nCol = int((points[j].lng - LngMin) / 360.0 * pow(2.0, i + 1));
if (nRow >= 0 && nCol >= 0) {
int nTileNum = nTileBaseNum + nRow * int(pow(2.0, i + 1)) + nCol;
# pragma omp critical
vcPointsTiles[nTileNum].Points.push_back(points[j]);
}
}
}
end = clock();
cout << "Tiles cut time: " << (end - start) << " ms" << endl;
}

void outputPointsTiles(string pathFile, vector <PointsTile> vcPointsTiles) {
int flag = _mkdir(pathFile.c_str());
string pathPointsTiles = pathFile + "\\tilepoints";
flag = _mkdir(pathPointsTiles.c_str());
for (int i = 0; i < nTotalLevel; i++)
{
string sTileFile = "\\" + to_string(i);
string pathPointTileFile = pathPointsTiles + sTileFile;
flag = _mkdir(pathPointTileFile.c_str());
}
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < vcPointsTiles.size(); i++)
{
if (vcPointsTiles[i].Points.size() > 0)
{
ofstream output;
string sFilePath = pathPointsTiles + "\\" + to_string(vcPointsTiles[i].nLevel) + "\\" + \
to_string(vcPointsTiles[i].nLevel) + "_" + to_string(vcPointsTiles[i].nRow) + "_" + to_string(vcPointsTiles[i].nCol) + ".tilepoints";
output.open(sFilePath, ios::trunc);
for (int j = 0; j < vcPointsTiles[i].Points.size(); j++)
{
output << vcPointsTiles[i].Points[j].lat << "," << vcPointsTiles[i].Points[j].lng << "\n";
}
output.close();
}
}
cout << "points tiles output finished\n";
}



void transPointCountTiles(string pathFile, int nTotalLevel, vector <LevelCountMinMax> &vcLevelCountMinMax, int nResolution) {
vcLevelCountMinMax.resize(nTotalLevel);
for (int i = 0; i < nTotalLevel; i++)
{
vcLevelCountMinMax[i].nLevel = i;
vcLevelCountMinMax[i].nMin = INT_MAX;
vcLevelCountMinMax[i].nMax = -1;
}
int flag = _mkdir(pathFile.c_str());
string pathCountTiles = pathFile + "\\counttile";
flag = _mkdir(pathCountTiles.c_str());
for (int i = 0; i < nTotalLevel; i++)
{
string sTileFile = "\\" + to_string(i);
string pathCountTileFile = pathCountTiles + sTileFile;
flag = _mkdir(pathCountTileFile.c_str());
}
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < vcPointsTiles.size(); i++)
{
if (vcPointsTiles[i].Points.size() > 0) {
int *nPointCount = new int[nResolution * nResolution]();
int nLevel = vcPointsTiles[i].nLevel; int nRow = vcPointsTiles[i].nRow; int nCol = vcPointsTiles[i].nCol;
double dResolLat = 180.0 / (pow(2, nLevel + 1) * nResolution);
double dResolLnt = dResolLat * 2;
#pragma omp parallel for
for (int j = 0; j < vcPointsTiles[i].Points.size(); j++)
{
int nR = int((vcPointsTiles[i].Points[j].lat + 90.0 - (180.0 / pow(2, nLevel + 1))*nRow) / dResolLat);
int nC = int((vcPointsTiles[i].Points[j].lng + 180.0 - (360.0 / pow(2, nLevel + 1))*nCol) / dResolLnt);
#pragma omp atomic
nPointCount[nR * nResolution + nC] += 1;
}
int nCountMax = *max_element(nPointCount, nPointCount + nResolution * nResolution);
int nCountMin = *min_element(nPointCount, nPointCount + nResolution * nResolution);
if (nCountMax > vcLevelCountMinMax[nLevel].nMax) vcLevelCountMinMax[nLevel].nMax = nCountMax;
if (nCountMin < vcLevelCountMinMax[nLevel].nMin) vcLevelCountMinMax[nLevel].nMin = nCountMin;
ofstream output;
string sFilePath = pathCountTiles + "\\" + to_string(nLevel) + "\\" + \
to_string(nLevel) + "_" + to_string(nRow) + "_" + to_string(nCol) + ".counttile";
output.open(sFilePath, ios::trunc);
#pragma omp parallel for
for (int j = 0; j < nResolution * nResolution; j++)
{
if (nPointCount[j] > 0) {
output << j << "#" << nPointCount[j] << "\n";
}
}
output.close();
delete[] nPointCount;
}
}
ofstream output;
string sPath = pathCountTiles + "\\minmax.dat";
output.open(sPath, ios::trunc);
output << "level, mincount, maxcount\n";
for (int i = 0; i < nTotalLevel; i++)
{
output << vcLevelCountMinMax[i].nLevel << "," << vcLevelCountMinMax[i].nMin << "," << \
vcLevelCountMinMax[i].nMax << "\n";
}
}
