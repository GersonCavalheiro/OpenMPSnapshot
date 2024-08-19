




#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <algorithm> 


#include <Windows.h>
#include <process.h>


#define WINDOW_NAME "Camrea"		
#define MARGIN_SIZE_OF_QRCODE 10		


struct SquareContour
{
cv::Point ptArr[4];		
cv::Point2d midPt;
double length;			
};
struct PositionPattern
{
SquareContour outer;
SquareContour mid;	
SquareContour inner;
};
struct QrCodePose
{
std::vector<PositionPattern> tmpPositionPatterns;

PositionPattern positionPatternTopL;
PositionPattern positionPatternTopR;
PositionPattern positionPatternBotL;

cv::Point ptTopL;	
cv::Point ptTopR;	
cv::Point ptTopR_neighbor;
cv::Point ptBotL;	
cv::Point ptBotL_neighbor;
cv::Point ptBotR;	
cv::Point ptMid;

cv::Rect boundBox;
};
struct QrCode
{
bool flagDecoded = false;	
QrCodePose expectedpose;	
cv::Mat image;				
cv::String str;				
cv::Point detectedPose[4];	
cv::Rect boundBox;
};


struct pointFindingHelper
{
cv::Point pt;
double distance;
};
struct angleFindingHelper
{
cv::Point edgePt;
cv::Point supportingPt;
double angle;
PositionPattern tmpPositionPattern;
};


class OpenQR
{
private:
cv::VideoCapture* videoStream;
int threadNum;

cv::Mat imgSource;
cv::Mat imgGray;
cv::Mat imgBin;
cv::Mat imgMorphology;
cv::Mat imgEdge;

std::vector<std::vector<cv::Point>> outlineContours;	
std::vector<std::vector<cv::Point>> inlineContours;
std::vector<SquareContour> squareContours;
std::vector<PositionPattern> positionPatterns;
std::vector<QrCodePose> qrCodeList;						
std::vector<QrCodePose> expectedQrCodes;				

cv::Mat imgOutput;
std::vector<QrCode> qrcodes;		

private:
int GetCoresSize();
void SortBubble(std::vector<std::vector<cv::Point>>& outlineCoutours);
void SortBubble(std::vector<SquareContour>& squareContours);
double GetRotatedAngle(cv::Point center, cv::Point point);
void RotatePt(cv::Point centerPt, cv::Point targetPt, cv::Point& outputPt, double angle);
cv::Point GetSlicedPoint(const cv::Point& start, const cv::Point& end, const double sliceNum, const double num);
int CorrectPtBotR(const cv::Point& ptBotL, const cv::Mat& imgBin, const cv::Point& refPtBotR, cv::Point& predictedPtBotR, const int pos = 0);
bool GetIntersectionPoint(cv::Point a, cv::Point b, cv::Point c, cv::Point d, cv::Point* outputPt);
QrCodePose GetQrCodePose(QrCodePose& qrcode, cv::Mat imgBin, int& flag);
void ExtractQrCode(const cv::Mat& bin, cv::Mat& output, const cv::Point& TopLeft, const cv::Point& BottomLeft, const cv::Point& TopRight, const cv::Point& BottomRight);
void FindMarginPoint(const cv::Point& orient, const cv::Point& target, cv::Point& out);
public:
OpenQR();
void SetThreadNum(int intputThreadNum);
void Reset();

int OpenCamera(int width, int height);
int ReadFrame();
int CheckEscKey();

void FindExpectedQrcodes();
void ConvertGray();
void ConvertBin();
void Morphology();
void FindContours();
void ExtractSquareContours();
void FindPositionPatterns();
void FindQrcodesList();
void FindQrcodesPose();

void FindBoundBox();
void SegmentWithTransform();
void SegmentWithNoTransform();

void DetectAndDecodeQrcodeWithOpenCV();

int GetExpectedNum();

void DrawExpectedQrcodesBoundBox();
void DrawExpectedQrcodes();
void DrawDecodedQrcodesOnNoTransform();
void DrawDecodingFailed();
void DrawDecodedStr();
void DrawText(cv::String str, cv::Point pos);
void ShowOutput();
};