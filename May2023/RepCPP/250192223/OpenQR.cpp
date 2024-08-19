



#include "OpenQR.h"


OpenQR::OpenQR()
{
threadNum = GetCoresSize();
}


void OpenQR::SetThreadNum(int intputThreadNum)
{
threadNum = intputThreadNum;
}

int OpenQR::GetCoresSize()
{
PSYSTEM_LOGICAL_PROCESSOR_INFORMATION pProcessorInformations = NULL;
DWORD length = 0;

BOOL result = GetLogicalProcessorInformation(pProcessorInformations, &length);
if (!result)
{
if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
{
pProcessorInformations = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)new BYTE[length];
}
}

result = GetLogicalProcessorInformation(pProcessorInformations, &length);
if (!result)
{
return -1;
}

int numOfCores = 0;
for (int i = 0; i < length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); i++)
{
if (pProcessorInformations[i].Relationship == RelationProcessorCore)
numOfCores++;
}

delete[] pProcessorInformations;
return numOfCores;
}


void OpenQR::SortBubble(std::vector<std::vector<cv::Point>>& outlineCoutours)
{
for (int i = 0; i < outlineCoutours.size(); i++) {
for (int j = 0; j < outlineCoutours.size() - (i + 1); j++) {
if (outlineCoutours[j].size() > outlineCoutours[j + 1].size())
std::swap(outlineCoutours[j], outlineCoutours[j + 1]);
}
}
}

void OpenQR::SortBubble(std::vector<SquareContour>& squareContours)
{
for (int i = 0; i < squareContours.size(); i++) {
for (int j = 0; j < squareContours.size() - (i + 1); j++) {
if (squareContours[j].length > squareContours[j + 1].length)
std::swap(squareContours[j], squareContours[j + 1]);
}
}
}


int OpenQR::OpenCamera(int width, int height)
{

videoStream = new cv::VideoCapture(cv::CAP_DSHOW);
videoStream->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
videoStream->set(cv::CAP_PROP_FRAME_WIDTH, width);
videoStream->set(cv::CAP_PROP_FRAME_HEIGHT, height);

if (!videoStream->isOpened())
{
std::cerr << " - ī޶   ϴ.\n";
return -1;
}

return 1;
}
int OpenQR::ReadFrame()
{
videoStream->read(imgSource);
if (imgSource.empty())
{
std::cerr << " -   ĸĵǾϴ.\n";
return -1;
}


imgSource.copyTo(imgOutput);
return 1;
}
int OpenQR::CheckEscKey()
{
if (cv::waitKey(25) >= 0)
return 1;

return 0;
}

double OpenQR::GetRotatedAngle(cv::Point center, cv::Point point)
{
double x = point.x - center.x;
double y = point.y - center.y;
double magnitude = sqrt(x * x + y * y);
double angle = 0;
if (magnitude > 0)
angle = acos(x / magnitude);

angle = angle * 180 / 3.14159265358979323846;
if (y < 0)
angle = 360 - angle;

return angle;
}

void OpenQR::RotatePt(cv::Point centerPt, cv::Point targetPt, cv::Point& outputPt, double angle)
{
angle = 3.14159265358979323846 / 180 * angle;
double cosq = cos(angle), sinq = sin(angle);

targetPt.x -= centerPt.x, targetPt.y -= centerPt.y;

outputPt.x = targetPt.x * cosq - targetPt.y * sinq;
outputPt.y = targetPt.y * cosq + targetPt.x * sinq;

outputPt.x += centerPt.x, outputPt.y += centerPt.y;
}

cv::Point OpenQR::GetSlicedPoint(const cv::Point& start, const cv::Point& end, const double sliceNum, const double num)
{
double x1;
double y1;
double x2;
double y2;
double x;
double y;
if (start.x < end.x)
{
x1 = start.x;
y1 = start.y;
x2 = end.x;
y2 = end.y;
double x = x1 + (x2 - x1) / sliceNum * num;
double y = (y2 - y1) / (x2 - x1) * (x - x1) + y1;
return cv::Point(x, y);
}
else if (start.x == end.x)
{
if (start.y == end.y)
{
return cv::Point(start.x, start.y);
}
else if (start.y < end.y)
{
x = start.x;
y1 = start.y;
y2 = end.y;
y = y1 + (y2 - y1) / sliceNum * num;
return cv::Point(x, y);
}
else if (start.y > end.y)
{
x = start.x;
y1 = start.y;
y2 = end.y;
y = y1 - (y1 - y2) / sliceNum * num;
return cv::Point(x, y);
}
}
else if (start.x > end.x)
{
x1 = start.x;
y1 = start.y;
x2 = end.x;
y2 = end.y;
double x = x1 - (x1 - x2) / sliceNum * num;
double y = (y2 - y1) / (x2 - x1) * (x - x1) + y1;
return cv::Point(x, y);
}
return cv::Point(0, 0);
}

int OpenQR::CorrectPtBotR(const cv::Point& ptBotL, const cv::Mat& imgBin, const cv::Point& refPtBotR, cv::Point& predictedPtBotR, const int pos)
{
int rotationDirection = pos;

double degreeSize = 0.01;	
double degreeSum = 0;		
double degreeMax = 5;		
double degreeMin = -degreeMax;		

cv::Point checkPt[20];		
int checkNum = 20;			
int sliceNum = 21;			

cv::Point searchPt;			
cv::Point endPt;			

int blackSum = 0;
int endFlag = false;
int findBlack = true;

endPt = refPtBotR;
searchPt = GetSlicedPoint(ptBotL, endPt, 2, 1);


for (int i = 0; i < checkNum; i++)
{
checkPt[i] = GetSlicedPoint(searchPt, endPt, checkNum + 1, i);
}

for (int i = 0; i < checkNum; i++)
{
if (imgBin.rows <= checkPt[i].y || checkPt[i].y < 0 || imgBin.cols <= checkPt[i].x || checkPt[i].x < 0)
return -1;
}
if (imgBin.rows <= endPt.y || endPt.y < 0 || imgBin.cols <= endPt.x || endPt.x < 0)
return -1;
if (imgBin.rows <= searchPt.y || searchPt.y < 0 || imgBin.cols <= searchPt.x || searchPt.x < 0)
return -1;

blackSum = 0;
uchar* dataBin = (uchar*)imgBin.data;   
for (int i = 0; i < checkNum; i++)
{
if (dataBin[checkPt[i].y * imgBin.cols + checkPt[i].x] == 0)
{
blackSum++;
}
}
if (blackSum > 0)
{
findBlack = true;
if (rotationDirection == 1)
{
degreeSize = -degreeSize;
}
}
else
{
findBlack = false;
if (rotationDirection == 1)
{
}
else
{
degreeSize = -degreeSize;
}
}

while (degreeMin <= degreeSum && degreeSum <= degreeMax)
{
degreeSum = degreeSum + degreeSize;
RotatePt(ptBotL, refPtBotR, endPt, degreeSum);
searchPt = GetSlicedPoint(ptBotL, endPt, 2, 1);
for (int i = 0; i < checkNum; i++)
{
checkPt[i] = GetSlicedPoint(searchPt, endPt, checkNum + 1, i);
}

for (int i = 0; i < checkNum; i++)
{
if (imgBin.rows <= checkPt[i].y || checkPt[i].y < 0 || imgBin.cols <= checkPt[i].x || checkPt[i].x < 0)
return -1;
}
if (imgBin.rows <= endPt.y || endPt.y < 0 || imgBin.cols <= endPt.x || endPt.x < 0)
return -1;
if (imgBin.rows <= searchPt.y || searchPt.y < 0 || imgBin.cols <= searchPt.x || searchPt.x < 0)
return -1;

if (findBlack)
{
blackSum = 0;
for (int i = 0; i < checkNum; i++)
{
if (dataBin[checkPt[i].y * imgBin.cols + checkPt[i].x] == 0)
{
blackSum++;
}
}
if (blackSum == 0)
{
endFlag = true;
break;
}
}
else
{
for (int i = 0; i < checkNum; i++)
{
if (dataBin[checkPt[i].y * imgBin.cols + checkPt[i].x] == 0)
{
endFlag = true;
break;
}
}
if (endFlag)
break;
}

}

predictedPtBotR = endPt;
return 1;
}

bool OpenQR::GetIntersectionPoint(cv::Point a, cv::Point b, cv::Point c, cv::Point d, cv::Point* outputPt)
{
double X;
double Y;
double distAB, theCos, theSin, newX, ABpos;

if (a.x == b.x && a.y == b.y || c.x == d.x && c.y == d.y) return false;

b.x -= a.x; b.y -= a.y;
c.x -= a.x; c.y -= a.y;
d.x -= a.x; d.y -= a.y;

distAB = sqrt(b.x * b.x + b.y * b.y);

theCos = b.x / distAB;
theSin = b.y / distAB;
newX = c.x * theCos + c.y * theSin;
c.y = c.y * theCos - c.x * theSin; c.x = newX;
newX = d.x * theCos + d.y * theSin;
d.y = d.y * theCos - d.x * theSin; d.x = newX;

if (c.y == d.y) return false;

ABpos = d.x + (c.x - d.x) * d.y / (d.y - c.y);

X = a.x + ABpos * theCos;
Y = a.y + ABpos * theSin;

outputPt->x = (int)X;
outputPt->y = (int)Y;

return true;
}

QrCodePose OpenQR::GetQrCodePose(QrCodePose& qrcode, cv::Mat imgBin, int& flag)
{
double distanceMarker0with1 = cv::norm(qrcode.tmpPositionPatterns[0].outer.midPt - qrcode.tmpPositionPatterns[1].outer.midPt);
double distanceMarker1with2 = cv::norm(qrcode.tmpPositionPatterns[1].outer.midPt - qrcode.tmpPositionPatterns[2].outer.midPt);
double distanceMarker2with0 = cv::norm(qrcode.tmpPositionPatterns[2].outer.midPt - qrcode.tmpPositionPatterns[0].outer.midPt);
if ((distanceMarker0with1 > distanceMarker1with2) && (distanceMarker0with1 > distanceMarker2with0))
{
qrcode.positionPatternTopL = qrcode.tmpPositionPatterns[2];
qrcode.ptMid = cv::Point((qrcode.tmpPositionPatterns[0].outer.midPt.x + qrcode.tmpPositionPatterns[1].outer.midPt.x) / 2.0,
(qrcode.tmpPositionPatterns[0].outer.midPt.y + qrcode.tmpPositionPatterns[1].outer.midPt.y) / 2.0);
qrcode.tmpPositionPatterns.erase(qrcode.tmpPositionPatterns.begin() + 2);
}
else if ((distanceMarker1with2 > distanceMarker0with1) && (distanceMarker1with2 > distanceMarker2with0))
{
qrcode.positionPatternTopL = qrcode.tmpPositionPatterns[0];
qrcode.ptMid = cv::Point((qrcode.tmpPositionPatterns[1].outer.midPt.x + qrcode.tmpPositionPatterns[2].outer.midPt.x) / 2.0,
(qrcode.tmpPositionPatterns[1].outer.midPt.y + qrcode.tmpPositionPatterns[2].outer.midPt.y) / 2.0);
qrcode.tmpPositionPatterns.erase(qrcode.tmpPositionPatterns.begin() + 0);
}
else if ((distanceMarker2with0 > distanceMarker0with1) && (distanceMarker2with0 > distanceMarker1with2))
{
qrcode.positionPatternTopL = qrcode.tmpPositionPatterns[1];
qrcode.ptMid = cv::Point((qrcode.tmpPositionPatterns[0].outer.midPt.x + qrcode.tmpPositionPatterns[2].outer.midPt.x) / 2.0,
(qrcode.tmpPositionPatterns[0].outer.midPt.y + qrcode.tmpPositionPatterns[2].outer.midPt.y) / 2.0);
qrcode.tmpPositionPatterns.erase(qrcode.tmpPositionPatterns.begin() + 1);
}


double maxDistancepositionPatternTopLOutPtWithMidPt = cv::norm(qrcode.positionPatternTopL.outer.ptArr[0] - qrcode.ptMid);
int maxIndexTopPt = 0;
for (int i = 1; i < 4; i++)
{
double distancepositionPatternTopLOutPtWithMidPt = cv::norm(qrcode.positionPatternTopL.outer.ptArr[i] - qrcode.ptMid);

if (maxDistancepositionPatternTopLOutPtWithMidPt < distancepositionPatternTopLOutPtWithMidPt)
{
maxDistancepositionPatternTopLOutPtWithMidPt = distancepositionPatternTopLOutPtWithMidPt;
maxIndexTopPt = i;
}
}
qrcode.ptTopL = qrcode.positionPatternTopL.outer.ptArr[maxIndexTopPt];



pointFindingHelper sortPoint1[4];
for (int i = 0; i < 4; i++)
{
sortPoint1[i] = { qrcode.tmpPositionPatterns[0].outer.ptArr[i],  cv::norm(qrcode.tmpPositionPatterns[0].outer.ptArr[i] - qrcode.ptTopL) };
}
pointFindingHelper tmpPoint1;
for (int i = 0; i < 4; i++) {
for (int j = 0; j < 4 - (i + 1); j++) {
if (sortPoint1[j].distance < sortPoint1[j + 1].distance)
{
tmpPoint1 = sortPoint1[j];
sortPoint1[j] = sortPoint1[j + 1];
sortPoint1[j + 1] = tmpPoint1;
}
}
}
double distance1PtOutNeiWithMid1 = cv::norm(sortPoint1[0].pt - qrcode.ptMid);
double distance1PtOutNeiWithMid2 = cv::norm(sortPoint1[1].pt - qrcode.ptMid);
if (distance1PtOutNeiWithMid1 > distance1PtOutNeiWithMid2)
{
qrcode.ptTopR = sortPoint1[0].pt;
qrcode.ptTopR_neighbor = sortPoint1[1].pt;
}
else
{
qrcode.ptTopR = sortPoint1[1].pt;
qrcode.ptTopR_neighbor = sortPoint1[0].pt;
}



pointFindingHelper sortPoint2[4];
for (int i = 0; i < 4; i++)
{
sortPoint2[i] = { qrcode.tmpPositionPatterns[1].outer.ptArr[i],  cv::norm(qrcode.tmpPositionPatterns[1].outer.ptArr[i] - qrcode.ptTopL) };
}
pointFindingHelper tmpPoint2;
for (int i = 0; i < 4; i++) {
for (int j = 0; j < 4 - (i + 1); j++) {
if (sortPoint2[j].distance < sortPoint2[j + 1].distance)
{
tmpPoint2 = sortPoint2[j];
sortPoint2[j] = sortPoint2[j + 1];
sortPoint2[j + 1] = tmpPoint2;
}
}
}
double distance2PtOutNeiWithMid1 = cv::norm(sortPoint2[0].pt - qrcode.ptMid);
double distance2PtOutNeiWithMid2 = cv::norm(sortPoint2[1].pt - qrcode.ptMid);
if (distance2PtOutNeiWithMid1 > distance2PtOutNeiWithMid2)
{
qrcode.ptBotL = sortPoint2[0].pt;
qrcode.ptBotL_neighbor = sortPoint2[1].pt;
}
else
{
qrcode.ptBotL = sortPoint2[1].pt;
qrcode.ptBotL_neighbor = sortPoint2[0].pt;
}



angleFindingHelper angleRef[3];	
angleRef[0] = { qrcode.ptTopL, cv::Point(0, 0), GetRotatedAngle(qrcode.ptMid, qrcode.ptTopL), };
angleRef[1] = { qrcode.ptBotL, qrcode.ptBotL_neighbor, GetRotatedAngle(qrcode.ptMid, qrcode.ptBotL), qrcode.tmpPositionPatterns[1] };
angleRef[2] = { qrcode.ptTopR, qrcode.ptTopR_neighbor, GetRotatedAngle(qrcode.ptMid, qrcode.ptTopR), qrcode.tmpPositionPatterns[0] };
for (int i = 0; i < 3 - 1; i++)
{
for (int j = 0; j < 3 - 1 - i; j++)
{
if (angleRef[j].angle < angleRef[j + 1].angle)
{
angleFindingHelper temp = angleRef[j];
angleRef[j] = angleRef[j + 1];
angleRef[j + 1] = temp;
}
}
}
for (int i = 0; i < 3; i++)
{
if (angleRef[i].edgePt == qrcode.ptTopL)
{
if (i == 0)
{
qrcode.ptBotL = angleRef[1].edgePt;
qrcode.ptBotL_neighbor = angleRef[1].supportingPt;
qrcode.ptTopR = angleRef[2].edgePt;
qrcode.ptTopR_neighbor = angleRef[2].supportingPt;

qrcode.positionPatternBotL = angleRef[1].tmpPositionPattern;
qrcode.positionPatternTopR = angleRef[2].tmpPositionPattern;
}
if (i == 1)
{
qrcode.ptBotL = angleRef[2].edgePt;
qrcode.ptBotL_neighbor = angleRef[2].supportingPt;
qrcode.ptTopR = angleRef[0].edgePt;
qrcode.ptTopR_neighbor = angleRef[0].supportingPt;

qrcode.positionPatternBotL = angleRef[2].tmpPositionPattern;
qrcode.positionPatternTopR = angleRef[0].tmpPositionPattern;
}
if (i == 2)
{
qrcode.ptBotL = angleRef[0].edgePt;
qrcode.ptBotL_neighbor = angleRef[0].supportingPt;
qrcode.ptTopR = angleRef[1].edgePt;
qrcode.ptTopR_neighbor = angleRef[1].supportingPt;

qrcode.positionPatternBotL = angleRef[0].tmpPositionPattern;
qrcode.positionPatternTopR = angleRef[1].tmpPositionPattern;
}
}
}


cv::Point predPtBotR;
GetIntersectionPoint(qrcode.ptTopR, qrcode.ptTopR_neighbor, qrcode.ptBotL, qrcode.ptBotL_neighbor, &predPtBotR);



cv::Point correctedPredDownLeft;
if (CorrectPtBotR(qrcode.ptBotL, imgBin, predPtBotR, correctedPredDownLeft) != 1)
{
flag = false;
return qrcode;
}

cv::Point correctedPredUpRight;
if (CorrectPtBotR(qrcode.ptTopR, imgBin, predPtBotR, correctedPredUpRight, 1) != 1)
{
flag = false;
return qrcode;
}




cv::Point correctedDownRightOut;
GetIntersectionPoint(qrcode.ptTopR, correctedPredUpRight, qrcode.ptBotL, correctedPredDownLeft, &correctedDownRightOut);
if (0 > correctedDownRightOut.x || imgBin.cols <= correctedDownRightOut.x || 0 > correctedDownRightOut.y || imgBin.rows <= correctedDownRightOut.y)
{
flag = false;
return qrcode;
}


qrcode.ptBotR = correctedDownRightOut;
flag = true;
return qrcode;
}

void OpenQR::ExtractQrCode(const cv::Mat& bin, cv::Mat& output, const cv::Point& TopLeft, const cv::Point& BottomLeft, const cv::Point& TopRight, const cv::Point& BottomRight)
{
cv::Point2f src[4], dst[4];

src[0] = cv::Point2f((float)TopLeft.x, (float)TopLeft.y);
src[1] = cv::Point2f((float)TopRight.x, (float)TopRight.y);
src[2] = cv::Point2f((float)BottomRight.x, (float)BottomRight.y);
src[3] = cv::Point2f((float)BottomLeft.x, (float)BottomLeft.y);

float size = sqrt(pow(TopLeft.x - TopRight.x, 2) + pow(TopLeft.y - TopRight.y, 2));

dst[0] = cv::Point2f((float)0, (float)0);
dst[1] = cv::Point2f(size, (float)0);
dst[2] = cv::Point2f(size, size);
dst[3] = cv::Point2f((float)0, size);

cv::Mat transformMatrix = getPerspectiveTransform(src, dst);
warpPerspective(bin, output, transformMatrix, cv::Size(size, size));
}

void OpenQR::FindExpectedQrcodes()
{
this->ConvertGray();
this->ConvertBin();
this->Morphology();
this->FindContours();
this->ExtractSquareContours();
this->FindPositionPatterns();
this->FindQrcodesList();
this->FindQrcodesPose();
this->FindBoundBox();
}

void OpenQR::ConvertGray()
{
cvtColor(imgSource, imgGray, cv::COLOR_BGR2GRAY);
}

void OpenQR::ConvertBin()
{
cv::threshold(imgGray, imgBin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void OpenQR::Morphology()
{
imgBin.copyTo(imgMorphology);
for (int i = 0; i < 7; i++)
{
erode(imgMorphology, imgMorphology, cv::Mat());
}
}
void OpenQR::FindContours()
{
std::vector<cv::Vec4i> outlineHierarchy;
std::vector<cv::Vec4i> inlineHierarchy;
#pragma omp parallel num_threads(2) 
{
#pragma omp sections
{
#pragma omp section
{
findContours(imgMorphology, outlineContours, outlineHierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

#pragma omp section
{
findContours(imgBin, inlineContours, inlineHierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}
}
}
}
void OpenQR::ExtractSquareContours()
{
squareContours.clear();

#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < inlineContours.size(); i++)
{
std::vector<cv::Point2f> approximatedContours;
approxPolyDP(inlineContours[i], approximatedContours, cv::arcLength(cv::Mat(inlineContours[i]), true) * 0.09, true);

if (fabs(cv::contourArea(approximatedContours)) > 20 && approximatedContours.size() == 4)
{
SquareContour tmp;
tmp.ptArr[0] = approximatedContours[0];
tmp.ptArr[1] = approximatedContours[1];
tmp.ptArr[2] = approximatedContours[2];
tmp.ptArr[3] = approximatedContours[3];
tmp.length = (cv::norm(approximatedContours[0] - approximatedContours[1])
+ cv::norm(approximatedContours[1] - approximatedContours[2])
+ cv::norm(approximatedContours[2] - approximatedContours[3])
+ cv::norm(approximatedContours[3] - approximatedContours[0])) / 4.0;
tmp.midPt = cv::Point2d((double)(approximatedContours[0].x + approximatedContours[1].x + approximatedContours[2].x + approximatedContours[3].x) / 4.0,
(double)(approximatedContours[0].y + approximatedContours[1].y + approximatedContours[2].y + approximatedContours[3].y) / 4.0);

#pragma omp critical
{
squareContours.push_back(tmp);
}
}
}
}

}
void OpenQR::FindPositionPatterns()
{
const double marginDistance = 5;

#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int a = 0; a < squareContours.size() - 2; a++)
{
for (int b = a + 1; b < squareContours.size() - 1; b++)
{
if (cv::norm(squareContours[a].midPt - squareContours[b].midPt) > marginDistance)
continue;

for (int c = b + 1; c < squareContours.size(); c++)
{
if (cv::norm(squareContours[a].midPt - squareContours[c].midPt) > marginDistance)
continue;
if (cv::norm(squareContours[b].midPt - squareContours[c].midPt) > marginDistance)
continue;

std::vector<SquareContour> sort;
sort.push_back(squareContours[a]);
sort.push_back(squareContours[b]);
sort.push_back(squareContours[c]);
SortBubble(sort);

PositionPattern tmp;
tmp.inner = sort[0];
tmp.mid = sort[1];
tmp.outer = sort[2];

#pragma omp critical
{
positionPatterns.push_back(tmp);
}
}
}
}
}
}
void OpenQR::FindQrcodesList()
{
std::vector<PositionPattern> expectedQrcode;
std::vector<int> index;

SortBubble(outlineContours);

for (int i = 0; i < outlineContours.size(); i++)
{
expectedQrcode.clear();
index.clear();
for (int j = 0; j < positionPatterns.size(); j++)
{
if (cv::pointPolygonTest(outlineContours[i], positionPatterns[j].mid.ptArr[0], false) == 1)
{
expectedQrcode.push_back(positionPatterns[j]);
index.push_back(j);
}
}

if (expectedQrcode.size() == 3)
{
QrCodePose tmp;
tmp.tmpPositionPatterns = expectedQrcode;
qrCodeList.push_back(tmp);

positionPatterns.erase(positionPatterns.begin() + index[2]);
positionPatterns.erase(positionPatterns.begin() + index[1]);
positionPatterns.erase(positionPatterns.begin() + index[0]);
}
}
}
void OpenQR::FindQrcodesPose()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < qrCodeList.size(); i++)
{
int flagCorrect = false;
QrCodePose qrcode = GetQrCodePose(qrCodeList[i], imgBin, flagCorrect);

if (flagCorrect)
{
#pragma omp critical
{
expectedQrCodes.push_back(qrcode);
}
}	
}	
}
}


void OpenQR::Reset()
{
outlineContours.clear();
inlineContours.clear();
squareContours.clear();
positionPatterns.clear();
qrCodeList.clear();
expectedQrCodes.clear();
qrcodes.clear();
}

void OpenQR::FindBoundBox()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < expectedQrCodes.size(); i++)
{
cv::Rect tmp;
tmp.x = min(min(expectedQrCodes[i].ptTopL.x, expectedQrCodes[i].ptTopR.x), min(expectedQrCodes[i].ptBotR.x, expectedQrCodes[i].ptBotL.x));
tmp.y = min(min(expectedQrCodes[i].ptTopL.y, expectedQrCodes[i].ptTopR.y), min(expectedQrCodes[i].ptBotR.y, expectedQrCodes[i].ptBotL.y));
int maxX = max(max(expectedQrCodes[i].ptTopL.x, expectedQrCodes[i].ptTopR.x), max(expectedQrCodes[i].ptBotR.x, expectedQrCodes[i].ptBotL.x));
int maxY = max(max(expectedQrCodes[i].ptTopL.y, expectedQrCodes[i].ptTopR.y), max(expectedQrCodes[i].ptBotR.y, expectedQrCodes[i].ptBotL.y));
tmp.width = maxX - tmp.x;
tmp.height = maxY - tmp.y;
expectedQrCodes[i].boundBox = tmp;
}
}
}

void OpenQR::SegmentWithTransform()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < expectedQrCodes.size(); i++)
{
cv::Mat extractedQrcode;
ExtractQrCode(imgBin, extractedQrcode, expectedQrCodes[i].ptTopL, expectedQrCodes[i].ptBotL, expectedQrCodes[i].ptTopR, expectedQrCodes[i].ptBotR);

cv::Mat expanded((int)extractedQrcode.rows + MARGIN_SIZE_OF_QRCODE * 2, (int)extractedQrcode.cols + MARGIN_SIZE_OF_QRCODE * 2, CV_8UC1, cv::Scalar(255, 255, 255));
uchar* out = (uchar*)expanded.data;
uchar* bin = (uchar*)extractedQrcode.data;
for (int y = 0; y < extractedQrcode.rows; y++) {
for (int x = 0; x < extractedQrcode.cols; x++) {

if (y >= expanded.rows || y < 0 || x >= expanded.cols || x < 0)
continue;

out[(MARGIN_SIZE_OF_QRCODE + y) * expanded.cols + (MARGIN_SIZE_OF_QRCODE + x)] = bin[y * extractedQrcode.cols * 1 + x * 1];
}
}

if (expanded.rows > 500 || expanded.cols >> 500)
resize(expanded, expanded, cv::Size(500, 500));

QrCode tmp;
tmp.expectedpose = expectedQrCodes[i];
tmp.image = expanded;
tmp.boundBox = expectedQrCodes[i].boundBox;

#pragma omp critical
{
qrcodes.push_back(tmp);
}
}
}
}


void OpenQR::FindMarginPoint(const cv::Point& orient, const cv::Point& target, cv::Point& out)
{
out = target;
out -= orient;
out.x *= 1.05;
out.y *= 1.05;
out += orient;
}

void OpenQR::SegmentWithNoTransform()
{
#pragma omp parallel num_threads(1) 
{
#pragma omp for
for (int i = 0; i < expectedQrCodes.size(); i++)
{
int minX = expectedQrCodes[i].boundBox.x;
int minY = expectedQrCodes[i].boundBox.y;
int maxX = expectedQrCodes[i].boundBox.x + expectedQrCodes[i].boundBox.width;
int maxY = expectedQrCodes[i].boundBox.y + expectedQrCodes[i].boundBox.height;

minX -= MARGIN_SIZE_OF_QRCODE;
minY -= MARGIN_SIZE_OF_QRCODE;
maxX += MARGIN_SIZE_OF_QRCODE;
maxY += MARGIN_SIZE_OF_QRCODE;

if (minX < 0 || minY < 0 || maxX >= imgBin.cols || maxY >= imgBin.rows)
continue;

cv::Rect rect(minX, minY, maxX - minX, maxY - minY);
cv::Mat cutted;
imgBin(rect).copyTo(cutted);

cv::Mat mask(cutted.rows, cutted.cols, CV_8UC1, cv::Scalar(255, 255, 255));
std::vector<cv::Point> pt;
cv::Point out1;
FindMarginPoint(expectedQrCodes[i].ptMid, cv::Point(expectedQrCodes[i].ptTopL.x, expectedQrCodes[i].ptTopL.y), out1);
pt.push_back(cv::Point(out1.x - minX, out1.y - minY));
FindMarginPoint(expectedQrCodes[i].ptMid, cv::Point(expectedQrCodes[i].ptTopR.x, expectedQrCodes[i].ptTopR.y), out1);
pt.push_back(cv::Point(out1.x - minX, out1.y - minY));
FindMarginPoint(expectedQrCodes[i].ptMid, cv::Point(expectedQrCodes[i].ptBotR.x, expectedQrCodes[i].ptBotR.y), out1);
pt.push_back(cv::Point(out1.x - minX, out1.y - minY));
FindMarginPoint(expectedQrCodes[i].ptMid, cv::Point(expectedQrCodes[i].ptBotL.x, expectedQrCodes[i].ptBotL.y), out1);
pt.push_back(cv::Point(out1.x - minX, out1.y - minY));
const cv::Point* pts3 = (const cv::Point*) cv::Mat(pt).data;
int var = 4;
cv::fillPoly(mask, &pts3, &var, 1, cv::Scalar(0, 0, 0), 8);
cutted = cutted + mask;

if(maxX - minX > 500 || maxY - minY >> 500)
resize(cutted, cutted, cv::Size(500, 500));

QrCode tmp;
tmp.expectedpose = expectedQrCodes[i];
tmp.image = cutted;
tmp.boundBox = expectedQrCodes[i].boundBox;

#pragma omp critical
{
qrcodes.push_back(tmp);
}
}
}
}

void OpenQR::DetectAndDecodeQrcodeWithOpenCV()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < qrcodes.size(); i++)
{
cv::QRCodeDetector qrDecoder = cv::QRCodeDetector::QRCodeDetector();
cv::Mat bbox, rectifiedImage;
std::string data = qrDecoder.detectAndDecode(qrcodes[i].image, bbox, rectifiedImage);
if (data.length() > 0)
{
qrcodes[i].flagDecoded = true;
qrcodes[i].str = data;

qrcodes[i].detectedPose[0] = cv::Point2i(bbox.at<float>(0, 0) + qrcodes[i].boundBox.x - MARGIN_SIZE_OF_QRCODE, bbox.at<float>(0, 1) + qrcodes[i].boundBox.y - MARGIN_SIZE_OF_QRCODE);
qrcodes[i].detectedPose[1] = cv::Point2i(bbox.at<float>(1, 0) + qrcodes[i].boundBox.x - MARGIN_SIZE_OF_QRCODE, bbox.at<float>(1, 1) + qrcodes[i].boundBox.y - MARGIN_SIZE_OF_QRCODE);
qrcodes[i].detectedPose[2] = cv::Point2i(bbox.at<float>(2, 0) + qrcodes[i].boundBox.x - MARGIN_SIZE_OF_QRCODE, bbox.at<float>(2, 1) + qrcodes[i].boundBox.y - MARGIN_SIZE_OF_QRCODE);
qrcodes[i].detectedPose[3] = cv::Point2i(bbox.at<float>(3, 0) + qrcodes[i].boundBox.x - MARGIN_SIZE_OF_QRCODE, bbox.at<float>(3, 1) + qrcodes[i].boundBox.y - MARGIN_SIZE_OF_QRCODE);
}
}
}
}

void OpenQR::DrawExpectedQrcodesBoundBox()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < qrcodes.size(); i++)
{
line(imgOutput, cv::Point(qrcodes[i].boundBox.x, qrcodes[i].boundBox.y), cv::Point(qrcodes[i].boundBox.x + qrcodes[i].boundBox.width, qrcodes[i].boundBox.y), cv::Scalar(0, 255, 0), 2);
line(imgOutput, cv::Point(qrcodes[i].boundBox.x + qrcodes[i].boundBox.width, qrcodes[i].boundBox.y), cv::Point(qrcodes[i].boundBox.x + qrcodes[i].boundBox.width, qrcodes[i].boundBox.y + qrcodes[i].boundBox.height), cv::Scalar(0, 255, 0), 2);
line(imgOutput, cv::Point(qrcodes[i].boundBox.x + qrcodes[i].boundBox.width, qrcodes[i].boundBox.y + qrcodes[i].boundBox.height), cv::Point(qrcodes[i].boundBox.x, qrcodes[i].boundBox.y + qrcodes[i].boundBox.height), cv::Scalar(0, 255, 0), 2);
line(imgOutput, cv::Point(qrcodes[i].boundBox.x, qrcodes[i].boundBox.y + qrcodes[i].boundBox.height), cv::Point(qrcodes[i].boundBox.x, qrcodes[i].boundBox.y), cv::Scalar(0, 255, 0), 2);
}	
}
}

void OpenQR::DrawExpectedQrcodes()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < qrcodes.size(); i++)
{
line(imgOutput, qrcodes[i].expectedpose.ptTopL, qrcodes[i].expectedpose.ptTopR, cv::Scalar(0, 255, 0), 2);
line(imgOutput, qrcodes[i].expectedpose.ptTopR, qrcodes[i].expectedpose.ptBotR, cv::Scalar(0, 255, 0), 2);
line(imgOutput, qrcodes[i].expectedpose.ptBotR, qrcodes[i].expectedpose.ptBotL, cv::Scalar(255, 0, 0), 2);
line(imgOutput, qrcodes[i].expectedpose.ptBotL, qrcodes[i].expectedpose.ptTopL, cv::Scalar(0, 255, 0), 2);
}	
}
}
void OpenQR::DrawDecodedQrcodesOnNoTransform()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < qrcodes.size(); i++)
{
if (qrcodes[i].flagDecoded)
{
line(imgOutput, qrcodes[i].detectedPose[0], qrcodes[i].detectedPose[1], cv::Scalar(0, 255, 0), 2);
line(imgOutput, qrcodes[i].detectedPose[1], qrcodes[i].detectedPose[2], cv::Scalar(0, 255, 0), 2);
line(imgOutput, qrcodes[i].detectedPose[2], qrcodes[i].detectedPose[3], cv::Scalar(255, 0, 0), 2);
line(imgOutput, qrcodes[i].detectedPose[3], qrcodes[i].detectedPose[0], cv::Scalar(0, 255, 0), 2);
}
}	
}
}
void OpenQR::DrawDecodingFailed()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < qrcodes.size(); i++)
{
if (qrcodes[i].flagDecoded == false)
{
line(imgOutput, qrcodes[i].detectedPose[0], qrcodes[i].detectedPose[1], cv::Scalar(0, 0, 255), 2);
line(imgOutput, qrcodes[i].detectedPose[1], qrcodes[i].detectedPose[2], cv::Scalar(0, 0, 255), 2);
line(imgOutput, qrcodes[i].detectedPose[2], qrcodes[i].detectedPose[3], cv::Scalar(0, 0, 255), 2);
line(imgOutput, qrcodes[i].detectedPose[3], qrcodes[i].detectedPose[0], cv::Scalar(0, 0, 255), 2);
}
}	
}
}
void OpenQR::DrawDecodedStr()
{
#pragma omp parallel num_threads(threadNum) 
{
#pragma omp for
for (int i = 0; i < qrcodes.size(); i++)
{
if (qrcodes[i].flagDecoded)
putText(imgOutput, qrcodes[i].str, qrcodes[i].expectedpose.ptMid, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
}	
}
}

void OpenQR::ShowOutput()
{
cv::imshow(WINDOW_NAME, imgOutput);
}

void OpenQR::DrawText(cv::String str, cv::Point pos)
{
putText(imgOutput, str, pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}

int OpenQR::GetExpectedNum()
{
return expectedQrCodes.size();
}