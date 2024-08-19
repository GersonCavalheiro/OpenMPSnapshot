#include <cv.h>
#include <highgui.h>
#include <math.h>
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "cvaux.lib")
#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "highgui.lib")
#define cimg_plugin1 "plugins\cimgIPL.h"
#include "CImg.h"
using namespace cimg_library;
int main(int argc, char* argv[]) {
int wid = 0;
CImg<> cImg(argv[1]);
cImg.display("cImg");
IplImage* ipl;
ipl = cImg.get_IPL();
IplImage *ipl8;
IplImage *ipl16, *ipl32, *ipl64;
IplImage *ipl16to8, *ipl32to8, *ipl64to8;
cvNamedWindow("origin", wid++);
cvNamedWindow("8bit_OK", wid++);
cvNamedWindow("16bit", wid++);
cvNamedWindow("32bit", wid++);
cvNamedWindow("64bit", wid++);
cvNamedWindow("16bitto8", wid++);
cvNamedWindow("32bitto8", wid++);
cvNamedWindow("64bitto8", wid++);
cvShowImage("origin", ipl);
ipl8 = cvCreateImage(cvGetSize(ipl), IPL_DEPTH_8U, ipl->nChannels);
cvConvert(ipl, ipl8);
ipl16 = cvCreateImage(cvGetSize(ipl), IPL_DEPTH_16U, ipl->nChannels);
cvConvert(ipl, ipl16);
ipl32 = cvCreateImage(cvGetSize(ipl), IPL_DEPTH_32F, ipl->nChannels);
cvConvert(ipl, ipl32);
ipl64 = cvCreateImage(cvGetSize(ipl), IPL_DEPTH_64F, ipl->nChannels);
cvConvert(ipl, ipl64);
cvShowImage("8bit_OK", ipl8);
cvShowImage("16bit", ipl16);
cvShowImage("32bit", ipl32);
cvShowImage("64bit", ipl64);
ipl16to8 = cvCreateImage(cvGetSize(ipl16), IPL_DEPTH_8U, ipl16->nChannels);
cvConvert(ipl16, ipl16to8);
ipl32to8 = cvCreateImage(cvGetSize(ipl32), IPL_DEPTH_8U, ipl32->nChannels);
cvConvert(ipl32, ipl32to8);
ipl64to8 = cvCreateImage(cvGetSize(ipl64), IPL_DEPTH_8U, ipl64->nChannels);
cvConvert(ipl64, ipl64to8);
cvShowImage("16bitto8", ipl16to8);    
cvShowImage("32bitto8", ipl32to8);    
cvShowImage("64bitto8", ipl64to8);    
cImg.assign(ipl8);
cImg.display("ipl8->cimg");
cImg.assign(ipl16);
cImg.display("ipl16->cimg");
cImg.assign(ipl32);
cImg.display("ipl32->cimg");
cImg.assign(ipl64);
cImg.display("ipl64->cimg");
cvWaitKey(0);
CImg<unsigned char> testCImg1(ipl16);
testCImg1.display("testCImg1");
CImg<unsigned char> testCImg2(ipl32);
testCImg2.display("testCImg2");
CImg<unsigned char> testCImg3(ipl64);
testCImg3.display("testCImg3");
CImg<double> testCImg4(ipl16);
testCImg4.display("testCImg4");
CImg<double> testCImg5(ipl32);
testCImg5.display("testCImg5");
CImg<double> testCImg6(ipl64);
testCImg6.display("testCImg6");
cvReleaseImage(&ipl);
cvReleaseImage(&ipl8);
cvReleaseImage(&ipl16);
cvReleaseImage(&ipl32);
cvReleaseImage(&ipl64);
cvReleaseImage(&ipl16to8);
cvReleaseImage(&ipl32to8);
cvReleaseImage(&ipl64to8);
cvDestroyWindow("origin");
cvDestroyWindow("8bit_OK");
cvDestroyWindow("16bit");
cvDestroyWindow("32bit");
cvDestroyWindow("64bit");
cvDestroyWindow("16bitto8");
cvDestroyWindow("32bitto8");
cvDestroyWindow("64bitto8");
return 0;
}
