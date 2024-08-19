#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string>

#include "capturev4l2.hpp"

static int xioctl(int fd, int request, void *arg)
{
int r;

do r = ioctl (fd, request, arg);
while (-1 == r && EINTR == errno);

return r;
}


CaptureV4L2::CaptureV4L2(std::string device, int shift){
if(open_camera(device))
return;

shift_ = shift;
}

int CaptureV4L2::print_caps()
{
struct v4l2_capability caps = {};
if (-1 == xioctl(fd_, VIDIOC_QUERYCAP, &caps))
{
perror("Querying Capabilities");
return 1;
}

printf( "Driver Caps:\n"
"  Driver: \"%s\"\n"
"  Card: \"%s\"\n"
"  Bus: \"%s\"\n"
"  Version: %d.%d\n"
"  Capabilities: %08x\n",
caps.driver,
caps.card,
caps.bus_info,
(caps.version>>16)&&0xff,
(caps.version>>24)&&0xff,
caps.capabilities);


struct v4l2_cropcap cropcap = {0};
cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
if (-1 == xioctl (fd_, VIDIOC_CROPCAP, &cropcap))
{
perror("Querying Cropping Capabilities");
return 1;
}

printf( "Camera Cropping:\n"
"  Bounds: %dx%d+%d+%d\n"
"  Default: %dx%d+%d+%d\n"
"  Aspect: %d/%d\n",
cropcap.bounds.width, cropcap.bounds.height, cropcap.bounds.left, cropcap.bounds.top,
cropcap.defrect.width, cropcap.defrect.height, cropcap.defrect.left, cropcap.defrect.top,
cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);

int support_grbg10 = 0;

struct v4l2_fmtdesc fmtdesc = {0};
fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
char fourcc[5] = {0};
char c, e;
printf("  FMT : CE Desc\n--------------------\n");
while (0 == xioctl(fd_, VIDIOC_ENUM_FMT, &fmtdesc))
{
strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
if (fmtdesc.pixelformat == V4L2_PIX_FMT_SGRBG10)
support_grbg10 = 1;
c = fmtdesc.flags & 1? 'C' : ' ';
e = fmtdesc.flags & 2? 'E' : ' ';
printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
fmtdesc.index++;
}

struct v4l2_format fmt = {0};
fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
fmt.fmt.pix.width = cropcap.bounds.width;
fmt.fmt.pix.height = cropcap.bounds.height;
fmt.fmt.pix.pixelformat = fmtdesc.pixelformat;
fmt.fmt.pix.field = V4L2_FIELD_NONE;

if (-1 == xioctl(fd_, VIDIOC_S_FMT, &fmt))
{
perror("Setting Pixel Format");
return 1;
}

strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
printf( "Selected Camera Mode:\n"
"  Width: %d\n"
"  Height: %d\n"
"  PixFmt: %s\n"
"  Field: %d\n",
fmt.fmt.pix.width,
fmt.fmt.pix.height,
fourcc,
fmt.fmt.pix.field);
return 0;
}

int CaptureV4L2::init_mmap()
{
struct v4l2_requestbuffers req = {0};
req.count = 1;
req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
req.memory = V4L2_MEMORY_MMAP;

if (-1 == xioctl(fd_, VIDIOC_REQBUFS, &req))
{
perror("Requesting Buffer");
return 1;
}

struct v4l2_buffer buf = {0};
buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
buf.memory = V4L2_MEMORY_MMAP;
buf.index = 0;
if(-1 == xioctl(fd_, VIDIOC_QUERYBUF, &buf))
{
perror("Querying Buffer");
return 1;
}
buffer_.index = buf.index;
buffer_.length = buf.length;
buffer_.start = static_cast<unsigned char *>(mmap (NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset));

if (MAP_FAILED == buffer_.start) {
printf("Failed mapping device memory");
return 1;
}
printf("Length: %d\nAddress: %p\n", buf.length, buffer_.start);

return 0;
}

int CaptureV4L2::capture_image()
{
struct v4l2_buffer buf = {0};
buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
buf.memory = V4L2_MEMORY_MMAP;
buf.index = 0;
if(-1 == xioctl(fd_, VIDIOC_QBUF, &buf))
{
perror("Query Buffer");
return 1;
}

if(-1 == xioctl(fd_, VIDIOC_STREAMON, &buf.type))
{
perror("Start Capture");
return 1;
}

fd_set fds;
FD_ZERO(&fds);
FD_SET(fd_, &fds);
struct timeval tv = {0};
tv.tv_sec = 2;
int r = select(fd_+1, &fds, NULL, NULL, &tv);
if(-1 == r)
{
perror("Waiting for Frame");
return 1;
}

if(-1 == xioctl(fd_, VIDIOC_DQBUF, &buf))
{
perror("Retrieving Frame");
return 1;
}
printf("Image sucessfully captured\nImage Length: %d\n", buf.bytesused);

return 0;
}

int CaptureV4L2::process_buffer(const void *p){

unsigned char tmp;
unsigned short *src_short = (unsigned short *)p;
unsigned char *dst = (unsigned char *)p;
unsigned short ts;

if(shift_!=0){
#pragma omp for nowait
for (size_t i = 0; i < 1928; i++)
{
for (size_t j = 0; j < 3840; j++)
{
ts = *(src_short++);
tmp = ts >> shift_;
*(dst++) = (unsigned char)tmp;
}
}
}
return 0;
}

int CaptureV4L2::save_img(){
FILE * pFile = fopen ("image.raw", "wb");
if(pFile==NULL){
perror("ERROR: Cannot open output file");
return 1;
} 
fwrite((const int *)buffer_.start , sizeof(char), buffer_.length, pFile);
fclose(pFile);
printf("Raw Image sucessfully saved with name image.raw\n");
return 0;
}

int CaptureV4L2::open_camera(std::string device){
fd_ = open(device.c_str(), O_RDWR);
if (fd_ == -1)
{
perror("Opening video device");
}
printf("Sucessfuly open device %s.\n", device.c_str());
return 0;

}

void CaptureV4L2::close_camera(){
close(fd_);
}

void CaptureV4L2::run(){
if(print_caps())
return;

if(init_mmap())
return;

if(capture_image())
return;

if(process_buffer((const void *)buffer_.start))
return;	

if(save_img())
return;

close_camera();

}

int main(int argc, char** argv)
{
int videoNumber = 0;
int shift=0;
if(argc==2)
videoNumber = *argv[1];

else if(argc==3){
videoNumber = (int)*argv[1]-48;
if(std::string(argv[2]).compare(std::string("RAW8")))
shift = 0;
else if(std::string(argv[2]).compare(std::string("RAW10")))
shift = 2;
else if(std::string(argv[2]).compare(std::string("RAW12")))
shift = 4;
else{
perror("Wrong image format, you can choose RAW8, RAW10 or RAW12\n");
return 1;
}
}

CaptureV4L2 capture_v4l2(std::string("/dev/video"+std::string(std::to_string(videoNumber))).c_str(), shift);

capture_v4l2.run();
return 0;
}
