#include <omp.h>
#include <X11/Xlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
typedef struct complexType{
double real, imag;
}Compl;
int main (void) {
Display *display;
Window window;		
int screen;			
display = XOpenDisplay(NULL);
if(display == NULL) {
fprintf(stderr, "cannot open display\n");
return 0;
}
screen = DefaultScreen(display);
int width = 400;
int height = 400;
int x = 0;
int y = 0;
int border_width = 0;
window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y,
width, height, border_width,
BlackPixel(display, screen), WhitePixel(display, screen));
GC gc;
XGCValues values;
long valuemask = 0;
gc = XCreateGC(display, window, valuemask, &values);
XSetForeground (display, gc, BlackPixel (display, screen));
XSetBackground(display, gc, 0X0000FF00);
XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
XMapWindow(display, window);
XSync(display, 0);
struct timespec t1, t2;
int result;
omp_lock_t lock;
omp_init_lock(&lock);
int i, j;
clock_gettime(CLOCK_REALTIME, &t1);
#pragma omp parallel for private(j) schedule(dynamic, 2)
for (i=0; i<width; i++) {
Compl z, c;
int repeats;
double temp, lengthsq;
for (j=0; j<height; j++) {
z.real = 0.0;
z.imag = 0.0;
c.real = -2.0 + (double)i * (4.0/(double)width);
c.imag = -2.0 + (double)j * (4.0/(double)height);
repeats = 0;
lengthsq = 0.0;
while (repeats < 100000 && lengthsq < 4.0) {
temp = z.real*z.real - z.imag*z.imag + c.real;
z.imag = 2*z.real*z.imag + c.imag;
z.real = temp;
lengthsq = z.real*z.real + z.imag*z.imag;
repeats++;
}
omp_set_lock(&lock);
XSetForeground(display, gc, 1024*1024*(repeats%256));
XDrawPoint(display, window, gc, i, j);
omp_unset_lock(&lock);
}
}
clock_gettime(CLOCK_REALTIME, &t2);
double timedif = 1000000*(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1000;
printf("It took %.5lf seconds to finish dynamic OpenMP calculation.\n", timedif/1000000);
printf("Going into sleep...\n");
XFlush(display);
sleep(3);
return 0;
}
