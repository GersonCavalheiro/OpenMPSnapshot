#include <stdio.h>
#include <stdlib.h>		
#include <string.h>		
#include <math.h>		
#include <complex.h> 		
#include <omp.h>		
#define VERSION 20210102
int NumberOfImages = 0;
typedef enum  {
Potential = 0,
Normal = 1
} RepresentationFunctionType; 
typedef enum  {
linear = 0,
step_linear = 1,
step_sqrt = 2
} GradientType; 
typedef enum  {
no = 0,
average = 1	
} BlendType; 
static unsigned int ixMin = 0;	
static unsigned int ixMax;	
static unsigned int iWidth;	
static unsigned int iyMin = 0;	
static unsigned int iyMax;	
static unsigned int iHeight = 10000;	
static unsigned int iMax;	
double radius ; 
double radius_0 = 2.5;
double zoom ; 
complex double center ; 
double  DisplayAspectRatio  = 1.0; 
double CxMin ;	
double CxMax ;	
double CyMin ;	
double CyMax ;	
int ExampleNumberMax;
double examples[][3] = {
{-0.75, 		0.0, 			2.5}, 
{-1.75, 		0.0, 			0.5},
{-1.77 ,		0.0, 			0.07 }, 
{-1.711065402413374, 	0.0, 			0.008}, 
{-1.711161027105541, 	0.0, 			0.0009}, 
{0.365557777904776,	0.613240370349204, 	0.12}, 
{0.391080345956122, 	0.570677592363374,  	0.01},
{0.296294860929836,	0.017184282646391,	0.001}, 
{-0.170337,		1.06506,  		0.32},
{-0.170337,		1.06506,  		0.064},
{-0.170337,		1.06506,  		0.0128}, 
{-0.170337,		1.06506,  		0.00256},
{-0.170337,		1.06506,  		0.000512}, 
{0.42884,		0.231345, 		0.06}, 
{0.42884,		0.231345, 		0.01}, 
{-1.711638937577389,	0.000449229252155, 	0.000001} 
};
double PixelWidth;	
double PixelHeight;	
double ratio;
static unsigned long int iterMax = 1000000;	
const int iterMax_pot = 4000; 
const int iterMax_normal = 2000; 
const double ER_normal = 1000; 
double ER = 200.0;		
double EscapeRadius=1000000; 
double ER_POT = 100000.0;  
double loger; 
static double TwoPi=2.0*M_PI; 
double MaxFinalRadius;
double MaxImagePotential = 0.0;
double potential_multiplier;
double potential_boundary;
double potential_noisy;
double BoundaryWidth = 3.0; 
double distanceMax; 
unsigned char iColorOfExterior = 250;
unsigned char iColorOfInterior = 127;
unsigned char iColorOfInterior1 = 210;
unsigned char iColorOfInterior2 = 180;
unsigned char iColorOfBoundary = 0;
unsigned char iColorOfUnknown = 30;
unsigned char iColorOfNoise = 255;
static unsigned int iSize;	
double *dData1;
double *dData2;
int iColorSize = 3 ; 
unsigned int iSize_rgb; 
unsigned char *rgbData1; 
unsigned char *rgbData2; 
unsigned char *rgbData3; 
double max(double n1, double n2)
{
return (n1 > n2 ) ? n1 : n2;
}
double min(double n1, double n2)
{
return (n1 < n2 ) ? n1 : n2;
}
double clip(double d){
return (d> 1.0) ? 1.0 : d;
}
double frac(double d){
double fraction = d - ((long)d);
return fraction;
}
double c_arg(complex double z)
{
double arg;
arg = carg(z);
if (arg<0.0) arg+= TwoPi ; 
return arg; 
}
double c_turn(complex double z)
{
double arg;
arg = c_arg(z);
return arg/TwoPi; 
}
double turn( double x, double y){
double t = atan2(y,x);
if ( t<0) t+= TwoPi ;
return t/TwoPi ;
}
double GiveCx ( int ix)
{
return (CxMin + ix * PixelWidth);
}
double GiveCy (int iy) {
return (CyMax - iy * PixelHeight);
}				
complex double GiveC( int ix, int iy){
double Cx = GiveCx(ix);
double Cy = GiveCy(iy);
return Cx + Cy*I;
}
int SetCPlane(complex double Center, double Radius, double a_ratio){
center = Center;
radius = Radius;
CxMin = creal(center) - radius*a_ratio;	
CxMax = creal(center) + radius*a_ratio;	
CyMin = cimag(center) - radius;	
CyMax = cimag(center) + radius;	
return 0;
}
#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr)[0]))
int SetCPlaneFromExamples(const int n, const double a_ratio){
int nMax = LEN(examples);
printf("n = %d \t nMax = %d \n",n,  nMax);
if (n> nMax)
{
SetCPlane(-0.75, 2.5, a_ratio);
fprintf(stderr, " error n>nMax\n");
return 1;
}
complex double c = examples[n][0] + I*examples[n][1];
double r = examples[n][2];
SetCPlane(c, r, a_ratio);
return 0;
}
unsigned int Give_i (unsigned int ix, unsigned int iy)
{
return ix + iy * iWidth;
}
int CheckCPlaneOrientation(unsigned char A[] )
{
double Cx, Cy; 
unsigned i; 
unsigned int ix, iy;		
fprintf(stderr, "compute image CheckOrientation\n");
#pragma omp parallel for schedule(dynamic) private(ix,iy, i, Cx, Cy) shared(A, ixMax , iyMax) 
for (iy = iyMin; iy <= iyMax; ++iy){
fprintf (stderr, " %d from %d \r", iy, iyMax);	
for (ix = ixMin; ix <= ixMax; ++ix){
Cy = GiveCy(iy);
Cx = GiveCx(ix);
i = Give_i(ix, iy); 
if (Cx>0 && Cy>0) A[i]=255-A[i];   
}
}
return 0;
}
double ComputePotential(const complex double c){
double potential = 0.0; 
double s = 0.5;
complex double z = 0.0;
double r;
int iter;
for (iter = 0; iter < iterMax_pot; ++iter){
z = z*z +c; 
s *= 0.5;  
r = cabs(z);
if (r > ER_POT) {break;}
}
if ( iter == iterMax_pot)
{ potential = -1.0; } 
else { 
potential =  s*log2(r); 
potential = fabs(log(potential)); 
}
if (potential >MaxImagePotential ) {MaxImagePotential  = potential;}
return potential;
}
unsigned char ComputePotentialColor(const double potential, const GradientType Gradient){
if ( potential > potential_boundary  ){ return iColorOfBoundary ;}
if ( potential > potential_noisy ) {return iColorOfNoise;} 
double p ; 
switch(Gradient){
case linear: {p = potential; break;}
case step_linear: {p = frac(potential); break;}
case step_sqrt: { p = frac(potential); p = sqrt(p); p = 1.0 - p; break;}
default: {}
}
return 255*p; 
}
double cdot(double complex a, double complex b) {
return creal(a) * creal(b) + cimag(a) * cimag(b);
}
double GiveReflection(double complex C, int iMax, double ER) {
int i = 0; 
double complex Z = 0.0; 
double complex dC = 0.0; 
double reflection = -1.0; 
double h2 = 1.5; 
double angle = 45.0 / 360.0; 
double complex v = cexp(2.0 * angle * M_PI * I); 
double complex u;
for (i = 0; i < iMax; i++) {
dC = 2.0 * dC * Z + 1.0;
Z = Z * Z + C;
if (cabs(Z) > ER) { 
u = Z / dC;
u = u / cabs(u);
reflection = cdot(u, v) + h2;
reflection = reflection / (1.0 + h2); 
if (reflection < 0.0) reflection = 0.0;
break;
}
}
return reflection;
}
unsigned char GiveNormalColor(const int i ) {
double complex c;
int ix;
int iy;
double reflection;
unsigned char g;
iy = i / iWidth;
if (iy>iHeight || iy<0) {fprintf(stderr, " bad iy = %d\n", iy);}
ix = i - iy*iWidth;
if (ix>iWidth || ix<0) {fprintf(stderr, " bad ix = %d\n", ix);}
c = GiveC(ix,iy);
reflection = GiveReflection(c, iterMax_normal, ER_normal);
g = 255*reflection; 
return g; 
}
int ComputePoint_dData (double A[], RepresentationFunctionType  RepresentationFunction, int ix, int iy)
{
int i;			
complex double c;
double d;
i = Give_i (ix, iy);		
c = GiveC(ix,iy);
switch (RepresentationFunction) {
case Potential : {d = ComputePotential(c); break;}
case Normal : { d = GiveReflection(c, iterMax_normal, ER_normal); break;}
default: {}
}
A[i] = d;		
return 0;
}
int Fill_dDataArray (double A[], RepresentationFunctionType  RepresentationFunction)
{
int ix, iy;		
#pragma omp parallel for schedule(dynamic) private(ix,iy) shared(A, ixMax , iyMax)
for (iy = iyMin; iy <= iyMax; ++iy){ 
fprintf (stderr, " %d from %d \r", iy, iyMax);	
for (ix = ixMin; ix <= ixMax; ++ix)
ComputePoint_dData(A, RepresentationFunction, ix, iy);	
}
return 0;
}
unsigned char GiveExteriorColor(const int i, const double D[], const double potential, RepresentationFunctionType RepresentationFunction, GradientType Gradient){
unsigned char g;
switch (RepresentationFunction){
case Potential: { g = ComputePotentialColor(potential, Gradient);  break;}  
case Normal: {g = GiveNormalColor(i); break;}
default: {}
}
return g;
} 
void ComputeAndSaveColor(const int i, const double D[], RepresentationFunctionType RepresentationFunction, GradientType Gradient, unsigned char  C[] ){
int iC = i*iColorSize; 
unsigned char t; 
double d = D[i]; 
if (d<0.0)
{	
C[iC] 	= 0;
C[iC+1] = 0;
C[iC+2] = iColorOfInterior; 
} 
else { 	
t = GiveExteriorColor(i, D, d, RepresentationFunction, Gradient);
C[iC] 	= t;
C[iC+1] = t;
C[iC+2] = t;
} 
}
int Fill_rgbData_from_dData (double D[], RepresentationFunctionType  RepresentationFunction, GradientType Gradient,  unsigned char C[])
{
int i=0;		
fprintf(stderr, "\nFill_rgbData_from_dData\n");
#pragma omp parallel for schedule(dynamic) private(i) shared( D, C, iSize)
for (i = 0; i < iSize; ++i){
ComputeAndSaveColor(i, D, RepresentationFunction, Gradient, C);	
}
return 0;
}
unsigned char GiveBlendedColor(const double c1, const double c2, const BlendType Blend){
unsigned char t;
switch (Blend){
case average: {t = (c1+c2)/2.0; break;}
default: {}
}
return  t;
}
void ComputeAndSaveBlendColor( const unsigned char C1[], const unsigned char C2[], const BlendType Blend, const int i, unsigned char C[]){
unsigned char t; 
int iC = i*iColorSize; 
double c1 = C1[iC];
double c2 = C2[iC];
if ( C1[iC+2] == iColorOfInterior && C1[iC]==0) 
{	
C[iC] 	= 0;
C[iC+1] = 0;
C[iC+2] = iColorOfInterior; 
} 
else { 	
t = GiveBlendedColor( c1 , c2, Blend);
C[iC] 	= t;
C[iC+1] = t;
C[iC+2] = t; 
} 
}
void MakeBlendImage(const unsigned char C1[], const unsigned char C2[], const BlendType Blend, unsigned char C[]){
int i=0;		
fprintf(stderr, "\nFill_rgbData_from_2_dData\n");
#pragma omp parallel for schedule(dynamic) private(i) shared(  C1, C2, C, iSize)
for (i = 0; i < iSize; ++i){
ComputeAndSaveBlendColor( C1, C2, Blend, i, C);
}
}
int Save_PPM( const unsigned char A[], const char* sName, const char* comment, const double radius  )
{
FILE * fp;
char name [100]; 
snprintf(name, sizeof name, "%s_%f", sName, radius); 
char *filename =strcat(name,".ppm");
char long_comment[200];
sprintf (long_comment, "fc(z)=z^2+ c %s", comment);
fp= fopen(filename,"wb"); 
if (!fp ) { fprintf( stderr, "ERROR saving ( cant open) file %s \n", filename); return 1; }
fprintf(fp,"P6\n%d %d\n255\n",  iWidth, iHeight);  
size_t rSize = fwrite(A, sizeof(A[0]), iSize_rgb,  fp);  
fclose(fp); 
if ( rSize == iSize_rgb) 
{
printf ("File %s saved ", filename);
if (long_comment == NULL || strlen (long_comment) == 0)
{printf ("\n"); }
else { printf (". Comment = %s \n", long_comment); }
}
else {printf("wrote %zu elements out of %u requested\n", rSize,  iSize);}
NumberOfImages +=1; 
return 0;
}
int setup ()
{
fprintf (stderr, "setup start\n");
iWidth = iHeight* DisplayAspectRatio;
iSize = iWidth * iHeight;	
iSize_rgb = iSize* iColorSize;
iyMax = iHeight - 1;		
ixMax = iWidth - 1;
iMax = iSize - 1;		
dData1 = malloc (iSize * sizeof (double));
dData2 = malloc (iSize * sizeof (double));
rgbData1 =  malloc (iSize_rgb * sizeof (unsigned char));
rgbData2 =  malloc (iSize_rgb * sizeof (unsigned char));
rgbData3 =  malloc (iSize_rgb * sizeof (unsigned char));
if (dData1 == NULL || dData2 == NULL  || rgbData1 == NULL || rgbData2 == NULL || rgbData3 == NULL){
fprintf (stderr, " Could not allocate memory");
return 1;
}
ExampleNumberMax = LEN ( examples);
fprintf (stderr," end of setup \n");
return 0;
} 
int local_setup(int example_number)
{
SetCPlaneFromExamples(example_number, DisplayAspectRatio );
potential_multiplier = 1+log10(radius_0/radius); 
switch(example_number) {
case 4:	; {
iColorOfBoundary = iColorOfNoise; 
potential_boundary = 26.0* potential_multiplier;
potential_noisy = 24.0 * potential_multiplier;
break;
}
case 15 : {
iColorOfBoundary = 255;
iColorOfNoise = 180 ; 
potential_boundary = 250.0; 
potential_noisy = 220.0; 
break;
}
default : {
potential_boundary = 25.0* potential_multiplier;
potential_noisy = 10.0 * potential_multiplier;
}
} 
PixelWidth = (CxMax - CxMin) / ixMax;	
PixelHeight = (CyMax - CyMin) / iyMax;
ratio = ((CxMax - CxMin) / (CyMax - CyMin)) / ((double) iWidth / (double) iHeight);	
return 0; 
}  
int PrintInfoAboutProgam(int example_number)
{
printf ("Numerical approximation of M set for fc(z)= z^2 + c \n");
printf ("Image Width = %f in world coordinate\n", CxMax - CxMin);
printf ("PixelWidth = %f  = %.16f * Image Width\n", PixelWidth, PixelWidth/ (CxMax - CxMin));
printf ("example number = %d \n", example_number);
printf ("plane center c = ( %.16f ; %.16f ) \n", creal (center), cimag (center));
printf ("plane radius = %.16f \n", radius);
printf ("plane zoom = 1/radius = %.16f \n", 1.0/radius);
printf("\n\n potential \n");
printf("\t iterMax_pot = %d \n", iterMax_pot);
printf("\t ER_POT = %.16f \n" , ER_POT  ); 
printf ("\t MaxImagePotential  = %.16f \n", MaxImagePotential );
printf ("\t plane  potential_multiplier = %.16f \n", potential_multiplier );
printf("\t black area : potential > potential_boundary =  %.16f  = %.16f * MaxImagePotential \n",potential_boundary, potential_boundary / MaxImagePotential);
printf("\t white area : potential > potential_noisy  = %.16f  = %.16f * MaxImagePotential \n", potential_noisy, potential_noisy / MaxImagePotential);
printf("\n");
printf ("Maximal number of iterations = iterMax = %ld \n", iterMax);
printf("Number of pgm images = %d \n", NumberOfImages);	
printf ("ratio of image  = %f ; it should be 1.000 ...\n", ratio);
printf("gcc version: %d.%d.%d\n",__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__); 
return 0;
}
int MakeExampleImages(int example_number){
local_setup(example_number);
Fill_dDataArray(dData1, Potential);
Fill_rgbData_from_dData (dData1, Potential, step_sqrt, rgbData1);
Fill_dDataArray(dData2, Normal);
Fill_rgbData_from_dData (dData2, Normal, linear, rgbData2);
Save_PPM(rgbData2, "normal_linear", "normal_linear", radius);
MakeBlendImage(rgbData1, rgbData2, average, rgbData3);
Save_PPM(rgbData3, "average", "average blend = (potential + normal)/2", radius);
PrintInfoAboutProgam(example_number);
return 0;
}
int end(){
fprintf (stderr," allways free memory (deallocate )  to avoid memory leaks \n"); 
free(dData1);
free(dData2);
free(rgbData1);
free(rgbData2);
free(rgbData3);
return 0;
}
int main () {
setup ();
int example_number = 15;
{
MakeExampleImages(example_number);
}
end();
return 0;
}
