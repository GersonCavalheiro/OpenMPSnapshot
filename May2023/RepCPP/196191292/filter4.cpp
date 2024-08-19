#include <iostream> 
#include <fstream> 
#include <string> 
#include <math.h> 
#include <sys/time.h> 
#include <omp.h> 

using namespace std;



int main(int argc, char** argv) {


if(argc!=2) { 
cout<<"Lutfen bir adet arguman giriniz! \n";
cout<<"Kullanim bu sekildedir: ./a.out GirisDosyasininAdi.txt \n";
exit(1);
}

int satir, sutun, siralananDizi[9];
int medyan;
int basamak = 0;
int nThreads;


ifstream okunacakDosya(argv[1]);

string cikisDosyasininAdi(argv[1]);

if (cikisDosyasininAdi.length()==19)
cikisDosyasininAdi = cikisDosyasininAdi.insert(15, "_filtered");
else
cikisDosyasininAdi = cikisDosyasininAdi.insert(16, "_filtered");
remove(cikisDosyasininAdi.c_str());
ofstream cikisDosyasi(cikisDosyasininAdi.c_str(), ios::app); 

okunacakDosya>>satir>>sutun;

if(okunacakDosya.fail()){
cout<<"Dosya Bulunamadi!\n";
exit(1);
}



int **girisMatrisiPtr = new int*[satir]; 

for(int i=0; i<satir; i++)  
girisMatrisiPtr[i] = new int[sutun]; 

int **sonucMatrisiPtr = new int*[satir]; 

for(int i=0; i<satir; i++)  
sonucMatrisiPtr[i] = new int[sutun]; 




for(int i=0; i<satir; i++){
for(int j=0; j<sutun; j++){
okunacakDosya>>girisMatrisiPtr[i][j];
}
}




struct timeval currentTime;
double startTime, endTime, elapsedTime;

gettimeofday(&currentTime, NULL);
startTime = currentTime.tv_sec+(currentTime.tv_usec/1000000.0);
#pragma omp parallel private(medyan,siralananDizi)
{

nThreads = omp_get_num_threads();

#pragma omp for collapse(2) 

for(int i=0; i<satir; i++){
for(int j=0; j<sutun; j++) {
if(i==0 || j==0 || i==satir-1 || j==sutun-1) {
sonucMatrisiPtr[i][j] = girisMatrisiPtr[i][j];
}
else {

siralananDizi[0] = girisMatrisiPtr[i-1][j-1];
siralananDizi[1] = girisMatrisiPtr[i-1][j];
siralananDizi[2] = girisMatrisiPtr[i-1][j+1];
siralananDizi[3] = girisMatrisiPtr[i][j-1];
siralananDizi[4] = girisMatrisiPtr[i][j];
siralananDizi[5] = girisMatrisiPtr[i][j+1];
siralananDizi[6] = girisMatrisiPtr[i+1][j-1];
siralananDizi[7] = girisMatrisiPtr[i+1][j];
siralananDizi[8] = girisMatrisiPtr[i+1][j+1];


for(int x=0; x<9; x++) {
for(int y=0; y<8; y++) {
if(siralananDizi[y]>siralananDizi[y+1]) {
int geciciDegisken = siralananDizi[y+1];
siralananDizi[y+1] = siralananDizi[y];
siralananDizi[y] = geciciDegisken;
}
}
}


medyan = siralananDizi[9/2];
sonucMatrisiPtr[i][j] = medyan;

}

} 

} 
} 
for(int i=0; i<satir; i++)
{
for(int j=0; j<sutun; j++) 
{			
basamak=((sonucMatrisiPtr[i][j]<=1)? 1 : log10(sonucMatrisiPtr[i][j])+1);

if(basamak==1) {
cikisDosyasi<<sonucMatrisiPtr[i][j]<<"   ";
}
else if(basamak==2) {
cikisDosyasi<<sonucMatrisiPtr[i][j]<<"  ";
}
else
cikisDosyasi<<sonucMatrisiPtr[i][j]<<" ";
}
cikisDosyasi<<endl;
}

gettimeofday(&currentTime, NULL);
endTime = currentTime.tv_sec+(currentTime.tv_usec/1000000.0);

elapsedTime = endTime-startTime;
printf("%d tane thread kullanarak olculen toplam zaman: %lf ms\n", nThreads, elapsedTime*1000);


for(int i=0; i<satir; i++) 
delete [] girisMatrisiPtr[i]; 

delete [] girisMatrisiPtr; 

for(int i=0; i<satir; i++) 
delete [] sonucMatrisiPtr[i]; 

delete [] sonucMatrisiPtr; 


cikisDosyasi.close();
okunacakDosya.close();

return 0;
}
