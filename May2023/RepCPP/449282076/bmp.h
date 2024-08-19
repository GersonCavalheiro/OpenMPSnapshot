typedef unsigned char BYTE;
typedef unsigned short int WORD;
typedef unsigned int DWORD;
typedef int LONG;

#pragma pack( push, 1 )                     
typedef struct tagBITMAPFILEHEADER {        
WORD bfType;                        
DWORD bfSize;                       
WORD bfReserved1;                   
WORD bfReserved2;                   
DWORD bfOffbytes;                   
} BMPHEADER; 
#pragma pack( pop )

typedef struct tagBITMAPINFOHEADER{         
DWORD biSize;                       
LONG biWidth;                       
LONG biHeight;                      
WORD biPlanes;                      
WORD biBitCount;                    
DWORD biCompression;                
DWORD biSizeImage;                  
LONG biXPelsPerMeter;               
LONG biYPelsPerMeter;               
DWORD biClrUsed;                    
DWORD biClrImportant;               
} BMPINFO; 

typedef struct tagRGBTRIPLE{                
BYTE rgbBlue;                       
BYTE rgbGreen;                      
BYTE rgbRed;                        
} RGBTRIPLE; 
