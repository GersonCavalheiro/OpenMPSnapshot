



#ifndef __BIG_ENDIAN__
#define LITTLE_MACHINE
#endif

#ifdef WIN32
#define assert ASSERT
#else
#include <assert.h>
#endif
#ifdef _MSC_VER
#pragma warning( disable : 4267 )
#define strtok_r strtok_s
#endif

#ifdef WIN32
#include <direct.h>
#endif

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include <vector>
#include <algorithm>

#include "plylib.h"
using namespace std;
namespace vcg{
namespace ply{
typedef unsigned short ushort;
typedef unsigned long ulong;
typedef unsigned char uchar;
typedef unsigned int uint;

#define XFILE  FILE
#define pb_fclose fclose
#define pb_fopen  fopen
#define pb_fgets(s,n,f)  fgets(s,n,f)
#define pb_fread(b,s,n,f) fread(b,s,n,f)


#define pb_mkdir(n)  _mkdir(n)
#define pb_access _access
#define pb_stat   _stat
#define pb_open   _open
#define pb_close  _close




int ReadBin  ( XFILE * fp, const PlyProperty * pr, void * mem, int fmt );
int ReadAscii( XFILE * fp, const PlyProperty * pr, void * mem, int fmt );


const char * ::vcg::ply::PlyFile::typenames[9]=
{
"none",
"char",
"short",
"int",
"uchar",
"ushort",
"uint",
"float",
"double"
};
const char * PlyFile::newtypenames[9]=
{
"none",
"int8",
"int16",
"int32",
"uint8",
"uint16",
"uint32",
"float32",
"float64"
};

static int TypeSize[] = {
0, 1, 2, 4, 1, 2, 4, 4, 8
};

size_t PropDescriptor::memtypesize() const {return TypeSize[memtype1];}
size_t PropDescriptor::stotypesize() const {return TypeSize[stotype1];}
const char *PropDescriptor::memtypename() const {return PlyFile::typenames[memtype1];}
const char *PropDescriptor::stotypename() const {return PlyFile::typenames[stotype1];}

static char CrossType[9][9]=
{
{0,0,0,0,0,0,0,0,0},
{0,1,1,1,1,1,1,0,0},
{0,0,1,1,0,1,1,0,0},
{0,0,0,1,0,0,1,0,0},
{0,1,1,1,1,1,1,0,0},
{0,0,1,1,0,1,1,0,0},
{0,0,0,1,0,0,1,0,0},
{0,0,0,0,0,0,0,1,1},
{0,0,0,0,0,0,0,1,1}
};



static inline void SwapShort( ushort * s )
{
assert(s);
*s = ushort( (int(*s)>>8) | (int(*s)<<8) );
}


static inline void SwapInt( uint * x )
{
assert(x);
*x = 
( ((*x)>>24) & 0x000000FF ) |
( ((*x)>> 8) & 0x0000FF00 ) |
( ((*x)<< 8) & 0x00FF0000 ) |
( ((*x)<<24) & 0xFF000000 ) ;
}


static inline void SwapDouble( double *  )
{
assert(0);
}



static inline int ReadCharB( XFILE * fp, char * c, int  )
{
assert(fp);
assert(c);

return pb_fread(c,1,1,fp);
}


static inline int ReadShortB( XFILE * fp, short * s, int format )
{
assert(fp);
assert(s);

int r;
r = pb_fread(s,sizeof(short),1,fp);

#ifdef LITTLE_MACHINE
if(format==F_BINBIG)
#else
if(format==F_BINLITTLE)
#endif
SwapShort((ushort *)s);

return r;
}


static inline int ReadIntB( XFILE * fp, int * i, int format )
{
assert(fp);
assert(i);

int r;
r = pb_fread(i,sizeof(int),1,fp);

#ifdef LITTLE_MACHINE
if(format==F_BINBIG)
#else
if(format==F_BINLITTLE)
#endif
SwapInt((uint *)i);

return r;
}


static inline int ReadUCharB( XFILE * fp, uchar * uc, int  )
{
assert(fp);
assert(uc);

return pb_fread(uc,1,1,fp);
}


static inline int ReadUShortB( XFILE * fp, ushort * us, int format )
{
assert(fp);
assert(us);

int r;
r = pb_fread(us,sizeof(ushort),1,fp);

#ifdef LITTLE_MACHINE
if(format==F_BINBIG)
#else
if(format==F_BINLITTLE)
#endif
SwapShort(us);

return r;
}


static inline int ReadUIntB( XFILE * fp, uint * ui, int format )
{
assert(fp);
assert(ui);

int r;
r = pb_fread(ui,sizeof(uint),1,fp);

#ifdef LITTLE_MACHINE
if(format==F_BINBIG)
#else
if(format==F_BINLITTLE)
#endif
SwapInt(ui);

return r;
}


static inline int ReadFloatB( XFILE * fp, float * f, int format )
{
assert(fp);
assert(f);

int r;
r = pb_fread(f,sizeof(float),1,fp);

#ifdef LITTLE_MACHINE
if(format==F_BINBIG)
#else
if(format==F_BINLITTLE)
#endif
SwapInt((uint *)f);

return r;
}


static inline int ReadDoubleB( XFILE * fp, double * d, int format )
{
assert(fp);
assert(d);

int r;
r = pb_fread(d,sizeof(double),1,fp);

#ifdef LITTLE_MACHINE
if(format==F_BINBIG)
#else
if(format==F_BINLITTLE)
#endif
SwapDouble(d);

return r;
}



static void InitSBuffer()
{
}

static inline int ReadInt( XFILE * fp, int & t )
{

int r =  fscanf(fp,"%d",&t);
if(r==EOF) r = 0;
return r;
}


static inline int ReadUInt( XFILE * fp, unsigned int & t )
{

int r =  fscanf(fp,"%u",&t);
if(r==EOF) r = 0;
return r;
}


static inline int ReadFloat( XFILE * fp, float & f )
{

int r = fscanf(fp,"%f",&f);
if(r==EOF) r = 0;
return r;
}

static inline int ReadDouble( XFILE * fp, double & d )
{

int r = fscanf(fp,"%lf",&d);
if(r==EOF) r = 0;
return r;
}



static inline int ReadCharA( XFILE * fp, char * c )
{
assert(fp);
assert(c);

int r,t;
r = ReadInt(fp,t);
*c = (char)t;
return r;
}


static inline int ReadShortA( XFILE * fp, short * s )
{
assert(fp);
assert(s);

int r,t;
r = ReadInt(fp,t);
*s = (short)t;
return r;
}


static inline int ReadIntA( XFILE * fp, int * i )
{
assert(fp);
assert(i);

return ReadInt(fp,*i);
}

static inline int ReadUCharA( XFILE * fp, uchar * uc )
{
assert(fp);
assert(uc);

int r;
uint t;
r = ReadUInt(fp,t);
*uc = (uchar)t;
return r;
}


static inline int ReadUShortA( XFILE * fp, ushort * us )
{
assert(fp);
assert(us);

int r;
uint t;
r = ReadUInt(fp,t);
*us = (ushort)t;
return r;
}


static inline int ReadUIntA( XFILE * fp, uint * ui )
{
assert(fp);
assert(ui);

return ReadUInt(fp,*ui);
}

static inline int ReadFloatA( XFILE * fp, float * f )
{
assert(fp);
assert(f);

return ReadFloat(fp,*f);
}


static inline int ReadDoubleA( XFILE * fp, double * d )
{
assert(fp);
assert(d);

return ReadDouble(fp,*d);
}




static inline void StoreInt( void * mem, const int tm, const int val )
{
assert(mem);

switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )val; break;
case T_SHORT:	*(short  *)mem = (short )val; break;
case T_INT:		*(int    *)mem = (int   )val; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )val; break;
case T_USHORT:	*(ushort *)mem = (ushort)val; break;
case T_UINT:	*(uint   *)mem = (uint  )val; break;
case T_FLOAT:	*(float  *)mem = (float )val; break;
case T_DOUBLE:	*(double *)mem = (double)val; break;
default: assert(0);
}
}


static inline int SkipScalarA( XFILE * fp, const int tf )
{
int t;
float f;

assert(fp);

switch(tf)
{
case T_CHAR:
case T_SHORT:
case T_INT:
case T_UCHAR:
case T_USHORT:
case T_UINT:
return ReadInt(fp,t);
case T_FLOAT:
case T_DOUBLE:
return ReadFloat(fp,f);
default:
assert(0);
return 0;
}
}



static inline int SkipScalarB( XFILE * fp, const int tf)
{
char dummy[8];

assert(fp);
return pb_fread(dummy,1,TypeSize[tf],fp);
}

static int ReadScalarB( XFILE * fp, void * mem, const int tf, const int tm, int fmt )
{
char		ch;
short	sh;
int		in;
uchar	uc;
ushort	us;
uint		ui;
float	fl;
double	dd;

int r = 0;

switch(tf)
{
case T_CHAR:	
r = ReadCharB(fp,&ch,fmt);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )ch; break;
case T_SHORT:	*(short  *)mem = (short )ch; break;
case T_INT:		*(int    *)mem = (int   )ch; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )ch; break;
case T_USHORT:	*(ushort *)mem = (ushort)ch; break;
case T_UINT:	*(uint   *)mem = (uint  )ch; break;
case T_FLOAT:	*(float  *)mem = (float )ch; break;
case T_DOUBLE:	*(double *)mem = (double)ch; break;
default: assert(0);
}
break;
case T_SHORT:	
r = ReadShortB(fp,&sh,fmt);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )sh; break;
case T_SHORT:	*(short  *)mem = (short )sh; break;
case T_INT:		*(int    *)mem = (int   )sh; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )sh; break;
case T_USHORT:	*(ushort *)mem = (ushort)sh; break;
case T_UINT:	*(uint   *)mem = (uint  )sh; break;
case T_FLOAT:	*(float  *)mem = (float )sh; break;
case T_DOUBLE:	*(double *)mem = (double)sh; break;
default: assert(0);
}
break;
case T_INT:		
r = ReadIntB(fp,&in,fmt);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )in; break;
case T_SHORT:	*(short  *)mem = (short )in; break;
case T_INT:		*(int    *)mem = (int   )in; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )in; break;
case T_USHORT:	*(ushort *)mem = (ushort)in; break;
case T_UINT:	*(uint   *)mem = (uint  )in; break;
case T_FLOAT:	*(float  *)mem = (float )in; break;
case T_DOUBLE:	*(double *)mem = (double)in; break;
default: assert(0);
}
break;
case T_UCHAR:	
r = ReadUCharB(fp,&uc,fmt);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )uc; break;
case T_SHORT:	*(short  *)mem = (short )uc; break;
case T_INT:		*(int    *)mem = (int   )uc; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )uc; break;
case T_USHORT:	*(ushort *)mem = (ushort)uc; break;
case T_UINT:	*(uint   *)mem = (uint  )uc; break;
case T_FLOAT:	*(float  *)mem = (float )uc; break;
case T_DOUBLE:	*(double *)mem = (double)uc; break;
default: assert(0);
}
break;
case T_USHORT:	
r = ReadUShortB(fp,&us,fmt);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )us; break;
case T_SHORT:	*(short  *)mem = (short )us; break;
case T_INT:		*(int    *)mem = (int   )us; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )us; break;
case T_USHORT:	*(ushort *)mem = (ushort)us; break;
case T_UINT:	*(uint   *)mem = (uint  )us; break;
case T_FLOAT:	*(float  *)mem = (float )us; break;
case T_DOUBLE:	*(double *)mem = (double)us; break;
default: assert(0);
}
break;
case T_UINT:	
r = ReadUIntB(fp,&ui,fmt);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )ui; break;
case T_SHORT:	*(short  *)mem = (short )ui; break;
case T_INT:		*(int    *)mem = (int   )ui; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )ui; break;
case T_USHORT:	*(ushort *)mem = (ushort)ui; break;
case T_UINT:	*(uint   *)mem = (uint  )ui; break;
case T_FLOAT:	*(float  *)mem = (float )ui; break;
case T_DOUBLE:	*(double *)mem = (double)ui; break;
default: assert(0);
}
break;
case T_FLOAT:	
r = ReadFloatB(fp,&fl,fmt);
switch(tm)
{
case T_FLOAT:  *(float  *)mem = fl; break;
case T_DOUBLE: *(double *)mem = fl; break;
default: assert(0);
}
break;
case T_DOUBLE:	
r = ReadDoubleB(fp,&dd,fmt);
switch(tm)
{
case T_FLOAT:  *(float  *)mem = (float)dd; break;
case T_DOUBLE: *(double *)mem = dd;        break;
default: assert(0);
}
break;
default:
assert(0);
}

return r;
}


static int ReadScalarA( XFILE * fp, void * mem, const int tf, const int tm )
{
char		ch;
short	sh;
int		in;
uchar	uc;
ushort	us;
uint		ui;
float	fl;
double	dd;

int r = 0;

switch(tf)
{
case T_CHAR:	
r = ReadCharA(fp,&ch);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )ch; break;
case T_SHORT:	*(short  *)mem = (short )ch; break;
case T_INT:		*(int    *)mem = (int   )ch; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )ch; break;
case T_USHORT:	*(ushort *)mem = (ushort)ch; break;
case T_UINT:	*(uint   *)mem = (uint  )ch; break;
case T_FLOAT:	*(float  *)mem = (float )ch; break;
case T_DOUBLE:	*(double *)mem = (double)ch; break;
default: assert(0);
}
break;
case T_SHORT:	
r = ReadShortA(fp,&sh);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )sh; break;
case T_SHORT:	*(short  *)mem = (short )sh; break;
case T_INT:		*(int    *)mem = (int   )sh; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )sh; break;
case T_USHORT:	*(ushort *)mem = (ushort)sh; break;
case T_UINT:	*(uint   *)mem = (uint  )sh; break;
case T_FLOAT:	*(float  *)mem = (float )sh; break;
case T_DOUBLE:	*(double *)mem = (double)sh; break;
default: assert(0);
}
break;
case T_INT:		
r = ReadIntA(fp,&in);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )in; break;
case T_SHORT:	*(short  *)mem = (short )in; break;
case T_INT:		*(int    *)mem = (int   )in; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )in; break;
case T_USHORT:	*(ushort *)mem = (ushort)in; break;
case T_UINT:	*(uint   *)mem = (uint  )in; break;
case T_FLOAT:	*(float  *)mem = (float )in; break;
case T_DOUBLE:	*(double *)mem = (double)in; break;
default: assert(0);
}
break;
case T_UCHAR:	
r = ReadUCharA(fp,&uc);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )uc; break;
case T_SHORT:	*(short  *)mem = (short )uc; break;
case T_INT:		*(int    *)mem = (int   )uc; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )uc; break;
case T_USHORT:	*(ushort *)mem = (ushort)uc; break;
case T_UINT:	*(uint   *)mem = (uint  )uc; break;
case T_FLOAT:	*(float  *)mem = (float )uc; break;
case T_DOUBLE:	*(double *)mem = (double)uc; break;
default: assert(0);
}
break;
case T_USHORT:	
r = ReadUShortA(fp,&us);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )us; break;
case T_SHORT:	*(short  *)mem = (short )us; break;
case T_INT:		*(int    *)mem = (int   )us; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )us; break;
case T_USHORT:	*(ushort *)mem = (ushort)us; break;
case T_UINT:	*(uint   *)mem = (uint  )us; break;
case T_FLOAT:	*(float  *)mem = (float )us; break;
case T_DOUBLE:	*(double *)mem = (double)us; break;
default: assert(0);
}
break;
case T_UINT:	
r = ReadUIntA(fp,&ui);
switch(tm)
{
case T_CHAR:	*(char   *)mem = (char  )ui; break;
case T_SHORT:	*(short  *)mem = (short )ui; break;
case T_INT:		*(int    *)mem = (int   )ui; break;
case T_UCHAR:	*(uchar  *)mem = (uchar )ui; break;
case T_USHORT:	*(ushort *)mem = (ushort)ui; break;
case T_UINT:	*(uint   *)mem = (uint  )ui; break;
case T_FLOAT:	*(float  *)mem = (float )ui; break;
case T_DOUBLE:	*(double *)mem = (double)ui; break;
default: assert(0);
}
break;
case T_FLOAT:	
r = ReadFloatA(fp,&fl);
switch(tm)
{
case T_FLOAT:  *(float  *)mem = fl; break;
case T_DOUBLE: *(double *)mem = fl; break;
default: assert(0);
}
break;
case T_DOUBLE:	
r = ReadDoubleA(fp,&dd);
switch(tm)
{
case T_FLOAT:  *(float  *)mem = (float)dd; break;
case T_DOUBLE: *(double *)mem = dd;        break;
default: assert(0);
}
break;
default:
assert(0);
}

return r;
}



void PlyElement::AddProp( const char * na, int ti, int isl, int t2 )
{
assert(na);
assert(ti>0);
assert(ti<T_MAXTYPE);
assert( t2>0 || (t2==0 && isl==0));
assert(t2<T_MAXTYPE);

props.push_back( PlyProperty (na,ti,isl,t2) );
}

PlyProperty * PlyElement::FindProp( const char * na )
{
assert(na);
vector<PlyProperty>::iterator i;

for(i=props.begin();i!=props.end();++i)
if( i->name == na )
return &*i;
return 0;
}

int PlyElement::AddToRead(
const char * propname,
int	stotype1,
int memtype1,
size_t offset1,
int islist,
int alloclist,
int stotype2,
int memtype2,
size_t offset2
)	
{
assert(propname);

PlyProperty * p = FindProp(propname);
if(p==0)
{
return E_PROPNOTFOUND;
}

if( stotype1<1 || stotype1>=T_MAXTYPE ||
memtype1<1 || memtype1>=T_MAXTYPE )
{
return E_BADTYPE;
}

if( islist && (stotype2<1 || stotype2>=T_MAXTYPE ||
memtype2<1 || memtype2>=T_MAXTYPE ) )
{
return E_BADTYPE;
}

if( islist!= p->islist || stotype1 != p->tipo ||
( islist && stotype2!=p->tipoindex) )
{
return E_INCOMPATIBLETYPE;
}

if( !CrossType[p->tipo][stotype1] ||
(islist && !CrossType[p->tipoindex][stotype2] ) )
{
return E_BADCAST;
}

p->bestored = 1;

p->desc.stotype1	= stotype1;
p->desc.memtype1	= memtype1;
p->desc.offset1		= offset1;
p->desc.islist		= islist;
p->desc.alloclist	= alloclist;
p->desc.stotype2	= stotype2;
p->desc.memtype2	= memtype2;
p->desc.offset2		= offset2;


return E_NOERROR;
}




PlyFile::PlyFile( void )
{
gzfp		= 0;
version		= 0.0f;
error		= E_NOERROR;
format		= F_UNSPECIFIED;
cure		= 0;
ReadCB		= 0;
InitSBuffer();
}

PlyFile::~PlyFile( void )
{
Destroy();
}

int PlyFile::Open( const char * filename, int mode )
{
if(filename==0 || (mode!=MODE_READ && mode!=MODE_WRITE) )
{
error = E_CANTOPEN;
return -1;
}
if(mode==MODE_READ)
return OpenRead(filename);
else
return OpenWrite(filename);
}

void PlyFile::Destroy( void )
{
if(gzfp!=0)
{
pb_fclose(gzfp);
gzfp = 0;
}

ReadCB = 0;
}

int PlyFile::OpenRead( const char * filename )
{

const char * SEP			= " \t\n\r";
const char * HEADER		= "ply";
const char * FORMAT		= "format";
const char * TASCII		= "ascii";
const char * TBINBIG		= "binary_big_endian";
const char * TBINLITTLE	= "binary_little_endian";
const char * COMMENT		= "comment";
const char * OBJ_INFO	= "obj_info";
const char * ELEMENT		= "element";
const char * PROPERTY	= "property";
const char * ENDHEADER	= "end_header";
const char * LIST		= "list";

const int MAXB = 512;
char buf[MAXB];
char * token;
PlyElement * curelement;


Destroy();

gzfp = pb_fopen(filename,"rb");
if(gzfp==0)
{
error = E_CANTOPEN;
goto error;
}

header.clear();
header.reserve(1536);

if( pb_fgets(buf,MAXB-1,gzfp)==0 )
{
error = E_UNESPECTEDEOF;
goto error;
}
header.append(buf);

if( strncmp(buf,HEADER,strlen(HEADER)) )
{
error = E_NOTHEADER;
goto error;
}



if( pb_fgets(buf,MAXB-1,gzfp)==0 )
{
error = E_UNESPECTEDEOF;
goto error;
}
header.append(buf);
#ifdef __MINGW32__
token = strtok(buf,SEP);
#else
char *tokenPtr;
token = strtok_r(buf,SEP,&tokenPtr);
#endif
if(token==0)
{
error = E_UNESPECTEDEOF;
goto error;
}

while (!strcmp(token, COMMENT))
{
comments.push_back(string(token + strlen(token) + 1));
if (pb_fgets(buf, MAXB - 1, gzfp) == 0)
{
error = E_UNESPECTEDEOF;
goto error;
}
header.append(buf);

#ifdef __MINGW32__
token = strtok(buf, SEP);
#else
token = strtok_r(buf, SEP, &tokenPtr);
#endif
}

if( strcmp(token,FORMAT) )
{
error = E_NOFORMAT;
goto error;
}
#ifdef __MINGW32__
token = strtok(0,SEP);
#else
token = strtok_r(0,SEP,&tokenPtr);
#endif
if(token==0)
{
error = E_UNESPECTEDEOF;
goto error;
}
if( !strcmp(token,TASCII) )
format = F_ASCII;
else if( !strcmp(token,TBINBIG) )
format = F_BINBIG;
else if( !strcmp(token,TBINLITTLE) )
format = F_BINLITTLE;
else
{
error = E_NOFORMAT;
goto error;
}
#ifdef __MINGW32__
token = strtok(0,SEP);
#else
token = strtok_r(0,SEP,&tokenPtr);
#endif
if(token==0)
{
error = E_UNESPECTEDEOF;
goto error;
}
version = float(atof(token));
/ )
{
assert(0);
return -1;
}



int PlyFile::FindType( const char * name ) const
{
int i;
assert(name);

for(i=1;i<9;++i)
if( !strcmp(name,typenames[i]) || !strcmp(name,newtypenames[i]))
return i;
return -1;
}

PlyElement * PlyFile::FindElement( const char * na ) 
{
assert(na);
vector<PlyElement>::iterator i;

for(i=elements.begin();i!=elements.end();++i)
if( i->name == na )
return &*i;
return 0;
}

int PlyFile::AddToRead(
const char * elemname,
const char * propname,
int	stotype1,
int memtype1,
size_t offset1,
int islist,
int alloclist,
int stotype2,
int memtype2,
size_t offset2
)	
{
assert(elemname);
PlyElement * e = FindElement(elemname);
if(e==0)
{
error = E_ELEMNOTFOUND;
return -1;
}

int r = e->AddToRead(propname,stotype1,memtype1,offset1,islist,
alloclist,stotype2,memtype2,offset2 );

if(r==E_NOERROR)
return 0;
else
{
error = r;
return -1;
}
}

const char * PlyFile::ElemName( int i )
{
if(i<0 || i>=int(elements.size()))
return 0;
else
return elements[i].name.c_str();
}

int PlyFile::ElemNumber( int i ) const
{
if(i<0 || i>=int(elements.size()))
return 0;
else
return elements[i].number;
}


static bool cb_skip_bin1( GZFILE fp, void * , PropDescriptor *  )
{
char dummy[1];

assert(fp);
return pb_fread(dummy,1,1,fp)!=0;
}

static bool cb_skip_bin2( GZFILE fp, void * , PropDescriptor *  )
{
char dummy[2];

assert(fp);
return pb_fread(dummy,1,2,fp)!=0;
}

static bool cb_skip_bin4( GZFILE fp, void * , PropDescriptor *  )
{
char dummy[4];

assert(fp);
return pb_fread(dummy,1,4,fp)!=0;
}

static bool cb_skip_bin8( GZFILE fp, void * , PropDescriptor *  )
{
char dummy[8];

assert(fp);
return pb_fread(dummy,1,8,fp)!=0;
}

static bool cb_skip_float_ascii( GZFILE fp, void * , PropDescriptor *  )
{
float dummy;

assert(fp);
return fscanf(fp,"%f",&dummy)!=EOF;
}

static bool cb_skip_int_ascii( GZFILE fp, void * , PropDescriptor *  )
{
int dummy;

assert(fp);
return fscanf(fp,"%d",&dummy)!=EOF;
}


static bool cb_read_chch( GZFILE fp, void * mem, PropDescriptor * d )
{
return pb_fread( ((char *)mem)+d->offset1,1,1,fp)!=0;
}

static bool cb_read_chsh( GZFILE fp, void * mem, PropDescriptor * d )
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
return true;

*(short *)(((char *)mem)+d->offset1) = short(c);
return true;
}

static bool cb_read_chin( GZFILE fp, void * mem, PropDescriptor * d )
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
return true;

*(int *)(((char *)mem)+d->offset1) = int(c);
return true;
}

static bool cb_read_chuc( GZFILE fp, void * mem, PropDescriptor * d )
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
return true;

*(uchar *)(((char *)mem)+d->offset1) = uchar(c);
return true;
}

static bool cb_read_chus( GZFILE fp, void * mem, PropDescriptor * d )
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
return true;

*(ushort *)(((char *)mem)+d->offset1) = ushort(c);
return true;
}

static bool cb_read_chui( GZFILE fp, void * mem, PropDescriptor * d )
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
return true;

*(uint *)(((char *)mem)+d->offset1) = uint(c);
return true;
}

static bool cb_read_chfl( GZFILE fp, void * mem, PropDescriptor * d )
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
return true;

*(float *)(((char *)mem)+d->offset1) = float(c);
return true;
}

static bool cb_read_chdo( GZFILE fp, void * mem, PropDescriptor * d )
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
return true;

*(double *)(((char *)mem)+d->offset1) = double(c);
return true;
}



static bool cb_read_shch( GZFILE fp, void * mem, PropDescriptor * d )
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;

*(char *)(((char *)mem)+d->offset1) = char(c);
return true;
}

static bool cb_read_shsh( GZFILE fp, void * mem, PropDescriptor * d )
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;

*(short *)(((char *)mem)+d->offset1) = short(c);
return true;
}

static bool cb_read_shin( GZFILE fp, void * mem, PropDescriptor * d )
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;

*(int *)(((char *)mem)+d->offset1) = int(c);
return true;
}

static bool cb_read_shuc( GZFILE fp, void * mem, PropDescriptor * d )
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;

*(uchar *)(((char *)mem)+d->offset1) = uchar(c);
return true;
}

static bool cb_read_shus( GZFILE fp, void * mem, PropDescriptor * d )
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;

*(ushort *)(((char *)mem)+d->offset1) = ushort(c);
return true;
}

static bool cb_read_shui( GZFILE fp, void * mem, PropDescriptor * d )
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;

*(uint *)(((char *)mem)+d->offset1) = uint(c);
return true;
}

static bool cb_read_shfl( GZFILE fp, void * mem, PropDescriptor * d )
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;

*(float *)(((char *)mem)+d->offset1) = float(c);
return true;
}

static bool cb_read_shdo( GZFILE fp, void * mem, PropDescriptor * d )
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;

*(double *)(((char *)mem)+d->offset1) = double(c);
return true;
}




static bool cb_read_inch( GZFILE fp, void * mem, PropDescriptor * d )
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;

*(char *)(((char *)mem)+d->offset1) = char(c);
return true;
}

static bool cb_read_insh( GZFILE fp, void * mem, PropDescriptor * d )
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;

*(short *)(((char *)mem)+d->offset1) = short(c);
return true;
}

static bool cb_read_inin( GZFILE fp, void * mem, PropDescriptor * d )
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;

*(int *)(((char *)mem)+d->offset1) = int(c);
return true;
}

static bool cb_read_inuc( GZFILE fp, void * mem, PropDescriptor * d )
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;

*(uchar *)(((char *)mem)+d->offset1) = uchar(c);
return true;
}

static bool cb_read_inus( GZFILE fp, void * mem, PropDescriptor * d )
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;

*(ushort *)(((char *)mem)+d->offset1) = ushort(c);
return true;
}

static bool cb_read_inui( GZFILE fp, void * mem, PropDescriptor * d )
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;

*(uint *)(((char *)mem)+d->offset1) = uint(c);
return true;
}

static bool cb_read_infl( GZFILE fp, void * mem, PropDescriptor * d )
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;

*(float *)(((char *)mem)+d->offset1) = float(c);
return true;
}

static bool cb_read_indo( GZFILE fp, void * mem, PropDescriptor * d )
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;

*(double *)(((char *)mem)+d->offset1) = double(c);
return true;
}



static bool cb_read_ucch( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;

*(char *)(((char *)mem)+d->offset1) = char(c);
return true;
}

static bool cb_read_ucsh( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;

*(short *)(((char *)mem)+d->offset1) = short(c);
return true;
}

static bool cb_read_ucin( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;

*(int *)(((char *)mem)+d->offset1) = int(c);
return true;
}

static bool cb_read_ucuc( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;

*(uchar *)(((char *)mem)+d->offset1) = uchar(c);
return true;
}

static bool cb_read_ucus( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;

*(ushort *)(((char *)mem)+d->offset1) = ushort(c);
return true;
}

static bool cb_read_ucui( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;

*(uint *)(((char *)mem)+d->offset1) = uint(c);
return true;
}

static bool cb_read_ucfl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;

*(float *)(((char *)mem)+d->offset1) = float(c);
return true;
}

static bool cb_read_ucdo( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;

*(double *)(((char *)mem)+d->offset1) = double(c);
return true;
}




static bool cb_read_usch( GZFILE fp, void * mem, PropDescriptor * d )
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;

*(char *)(((char *)mem)+d->offset1) = char(c);
return true;
}

static bool cb_read_ussh( GZFILE fp, void * mem, PropDescriptor * d )
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;

*(short *)(((char *)mem)+d->offset1) = short(c);
return true;
}

static bool cb_read_usin( GZFILE fp, void * mem, PropDescriptor * d )
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;

*(int *)(((char *)mem)+d->offset1) = int(c);
return true;
}

static bool cb_read_usuc( GZFILE fp, void * mem, PropDescriptor * d )
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;

*(uchar *)(((char *)mem)+d->offset1) = uchar(c);
return true;
}

static bool cb_read_usus( GZFILE fp, void * mem, PropDescriptor * d )
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;

*(ushort *)(((char *)mem)+d->offset1) = ushort(c);
return true;
}

static bool cb_read_usui( GZFILE fp, void * mem, PropDescriptor * d )
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;

*(uint *)(((char *)mem)+d->offset1) = uint(c);
return true;
}

static bool cb_read_usfl( GZFILE fp, void * mem, PropDescriptor * d )
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;

*(float *)(((char *)mem)+d->offset1) = float(c);
return true;
}

static bool cb_read_usdo( GZFILE fp, void * mem, PropDescriptor * d )
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;

*(double *)(((char *)mem)+d->offset1) = double(c);
return true;
}




static bool cb_read_uich( GZFILE fp, void * mem, PropDescriptor * d )
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;

*(char *)(((char *)mem)+d->offset1) = char(c);
return true;
}

static bool cb_read_uish( GZFILE fp, void * mem, PropDescriptor * d )
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;

*(short *)(((char *)mem)+d->offset1) = short(c);
return true;
}

static bool cb_read_uiin( GZFILE fp, void * mem, PropDescriptor * d )
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;

*(int *)(((char *)mem)+d->offset1) = int(c);
return true;
}

static bool cb_read_uiuc( GZFILE fp, void * mem, PropDescriptor * d )
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;

*(uchar *)(((char *)mem)+d->offset1) = uchar(c);
return true;
}

static bool cb_read_uius( GZFILE fp, void * mem, PropDescriptor * d )
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;

*(ushort *)(((char *)mem)+d->offset1) = ushort(c);
return true;
}

static bool cb_read_uiui( GZFILE fp, void * mem, PropDescriptor * d )
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;

*(uint *)(((char *)mem)+d->offset1) = uint(c);
return true;
}

static bool cb_read_uifl( GZFILE fp, void * mem, PropDescriptor * d )
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;

*(float *)(((char *)mem)+d->offset1) = float(c);
return true;
}

static bool cb_read_uido( GZFILE fp, void * mem, PropDescriptor * d )
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;

*(double *)(((char *)mem)+d->offset1) = double(c);
return true;
}



static bool cb_read_flfl( GZFILE fp, void * mem, PropDescriptor * d )
{
float c; if( ReadFloatB(fp,&c,d->format)==0 ) return false;

*(float *)(((char *)mem)+d->offset1) = float(c);
return true;
}

static bool cb_read_fldo( GZFILE fp, void * mem, PropDescriptor * d )
{
float c; if( ReadFloatB(fp,&c,d->format)==0 ) return false;

*(double *)(((char *)mem)+d->offset1) = double(c);
return true;
}

static bool cb_read_dofl( GZFILE fp, void * mem, PropDescriptor * d )
{
double c; if( ReadDoubleB(fp,&c,d->format)==0 ) return false;

*(float *)(((char *)mem)+d->offset1) = float(c);
return true;
}

static bool cb_read_dodo( GZFILE fp, void * mem, PropDescriptor * d )
{
double c; if( ReadDoubleB(fp,&c,d->format)==0 ) return false;

*(double *)(((char *)mem)+d->offset1) = double(c);
return true;
}


static bool cb_read_ascii( GZFILE fp, void * mem, PropDescriptor * d )
{
return ReadScalarA(fp, ((char *)mem)+d->offset1, d->stotype1, d->memtype1)!=0;
}


const int SKIP_MAX_BUF = 512;

static bool cb_skip_list_bin1( GZFILE fp, void * , PropDescriptor *  )
{
char skip_buf[SKIP_MAX_BUF];
uchar n;
if( pb_fread(&n,1,1,fp)==0 ) return false;

if( pb_fread(skip_buf,1,n,fp)==0) return false;
return true;
}

static bool cb_skip_list_bin2( GZFILE fp, void * , PropDescriptor *  )
{
char skip_buf[SKIP_MAX_BUF];
uchar n;
if( pb_fread(&n,1,1,fp)==0 ) return false;

if( pb_fread(skip_buf,2,n,fp)==0) return false;
return true;
}

static bool cb_skip_list_bin4( GZFILE fp, void * , PropDescriptor *  )
{
char skip_buf[SKIP_MAX_BUF];
uchar n;
if( pb_fread(&n,1,1,fp)==0 ) return false;

if( pb_fread(skip_buf,4,n,fp)==0) return false;
return true;
}

static bool cb_skip_list_bin8( GZFILE fp, void * , PropDescriptor *  )
{
char skip_buf[SKIP_MAX_BUF];
uchar n;
if( pb_fread(&n,1,1,fp)==0 ) return false;

if( pb_fread(skip_buf,8,n,fp)==0) return false;
return true;
}

static bool cb_skip_list_ascii ( GZFILE fp, void * , PropDescriptor *  )
{
int i,n;

if( !ReadScalarA(fp,&n,T_INT, T_INT) )return false;
for(i=0;i<n;++i)
if( !SkipScalarA(fp,T_FLOAT) ) 
return false;
return true;
}

static bool cb_read_list_ascii( GZFILE fp, void * mem, PropDescriptor * d )
{
int i,n;

if( ReadIntA(fp,&n)==0) return false;

char * store;

StoreInt( ((char *)mem)+d->offset2, d->memtype2, n);
if(d->alloclist)
{
store = (char *)calloc(n,TypeSize[d->memtype1]);
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
{
store = ((char *)mem)+d->offset1;
}

for(i=0;i<n;++i)
{
if( !ReadScalarA(
fp, 
store+i*TypeSize[d->memtype1],
d->stotype1,
d->memtype1
) )
return 0;
}
return true;
}


static bool cb_read_list_chch( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(char));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
if( pb_fread( store+i*sizeof(char),1,1,fp)==0 ) return false;
return true;
}

static bool cb_read_list_chsh( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(short));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
*(short *)(store+i*sizeof(short)) = short(c);
}
return true;
}

static bool cb_read_list_chin( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(int));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
*(int *)(store+i*sizeof(int)) = int(c);
}
return true;
}

static bool cb_read_list_chuc( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uchar));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
*(uchar *)(store+i*sizeof(uchar)) = uchar(c);
}
return true;
}

static bool cb_read_list_chus( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(ushort));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
*(ushort *)(store+i*sizeof(ushort)) = ushort(c);
}
return true;
}

static bool cb_read_list_chui( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uint));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
*(uint *)(store+i*sizeof(uint)) = uint(c);
}
return true;
}

static bool cb_read_list_chfl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(float));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
*(float *)(store+i*sizeof(float)) = float(c);
}
return true;
}

static bool cb_read_list_chdo( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(double));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
char c;
if( pb_fread(&c,1,1,fp)==0) return false;
*(double *)(store+i*sizeof(float)) = double(c);
}
return true;
}



static bool cb_read_list_shch( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(char));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;
*(char *)(store+i*sizeof(char)) = char(c);
}
return true;
}

static bool cb_read_list_shsh( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(short));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;
*(short *)(store+i*sizeof(short)) = short(c);
}
return true;
}

static bool cb_read_list_shin( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(int));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;
*(int *)(store+i*sizeof(int)) = int(c);
}
return true;
}

static bool cb_read_list_shuc( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uchar));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;
*(uchar *)(store+i*sizeof(uchar)) = uchar(c);
}
return true;
}

static bool cb_read_list_shus( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(ushort));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;
*(ushort *)(store+i*sizeof(ushort)) = ushort(c);
}
return true;
}

static bool cb_read_list_shui( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uint));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;
*(uint *)(store+i*sizeof(uint)) = uint(c);
}
return true;
}

static bool cb_read_list_shfl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(float));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;
*(float *)(store+i*sizeof(float)) = float(c);
}
return true;
}

static bool cb_read_list_shdo( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(double));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
short c; if( ReadShortB(fp,&c,d->format)==0) return false;
*(double *)(store+i*sizeof(double)) = double(c);
}
return true;
}




static bool cb_read_list_inch( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(char));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;
*(char *)(store+i*sizeof(char)) = char(c);
}
return true;
}

static bool cb_read_list_insh( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(short));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;
*(short *)(store+i*sizeof(short)) = short(c);
}
return true;
}

static bool cb_read_list_inin( GZFILE fp, void * mem, PropDescriptor * d )
{
int n=0,i;
switch(d->stotype2)
{ 
case T_CHAR : { char val; if( ReadCharB(fp,&val,d->format)==0 ) return false; n=val; } break;
case T_UCHAR : { uchar val; if( ReadUCharB(fp,&val,d->format)==0 ) return false; n=val; } break;
case T_SHORT : { short val; if( ReadShortB(fp,&val,d->format)==0 ) return false; n=val; } break;
case T_UINT : { uint val; if( ReadUIntB(fp,&val,d->format)==0 ) return false; n=val; } break;
case T_INT : { int val; if( ReadIntB(fp,&val,d->format)==0 ) return false; n=val; } break;
default: assert(0); break;
}
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(int));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
if( ReadIntB(fp,(int *)(store+i*sizeof(int)),d->format)==0 ) return false;
}
return true;
}

static bool cb_read_list_inuc( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uchar));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;
*(uchar *)(store+i*sizeof(uchar)) = uchar(c);
}
return true;
}

static bool cb_read_list_inus( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(ushort));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;
*(ushort *)(store+i*sizeof(ushort)) = ushort(c);
}
return true;
}

static bool cb_read_list_inui( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uint));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;
*(uint *)(store+i*sizeof(uint)) = uint(c);
}
return true;
}

static bool cb_read_list_infl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(float));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;
*(float *)(store+i*sizeof(float)) = float(c);
}
return true;
}

static bool cb_read_list_indo( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(double));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
int c; if( ReadIntB(fp,&c,d->format)==0 ) return false;
*(double *)(store+i*sizeof(double)) = double(c);
}
return true;
}



static bool cb_read_list_ucch( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(char));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;
*(char *)(store+i*sizeof(char)) = char(c);
}
return true;
}

static bool cb_read_list_ucsh( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(short));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;
*(short *)(store+i*sizeof(short)) = short(c);
}
return true;
}

static bool cb_read_list_ucin( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(int));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;
*(int *)(store+i*sizeof(int)) = int(c);
}
return true;
}

static bool cb_read_list_ucuc( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uchar));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;
*(uchar *)(store+i*sizeof(uchar)) = uchar(c);
}
return true;
}

static bool cb_read_list_ucus( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(ushort));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;
*(ushort *)(store+i*sizeof(ushort)) = ushort(c);
}
return true;
}

static bool cb_read_list_ucui( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uint));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;
*(uint *)(store+i*sizeof(uint)) = uint(c);
}
return true;
}

static bool cb_read_list_ucfl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(float));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;
*(float *)(store+i*sizeof(float)) = float(c);
}
return true;
}

static bool cb_read_list_ucdo( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(double));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uchar c; if( pb_fread(&c,1,1,fp)==0 ) return false;
*(double *)(store+i*sizeof(double)) = double(c);
}
return true;
}




static bool cb_read_list_usch( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(char));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;
*(char *)(store+i*sizeof(char)) = char(c);
}
return true;
}

static bool cb_read_list_ussh( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(short));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;
*(short *)(store+i*sizeof(short)) = short(c);
}
return true;
}

static bool cb_read_list_usin( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(int));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;
*(int *)(store+i*sizeof(int)) = int(c);
}
return true;
}

static bool cb_read_list_usuc( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uchar));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;
*(uchar *)(store+i*sizeof(uchar)) = uchar(c);
}
return true;
}

static bool cb_read_list_usus( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(ushort));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;
*(ushort *)(store+i*sizeof(ushort)) = ushort(c);
}
return true;
}

static bool cb_read_list_usui( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uint));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;
*(uint *)(store+i*sizeof(uint)) = uint(c);
}
return true;
}

static bool cb_read_list_usfl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(float));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;
*(float *)(store+i*sizeof(float)) = float(c);
}
return true;
}

static bool cb_read_list_usdo( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(double));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
ushort c; if( ReadUShortB(fp,&c,d->format)==0) return false;
*(double *)(store+i*sizeof(double)) = double(c);
}
return true;
}




static bool cb_read_list_uich( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(char));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;
*(char *)(store+i*sizeof(char)) = char(c);
}
return true;
}

static bool cb_read_list_uish( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(short));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;
*(short *)(store+i*sizeof(short)) = short(c);
}
return true;
}

static bool cb_read_list_uiin( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(int));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;
*(int *)(store+i*sizeof(int)) = int(c);
}
return true;
}

static bool cb_read_list_uiuc( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uchar));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;
*(uchar *)(store+i*sizeof(uchar)) = uchar(c);
}
return true;
}

static bool cb_read_list_uius( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(ushort));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;
*(ushort *)(store+i*sizeof(ushort)) = ushort(c);
}
return true;
}

static bool cb_read_list_uiui( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(uint));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;
*(uint *)(store+i*sizeof(uint)) = uint(c);
}
return true;
}

static bool cb_read_list_uifl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(float));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;
*(float *)(store+i*sizeof(float)) = float(c);
}
return true;
}

static bool cb_read_list_uido( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(double));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
uint c; if( ReadUIntB(fp,&c,d->format)==0 ) return false;
*(double *)(store+i*sizeof(double)) = double(c);
}
return true;
}



static bool cb_read_list_flfl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(float));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
float c; if( ReadFloatB(fp,&c,d->format)==0 ) return false;
*(float *)(store+i*sizeof(float)) = float(c);
}
return true;
}

static bool cb_read_list_fldo( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(double));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
float c; if( ReadFloatB(fp,&c,d->format)==0 ) return false;
*(double *)(store+i*sizeof(double)) = double(c);
}
return true;
}

static bool cb_read_list_dofl( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(float));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
double c; if( ReadDoubleB(fp,&c,d->format)==0 ) return false;
*(float *)(store+i*sizeof(float)) = float(c);
}
return true;
}

static bool cb_read_list_dodo( GZFILE fp, void * mem, PropDescriptor * d )
{
uchar n,i;
if( pb_fread(&n,1,1,fp)==0 ) return false;
char * store;
StoreInt( ((char *)mem)+d->offset2, d->memtype2, int(n));
if(d->alloclist)
{
store = (char *)calloc(n,sizeof(double));
assert(store);
*(char **)(((char *)mem)+d->offset1) = store;
}
else
store = ((char *)mem)+d->offset1;

for(i=0;i<n;++i)
{
double c; if( ReadDoubleB(fp,&c,d->format)==0 ) return false;
*(double *)(store+i*sizeof(double)) = double(c);
}
return true;
}



void PlyFile::compile( PlyProperty * p )
{
p->desc.format = format;		

if(format==F_ASCII)
{
if(p->islist)
{
if(p->bestored)
p->cb = cb_read_list_ascii;
else
p->cb = cb_skip_list_ascii;
}
else
{
if(p->bestored)
{
p->cb = cb_read_ascii;
}
else
{
switch(p->tipo)
{
case T_CHAR:
case T_SHORT:
case T_INT:
case T_UCHAR:
case T_USHORT:
case T_UINT:
p->cb = cb_skip_int_ascii;
break;
case T_FLOAT:
case T_DOUBLE: 
p->cb = cb_skip_float_ascii;
break;
default: p->cb = 0; assert(0); break;
}
}
}
}
else
{
if(p->islist)
{
if(p->bestored)
{
switch(p->desc.stotype1)
{
case T_CHAR:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_list_chch; break;
case T_SHORT:	p->cb = cb_read_list_chsh; break;
case T_INT:		p->cb = cb_read_list_chin; break;
case T_UCHAR:	p->cb = cb_read_list_chuc; break;
case T_USHORT:	p->cb = cb_read_list_chus; break;
case T_UINT:	p->cb = cb_read_list_chui; break;
case T_FLOAT:	p->cb = cb_read_list_chfl; break;
case T_DOUBLE:	p->cb = cb_read_list_chdo; break;
default: assert(0);
}
break;
case T_SHORT:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_list_shch; break;
case T_SHORT:	p->cb = cb_read_list_shsh; break;
case T_INT:		p->cb = cb_read_list_shin; break;
case T_UCHAR:	p->cb = cb_read_list_shuc; break;
case T_USHORT:	p->cb = cb_read_list_shus; break;
case T_UINT:	p->cb = cb_read_list_shui; break;
case T_FLOAT:	p->cb = cb_read_list_shfl; break;
case T_DOUBLE:	p->cb = cb_read_list_shdo; break;
default: assert(0);
}
break;
case T_INT:		
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_list_inch; break;
case T_SHORT:	p->cb = cb_read_list_insh; break;
case T_INT:		p->cb = cb_read_list_inin; break;
case T_UCHAR:	p->cb = cb_read_list_inuc; break;
case T_USHORT:	p->cb = cb_read_list_inus; break;
case T_UINT:	p->cb = cb_read_list_inui; break;
case T_FLOAT:	p->cb = cb_read_list_infl; break;
case T_DOUBLE:	p->cb = cb_read_list_indo; break;
default: assert(0);
}
break;
case T_UCHAR:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_list_ucch; break;
case T_SHORT:	p->cb = cb_read_list_ucsh; break;
case T_INT:		p->cb = cb_read_list_ucin; break;
case T_UCHAR:	p->cb = cb_read_list_ucuc; break;
case T_USHORT:	p->cb = cb_read_list_ucus; break;
case T_UINT:	p->cb = cb_read_list_ucui; break;
case T_FLOAT:	p->cb = cb_read_list_ucfl; break;
case T_DOUBLE:	p->cb = cb_read_list_ucdo; break;
default: assert(0);
}
break;
case T_USHORT:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_list_usch; break;
case T_SHORT:	p->cb = cb_read_list_ussh; break;
case T_INT:		p->cb = cb_read_list_usin; break;
case T_UCHAR:	p->cb = cb_read_list_usuc; break;
case T_USHORT:	p->cb = cb_read_list_usus; break;
case T_UINT:	p->cb = cb_read_list_usui; break;
case T_FLOAT:	p->cb = cb_read_list_usfl; break;
case T_DOUBLE:	p->cb = cb_read_list_usdo; break;
default: assert(0);
}
break;
case T_UINT:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_list_uich; break;
case T_SHORT:	p->cb = cb_read_list_uish; break;
case T_INT:		p->cb = cb_read_list_uiin; break;
case T_UCHAR:	p->cb = cb_read_list_uiuc; break;
case T_USHORT:	p->cb = cb_read_list_uius; break;
case T_UINT:	p->cb = cb_read_list_uiui; break;
case T_FLOAT:	p->cb = cb_read_list_uifl; break;
case T_DOUBLE:	p->cb = cb_read_list_uido; break;
default: assert(0);
}
break;
case T_FLOAT:	
switch(p->desc.memtype1)
{
case T_FLOAT:  p->cb = cb_read_list_flfl; break;
case T_DOUBLE: p->cb = cb_read_list_fldo; break;
default: assert(0);
}
break;
case T_DOUBLE:	
switch(p->desc.memtype1)
{
case T_FLOAT:  p->cb = cb_read_list_dofl; break;
case T_DOUBLE: p->cb = cb_read_list_dodo; break;
default: assert(0);
}
break;
default:
assert(0);
}
}
else
{
switch(TypeSize[p->tipo])
{
case 1: p->cb = cb_skip_list_bin1; break;
case 2: p->cb = cb_skip_list_bin2; break;
case 4: p->cb = cb_skip_list_bin4; break;
case 8: p->cb = cb_skip_list_bin8; break; 
default:p->cb = 0; assert(0); break;
}
}
}
else
{
if(p->bestored)
{
switch(p->desc.stotype1)
{
case T_CHAR:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_chch; break;
case T_SHORT:	p->cb = cb_read_chsh; break;
case T_INT:		p->cb = cb_read_chin; break;
case T_UCHAR:	p->cb = cb_read_chuc; break;
case T_USHORT:	p->cb = cb_read_chus; break;
case T_UINT:	p->cb = cb_read_chui; break;
case T_FLOAT:	p->cb = cb_read_chfl; break;
case T_DOUBLE:	p->cb = cb_read_chdo; break;
default: assert(0);
}
break;
case T_SHORT:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_shch; break;
case T_SHORT:	p->cb = cb_read_shsh; break;
case T_INT:		p->cb = cb_read_shin; break;
case T_UCHAR:	p->cb = cb_read_shuc; break;
case T_USHORT:	p->cb = cb_read_shus; break;
case T_UINT:	p->cb = cb_read_shui; break;
case T_FLOAT:	p->cb = cb_read_shfl; break;
case T_DOUBLE:	p->cb = cb_read_shdo; break;
default: assert(0);
}
break;
case T_INT:		
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_inch; break;
case T_SHORT:	p->cb = cb_read_insh; break;
case T_INT:		p->cb = cb_read_inin; break;
case T_UCHAR:	p->cb = cb_read_inuc; break;
case T_USHORT:	p->cb = cb_read_inus; break;
case T_UINT:	p->cb = cb_read_inui; break;
case T_FLOAT:	p->cb = cb_read_infl; break;
case T_DOUBLE:	p->cb = cb_read_indo; break;
default: assert(0);
}
break;
case T_UCHAR:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_ucch; break;
case T_SHORT:	p->cb = cb_read_ucsh; break;
case T_INT:		p->cb = cb_read_ucin; break;
case T_UCHAR:	p->cb = cb_read_ucuc; break;
case T_USHORT:	p->cb = cb_read_ucus; break;
case T_UINT:	p->cb = cb_read_ucui; break;
case T_FLOAT:	p->cb = cb_read_ucfl; break;
case T_DOUBLE:	p->cb = cb_read_ucdo; break;
default: assert(0);
}
break;
case T_USHORT:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_usch; break;
case T_SHORT:	p->cb = cb_read_ussh; break;
case T_INT:		p->cb = cb_read_usin; break;
case T_UCHAR:	p->cb = cb_read_usuc; break;
case T_USHORT:	p->cb = cb_read_usus; break;
case T_UINT:	p->cb = cb_read_usui; break;
case T_FLOAT:	p->cb = cb_read_usfl; break;
case T_DOUBLE:	p->cb = cb_read_usdo; break;
default: assert(0);
}
break;
case T_UINT:	
switch(p->desc.memtype1)
{
case T_CHAR:	p->cb = cb_read_uich; break;
case T_SHORT:	p->cb = cb_read_uish; break;
case T_INT:		p->cb = cb_read_uiin; break;
case T_UCHAR:	p->cb = cb_read_uiuc; break;
case T_USHORT:	p->cb = cb_read_uius; break;
case T_UINT:	p->cb = cb_read_uiui; break;
case T_FLOAT:	p->cb = cb_read_uifl; break;
case T_DOUBLE:	p->cb = cb_read_uido; break;
default: assert(0);
}
break;
case T_FLOAT:	
switch(p->desc.memtype1)
{
case T_FLOAT:  p->cb = cb_read_flfl; break;
case T_DOUBLE: p->cb = cb_read_fldo; break;
default: assert(0);
}
break;
case T_DOUBLE:	
switch(p->desc.memtype1)
{
case T_FLOAT:  p->cb = cb_read_dofl; break;
case T_DOUBLE: p->cb = cb_read_dodo; break;
default: assert(0);
}
break;
default:
assert(0);
}
}
else
{
switch(TypeSize[p->tipo])
{
case 1: p->cb = cb_skip_bin1; break;
case 2: p->cb = cb_skip_bin2; break;
case 4: p->cb = cb_skip_bin4; break;
case 8: p->cb = cb_skip_bin8; break;
default:p->cb = 0; assert(0); break;
}
}
}
}
}

void PlyFile::compile( PlyElement * e )
{
vector<PlyProperty>::iterator i;
for(i=e->props.begin();i!=e->props.end();++i)
compile(&*i);
}




int ReadBin  ( XFILE * fp, const PlyProperty * pr, void * mem, int fmt )
{
assert(pr);

if(pr->islist)
{
int i,n;

if( !ReadScalarB(fp,&n,pr->tipoindex, T_INT, fmt) )
return 0;

assert(n<12);	

if(pr->bestored)
{
char * store;

StoreInt( ((char *)mem)+pr->desc.offset2, pr->desc.memtype2, n);
if(pr->desc.alloclist)
{
store = (char *)calloc(n,TypeSize[pr->desc.memtype1]);
assert(store);
*(char **)(((char *)mem)+pr->desc.offset1) = store;
}
else
{
store = ((char *)mem)+pr->desc.offset1;
}

for(i=0;i<n;++i)
{
if( !ReadScalarB(
fp, 
store+i*TypeSize[pr->desc.memtype1],
pr->desc.stotype1,
pr->desc.memtype1,
fmt
) )
return 0;
}
}
else
{
for(i=0;i<n;i++)
if( !SkipScalarB(fp,pr->tipo) )
return 0;
}
}
else
{
if(pr->bestored)
return ReadScalarB(
fp, 
((char *)mem)+pr->desc.offset1,
pr->desc.stotype1, pr->desc.memtype1,
fmt
);
else
return SkipScalarB(fp,pr->tipo);
}

return 1;
}


int ReadAscii( XFILE * fp, const PlyProperty * pr, void * mem, int  )
{
assert(pr);
assert(mem);


if(pr->islist)
{
int i,n;

if( !ReadScalarA(fp,&n,pr->tipoindex, T_INT) )
return 0;

assert(n<12);	

if(pr->bestored)
{
char * store;

StoreInt( ((char *)mem)+pr->desc.offset2, pr->desc.memtype2, n);
if(pr->desc.alloclist)
{
store = (char *)calloc(n,TypeSize[pr->desc.memtype1]);
assert(store);
*(char **)(((char *)mem)+pr->desc.offset1) = store;
}
else
{
store = ((char *)mem)+pr->desc.offset1;
}

for(i=0;i<n;++i)
{
if( !ReadScalarA(
fp, 
store+i*TypeSize[pr->desc.memtype1],
pr->desc.stotype1,
pr->desc.memtype1
) )
return 0;
}
}
else
{
for(i=0;i<n;++i)
if( !SkipScalarA(fp,pr->tipo) )
return 0;
}
}
else
{
if(pr->bestored)
return ReadScalarA(
fp, 
((char *)mem)+pr->desc.offset1,
pr->desc.stotype1, pr->desc.memtype1
);
else
return SkipScalarA(fp,pr->tipo);
}
return 1;
}



int PlyFile::Read( void * mem )
{
assert(cure);
assert(ReadCB);

vector<PlyProperty>::iterator i;

for(i=cure->props.begin();i!=cure->props.end();++i)
{
if( ! i->cb(gzfp,mem,&(i->desc)) ) return -1;

}

return 0;
}

void interpret_texture_name(const char*a, const char*fn, char*output){
int ia=0,io=0;
output[0]=0;
while (a[ia]!=0){
if (a[ia]=='<') {
if (static_cast<int>(strlen(a)) > ia+5) {
if ( ( (a[ia+1]=='t') || (a[ia+1]=='T') ) &&
( (a[ia+2]=='h') || (a[ia+2]=='H') ) &&
( (a[ia+3]=='i') || (a[ia+3]=='I') ) &&
( (a[ia+4]=='s') || (a[ia+4]=='S') ) &&
( a[ia+5]=='>' ) )
{
int lastbar=0;
int ifn=0;
while (fn[ifn]!=0) { if ((fn[ifn]=='/') || (fn[ifn]=='\\')) lastbar=ifn+1; ifn++;}
ifn=lastbar;
char fn2[255];
while (fn[ifn]!=0) { fn2[ifn-lastbar]=fn[ifn]; ifn++;}
fn2[ifn-lastbar]=0;

int l=ifn-lastbar;
if ((fn2[l-4]=='.') 
&& ((fn2[l-3]=='P') || (fn2[l-3]=='p')) 
&& ((fn2[l-2]=='L') || (fn2[l-2]=='l')) 
&& ((fn2[l-1]=='Y') || (fn2[l-1]=='y')) )
fn2[l-4]=0;

output[io]=0;
sprintf(output,"%s%s",output,fn2);
io=strlen(output);
ia+=6; 
continue;
};
}
}
output[io++]=a[ia++]; 
};
output[io]=0;
};
} 
}
