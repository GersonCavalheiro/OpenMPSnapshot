#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
void get_info(char * argv[], int * typep, char * classp);
void check_info(int type, char class);
void read_info(int type, char * classp);
void write_info(int type, char class);
void write_sp_info(FILE * fp, char class);
void write_bt_info(FILE * fp, char class);
void write_lu_info(FILE * fp, char class);
void write_mg_info(FILE * fp, char class);
void write_cg_info(FILE * fp, char class);
void write_ft_info(FILE * fp, char class);
void write_ep_info(FILE * fp, char class);
void write_is_info(FILE * fp, char class);
void write_compiler_info(int type, FILE * fp);
void write_convertdouble_info(int type, FILE * fp);
void check_line(char * line, char * label, char * val);
int check_include_line(char * line, char * filename);
void put_string(FILE * fp, char * name, char * val);
void put_def_string(FILE * fp, char * name, char * val);
void put_def_variable(FILE * fp, char * name, char * val);
int ilog2(int i);
enum benchmark_types { SP, BT, LU, MG, FT, IS, EP, CG };
main(int argc, char * argv[])
{
int type;
char class, class_old;
int _ret_val_0;
if (argc!=3)
{
printf("Usage: %s benchmark-name class\n", argv[0]);
exit(1);
}
get_info(argv,  & type,  & class);
if (class!='U')
{
check_info(type, class);
}
read_info(type,  & class_old);
if (class!='U')
{
if (class_old!='X')
{
}
}
else
{
printf("setparams:\n  *********************************************************************\n  * You must specify CLASS to build this benchmark                    *\n  * For example, to build a class A benchmark, type                   *\n  *       make {benchmark-name} CLASS=A                               *\n  *********************************************************************\n\n");
if (class_old!='X')
{
}
exit(1);
}
if (class!=class_old)
{
write_info(type, class);
}
else
{
}
exit(0);
return _ret_val_0;
}
void get_info(char * argv[], int * typep, char * classp)
{
( * classp)=( * argv[2]);
if (( ! strcmp(argv[1], "sp"))||( ! strcmp(argv[1], "SP")))
{
( * typep)=SP;
}
else
{
if (( ! strcmp(argv[1], "bt"))||( ! strcmp(argv[1], "BT")))
{
( * typep)=BT;
}
else
{
if (( ! strcmp(argv[1], "ft"))||( ! strcmp(argv[1], "FT")))
{
( * typep)=FT;
}
else
{
if (( ! strcmp(argv[1], "lu"))||( ! strcmp(argv[1], "LU")))
{
( * typep)=LU;
}
else
{
if (( ! strcmp(argv[1], "mg"))||( ! strcmp(argv[1], "MG")))
{
( * typep)=MG;
}
else
{
if (( ! strcmp(argv[1], "is"))||( ! strcmp(argv[1], "IS")))
{
( * typep)=IS;
}
else
{
if (( ! strcmp(argv[1], "ep"))||( ! strcmp(argv[1], "EP")))
{
( * typep)=EP;
}
else
{
if (( ! strcmp(argv[1], "cg"))||( ! strcmp(argv[1], "CG")))
{
( * typep)=CG;
}
else
{
printf("setparams: Error: unknown benchmark type %s\n", argv[1]);
exit(1);
}
}
}
}
}
}
}
}
return ;
}
void check_info(int type, char class)
{
int tmplog;
if ((((((class!='S')&&(class!='A'))&&(class!='B'))&&(class!='R'))&&(class!='W'))&&(class!='C'))
{
printf("setparams: Unknown benchmark class %c\n", class);
printf("setparams: Allowed classes are \"S\", \"A\", \"B\" and \"C\"\n");
exit(1);
}
return ;
}
void read_info(int type, char * classp)
{
int nread, gotem = 0;
char line[200];
FILE * fp;
fp=fopen("npbparams.h", "r");
if (fp==((void * )0))
{
goto abort;
}
switch (type)
{
case SP:
case BT:
case FT:
case MG:
case LU:
case EP:
case CG:
nread=fscanf(fp, "\n", classp);
if (nread!=1)
{
printf("setparams: Error parsing config file %s. Ignoring previous settings\n", "npbparams.h");
goto abort;
}
break;
case IS:
nread=fscanf(fp, "#define CLASS '%c'\n", classp);
if (nread!=1)
{
printf("setparams: Error parsing config file %s. Ignoring previous settings\n", "npbparams.h");
goto abort;
}
break;
default:
printf("setparams: (Internal Error) Benchmark type %d unknown to this program\n", type);
exit(1);
}
normal_return:
( * classp)=( * classp);
fclose(fp);
return ;
abort:
( * classp)='X';
return ;
}
void write_info(int type, char class)
{
FILE * fp;
fp=fopen("npbparams.h", "w");
if (fp==((void * )0))
{
printf("setparams: Can't open file %d for writing\n", "npbparams.h");
exit(1);
}
switch (type)
{
case SP:
case BT:
case FT:
case MG:
case LU:
case EP:
case CG:
fprintf(fp, "\n", class);
fprintf(fp, "\n");
break;
case IS:
fprintf(fp, "#define CLASS '%c'\n", class);
fprintf(fp, "\n   \n");
break;
default:
printf("setparams: (Internal error): Unknown benchmark type %d\n", type);
exit(1);
}
switch (type)
{
case SP:
write_sp_info(fp, class);
break;
case BT:
write_bt_info(fp, class);
break;
case LU:
write_lu_info(fp, class);
break;
case MG:
write_mg_info(fp, class);
break;
case IS:
write_is_info(fp, class);
break;
case FT:
write_ft_info(fp, class);
break;
case EP:
write_ep_info(fp, class);
break;
case CG:
write_cg_info(fp, class);
break;
default:
printf("setparams: (Internal error): Unknown benchmark type %d\n", type);
exit(1);
}
write_convertdouble_info(type, fp);
write_compiler_info(type, fp);
fclose(fp);
return ;
}
void write_sp_info(FILE * fp, char class)
{
int problem_size, niter;
char * dt;
if (class=='S')
{
problem_size=12;
dt="0.015";
niter=100;
}
else
{
if (class=='W')
{
problem_size=36;
dt="0.0015";
niter=400;
}
else
{
if (class=='A')
{
problem_size=64;
dt="0.0015";
niter=400;
}
else
{
if (class=='B')
{
problem_size=102;
dt="0.001";
niter=400;
}
else
{
if (class=='C')
{
problem_size=162;
dt="0.00067";
niter=400;
}
else
{
printf("setparams: Internal error: invalid class %c\n", class);
exit(1);
}
}
}
}
}
fprintf(fp, "#define\tPROBLEM_SIZE\t%d\n", problem_size);
fprintf(fp, "#define\tNITER_DEFAULT\t%d\n", niter);
fprintf(fp, "#define\tDT_DEFAULT\t%s\n", dt);
return ;
}
void write_bt_info(FILE * fp, char class)
{
int problem_size, niter;
char * dt;
if (class=='S')
{
problem_size=12;
dt="0.010";
niter=60;
}
else
{
if (class=='W')
{
problem_size=24;
dt="0.0008";
niter=200;
}
else
{
if (class=='A')
{
problem_size=64;
dt="0.0008";
niter=200;
}
else
{
if (class=='B')
{
problem_size=102;
dt="0.0003";
niter=200;
}
else
{
if (class=='C')
{
problem_size=162;
dt="0.0001";
niter=200;
}
else
{
printf("setparams: Internal error: invalid class %c\n", class);
exit(1);
}
}
}
}
}
fprintf(fp, "#define\tPROBLEM_SIZE\t%d\n", problem_size);
fprintf(fp, "#define\tNITER_DEFAULT\t%d\n", niter);
fprintf(fp, "#define\tDT_DEFAULT\t%s\n", dt);
return ;
}
void write_lu_info(FILE * fp, char class)
{
int isiz1, isiz2, itmax, inorm, problem_size;
int xdiv, ydiv;
char * dt_default;
if (class=='S')
{
problem_size=12;
dt_default="0.5";
itmax=50;
}
else
{
if (class=='W')
{
problem_size=33;
dt_default="1.5e-3";
itmax=300;
}
else
{
if (class=='A')
{
problem_size=64;
dt_default="2.0";
itmax=250;
}
else
{
if (class=='B')
{
problem_size=102;
dt_default="2.0";
itmax=250;
}
else
{
if (class=='C')
{
problem_size=162;
dt_default="2.0";
itmax=250;
}
else
{
printf("setparams: Internal error: invalid class %c\n", class);
exit(1);
}
}
}
}
}
inorm=itmax;
isiz1=problem_size;
isiz2=problem_size;
fprintf(fp, "\n\n");
fprintf(fp, "#define\tISIZ1\t%d\n", problem_size);
fprintf(fp, "#define\tISIZ2\t%d\n", problem_size);
fprintf(fp, "#define\tISIZ3\t%d\n", problem_size);
fprintf(fp, "\n");
fprintf(fp, "#define\tITMAX_DEFAULT\t%d\n", itmax);
fprintf(fp, "#define\tINORM_DEFAULT\t%d\n", inorm);
fprintf(fp, "#define\tDT_DEFAULT\t%s\n", dt_default);
return ;
}
void write_mg_info(FILE * fp, char class)
{
int problem_size, nit, log2_size, lt_default, lm;
int ndim1, ndim2, ndim3;
if (class=='S')
{
problem_size=32;
nit=4;
}
else
{
if (class=='W')
{
problem_size=64;
nit=40;
}
else
{
if (class=='A')
{
problem_size=256;
nit=4;
}
else
{
if (class=='B')
{
problem_size=256;
nit=20;
}
else
{
if (class=='C')
{
problem_size=512;
nit=20;
}
else
{
printf("setparams: Internal error: invalid class type %c\n", class);
exit(1);
}
}
}
}
}
log2_size=ilog2(problem_size);
lt_default=log2_size;
lm=log2_size;
ndim1=lm;
ndim3=log2_size;
ndim2=log2_size;
fprintf(fp, "#define\tNX_DEFAULT\t%d\n", problem_size);
fprintf(fp, "#define\tNY_DEFAULT\t%d\n", problem_size);
fprintf(fp, "#define\tNZ_DEFAULT\t%d\n", problem_size);
fprintf(fp, "#define\tNIT_DEFAULT\t%d\n", nit);
fprintf(fp, "#define\tLM\t%d\n", lm);
fprintf(fp, "#define\tLT_DEFAULT\t%d\n", lt_default);
fprintf(fp, "#define\tDEBUG_DEFAULT\t%d\n", 0);
fprintf(fp, "#define\tNDIM1\t%d\n", ndim1);
fprintf(fp, "#define\tNDIM2\t%d\n", ndim2);
fprintf(fp, "#define\tNDIM3\t%d\n", ndim3);
return ;
}
void write_is_info(FILE * fp, char class)
{
int m1, m2, m3;
if (((((class!='S')&&(class!='W'))&&(class!='A'))&&(class!='B'))&&(class!='C'))
{
printf("setparams: Internal error: invalid class type %c\n", class);
exit(1);
}
return ;
}
void write_cg_info(FILE * fp, char class)
{
int na, nonzer, niter;
char * shift, * rcond = "1.0e-1";
char * shiftS = "10.0", * shiftW = "12.0", * shiftA = "20.0", * shiftB = "60.0", * shiftC = "110.0";
if (class=='S')
{
na=1400;
nonzer=7;
niter=15;
shift=shiftS;
}
else
{
if (class=='W')
{
na=7000;
nonzer=8;
niter=15;
shift=shiftW;
}
else
{
if (class=='A')
{
na=14000;
nonzer=11;
niter=15;
shift=shiftA;
}
else
{
if (class=='B')
{
na=75000;
nonzer=13;
niter=75;
shift=shiftB;
}
else
{
if (class=='C')
{
na=150000;
nonzer=15;
niter=75;
shift=shiftC;
}
else
{
printf("setparams: Internal error: invalid class type %c\n", class);
exit(1);
}
}
}
}
}
fprintf(fp, "#define\tNA\t%d\n", na);
fprintf(fp, "#define\tNONZER\t%d\n", nonzer);
fprintf(fp, "#define\tNITER\t%d\n", niter);
fprintf(fp, "#define\tSHIFT\t%s\n", shift);
fprintf(fp, "#define\tRCOND\t%s\n", rcond);
return ;
}
void write_ft_info(FILE * fp, char class)
{
int nx, ny, nz, maxdim, niter, np_min;
if (class=='S')
{
nx=64;
ny=64;
nz=64;
niter=6;
}
else
{
if (class=='W')
{
nx=128;
ny=128;
nz=32;
niter=6;
}
else
{
if (class=='A')
{
nx=256;
ny=256;
nz=128;
niter=6;
}
else
{
if (class=='B')
{
nx=512;
ny=256;
nz=256;
niter=20;
}
else
{
if (class=='C')
{
nx=512;
ny=512;
nz=512;
niter=20;
}
else
{
printf("setparams: Internal error: invalid class type %c\n", class);
exit(1);
}
}
}
}
}
maxdim=nx;
if (ny>maxdim)
{
maxdim=ny;
}
if (nz>maxdim)
{
maxdim=nz;
}
fprintf(fp, "#define\tNX\t%d\n", nx);
fprintf(fp, "#define\tNY\t%d\n", ny);
fprintf(fp, "#define\tNZ\t%d\n", nz);
fprintf(fp, "#define\tMAXDIM\t%d\n", maxdim);
fprintf(fp, "#define\tNITER_DEFAULT\t%d\n", niter);
fprintf(fp, "#define\tNTOTAL\t%d\n", (nx*ny)*nz);
return ;
}
void write_ep_info(FILE * fp, char class)
{
int m;
if (class=='S')
{
m=24;
}
else
{
if (class=='W')
{
m=25;
}
else
{
if (class=='A')
{
m=28;
}
else
{
if (class=='B')
{
m=30;
}
else
{
if (class=='C')
{
m=32;
}
else
{
printf("setparams: Internal error: invalid class type %c\n", class);
exit(1);
}
}
}
}
}
fprintf(fp, "#define\tCLASS\t \'%c\'\n", class);
fprintf(fp, "#define\tM\t%d\n", m);
return ;
}
#include <stdio.h>
void write_compiler_info(int type, FILE * fp)
{
FILE * deffile;
char line[400];
char f77[400], flink[400], f_lib[400], f_inc[400], fflags[400], flinkflags[400];
char compiletime[400], randfile[400];
char cc[400], cflags[400], clink[400], clinkflags[400], c_lib[400], c_inc[400];
struct tm * tmp;
time_t t;
deffile=fopen("../config/make.def", "r");
if (deffile==((void * )0))
{
printf("\nsetparams: File %s doesn't exist. To build the NAS benchmarks\n           you need to create is according to the instructions\n           in the README in the main directory and comments in \n           the file config/make.def.template\n", "../config/make.def");
exit(1);
}
strcpy(f77, "(none)");
strcpy(flink, "(none)");
strcpy(f_lib, "(none)");
strcpy(f_inc, "(none)");
strcpy(fflags, "(none)");
strcpy(flinkflags, "(none)");
strcpy(randfile, "(none)");
strcpy(cc, "(none)");
strcpy(cflags, "(none)");
strcpy(clink, "(none)");
strcpy(clinkflags, "(none)");
strcpy(c_lib, "(none)");
strcpy(c_inc, "(none)");
while (fgets(line, 400, deffile)!=((void * )0))
{
if (( * line)=='#')
{
continue;
}
check_line(line, "F77", f77);
check_line(line, "FLINK", flink);
check_line(line, "F_LIB", f_lib);
check_line(line, "F_INC", f_inc);
check_line(line, "FFLAGS", fflags);
check_line(line, "FLINKFLAGS", flinkflags);
check_line(line, "RAND", randfile);
check_line(line, "CC", cc);
check_line(line, "CFLAGS", cflags);
check_line(line, "CLINK", clink);
check_line(line, "CLINKFLAGS", clinkflags);
check_line(line, "C_LIB", c_lib);
check_line(line, "C_INC", c_inc);
}
(void)time( & t);
tmp=localtime( & t);
(void)strftime(compiletime, (size_t)400, "%d %b %Y", tmp);
switch (type)
{
case FT:
case SP:
case BT:
case MG:
case LU:
case EP:
case CG:
put_def_string(fp, "COMPILETIME", compiletime);
put_def_string(fp, "NPBVERSION", "3.0 structured");
put_def_string(fp, "CS1", cc);
put_def_string(fp, "CS2", clink);
put_def_string(fp, "CS3", c_lib);
put_def_string(fp, "CS4", c_inc);
put_def_string(fp, "CS5", cflags);
put_def_string(fp, "CS6", clinkflags);
put_def_string(fp, "CS7", randfile);
break;
case IS:
put_def_string(fp, "COMPILETIME", compiletime);
put_def_string(fp, "NPBVERSION", "3.0 structured");
put_def_string(fp, "CC", cc);
put_def_string(fp, "CFLAGS", cflags);
put_def_string(fp, "CLINK", clink);
put_def_string(fp, "CLINKFLAGS", clinkflags);
put_def_string(fp, "C_LIB", c_lib);
put_def_string(fp, "C_INC", c_inc);
break;
default:
printf("setparams: (Internal error): Unknown benchmark type %d\n", type);
exit(1);
}
return ;
}
void check_line(char * line, char * label, char * val)
{
char * original_line;
original_line=line;
while ((( * label)!='\0')&&(( * line)==( * label)))
{
line ++ ;
label ++ ;
}
if (( * label)!='\0')
{
return ;
}
if (( ! (( * __ctype_b_loc())[(int)( * line)]&((unsigned short int)_ISspace)))&&(( * line)!='='))
{
return ;
}
while (( * __ctype_b_loc())[(int)( * line)]&((unsigned short int)_ISspace))
{
line ++ ;
}
if (( * line)!='=')
{
return ;
}
while (( * __ctype_b_loc())[(int)( * ( ++ line))]&((unsigned short int)_ISspace))
{
;
}
if (( * line)=='\0')
{
return ;
}
strcpy(val, line);
val[strlen(val)-1]='\0';
if (val[strlen(val)-1]=='\\')
{
printf("\nsetparams: Error in file make.def. Because of the way in which\n           command line arguments are incorporated into the\n           executable benchmark, you can't have any continued\n           lines in the file make.def, that is, lines ending\n           with the character \"\\\". Although it may be ugly, \n           you should be able to reformat without continuation\n           lines. The offending line is\n  %s\n", original_line);
exit(1);
}
return ;
}
int check_include_line(char * line, char * filename)
{
char * include_string = "include";
int _ret_val_0;
while ((( * include_string)!='\0')&&(( * line)==( * include_string)))
{
line ++ ;
include_string ++ ;
}
if (( * include_string)!='\0')
{
_ret_val_0=0;
return _ret_val_0;
}
if ( ! (( * __ctype_b_loc())[(int)( * line)]&((unsigned short int)_ISspace)))
{
_ret_val_0=0;
return _ret_val_0;
}
while (( * __ctype_b_loc())[(int)( * ( ++ line))]&((unsigned short int)_ISspace))
{
;
}
if (( * line)=='\0')
{
_ret_val_0=0;
return _ret_val_0;
}
while ((( * filename)!='\0')&&(( * line)==( * filename)))
{
line ++ ;
filename ++ ;
}
if ((( * filename)!='\0')||(((( * line)!=' ')&&(( * line)!='\0'))&&(( * line)!='\n')))
{
_ret_val_0=0;
return _ret_val_0;
}
else
{
_ret_val_0=1;
return _ret_val_0;
}
return _ret_val_0;
}
void put_string(FILE * fp, char * name, char * val)
{
int len;
len=strlen(val);
if (len>46)
{
val[46]='\0';
val[46-1]='.';
val[46-2]='.';
val[46-3]='.';
len=46;
}
fprintf(fp, "%scharacter*%d %s\n", "        ", len, name);
fprintf(fp, "%sparameter (%s=\'%s\')\n", "        ", name, val);
return ;
}
void put_def_string(FILE * fp, char * name, char * val)
{
int len;
len=strlen(val);
if (len>46)
{
val[46]='\0';
val[46-1]='.';
val[46-2]='.';
val[46-3]='.';
len=46;
}
fprintf(fp, "#define %s \"%s\"\n", name, val);
return ;
}
void put_def_variable(FILE * fp, char * name, char * val)
{
int len;
len=strlen(val);
if (len>46)
{
val[46]='\0';
val[46-1]='.';
val[46-2]='.';
val[46-3]='.';
len=46;
}
fprintf(fp, "#define %s %s\n", name, val);
return ;
}
int ilog2(int i)
{
int log2;
int exp2 = 1;
int _ret_val_0;
if (i<=0)
{
_ret_val_0=( - 1);
return _ret_val_0;
}
#pragma loop name ilog2#0 
for (log2=0; log2<20; log2 ++ )
{
if (exp2==i)
{
return log2;
}
exp2*=2;
}
_ret_val_0=( - 1);
return _ret_val_0;
}
void write_convertdouble_info(int type, FILE * fp)
{
switch (type)
{
case SP:
case BT:
case LU:
case FT:
case MG:
case EP:
case CG:
fprintf(fp, "#define\tCONVERTDOUBLE\tFALSE\n");
break;
}
return ;
}
