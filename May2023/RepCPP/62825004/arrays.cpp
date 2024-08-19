
#include "arrays.h"

void CargarVectorDesordenado(ty_vec&v)
{
cout<<"\nCargando vector desordenado...\n";
int f;
int n;
for(f=0; f<TOPEF; f++)
{
n= rand() %TOPEF+1;
v[f]=n;
}
cout<<"Fin!\n";
}

void Burbujeo(ty_vec v2)
{
int i;
int iAux;
int iPaso=1;
bool bOrd;
clock_t t;
t = clock();

cout<<"Realizando ordenamiento utilizando burbujeo LINEAL...\n";
do
{
bOrd=TRUE;
for(i=0;i<TOPEF-iPaso;i++)
{
if(v2[i]>v2[i+1])
{
iAux	= v2[i];
v2[i]	= v2[i+1];
v2[i+1]	= iAux;
bOrd=FALSE;
}
}
iPaso++;
}while(!bOrd);
t = clock() - t;
cout<<"Ordenamiento realizado.\n";
cout<<"Tardo "<<t<<" ticks del reloj. ( "<< ((float)t)/CLOCKS_PER_SEC<<" segundos)";
}

void BurbujeoParalelo(ty_vec v2)
{
int i;
int fase;
int tmp;
int iPos;
clock_t t;
t = clock();

cout<<"Realizando ordenamiento utilizando burbujeo PARALEO...\n";
#pragma omp parallel default(none) shared(v2) private(i, tmp, fase)
for (fase=0; fase<TOPEF;fase++)
{
if(fase % 2 == 0)
for (i=1;i<TOPEF;i+=2)
{
if(v2[i-1]>v2[i])
{
tmp = v2[i-1];
v2[i-1] = v2[i];
v2[i] = tmp;
}
}
else
for(i=1;i<TOPEF-1;i+=2)
{
if(v2[i]>v2[i+1])
{
tmp=v2[i+1];
v2[i+1] = v2[i];
v2[i]= tmp;
}
}
}
t = clock() - t;
cout<<"Ordenamiento realizado.\n";
cout<<"Tardo "<<t<<" ticks del reloj. ( "<< ((float)t)/CLOCKS_PER_SEC<<" segundos)\n";
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
while(!(cin>>iPos))
{
LimpiarCin();
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}
while(iPos!=-1)
{
cout<<"En la posicion "<<iPos<<" se encuentra el nro "<<v2[iPos]<<endl;
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
while(!(cin>>iPos))
{
LimpiarCin();
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}
}
}

void BurbujeoBidireccional(ty_vec v2)
{
register int a;
int exchange;
int t;
clock_t ti;
ti = clock();

cout<<"Realizando ordenamiento utilizando burbujeo bidireccional LINEAL...\n";
do 
{
exchange = 0;
for(a=TOPEF-1; a>0; --a)
{
if(v2[a-1] > v2[a])
{
t = v2[a-1];
v2[a-1] = v2[a];
v2[a] = t;
exchange = 1;
}
}
for(a=1; a < TOPEF; ++a)
{
if(v2[a-1] > v2[a])
{
t = v2[a-1];
v2[a-1] = v2[a];
v2[a] = t;
exchange = 1;
}
}
}while(exchange);
ti = clock() - ti;
cout<<"Ordenamiento realizado.\n";
cout<<"Tardo "<<ti<<" ticks del reloj. ( "<< ((float)ti)/CLOCKS_PER_SEC<<" segundos)\n";
}

void BurbujeoBidireccionalParalelo(ty_vec v2)
{
register int a;
int exchange;
int t=0;
long iPos;
clock_t ti;
ti = clock();

cout<<"Realizando ordenamiento utilizando burbujeo bidireccional PARALELO...\n";
#pragma omp parallel default(none) shared(v2) private(a, t, exchange)
do 
{
exchange = 0;
for(a=TOPEF-1; a>0; --a)
{
if(v2[a-1] > v2[a])
{
t = v2[a-1];
v2[a-1] = v2[a];
v2[a] = t;
exchange = 1;
}
}
for(a=1; a < TOPEF; ++a)
{
if(v2[a-1] > v2[a])
{
t = v2[a-1];
v2[a-1] = v2[a];
v2[a] = t;
exchange = 1;
}
}
}while(exchange);
ti = clock() - ti;
cout<<"Ordenamiento realizado.\n";
cout<<"Tardo "<<ti<<" ticks del reloj. ( "<< ((float)ti)/CLOCKS_PER_SEC<<" segundos)\n";
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
while(!(cin>>iPos))
{
LimpiarCin();
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}
while(iPos!=-1)
{
cout<<"En la posicion "<<iPos<<" se encuentra el nro "<<v2[iPos]<<endl;
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
while(!(cin>>iPos))
{
LimpiarCin();
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}
}
}

void Shell(ty_vec v2)
{
int i, j, incrmnt=0, temp;
clock_t t;
t = clock();

cout<<"Realizando ordenamiento utilizando Shell LINEAL...";
incrmnt = TOPEF/2;
while (incrmnt > 0)
{
for (i=incrmnt; i < TOPEF; i++)
{
j = i;
temp = v2[i];
while ((j >= incrmnt) && (v2[j-incrmnt] > temp))
{
v2[j] = v2[j - incrmnt];
j = j - incrmnt;
}
v2[j] = temp;
}
incrmnt /= 2;
}
t = clock() - t;
cout<<"Ordenamiento realizado.\n";
cout<<"Tardo "<<t<<" ticks del reloj. ( "<< ((float)t)/CLOCKS_PER_SEC<<" segundos)\n";
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}

void ShellParalelo(ty_vec v2)
{
int i=0, j=0, incrmnt=0, temp=0;
long iPos;
clock_t t;
t = clock();

cout<<"Realizando ordenamiento utilizando Shell PARALEO...";
incrmnt = TOPEF/2;
#pragma omp parallel default(none) firstprivate(incrmnt) shared(v2) private(i, j, temp)
{
while (incrmnt > 0)
{
#pragma omp for
for (i=incrmnt; i < TOPEF; i++)
{
j = i;
temp = v2[i];
while ((j >= incrmnt) && (v2[j-incrmnt] > temp))
{
v2[j] = v2[j - incrmnt];
j = j - incrmnt;
}
v2[j] = temp;
}
incrmnt /= 2;
}
}
t = clock() - t;
cout<<"Ordenamiento realizado.\n";
cout<<"Tardo "<<t<<" ticks del reloj. ( "<< ((float)t)/CLOCKS_PER_SEC<<" segundos)\n";
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
while(!(cin>>iPos))
{
LimpiarCin();
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}
while(iPos!=-1)
{
cout<<"En la posicion "<<iPos<<" se encuentra el nro "<<v2[iPos]<<endl;
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
while(!(cin>>iPos))
{
LimpiarCin();
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}
}
}

void InsertionSort(ty_vec v2)
{
int i, j, index;
clock_t t;
t = clock();

cout<<"Realizando ordenamiento utilizando InsertionSort LINEAL...";
for (i=1; i<TOPEF; i++)
{
index = v2[i];
j = i;
while ((j>0) && (v2[j-1] > index))
{
v2[j] = v2[j-1];
j = j - 1;
}
v2[j] = index;
}
t = clock() - t;
cout<<"Ordenamiento realizado.\n";
cout<<"Tardo "<<t<<" ticks del reloj. ( "<< ((float)t)/CLOCKS_PER_SEC<<" segundos)\n";
}

void InsertionSortParalelo(ty_vec v2)
{
int i, j, index;
long iPos;
clock_t t;
t = clock();

cout<<"Realizando ordenamiento utilizando InsertionSort PARALELO...";
#pragma omp parallel for default(none) shared(v2) private(i, j, index)
for (i=1; i<TOPEF; i++)
{
index = v2[i];
j = i;
while ((j>0) && (v2[j-1] > index))
{
v2[j] = v2[j-1];
j = j - 1;
}
v2[j] = index;
}
t = clock() - t;
cout<<"Ordenamiento realizado.\n";
cout<<"Tardo "<<t<<" ticks del reloj. ( "<< ((float)t)/CLOCKS_PER_SEC<<" segundos)\n";
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
while(!(cin>>iPos))
{
LimpiarCin();
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}
while(iPos!=-1)
{
cout<<"En la posicion "<<iPos<<" se encuentra el nro "<<v2[iPos]<<endl;
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
while(!(cin>>iPos))
{
LimpiarCin();
cout<<"Ingrese la posicion que desea ver (-1 para volver al menu principal):";
}
}
}

void Menu()
{
cout<<"Presione:\n";
cout<<"0 para salir.\n";
cout<<"1 para utilizar burbujeo.\n";
cout<<"2 para utilizar de a burbujeo bidireccional.\n";
cout<<"3 para utilizar shell.\n";
cout<<"4 para utilizar InsertionSort.\n";
}