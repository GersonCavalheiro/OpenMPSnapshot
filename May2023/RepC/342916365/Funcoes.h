

void DefineNucleos(){

#ifdef _WIN32 
#ifndef _SC_NPROCESSORS_ONLN
SYSTEM_INFO info;
GetSystemInfo(&info);
#define sysconf(a) info.dwNumberOfProcessors
#define _SC_NPROCESSORS_ONLN
#endif
#endif

nprocs = sysconf(_SC_NPROCESSORS_ONLN); 

}	

void IniciaFila(struct Fila *fila){
fila->frente = 0;
fila->tras = 0;
strcpy(fila->linha[fila->frente].conteudo, "");
}

void Enfileira(struct Fila *fila){

struct Linha l;

FILE *pont_arq;
char texto_str[tamLinha];

pont_arq = fopen("./Arquivo/arquivo.txt", "r");

while(fgets(texto_str, tamLinha, pont_arq) != NULL){

strcpy(l.conteudo, texto_str);

strcpy(fila->linha[fila->tras].conteudo, l.conteudo);
fila->tras=fila->tras+1;	
}
}

int Desenfileira(struct Fila *fila){

int i=0, j=0, k=0;

char l[tamLinha] = "";

#pragma omp critical 
{
if (fila->frente < fila->tras){

for (i=0;i<strlen(fila->linha[fila->frente].conteudo);i++){
l[i] = toupper (fila->linha[fila->frente].conteudo[i]);	
}

Substring_count(l, palavra);
}

strcpy(fila->linha[fila->frente].conteudo, " ");

fila->frente += 1;
}

if (fila->frente > fila->tras){
return 0;
}

return 1;
}

void Substring_count(char* string, char* substring) {
int i, j, l1, l2;
int found = 0;

l1 = strlen(string);
l2 = strlen(substring);

for(i = 0; i < l1; i++) {
found = 1;
for(j = 0; j < l2; j++) {
if(string[i+j] != substring[j]) {
found = 0;
break;
}
}

if(found == 1) {
if (i == 0 && i != (l1-l2)){
if (string[i+l2] == ' ' || string[i+l2] == '?'|| string[i+l2] == '!' || string[i+l2] == '.' || string[i+l2] == ','){
contador++;
i = i + l2 -1;	
}		
}
else if (string[i-1] == ' ' && (string[i+l2] == ' ' || string[i+l2] == '?'|| string[i+l2] == '!' || string[i+l2] == '.' || string[i+l2] == ',')){
contador++;
i = i + l2 -1;	

}
else if (i == (l1-l2-1)){
if (string[i-1] == ' '){
contador++;
i = i + l2 -1;	
}
}
else if (i == 0 && i == (l1-l2)){
contador++;
i = i + l2 - 1;
}
else if ( i == l1-l2){
if (string[i-1] == ' '){
contador++;
i = i+l2-1;
}
}
}			
}

}

void ExibeFila(struct Fila *fila){
int i;
for (i=fila->frente;i<fila->tras;i++){
printf ("%s", fila->linha[i].conteudo);
}

}

void TransformaPalavra(char palavraPesquisada[]){
strcpy(palavra, palavraPesquisada);

int i; 

for (i=0;i<strlen(palavra);i++){
palavra[i] = toupper(palavra[i]);
}
}

void ExibeOcorrencia(){

printf ("----------- INFORMACOES DO SISTEMA -------------\n\n");
printf ("       %d NUCLEOS DO PROCESSADOR EM USO\n\n\n", nprocs);

printf ("----------------- OCORRENCIAS ------------------\n\n");
printf (" %d OCORRENCIA(S) DA PALAVRA '%s' NO TEXTO\n", contador, palavra);
}

