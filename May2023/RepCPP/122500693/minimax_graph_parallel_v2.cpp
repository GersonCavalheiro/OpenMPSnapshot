#include <stdio.h>

#include <iostream>
#include <ratio>
#include <thread>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#define VALMAX 48


#include "minimax_graph_parallel.h"

CaseSuivante cs;

void init_position(Position* pos) {
for(int i=0;i<6;i++){
pos->_Cases[0][i]=0;
pos->_Cases[1][i]=0;
}
pos->_PionsPris[0]=pos->_PionsPris[1]=0;
}

void print_position(Position* pos){
printf("--------------------------\n");
for(int i = 5 ; i >= 0 ; i--){
printf("[%d] " , pos->_Cases[0][i]);
}
printf(" PP=%d\n",pos->_PionsPris[0]);
for(int i = 5 ; i >= 0 ; i--){
printf("[%d] " , pos->_Cases[1][i]);
}
printf(" PP=%d\n--------------------------\n",pos->_PionsPris[1]);
}

void print_position_ordi_bas_inv(Position* pos){
printf("--------------------------\n");
for(int i = 0 ; i <6 ; i++){
printf("[%d] " , pos->_Cases[1][i]);
}
printf(" PP=%d\n",pos->_PionsPris[1]);
for(int i = 0 ; i <6 ; i++){
printf("[%d] " , pos->_Cases[0][i]);
}
printf(" PP=%d\n--------------------------\n",pos->_PionsPris[0]);
}

void print_position_ordi_haut_inv(Position* pos){
printf("--------------------------\n");
for(int i = 5 ; i >= 0 ; i--){
printf("[%d] " , pos->_Cases[0][i]);
}
printf(" PP=%d\n",pos->_PionsPris[0]);
for(int i = 5 ; i >=0 ; i--){
printf("[%d] " , pos->_Cases[1][i]);
}
printf(" PP=%d\n--------------------------\n",pos->_PionsPris[1]);
}


void print_position_ordi_bas(Position* pos){
printf("--------------------------\n");
for(int i = 5 ; i >= 0 ; i--){
printf("[%d] " , pos->_Cases[1][i]);
}
printf(" PP=%d\n",pos->_PionsPris[1]);
for(int i = 5 ; i >= 0 ; i--){
printf("[%d] " , pos->_Cases[0][i]);
}
printf(" PP=%d\n--------------------------\n",pos->_PionsPris[0]);
}

void print_position_ordi_haut(Position* pos){
printf("--------------------------\n");
for(int i = 0 ; i < 6 ; i++){
printf("[%d] " , pos->_Cases[0][i]);
}
printf(" PP=%d\n",pos->_PionsPris[0]);
for(int i = 0 ; i <6 ; i++){
printf("[%d] " , pos->_Cases[1][i]);
}
printf(" PP=%d\n--------------------------\n",pos->_PionsPris[1]);
}

int est_affame(Position* pos, const int joueur){
int somme=0;
for(int i=0;i<6;i++){
somme += pos->_Cases[joueur][i];
}
return !somme;
}

void copier(Position* newPos, Position* pos){
for(int i=0;i<6;i++){
newPos->_Cases[0][i]=pos->_Cases[0][i];
newPos->_Cases[1][i]=pos->_Cases[1][i];
}
newPos->_PionsPris[0]=pos->_PionsPris[0];
newPos->_PionsPris[1]=pos->_PionsPris[1];
}

int jouer_coup(Position* newPos, Position* pos, const int joueur, const int coup){
const int nbpions=pos->_Cases[joueur][coup];
if (nbpions == 0){
return 0; 
}
copier(newPos,pos);
newPos->_Cases[joueur][coup]=0;
int j=joueur;
int c=coup;
for(int i=0;i<nbpions;i++){
const int tj=j;
j=cs._Jnext[j][c];
c=cs._Cnext[tj][c];
newPos->_Cases[j][c]++;
}
int nbp=newPos->_Cases[joueur][coup];
while (nbp != 0){
newPos->_Cases[joueur][coup]=0;
for(int i=0;i<nbp;i++){
const int tj=j;
j=cs._Jnext[j][c];
c=cs._Cnext[tj][c];
newPos->_Cases[j][c]++;
}
nbp=newPos->_Cases[joueur][coup];
}
if (j != joueur){
if (j ==0){
for(int i=c;i<=5;i++){
if (newPos->_Cases[j][i] == 2 || newPos->_Cases[j][i] == 3){
newPos->_PionsPris[joueur] += newPos->_Cases[j][i];
newPos->_Cases[j][i]=0;
} else {
break;
}
}
} else {
for(int i=c;i>=0;i--){
if (newPos->_Cases[j][i] == 2 || newPos->_Cases[j][i] == 3){
newPos->_PionsPris[joueur] += newPos->_Cases[j][i];
newPos->_Cases[j][i]=0;
} else {
break;
}
}
}
}
if (est_affame(newPos,!joueur)) return 0;
return 1;
}

inline
int evaluer(Position* pos){
return pos->_PionsPris[0] - pos->_PionsPris[1];
}

int test_fin(Position* pos){
if (pos->_PionsPris[0] + pos->_PionsPris[1] > 24){
if (48-pos->_PionsPris[0]-pos->_PionsPris[1] <= 6){
return 1;
}
if (pos->_PionsPris[0] >= 25) return 1;
if (pos->_PionsPris[1] >= 25) return 1;
}
return 0;
}


struct ECoup {
int _Val[6];
int _Coup[6];
};

typedef struct ECoup EvalCoup;

long long NUM_MINIMAX=0;
int VALMM=0;

void print_eval_coup(EvalCoup* ec, int nb){
for(int i=0;i<nb;i++){
std::cout << "coup: " << ec->_Coup[i] << " eval: " << ec->_Val[i] << std::endl;
}
}


int calculer_eval_coup(EvalCoup* ec, Position* pos,const int joueur, const int alpha, const int beta, const int pmax){
int nbv=0;
Position newPos;
for(int i=0;i<6;i++){
if (jouer_coup(&newPos,pos,joueur,i)){
ec->_Val[nbv]=valeur_minimax(&newPos,!joueur,alpha,beta,pmax-1);
ec->_Coup[nbv]=i;
nbv++;
}
}
return nbv;
}

int valeur_minimax(Position* pos,const int joueur, int alpha,int beta, const int pmax){
NUM_MINIMAX++;
if (pos->_PionsPris[0] + pos->_PionsPris[1] > 24){
if (48-pos->_PionsPris[0]-pos->_PionsPris[1] <= 6){
if (pos->_PionsPris[0] > pos->_PionsPris[1]) return VALMAX;
if (pos->_PionsPris[0] < pos->_PionsPris[1]) return -VALMAX;
return 0;
}
if (pos->_PionsPris[0] >= 25) return VALMAX;
if (pos->_PionsPris[1] >= 25) return -VALMAX;
}
if (pmax == 0) return evaluer(pos);
EvalCoup ec;
const int nbv=calculer_eval_coup(&ec,pos,joueur,alpha,beta,pmax);
int imin=0;
if (joueur==0){ 
for(int i=1;i<nbv;i++){
if (ec._Val[i] > ec._Val[imin]){
imin=i;
}
}
} else { 
for(int i=1;i<nbv;i++){
if (ec._Val[i] < ec._Val[imin]){
imin=i;
}
}
}
return ec._Val[imin];
}

int decision(Position* pos,int pmax){
int k=0;
for(int i=0;i<6;i++){
if (pos->_Cases[0][i] == 0) k++;
}
if (k >0){
if (pos->_PionsPris[1] + pos->_PionsPris[0] >= 20) pmax++;

if (k <= 3) pmax++;
if (k > 3) pmax +=2;
}
int alpha=-VALMAX-50;
int beta=VALMAX+50;
EvalCoup ec;
const int nbv=calculer_eval_coup(&ec,pos,0,alpha,beta,pmax);
int imin=0;
for(int i=1;i<nbv;i++){
if (ec._Val[i] > ec._Val[imin]){
imin=i;
}
}
VALMM=ec._Val[imin];
return ec._Coup[imin];
}

int valeur_minimaxAB(Position* pos,const int joueur, int alpha,int beta, const int pmax, const bool gagne);

int calculer_coup(Position* pos,const int joueur, int alpha, int beta, const int pmax,const bool gagne){
Position newPos;
if (joueur==0){
for(int i=0;i<6;i++){
if (jouer_coup(&newPos,pos,joueur,i)){
const int val=valeur_minimaxAB(&newPos,!joueur,alpha,beta,pmax-1,gagne);
if (val > alpha) {
alpha=val;
}
if (alpha >= beta){
return alpha;
}
}
}
return alpha;
}
for(int i=0;i<6;i++){
if (jouer_coup(&newPos,pos,joueur,i)){
const int val=valeur_minimaxAB(&newPos,!joueur,alpha,beta,pmax-1,gagne);
if (val < beta){
beta=val;
}
if (beta <= alpha){
return beta;
}
}
}
return beta;
}

int valeur_minimaxAB(Position* pos,const int joueur, int alpha,int beta, const int pmax, const bool gagne){
NUM_MINIMAX++;
int ajoutProf=(gagne) ? pmax : 0;
if (pos->_PionsPris[0] + pos->_PionsPris[1] > 24){
if (48-pos->_PionsPris[0]-pos->_PionsPris[1] <= 6){
if (pos->_PionsPris[0] > pos->_PionsPris[1]){
return VALMAX + ajoutProf;
}
if (pos->_PionsPris[0] < pos->_PionsPris[1]) return -VALMAX-ajoutProf;
return 0;
}
if (pos->_PionsPris[0] >= 25){
return VALMAX+ajoutProf;
}
if (pos->_PionsPris[1] >= 25) return -VALMAX-ajoutProf;
}
if (pmax == 0) return evaluer(pos);
return calculer_coup(pos,joueur,alpha,beta,pmax,gagne);
}


int decisionAB(Position* pos,int pmax, bool gagne){
int k=0;
for(int i=0;i<6;i++){
if (pos->_Cases[0][i] == 0) k++;
}
if (k >0){
if (pos->_PionsPris[1] + pos->_PionsPris[0] >= 20) pmax++;
if (pos->_PionsPris[1] + pos->_PionsPris[0] >= 25) pmax++;

if (k <= 3) pmax++;
if (k > 3) pmax +=2;
}

int alpha=-VALMAX-50; 
int beta=VALMAX+50; 
int coup;
int valeurs[81];
for(int i = 0; i < 6; i++){
valeurs[16 * i] = alpha;
}
for(coup=0;coup<6;coup++){
Position newPos;
if (jouer_coup(&newPos,pos,0,coup)){
alpha=valeur_minimaxAB(&newPos,1,alpha,beta,pmax-1,gagne);
valeurs[16 * coup] = alpha;
if(coup < 4){
break;
}
}
}
#pragma omp parallel for
for (int k = coup + 1; k < 6; k++) {
Position newPos;
if (jouer_coup(&newPos,pos,0,k)){
valeurs[16 * k]=valeur_minimaxAB(&newPos,1,alpha,beta,pmax-1,gagne);
}
}
coup = 0;
for(int i = 1; i < 6; i++){
if(valeurs[16 * i] > valeurs[16 * coup]){
coup = i;
}
}
VALMM=valeurs[16 * coup];
return coup;
}

void position_debut(Position* pos) {
for(int i=0;i<6;i++){
pos->_Cases[0][i]=4;
pos->_Cases[1][i]=4;
}
pos->_PionsPris[0]=pos->_PionsPris[1]=0;

}
int main(int argc, char* argv[]){
std::chrono::time_point<std::chrono::system_clock> start, end;

cs._Cnext[0][0]=0;
cs._Jnext[0][0]=1;
cs._Cnext[0][1]=0;
cs._Jnext[0][1]=0;
cs._Cnext[0][2]=1;
cs._Jnext[0][2]=0;
cs._Cnext[0][3]=2;
cs._Jnext[0][3]=0;
cs._Cnext[0][4]=3;
cs._Jnext[0][4]=0;
cs._Cnext[0][5]=4;
cs._Jnext[0][5]=0;

cs._Cnext[1][0]=1;
cs._Jnext[1][0]=1;
cs._Cnext[1][1]=2;
cs._Jnext[1][1]=1;
cs._Cnext[1][2]=3;
cs._Jnext[1][2]=1;
cs._Cnext[1][3]=4;
cs._Jnext[1][3]=1;
cs._Cnext[1][4]=5;
cs._Jnext[1][4]=1;
cs._Cnext[1][5]=5;
cs._Jnext[1][5]=0;

start = std::chrono::system_clock::now();

Position pos;
Position newPos;
position_debut(&pos);


int joueur;
if(scanf("%d",&joueur)){}

int ordiCommence= (joueur==0)? 1 : 0;
int fin=0;
bool gagne=false;
int numeroCoup = 1;
while(!fin){
int coup;
if (joueur == 0){ 
coup=decisionAB(&pos,17,gagne); 
if (!gagne && VALMM==48){gagne=true;}
int cj;
if (ordiCommence){
cj=6-coup;
} else {
cj=12-coup;
}
NUM_MINIMAX=0;
jouer_coup(&newPos,&pos,joueur,coup);
copier(&pos,&newPos);

} else { 
if (ordiCommence){
if(scanf("%d",&coup)){}
coup -=7;
} else {
if(scanf("%d",&coup)){}
coup--;
}

jouer_coup(&newPos,&pos,joueur,coup);

copier(&pos,&newPos);


}
fin=test_fin(&pos);
joueur = !joueur;
}
end = std::chrono::system_clock::now();
std::chrono::duration<double> elapsed_seconds = end - start;
std::cout << elapsed_seconds.count() << std::endl;
}


