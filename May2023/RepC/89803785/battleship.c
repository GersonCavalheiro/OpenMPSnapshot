#include<stdio.h>
#include<curses.h>
#include<unistd.h>
int active_row,active_col,counter=1;
int npc_active_row=0,npc_active_col=0;
int npcRemShips=5, playerRemShips=5;
int playerRemNodes[]={2,3,3,4,5};
int shipFinal=0;
int npcRemNodes[]={2,3,3,4,5}; 
char* shipNames[]={"Destroyer","Submarine","Cruiser","Battleship","Carrier"};
char letters[]={'A','B','C','D','E','F','G','H','I','J'};
WINDOW* logWindow;
WINDOW* targeterWindow;
WINDOW* shipWindow;
int shipNodes[5];
int npcShips[10][10]={
{0,0,0,0,0,0,0,0,0,0},
{0,0,0,0,0,5,0,1,0,0},
{0,0,0,0,0,5,0,1,0,0},  
{0,0,0,0,0,5,0,0,0,0},  
{0,0,0,0,0,5,0,0,0,0},  
{4,4,4,4,0,5,0,0,0,0},  
{0,0,0,0,0,0,0,0,0,0},  
{0,2,0,0,0,0,0,0,0,0},
{0,2,0,0,0,3,3,3,0,0},
{0,2,0,0,0,0,0,0,0,0}
};
int playerShootMap[10][10];
int npcShootMap[10][10];
int playerShips[10][10];
int isValueInArray(int value,int array[],int size){
#pragma omp parallel for
for(int j=0;j<size;j++){
if(value==array[j])
return 1;
}
return 0;
}
void generateTargetMap(){
wclear(targeterWindow);
wprintw(targeterWindow,"   1 2 3 4 5 6 7 8 9 10\n\n");
#pragma omp parallel for
for(int i=0;i<10;i++){
wprintw(targeterWindow,"%c ",letters[i]);
for(int j=0;j<10;j++){
if(i==active_row && j==active_col)
wprintw(targeterWindow," +");
else if(playerShootMap[i][j]==1)
wprintw(targeterWindow," x");
else if(playerShootMap[i][j]==0)
wprintw(targeterWindow," .");
else wprintw(targeterWindow," o");
}
wprintw(targeterWindow,"\n");
}	
wrefresh(targeterWindow);
}
void generateShipMap(int vertical,int num_nodes){
wclear(shipWindow);
wprintw(shipWindow,"   1 2 3 4 5 6 7 8 9 10\n\n");
if(vertical){
#pragma omp parallel for
for(int i=0;i<num_nodes;i++){
shipNodes[i]=i+active_row;
}
#pragma omp parallel for
for(int i=0;i<10;i++){
wprintw(shipWindow,"%c ",letters[i]);
for(int j=0;j<10;j++){
if(npcShootMap[i][j]!=0)
wprintw(shipWindow," x");
else if(playerShips[i][j]!=0)
wprintw(shipWindow," +");
else if((!shipFinal)&&isValueInArray(i,shipNodes,num_nodes) && j==active_col)
wprintw(shipWindow," |");
else wprintw(shipWindow," .");
}
wprintw(shipWindow,"\n");
}	
}
else{
#pragma omp parallel for
for(int i=0;i<num_nodes;i++){
shipNodes[i]=i+active_col;
}
#pragma omp parallel for
for(int i=0;i<10;i++){
wprintw(shipWindow,"%c ",letters[i]);
for(int j=0;j<10;j++){
if(npcShootMap[i][j]!=0)
wprintw(shipWindow," x");
else if(playerShips[i][j]!=0)
wprintw(shipWindow," +");
else if((!shipFinal)&&isValueInArray(j,shipNodes,num_nodes) && i==active_row)
wprintw(shipWindow," _");
else wprintw(shipWindow," .");
}
wprintw(shipWindow,"\n");
}	
}
wrefresh(shipWindow);
}
void place(int vertical,int num_nodes){
#pragma omp parallel for
for(int i=0;i<num_nodes;i++){
if(vertical)
playerShips[shipNodes[i]][active_col]=counter;
else playerShips[active_row][shipNodes[i]]=counter;
}
counter++;
}
void shoot(){
if(playerShootMap[active_row][active_col]==0){
if(npcShips[active_row][active_col]!=0){
playerShootMap[active_row][active_col]=1;
if(--npcRemNodes[npcShips[active_row][active_col]-1]==0){
wprintw(logWindow,"\n%c-%d: Successfully destroyed the %s",letters[active_row],active_col+1,shipNames[npcShips[active_row][active_col]-1]);
npcRemShips--;
}
else wprintw(logWindow,"\n%c-%d: Successfully damaged the %s",letters[active_row],active_col+1,shipNames[npcShips[active_row][active_col]-1]);	
npcShips[active_row][active_col]=0;
if(npcRemShips==0){
wclear(logWindow);
wprintw(logWindow,"\nYOU WON!");
}
}	
else {
wprintw(logWindow,"\n%c-%d: Failed to hit any target",letters[active_row],active_col+1);
playerShootMap[active_row][active_col]=-1;
}
wrefresh(logWindow);
generateTargetMap();
usleep(50000);
npcShoot();
}	
}
void npcShoot(){
if(npcShootMap[npc_active_row][npc_active_col]==0){
if(playerShips[npc_active_row][npc_active_col]!=0){
npcShootMap[npc_active_row][npc_active_col]=1;
if(--playerRemNodes[playerShips[npc_active_row][npc_active_col]-1]==0){
wprintw(logWindow,"\n%c-%d: Computer successfully destroyed your %s",letters[npc_active_row],npc_active_col+1,shipNames[playerShips[npc_active_row][npc_active_col]-1]);
playerRemShips--;
}
else wprintw(logWindow,"\n%c-%d: Computer successfully damaged your %s",letters[npc_active_row],npc_active_col+1,shipNames[playerShips[npc_active_row][npc_active_col]-1]);	
playerShips[npc_active_row][npc_active_col]=0;
if(playerRemShips==0){
wclear(logWindow);
wprintw(logWindow,"\nComputer WON!");
}
}	
else {
wprintw(logWindow,"\n%c-%d: Computer failed to hit any target",letters[npc_active_row],npc_active_col+1);
npcShootMap[npc_active_row][npc_active_col]=-1;
}
if(npc_active_col<9)
npc_active_col++;
else {
npc_active_row++;
npc_active_col=0;
}
wrefresh(logWindow);
generateTargetMap();
generateShipMap(1,npcRemNodes[0]);
}	
}
int main(){
initscr();
noecho();
curs_set(0);
#pragma omp parallel for
for(int i=0;i<10;i++){
for(int j=0;j<10;j++)
playerShootMap[i][j]=npcShootMap[i][j]=playerShips[i][j]=0;
}	
printw("        ---BATTLESHIP---\n KEY: x->Hit shot o->Missed shot +->Crosshair\n CONTROLS: LEFT-ARROW->Move Left RIGHT-ARROW->Move Right\n           UP-ARROW->Move Up DOWN-ARROW->Move Down SPACE->Shoot");
refresh();
targeterWindow=newwin(15,25,5,2);
shipWindow=newwin(25,25,20,2);
logWindow=newwin(60,60,5,30);
active_row=0;
active_col=0;
int vertical=1;
char input;
generateShipMap(vertical,npcRemNodes[0]);
for(int i=0;i<5;i++){
while(1){	
input=getch();
if (input == '\033') {
getch();
switch(getch()) { 
case 'A':
if(active_row>0)
active_row--;
generateShipMap(vertical,npcRemNodes[i]);
break;
case 'B':
if((active_row<(10-npcRemNodes[i]) && vertical==1)||(active_row<9 && vertical==0))
active_row++;
generateShipMap(vertical,npcRemNodes[i]);
break;
case 'C':
if((active_col<(10-npcRemNodes[i]) && vertical==0)||(active_col<9 && vertical==1))
active_col++;
generateShipMap(vertical,npcRemNodes[i]);
break;
case 'D':
if(active_col>0)
active_col--;
generateShipMap(vertical,npcRemNodes[i]);
break;
}
} 
else if (input==' '){
place(vertical,npcRemNodes[i]);
break;
}
else if (input=='r'){
vertical=!vertical;
generateShipMap(vertical,npcRemNodes[i]);
}
}
generateShipMap(vertical,npcRemNodes[i]);
}		
shipFinal=1;
active_row=4;
active_col=4;
generateTargetMap();
while(1){	
input=getch();
if (input == '\033') {
getch();
switch(getch()) { 
case 'A':
if(active_row==0)
active_row=9;
else active_row--;
generateTargetMap();
break;
case 'B':
if(active_row==9)
active_row=0;
else active_row++;
generateTargetMap();
break;
case 'C':
if(active_col==9)
active_col=0;
else active_col++;
generateTargetMap();
break;
case 'D':
if(active_col==0)
active_col=9;
else active_col--;
generateTargetMap();
break;
}
} 
else if (input==' ')
shoot();
}
endwin();
return 0;
}