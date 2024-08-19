#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<float.h>
#include<cputils.h>
#define RADIUS_TYPE_1		3
#define RADIUS_TYPE_2_3		9
#define THRESHOLD	0.1f
typedef struct {
int x,y;
int type;
int target;
} Team;
typedef struct {
int x,y;
int start;
int heat;
int active; 
} FocalPoint;
#define accessMat( arr, exp1, exp2 )	arr[ (exp1) * columns + (exp2) ]
void show_usage( char *program_name ) {
fprintf(stderr,"Usage: %s <config_file> | <command_line_args>\n", program_name );
fprintf(stderr,"\t<config_file> ::= -f <file_name>\n");
fprintf(stderr,"\t<command_line_args> ::= <rows> <columns> <maxIter> <numTeams> [ <teamX> <teamY> <teamType> ... ] <numFocalPoints> [ <focalX> <focalY> <focalStart> <focalTemperature> ... ]\n");
fprintf(stderr,"\n");
}
#ifdef DEBUG
void print_status( int iteration, int rows, int columns, float *surface, int num_teams, Team *teams, int num_focal, FocalPoint *focal, float global_residual ) {
int i,j;
printf("Iteration: %d\n", iteration );
printf("+");
for( j=0; j<columns; j++ ) printf("---");
printf("+\n");
for( i=0; i<rows; i++ ) {
printf("|");
for( j=0; j<columns; j++ ) {
char symbol;
if ( accessMat( surface, i, j ) >= 1000 ) symbol = '*';
else if ( accessMat( surface, i, j ) >= 100 ) symbol = '0' + (int)(accessMat( surface, i, j )/100);
else if ( accessMat( surface, i, j ) >= 50 ) symbol = '+';
else if ( accessMat( surface, i, j ) >= 25 ) symbol = '.';
else symbol = '0';
int t;
int flag_team = 0;
for( t=0; t<num_teams; t++ ) 
if ( teams[t].x == i && teams[t].y == j ) { flag_team = 1; break; }
if ( flag_team ) printf("[%c]", symbol );
else {
int f;
int flag_focal = 0;
for( f=0; f<num_focal; f++ ) 
if ( focal[f].x == i && focal[f].y == j && focal[f].active == 1 ) { flag_focal = 1; break; }
if ( flag_focal ) printf("(%c)", symbol );
else printf(" %c ", symbol );
}
}
printf("|\n");
}
printf("+");
for( j=0; j<columns; j++ ) printf("---");
printf("+\n");
printf("Global residual: %f\n\n", global_residual);
}
#endif
int main(int argc, char *argv[]) {
int i,j,t;
int rows, columns, max_iter;
float *surface, *surfaceCopy;
int num_teams, num_focal;
Team *teams;
FocalPoint *focal;
if (argc<2) {
fprintf(stderr,"-- Error in arguments: No arguments\n");
show_usage( argv[0] );
exit( EXIT_FAILURE );
}
int read_from_file = ! strcmp( argv[1], "-f" );
if ( read_from_file ) {
if (argc<3) {
fprintf(stderr,"-- Error in arguments: file-name argument missing\n");
show_usage( argv[0] );
exit( EXIT_FAILURE );
}
FILE *args = cp_abrir_fichero( argv[2] );
if ( args == NULL ) {
fprintf(stderr,"-- Error in file: not found: %s\n", argv[1]);
exit( EXIT_FAILURE );
}	
int ok;
ok = fscanf(args, "%d %d %d", &rows, &columns, &max_iter);
if ( ok != 3 ) {
fprintf(stderr,"-- Error in file: reading rows, columns, max_iter from file: %s\n", argv[1]);
exit( EXIT_FAILURE );
}
surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
surfaceCopy = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
if ( surface == NULL || surfaceCopy == NULL ) {
fprintf(stderr,"-- Error allocating: surface structures\n");
exit( EXIT_FAILURE );
}
ok = fscanf(args, "%d", &num_teams );
if ( ok != 1 ) {
fprintf(stderr,"-- Error file, reading num_teams from file: %s\n", argv[1]);
exit( EXIT_FAILURE );
}
teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
if ( teams == NULL ) {
fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
exit( EXIT_FAILURE );
}
for( i=0; i<num_teams; i++ ) {
ok = fscanf(args, "%d %d %d", &teams[i].x, &teams[i].y, &teams[i].type);
if ( ok != 3 ) {
fprintf(stderr,"-- Error in file: reading team %d from file: %s\n", i, argv[1]);
exit( EXIT_FAILURE );
}
}
ok = fscanf(args, "%d", &num_focal );
if ( ok != 1 ) {
fprintf(stderr,"-- Error in file: reading num_focal from file: %s\n", argv[1]);
exit( EXIT_FAILURE );
}
focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
if ( focal == NULL ) {
fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
exit( EXIT_FAILURE );
}
for( i=0; i<num_focal; i++ ) {
ok = fscanf(args, "%d %d %d %d", &focal[i].x, &focal[i].y, &focal[i].start, &focal[i].heat);
if ( ok != 4 ) {
fprintf(stderr,"-- Error in file: reading focal point %d from file: %s\n", i, argv[1]);
exit( EXIT_FAILURE );
}
focal[i].active = 0;
}
}
else {
if (argc<6) {
fprintf(stderr, "-- Error in arguments: not enough arguments when reading configuration from the command line\n");
show_usage( argv[0] );
exit( EXIT_FAILURE );
}
rows = atoi( argv[1] );
columns = atoi( argv[2] );
max_iter = atoi( argv[3] );
surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
surfaceCopy = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
num_teams = atoi( argv[4] );
teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
if ( teams == NULL ) {
fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
exit( EXIT_FAILURE );
}
if ( argc < num_teams*3 + 5 ) {
fprintf(stderr,"-- Error in arguments: not enough arguments for %d teams\n", num_teams );
exit( EXIT_FAILURE );
}
for( i=0; i<num_teams; i++ ) {
teams[i].x = atoi( argv[5+i*3] );
teams[i].y = atoi( argv[6+i*3] );
teams[i].type = atoi( argv[7+i*3] );
}
int focal_args = 5 + i*3;
if ( argc < focal_args+1 ) {
fprintf(stderr,"-- Error in arguments: not enough arguments for the number of focal points\n");
show_usage( argv[0] );
exit( EXIT_FAILURE );
}
num_focal = atoi( argv[focal_args] );
focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
if ( teams == NULL ) {
fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
exit( EXIT_FAILURE );
}
if ( argc < focal_args + 1 + num_focal*4 ) {
fprintf(stderr,"-- Error in arguments: not enough arguments for %d focal points\n", num_focal );
exit( EXIT_FAILURE );
}
for( i=0; i<num_focal; i++ ) {
focal[i].x = atoi( argv[focal_args+i*4+1] );
focal[i].y = atoi( argv[focal_args+i*4+2] );
focal[i].start = atoi( argv[focal_args+i*4+3] );
focal[i].heat = atoi( argv[focal_args+i*4+4] );
focal[i].active = 0;
}
if ( argc > focal_args+i*4+1 ) {
fprintf(stderr,"-- Error in arguments: extra arguments at the end of the command line\n");
show_usage( argv[0] );
exit( EXIT_FAILURE );
}
}
#ifdef DEBUG
printf("Arguments, Rows: %d, Columns: %d, max_iter: %d\n", rows, columns, max_iter);
printf("Arguments, Teams: %d, Focal points: %d\n", num_teams, num_focal );
for( i=0; i<num_teams; i++ ) {
printf("\tTeam %d, position (%d,%d), type: %d\n", i, teams[i].x, teams[i].y, teams[i].type );
}
for( i=0; i<num_focal; i++ ) {
printf("\tFocal_point %d, position (%d,%d), start time: %d, temperature: %d\n", i, 
focal[i].x,
focal[i].y,
focal[i].start,
focal[i].heat );
}
#endif 
double ttotal = cp_Wtime();
#ifdef TIME
double tparallel = cp_Wtime(); 
double t3 = cp_Wtime();
double t41;
double t42;
double t43;
double t44;
#endif 
#pragma omp parallel for private(i,j) 
for( i=0; i<rows; i++ )
for( j=0; j<columns; j++ )
accessMat( surface, i, j ) = 0.0;
#ifdef TIME
t3 = cp_Wtime() - t3;
#endif
int iter;
int flag_stability = 0;
int mColumns = columns -1;
int mRows = rows -1;
int first_activation = 0;
for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {
#ifdef TIME
t41 = cp_Wtime();
#endif
int num_deactivated = 0;
for( i=0; i<num_focal; i++ ) {
if ( focal[i].start == iter ) {
focal[i].active = 1;
if ( ! first_activation ) first_activation = 1;
}
if ( focal[i].active == 2 ) {
num_deactivated++; 
}
}
#ifdef TIME
t41 = cp_Wtime() - t41;
#endif
#ifdef TIME
t42 = cp_Wtime();
#endif
float global_residual = 0.0f;
int step;
for( step=0; first_activation && step<10; step++ )	{		
for( i=0; i<num_focal; i++ ) {
if ( focal[i].active != 1 ) continue;
int x = focal[i].x;
int y = focal[i].y;
accessMat( surface, x, y ) = focal[i].heat;
}
float *tmp = surface;			
surface = surfaceCopy;
surfaceCopy = tmp;
#pragma omp parallel
{
#pragma omp for private(i, j) nowait				
for( i=1; i<mRows; i++ )	
for( j=1; j<mColumns; j++ ) 
accessMat( surface, i, j ) = (accessMat(surfaceCopy, i-1, j ) + accessMat(surfaceCopy, i+1, j ) +
accessMat(surfaceCopy, i, j-1 ) + accessMat(surfaceCopy, i, j+1 ) ) / 4;	
#pragma omp for private(i,j) reduction(max:global_residual) nowait
for( i=1; i<mRows; i++ ) {
for( j=1; j<mColumns; j++ ) {				
float difference = accessMat( surface, i, j ) - accessMat( surfaceCopy, i, j ); 
if ( difference > global_residual  ) global_residual = difference;
if(global_residual >= THRESHOLD) break;
}	
}
}
}	
#ifdef TIME
t42 = cp_Wtime() - t42;
#endif
if( num_deactivated == num_focal && global_residual < THRESHOLD) flag_stability = 1;
global_residual = 0.0f;
#ifdef TIME
t43 = cp_Wtime();
#endif
#pragma omp parallel for private(t) 
for( t=0; t<num_teams; t++ ) {
float distance = FLT_MAX;
int target = -1;			
for( j=0; j<num_focal; j++ ) { 
if ( focal[j].active != 1 ) continue; 
float dx = focal[j].x - teams[t].x;
float dy = focal[j].y - teams[t].y;
float local_square_distance = dx*dx + dy*dy;	
if ( local_square_distance < distance ) {
distance = local_square_distance;
target = j;
}
}
teams[t].target = target;
if ( target == -1 ) continue; 
if ( teams[t].type == 1 ) { 
if ( focal[target].x < teams[t].x ) teams[t].x--;		
else if ( focal[target].x > teams[t].x ) teams[t].x++;	
if ( focal[target].y < teams[t].y ) teams[t].y--;		
else if ( focal[target].y > teams[t].y ) teams[t].y++;	
}
else if ( teams[t].type == 2 ) { 
if ( focal[target].y < teams[t].y ) teams[t].y--;
else if ( focal[target].y > teams[t].y ) teams[t].y++;
else if ( focal[target].x < teams[t].x ) teams[t].x--;
else if ( focal[target].x > teams[t].x ) teams[t].x++;
}
else {
if ( focal[target].x < teams[t].x ) teams[t].x--;
else if ( focal[target].x > teams[t].x ) teams[t].x++;
else if ( focal[target].y < teams[t].y ) teams[t].y--;
else if ( focal[target].y > teams[t].y ) teams[t].y++;
}
}
#ifdef TIME
t43 = cp_Wtime() - t43;
#endif 
#ifdef TIME
t44 = cp_Wtime() - t44;
#endif 
for( t=0; t<num_teams; t++ ) {
int target = teams[t].target;
if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y && focal[target].active == 1 )
focal[target].active = 2;
int radius;
if ( teams[t].type == 1 ) radius = RADIUS_TYPE_1;
else radius = RADIUS_TYPE_2_3; 
int squareRadius = radius*radius;			
int xR = teams[t].x+radius;
int yR = teams[t].y+radius;
for( i=teams[t].x-radius; i<=xR; i++ ) {
for( j=teams[t].y-radius; j<=yR; j++ ) {
if ( i<1 || i>=mRows || j<1 || j>=mColumns ) continue; 
float dx = teams[t].x - i;
float dy = teams[t].y - j;
float squareDistance = dx*dx + dy*dy;			
if ( squareDistance <= squareRadius) {	
accessMat( surface, i, j ) *= ( 0.75 ); 
}
}
}
}
#ifdef TIME 
t44 = cp_Wtime() - t44;
#endif 
#ifdef DEBUG
print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif 
}
#ifdef TIME 
tparallel = cp_Wtime() -tparallel;
#endif
ttotal = cp_Wtime() - ttotal;
printf("\nTime: %lf\n", ttotal );
#ifdef TIME
printf("Time parallel: %lf\n",tparallel );
printf("Time 3: %lf\n",   t3);
printf("Time 4.1: %lf\n", t41*iter );
printf("Time 4.2: %lf\n", t42*iter );
printf("Time 4.3: %lf\n", t43*iter );
printf("Time 4.4: %lf\n", t44*iter );
#endif
printf("Result: %d", iter);
for (i=0; i<num_focal; i++)
printf(" %.6f", accessMat( surface, focal[i].x, focal[i].y ) );
printf("\n");
free( teams );
free( focal );
free( surface );
free( surfaceCopy );
return 0;
}
