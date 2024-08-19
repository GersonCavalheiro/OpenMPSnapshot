#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#define NUM_THRDS 4 
int chromo_length; 
int popl_size; 
int no_generation; 
float **dist_matrix; 
char citiesNameFile[255]; 
typedef struct{ 
int id;
float x;
float y;
} City;
City *cities;
typedef struct{
int * genes;
float fitness;
} Chromosome;
Chromosome * population;
void print_chromosome(Chromosome *ptr_chromosome){
printf("\nFitness = %f\t, Genes = ", ptr_chromosome->fitness);
for (int i = 0; i < chromo_length; i++)
printf("%d_", ptr_chromosome->genes[i]);
printf("\n");
}
void print_population(Chromosome *population){
for(int i = 0; i < popl_size; i++)
print_chromosome(&population[i]);
}
void parse_arguments(int argc, char **argv){
if (argc != 2){
printf("Must submit TSP locations test file name as argument.\n");
exit(0);
} 
else{
strcpy(citiesNameFile, argv[1]);
}
}
void read_cities_from_file(){
int nb_cities, line;
char buffer[1024];
float x, y;
FILE *fp;
fp = fopen(citiesNameFile, "r");
if (fscanf(fp, "DIMENSION : %d", &nb_cities) == 0){
printf("Illegal TSP locations file format. Expecting the DIMENSION at line 5.\n");
exit(0);
}
chromo_length = nb_cities;
cities = (City *) malloc(sizeof (City) * nb_cities);
rewind(fp);
for (int i = 0; i < 7; i++)
fgets(buffer, 1024, fp);
while (fscanf(fp, "%d %f %f", &line, &x, &y) > 0 && line <= nb_cities){
cities[line - 1].id = line;
cities[line - 1].x = x;
cities[line - 1].y = y;
}
fclose(fp);
}
float get_distance(City city1, City city2){
return sqrt(pow(city1.x - city2.x, 2) + pow(city1.y - city2.y, 2));
}
void print_dist_matrix(){
printf("\n\t");
for (int i = 0; i < chromo_length; ++i){
printf("\n|%d|\t", i);
for (int j = 0; j < chromo_length; j++)
printf("(%d,%d) = %.4f\t",i,j, dist_matrix[i][j]);
}
}
void init_dist_matrix(){
read_cities_from_file();
dist_matrix = malloc(sizeof (float *) * chromo_length);
#pragma omp parallel for num_threads(NUM_THRDS)
for (int i = 0; i < chromo_length; i++)
dist_matrix[i] = calloc(chromo_length, sizeof (float));
#pragma omp parallel for num_threads(NUM_THRDS)
for (int i = 0; i < chromo_length - 1; i++){
for (int j = i + 1; j < chromo_length; j++){
float distance = get_distance(cities[i], cities[j]);
dist_matrix[i][j] = dist_matrix[j][i] = distance;
}
}
free(cities);
}
void calculate_fitness(Chromosome *ptr_chromosome){
float fitness = 0;
int i = 0;
for (i = 0; i < chromo_length-1; i++)
fitness += dist_matrix[ptr_chromosome->genes[i] -1][ptr_chromosome->genes[i+1] -1];
fitness += dist_matrix[ptr_chromosome->genes[i]-1 ][0];
fitness = 10/log10(fitness);
ptr_chromosome-> fitness= fitness;
}
void calculate_population_fitness(Chromosome *population){
#pragma omp parallel for num_threads(NUM_THRDS)
for(int i = 0; i < popl_size; i++)
calculate_fitness(&population[i]);
}
int getRandomNumber(){
int seed = (unsigned)(time(NULL)+rand());
srand(seed);
return rand()%chromo_length;
}
void fill_randomly_the_chromosome(Chromosome *chrom){
int array[chromo_length];
chrom->genes = malloc(chromo_length * sizeof (int));
#pragma omp parallel for num_threads(NUM_THRDS)
for(int i = 0; i < chromo_length; i++)
array[i] = i+1;
for(int i = 0; i < chromo_length; i++){
int nbRand = getRandomNumber()%(chromo_length -i);
int tmp = array[nbRand];
array[nbRand] = array[chromo_length - i - 1];
array[chromo_length - i - 1] = tmp;
chrom->genes[i] = tmp;
}
calculate_fitness(chrom); 
}
void swap_chromosomes( Chromosome *pop, int src, int dest){
Chromosome chrom  = pop[src];
pop[src] = pop[dest];
pop[dest] = chrom;
}
void sort_population(Chromosome *population){
for(int i = 0; i < popl_size; i++)
for(int j = i+1; j < popl_size; j++)
if(population[i].fitness <population[j].fitness)
swap_chromosomes(population, i, j);
}
int get_random_index_of_chrom(){
int seed = (unsigned)(time(NULL)+rand());
srand(seed);
return rand()%popl_size;
}
void selection(Chromosome *pop){
int n = (40*popl_size)/100;
#pragma omp parallel for num_threads(NUM_THRDS)
for(int i = 0; i < (10*popl_size)/100; i++){ 
int randNb =(popl_size/2) + get_random_index_of_chrom()%(popl_size/2);
swap_chromosomes(population, n+i, randNb);
}
}
void print_fitness(){
printf("\n------------------------------------------------------Fitness-------------------------------------------\n");
for(int i = 0; i < popl_size; i++)
printf("%.3f - ",population[i].fitness);
printf("\n------------------------------------------------------Fitness-------------------------------------------------\n");
}
float percentage_of_difference(Chromosome chro1, Chromosome chro2){
float sum = 0;
for(int i = 0; i < chromo_length; i++)
if(chro1.genes[i] != chro2.genes[i])
sum++;
return (sum*100)/chromo_length;
}
int if_exist(Chromosome *chrom, int x){
for(int i=0; i<chromo_length; i++)
if(x == chrom->genes[i])
return 1;
return 0;
}
void create_ChildV2(Chromosome p, Chromosome m, Chromosome *Chro){
int z=1;
int n = getRandomNumber() % (chromo_length);
#pragma omp parallel for num_threads(NUM_THRDS)
for(int i=0; i < chromo_length;i++)
Chro->genes[i] =0;
for(int i = n; i < n+((chromo_length*30)/100);i++){
z=i%chromo_length;
Chro->genes[z]=p.genes[z];
}
int c=0;
int i=(z+1) % chromo_length;
while(i!=z){
c = c%chromo_length;
if(if_exist(Chro, m.genes[c]) != 1)
Chro->genes[i] = m.genes[c];
else{
if(Chro->genes[i] == 0){
while(if_exist(Chro, m.genes[c]) == 1)
c++;
Chro->genes[i] = m.genes[c];
}
}
c++;
i++;
i = i % chromo_length;
}
}
void crossoverV2(Chromosome *pop){
int nb=0;
for(int i = 0; i < (popl_size/2) ; i++){
do{
nb= getRandomNumber() % (popl_size/2);
}while(nb == i && percentage_of_difference(pop[i], pop[nb]) < 70);
create_ChildV2(pop[i], pop[nb], &pop[(popl_size/2) +i]);
}
}
void mutation(Chromosome *pop){
for(int z=0; z<5; z++){
int i = getRandomNumber()%(chromo_length);
int j = getRandomNumber()%(chromo_length);
int k = getRandomNumber()%(popl_size -(20*popl_size/100));
int temp = pop[(20*popl_size/100)+k].genes[j];
pop[(20*popl_size/100)+k].genes[j] = pop[(20*popl_size/100)+k].genes[i];
pop[(20*popl_size/100)+k].genes[i] = temp;
}
}
void main(int argc, char **argv){
popl_size = 200;
no_generation = 1000;
parse_arguments(argc, argv);
init_dist_matrix();
population = (Chromosome *)malloc(popl_size*sizeof(Chromosome));
double start= omp_get_wtime();
for(int i = 0; i < popl_size ; i++)
fill_randomly_the_chromosome(&population[i]);
sort_population(population); 
int  i = 0;
while(i < no_generation){
selection(population); 
crossoverV2(population); 
mutation(population); 
calculate_population_fitness(population); 
sort_population(population); 
i++;
}
double end= omp_get_wtime();
printf("Time: \t %f \n", ((end-start)));
}