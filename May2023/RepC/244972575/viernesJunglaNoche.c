#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <cputils.h>
typedef struct
{
float pos_row, pos_col;		  
float mov_row, mov_col;		  
float choose_mov[3];		  
float storage;				  
int age;					  
unsigned short random_seq[3]; 
bool alive;					  
} Cell;
typedef struct
{
int history_total_cells;	 
int history_dead_cells;		 
int history_max_alive_cells; 
int history_max_new_cells;   
int history_max_dead_cells;  
int history_max_age;		 
float history_max_food;		 
} Statistics;
#define accessMat(arr, exp1, exp2) arr[(int)(exp1)*columns + (int)(exp2)]
void cell_new_direction(Cell *cell)
{
float angle = (float)(2 * M_PI * erand48(cell->random_seq));
cell->mov_row = sinf(angle);
cell->mov_col = cosf(angle);
}
void cell_mutation(Cell *cell)
{
int mutation_type = (int)(4 * erand48(cell->random_seq));
float mutation_percentage = (float)(0.5 * erand48(cell->random_seq));
float mutation_value;
switch (mutation_type)
{
case 0:
mutation_value = cell->choose_mov[1] * mutation_percentage;
cell->choose_mov[1] -= mutation_value;
cell->choose_mov[0] += mutation_value;
break;
case 1:
mutation_value = cell->choose_mov[0] * mutation_percentage;
cell->choose_mov[0] -= mutation_value;
cell->choose_mov[1] += mutation_value;
break;
case 2:
mutation_value = cell->choose_mov[2] * mutation_percentage;
cell->choose_mov[2] -= mutation_value;
cell->choose_mov[1] += mutation_value;
break;
case 3:
mutation_value = cell->choose_mov[1] * mutation_percentage;
cell->choose_mov[1] -= mutation_value;
cell->choose_mov[2] += mutation_value;
break;
default:
fprintf(stderr, "Error: Imposible type of mutation\n");
exit(EXIT_FAILURE);
}
cell->choose_mov[2] = 1.0f - cell->choose_mov[1] - cell->choose_mov[0];
}
#ifdef DEBUG
void print_status(int iteration, int rows, int columns, float *culture, int num_cells, Cell *cells, int num_cells_alive, Statistics sim_stat)
{
int i, j;
printf("Iteration: %d\n", iteration);
printf("+");
for (j = 0; j < columns; j++)
printf("---");
printf("+\n");
for (i = 0; i < rows; i++)
{
printf("|");
for (j = 0; j < columns; j++)
{
char symbol;
if (accessMat(culture, i, j) >= 20)
symbol = '+';
else if (accessMat(culture, i, j) >= 10)
symbol = '*';
else if (accessMat(culture, i, j) >= 5)
symbol = '.';
else
symbol = ' ';
int t;
int counter = 0;
for (t = 0; t < num_cells; t++)
{
int row = (int)(cells[t].pos_row);
int col = (int)(cells[t].pos_col);
if (cells[t].alive && row == i && col == j)
{
counter++;
}
}
if (counter > 9)
printf("(M)");
else if (counter > 0)
printf("(%1d)", counter);
else
printf(" %c ", symbol);
}
printf("|\n");
}
printf("+");
for (j = 0; j < columns; j++)
printf("---");
printf("+\n");
printf("Num_cells_alive: %04d\nHistory( Cells: %04d, Dead: %04d, Max.alive: %04d, Max.new: %04d, Max.dead: %04d, Max.age: %04d, Max.food: %6f )\n\n",
num_cells_alive,
sim_stat.history_total_cells,
sim_stat.history_dead_cells,
sim_stat.history_max_alive_cells,
sim_stat.history_max_new_cells,
sim_stat.history_max_dead_cells,
sim_stat.history_max_age,
sim_stat.history_max_food);
}
#endif
void show_usage(char *program_name)
{
fprintf(stderr, "Usage: %s ", program_name);
fprintf(stderr, "<rows> <columns> <maxIter> <max_food> <food_density> <food_level> <short_rnd1> <short_rnd2> <short_rnd3> <num_cells>\n");
fprintf(stderr, "\tOptional arguments for special food spot: [ <row> <col> <size_rows> <size_cols> <density> <level> ]\n");
fprintf(stderr, "\n");
}
int main(int argc, char *argv[])
{
int i, j;
int max_iter;		  
int rows, columns;	
float *culture;		  
short *culture_cells; 
float max_food;		
float food_density; 
float food_level;   
bool food_spot_active = false;  
int food_spot_row = 0;			
int food_spot_col = 0;			
int food_spot_size_rows = 0;	
int food_spot_size_cols = 0;	
float food_spot_density = 0.0f; 
float food_spot_level = 0.0f;   
unsigned short init_random_seq[3];		
unsigned short food_random_seq[3];		
unsigned short food_spot_random_seq[3]; 
int num_cells; 
Cell *cells;   
Statistics sim_stat;
sim_stat.history_total_cells = 0;
sim_stat.history_dead_cells = 0;
sim_stat.history_max_alive_cells = 0;
sim_stat.history_max_new_cells = 0;
sim_stat.history_max_dead_cells = 0;
sim_stat.history_max_age = 0;
sim_stat.history_max_food = 0.0f;
if (argc < 11)
{
fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
show_usage(argv[0]);
exit(EXIT_FAILURE);
}
rows = atoi(argv[1]);
columns = atoi(argv[2]);
max_iter = atoi(argv[3]);
max_food = atof(argv[4]);
food_density = atof(argv[5]);
food_level = atof(argv[6]);
for (i = 0; i < 3; i++)
{
init_random_seq[i] = (unsigned short)atoi(argv[7 + i]);
}
num_cells = atoi(argv[10]);
if (argc > 11)
{
if (argc < 17)
{
fprintf(stderr, "-- Error in number of special-food-spot arguments in the command line\n\n");
show_usage(argv[0]);
exit(EXIT_FAILURE);
}
else
{
food_spot_active = true;
food_spot_row = atoi(argv[11]);
food_spot_col = atoi(argv[12]);
food_spot_size_rows = atoi(argv[13]);
food_spot_size_cols = atoi(argv[14]);
food_spot_density = atof(argv[15]);
food_spot_level = atof(argv[16]);
if (argc > 17)
{
fprintf(stderr, "-- Error: too many arguments in the command line\n\n");
show_usage(argv[0]);
exit(EXIT_FAILURE);
}
}
}
#ifdef DEBUG
printf("Arguments, Rows: %d, Columns: %d, max_iter: %d\n", rows, columns, max_iter);
printf("Arguments, Max.food: %f, Food density: %f, Food level: %f\n", max_food, food_density, food_level);
printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", init_random_seq[0], init_random_seq[1], init_random_seq[2]);
if (food_spot_active)
{
printf("Arguments, Food_spot, pos(%d,%d), size(%d,%d), Density: %f, Level: %f\n",
food_spot_row, food_spot_col, food_spot_size_rows, food_spot_size_cols, food_spot_density, food_spot_level);
}
printf("Initial cells: %d\n", num_cells);
#endif 
for (i = 0; i < 3; i++)
{
food_random_seq[i] = (unsigned short)nrand48(init_random_seq);
food_spot_random_seq[i] = (unsigned short)nrand48(init_random_seq);
}
cells = (Cell *)malloc(sizeof(Cell) * (size_t)num_cells);
if (cells == NULL)
{
fprintf(stderr, "-- Error allocating: %d cells\n", num_cells);
exit(EXIT_FAILURE);
}
for (i = 0; i < num_cells; i++)
{
for (j = 0; j < 3; j++)
cells[i].random_seq[j] = (unsigned short)nrand48(init_random_seq);
}
#ifdef DEBUG
#endif 
double ttotal = cp_Wtime();
culture = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)columns);
culture_cells = (short *)malloc(sizeof(short) * (size_t)rows * (size_t)columns);
if (culture == NULL || culture_cells == NULL)
{
fprintf(stderr, "-- Error allocating culture structures for size: %d x %d \n", rows, columns);
exit(EXIT_FAILURE);
}
#pragma omp parallel for default(none) shared(rows, columns, culture)     schedule(guided)
for (i = 0; i < rows * columns; i++)
culture[i] = 0.0f;
#pragma omp parallel for default(shared) schedule(guided)
for (i = 0; i < num_cells; i++)
{
cells[i].alive = true;
cells[i].age = 1 + (int)(19 * erand48(cells[i].random_seq));
cells[i].storage = (float)(10 + 10 * erand48(cells[i].random_seq));
cells[i].pos_row = (float)(rows * erand48(cells[i].random_seq));
cells[i].pos_col = (float)(columns * erand48(cells[i].random_seq));
cell_new_direction(&cells[i]);
cells[i].choose_mov[0] = 0.33f;
cells[i].choose_mov[1] = 0.34f;
cells[i].choose_mov[2] = 0.33f;
}
sim_stat.history_total_cells = num_cells;
sim_stat.history_max_alive_cells = num_cells;
#ifdef DEBUG
printf("Initial cells data: %d\n", num_cells);
for (i = 0; i < num_cells; i++)
{
printf("\tCell %d, Pos(%f,%f), Mov(%f,%f), Choose_mov(%f,%f,%f), Storage: %f, Age: %d\n",
i,
cells[i].pos_row,
cells[i].pos_col,
cells[i].mov_row,
cells[i].mov_col,
cells[i].choose_mov[0],
cells[i].choose_mov[1],
cells[i].choose_mov[2],
cells[i].storage,
cells[i].age);
}
#endif 
float current_max_food = 0.0f;
int num_cells_alive = num_cells;
int iter;
int max_age = 0;
int num_new_sources = (int)(rows * columns * food_density);
int num_new_sources_spot = food_spot_active ? (int)(food_spot_size_rows * food_spot_size_cols * food_spot_density) : 0;
double rand41[3 * num_new_sources];
double rand41s[3 * num_new_sources_spot];
for (iter = 0; iter < max_iter && current_max_food <= max_food && num_cells_alive > 0; iter++)
{
int step_new_cells = 0;
int step_dead_cells = 0;
for (i = 0; i < num_new_sources; i++)
{
rand41[3 * i] = erand48(food_random_seq);
rand41[3 * i + 1] = erand48(food_random_seq);
rand41[3 * i + 2] = erand48(food_random_seq);
}
for (i = 0; i < num_new_sources; i++)
{
int row = (int)(rows * rand41[3 * i]);
int col = (int)(columns * rand41[3 * i + 1]);
float food = (float)(food_level * rand41[3 * i + 2]);
accessMat( culture, row, col ) += food;
}
if (food_spot_active)
{
for (i = 0; i < num_new_sources_spot; i++)
{
rand41s[3 * i] = erand48(food_spot_random_seq);
rand41s[3 * i + 1] = erand48(food_spot_random_seq);
rand41s[3 * i + 2] = erand48(food_spot_random_seq);
}
for (i = 0; i < num_new_sources_spot; i++)
{
int row = food_spot_row + (int)(food_spot_size_rows * rand41s[3 * i]);
int col = food_spot_col + (int)(food_spot_size_cols * rand41s[3 * i + 1]);
float food = (float)(food_spot_level * rand41s[3 * i + 2]);
accessMat( culture, row, col ) += food;
}
}
#pragma omp parallel for default(none)   shared(rows, columns, culture_cells) schedule(static)
for (i = 0; i < rows * columns; i++)
culture_cells[i] = 0;
float *food_to_share = (float *)malloc(sizeof(float) * num_cells);
if (culture == NULL || culture_cells == NULL)
{
fprintf(stderr, "-- Error allocating culture structures for size: %d x %d \n", rows, columns);
exit(EXIT_FAILURE);
}
#pragma omp parallel for schedule(guided) reduction(+ : step_dead_cells) reduction(max : max_age)
for (i = 0; i < num_cells; i++)
{
if (cells[i].alive)
{
cells[i].age++;
if (cells[i].age > max_age)
max_age = cells[i].age;
if (cells[i].storage < 0.1f)
{
cells[i].alive = false;
step_dead_cells++;
continue;
}
if (cells[i].storage < 1.0f)
{
cells[i].storage -= 0.2f;
}
else
{
cells[i].storage -= 1.0f;
float prob = (float)erand48(cells[i].random_seq);
if (prob < cells[i].choose_mov[0])
{
float tmp = cells[i].mov_col;
cells[i].mov_col = cells[i].mov_row;
cells[i].mov_row = -tmp;
}
else if (prob >= cells[i].choose_mov[0] + cells[i].choose_mov[1])
{
float tmp = cells[i].mov_row;
cells[i].mov_row = cells[i].mov_col;
cells[i].mov_col = -tmp;
}
cells[i].pos_row += cells[i].mov_row;
cells[i].pos_col += cells[i].mov_col;
if (cells[i].pos_row < 0)
cells[i].pos_row += rows;
if (cells[i].pos_row >= rows)	
cells[i].pos_row -= rows;
if (cells[i].pos_col < 0)
cells[i].pos_col += columns;
if (cells[i].pos_col >= columns)
cells[i].pos_col -= columns;
}
#pragma omp atomic
accessMat(culture_cells, cells[i].pos_row, cells[i].pos_col)++;
food_to_share[i] = accessMat(culture, cells[i].pos_row, cells[i].pos_col);
}
} 
sim_stat.history_max_age = max_age;
Cell *new_cells = (Cell *)malloc(sizeof(Cell) * num_cells);
if (new_cells == NULL)
{
fprintf(stderr, "-- Error allocating new cells structures for: %d cells\n", num_cells);
exit(EXIT_FAILURE);
}
int free_position = 0;
for (i = 0; i < num_cells; i++)
{
if (cells[i].alive)
{
float food = food_to_share[i];
short count = accessMat(culture_cells, cells[i].pos_row, cells[i].pos_col);
float my_food = food / count;
cells[i].storage += my_food;
if (cells[i].age > 30 && cells[i].storage > 20)
{
step_new_cells++;
cells[i].storage /= 2.0f;
cells[i].age = 1;
new_cells[step_new_cells - 1] = cells[i];
new_cells[step_new_cells - 1].random_seq[0] = (unsigned short)nrand48(cells[i].random_seq);
new_cells[step_new_cells - 1].random_seq[1] = (unsigned short)nrand48(cells[i].random_seq);
new_cells[step_new_cells - 1].random_seq[2] = (unsigned short)nrand48(cells[i].random_seq);
float angle = (float)(2 * M_PI * erand48(cells[i].random_seq));
cells[i].mov_row = sinf(angle);
cells[i].mov_col = cosf(angle);
angle = (float)(2 * M_PI * erand48(new_cells[step_new_cells - 1].random_seq));
new_cells[step_new_cells - 1].mov_row = sinf(angle);
new_cells[step_new_cells - 1].mov_col = cosf(angle);
cell_mutation(&cells[i]);
cell_mutation(&new_cells[step_new_cells - 1]);
}
accessMat(culture, cells[i].pos_row, cells[i].pos_col) = 0.0f;
if (free_position != i)
{
cells[free_position] = cells[i];
}
free_position++;
}
} 
num_cells_alive = num_cells_alive - step_dead_cells + step_new_cells;
sim_stat.history_total_cells += step_new_cells;
free(food_to_share);
num_cells = free_position;
cells = (Cell *)realloc(cells, sizeof(Cell) * (num_cells + step_new_cells));
current_max_food = 0.0f;
if (step_new_cells > 0)
{
#pragma omp parallel for default(none)                  shared(step_new_cells, cells, new_cells, num_cells) schedule(static)
for (j = 0; j < step_new_cells; j++)
cells[num_cells + j] = new_cells[j];
num_cells += step_new_cells;
}
#pragma omp parallel for default(none) shared(rows, columns, culture)     schedule(guided)               reduction(max              : current_max_food)
for (i = 0; i < rows * columns; i++)
{
culture[i] *= 0.95f; 
if (culture[i] > current_max_food)
current_max_food = culture[i];
}
free(new_cells);
if (current_max_food > sim_stat.history_max_food)
sim_stat.history_max_food = current_max_food;
if (step_new_cells > sim_stat.history_max_new_cells)
sim_stat.history_max_new_cells = step_new_cells;
sim_stat.history_dead_cells += step_dead_cells;
if (step_dead_cells > sim_stat.history_max_dead_cells)
sim_stat.history_max_dead_cells = step_dead_cells;
if (num_cells_alive > sim_stat.history_max_alive_cells)
sim_stat.history_max_alive_cells = num_cells_alive;
#ifdef DEBUG
print_status(iter, rows, columns, culture, num_cells, cells, num_cells_alive, sim_stat);
#endif 
}
ttotal = cp_Wtime() - ttotal;
#ifdef DEBUG
printf("List of cells at the end of the simulation: %d\n\n", num_cells);
for (i = 0; i < num_cells; i++)
{
printf("Cell %d, Alive: %d, Pos(%f,%f), Mov(%f,%f), Choose_mov(%f,%f,%f), Storage: %f, Age: %d\n",
i,
cells[i].alive,
cells[i].pos_row,
cells[i].pos_col,
cells[i].mov_row,
cells[i].mov_col,
cells[i].choose_mov[0],
cells[i].choose_mov[1],
cells[i].choose_mov[2],
cells[i].storage,
cells[i].age);
}
#endif 
printf("\n");
printf("Time: %lf\n", ttotal);
printf("Result: %d, ", iter);
printf("%d, %d, %d, %d, %d, %d, %d, %f\n",
num_cells_alive,
sim_stat.history_total_cells,
sim_stat.history_dead_cells,
sim_stat.history_max_alive_cells,
sim_stat.history_max_new_cells,
sim_stat.history_max_dead_cells,
sim_stat.history_max_age,
sim_stat.history_max_food);
free(culture);
free(culture_cells);
free(cells);
return 0;
}
