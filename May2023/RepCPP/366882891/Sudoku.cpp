#include <iostream>
#include <sstream>
#include <fstream>
#include <omp.h>
#include <vector>

void print_field(int LEVEL, std::vector<std::vector<int>>& game_field) {
int LEVEL_2 = LEVEL * LEVEL;
for (int ix = 0; ix < LEVEL_2; ++ix) {
for (int iy = 0; iy < LEVEL_2; ++iy) {
if(game_field[ix][iy] > 9) {std::cout << game_field[ix][iy] << " ";}
else { std::cout << game_field[ix][iy] << "  "; }

if ((iy + 1) % LEVEL == 0 && iy != LEVEL_2 - 1) { std::cout << "|  "; }
}
std::cout << std::endl;

if ((ix + 1) % LEVEL == 0 && ix != LEVEL_2 - 1) { 
for (int i = 0; i < LEVEL_2 + LEVEL - 1; ++i) { std::cout << "_ _"; }
std::cout << std::endl << std::endl;
}
}
}

void read_matrix(std::string file_name, int& LEVEL, std::vector<std::vector<int>>& game_field) {
std::ifstream fin;
std::string buf_str;
int buf_int;

fin.open(file_name);

getline(fin, buf_str, ',');
std::istringstream(buf_str) >> buf_int;

LEVEL = buf_int;
int LEVEL_2 = LEVEL * LEVEL;

game_field.resize(LEVEL_2);
for (int i = 0; i < LEVEL_2; ++i) { game_field[i].resize(LEVEL_2); }


int i = 0;
while (getline(fin, buf_str, ',')) {
std::istringstream(buf_str) >> buf_int;
game_field[i / LEVEL_2][i % LEVEL_2] = buf_int;
++i;
}

if (i != LEVEL_2 * LEVEL_2) {
std::cout << "Error! Wrong file!" << std::endl;
exit(1);
}
}

void find_possible_values(int LEVEL, std::vector<std::vector<int>>& game_field, std::vector<std::vector<bool>>& is_known,
std::vector<bool>& possibles, int cell) {

int LEVEL_2 = LEVEL * LEVEL;


int cell_x = cell / LEVEL_2;
int cell_y = cell % LEVEL_2;


for (int iy = 0; iy < LEVEL_2; ++iy) {
if (is_known[cell_x][iy] == true) {
possibles[game_field[cell_x][iy]] = true;
}
}

for (int ix = 0; ix < LEVEL_2; ++ix) {
if (is_known[ix][cell_y] == true) {
possibles[game_field[ix][cell_y]] = true;
}
}

int start_x = (cell_x / LEVEL) * LEVEL;
int start_y = (cell_y / LEVEL) * LEVEL;

for (int ix = 0; ix < LEVEL; ++ix) {
for (int iy = 0; iy < LEVEL; ++iy) {
if (is_known[start_x + ix][start_y + iy] == true) {
possibles[game_field[start_x + ix][start_y + iy]] = true;
}
}
}
}

bool PLEASE_STOP = false;

int solve_sudoku(int LEVEL, std::vector<std::vector<int>>& game_field) {

bool JOB_IS_DONE = false;

int LEVEL_2;
int LEVEL_4;

LEVEL_2 = LEVEL * LEVEL;
LEVEL_4 = LEVEL_2 * LEVEL_2;

std::vector<std::vector<bool>> is_known(LEVEL_2);   

std::vector<int> tasks;                             
std::vector<int> next_tasks;                        

for (int i = 0; i < LEVEL_2; ++i) { is_known[i].resize(LEVEL_2); }


for (int ix = 0; ix < LEVEL_2; ++ix) {
for (int iy = 0; iy < LEVEL_2; ++iy) {
if (game_field[ix][iy] != 0) {
is_known[ix][iy] = true;
}
else {
is_known[ix][iy] = false;
tasks.push_back(ix * LEVEL_2 + iy);
}
}
}

while (JOB_IS_DONE == false) {

std::vector<int> potential_vector(LEVEL_4, LEVEL_2 + 1); 


for (int cell : tasks) {
int cell_x = cell / LEVEL_2;
int cell_y = cell % LEVEL_2;

std::vector<bool> possibles(LEVEL_2 + 1, false);

find_possible_values(LEVEL, game_field, is_known, possibles, cell);


int first_false = 0;    
int false_counter = 0;  
for (int i = 1; i < LEVEL_2 + 1; ++i) {
if (possibles[i] == false) {
false_counter++;
if (false_counter == 1) { first_false = i; }
}
}
potential_vector[cell] = false_counter;

if (false_counter == 0) { 
return 1;
}
else if (false_counter == 1) {
is_known[cell_x][cell_y] = true;
game_field[cell_x][cell_y] = first_false;
}
else {
next_tasks.push_back(cell);
}

}

if (tasks.empty() == true) { JOB_IS_DONE = true; }

else if (tasks.size() == next_tasks.size()) {

std::pair<int, int> min_cell = std::pair<int, int>(-1, LEVEL_2 + 1); 
for (int i = 0; i < LEVEL_4; ++i) {
if (potential_vector[i] < min_cell.second) { 
min_cell.first = i;
min_cell.second = potential_vector[i];
if(min_cell.second == 2) { i = LEVEL_4; }
}
}

std::vector<bool> possibles(LEVEL_2 + 1, false);
find_possible_values(LEVEL, game_field, is_known, possibles, min_cell.first);


for (int i = 1; i < LEVEL_2 + 1; ++i) {
if (possibles[i] == false) {

#pragma omp task firstprivate(i, min_cell) shared(JOB_IS_DONE, game_field)
{
std::vector<std::vector<int>> game_field_copy(LEVEL_2);
for (int i = 0; i < LEVEL_2; ++i) { game_field_copy[i].resize(LEVEL_2); }
game_field_copy = game_field;
game_field_copy[min_cell.first / LEVEL_2][min_cell.first % LEVEL_2] = i;

if(JOB_IS_DONE == false && PLEASE_STOP == false) {
if (solve_sudoku(LEVEL, game_field_copy) == 0) { 
PLEASE_STOP = true;
JOB_IS_DONE = true;
game_field = game_field_copy;
}
}
}
}

}
#pragma omp taskwait

if(JOB_IS_DONE == true) { return 0; }
return 1;
}
tasks = next_tasks;
next_tasks.clear();
}
return 0;
}


int main() {
int LEVEL;                                                      
int LEVEL_2;
int LEVEL_4;


std::vector<std::vector<int>> game_field;                       

std::string file_name = "Sudoku_4_e.txt";

read_matrix(file_name, LEVEL, game_field);
LEVEL_2 = LEVEL * LEVEL;
LEVEL_4 = LEVEL_2 * LEVEL_2;

print_field(LEVEL, game_field);

double time_d = 0;
double time_end = 0;
double time_start = 0;
time_start = omp_get_wtime();

#pragma omp parallel
{
#pragma omp single
if (solve_sudoku(LEVEL, game_field) == 1) { 
std::cout << "Error!" << std::endl;
}
}

time_end = omp_get_wtime();
time_d = time_end - time_start;

std::cout << std::endl;
std::cout << std::endl;
print_field(LEVEL, game_field);

std::cout << std::endl << "Time: " << time_d << " seconds" << std::endl;
}
