#include "sequential.hpp"
int main(int argc, char* argv[]){
string file;                            
int generations = 0;                    
int cube_size = 0;                      
vector< vector< set <int> > > graph;    
set<Cell> cell_set;                                            
set<Cell>::const_iterator it;
parse_args(argc, argv, file, generations);
double start = omp_get_wtime();  
parse_file(file, cube_size, graph, cell_set);
for(int g = 1; g <= generations; g++){
set<Cell> live = cell_set;
for(it = live.begin(); it != live.end(); ++it){
vector<Cell> neighbours = getNeighbours(it->x, it->y, it->z, cube_size);
for (vector<Cell>::const_iterator j = neighbours.begin(); j != neighbours.end(); ++j){
cell_set.insert((Cell)*j);
}
}
Cell** cell_array;
int size = cell_set.size();
cell_array = new Cell*[size];
int a = 0;
for(it = cell_set.begin(); it != cell_set.end(); ++it){
cell_array[a++] = (Cell*) &(*it);
}
for(int i = 0; i < size; i++){
Cell* temp = cell_array[i];
temp->next_state = setNextState(temp->state, graph, cube_size, temp->x, temp->y, temp->z);
cout << "[" << ((i * 1.0)/size * 100.0) << "\%] Processing node " << i << " of " << size << endl;
}
for(int i = 0; i < size; i++){
Cell* temp = cell_array[i];
if(temp->state != temp->next_state){
if(temp->state == ALIVE){
graph[temp->x][temp->y].erase(temp->z);
}
else if(temp->state == DEAD){
graph[temp->x][temp->y].insert(temp->z);
}
temp->state = temp->next_state;
}    
}
for(it = cell_set.begin(); it != cell_set.end(); ++it){
cout << it->x << " " << it->y << " " << it-> z << endl;    
}
for(std::set<Cell>::iterator it = cell_set.begin(); it != cell_set.end();){
if(it->state == DEAD){
cell_set.erase(it++);    
}else{
++it;
}
}
delete[] cell_array;
}
for(it = cell_set.begin(); it != cell_set.end(); ++it){
cout << it->x << " " << it->y << " " << it-> z << endl;    
}
double end = omp_get_wtime();   
cout << "Total Runtime: " << (end - start) << endl; 
return 0;
}
bool setNextState(bool state, vector<vector<set<int> > > graph, int cube_size, int x, int y, int z){
int live_neighbours = liveNeighbors(x, y, z, graph, cube_size);
if (state == ALIVE){
if(live_neighbours < 2 || live_neighbours > 4){
return DEAD;
}
}else if(state == DEAD){
if(live_neighbours == 2 || live_neighbours == 3){
return ALIVE;
}   
}
return state;
}
vector<Cell> getNeighbours(int x, int y, int z, int cube_size){
std::vector<Cell> neighbours;
neighbours.push_back(Cell((x+1)%cube_size,y,z,DEAD));
neighbours.push_back(Cell((x-1) < 0 ? (cube_size-1) : (x-1),y,z,DEAD));
neighbours.push_back(Cell(x,(y+1)%cube_size,z,DEAD));
neighbours.push_back(Cell(x,((y-1) < 0 ? (cube_size-1) : (y-1)),z,DEAD));
neighbours.push_back(Cell(x,y,(z+1)%cube_size,DEAD));
neighbours.push_back(Cell(x,y,((z-1) < 0 ? (cube_size-1) : (z-1)),DEAD));
return neighbours;
}
int liveNeighbors(int i, int j, int k, vector<vector<set<int> > > graph, int cube_size){
int live_neighbours = 0;
if(setContains(k, graph[(i+1)%cube_size][j])) {  live_neighbours++;  }
if(setContains(k, graph[((i-1) < 0 ? (cube_size-1) : (i-1))][j])) {  live_neighbours++;  }
if(setContains(k, graph[i][(j+1)%cube_size])) {  live_neighbours++;  }
if(setContains(k, graph[i][((j-1) < 0 ? (cube_size-1) : (j-1))])) {  live_neighbours++;  }
if(setContains((k+1)%cube_size, graph[i][j])) {  live_neighbours++;  }
if(setContains(((k-1) < 0 ? (cube_size-1) : (k-1)), graph[i][j])) {  live_neighbours++;  }
return live_neighbours;
}
bool setContains(int k, std::set<int> _set){
if (_set.count(k)){
return true;
}
return false;
}
void parse_args(int argc, char* argv[], string &file, int &generations){
if (argc == 3){
file = argv[1];
generations = atoi(argv[2]);
if (generations > 0 && !file.empty())
return;
}    
cout << "Usage: " << argv[0] << " [data_file.in] [number_generations]" << endl;
exit(EXIT_FAILURE);
}
void parse_file(string file, int &cube_size, vector<vector<set<int> > > &graph, set<Cell> &cell_set){
int x,y,z;
std::ifstream infile(file.c_str());
std::string line;
if(!infile.good()){
cout << "ERROR: Could not find file." << endl;
exit(EXIT_FAILURE);
}
getline(infile, line);
cube_size = atoi(line.data());
graph.resize(cube_size);
for(int i = 0; i < cube_size; i++)
graph[i].resize(cube_size);
while (getline(infile, line)){
std::istringstream iss(line);
if ((iss >> x >> y >> z)) { 
graph[x][y].insert(z);
Cell new_cell(x,y,z,ALIVE);
cell_set.insert(new_cell);
}
}
}