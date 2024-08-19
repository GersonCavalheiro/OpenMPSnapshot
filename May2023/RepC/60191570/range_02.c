enum { N = 4, BS = 8 };
unsigned long long int square[N][N][BS][BS];
typedef unsigned long long int (*p_block_t)[BS];
void process_corner(p_block_t block);
void process_left_edge(p_block_t north, p_block_t block);
void process_upper_edge(p_block_t west, p_block_t block);
void process_inner(p_block_t north, p_block_t west,
p_block_t north_west, p_block_t block);
unsigned long long wavefront(void)
{
int i, j;
#pragma analysis_check assert range(N:4:4:0; BS:8:8:0; i:0:4:0)
for (i=0; i<N; ++i) {
#pragma analysis_check assert range(j:0:4:0)
for (j=0; j<N; ++j) {
if (j == 0)
{
if (i == 0)
{   
p_block_t block = square[i][j];
#pragma analysis_check assert range(i:0:0:0; j:0:0:0)
process_corner(block);
}
else
{   
p_block_t north = square[i-1][j];
p_block_t block = square[i][j];
#pragma analysis_check assert range(i:1:3:0; j:0:0:0)
process_left_edge(north, block);
}
}
else if (i == 0)
{   
p_block_t west = square[i][j-1];
p_block_t block = square[i][j];
#pragma analysis_check assert range(i:0:0:0; j:1:3:0)
process_upper_edge(west, block);
}
else
{   
p_block_t north = square[i-1][j];
p_block_t west = square[i][j-1];
p_block_t north_west = square[i-1][j-1];
p_block_t block = square[i][j];
#pragma analysis_check assert range(i:1:3:0; j:1:3:0)
process_inner(north, west, north_west, block);
}
}
}
return square[N - 1][N - 1][BS - 1][BS - 1];
}
