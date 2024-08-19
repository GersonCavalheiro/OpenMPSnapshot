#include "util.h"
#define INVALID_NUM_INIT_VALUE (-666)
#define GET_TIME omp_get_wtime()
u_int8_t apply_filter_on_pixel(int** img, int row, int col)
{
u_int8_t decimal = 0;
int center = img[row][col];
decimal |= img[row - 1][col - 1] > center;
decimal = decimal << 1u;
decimal |= img[row - 1][col]     > center;
decimal = decimal << 1u;
decimal |= img[row - 1][col + 1] > center;
decimal = decimal << 1u;
decimal |= img[row][col + 1]     > center;
decimal = decimal << 1u;
decimal |= img[row + 1][col + 1] > center;
decimal = decimal << 1u;
decimal |= img[row + 1][col]     > center;
decimal = decimal << 1u;
decimal |= img[row + 1][col - 1] > center;
decimal = decimal << 1u;
decimal |= img[row][col - 1]     > center;
return decimal;
}
int flag = 15;
void create_histogram(int * hist, int ** img, int num_rows, int num_cols)
{
int** img_lbp;
#if DEBUG_LBP_WRITE
img_lbp = alloc_2d_matrix(num_rows, num_cols);
printf("PARALLEL_DEBUGDEBUGDEBUGDEBUG\n");
if (img_lbp == NULL)
printf("NULL Pointer\n");
#endif
#if DEBUG_OPT_HIST
#pragma omp parallel for
#endif
for (int i = 1; i < IMAGE_HEIGHT - 1; ++i)
{
for (int j = 1; j < IMAGE_WIDTH - 1; ++j) {
int tmp = apply_filter_on_pixel(img, i, j);
#if DEBUG_LBP_WRITE
img_lbp[i][j] = tmp;
#endif
#pragma omp atomic
((int*)hist)[tmp]++;
}
}
hist[0] += (IMAGE_WIDTH + IMAGE_HEIGHT) * 2 - 2;
#if DEBUG_LBP_WRITE
for (int k = 0; k < IMAGE_HEIGHT; ++k) {
img_lbp[k][0] = 0;
img_lbp[k][IMAGE_WIDTH - 1] = 0;
hist[img[k][IMAGE_WIDTH - 1]]++;
hist[img[k][0]]++;
}
for (int l = 0; l < IMAGE_WIDTH; ++l) {
img_lbp[0][l] = 0;
img_lbp[IMAGE_HEIGHT - 1][l] = 0;
hist[img[0][l]]++;
hist[img[IMAGE_HEIGHT - 1][l]]++;
}
if (DEBUG_IMG_WRITE & flag == 15) {
FILE *fptr;
if (fptr = fopen("mat.out", "w")) {
for (int i = 0; i < IMAGE_HEIGHT; ++i) {
for (int j = 0; j < IMAGE_WIDTH; ++j) {
fprintf(fptr, "%d ", img_lbp[i][j]);
}
fprintf(fptr, "\n");
}
} else
printf("ADGFXNCH\n");
flag = 1;
}
dealloc_2d_matrix((int **) img_lbp, num_rows, num_cols); 
#endif
}
double distance(int * a, int *b, int size)
{
double sum = 0;
for (int i = 0; i < size; ++i) {
if ((double)(a[i] + b[i]) != 0)
sum += pow(a[i] - b[i], 2) / (double)(a[i] + b[i]);
}
return sum / 2;
}
int find_closest(int ***training_set, int num_persons, int num_training, int size,
int * test_image)
{
double min = INT_MAX;
int min_id_i = INVALID_NUM_INIT_VALUE;
int min_id_j;
#if DEBUG_OPT_DIST
#pragma omp parallel for collapse(2)
#endif
for (int i = 0; i < num_persons; ++i) {
for (int j = 0; j < num_training; ++j) {
double tmp = distance(training_set[i][j], test_image, size);
if (tmp < min) {
min = tmp;
min_id_i = i;
min_id_j = j;
}
}
}
return min_id_i;
}
int main(int argc, char* argv[])
{
double start_seq = GET_TIME;
int k = atoi(argv[1]), people_count = 18, sample_count_per_person = 20;
int**** original_images = malloc(people_count * sizeof(int***));
int*** histogram_array = malloc(people_count * sizeof(int**));
int j = 0;
int **image;
char buff[32];
double end_seq1 = GET_TIME - start_seq;
double start_parallel = GET_TIME;
#if DEBUG_OPT_MAIN
#pragma omp parallel for private(j, buff, image) shared(original_images)
#endif
for (int i = 0; i < people_count; ++i)
{
histogram_array[i] = malloc(sample_count_per_person * sizeof(int*));
original_images[i] = malloc(sample_count_per_person * sizeof(int**));
for (int j = 0; j < sample_count_per_person; ++j)
{
sprintf(buff, "images/%d.%d.txt", i + 1, j + 1); 
int** image = read_pgm_file(buff, IMAGE_HEIGHT, IMAGE_WIDTH);
original_images[i][j] = image;
histogram_array[i][j] = calloc(256, sizeof(int));
create_histogram((int*)histogram_array[i][j], original_images[i][j], IMAGE_HEIGHT, IMAGE_WIDTH);
}
}
int correct_count = 0;
int incorrect_count = 0;
u_int8_t found_people_array[people_count][sample_count_per_person - k];
#if DEBUG_OPT_TEST
#pragma omp parallel for collapse(2)
#endif
for (int i = 0; i < people_count; ++i)
{
for (int j = k; j < sample_count_per_person; ++j)
{
sprintf(buff, "%d.%d.txt", i + 1, j + 1); 
int found_person_id = find_closest(histogram_array, people_count, k, 256, histogram_array[i][j]);
found_people_array[i][j] = (u_int8_t)found_person_id;
if (found_person_id == i)
#if DEBUG_OPT_TEST
#pragma omp atomic
#endif
correct_count++;
else
#if DEBUG_OPT_TEST
#pragma omp atomic
#endif
incorrect_count++;
}
}
double end_parallel = GET_TIME;
start_seq = GET_TIME;
double parallel_time = end_parallel - start_parallel;
for (int i = 0; i < people_count; ++i) {
for (int j = k; j < sample_count_per_person; ++j) {
sprintf(buff, "%d.%d.txt", i + 1, j + 1); 
printf("%s %d %d\n", buff, found_people_array[i][j] + 1, i + 1);
}
}
printf("Accuracy: %d correct answers for %d tests\n", correct_count,
people_count * sample_count_per_person - k * people_count);
for (int l = 0; l < people_count; ++l) {
for (int i = 0; i < sample_count_per_person; ++i) {
free(histogram_array[l][i]);
dealloc_2d_matrix(original_images[l][i], IMAGE_HEIGHT, IMAGE_WIDTH);
}
free(histogram_array[l]);
free(original_images[l]);
}
free(histogram_array);
free(original_images);
double sequential_time = GET_TIME - start_seq + end_seq1;
printf("Parallel Time: %lf ms\n", (parallel_time) * 1000);
printf("Sequential Time: %lf ms\n", (sequential_time) * 1000);
return 0;
}