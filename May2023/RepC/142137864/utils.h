#pragma once
#define MAX_WIDTH 15000 
#define MAX_HEIGHT 15000 
#define MAX_SEED 50 
#define FILTER_SIZE 3 
#define FILTER_MAX_VALUE 10 
#define FILTER_MIN_VALUE -10
#define MAX_ITERATIONS 2000
#define PROCESSES_LIMIT 100
#define NUM_THREADS 1 
#define NUM_NEIGHBOURS 8
#define N 0
#define NE 1
#define E 2
#define SE 3
#define S 4
#define SW 5
#define W 6
#define NW 7
typedef struct Args_type{
int image_type, image_width, image_height, image_seed;
double filter[FILTER_SIZE][FILTER_SIZE];
int width_per_process, width_remaining;
int height_per_process, height_remaining;
int iterations;
}Args_type;
int read_filter(double[FILTER_SIZE][FILTER_SIZE]);
int read_user_input(Args_type*,int);
void printImage(int**,int,int,int,int);
