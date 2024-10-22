#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
for (int row = 1; row < NUM/2+1; row++)
{

int NUM_2 = NUM >> 1;

pres_black(0, row) = pres_red(1, row);
pres_red(0, row) = pres_black(1, row);

pres_black(NUM + 1, row) = pres_red(NUM, row);
pres_red(NUM + 1, row) = pres_black(NUM, row);
}
