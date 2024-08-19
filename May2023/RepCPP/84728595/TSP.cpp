inline float distanceC(int i, int j){
return (i<j)?m[i*MAXIMAL+j]:m[j*MAXIMAL+i];
}

void rek(const int actual_city,const unsigned int no_visited, const float tmp_dist, const int depth) {

if (no_visited==((1<<no_cities)-1)) {
if (actual_city==0) {
if (tmp_dist<tmp_min) {
tmp_min=tmp_dist;
}
}
return;
}

float sum;
bool chosen = false;
int chosenJ = 0;
float prevSum;

int j = (depth == no_cities-1) ? 0 : 1;

for(;j<no_cities &&  !(no_visited & (1<<(no_cities-j)-1) ==  (1<<(no_cities-j)-1))   ;j++) {

if ((no_visited>>j)&1 || j == actual_city || tmp_dist+distanceC(actual_city,j) > tmp_min){
continue;
}
chosen = true;
chosenJ = j;
prevSum = tmp_dist+distanceC(actual_city,j) ;
j++;
break;
}

for(;j<no_cities &&  !(no_visited & (1<<(no_cities-j)-1) ==  (1<<(no_cities-j)-1))   ;j++) {

if ((no_visited>>j)&1 || j == actual_city){
continue;
}

sum = tmp_dist+distanceC(actual_city,j) ;
if (sum > tmp_min) {
continue;
}

#pragma omp task firstprivate(chosenJ,no_visited,prevSum,depth) if (depth<4) 
rek(chosenJ,(no_visited|(1<<chosenJ)),prevSum,depth+1);


chosenJ = j;
prevSum = sum;

}

if (chosen) {
rek(chosenJ,(no_visited|(1<<chosenJ)),prevSum,depth+1);
}
}


