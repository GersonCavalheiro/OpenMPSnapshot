#ifndef ___HLS__VIDEO_MEM__
#define ___HLS__VIDEO_MEM__
namespace hls {
template<int ROWS, int COLS, typename T>
class Window {
public:
Window() {
#pragma HLS ARRAY_PARTITION variable=val dim=0 complete
};
void shift_right(){};
void shift_left(){};
void shift_up(){};
void shift_down(){};
void insert(T value, int row, int col){};
void insert_bottom(T value[COLS]){};
void insert_top(T value[COLS]){};
void insert_left(T value[ROWS]){};
void insert_right(T value[ROWS]){};
T getval(int row, int col){};
T val[ROWS][COLS];
};
} 
#endif
