







#include <iostream>
#include <sstream>
#include <bitset>
#include <random>
#include <memory>
#include <omp.h>

#include <allegro5/allegro5.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_primitives.h>





#define GAME_CYCLES_PER_SECOND      10.0
#define GAME_FRAME_PER_SECOND       60.0


#ifndef VIEWPORT_WIDTH
#define VIEWPORT_WIDTH              1280
#endif

#ifndef VIEWPORT_HEIGHT
#define VIEWPORT_HEIGHT             720
#endif

#define VIEWPORT_INITIAL_ZOOM       20
#define VIEWPORT_MAX_ZOOM           60
#define VIEWPORT_MIN_ZOOM           3


#define POS_TO_X(pos)               ((pos) % VIEWPORT_WIDTH)
#define POS_TO_Y(pos)               ((pos) / VIEWPORT_WIDTH)
#define XY_TO_POS(x, y)             (((y) * VIEWPORT_WIDTH) + (x))




class GameOfLife {

public:

GameOfLife() {

mainMatrix = std::make_unique<std::bitset<VIEWPORT_WIDTH * VIEWPORT_HEIGHT>>();
supportMatrix = std::make_unique<std::bitset<VIEWPORT_WIDTH * VIEWPORT_HEIGHT>>();

#if defined(BENCH)
mainMatrix->set();
#endif


mainMatrix->set(XY_TO_POS((VIEWPORT_WIDTH / 2) + 15, (VIEWPORT_HEIGHT / 2) + 14));
mainMatrix->set(XY_TO_POS((VIEWPORT_WIDTH / 2) + 17, (VIEWPORT_HEIGHT / 2) + 14));
mainMatrix->set(XY_TO_POS((VIEWPORT_WIDTH / 2) + 15, (VIEWPORT_HEIGHT / 2) + 15));
mainMatrix->set(XY_TO_POS((VIEWPORT_WIDTH / 2) + 16, (VIEWPORT_HEIGHT / 2) + 15));
mainMatrix->set(XY_TO_POS((VIEWPORT_WIDTH / 2) + 15, (VIEWPORT_HEIGHT / 2) + 16));

}


void setCell(size_t x, size_t y) {

if(x < VIEWPORT_WIDTH && y < VIEWPORT_HEIGHT)
mainMatrix->set(XY_TO_POS(x, y));

}

const bool getCell(size_t x, size_t y) const {

if(x < VIEWPORT_WIDTH && y < VIEWPORT_HEIGHT)
return mainMatrix->test(XY_TO_POS(x, y));

return false;
}

const size_t getHeat(size_t x, size_t y) const {

if(x < VIEWPORT_WIDTH && y < VIEWPORT_HEIGHT)
return getNeighborhood(XY_TO_POS(x, y)); 

return 0L;
}


const auto getPopulation() const { 
return mainMatrix->count();
}



void step() {

#if defined(BENCH)
const double tss = omp_get_wtime();
#endif

#pragma omp parallel for default(none) shared(mainMatrix, supportMatrix) schedule(static)
for(size_t i = 0; i < mainMatrix->size(); i++) {


const size_t n = getNeighborhood(i);



if(mainMatrix->test(i) && (n == 2 || n == 3))
supportMatrix->set(i, true);

else if(!mainMatrix->test(i) && (n == 3))
supportMatrix->set(i, true);

else
supportMatrix->set(i, false);




}


mainMatrix.swap(supportMatrix);


#if defined(BENCH)

const double tse = omp_get_wtime();
std::cout << std::fixed << (tse - tss) << std::endl;

#endif

}





private:

std::unique_ptr<std::bitset<VIEWPORT_WIDTH * VIEWPORT_HEIGHT>> mainMatrix;
std::unique_ptr<std::bitset<VIEWPORT_WIDTH * VIEWPORT_HEIGHT>> supportMatrix;


const size_t getNeighborhood(const size_t pos) const {

size_t c = 0;
const size_t x = POS_TO_X(pos);
const size_t y = POS_TO_Y(pos);


if(x > 0)
c += (mainMatrix->test(pos - 1) == true);                           

if(x < VIEWPORT_WIDTH - 1)
c += (mainMatrix->test(pos + 1) == true);                           

if(y > 0)
c += (mainMatrix->test(pos - VIEWPORT_WIDTH) == true);              

if(y < VIEWPORT_HEIGHT - 1)
c += (mainMatrix->test(pos + VIEWPORT_WIDTH) == true);              

if(y > 0 && x > 0)
c += (mainMatrix->test(pos - VIEWPORT_WIDTH - 1) == true);          

if(y > 0 && x < VIEWPORT_WIDTH - 1)
c += (mainMatrix->test(pos - VIEWPORT_WIDTH + 1) == true);          

if(y < VIEWPORT_HEIGHT - 1 && x > 0)
c += (mainMatrix->test(pos + VIEWPORT_WIDTH - 1) == true);          

if(y < VIEWPORT_HEIGHT - 1 && x < VIEWPORT_WIDTH - 1)
c += (mainMatrix->test(pos + VIEWPORT_WIDTH + 1) == true);          


return c;

}

};










class Application {

public:

Application(ALLEGRO_FONT* font) {

static_assert(VIEWPORT_WIDTH);
static_assert(VIEWPORT_HEIGHT);

this->viewport_zoom_factor = VIEWPORT_INITIAL_ZOOM;
this->viewport_x = -(VIEWPORT_WIDTH  / 2);
this->viewport_y = -(VIEWPORT_HEIGHT / 2);

this->font = font;
this->draw_user_instruction = true;
this->draw_user_interaction = true;
this->draw_user_map = true;
this->paused = false;


heats[0] = al_map_rgb(160, 0, 0);
heats[1] = al_map_rgb(160, 20, 10);
heats[2] = al_map_rgb(160, 40, 20);
heats[3] = al_map_rgb(160, 60, 30);
heats[4] = al_map_rgb(160, 80, 40);
heats[5] = al_map_rgb(160, 120, 70);
heats[6] = al_map_rgb(160, 140, 90);
heats[7] = al_map_rgb(160, 160, 160);
heats[8] = al_map_rgb(220, 220, 220);


this->running = true;

}


void update(ALLEGRO_EVENT* e) {

switch(e->type) {

case ALLEGRO_EVENT_KEY_DOWN:

viewport_x += (e->keyboard.keycode == ALLEGRO_KEY_LEFT)  * (VIEWPORT_MAX_ZOOM / viewport_zoom_factor);
viewport_x -= (e->keyboard.keycode == ALLEGRO_KEY_RIGHT) * (VIEWPORT_MAX_ZOOM / viewport_zoom_factor);
viewport_y += (e->keyboard.keycode == ALLEGRO_KEY_UP)    * (VIEWPORT_MAX_ZOOM / viewport_zoom_factor);
viewport_y -= (e->keyboard.keycode == ALLEGRO_KEY_DOWN)  * (VIEWPORT_MAX_ZOOM / viewport_zoom_factor);

viewport_x += (e->keyboard.keycode == ALLEGRO_KEY_A) * (VIEWPORT_MAX_ZOOM / viewport_zoom_factor);
viewport_x -= (e->keyboard.keycode == ALLEGRO_KEY_D) * (VIEWPORT_MAX_ZOOM / viewport_zoom_factor);
viewport_y += (e->keyboard.keycode == ALLEGRO_KEY_W) * (VIEWPORT_MAX_ZOOM / viewport_zoom_factor);
viewport_y -= (e->keyboard.keycode == ALLEGRO_KEY_S) * (VIEWPORT_MAX_ZOOM / viewport_zoom_factor);


if(e->keyboard.keycode == ALLEGRO_KEY_F)
this->paused = !this->paused;

if(e->keyboard.keycode == ALLEGRO_KEY_I)
this->draw_user_instruction = !this->draw_user_instruction;

if(e->keyboard.keycode == ALLEGRO_KEY_M)
this->draw_user_map = !this->draw_user_map;

if(e->keyboard.keycode == ALLEGRO_KEY_ESCAPE)
this->running = false;

break;


case ALLEGRO_EVENT_MOUSE_AXES:


if(e->mouse.dz != 0) {


const double odx = (VIEWPORT_WIDTH  - viewport_x) / viewport_zoom_factor;
const double ody = (VIEWPORT_HEIGHT - viewport_y) / viewport_zoom_factor;


viewport_zoom_factor += (e->mouse.dz);

if(viewport_zoom_factor < VIEWPORT_MIN_ZOOM)
viewport_zoom_factor = VIEWPORT_MIN_ZOOM;

if(viewport_zoom_factor > VIEWPORT_MAX_ZOOM)
viewport_zoom_factor = VIEWPORT_MAX_ZOOM;



const double ndx = (VIEWPORT_WIDTH  - viewport_x) / viewport_zoom_factor;
const double ndy = (VIEWPORT_HEIGHT - viewport_y) / viewport_zoom_factor;

viewport_x -= (odx - ndx) / (2 + 2 * (-viewport_x / VIEWPORT_WIDTH));
viewport_y -= (ody - ndy) / (2 + 2 * (-viewport_y / VIEWPORT_HEIGHT));


}



if(e->mouse.pressure > 0) {

ALLEGRO_MOUSE_STATE state;
al_get_mouse_state(&state);


if(al_mouse_button_down(&state, 3) 
|| al_mouse_button_down(&state, 2)) {

viewport_x += e->mouse.dx / viewport_zoom_factor;
viewport_y += e->mouse.dy / viewport_zoom_factor;


if(viewport_x > 0)
viewport_x = 0;

if(viewport_y > 0)
viewport_y = 0;

if(viewport_x < (-VIEWPORT_WIDTH + (VIEWPORT_WIDTH / viewport_zoom_factor) + 1))
viewport_x = (-VIEWPORT_WIDTH + (VIEWPORT_WIDTH / viewport_zoom_factor) + 1);

if(viewport_y < (-VIEWPORT_HEIGHT + (VIEWPORT_HEIGHT / viewport_zoom_factor) + 1))
viewport_y = (-VIEWPORT_HEIGHT + (VIEWPORT_HEIGHT / viewport_zoom_factor) + 1);


}


if(draw_user_interaction) {

if(al_mouse_button_down(&state, 1)) {

gameOfLife.setCell(draw_user_x, draw_user_y);

}

}

}


if(draw_user_interaction) {

draw_user_x = round(e->mouse.x / viewport_zoom_factor) - viewport_x + (viewport_x - floor(viewport_x)) - 1;
draw_user_y = round(e->mouse.y / viewport_zoom_factor) - viewport_y + (viewport_y - floor(viewport_y)) - 1;

}

break;


case ALLEGRO_EVENT_MOUSE_ENTER_DISPLAY:

draw_user_interaction = true;
break;


case ALLEGRO_EVENT_MOUSE_LEAVE_DISPLAY:

draw_user_interaction = false;
break;

case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:

if(draw_user_interaction) {

ALLEGRO_MOUSE_STATE state;
al_get_mouse_state(&state);


if(al_mouse_button_down(&state, 1)) {

gameOfLife.setCell(draw_user_x, draw_user_y);

}

}


}





}



void redraw(ALLEGRO_EVENT* e) {

al_clear_to_color(al_map_rgb(0, 0, 0));

redraw_game();
redraw_user();
redraw_ui(); 


if(!this->paused) {

if((e->timer.count % (int64_t) (GAME_FRAME_PER_SECOND / GAME_CYCLES_PER_SECOND)) == 0LL)
gameOfLife.step();

}

}


void exit() {
running = false;
}

void pause() {
paused = true;
}

void resume() {
paused = false;
}


inline const auto isRunning() const { return running; }



private:

ALLEGRO_FONT* font; 
ALLEGRO_COLOR heats[9];

GameOfLife gameOfLife;

double viewport_x;
double viewport_y;
double viewport_zoom_factor;

bool running;
bool paused;
bool draw_user_instruction;
bool draw_user_interaction;
bool draw_user_map;

double draw_user_x;
double draw_user_y;

size_t drawn_cells;



void redraw_game() {


drawn_cells = 0;


for(size_t x = 0; x < VIEWPORT_WIDTH; x++) {

for(size_t y = 0; y < VIEWPORT_WIDTH; y++) {

if(gameOfLife.getCell(x, y) != true)
continue;


const double dx = (x + viewport_x) * viewport_zoom_factor;
const double dy = (y + viewport_y) * viewport_zoom_factor;

if(dx < 0 || dx > VIEWPORT_WIDTH - 1)
continue;

if(dy < 0 || dy > VIEWPORT_HEIGHT - 1)
continue;



al_draw_filled_rectangle (
dx,
dy,
dx + viewport_zoom_factor,
dy + viewport_zoom_factor, heats[gameOfLife.getHeat(x, y)]
);

al_draw_rectangle (
dx,
dy,
dx + viewport_zoom_factor,
dy + viewport_zoom_factor, al_map_rgb_f(0.5f, 0, 0), viewport_zoom_factor / 5
); 


drawn_cells++;

}

}

}


void redraw_user() {

if(this->draw_user_interaction) {

al_draw_rectangle (
((draw_user_x + viewport_x) * viewport_zoom_factor),
((draw_user_y + viewport_y) * viewport_zoom_factor),
((draw_user_x + viewport_x) * viewport_zoom_factor) + viewport_zoom_factor,
((draw_user_y + viewport_y) * viewport_zoom_factor) + viewport_zoom_factor, al_map_rgb_f(0, 0.3f, 0.1f), viewport_zoom_factor / 5
);

}


if(this->draw_user_map) {

constexpr double m_sfactor = 10;
constexpr double m_padding = 30;

constexpr double x1 = (VIEWPORT_WIDTH) - (VIEWPORT_WIDTH / m_sfactor) - m_padding;
constexpr double x2 = (VIEWPORT_WIDTH) - m_padding;
constexpr double y1 = (m_padding);
constexpr double y2 = (VIEWPORT_HEIGHT / m_sfactor) + m_padding;


al_draw_filled_rectangle (x1, y1, x2, y2, al_map_rgb_f(0.20f, 0.20f, 0.20f));
al_draw_rectangle        (x1, y1, x2, y2, al_map_rgb_f(0.35f, 0.35f, 0.35f), 2);


al_draw_filled_rectangle (
x1 + (-viewport_x / m_sfactor),
y1 + (-viewport_y / m_sfactor),
x1 + (-viewport_x / m_sfactor) + (VIEWPORT_WIDTH  / m_sfactor / viewport_zoom_factor),
y1 + (-viewport_y / m_sfactor) + (VIEWPORT_HEIGHT / m_sfactor / viewport_zoom_factor), al_map_rgb_f(0.25, 0.30, 0.25)
);

al_draw_rectangle (
x1 + (-viewport_x / m_sfactor),
y1 + (-viewport_y / m_sfactor),
x1 + (-viewport_x / m_sfactor) + (VIEWPORT_WIDTH  / m_sfactor / viewport_zoom_factor),
y1 + (-viewport_y / m_sfactor) + (VIEWPORT_HEIGHT / m_sfactor / viewport_zoom_factor), al_map_rgb_f(0.25, 0.50, 0.25), 1
);

}

}

void redraw_ui() {

std::stringstream ss;
ss << "Population: "  << gameOfLife.getPopulation() << ", "
<< "Drawn Cells: " << drawn_cells;


al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), 10, 10, 0, ss.str().c_str());



if(this->draw_user_instruction) {

al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), 10, VIEWPORT_HEIGHT - 140, 0, "Instructions:");
al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), 10, VIEWPORT_HEIGHT - 120, 0, "  Left Click: generate cells");
al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), 10, VIEWPORT_HEIGHT - 100, 0, "  Mouse Wheel: increase/decrease zoom");
al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), 10, VIEWPORT_HEIGHT - 80, 0,  "  WASD, Arrows or Right Click: move around the map");
al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), 10, VIEWPORT_HEIGHT - 60, 0,  "  F: freeze life generation");
al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), 10, VIEWPORT_HEIGHT - 40, 0,  "  M: show/hide minimap");
al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), 10, VIEWPORT_HEIGHT - 20, 0,  "  I: show/hide this menu");

}

if(this->paused)
al_draw_text(font, al_map_rgb_f(0.75f, 0.75f, 0.75f), VIEWPORT_WIDTH - 220, VIEWPORT_HEIGHT - 20, 0, "Paused, press P to resume");

}

};








int main(int argc, char** argv) {


#if defined(BENCH)

GameOfLife gameOfLife;
gameOfLife.step();
exit(0);

#endif


al_init();
al_install_keyboard();
al_install_mouse();

al_set_app_name("Game of Life");


ALLEGRO_EVENT_QUEUE* queue;
if((queue = al_create_event_queue()) == NULL)
return std::cerr << "al_create_event_queue() failed!" << std::endl, 1;

ALLEGRO_DISPLAY* disp;
if((disp = al_create_display(VIEWPORT_WIDTH, VIEWPORT_HEIGHT)) == NULL)
return std::cerr << "al_create_display() failed!" << std::endl, 1;

ALLEGRO_TIMER* timer;
if((timer = al_create_timer(1 / GAME_FRAME_PER_SECOND)) == NULL)
return std::cerr << "al_create_timer() failed!" << std::endl, 1;

ALLEGRO_FONT* font;
if((font = al_create_builtin_font()) == NULL)
return std::cerr << "al_create_builtin_font() failed!" << std::endl, 1;



al_set_window_title(disp, "Game of Life");


al_register_event_source(queue, al_get_keyboard_event_source());
al_register_event_source(queue, al_get_mouse_event_source());
al_register_event_source(queue, al_get_display_event_source(disp));
al_register_event_source(queue, al_get_timer_event_source(timer));

al_start_timer(timer);


Application app(font);

do {

ALLEGRO_EVENT e;
al_wait_for_event(queue, &e);


switch(e.type) {

case ALLEGRO_EVENT_TIMER:
app.redraw(&e);
al_flip_display();
break;

case ALLEGRO_EVENT_KEY_DOWN:
case ALLEGRO_EVENT_KEY_UP:
case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
case ALLEGRO_EVENT_MOUSE_BUTTON_UP:
case ALLEGRO_EVENT_MOUSE_AXES:
case ALLEGRO_EVENT_MOUSE_ENTER_DISPLAY:
case ALLEGRO_EVENT_MOUSE_LEAVE_DISPLAY:
app.update(&e);
break;

case ALLEGRO_EVENT_DISPLAY_CLOSE:
app.exit();
break;

}


} while(app.isRunning());


return 0;

}
