#include <SFML/Graphics.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include <string>
#include <sstream>
#include <iomanip>

#define WWIDTH 800
#define WHEIGHT 600

#define WIDTH 200
#define HEIGHT WIDTH
#define CELLS 5
#define CELLT 1
#define CAMERAVX 4.0f
#define CAMERAVY CAMERAVX
#define ZOOMS 0.9f
#define STARTF 1.0f
#define BASEF 8.0f

template<typename T> T lerp( T, T, T );

int main() {
sf::RenderWindow window( sf::VideoMode( WWIDTH, WHEIGHT ), "Conway's Game of Life" );

sf::Font font;
if ( !font.loadFromFile( "Fonts\\cour.ttf" ) )
std::cerr << "ERROR opening font." << std::endl;

float epochs_frec = STARTF;
float zoom = 1.0f;
bool left = false, right = false, up = false, down = false, zooming = false, unzooming = false, grid = true, play = false;
bool board[WIDTH][HEIGHT] = {false}, board_temp[WIDTH][HEIGHT];

sf::RectangleShape cell;
cell.setSize( sf::Vector2<float>( CELLS, CELLS ) );
cell.setOutlineColor( sf::Color( 150, 150, 150 ) );
cell.setOutlineThickness( CELLT );
sf::Text paused_vel_text;
paused_vel_text.setString( "Paused" );
paused_vel_text.setFont( font );
paused_vel_text.setCharacterSize( 24 );
paused_vel_text.setFillColor( sf::Color::Red );
paused_vel_text.setStyle( sf::Text::Bold );
paused_vel_text.setPosition( 15.0f, 5.0f );
sf::View camera( sf::Vector2f( WIDTH * 5 / 2, HEIGHT * 5 / 2 ), sf::Vector2f( WWIDTH, WHEIGHT ) );

sf::Clock epoch_t, frame_c;

while ( window.isOpen() ) {
window.setView( camera );
sf::Event event;
while ( window.pollEvent( event ) ) {
if ( event.type == sf::Event::Closed )
window.close();
else {
if ( event.type == sf::Event::Resized ) {
sf::Vector2<float> temp_c = camera.getCenter();
camera = sf::View( sf::FloatRect( 0, 0, event.size.width, event.size.height ) );
camera.setCenter( temp_c );
camera.zoom( zoom );
}
if ( event.type == sf::Event::KeyPressed ) {
if ( event.key.code == sf::Keyboard::Left )
left = true;
if ( event.key.code == sf::Keyboard::Up )
up = true;
if ( event.key.code == sf::Keyboard::Right )
right = true;
if ( event.key.code == sf::Keyboard::Down )
down = true;
if ( event.key.code == sf::Keyboard::Z )
zooming = true;
if ( event.key.code == sf::Keyboard::X )
unzooming = true;
}
if ( event.type == sf::Event::KeyReleased ) {
if ( event.key.code == sf::Keyboard::Left )
left = false;
if ( event.key.code == sf::Keyboard::Up )
up = false;
if ( event.key.code == sf::Keyboard::Right )
right = false;
if ( event.key.code == sf::Keyboard::Down )
down = false;
if ( event.key.code == sf::Keyboard::Z )
zooming = false;
if ( event.key.code == sf::Keyboard::X )
unzooming = false;
if ( event.key.code == sf::Keyboard::Space ) {
play = !play;
if ( play ) {
std::stringstream temp( "" );
temp << "Curr. velocity: " << std::setprecision( 2 ) << epochs_frec*BASEF << " epochs/s";
paused_vel_text.setString( temp.str() );
paused_vel_text.setFillColor( sf::Color::Green );
}
else {
paused_vel_text.setString( "Paused" );
paused_vel_text.setFillColor( sf::Color::Red );
}
}
if ( event.key.code == sf::Keyboard::Q ) {
epochs_frec = std::max( 1.0f / 32.0f, epochs_frec / 2.0f );
if ( play ) {
std::stringstream temp( "" );
temp << "Curr. velocity: " << std::setprecision( 2 ) << epochs_frec*BASEF << " epochs/s";
paused_vel_text.setString( temp.str() );
paused_vel_text.setFillColor( sf::Color::Green );
}
}
if ( event.key.code == sf::Keyboard::W ) {
epochs_frec = std::min( 8.0f, epochs_frec * 2.0f );
if ( play ) {
std::stringstream temp( "" );
temp << "Curr. velocity: " << std::setprecision( 2 ) << epochs_frec*BASEF << " epochs/s";
paused_vel_text.setString( temp.str() );
paused_vel_text.setFillColor( sf::Color::Green );
}
}
if ( event.key.code == sf::Keyboard::G )
grid = !grid;
if ( grid )
cell.setOutlineThickness( CELLT );
else
cell.setOutlineThickness( 0 );
}
if ( event.type == sf::Event::MouseButtonReleased ) {
if ( event.mouseButton.button == sf::Mouse::Left ) {
sf::Vector2<float> mouse_pos = window.mapPixelToCoords( sf::Mouse::getPosition( window ) );
if ( mouse_pos.x >= 0 && mouse_pos.y >= 0 && mouse_pos.x < WIDTH * CELLS && mouse_pos.y < HEIGHT * CELLS ) {
int x = ( int )std::floor( mouse_pos.x / ( float )CELLS ), y = ( int )std::floor( mouse_pos.y / ( float )CELLS );
board[x][y] = !board[x][y];
}
}
}
if ( event.type == sf::Event::MouseWheelScrolled ) {
if ( event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel ) {
sf::Vector2<float> temp_c = camera.getCenter();
sf::Vector2<float> mouse_pos = window.mapPixelToCoords( sf::Vector2<int>( event.mouseWheelScroll.x, event.mouseWheelScroll.y ) );
float new_x, new_y;
if ( event.mouseWheelScroll.delta > 0 ) {
new_x = lerp( ( float )temp_c.x, ( float )mouse_pos.x, 0.1f );
new_y = lerp( ( float )temp_c.y, ( float )mouse_pos.y, 0.1f );
camera.zoom( 0.9f );
zoom *= ZOOMS;
}
else {
new_x = lerp( ( float )temp_c.x, ( float )mouse_pos.x, 1.0f - 1.0f / ZOOMS );
new_y = lerp( ( float )temp_c.y, ( float )mouse_pos.y, 1.0f - 1.0f / ZOOMS );
camera.zoom( 1.0f / ZOOMS );
zoom *= 1.0f / ZOOMS;
}
sf::Vector2<float> new_c( std::max( 0.0f, std::min( ( float )WIDTH * CELLS, new_x ) ),
std::max( 0.0f, std::min( ( float )HEIGHT * CELLS, new_y ) ) );
camera.setCenter( new_c );
}
}
}
}
float delta_t = ( ( float )frame_c.restart().asMilliseconds() / 1000.0f );
if ( left ) {
sf::Vector2<float> temp = camera.getCenter();
temp.x = std::max( 0.0f, temp.x - ( camera.getSize().x / ( 2 * CAMERAVX ) ) * delta_t );
camera.setCenter( temp );
}
if ( up ) {
sf::Vector2<float> temp = camera.getCenter();
temp.y = std::max( 0.0f, temp.y - ( camera.getSize().y / ( 2 * CAMERAVY ) ) * delta_t );
camera.setCenter( temp );
}
if ( right ) {
sf::Vector2<float> temp = camera.getCenter();
temp.x = std::min( ( float )WIDTH * CELLS, temp.x + ( camera.getSize().x / ( 2 * CAMERAVX ) ) * delta_t );
camera.setCenter( temp );
}
if ( down ) {
sf::Vector2<float> temp = camera.getCenter();
temp.y = std::min( ( float )HEIGHT * CELLS, temp.y + ( camera.getSize().y / ( 2 * CAMERAVY ) ) * delta_t );
camera.setCenter( temp );
}
if ( zooming ) {
camera.zoom( ZOOMS );
zoom = zoom * ZOOMS;
}
if ( unzooming ) {
camera.zoom( 1.0f / ZOOMS );
zoom = zoom * 1.0f / ZOOMS;
}
window.clear( sf::Color( 120, 120, 120 ) );
memcpy( &board_temp[0][0], &board[0][0], WIDTH * HEIGHT * sizeof( bool ) );
if ( play && epoch_t.getElapsedTime().asMilliseconds() >= ( 1000.0f / ( BASEF * epochs_frec ) ) ) {
#pragma omp parallel for simd
for ( int x = 0 ; x < WIDTH; x++ ) {
for ( int y = 0 ; y < HEIGHT; y++ ) {
int neighbours = 0;
if ( x > 0 ) {
if ( y > 0 && board_temp[x - 1][y - 1] )
neighbours++;
if ( board_temp[x - 1][y] )
neighbours++;
if ( y < HEIGHT - 2 && board_temp[x - 1][y + 1] )
neighbours++;
}
if ( y > 0 && board_temp[x][y - 1] )
neighbours++;
if ( y < HEIGHT - 2 && board_temp[x][y + 1] )
neighbours++;
if ( x < WIDTH - 2 ) {
if ( y > 0 && board_temp[x + 1][y - 1] )
neighbours++;
if ( board_temp[x + 1][y] )
neighbours++;
if ( y < HEIGHT - 2 && board_temp[x + 1][y + 1] )
neighbours++;
}
if ( board_temp[x][y] && ( neighbours < 2 || neighbours > 3 ) )
board[x][y] =  false;
else if ( !board_temp[x][y] && neighbours == 3 )
board[x][y] =  true;
}
}
epoch_t.restart();
}
for ( int x = 0 ; x < WIDTH; x++ ) {
for ( int y = 0 ; y < HEIGHT; y++ ) {
cell.setPosition( sf::Vector2<float>( x * CELLS, y * CELLS ) );
if ( board[x][y] )
cell.setFillColor( sf::Color::Black );
else
cell.setFillColor( sf::Color::White );
window.draw( cell );
}
}
window.setView( window.getDefaultView() );
window.draw( paused_vel_text );
window.display();
}
return EXIT_SUCCESS;
}

template<typename T> T lerp( T a, T b, T v ) {
return a + v * ( b - a );
}
