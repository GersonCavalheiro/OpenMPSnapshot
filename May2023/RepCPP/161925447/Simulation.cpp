#include "Simulation.h"
#include "Timer.h"
#include "Util.h"

#include <omp.h>
#include <random>

Simulation::Simulation(unsigned int width, unsigned int height)
: _width(width),
_height(height),
_bodies(Body::generate(NUM_BODIES)),
_zoom(1.2f) {
_window.create(sf::VideoMode(_width, _height), "N-body simulation",
sf::Style::Default);
_view.reset(sf::FloatRect(0, 0, _width, _height));
_view.zoom(_zoom);
_view.setViewport(sf::FloatRect(0.f, 0.f, 1.f, 1.f));
_window.setView(_view);
}

void Simulation::start() {
while (_window.isOpen()) {
poll_events();  
update();       
render();       
}

std::cout << "Done!" << std::endl;
}

void Simulation::poll_events() {
sf::Event event{};

while (_window.pollEvent(event)) {
if (event.type == sf::Event::Closed) {  
_window.close();
}
if (event.type ==
sf::Event::MouseWheelScrolled)  
{
_zoom *= 1.f + (-event.mouseWheelScroll.delta / 10.f);
_view.zoom(1.f + (-event.mouseWheelScroll.delta / 10.f));
}
}

if (sf::Mouse::getPosition().x > (_width - 20)) _view.move(2 * _zoom, 0);
if (sf::Mouse::getPosition().x < (0 + 20)) _view.move(-2 * _zoom, 0);
if (sf::Mouse::getPosition().y > (_height - 20)) _view.move(0, 2 * _zoom);
if (sf::Mouse::getPosition().y < (0 + 20)) _view.move(0, -2 * _zoom);

_window.setView(_view);
}

void Simulation::update() {
static sf::Clock clock;
auto dt = clock.restart().asSeconds();

Timer t(__func__);

#pragma omp parallel for schedule(static, 1)
for (int i = 0; i < NUM_BODIES; ++i) {
#pragma omp parallel for
for (int j = i + 1; j < NUM_BODIES; ++j) {
_bodies[i].interact(_bodies[j]);
}
}

#pragma omp parallel for
for (int i = 0; i < NUM_BODIES; ++i) {
_bodies[i].update(dt);  
}
}

void Simulation::render() {
_window.clear(sf::Color::Black);

sf::CircleShape star(DOT_SIZE, 50);
star.setOrigin(sf::Vector2f(DOT_SIZE / 2.0f, DOT_SIZE / 2.0f));

for (size_t i = 0; i < NUM_BODIES; ++i) {
auto current = &_bodies[i];
auto pos = current->position();
auto vel = current->velocity();
auto mag = vel.magnitude();

auto x = static_cast<float>(Util::to_pixel_space(pos.x, WIDTH, _zoom));
auto y = static_cast<float>(Util::to_pixel_space(pos.y, HEIGHT, _zoom));
star.setFillColor(Util::get_dot_colour(x, y, mag));
star.setPosition(sf::Vector2f(x, y));
star.setScale(sf::Vector2f(PARTICLE_MAX_SIZE, PARTICLE_MAX_SIZE));
if (i == 0) {
star.setScale(sf::Vector2f(2.5f, 2.5f));
star.setFillColor(sf::Color::Red);
}
_window.draw(star);
}

_window.display();
}
