#pragma once

#include <SDL2/SDL.h>
#include <memory>

namespace sdl2 {

template <typename Creator, typename Destructor, typename... Arguments>
auto make_resource(Creator c, Destructor d, Arguments&&... args)
{
using std::decay_t;
using std::forward;
using std::unique_ptr;

auto r = c(forward<Arguments>(args)...);
return unique_ptr<decay_t<decltype(*r)>, decltype(d)>(r, d);
}

using SDL_System = int;

inline SDL_System* SDL_CreateSDL(Uint32 flags)
{
auto init_status = new SDL_System;
*init_status = SDL_Init(flags);
return init_status;
}

inline void SDL_DestroySDL(SDL_System* init_status)
{
delete init_status; 
SDL_Quit();
}

using sdlsystem_ptr_t = std::unique_ptr<SDL_System, decltype(&SDL_DestroySDL)>;
using window_ptr_t = std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)>;
using renderer_ptr_t = std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)>;
using surf_ptr_t = std::unique_ptr<SDL_Surface, decltype(&SDL_FreeSurface)>;
using texture_ptr_t = std::unique_ptr<SDL_Texture, decltype(&SDL_DestroyTexture)>;

inline sdlsystem_ptr_t make_sdlsystem(Uint32 flags)
{
return make_resource(SDL_CreateSDL, SDL_DestroySDL, flags);
}

inline window_ptr_t make_window(const char* title, int x, int y, int w, int h, Uint32 flags)
{
return make_resource(SDL_CreateWindow, SDL_DestroyWindow, title, x, y, w, h, flags);
}

inline renderer_ptr_t make_renderer(SDL_Window* win, int x, Uint32 flags)
{
return make_resource(SDL_CreateRenderer, SDL_DestroyRenderer, win, x, flags);
}

inline surf_ptr_t make_bmp(SDL_RWops* sdlfile)
{
return make_resource(SDL_LoadBMP_RW, SDL_FreeSurface, sdlfile, 1);
}

inline texture_ptr_t make_texture(SDL_Renderer* ren, SDL_Surface* surf)
{
return make_resource(SDL_CreateTextureFromSurface, SDL_DestroyTexture, ren, surf);
}

inline texture_ptr_t make_buffer_texture(SDL_Renderer* ren, int w, int h)
{
return make_resource(SDL_CreateTexture, SDL_DestroyTexture, ren, SDL_PIXELFORMAT_ARGB8888,
SDL_TEXTUREACCESS_STREAMING, w, h);
}

} 
