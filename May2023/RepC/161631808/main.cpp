#include "ThirdParty/include/SDL2/SDL.h"
#include <stdlib.h>
#include <vector>
#include "stdio.h"
#include <omp.h>
#include <time.h>
namespace
{
inline int Get1DIndexFrom2D(const int x, const int y, const int xMax)
{
return x + y * xMax;
}
struct Window
{
int Width, Height;
};
struct GameContext
{
SDL_Renderer*    pRender;
std::vector<int> CurrentFrame, LastFrame;
Window           WindowSize;
bool             bStopGame;
};
}
void DrawPoints(GameContext* context)
{
SDL_SetRenderDrawColor(context->pRender, 0, 0, 0, 1);
SDL_RenderClear(context->pRender);
SDL_SetRenderDrawColor(context->pRender, 255, 255, 255, 1);
for(int y = 0; y < context->WindowSize.Height; ++y)
for (int x = 0; x < context->WindowSize.Width; ++x)
{
const int index = Get1DIndexFrom2D(x, y, context->WindowSize.Width - 1);
const int colorValue = context->CurrentFrame[index];
if(colorValue != 0)
SDL_RenderDrawPoint(context->pRender, x, y);
}
SDL_RenderPresent(context->pRender);
}
void DrawInitialPoints(GameContext* context)
{
SDL_Event event;
bool bStopPointsFilling = false;
bool bDrawPoints = false;
int lastX = 0;
int lastY = 0;
while (bStopPointsFilling != true)
{
if (SDL_PollEvent(&event))
{
int pointIndex = 0;
int colorValue = 0;
switch (event.type)
{
case SDL_MOUSEBUTTONDOWN:
bDrawPoints = true;
pointIndex = Get1DIndexFrom2D(lastX, lastY, context->WindowSize.Width - 1);
colorValue = context->CurrentFrame[pointIndex] == 0 ? 255 : 0;
SDL_SetRenderDrawColor(context->pRender, colorValue, colorValue, colorValue, 1);
SDL_RenderDrawPoint(context->pRender, lastX, lastY);
SDL_RenderPresent(context->pRender);
context->CurrentFrame[pointIndex] = context->CurrentFrame[pointIndex] == 0 ? 255 : 0;
break;
case SDL_MOUSEBUTTONUP:
bDrawPoints = false;
break;
case SDL_MOUSEMOTION:
lastX = event.motion.x;
lastY = event.motion.y;
printf("x:%d, y:%d\n", lastX, lastY);
if (bDrawPoints)
{
pointIndex = Get1DIndexFrom2D(lastX, lastY, context->WindowSize.Width - 1);
colorValue = context->CurrentFrame[pointIndex] == 0 ? 255 : 0;
SDL_SetRenderDrawColor(context->pRender, colorValue, colorValue, colorValue, 1);
SDL_RenderDrawPoint(context->pRender, lastX, lastY);
SDL_RenderPresent(context->pRender);
context->CurrentFrame[pointIndex] = context->CurrentFrame[pointIndex] == 0 ? 255 : 0;
}
break;
case SDL_KEYDOWN:
if (event.key.keysym.scancode == SDL_SCANCODE_RETURN)
return;
break;
}
}
}
}
void GameStepWithoutBorders(GameContext* context)
{
context->LastFrame = context->CurrentFrame;
#pragma omp parallel for
for (int y = 1; y < context->WindowSize.Height - 1; ++y)
{
for (int x = 1; x < context->WindowSize.Width - 1; ++x)
{
int topLeftIndex = Get1DIndexFrom2D(x - 1, y + 1, context->WindowSize.Width - 1);
int topMidleIndex = Get1DIndexFrom2D(x, y + 1, context->WindowSize.Width - 1);
int topRighIndex = Get1DIndexFrom2D(x + 1, y + 1, context->WindowSize.Width - 1);
int midleLeftIndex = Get1DIndexFrom2D(x - 1, y, context->WindowSize.Width - 1);
int midleRightIndex = Get1DIndexFrom2D(x + 1, y, context->WindowSize.Width - 1);
int botLeftIndex = Get1DIndexFrom2D(x - 1, y - 1, context->WindowSize.Width - 1);
int botMidleIndex = Get1DIndexFrom2D(x, y - 1, context->WindowSize.Width - 1);
int botRightIndex = Get1DIndexFrom2D(x + 1, y - 1, context->WindowSize.Width - 1);
unsigned int aliveNeighbours = 0;
if (context->LastFrame[topLeftIndex] != 0)
++aliveNeighbours;
if (context->LastFrame[topMidleIndex] != 0)
++aliveNeighbours;
if (context->LastFrame[topRighIndex] != 0)
++aliveNeighbours;
if (context->LastFrame[midleLeftIndex] != 0)
++aliveNeighbours;
if (context->LastFrame[midleRightIndex] != 0)
++aliveNeighbours;
if (context->LastFrame[botLeftIndex] != 0)
++aliveNeighbours;
if (context->LastFrame[botMidleIndex] != 0)
++aliveNeighbours;
if (context->LastFrame[botRightIndex] != 0)
++aliveNeighbours;
const int currentPointIndex = Get1DIndexFrom2D(x, y, context->WindowSize.Width - 1);
if (aliveNeighbours == 3)
{
context->CurrentFrame[currentPointIndex] = 255;
}
else if ((aliveNeighbours < 2) || (aliveNeighbours > 3))
{
context->CurrentFrame[currentPointIndex] = 0;
}
}
}
}
void CheckIfGameShouldStop(GameContext* context)
{
SDL_Event event;
if(SDL_PollEvent(&event))
switch (event.type)
{
case SDL_KEYDOWN:
if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE)
context->bStopGame = true;
}
}
void StartGameLoop(GameContext* context)
{
clock_t t1,t2;
while (context->bStopGame != true)
{
CheckIfGameShouldStop(context);
t1 = clock();
GameStepWithoutBorders(context);
DrawPoints(context);
t2 = clock();
double dt = t2 - t1;
std::printf("dt: %.2fms\n", dt * 1000 / CLOCKS_PER_SEC);
}
}
int main(int argc, char* argv[])
{
const unsigned int threadsNumber = 2;
if (threadsNumber < 0)
return -1;
SDL_Window* pWindow = nullptr;
SDL_Renderer* pRender = nullptr;
omp_set_num_threads(4);
const unsigned int width = 800;
const unsigned int height = 600;
SDL_CreateWindowAndRenderer(width, height,0, &pWindow, &pRender);
GameContext context;
context.CurrentFrame = std::vector<int>(width * height, 0);
context.LastFrame    = std::vector<int>(width * height, 0);
context.bStopGame = false;
context.pRender = pRender;
Window wndSize;
wndSize.Width = 800;
wndSize.Height = 600;
context.WindowSize = wndSize;
DrawInitialPoints(&context);
StartGameLoop(&context);
SDL_Quit();
return 0;
}