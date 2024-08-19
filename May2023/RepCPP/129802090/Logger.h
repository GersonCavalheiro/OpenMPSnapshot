#pragma once
#ifndef OPENIMAGEFILTER_LOGGER_H
#define OPENIMAGEFILTER_LOGGER_H


class Logger {

~Logger();

void PrintThreadForIteraction(int x, int y);

void PrintCurrentThread();

public:
Logger();

double CetCurrentTime();

void GetElapsedTime(double startTime);
};


#endif 
