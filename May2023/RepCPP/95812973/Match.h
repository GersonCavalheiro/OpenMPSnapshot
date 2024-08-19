#pragma once

class Match {
private:
unsigned short red1;
unsigned short red2;
unsigned short red3;

unsigned short blue1;
unsigned short blue2;
unsigned short blue3;

double redWinProbability;

public:
Match(
unsigned short red1, 
unsigned short red2,
unsigned short red3,
unsigned short blue1,
unsigned short blue2,
unsigned short blue3,
double redWinProbability = .5
);

~Match();

unsigned short getRed1() const;
unsigned short getRed2() const;
unsigned short getRed3() const;

unsigned short getBlue1() const;
unsigned short getBlue2() const;
unsigned short getBlue3() const;

double getRedWinProbability() const;
double getBlueWinProbability() const;

};