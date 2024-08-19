#pragma once

class Team {
private:
unsigned short number;

unsigned short qs;

unsigned short firstSort;
unsigned short secondSort;
unsigned short thirdSort;
unsigned short fourthSort;

public:
Team(
unsigned short number,
unsigned short qs,
unsigned short firstSort,
unsigned short secondSort,
unsigned short thirdSort,
unsigned short fourthSort
);
~Team();

unsigned short getNumber() const;

unsigned short getQS() const;

unsigned short getFirstSort() const;
unsigned short getSecondSort() const;
unsigned short getThirdSort() const;
unsigned short getFourthSort() const;
unsigned short getFifthSort() const;

void win();

bool lessThan (const Team &t2) const;
bool operator < (const Team &t2) const;
};