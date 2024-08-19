#pragma once





class Input {

public:

enum class Key {
Space = 32,
A = 65,
B = 66,
C = 67,
D = 68,
E = 69,
F = 70,
G = 71,
H = 72,
I = 73,
J = 74,
K = 75,
L = 76,
M = 77,
N = 78,
O = 79,
P = 80,
Q = 81,
R = 82,
S = 83,
T = 84,
U = 85,
V = 86,
W = 87,
X = 88,
Y = 89,
Z = 90,
Escape = 256,
Enter = 257,
Tab = 258,
Backspace = 259,
Insert = 260,
Delete = 261,
Right = 262,
Left = 263,
Down = 264,
Up = 265,
PageUp = 266,
PageDown = 267,
Home = 268,
End = 269,
F1 = 290,
F2 = 291,
F3 = 292,
LeftShift = 340,
LeftControl = 341,
LeftAlt = 342,
LeftSuper = 343,
};

enum class Button {
Left = 0,
Middle = 2,
Right = 1
};


public:

static bool IsKeyPressed(Key key);
static bool IsButtonPressed(Button button);

};
