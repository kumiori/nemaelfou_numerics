T = $T;
rad = $rad;
h = $meshsize;

theta = Atan(T);

P1x = rad;
P1y = 0;

P2x = -rad*Cos(theta);
P2y = rad*Sin(theta);

P3x = -rad;
P3y = 0;


P4x = -rad*Cos(theta);
P4y = -rad*Sin(theta);

Point(0) = { 0, 0, 0, h};
Point(2) = { -rad*Cos(theta), rad*Sin(theta), 0, h};
Point(3) = { -rad, 0, 0, h};
Point(4) = { -rad*Cos(theta), -rad*Sin(theta), 0, h};
Point(5) = { rad, 0, 0, h};


Line(2) = {2, 0};
Line(3) = {0, 4};
Line(4) = {3, 0};
Line(5) = {0, 5};

Circle(6) = {5, 0, 2};
Circle(7) = {2, 0, 3};
Circle(8) = {3, 0, 4};
Circle(9) = {4, 0, 5};


Line Loop(2) = {2, 5, 6};
Line Loop(1) = {7, 4, -2};
Line Loop(4) = {3, 9, -5};
Line Loop(3) = {8, -3, -4};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};

Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Surface(4) = {4};