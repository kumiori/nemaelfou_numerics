rad = .5;
rad2 = .1;
h = .05;

Point(0) = { 0, 0, 0, h};
Point(1) = { rad, 0, 0, h};
Point(2) = { 0, rad, 0, h};
Point(3) = { -rad, 0, 0, h};
Point(4) = { 0, -rad, 0, h};

Point(10) = { rad2, 0, 0, h};
Point(20) = { 0, rad2, 0, h};
Point(30) = { -rad2, 0, 0, h};
Point(40) = { 0, -rad2, 0, h};


Circle(1) = {1, 0, 2};
Circle(2) = {2, 0, 3};
Circle(3) = {3, 0, 4};
Circle(4) = {4, 0, 1};

Circle(10) = {10, 0, 20};
Circle(20) = {20, 0, 30};
Circle(30) = {30, 0, 40};
Circle(40) = {40, 0, 10};

Line Loop(1) = {1, 2, 3, 4, -10, -20, -30, -40};
Line Loop(2) = {10, 20, 30, 40};
Plane Surface(1) = {1};
Plane Surface(2) = {2};

Physical Surface(1) = {1};
Physical Surface(2) = {2};
