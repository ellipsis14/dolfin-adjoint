Point(1) = {0.5, 0, 0, 0.01};
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Point{1};
}
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Point{2};
}
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Point{4};
}
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Point{5};
}
Line Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};
Physical Line(1) = {1, 2, 3, 4};
Physical Surface(1) = {6};