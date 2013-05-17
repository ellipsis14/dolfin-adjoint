Point(1) = {0, 0, 0, 0.01};
Extrude {1, 0, 0} {
  Point{1};
}
Extrude {0, 1, 0} {
  Line{1};
}
Physical Line(1) = {3};
Physical Line(2) = {4};
Physical Line(3) = {1};
Physical Line(4) = {2};
Physical Surface(1) = {5};
