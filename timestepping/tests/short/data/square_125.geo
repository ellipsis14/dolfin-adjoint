Point(1) = {0, 0, 0, 1600000.0};
Extrude {200000000, 0, 0} {
  Point{1};
}
Extrude {0, 200000000, 0} {
  Line{1};
}
Physical Line(1) = {3};
Physical Line(2) = {4};
Physical Line(3) = {1};
Physical Line(4) = {2};
Physical Surface(1) = {5};
