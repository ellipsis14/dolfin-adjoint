Point(1) = {0, 0, 0, 0.1};
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

Field[1] = MathEval;
Field[1].F = "min(0.005 + (0.1 - 0.005) * 2.0 * x, 0.005 + (0.1 - 0.005) * 2.0 * (1.0 - x))";
Field[2] = MathEval;
Field[2].F = "min(0.005 + (0.1 - 0.005) * 2.0 * y, 0.005 + (0.1 - 0.005) * 2.0 * (1.0 - y))";
Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;
Mesh.CharacteristicLengthExtendFromBoundary = 0;