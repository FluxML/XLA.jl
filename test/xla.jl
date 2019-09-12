using XLATools, Test
using XLATools: XArray, Shape, xlaclient

x = XArray([1, 2, 3])

builder = xlaclient.ComputationBuilder("test")
arg = builder.ParameterWithShape(Shape(x))
builder.Neg(arg)
comp = builder.Build().Compile()

y = comp.Execute([x]) |> XArray

@test collect(y) == [-1, -2, -3]
