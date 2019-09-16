using XLATools, Test
using XLATools: XArray, Shape, Add, Neg, Mul, Ge, XTuple, Conditional, xlaclient, compile
using IRTools: IR, xcall, argument!

@test collect(Add()([1, 2, 3], [4, 5, 6])) == [5, 7, 9]

ir = IR()
x = argument!(ir, Shape(Int, (3,)))
push!(ir, xcall(Neg(), x))
f = compile(ir)
y = f([1, 2, 3])

@test collect(y) == [-1, -2, -3]

ir = IR()
x = argument!(ir, Shape(Int, (3,)))
mx = push!(ir, xcall(Neg(), x))
y = push!(ir, xcall(Add(), x, mx))
f = compile(ir)
y = f([1, 2, 3])

@test collect(y) == [0, 0, 0]

ir = IR()
x = argument!(ir, Int)
y = push!(ir, xcall(Mul(), x, 2))
f = compile(ir)

@test f(4) == 8

ir = IR()
argument!(ir, Int)
@test compile(ir)(5) == 5

ir = IR()
push!(ir, 5)
@test compile(ir)() == 5

ir = IR()
push!(ir, ())
@test compile(ir)() == ()

ir = IR()
push!(ir, xcall(XTuple(), argument!(ir, Int), argument!(ir, Int)))
@test compile(ir)(1, 2) == (1, 2)

relu = let
  ir = IR()
  x = argument!(ir, Int)
  cond = push!(ir, xcall(Ge(), x, 0))

  tb = IR()
  x = argument!(tb, Int)
  t = push!(ir, tb)
  fb = IR()
  argument!(fb, ())
  push!(fb, 0)
  f = push!(ir, fb)

  push!(ir, xcall(Conditional(), cond, x, t, (), f))
  compile(ir)
end

@test relu(5) == 5
@test relu(-5) == 0
