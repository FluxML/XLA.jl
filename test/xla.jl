using XLA, IRTools, Test
using XLA: XArray, Shape, xlaclient, compile
using XLA: Add, Neg, Mul, Sub, Ge, Gt, XTuple, Conditional, While, GetTupleElement
using IRTools: IR, xcall, argument!

@test collect(Add()([1, 2, 3], [4, 5, 6])) == [5, 7, 9]

@test GetTupleElement(1)((3, 4)) == 4

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
x = argument!(ir, (Int, Int))
push!(ir, xcall(GetTupleElement(1), x))
@test compile(ir)((1, 2)) == 2

ir = IR()
push!(ir, xcall(XTuple(), argument!(ir, Int), argument!(ir, Int)))
@test compile(ir)(1, 2) == (1, 2)

let
  relu = let
    ir = IR()
    x = argument!(ir, Int)
    cond = push!(ir, xcall(Ge(), x, 0))

    tb = IR()
    x = argument!(tb)
    t = push!(ir, tb)
    fb = IR()
    argument!(fb)
    push!(fb, 0)
    f = push!(ir, fb)

    push!(ir, xcall(Conditional(), cond, x, t, (), f))
    compile(ir)
  end

  @test relu(5) == 5
  @test relu(-5) == 0
end

ir = IR()
x = argument!(ir, Int)
n = argument!(ir, Int)
xnr = push!(ir, xcall(XTuple(), x, n, 1))
cond = let ir = IR()
  xnr = argument!(ir)
  n = push!(ir, xcall(GetTupleElement(1), xnr))
  push!(ir, xcall(Gt(), n, 0))
  ir
end
cond = push!(ir, cond)
body = let ir = IR()
  xnr = argument!(ir)
  x = push!(ir, xcall(GetTupleElement(0), xnr))
  n = push!(ir, xcall(GetTupleElement(1), xnr))
  r = push!(ir, xcall(GetTupleElement(2), xnr))
  n = push!(ir, xcall(Sub(), n, 1))
  r = push!(ir, xcall(Mul(), x, r))
  xnr = push!(ir, xcall(XTuple(), x, n, r))
  ir
end
body = push!(ir, body)
xnr = push!(ir, xcall(While(), cond, body, xnr))
push!(ir, xcall(GetTupleElement(2), xnr))
pow = compile(ir)

@test pow(2, 3) == 2^3
@test pow(5, 10) == 5^10

let
  @eval relu(x) = $(Gt())(x, 0) ? x : 0
  ir = @code_ir relu(1)
  IRTools.argtypes(ir)[:] = [(), Int]

  f = compile(ir)
  @test f((), 5) == 5
  @test f((), -5) == 0
end

let
  @eval relu(x) = $(Gt())(x, 0) ? $(Mul())(2, x) : $(Mul())(3, x)
  ir = @code_ir relu(1)
  IRTools.argtypes(ir)[:] = [(), Int]

  f = compile(ir)
  @test f((), 5) == 10
  @test f((), -5) == -15
end
