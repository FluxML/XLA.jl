using XLATools, Test
using XLATools: XArray, Shape, Add, Neg, Mul, xlaclient, compile
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
