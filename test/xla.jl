using XLATools, Test
using XLATools: XArray, Shape, Add, Neg, xlaclient, compile
using IRTools: IR, xcall, argument!

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
