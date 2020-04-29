using XLA, IRTools, Test
using XLA: XArray, Shape, xlaclient, compile
using XLA: Add, Neg, Mul, Sub, Ge, Gt, XTuple, Conditional, While, GetTupleElement
using IRTools: IR, xcall, argument!

@test collect(Add()([1, 2, 3], [4, 5, 6])) == [5, 7, 9]

# TODO: tuples broke when we switched to flat inputs/outputs, which was only
# implemented for the Julia compiler.
@test_broken GetTupleElement(1)((3, 4)) == 4
