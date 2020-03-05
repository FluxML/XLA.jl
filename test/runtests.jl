using XLATools, Test

@testset "XLATools" begin

@testset "XLA IR" begin
  include("xla.jl")
end

@testset "Compile" begin
  include("compile.jl")
end

end
