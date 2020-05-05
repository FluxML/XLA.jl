using XLA, Test

@testset "XLA" begin

@testset "Compile" begin
  include("compile.jl")
end

@testset "Flux" begin
  include("flux.jl")
end

end
