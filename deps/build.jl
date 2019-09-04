using PyCall, Conda

if PyCall.conda
  Conda.add("pip")
  pip = joinpath(Conda.BINDIR, "pip")
	run(`$pip install --upgrade jaxlib`)
else
  try
		pyimport("jaxlib")
	catch e
		e isa PyCall.PyError || rethrow(e)
		warn("""
      Python Dependancies not installed
      Please either:
       - Rebuild PyCall to use Conda, by running in the julia REPL:
          - `ENV[PYTHON]=""; Pkg.build("PyCall"); Pkg.build("XLATools")`
       - Or install the depencences, eg by running pip
      	- `pip install jaxlib`
    	""")
	end
end
