# Matrix Market IO

These functions require the optional SciPy dependency. See the
[interoperability guide](interoperability.md#matrix-market-files) for a round trip.
Paths and open binary streams are accepted. Arrays must be two-dimensional.

`mmread(filename, order="C")` defaults to row storage; pass
`order="F"` to use column storage. Real files are converted to `float64`
integer files to `int64`, and complex files to `complex128`.

`mmwrite(filename, A, comment="", field=None, precision=None)` writes general
symmetry explicitly. `field` and `precision` are forwarded to SciPy's writer.

::: sppy.io.matrix_market
    options:
      members: [mmread, mmwrite]
