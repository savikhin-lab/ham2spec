# ham2spec

This is a Python extension written in Rust using [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

This module computes absorption and circular dichroism (CD) spectra from a Hamiltonian and a set of pigment positions and transition dipole moments. It's primarily used by the [fmo_analysis](https://github.com/savikhin-lab/fmo_analysis) tool.

## Installation
Right now this only works on macOS because I only have a macOS system to test on. The primary hurdle to building on other systems is the dependency on LAPACK. Your system will need to have a LAPACK implementation installed, and you'll need to set the correct `lapack_src` feature.

## Development
The Python extension module requires a `crate-type` of `"cdylib"`, but running examples and requires a `crate-type` of `"rlib"`. In order to accommodate both you'll need to run tests via
```
$ cargo test --lib
```
and examples via
```
$ cargo run --no-default-features --example example_name
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.