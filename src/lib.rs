extern crate lapack_src;
#[cfg(test)]
use approx::{assert_abs_diff_eq, assert_relative_eq};
use lapack::dgeev;
use ndarray::{
    arr1, arr2, s, Array, Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, Axis,
    Zip,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    ToPyArray,
};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::ops::AddAssign;

/// A stick spectrum computed from a single Hamiltonian and associated pigments.
#[derive(Debug, Clone)]
pub struct StickSpectrum {
    /// The eigenvectors, one per column.
    pub e_vecs: Array2<f64>,

    /// The energies of the excitons.
    pub e_vals: Array1<f64>,

    /// The transition dipole moments of the excitons.
    pub mus: Array2<f64>,

    /// The absorption (dipole strength) of each exciton.
    pub stick_abs: Array1<f64>,

    /// The circular dichroism (rotational strength) of each exciton.
    pub stick_cd: Array1<f64>,
}

impl ToPyObject for StickSpectrum {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        // Don't mind the `unwrap`s here, setting a dictionary entry is unlikely to fail
        dict.set_item("e_vals", self.e_vals.to_pyarray(py)).unwrap();
        dict.set_item("e_vecs", self.e_vecs.to_pyarray(py)).unwrap();
        dict.set_item("exciton_mus", self.mus.to_pyarray(py))
            .unwrap();
        dict.set_item("stick_abs", self.stick_abs.to_pyarray(py))
            .unwrap();
        dict.set_item("stick_cd", self.stick_cd.to_pyarray(py))
            .unwrap();
        dict.to_object(py)
    }
}

/// The configuration for computing a broadened spectrum from a stick spectrum
#[derive(Debug, Clone, FromPyObject)]
pub struct BroadeningConfig {
    /// The starting point for the x-axis in wavenumbers (cm^-1)
    #[pyo3(attribute("xfrom"))]
    pub x_from: f64,

    /// The stopping point for the x-axis in wavenumbers (cm^-1)
    #[pyo3(attribute("xto"))]
    pub x_to: f64,

    /// The step size for the x-axis in wavenumbers (cm^-1)
    #[pyo3(attribute("xstep"))]
    pub x_step: f64,

    /// The bandwidth for each transition in wavenumbers (cm^-1)
    #[pyo3(attribute("bandwidth"))]
    pub bw: f64,

    /// The absorption bandwidths in wavenumbers (cm^-1) when using heterogenous bandwidths
    #[pyo3(attribute("abs_bws"))]
    pub abs_bws: Vec<f64>,

    /// The CD bandwidths in wavenumbers (cm^-1) when using heterogenous bandwidths
    #[pyo3(attribute("cd_bws"))]
    pub cd_bws: Vec<f64>,

    /// The number of `bw`s away from the band center outside of which calculations will be skipped
    #[pyo3(attribute("band_cutoff"))]
    pub band_cutoff: f64,
}

/// A broadened spectrum
#[derive(Debug, Clone)]
pub struct BroadenedSpectrum {
    /// The x-axis for the spectra
    pub x: Array1<f64>,

    /// The absorption spectrum
    pub abs: Array1<f64>,

    /// The circular-dichroism spectrum
    pub cd: Array1<f64>,
}

impl ToPyObject for BroadenedSpectrum {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        // Don't mind the `unwrap`s here, setting a dictionary entry is unlikely to fail
        dict.set_item("x", self.x.to_pyarray(py)).unwrap();
        dict.set_item("abs", self.abs.to_pyarray(py)).unwrap();
        dict.set_item("cd", self.cd.to_pyarray(py)).unwrap();
        dict.to_object(py)
    }
}

/// Compute the dot product of 2 3-vectors
fn dot(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the cross-product of 2 3-vectors
fn cross(a: ArrayView1<f64>, b: ArrayView1<f64>) -> Array1<f64> {
    arr1(&[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

/// Compute the absorbance stick spectrum
///
/// The eigenvectors must be arranged into columns, and the pigment dipole moments
/// must be arranged into rows. See [`compute_stick_spectrum`] for the expected
/// layout of `mus`.
pub fn stick_abs_single(mus: ArrayView2<f64>) -> Array1<f64> {
    let n_pigs = mus.nrows();
    let mut stick_abs = Array1::zeros(n_pigs);
    Zip::from(&mut stick_abs)
        .and(mus.rows())
        .for_each(|a, mu| *a = mu.dot(&mu));
    stick_abs
}

/// Compute the CD stick spectrum
///
/// The eigenvectors must be arranged into columns, and the pigment dipole moments
/// must be arranged into rows. See [`compute_stick_spectrum`] for the expected
/// layout of `mus` and `rs`.
pub fn stick_cd_single(
    e_vecs: ArrayView2<f64>,
    mus: ArrayView2<f64>,
    rs: ArrayView2<f64>,
    energies: ArrayView1<f64>,
) -> Array1<f64> {
    let coeffs: Vec<f64> = energies
        .iter()
        .map(|e| {
            let wavelength = if e < &1.0 { 1e8 / 100_000f64 } else { 1e8 / e };
            2f64 * core::f64::consts::PI / wavelength
        })
        .collect();
    let mut cd = Array1::zeros(energies.raw_dim());
    let n_pigs = e_vecs.ncols();
    let r_mu_cross_cache = populate_r_mu_cross_cache(mus, rs);
    for i in 0..n_pigs {
        for j in 0..n_pigs {
            for k in j..n_pigs {
                cd[i] += 2f64 * e_vecs[[j, i]] * e_vecs[[k, i]] * r_mu_cross_cache[[j, k]];
            }
        }
        cd[i] *= coeffs[i];
    }
    cd
}

/// Creates a cache of (r_i - r_j) * (mu_i x mu_j).
///
/// These values are used in each iteration of the CD calculation but do not
/// change between iterations.
pub fn populate_r_mu_cross_cache(mus: ArrayView2<f64>, rs: ArrayView2<f64>) -> Array2<f64> {
    let n = mus.nrows();
    let mut cache = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let r_i = rs.row(i);
            let r_j = rs.row(j);
            let r = &r_i - &r_j;
            let mu_i = mus.row(i);
            let mu_j = mus.row(j);
            let mu_cross = cross(mu_i, mu_j);
            cache[[i, j]] = dot(r.view(), mu_cross.view());
        }
    }
    cache
}

/// Computes the transition dipole moments for each exciton
///
/// The exciton dipole moments are superpositions of the individual pigment
/// dipole moments where the weights of the superposition come from the eigenvectors
/// of the Hamiltonian.
pub fn exciton_dipole_moments(e_vecs: ArrayView2<f64>, p_mus: ArrayView2<f64>) -> Array2<f64> {
    let n_pigs = e_vecs.ncols();
    let mut e_mus = Array2::zeros(p_mus.raw_dim());
    for i in 0..n_pigs {
        let weighted_mu = Zip::from(e_vecs.column(i))
            .and(p_mus.rows())
            .fold(Array1::zeros(3), |acc, &w, mu| acc + w * &mu);
        e_mus.row_mut(i).assign(&weighted_mu);
    }
    e_mus
}

/// Diagonalize a Hamiltonian, returns eigenvalues and eigenvectors
pub fn diagonalize(ham: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
    // Normally you would need to convert the Hamiltonian to an array with Fortran
    // memory ordering, but the matrix is symmetric so the transpose doesn't actually
    // change the matrix.
    let ham = ham.clone();
    let ham_size = ham.nrows() as i32;
    let mut e_vals_real = Vec::with_capacity(ham_size as usize);
    e_vals_real.resize(ham_size as usize, 0.0);
    let mut e_vals_imag = Vec::with_capacity(ham_size as usize);
    e_vals_imag.resize(ham_size as usize, 0.0);
    let mut e_vecs_left = Vec::with_capacity((ham_size as usize).pow(2));
    e_vecs_left.resize((ham_size as usize).pow(2), 0.0);
    let mut e_vecs_right = Vec::with_capacity((ham_size as usize).pow(2));
    e_vecs_right.resize((ham_size as usize).pow(2), 0.0);
    let mut work_arr = Vec::with_capacity(8 * ham_size as usize);
    work_arr.resize(8 * ham_size as usize, 0.0);
    let mut info: i32 = 0;
    unsafe {
        dgeev(
            b'N',                                   // Don't calculate left eigenvectors
            b'V',                                   // Do calculate the right eigenvectors
            ham_size,                               // The dimensions of the Hamiltonian
            ham.to_owned().as_slice_mut().unwrap(), // The underlying data in the Hamiltonian array
            ham_size,                    // The "leading" dimension of `ham`, `ham` is square
            e_vals_real.as_mut_slice(),  // The place to put the real parts of the eigenvalues
            &mut e_vals_imag,            // The place to put the imaginary parts of the eigenvalues
            e_vecs_left.as_mut_slice(),  // Where the left eigenvectors will be stored
            ham_size,                    // The leading dimension of the left eigenvectors array
            e_vecs_right.as_mut_slice(), // Where the right eigenvectors will be stored
            ham_size,                    // The leading dimension of the right eigenvectors array
            work_arr.as_mut_slice(),     // I don't know what this is for
            8 * ham_size,                // The size of the work array
            &mut info,                   // Will contain the status of the operation when done
        );
    }
    let e_vals: Array1<f64> = Array1::from_vec(e_vals_real).reversed_axes();
    let e_vecs = Array2::from_shape_vec((ham_size as usize, ham_size as usize), e_vecs_right)
        .unwrap()
        .reversed_axes()
        .as_standard_layout()
        .to_owned();
    (e_vals, e_vecs)
}

/// Compute the stick spectrum of a Hamiltonian
///
/// `ham`: An NxN Hamiltonian matrix
/// `mus`: An Nx3 array of dipole moments, one row for each pigment
/// `pos`: An Nx3 array of positions, one row for each pigment
pub fn compute_stick_spectrum(
    ham: ArrayView2<f64>,
    mus: ArrayView2<f64>,
    rs: ArrayView2<f64>,
) -> StickSpectrum {
    let (e_vals, e_vecs) = diagonalize(ham);
    let exc_mus = exciton_dipole_moments(e_vecs.view(), mus);
    let stick_abs = stick_abs_single(exc_mus.view());
    let stick_cd = stick_cd_single(e_vecs.view(), mus.view(), rs.view(), e_vals.view());
    StickSpectrum {
        e_vecs,
        e_vals,
        mus: exc_mus,
        stick_abs,
        stick_cd,
    }
}

/// Compute the stick spectra of multiple Hamiltonians
///
/// `ham`: An mxNxN array of `m` `NxN` Hamiltonians
/// `mus`: An mxNx3 array of `m` dipole moments
/// `rs`: An mxNx3 array of `m` pigment positions
pub fn compute_stick_spectra(
    hams: ArrayView3<f64>,
    mus: ArrayView3<f64>,
    rs: ArrayView3<f64>,
) -> Vec<StickSpectrum> {
    let dummy_stick = StickSpectrum {
        e_vals: arr1(&[]),
        e_vecs: arr2(&[[], []]),
        mus: arr2(&[[], []]),
        stick_abs: arr1(&[]),
        stick_cd: arr1(&[]),
    };
    let mut sticks: Vec<StickSpectrum> = Vec::with_capacity(hams.dim().0);
    sticks.resize(hams.dim().0, dummy_stick);
    Zip::from(hams.axis_iter(Axis(0)))
        .and(mus.axis_iter(Axis(0)))
        .and(rs.axis_iter(Axis(0)))
        .and(&mut sticks)
        .par_for_each(|h, m, r, s| *s = compute_stick_spectrum(h, m, r));
    sticks
}

/// Converts a bandwidth given as FWHM into 2*sigma^2
///
/// This is the denominator of the exponent of a Gaussian:
/// e^(-(x - mu)^2 / (2 * sigma^2))
///
/// The formula for converting between FWHM and sigma is:
/// FWHM = sigma * sqrt(8 * ln(2))
fn gauss_denom(fwhm: f64) -> f64 {
    fwhm.powi(2) / (4. * 2_f64.ln())
}

/// Computes each band and adds it to the spectrum
pub fn add_bands(
    mut spec: ArrayViewMut1<f64>,
    energies: ArrayView1<f64>,
    stick_strengths: ArrayView1<f64>,
    bw: f64,
    x: ArrayView1<f64>,
) {
    let denom = gauss_denom(bw);
    spec.assign(&x.mapv(|x_i| abs_at_x(x_i, denom, energies, stick_strengths)));
}

/// Determine the indices for which you actually need to compute the contribution of a band
pub fn band_cutoff_indices(center: f64, bw: f64, cutoff: f64, xs: &[f64]) -> (usize, usize) {
    let lower = xs.partition_point(|&x| x < (center - cutoff * bw));
    let upper = xs.partition_point(|&x| x < (center + cutoff * bw));
    (lower, upper)
}

/// Computes the band and adds it to the spectrum
pub fn add_cutoff_bands(
    mut spec: ArrayViewMut1<f64>,
    energies: ArrayView1<f64>,
    stick_strengths: ArrayView1<f64>,
    bws: &[f64],
    cutoff: f64,
    x: ArrayView1<f64>,
) {
    Zip::from(energies)
        .and(stick_strengths)
        .and(bws)
        .for_each(|&e, &strength, &bw| {
            let denom = gauss_denom(bw);
            let (lower, upper) = band_cutoff_indices(e, bw, cutoff, x.as_slice().unwrap());
            let band = x
                .slice(s![lower..upper])
                .mapv(|x_i| strength * (-(x_i - e).powi(2) / denom).exp());
            spec.slice_mut(s![lower..upper]).add_assign(&band);
        });
}

/// Compute the broadened spectrum of a stick spectrum
pub fn compute_broadened_spectrum_from_stick(
    energies: ArrayView1<f64>,
    dip_strengths: ArrayView1<f64>,
    rot_strengths: ArrayView1<f64>,
    config: &BroadeningConfig,
) -> BroadenedSpectrum {
    let x = Array::range(config.x_from, config.x_to, config.x_step);
    let mut abs = Array1::zeros(x.dim());
    let mut cd = Array1::zeros(x.dim());
    // add_bands(abs.view_mut(), energies, dip_strengths, config.bw, x.view());
    // add_bands(cd.view_mut(), energies, rot_strengths, config.bw, x.view());
    let bws = vec![config.bw; energies.len()];
    add_cutoff_bands(
        abs.view_mut(),
        energies,
        dip_strengths,
        bws.as_slice(),
        config.band_cutoff,
        x.view(),
    );
    add_cutoff_bands(
        cd.view_mut(),
        energies,
        rot_strengths,
        bws.as_slice(),
        config.band_cutoff,
        x.view(),
    );
    BroadenedSpectrum { x, abs, cd }
}

/// Compute the broadened spectrum of a stick spectrum with different bandwidths for each transition
pub fn compute_het_broadened_spectrum_from_stick(
    energies: ArrayView1<f64>,
    dipole_strengths: ArrayView1<f64>,
    rotational_strengths: ArrayView1<f64>,
    config: &BroadeningConfig,
) -> BroadenedSpectrum {
    let x = Array::range(config.x_from, config.x_to, config.x_step);
    let mut abs = Array1::zeros(x.dim());
    let mut cd = Array1::zeros(x.dim());
    add_cutoff_bands(
        abs.view_mut(),
        energies,
        dipole_strengths,
        config.abs_bws.as_slice(),
        config.band_cutoff,
        x.view(),
    );
    add_cutoff_bands(
        cd.view_mut(),
        energies,
        rotational_strengths,
        config.cd_bws.as_slice(),
        config.band_cutoff,
        x.view(),
    );
    BroadenedSpectrum { x, abs, cd }
}

/// Computes the absorption at a point given the dipole strengths and energies.
///
/// Note, this function works just as well for circular dichroism if you supply
/// rotational strengths instead of dipole strengths.
fn abs_at_x(x: f64, denom: f64, energies: ArrayView1<f64>, strengths: ArrayView1<f64>) -> f64 {
    Zip::from(&energies)
        .and(&strengths)
        .fold(0f64, |acc, &e, &s| {
            acc + s * (-(x - e).powi(2) / denom).exp()
        })
}

/// Compute the broadened spectra of a single Hamiltonian
pub fn compute_broadened_spectrum_from_ham(
    ham: ArrayView2<f64>,
    mus: ArrayView2<f64>,
    rs: ArrayView2<f64>,
    config: &BroadeningConfig,
) -> BroadenedSpectrum {
    let stick = compute_stick_spectrum(ham, mus, rs);
    compute_broadened_spectrum_from_stick(
        stick.e_vals.view(),
        stick.stick_abs.view(),
        stick.stick_cd.view(),
        config,
    )
}

/// Compute the broadened spectrum of a single Hamiltonian with different bandwidths for each transition
pub fn compute_het_broadened_spectrum_from_ham(
    ham: ArrayView2<f64>,
    mus: ArrayView2<f64>,
    rs: ArrayView2<f64>,
    config: &BroadeningConfig,
) -> BroadenedSpectrum {
    let stick = compute_stick_spectrum(ham, mus, rs);
    compute_het_broadened_spectrum_from_stick(
        stick.e_vals.view(),
        stick.stick_abs.view(),
        stick.stick_cd.view(),
        config,
    )
}

/// Compute a broadened spectrum from multiple Hamiltonians
///
/// `ham`: An mxNxN array of `m` `NxN` Hamiltonians
/// `mus`: An mxNx3 array of `m` dipole moments
/// `rs`: An mxNx3 array of `m` pigment positions
pub fn compute_broadened_spectrum_from_hams(
    hams: ArrayView3<f64>,
    mus: ArrayView3<f64>,
    rs: ArrayView3<f64>,
    config: &BroadeningConfig,
) -> BroadenedSpectrum {
    let x = Array::range(config.x_from, config.x_to, config.x_step);
    let n_hams = hams.dim().0;
    let mut abs_arr = Array2::zeros((x.dim(), n_hams));
    let mut cd_arr = Array2::zeros((x.dim(), n_hams));
    Zip::from(abs_arr.columns_mut())
        .and(cd_arr.columns_mut())
        .and(hams.axis_iter(Axis(0)))
        .and(mus.axis_iter(Axis(0)))
        .and(rs.axis_iter(Axis(0)))
        .par_for_each(|mut abs_col, mut cd_col, h, m, r| {
            let stick = compute_stick_spectrum(h, m, r);
            let broadened = compute_broadened_spectrum_from_stick(
                stick.e_vals.view(),
                stick.stick_abs.view(),
                stick.stick_cd.view(),
                config,
            );
            abs_col.assign(&broadened.abs);
            cd_col.assign(&broadened.cd);
        });
    // Unwrapping because we know the length of the axis can't be zero
    let abs = abs_arr.mean_axis(Axis(1)).unwrap();
    let cd = cd_arr.mean_axis(Axis(1)).unwrap();
    BroadenedSpectrum { x, abs, cd }
}

/// Compute the broadened spectra of multiple Hamiltonians with different bandwidths for each transition
pub fn compute_het_broadened_spectrum_from_hams(
    hams: ArrayView3<f64>,
    mus: ArrayView3<f64>,
    rs: ArrayView3<f64>,
    config: &BroadeningConfig,
) -> BroadenedSpectrum {
    let x = Array::range(config.x_from, config.x_to, config.x_step);
    let n_hams = hams.dim().0;
    let mut abs_arr = Array2::zeros((x.dim(), n_hams));
    let mut cd_arr = Array2::zeros((x.dim(), n_hams));
    Zip::from(abs_arr.columns_mut())
        .and(cd_arr.columns_mut())
        .and(hams.axis_iter(Axis(0)))
        .and(mus.axis_iter(Axis(0)))
        .and(rs.axis_iter(Axis(0)))
        .par_for_each(|mut abs_col, mut cd_col, h, m, r| {
            let stick = compute_stick_spectrum(h, m, r);
            let broadened = compute_het_broadened_spectrum_from_stick(
                stick.e_vals.view(),
                stick.stick_abs.view(),
                stick.stick_cd.view(),
                config,
            );
            abs_col.assign(&broadened.abs);
            cd_col.assign(&broadened.cd);
        });
    // Unwrapping because we know the length of the axis can't be zero
    let abs = abs_arr.mean_axis(Axis(1)).unwrap();
    let cd = cd_arr.mean_axis(Axis(1)).unwrap();
    BroadenedSpectrum { x, abs, cd }
}

/// Compute absorbance and CD spectra from first principles.
#[pymodule]
fn ham2spec(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Computes the transition dipole moments for each exciton.
    ///
    /// `e_vecs`: An NxN array of eigenvectors of the Hamiltonian, one vector per column
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    #[pyfn(m)]
    #[pyo3(name = "exciton_mus")]
    fn exciton_dipole_moments_py<'py>(
        py: Python<'py>,
        e_vecs: PyReadonlyArray2<f64>,
        pig_mus: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        exciton_dipole_moments(e_vecs.as_array().view(), pig_mus.as_array().view()).into_pyarray(py)
    }

    /// Compute the absorbance stick spectrum
    ///
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    #[pyfn(m)]
    #[pyo3(name = "stick_abs_single")]
    fn stick_abs_single_py<'py>(py: Python<'py>, mus: PyReadonlyArray2<f64>) -> &'py PyArray1<f64> {
        stick_abs_single(mus.as_array().view()).into_pyarray(py)
    }

    /// Compute the CD stick spectrum
    ///
    /// `e_vecs`: An NxN array of eigenvectors of the Hamiltonian, one vector per column
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    /// `rs`: An Nx3 array of positions, one row for each pigment
    /// `energies`: An Nx1 array of eigenvalues of the Hamiltonian
    #[pyfn(m)]
    #[pyo3(name = "stick_cd_single")]
    fn stick_cd_single_py<'py>(
        py: Python<'py>,
        e_vecs: PyReadonlyArray2<f64>,
        mus: PyReadonlyArray2<f64>,
        rs: PyReadonlyArray2<f64>,
        energies: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        stick_cd_single(
            e_vecs.as_array(),
            mus.as_array(),
            rs.as_array(),
            energies.as_array(),
        )
        .into_pyarray(py)
    }

    /// Compute the absorbance and CD stick spectrum of a single Hamiltonian
    ///
    /// `ham`: An NxN Hamiltonian matrix
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    /// `rs`: An Nx3 array of positions, one row for each pigment
    #[pyfn(m)]
    #[pyo3(name = "compute_stick_spectrum")]
    fn compute_stick_spectrum_py<'py>(
        py: Python<'py>,
        ham: PyReadonlyArray2<f64>,
        mus: PyReadonlyArray2<f64>,
        rs: PyReadonlyArray2<f64>,
    ) -> PyObject {
        let stick = compute_stick_spectrum(ham.as_array(), mus.as_array(), rs.as_array());
        stick.to_object(py)
    }

    /// Compute the broadened spectra of a single stick spectrum
    #[pyfn(m)]
    #[pyo3(name = "compute_broadened_spectrum_from_stick")]
    fn compute_broadened_spectrum_from_stick_py<'py>(
        py: Python<'py>,
        stick: &'py PyDict,
        config: PyObject,
    ) -> PyResult<&'py PyDict> {
        let energies = stick
            .get_item("e_vals")
            .ok_or(PyKeyError::new_err("e_vals"))?
            .downcast::<PyArray1<f64>>()?
            .to_owned_array();
        let stick_abs = stick
            .get_item("stick_abs")
            .ok_or(PyKeyError::new_err("stick_abs"))?
            .downcast::<PyArray1<f64>>()?
            .to_owned_array();
        let stick_cd = stick
            .get_item("stick_cd")
            .ok_or(PyKeyError::new_err("stick_cd"))?
            .downcast::<PyArray1<f64>>()?
            .to_owned_array();
        let b_config: BroadeningConfig = config.extract(py)?;
        let broadened = compute_broadened_spectrum_from_stick(
            energies.view(),
            stick_abs.view(),
            stick_cd.view(),
            &b_config,
        );
        let dict = PyDict::new(py);
        dict.set_item("x", broadened.x.to_pyarray(py))?;
        dict.set_item("abs", broadened.abs.to_pyarray(py))?;
        dict.set_item("cd", broadened.cd.to_pyarray(py))?;
        Ok(dict)
    }

    /// Compute the broadened spectra of a single Hamiltonian
    ///
    /// `ham`: An NxN Hamiltonian matrix
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    /// `rs`: An Nx3 array of positions, one row for each pigment
    /// `config`: An object (`fmo_analysis.util.Config`) containing the configuration for broadening
    #[pyfn(m)]
    #[pyo3(name = "compute_broadened_spectrum_from_ham")]
    fn compute_broadened_spectrum_from_ham_py<'py>(
        py: Python<'py>,
        ham: PyReadonlyArray2<f64>,
        mus: PyReadonlyArray2<f64>,
        rs: PyReadonlyArray2<f64>,
        config: PyObject,
    ) -> PyResult<&'py PyDict> {
        let b_config: BroadeningConfig = config.extract(py)?;
        let broadened = compute_broadened_spectrum_from_ham(
            ham.as_array(),
            mus.as_array(),
            rs.as_array(),
            &b_config,
        );
        let dict = PyDict::new(py);
        dict.set_item("x", broadened.x.to_pyarray(py))?;
        dict.set_item("abs", broadened.abs.to_pyarray(py))?;
        dict.set_item("cd", broadened.cd.to_pyarray(py))?;
        Ok(dict)
    }

    /// Compute the broadened spectrum from a stick spectrum using different bandwidths for each transition
    #[pyfn(m)]
    #[pyo3(name = "compute_het_broadened_spectrum_from_stick")]
    fn compute_het_broadened_spectrum_from_stick_py<'py>(
        py: Python<'py>,
        stick: &'py PyDict,
        config: PyObject,
    ) -> PyResult<&'py PyDict> {
        let energies = stick
            .get_item("e_vals")
            .ok_or(PyKeyError::new_err("e_vals"))?
            .downcast::<PyArray1<f64>>()?
            .to_owned_array();
        let stick_abs = stick
            .get_item("stick_abs")
            .ok_or(PyKeyError::new_err("stick_abs"))?
            .downcast::<PyArray1<f64>>()?
            .to_owned_array();
        let stick_cd = stick
            .get_item("stick_cd")
            .ok_or(PyKeyError::new_err("stick_cd"))?
            .downcast::<PyArray1<f64>>()?
            .to_owned_array();
        let b_config: BroadeningConfig = config.extract(py)?;
        let broadened = compute_het_broadened_spectrum_from_stick(
            energies.view(),
            stick_abs.view(),
            stick_cd.view(),
            &b_config,
        );
        let dict = PyDict::new(py);
        dict.set_item("x", broadened.x.to_pyarray(py))?;
        dict.set_item("abs", broadened.abs.to_pyarray(py))?;
        dict.set_item("cd", broadened.cd.to_pyarray(py))?;
        Ok(dict)
    }

    /// Compute the broadened spectra of a single Hamiltonian
    ///
    /// `ham`: An NxN Hamiltonian matrix
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    /// `rs`: An Nx3 array of positions, one row for each pigment
    /// `config`: An object (`fmo_analysis.util.Config`) containing the configuration for broadening
    #[pyfn(m)]
    #[pyo3(name = "compute_het_broadened_spectrum_from_ham")]
    fn compute_het_broadened_spectrum_from_ham_py<'py>(
        py: Python<'py>,
        ham: PyReadonlyArray2<f64>,
        mus: PyReadonlyArray2<f64>,
        rs: PyReadonlyArray2<f64>,
        config: PyObject,
    ) -> PyResult<&'py PyDict> {
        let b_config: BroadeningConfig = config.extract(py)?;
        let broadened = compute_het_broadened_spectrum_from_ham(
            ham.as_array(),
            mus.as_array(),
            rs.as_array(),
            &b_config,
        );
        let dict = PyDict::new(py);
        dict.set_item("x", broadened.x.to_pyarray(py))?;
        dict.set_item("abs", broadened.abs.to_pyarray(py))?;
        dict.set_item("cd", broadened.cd.to_pyarray(py))?;
        Ok(dict)
    }

    /// Compute the stick spectra of multiple Hamiltonians
    ///
    /// `ham`: An mxNxN array of `m` `NxN` Hamiltonians
    /// `mus`: An mxNx3 array of `m` dipole moments
    /// `rs`: An mxNx3 array of `m` pigment positions
    #[pyfn(m)]
    #[pyo3(name = "compute_stick_spectra")]
    fn compute_stick_spectra_py<'py>(
        py: Python<'py>,
        hams: PyReadonlyArray3<f64>,
        mus: PyReadonlyArray3<f64>,
        rs: PyReadonlyArray3<f64>,
    ) -> &'py PyList {
        let sticks = compute_stick_spectra(hams.as_array(), mus.as_array(), rs.as_array());
        PyList::new(py, sticks)
    }

    /// Compute the broadened spectrum from multiple Hamiltonians
    ///
    /// `ham`: An mxNxN array of `m` `NxN` Hamiltonians
    /// `mus`: An mxNx3 array of `m` dipole moments
    /// `rs`: An mxNx3 array of `m` pigment positions
    /// `config`: An object (`fmo_analysis.util.Config`) containing the configuration for broadening
    #[pyfn(m)]
    #[pyo3(name = "compute_broadened_spectrum_from_hams")]
    fn compute_broadened_spectrum_from_hams_py<'py>(
        py: Python<'py>,
        hams: PyReadonlyArray3<f64>,
        mus: PyReadonlyArray3<f64>,
        rs: PyReadonlyArray3<f64>,
        config: BroadeningConfig,
    ) -> PyObject {
        let spec = compute_broadened_spectrum_from_hams(
            hams.as_array(),
            mus.as_array(),
            rs.as_array(),
            &config,
        );
        spec.to_object(py)
    }

    /// Compute the broadened spectra from multiple Hamiltonians
    ///
    /// `ham`: An mxNxN array of `m` `NxN` Hamiltonians
    /// `mus`: An mxNx3 array of `m` dipole moments
    /// `rs`: An mxNx3 array of `m` pigment positions
    /// `config`: An object (`fmo_analysis.util.Config`) containing the configuration for broadening
    #[pyfn(m)]
    #[pyo3(name = "compute_het_broadened_spectrum_from_hams")]
    fn compute_het_broadened_spectrum_from_hams_py<'py>(
        py: Python<'py>,
        hams: PyReadonlyArray3<f64>,
        mus: PyReadonlyArray3<f64>,
        rs: PyReadonlyArray3<f64>,
        config: BroadeningConfig,
    ) -> PyObject {
        let spec = compute_het_broadened_spectrum_from_hams(
            hams.as_array(),
            mus.as_array(),
            rs.as_array(),
            &config,
        );
        spec.to_object(py)
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{s, Array2, Array3};

    fn load_ham() -> Array2<f64> {
        let contents = include_str!("../validation_data/hamiltonian.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array2::from_shape_vec((7, 7), data).unwrap()
    }

    fn load_dipole_moments() -> Array2<f64> {
        let contents = include_str!("../validation_data/dipole_moments.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array2::from_shape_vec((7, 3), data).unwrap()
    }

    fn load_positions() -> Array2<f64> {
        let contents = include_str!("../validation_data/positions.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array2::from_shape_vec((7, 3), data).unwrap()
    }

    fn load_eigenvalues() -> Array1<f64> {
        let contents = include_str!("../validation_data/eigenvalues.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array1::from_vec(data)
    }

    fn load_eigenvectors() -> Array2<f64> {
        let contents = include_str!("../validation_data/eigenvectors.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array2::from_shape_vec((7, 7), data).unwrap()
    }

    fn load_exciton_dipole_moments() -> Array2<f64> {
        let contents = include_str!("../validation_data/exciton_dipole_moments.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array2::from_shape_vec((7, 3), data).unwrap()
    }

    fn load_dipole_strengths() -> Array1<f64> {
        let contents = include_str!("../validation_data/dipole_strengths.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array1::from_vec(data)
    }

    fn load_rotational_strengths() -> Array1<f64> {
        let contents = include_str!("../validation_data/rotational_strengths.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array1::from_vec(data)
    }

    fn load_x() -> Array1<f64> {
        let contents = include_str!("../validation_data/x.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array1::from_vec(data)
    }

    fn load_abs() -> Array1<f64> {
        let contents = include_str!("../validation_data/abs.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array1::from_vec(data)
    }

    fn load_cd() -> Array1<f64> {
        let contents = include_str!("../validation_data/cd.txt");
        let data: Vec<f64> = contents
            .split("\n")
            .map(|line| line.parse::<f64>().unwrap())
            .collect();
        Array1::from_vec(data)
    }

    fn load_config() -> BroadeningConfig {
        BroadeningConfig {
            x_from: 11790.0,
            x_to: 13300.0,
            x_step: 1.0,
            bw: 120.0,
            abs_bws: vec![120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
            cd_bws: vec![120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
            band_cutoff: 3.0,
        }
    }

    #[test]
    fn correctly_loads_hamiltonian() {
        let ham = load_ham();
        assert_relative_eq!(ham[[1, 2]], 28.0);
    }

    #[test]
    fn diagonalizes_brixner_hamiltonian() {
        let ham = load_ham();
        let good_e_vals = load_eigenvalues();
        let good_e_vecs = load_eigenvectors();
        let (test_e_vals, test_e_vecs) = diagonalize(ham.view());
        println!("{}", test_e_vecs);
        assert_abs_diff_eq!(test_e_vals, good_e_vals, epsilon = 1.0);
        assert_relative_eq!(test_e_vecs, good_e_vecs, epsilon = 1e-4);
    }

    #[test]
    fn computes_brixner_exciton_dipole_moments() {
        let e_vecs = load_eigenvectors();
        let dipole_moments = load_dipole_moments();
        let test_exc_dipole_moments = exciton_dipole_moments(e_vecs.view(), dipole_moments.view());
        let good_exc_dipole_moments = load_exciton_dipole_moments();
        assert_abs_diff_eq!(
            test_exc_dipole_moments,
            good_exc_dipole_moments,
            epsilon = 1e-4
        );
    }

    #[test]
    fn computes_brixner_stick_abs() {
        let exciton_dpm = load_exciton_dipole_moments();
        let test_stick_abs = stick_abs_single(exciton_dpm.view());
        let good_stick_abs = load_dipole_strengths();
        assert_abs_diff_eq!(test_stick_abs, good_stick_abs, epsilon = 1e-4);
    }

    #[test]
    fn computes_brixner_stick_cd() {
        let dipole_moments = load_dipole_moments();
        let e_vecs = load_eigenvectors();
        let e_vals = load_eigenvalues();
        let pig_pos = load_positions();
        let good_stick_cd = load_rotational_strengths();
        let test_stick_cd = stick_cd_single(
            e_vecs.view(),
            dipole_moments.view(),
            pig_pos.view(),
            e_vals.view(),
        );
        assert_abs_diff_eq!(test_stick_cd, good_stick_cd, epsilon = 1e-4);
    }

    #[test]
    fn computes_broadened_spectra() {
        let x = load_x();
        let abs = load_abs();
        let cd = load_cd();
        let ham = load_ham();
        let mus = load_dipole_moments();
        let rs = load_positions();
        let config = load_config();
        let spec = compute_broadened_spectrum_from_ham(ham.view(), mus.view(), rs.view(), &config);
        assert_abs_diff_eq!(x, spec.x, epsilon = 1e-4);
        assert_abs_diff_eq!(abs, spec.abs, epsilon = 1e-4);
        assert_abs_diff_eq!(cd, spec.cd, epsilon = 1e-4);
    }

    #[test]
    fn computes_multiple_stick_spectra() {
        let n_hams = 100;
        let ham = load_ham();
        let mus = load_dipole_moments();
        let rs = load_positions();
        let dipole_strengths = load_dipole_strengths();
        let rotational_strengths = load_rotational_strengths();
        let mut ham_multi = Array3::zeros((n_hams, 7, 7));
        let mut mus_multi = Array3::zeros((n_hams, 7, 3));
        let mut rs_multi = Array3::zeros((n_hams, 7, 3));
        for i in 0..n_hams {
            ham_multi.slice_mut(s![i, .., ..]).assign(&ham);
            mus_multi.slice_mut(s![i, .., ..]).assign(&mus);
            rs_multi.slice_mut(s![i, .., ..]).assign(&rs);
        }
        let sticks = compute_stick_spectra(ham_multi.view(), mus_multi.view(), rs_multi.view());
        for s in sticks.iter() {
            assert_abs_diff_eq!(dipole_strengths, s.stick_abs, epsilon = 1e-4);
            assert_abs_diff_eq!(rotational_strengths, s.stick_cd, epsilon = 1e-4);
        }
    }

    #[test]
    fn computes_multiple_broadened_spectra() {
        let n_hams = 100;
        let ham = load_ham();
        let mus = load_dipole_moments();
        let rs = load_positions();
        let abs = load_abs();
        let cd = load_cd();
        let mut ham_multi = Array3::zeros((n_hams, 7, 7));
        let mut mus_multi = Array3::zeros((n_hams, 7, 3));
        let mut rs_multi = Array3::zeros((n_hams, 7, 3));
        for i in 0..n_hams {
            ham_multi.slice_mut(s![i, .., ..]).assign(&ham);
            mus_multi.slice_mut(s![i, .., ..]).assign(&mus);
            rs_multi.slice_mut(s![i, .., ..]).assign(&rs);
        }
        let config = load_config();
        let spec = compute_broadened_spectrum_from_hams(
            ham_multi.view(),
            mus_multi.view(),
            rs_multi.view(),
            &config,
        );
        assert_abs_diff_eq!(abs, spec.abs, epsilon = 1e-4);
        assert_abs_diff_eq!(cd, spec.cd, epsilon = 1e-4);
    }

    #[test]
    fn finds_band_cutoff_indices() {
        let bw = 5.0;
        let cutoff = 1.0;
        let center = 50.0;
        let expected_lower = (center - bw) as usize;
        let expected_upper = (center + bw) as usize;
        let xs: Array1<f64> = Array1::range(0.0, 100.0, 1.0);
        let (lower, upper) = band_cutoff_indices(center, bw, cutoff, xs.as_slice().unwrap());
        assert_eq!(lower, expected_lower);
        assert_eq!(upper, expected_upper);
    }

    #[test]
    fn band_indices_are_whole_range_with_large_bw() {
        let bw = 100.0;
        let cutoff = 3.0;
        let center = 50.0;
        let expected_lower = 0;
        let expected_upper = 100;
        let xs: Array1<f64> = Array1::range(0.0, 100.0, 1.0);
        let (lower, upper) = band_cutoff_indices(center, bw, cutoff, xs.as_slice().unwrap());
        assert_eq!(lower, expected_lower);
        assert_eq!(upper, expected_upper);
    }

    #[test]
    fn computes_het_broadened_spectrum_from_stick() {
        let ham = load_ham();
        let mus = load_dipole_moments();
        let rs = load_positions();
        let stick = compute_stick_spectrum(ham.view(), mus.view(), rs.view());
        let known_abs = load_abs();
        let known_cd = load_cd();
        let config = load_config();
        let broadened = compute_het_broadened_spectrum_from_stick(
            stick.e_vals.view(),
            stick.stick_abs.view(),
            stick.stick_cd.view(),
            &config,
        );
        assert_abs_diff_eq!(broadened.abs, known_abs, epsilon = 1e-4);
        assert_abs_diff_eq!(broadened.cd, known_cd, epsilon = 1e-4);
    }
}
