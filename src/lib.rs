extern crate lapack_src;
#[cfg(test)]
use approx::{assert_abs_diff_eq, assert_relative_eq};
use lapack::sgeev;
use numpy::ndarray::{arr1, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A stick spectrum computed from a single Hamiltonian and associated pigments.
#[derive(Debug, Clone)]
pub struct StickSpectrum {
    /// The eigenvectors, one per column.
    pub e_vecs: Array2<f32>,

    /// The energies of the excitons.
    pub e_vals: Array1<f32>,

    /// The transition dipole moments of the excitons.
    pub mus: Array2<f32>,

    /// The absorption (dipole strength) of each exciton.
    pub stick_abs: Array1<f32>,

    /// The circular dichroism of each exciton.
    pub stick_cd: Array1<f32>,
}

/// Compute the dot product of 2 3-vectors
fn dot(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the cross-product of 2 3-vectors
fn cross(a: ArrayView1<f32>, b: ArrayView1<f32>) -> Array1<f32> {
    arr1(&[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

/// Deletes a pigment from the Hamiltonian
pub fn delete_pigment_single(mut ham: ArrayViewMut2<f32>, mut mus: ArrayViewMut2<f32>, del: usize) {
    ham.row_mut(del).fill(0f32);
    ham.column_mut(del).fill(0f32);
    mus.row_mut(del).fill(0f32);
}

/// Compute the absorbance stick spectrum
///
/// The eigenvectors must be arranged into columns, and the pigment dipole moments
/// must be arranged into rows. See [`compute_stick_spectrum`] for the expected
/// layout of `mus`.
pub fn stick_abs_single(mus: &Array2<f32>) -> Array1<f32> {
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
/// layout of `mus` and `pos`.
pub fn stick_cd_single(
    e_vecs: ArrayView2<f32>,
    mus: ArrayView2<f32>,
    pos: ArrayView2<f32>,
    energies: ArrayView1<f32>,
) -> Array1<f32> {
    let coeffs: Vec<f32> = energies
        .iter()
        .map(|e| {
            let wavelength = if e < &1.0 { 1e8 / 100_000f32 } else { 1e8 / e };
            2f32 * core::f32::consts::PI / wavelength
        })
        .collect();
    let mut cd = Array1::zeros(energies.raw_dim());
    let n_pigs = e_vecs.ncols();
    let r_mu_cross_cache = populate_r_mu_cross_cache(mus, pos);
    for i in 0..n_pigs {
        for j in 0..n_pigs {
            for k in j..n_pigs {
                cd[i] += 2f32 * e_vecs[[j, i]] * e_vecs[[k, i]] * r_mu_cross_cache[[j, k]];
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
pub fn populate_r_mu_cross_cache(mus: ArrayView2<f32>, pos: ArrayView2<f32>) -> Array2<f32> {
    let n = mus.nrows();
    let mut cache = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let r_i = pos.row(i);
            let r_j = pos.row(j);
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
pub fn exciton_dipole_moments(e_vecs: &Array2<f32>, p_mus: &Array2<f32>) -> Array2<f32> {
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
pub fn diagonalize(ham: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    // Normally you would need to convert the Hamiltonian to an array with Fortran
    // memory ordering, but the matrix is symmetric so the transpose doesn't actually
    // change the matrix.
    let mut ham = ham.clone();
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
        sgeev(
            b'N',                        // Don't calculate left eigenvectors
            b'V',                        // Do calculate the right eigenvectors
            ham_size,                    // The dimensions of the Hamiltonian
            ham.as_slice_mut().unwrap(), // The underlying data in the Hamiltonian array
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
    let e_vals: Array1<f32> = Array1::from_vec(e_vals_real).reversed_axes();
    let e_vecs = Array2::from_shape_vec((ham_size as usize, ham_size as usize), e_vecs_right)
        .unwrap()
        .reversed_axes()
        .as_standard_layout()
        .to_owned();
    (e_vals, e_vecs)
}

/// Compute the stick spectra of a Hamiltonian
///
/// `ham`: An NxN Hamiltonian matrix
/// `mus`: An Nx3 array of dipole moments, one row for each pigment
/// `pos`: An Nx3 array of positions, one row for each pigment
pub fn compute_stick_spectrum(
    ham: ArrayView2<f32>,
    mus: ArrayView2<f32>,
    pos: ArrayView2<f32>,
) -> StickSpectrum {
    let (e_vals, e_vecs) = diagonalize(&ham.to_owned());
    let exc_mus = exciton_dipole_moments(&e_vecs, &mus.to_owned());
    let stick_abs = stick_abs_single(&exc_mus);
    let stick_cd = stick_cd_single(e_vecs.view(), mus.view(), pos.view(), e_vals.view());
    StickSpectrum {
        e_vecs,
        e_vals,
        mus: exc_mus,
        stick_abs,
        stick_cd,
    }
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
        e_vecs: PyReadonlyArray2<f32>,
        pig_mus: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        exciton_dipole_moments(
            &e_vecs.as_array().to_owned(),
            &pig_mus.as_array().to_owned(),
        )
        .into_pyarray(py)
    }

    /// Compute the absorbance stick spectrum
    ///
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    #[pyfn(m)]
    #[pyo3(name = "stick_abs_single")]
    fn stick_abs_single_py<'py>(py: Python<'py>, mus: PyReadonlyArray2<f32>) -> &'py PyArray1<f32> {
        stick_abs_single(&mus.as_array().to_owned()).into_pyarray(py)
    }

    /// Compute the CD stick spectrum
    ///
    /// `e_vecs`: An NxN array of eigenvectors of the Hamiltonian, one vector per column
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    /// `pos`: An Nx3 array of positions, one row for each pigment
    /// `energies`: An Nx1 array of eigenvalues of the Hamiltonian
    #[pyfn(m)]
    #[pyo3(name = "stick_cd_single")]
    fn stick_cd_single_py<'py>(
        py: Python<'py>,
        e_vecs: PyReadonlyArray2<f32>,
        mus: PyReadonlyArray2<f32>,
        pos: PyReadonlyArray2<f32>,
        energies: PyReadonlyArray1<f32>,
    ) -> &'py PyArray1<f32> {
        stick_cd_single(
            e_vecs.as_array(),
            mus.as_array(),
            pos.as_array(),
            energies.as_array(),
        )
        .into_pyarray(py)
    }

    /// Compute the absorbance and CD stick spectrum of a single Hamiltonian
    ///
    /// `ham`: An NxN Hamiltonian matrix
    /// `mus`: An Nx3 array of dipole moments, one row for each pigment
    /// `pos`: An Nx3 array of positions, one row for each pigment
    #[pyfn(m)]
    #[pyo3(name = "compute_stick_spectrum")]
    fn compute_stick_spectrum_py<'py>(
        py: Python<'py>,
        ham: PyReadonlyArray2<f32>,
        pig_mus: PyReadonlyArray2<f32>,
        pig_pos: PyReadonlyArray2<f32>,
    ) -> &'py PyDict {
        let sticks = compute_stick_spectrum(ham.as_array(), pig_mus.as_array(), pig_pos.as_array());
        let dict = PyDict::new(py);
        dict.set_item("e_vals", sticks.e_vals.into_pyarray(py))
            .unwrap();
        dict.set_item("e_vecs", sticks.e_vecs.into_pyarray(py))
            .unwrap();
        dict.set_item("exciton_mus", sticks.mus.into_pyarray(py))
            .unwrap();
        dict.set_item("stick_abs", sticks.stick_abs.into_pyarray(py))
            .unwrap();
        dict.set_item("stick_cd", sticks.stick_cd.into_pyarray(py))
            .unwrap();
        dict
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use numpy::ndarray::Array2;

    // These are known-good results we'll test against
    const BRIXNER_HAM: [f32; 49] = [
        // 7x7 matrix unraveled
        1.242000000000000000e+04,
        -1.060000000000000000e+02,
        8.000000000000000000e+00,
        -5.000000000000000000e+00,
        6.000000000000000000e+00,
        -8.000000000000000000e+00,
        -4.000000000000000000e+00,
        -1.060000000000000000e+02,
        1.256000000000000000e+04,
        2.800000000000000000e+01,
        6.000000000000000000e+00,
        2.000000000000000000e+00,
        1.300000000000000000e+01,
        1.000000000000000000e+00,
        8.000000000000000000e+00,
        2.800000000000000000e+01,
        1.214000000000000000e+04,
        -6.200000000000000000e+01,
        -1.000000000000000000e+00,
        -9.000000000000000000e+00,
        1.700000000000000000e+01,
        -5.000000000000000000e+00,
        6.000000000000000000e+00,
        -6.200000000000000000e+01,
        1.231500000000000000e+04,
        -7.000000000000000000e+01,
        -1.900000000000000000e+01,
        -5.700000000000000000e+01,
        6.000000000000000000e+00,
        2.000000000000000000e+00,
        -1.000000000000000000e+00,
        -7.000000000000000000e+01,
        1.246000000000000000e+04,
        4.000000000000000000e+01,
        -2.000000000000000000e+00,
        -8.000000000000000000e+00,
        1.300000000000000000e+01,
        -9.000000000000000000e+00,
        -1.900000000000000000e+01,
        4.000000000000000000e+01,
        1.250000000000000000e+04,
        3.200000000000000000e+01,
        -4.000000000000000000e+00,
        1.000000000000000000e+00,
        1.700000000000000000e+01,
        -5.700000000000000000e+01,
        -2.000000000000000000e+00,
        3.200000000000000000e+01,
        1.240000000000000000e+04,
    ];
    const BRIXNER_DIPOLE_MOMENTS: [f32; 21] = [
        // 7x3 matrix unraveled, one dipole moment per row
        -7.409999999999999920e-01,
        -5.605999999999999872e-01,
        -3.695999999999999841e-01,
        -8.570999999999999730e-01,
        5.038000000000000256e-01,
        -1.073000000000000065e-01,
        -1.970999999999999974e-01,
        9.574000000000000288e-01,
        -2.109999999999999931e-01,
        -7.992000000000000215e-01,
        -5.335999999999999632e-01,
        -2.766000000000000125e-01,
        -7.368999999999999995e-01,
        6.558000000000000496e-01,
        1.640999999999999959e-01,
        -1.350000000000000089e-01,
        -8.791999999999999815e-01,
        4.568999999999999728e-01,
        -4.950999999999999845e-01,
        -7.083000000000000407e-01,
        -5.030999999999999917e-01,
    ];
    const BRIXNER_PIG_POS: [f32; 21] = [
        // 7x3 matrix unraveled, one position vector per row
        2.625349998474121094e+01,
        2.782999992370605469e+00,
        -1.110700035095214844e+01,
        1.554899978637695312e+01,
        -1.585500001907348633e+00,
        -1.687949943542480469e+01,
        3.403500080108642578e+00,
        -1.370650005340576172e+01,
        -1.425450038909912109e+01,
        6.876500129699707031e+00,
        -2.091850090026855469e+01,
        -6.376999855041503906e+00,
        1.937750053405761719e+01,
        -1.849099922180175781e+01,
        -1.491500020027160645e+00,
        2.174550056457519531e+01,
        -7.323999881744384766e+00,
        3.115000128746032715e-01,
        1.058049964904785156e+01,
        -8.205499649047851562e+00,
        -5.826499938964843750e+00,
    ];
    const BRIXNER_E_VALS: [f32; 7] = [
        1.211586132812500000e+04,
        1.262016503906250000e+04,
        1.254819335937500000e+04,
        1.227897460937500000e+04,
        1.236427050781250000e+04,
        1.245148828125000000e+04,
        1.241604589843750000e+04,
    ];
    const BRIXNER_E_VECS: [f32; 49] = [
        // 7x7 matrix unraveled, one eigenvector per column
        -4.62779193e-02,
        -4.64960172e-01,
        7.95694100e-02,
        -6.59153288e-02,
        8.77070174e-01,
        3.97137859e-02,
        -1.28451052e-02,
        -7.58944119e-02,
        8.71062859e-01,
        -1.11174889e-01,
        -3.56131041e-02,
        4.60743931e-01,
        9.78584139e-02,
        -3.80231272e-04,
        9.39964910e-01,
        4.25812316e-02,
        3.40729894e-02,
        2.84851912e-01,
        8.92114997e-02,
        7.28455406e-02,
        1.38152312e-01,
        3.21242706e-01,
        -7.31556453e-03,
        -3.04078678e-01,
        -7.86447530e-01,
        -1.14269838e-02,
        -2.64925617e-01,
        -3.39808606e-01,
        6.55948819e-02,
        3.18615043e-02,
        5.60014918e-01,
        -3.18976754e-01,
        -9.03763572e-02,
        7.11933630e-01,
        -2.53578190e-01,
        3.22989047e-02,
        1.43941356e-01,
        7.10271801e-01,
        6.43476101e-02,
        4.29799612e-02,
        -6.29269202e-01,
        -2.67925391e-01,
        4.64504760e-03,
        3.82176052e-02,
        2.63782679e-01,
        -4.34579985e-01,
        -1.89343423e-02,
        -1.02592731e-01,
        8.53927980e-01,
    ];
    const BRIXNER_EXCITON_DIPOLE_MOMENTS: [f32; 21] = [
        // 7x3 matrix unraveled, one dipole moment per row
        -3.97660199e-01,
        7.27544933e-01,
        -2.38755973e-01,
        -4.66431013e-01,
        6.11441002e-01,
        1.23191081e-01,
        -3.66529533e-01,
        -3.49789113e-01,
        3.43151541e-01,
        1.09327940e+00,
        7.53429504e-01,
        3.81304753e-01,
        -9.83093328e-01,
        -2.51699821e-01,
        -3.74933231e-01,
        -3.04810553e-01,
        1.33095014e+00,
        -8.63407929e-02,
        5.44412889e-02,
        -2.14975460e-01,
        -5.24009187e-01,
    ];
    const BRIXNER_STICK_ABS: [f32; 7] = [
        7.44459679e-01,
        6.06594031e-01,
        3.74449303e-01,
        1.90830917e+00,
        1.17040022e+00,
        1.87179248e+00,
        3.23763930e-01,
    ];
    const BRIXNER_STICK_CD: [f32; 7] = [
        -3.68370364e-03,
        -2.10216543e-03,
        3.52796169e-04,
        6.69710466e-03,
        -2.78332787e-03,
        6.38344482e-03,
        -4.84810079e-03,
    ];

    macro_rules! brixner_ham {
        () => {
            Array2::from_shape_vec((7, 7), Vec::from(BRIXNER_HAM)).unwrap()
        };
    }
    macro_rules! brixner_e_vals {
        () => {
            Array1::from_vec(Vec::from(BRIXNER_E_VALS))
        };
    }
    macro_rules! brixner_e_vecs {
        () => {
            Array2::from_shape_vec((7, 7), Vec::from(BRIXNER_E_VECS)).unwrap()
        };
    }
    macro_rules! brixner_dipole_moments {
        () => {
            Array2::from_shape_vec((7, 3), Vec::from(BRIXNER_DIPOLE_MOMENTS)).unwrap()
        };
    }
    macro_rules! brixner_pig_pos {
        () => {
            Array2::from_shape_vec((7, 3), Vec::from(BRIXNER_PIG_POS)).unwrap()
        };
    }
    macro_rules! brixner_exciton_dipole_moments {
        () => {
            Array2::from_shape_vec((7, 3), Vec::from(BRIXNER_EXCITON_DIPOLE_MOMENTS)).unwrap()
        };
    }
    macro_rules! brixner_stick_abs {
        () => {
            Array1::from_vec(Vec::from(BRIXNER_STICK_ABS))
        };
    }
    macro_rules! brixner_stick_cd {
        () => {
            Array1::from_vec(Vec::from(BRIXNER_STICK_CD))
        };
    }

    #[test]
    fn correctly_loads_hamiltonian() {
        let ham = brixner_ham!();
        assert_relative_eq!(ham[[1, 2]], 28.0);
    }

    #[test]
    fn diagonalizes_brixner_hamiltonian() {
        let ham = brixner_ham!();
        let good_e_vals = brixner_e_vals!();
        // The 6th eigenvector has its sign flipped for some reason
        let mut good_e_vecs = brixner_e_vecs!();
        let inverse = -1.0 * &good_e_vecs.column(5);
        good_e_vecs.column_mut(5).assign(&inverse);
        let (test_e_vals, test_e_vecs) = diagonalize(&ham);
        assert_abs_diff_eq!(test_e_vals, good_e_vals, epsilon = 1.0);
        assert_relative_eq!(test_e_vecs, good_e_vecs, epsilon = 1e-4);
    }

    #[test]
    fn computes_brixner_exciton_dipole_moments() {
        let e_vecs = brixner_e_vecs!();
        let dipole_moments = brixner_dipole_moments!();
        let test_exc_dipole_moments = exciton_dipole_moments(&e_vecs, &dipole_moments);
        let good_exc_dipole_moments = brixner_exciton_dipole_moments!();
        assert_abs_diff_eq!(
            test_exc_dipole_moments,
            good_exc_dipole_moments,
            epsilon = 1e-4
        );
    }

    #[test]
    fn computes_brixner_stick_abs() {
        let exciton_dpm = brixner_exciton_dipole_moments!();
        let test_stick_abs = stick_abs_single(&exciton_dpm);
        let good_stick_abs = brixner_stick_abs!();
        assert_abs_diff_eq!(test_stick_abs, good_stick_abs, epsilon = 1e-4);
    }

    #[test]
    fn computes_brixner_stick_cd() {
        let dipole_moments = brixner_dipole_moments!();
        let e_vecs = brixner_e_vecs!();
        let e_vals = brixner_e_vals!();
        let pig_pos = brixner_pig_pos!();
        let good_stick_cd = brixner_stick_cd!();
        let test_stick_cd = stick_cd_single(
            e_vecs.view(),
            dipole_moments.view(),
            pig_pos.view(),
            e_vals.view(),
        );
        assert_abs_diff_eq!(test_stick_cd, good_stick_cd, epsilon = 1e-4);
    }
}
