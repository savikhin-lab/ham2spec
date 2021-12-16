extern crate lapack_src;
use approx::assert_relative_eq;
use lapack::sgeev;
use numpy::ndarray::{
    arr1, arr2, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, Zip,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct StickSpectrum {
    pub e_vecs: Array2<f32>,
    pub e_vals: Array1<f32>,
    pub mus: Array2<f32>,
    pub stick_abs: Array1<f32>,
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
fn delete_pigment_single(mut ham: ArrayViewMut2<f32>, mut mus: ArrayViewMut2<f32>, del: usize) {
    ham.row_mut(del).fill(0f32);
    ham.column_mut(del).fill(0f32);
    mus.row_mut(del).fill(0f32);
}

/// Compute the absorbance stick spectrum
///
/// The eigenvectors must be arranged into columns, and the pigment dipole moments
/// must be arranged into rows.
fn stick_abs_single(mus: &Array2<f32>) -> Array1<f32> {
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
/// must be arranged into rows.
fn stick_cd_single(
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
    for i in 0..n_pigs {
        for j in 0..n_pigs {
            for k in j..n_pigs {
                let r = &pos.row(j) - &pos.row(k);
                let mu_cross = cross(mus.row(j), mus.row(k));
                let r_mu_dot = dot(r.view(), mu_cross.view());
                cd[i] += 2f32 * e_vecs[[j, i]] * e_vecs[[k, i]] * r_mu_dot;
            }
        }
        cd[i] *= coeffs[i];
    }
    cd
}

/// Computes the transition dipole moments for each exciton
fn exciton_dipole_moments(e_vecs: &Array2<f32>, p_mus: &Array2<f32>) -> Array2<f32> {
    let n_pigs = e_vecs.ncols();
    let mut e_mus = Array2::zeros(p_mus.raw_dim());
    let mut weights = Array2::zeros(p_mus.raw_dim());
    for i in 0..n_pigs {
        weights.column_mut(0).assign(&e_vecs.column(i));
        weights.column_mut(1).assign(&e_vecs.column(i));
        weights.column_mut(2).assign(&e_vecs.column(i));
        let weighted_mu = (&weights * p_mus).sum_axis(Axis(0));
        e_mus.row_mut(i).assign(&weighted_mu);
        weights.fill(0.0);
    }
    e_mus
}

/// Diagonalize a Hamiltonian
fn diagonalize(ham: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let mut ham_fortran_order = ham.clone();
    ham_fortran_order.t().as_standard_layout().reversed_axes();
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
            b'N',                                      // Don't calculate left eigenvectors
            b'V',                                      // Do calculate the right eigenvectors
            ham_size,                                  // The dimensions of the Hamiltonian
            ham_fortran_order.as_slice_mut().unwrap(), // The underlying data in the Hamiltonian array
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
        .t()
        .as_standard_layout()
        .reversed_axes()
        .to_owned();
    (e_vals, e_vecs)
}

/// Compute the stick spectra of a Hamiltonian
fn compute_stick_spectrum(
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
    #[pyfn(m)]
    #[pyo3(name = "stick_abs_single")]
    fn stick_abs_single_py<'py>(py: Python<'py>, mus: PyReadonlyArray2<f32>) -> &'py PyArray1<f32> {
        stick_abs_single(&mus.as_array().to_owned()).into_pyarray(py)
    }

    /// Compute the CD stick spectrum
    #[pyfn(m)]
    #[pyo3(name = "stick_cd_single")]
    fn stick_cd_single_py<'py>(
        py: Python<'py>,
        e_vecs: PyReadonlyArray2<f32>,
        pig_mus: PyReadonlyArray2<f32>,
        pig_pos: PyReadonlyArray2<f32>,
        energies: PyReadonlyArray1<f32>,
    ) -> &'py PyArray1<f32> {
        stick_cd_single(
            e_vecs.as_array(),
            pig_mus.as_array(),
            pig_pos.as_array(),
            energies.as_array(),
        )
        .into_pyarray(py)
    }

    /// Compute the absorbance and CD stick spectrum of a single Hamiltonian
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

    #[test]
    fn diagonalize_identity_mat() {
        let ident_mat: Array2<f32> = Array2::eye(8);
        let (e_vals, e_vecs) = diagonalize(&ident_mat);
        for i in 0..8 {
            assert!((1.0 - e_vals[i]).abs() < 1e-4);
            assert!((1.0 - e_vecs[[i, i]]).abs() < 1e-4);
        }
    }

    #[test]
    fn diagonalizes_brixner_hamiltonian() {
        // This is a known-good result we're testing against
        let ham = arr2(&[
            [
                1.242000000000000000e+04,
                -1.060000000000000000e+02,
                8.000000000000000000e+00,
                -5.000000000000000000e+00,
                6.000000000000000000e+00,
                -8.000000000000000000e+00,
                -4.000000000000000000e+00,
            ],
            [
                -1.060000000000000000e+02,
                1.256000000000000000e+04,
                2.800000000000000000e+01,
                6.000000000000000000e+00,
                2.000000000000000000e+00,
                1.300000000000000000e+01,
                1.000000000000000000e+00,
            ],
            [
                8.000000000000000000e+00,
                2.800000000000000000e+01,
                1.214000000000000000e+04,
                -6.200000000000000000e+01,
                -1.000000000000000000e+00,
                -9.000000000000000000e+00,
                1.700000000000000000e+01,
            ],
            [
                -5.000000000000000000e+00,
                6.000000000000000000e+00,
                -6.200000000000000000e+01,
                1.231500000000000000e+04,
                -7.000000000000000000e+01,
                -1.900000000000000000e+01,
                -5.700000000000000000e+01,
            ],
            [
                6.000000000000000000e+00,
                2.000000000000000000e+00,
                -1.000000000000000000e+00,
                -7.000000000000000000e+01,
                1.246000000000000000e+04,
                4.000000000000000000e+01,
                -2.000000000000000000e+00,
            ],
            [
                -8.000000000000000000e+00,
                1.300000000000000000e+01,
                -9.000000000000000000e+00,
                -1.900000000000000000e+01,
                4.000000000000000000e+01,
                1.250000000000000000e+04,
                3.200000000000000000e+01,
            ],
            [
                -4.000000000000000000e+00,
                1.000000000000000000e+00,
                1.700000000000000000e+01,
                -5.700000000000000000e+01,
                -2.000000000000000000e+00,
                3.200000000000000000e+01,
                1.240000000000000000e+04,
            ],
        ]);
        let mus = arr2(&[
            [
                -7.409999999999999920e-01,
                -5.605999999999999872e-01,
                -3.695999999999999841e-01,
            ],
            [
                -8.570999999999999730e-01,
                5.038000000000000256e-01,
                -1.073000000000000065e-01,
            ],
            [
                -1.970999999999999974e-01,
                9.574000000000000288e-01,
                -2.109999999999999931e-01,
            ],
            [
                -7.992000000000000215e-01,
                -5.335999999999999632e-01,
                -2.766000000000000125e-01,
            ],
            [
                -7.368999999999999995e-01,
                6.558000000000000496e-01,
                1.640999999999999959e-01,
            ],
            [
                -1.350000000000000089e-01,
                -8.791999999999999815e-01,
                4.568999999999999728e-01,
            ],
            [
                -4.950999999999999845e-01,
                -7.083000000000000407e-01,
                -5.030999999999999917e-01,
            ],
        ]);
        let pos = arr2(&[
            [
                2.625349998474121094e+01,
                2.782999992370605469e+00,
                -1.110700035095214844e+01,
            ],
            [
                1.554899978637695312e+01,
                -1.585500001907348633e+00,
                -1.687949943542480469e+01,
            ],
            [
                3.403500080108642578e+00,
                -1.370650005340576172e+01,
                -1.425450038909912109e+01,
            ],
            [
                6.876500129699707031e+00,
                -2.091850090026855469e+01,
                -6.376999855041503906e+00,
            ],
            [
                1.937750053405761719e+01,
                -1.849099922180175781e+01,
                -1.491500020027160645e+00,
            ],
            [
                2.174550056457519531e+01,
                -7.323999881744384766e+00,
                3.115000128746032715e-01,
            ],
            [
                1.058049964904785156e+01,
                -8.205499649047851562e+00,
                -5.826499938964843750e+00,
            ],
        ]);
        let good_e_vals = arr1(&[
            1.211586132812500000e+04,
            1.262016503906250000e+04,
            1.254819335937500000e+04,
            1.227897460937500000e+04,
            1.236427050781250000e+04,
            1.245148828125000000e+04,
            1.241604589843750000e+04,
        ]);
        let good_e_vecs = arr2(&[
            [
                -4.627165570855140686e-02,
                -7.590184360742568970e-02,
                9.399629235267639160e-01,
                3.212469518184661865e-01,
                6.559748947620391846e-02,
                3.229876607656478882e-02,
                4.646750167012214661e-03,
            ],
            [
                -4.649672806262969971e-01,
                8.710590600967407227e-01,
                4.258818551898002625e-02,
                -7.314038462936878204e-03,
                3.185947611927986145e-02,
                1.439403444528579712e-01,
                3.821665793657302856e-02,
            ],
            [
                7.956331968307495117e-02,
                -1.111798435449600220e-01,
                3.407490253448486328e-02,
                -3.040867745876312256e-01,
                5.599942803382873535e-01,
                7.102859020233154297e-01,
                2.637786269187927246e-01,
            ],
            [
                -6.591442972421646118e-02,
                -3.561352565884590149e-02,
                2.848581373691558838e-01,
                -7.864409685134887695e-01,
                -3.189901113510131836e-01,
                6.433983892202377319e-02,
                -4.345791339874267578e-01,
            ],
            [
                8.770710229873657227e-01,
                4.607451856136322021e-01,
                8.920583873987197876e-02,
                -1.141071598976850510e-02,
                -9.036991000175476074e-02,
                4.298457875847816467e-02,
                -1.892157457768917084e-02,
            ],
            [
                -3.971533477306365967e-02,
                -9.785797446966171265e-02,
                -7.284726947546005249e-02,
                2.649361789226531982e-01,
                -7.119409441947937012e-01,
                6.292559504508972168e-01,
                1.025944426655769348e-01,
            ],
            [
                -1.284499187022447586e-02,
                -3.807079338002949953e-04,
                1.381517946720123291e-01,
                -3.398047685623168945e-01,
                -2.535800337791442871e-01,
                -2.679233551025390625e-01,
                8.539296984672546387e-01,
            ],
        ]);
        let (test_e_vals, test_e_vecs) = diagonalize(&ham);
        assert_relative_eq!(test_e_vals, good_e_vals, epsilon = 0.1);
        assert_relative_eq!(test_e_vecs, good_e_vecs, epsilon = 1e-4);
    }
}

/// Creates a cache of (r_i - r_j) * (mu_i x mu_j).
fn populate_r_mu_cross_cache(mus: &ArrayView2<f32>, pos: &ArrayView2<f32>) -> Array2<f32> {
    let n = mus.nrows();
    let mut cache = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let r_i = pos.row(i);
            let r_j = pos.row(j);
            let mut r = Array1::zeros(3);
            Zip::from(&mut r)
                .and(&r_i)
                .and(&r_j)
                .for_each(|r_out, &r_ii, &r_jj| *r_out = r_ii - r_jj);
            let mu_i = mus.row(i);
            let mu_j = mus.row(j);
            let mu_cross = cross(mu_i.view(), mu_j.view());
            cache[[i, j]] = dot(r.view(), mu_cross.view());
        }
    }
    cache
}
