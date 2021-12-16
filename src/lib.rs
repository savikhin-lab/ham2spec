extern crate lapack_src;
use lapack::sgeev;
use numpy::ndarray::{arr1, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, Zip};
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
    let e_vecs =
        Array2::from_shape_vec((ham_size as usize, ham_size as usize), e_vecs_right).unwrap();
    e_vecs.t().as_standard_layout().reversed_axes();
    (e_vals, e_vecs)
}

/// Compute the stick spectra of a Hamiltonian
fn compute_stick_spectrum(
    ham: ArrayView2<f32>,
    mus: ArrayView2<f32>,
    pos: ArrayView2<f32>,
) -> StickSpectrum {
    let (e_vals, e_vecs) = diagonalize(&ham.to_owned());
    // let exc_mus = exciton_mus(&e_vecs, &mus.to_owned());
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
}
