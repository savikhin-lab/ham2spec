use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::ndarray::{
    ArrayView1,
    ArrayView2, 
    ArrayViewMut2,
    Array1,
    Array2,
    Axis,
    arr1,
    s};
use numpy::{
    IntoPyArray,
    PyArray1,
    PyArray2, 
    PyReadonlyArray1,
    PyReadonlyArray2};
use peroxide::prelude::*;
use peroxide::numerical::eigen::{eigen, EigenMethod};

#[derive(Debug, Clone)]
pub struct StickSpectrum {
    pub e_vecs: Array2<f32>,
    pub e_vals: Array1<f32>,
    pub mus: Array2<f32>,
    pub stick_abs: Array1<f32>,
    pub stick_cd: Array1<f32>
}

/// Compute the dot product of 2 3-vectors
fn dot(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the cross-product of 2 3-vectors
fn cross(a: ArrayView1<f32>, b: ArrayView1<f32>) -> Array1<f32> {
    arr1(&[a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]])
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
fn stick_abs_single(e_vecs: ArrayView2<f32>, pig_mus: ArrayView2<f32>) -> Array1<f32> {
    let n_pigs = e_vecs.ncols();
    let mus = exciton_mus(e_vecs, pig_mus);
    let mut stick_abs = Array1::zeros(n_pigs);
    for i in 0_usize..n_pigs {
        let mu = mus.slice(s![i, ..]);
        // stick_abs[i] = mu.dot(&mu);
        stick_abs[i] = dot(mu, mu);
    }
    stick_abs
}

/// Compute the CD stick spectrum
/// 
/// The eigenvectors must be arranged into columns, and the pigment dipole moments
/// must be arranged into rows.
fn stick_cd_single(e_vecs: ArrayView2<f32>, mus: ArrayView2<f32>, pos: ArrayView2<f32>, energies: ArrayView1<f32>) -> Array1<f32> {
    let coeffs: Vec<f32> = energies.iter()
        .map(|e| {
        let wavelength = if e < &1.0 {
            1e8 / 100_000f32
        } else {
            1e8 / e
        };
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

/// Computes the transition dipole moments for each exciton.
fn exciton_mus(e_vecs: ArrayView2<f32>, pig_mus: ArrayView2<f32>) -> Array2<f32> {
    let n_pigs = e_vecs.ncols();
    // Each row of this array is a weighted sum of the dipole moments, where the
    // weights come from the elements of the corresponding eigenvector.
    let mut new_mus = Array2::zeros((n_pigs, 3));
    // Each row of this array holds a weighted dipole moment whose weight comes
    // from the corresponding element in the current eigenvector.
    let mut tmp_mus = Array2::zeros((n_pigs, 3));
    for i in 0_usize..n_pigs {
        tmp_mus.fill(0_f32); // clear out the previous iteration
        for j in 0_usize..n_pigs {
            let mut tmp_mu = tmp_mus.row_mut(j);
            let scaled_mu = pig_mus.row(j).mapv(|x| x * e_vecs[[j, i]]);
            tmp_mu.assign(&scaled_mu);
        }
        let mut new_mu = new_mus.row_mut(i);
        let weighted_mu_sum = tmp_mus.sum_axis(Axis(0));
        new_mu.assign(&weighted_mu_sum);
    }
    new_mus
}

/// Compute the stick spectra of a Hamiltonian
fn compute_stick_spectrum(ham: ArrayView2<f32>, mus: ArrayView2<f32>, pos: ArrayView2<f32>) -> StickSpectrum {
    let mat = ndarray_2d_to_matrix(ham);
    let eigs = eigen(&mat, EigenMethod::Jacobi);
    let e_vals = Array1::from_vec(eigs.eigenvalue.clone().iter().map(|x| *x as f32).collect());
    let e_vecs = matrix_to_2d_ndarray(mat);
    let exc_mus = exciton_mus(e_vecs.view(), mus);
    let abs_sticks = stick_abs_single(e_vecs.view(), mus);
    let cd_sticks = stick_cd_single(e_vecs.view(), mus, pos, e_vals.view());
    StickSpectrum {
        e_vecs: e_vecs,
        e_vals: e_vals,
        mus: exc_mus,
        stick_abs: abs_sticks,
        stick_cd: cd_sticks,
    }
}

/// Convert an ndarray into a peroxide Matrix
fn ndarray_2d_to_matrix(arr: ArrayView2<f32>) -> Matrix {
    let mut data = Vec::new();
    data.extend_from_slice(arr.to_slice().unwrap());
    let data = data.iter_mut().map(|x| *x as f64).collect();
    Matrix {
        data: data,
        row: arr.nrows(),
        col: arr.ncols(),
        shape: Row
    }
}

/// Convert a peroxide Matrix into a 2d ndarray
fn matrix_to_2d_ndarray(mat: Matrix) -> Array2<f32> {
    let data = mat.data.iter().map(|x| *x as f32).collect();
    Array2::from_shape_vec((mat.row, mat.col), data).unwrap()
}

/// Compute absorbance and CD spectra from first principles.
#[pymodule]
fn ham2spec(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Computes the transition dipole moments for each exciton.
    #[pyfn(m)]
    #[pyo3(name = "exciton_mus")]
    fn exciton_mus_py<'py>(
        py: Python<'py>,
        e_vecs: PyReadonlyArray2<f32>,
        pig_mus: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        exciton_mus(e_vecs.as_array(), pig_mus.as_array()).into_pyarray(py)
    }

    /// Compute the absorbance stick spectrum
    #[pyfn(m)]
    #[pyo3(name = "stick_abs_single")]
    fn stick_abs_single_py<'py>(
        py: Python<'py>,
        e_vecs: PyReadonlyArray2<f32>,
        pig_mus: PyReadonlyArray2<f32>
    ) -> &'py PyArray1<f32> {
        stick_abs_single(e_vecs.as_array(), pig_mus.as_array()).into_pyarray(py)
    }

    /// Compute the CD stick spectrum
    #[pyfn(m)]
    #[pyo3(name = "stick_cd_single")]
    fn stick_cd_single_py<'py>(
        py: Python<'py>,
        e_vecs: PyReadonlyArray2<f32>,
        pig_mus: PyReadonlyArray2<f32>,
        pig_pos: PyReadonlyArray2<f32>,
        energies: PyReadonlyArray1<f32>
    ) -> &'py PyArray1<f32> {
        stick_cd_single(
            e_vecs.as_array(),
            pig_mus.as_array(),
            pig_pos.as_array(),
            energies.as_array()
        ).into_pyarray(py)
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
        dict.set_item("e_vals", sticks.e_vals.into_pyarray(py)).unwrap();
        dict.set_item("e_vecs", sticks.e_vecs.into_pyarray(py)).unwrap();
        dict.set_item("exciton_mus", sticks.mus.into_pyarray(py)).unwrap();
        dict.set_item("stick_abs", sticks.stick_abs.into_pyarray(py)).unwrap();
        dict.set_item("stick_cd", sticks.stick_cd.into_pyarray(py)).unwrap();
        dict
    }

    Ok(())
}

#[cfg(test)]
mod test {
    todo!();
}