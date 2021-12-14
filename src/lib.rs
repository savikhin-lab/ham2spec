use pyo3::prelude::*;
use numpy::ndarray::{
    ArrayView1,
    ArrayView2, 
    ArrayViewMut2,
    Array1,
    Array2,
    Axis,
    arr1,
    s};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};

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

/// Computes the stick spectrum of the absorbance
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

    /// Compute the stick spectrum of the absorbance.
    #[pyfn(m)]
    #[pyo3(name = "stick_abs")]
    fn stick_abs_py<'py>(
        py: Python<'py>,
        e_vecs: PyReadonlyArray2<f32>,
        pig_mus: PyReadonlyArray2<f32>
    ) -> &'py PyArray1<f32> {
        stick_abs_single(e_vecs.as_array(), pig_mus.as_array()).into_pyarray(py)
    }

    Ok(())
}
