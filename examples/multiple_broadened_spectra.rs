// To enable examples, change the crate-type to "rlib", otherwise you'll get linker errors.
use ham2spec::{compute_broadened_spectrum_from_hams, BroadeningConfig};
use ndarray::{s, Array2, Array3};

// These are known-good results we'll test against
const BRIXNER_HAM: [f64; 49] = [
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
const BRIXNER_DIPOLE_MOMENTS: [f64; 21] = [
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
const BRIXNER_PIG_POS: [f64; 21] = [
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

macro_rules! ham {
    () => {
        Array2::from_shape_vec((7, 7), Vec::from(BRIXNER_HAM)).unwrap()
    };
}
macro_rules! dipole_moments {
    () => {
        Array2::from_shape_vec((7, 3), Vec::from(BRIXNER_DIPOLE_MOMENTS)).unwrap()
    };
}
macro_rules! positions {
    () => {
        Array2::from_shape_vec((7, 3), Vec::from(BRIXNER_PIG_POS)).unwrap()
    };
}

fn main() {
    let (hams, mus, rs) = assemble_data();
    let config = BroadeningConfig {
        x_from: 11790.0,
        x_to: 13300.0,
        x_step: 1.0,
        bw: 120.0,
        abs_bws: vec![120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
        cd_bws: vec![120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
        band_cutoff: 3.0,
    };
    let one_ham = ham!();
    let one_mus = dipole_moments!();
    let one_rs = positions!();
    let mut specs = Vec::new();
    for _ in 0..1_000 {
        let spec =
            compute_broadened_spectrum_from_hams(hams.view(), mus.view(), rs.view(), &config);
        // compute_broadened_spectrum_from_ham(one_ham.view(), one_mus.view(), one_rs.view(), &config);
        specs.push(spec);
    }
    // let spec =
    //     compute_broadened_spectrum_from_ham(one_ham.view(), one_mus.view(), one_rs.view(), &config);
    println!("{:?}", specs.len());
}

fn assemble_data() -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let single_ham = ham!();
    let single_mus = dipole_moments!();
    let single_rs = positions!();
    let mut hams = Array3::zeros((100, 7, 7));
    let mut mus = Array3::zeros((100, 7, 3));
    let mut rs = Array3::zeros((100, 7, 3));
    for i in 0..100 {
        hams.slice_mut(s![i, .., ..]).assign(&single_ham);
        mus.slice_mut(s![i, .., ..]).assign(&single_mus);
        rs.slice_mut(s![i, .., ..]).assign(&single_rs);
    }
    (hams, mus, rs)
}
