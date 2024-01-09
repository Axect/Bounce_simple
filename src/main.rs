use peroxide::fuga::*;
use std::f64::consts::PI;

const N: usize = 10000;

fn main() {
    let x = linspace(0, 1, 16);
    let u_omega = Uniform(0f64, PI / 3f64);
    let u_a = Uniform(0f64, 1f64);

    let omega_vec = u_omega.sample(N);
    let a_vec = u_a.sample(N);
    let xp = 1f64;

    let mut y_vec = vec![vec![0f64; x.len()]; a_vec.len()];
    let mut dy_vec = vec![vec![0f64; x.len()]; a_vec.len()];
    let mut s_vec = vec![0f64; a_vec.len()];

    for (i, (omega, a)) in omega_vec
        .iter()
        .zip(a_vec.iter())
        .enumerate()
    {
        let y = sho_potential(*omega, *a, &x);
        y_vec[i] = y;
        let dy = sho_derivative(*omega, *a, &x);
        dy_vec[i] = dy;
        let s = -0.5 * (xp + a).powi(2) * omega * omega.tan()
            + 0.5 * omega * a.powi(2) * (omega.tan() - omega);
        s_vec[i] = s;
    }

    let mut y_mat = py_matrix(y_vec);
    match y_mat.shape {
        Row => y_mat.change_shape_mut(),
        Col => (),
    };
    
    let mut dy_mat = py_matrix(dy_vec);
    match dy_mat.shape {
        Row => dy_mat.change_shape_mut(),
        Col => (),
    };

    let mut df = DataFrame::new(vec![]);
    df.push("omega", Series::new(omega_vec));
    df.push("a", Series::new(a_vec));
    df.push("s", Series::new(s_vec));
    for i in 0..x.len() {
        df.push(&format!("v{}", i), Series::new(y_mat.col(i)));
        df.push(&format!("dv{}", i), Series::new(dy_mat.col(i)));
    }

    df.write_parquet("sho.parquet", CompressionOptions::Uncompressed)
        .expect("failed to write parquet file");
}

fn sho_potential(omega: f64, a: f64, x: &Vec<f64>) -> Vec<f64> {
    x.fmap(|t| 0.5 * omega.powi(2) * (t - a).powi(2))
}

fn sho_derivative(omega: f64, a: f64, x: &Vec<f64>) -> Vec<f64> {
    x.fmap(|t| omega * (t - a))
}
