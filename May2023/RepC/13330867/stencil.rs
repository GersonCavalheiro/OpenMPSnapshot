use std::env;
use std::time::{Instant,Duration};
fn help() {
println!("Usage: <# iterations> <grid dimension> <radius>");
}
fn main()
{
println!("Parallel Research Kernels");
println!("Rust stencil execution on 2D grid");
let args : Vec<String> = env::args().collect();
let iterations : usize;
let n : usize;
let r : usize;
let grid : bool = if cfg!(grid) { true } else { false };
if args.len() == 4 {
iterations = match args[1].parse() {
Ok(n) => { n },
Err(_) => { help(); return; },
};
n = match args[2].parse() {
Ok(n) => { n },
Err(_) => { help(); return; },
};
r = match args[3].parse() {
Ok(n) => { n },
Err(_) => { 2 },
};
} else {
help();
return;
}
if iterations < 1 {
println!("ERROR: iterations must be >= 1");
}
if n < 1 {
println!("ERROR: grid dimension must be positive: {}", n);
}
if r < 1 {
println!("ERROR: Stencil radius {} should be positive ", r);
return;
} else if (2 * r + 1) > n {
println!("ERROR: Stencil radius {} exceeds grid size {}", r, n);
return;
}
println!("Grid size            = {}", n);
println!("Radius of stencil    = {}", r);
if grid {
println!("Type of stencil      = grid");
} else {
println!("Type of stencil      = star");
}
println!("Data type            = double precision");
println!("Compact representation of stencil loop body");
println!("Number of iterations = {}",iterations);
let mut a : Vec<Vec<f64>> = vec![vec![0.0; n]; n];
let mut b : Vec<Vec<f64>> = vec![vec![0.0; n]; n];
let wdim : usize = 2 * r + 1;
let mut w : Vec<Vec<f64>> = vec![vec![0.0; wdim]; wdim];
let stencil_size : usize;
if grid {
stencil_size = (2*r+1)*(2*r+1);
for j in 1..r+1 {
for i in 1-j..j {
let denom : f64 = (4*j*(2*j-1)*r) as f64;
w[r+i][r+j] =  1./denom;
w[r+i][r-j] = -1./denom;
w[r+j][r+i] =  1./denom;
w[r-j][r+i] = -1./denom;
}
let denom : f64 = (4*j*r) as f64;
w[r+j][r+j]   =  1./denom;
w[r-j][r-j]   = -1./denom;
}
}  else  {
stencil_size = 4*r+1;
for i in 1..r+1 {
let denom : f64 = (2 * i * r) as f64;
w[r][r+i] =  1./denom;
w[r][r-i] = -1./denom;
w[r+i][r] =  1./denom;
w[r-i][r] = -1./denom;
}
}
let active_points : usize = (n-2*r)*(n-2*r);
for j in 0..n {
for i in 0..n {
a[i][j] = (i+j) as f64;
b[i][j] = 0.0;
}
}
let timer = Instant::now();
let mut t0 : Duration = timer.elapsed();
for k in 0..iterations+1 {
if k == 1 { t0 = timer.elapsed(); }
for i in r..n-r {
for j in r..n-r {
if grid {
for ii in 0-r..r+1 {
for jj in 0-r..r+1 {
b[i][j] += w[r+ii][r+jj]*a[i+ii][j+jj];
}
}
} else {
b[i][j] += w[r][r]*a[i][j];
for jj in r..0 {
b[i][j] += w[r][r-jj]*a[i][j-jj];
}
for jj in 1..r+1 {
b[i][j] += w[r][r+jj]*a[i][j+jj];
}
for ii in r..0 {
b[i][j] += w[r-ii][r]*a[i-ii][j];
}
for ii in 1..r+1 {
b[i][j] += w[r+ii][r]*a[i+ii][j];
}
}
}
}
for j in 0..n {
for i in 0..n {
a[i][j] += 1.0;
}
}
}
let t1 = timer.elapsed();
let dt = (t1.checked_sub(t0)).unwrap();
let dtt : u64 = dt.as_secs() * 1_000_000_000 + dt.subsec_nanos() as u64;
let stencil_time : f64 = dtt as f64 / 1.0e9_f64 as f64;
let epsilon : f64 = 1.0e-8;
let mut norm : f64 = 0.0;
for i in r..n-r+1 {
for j in r..n-r+1 {
norm += (b[i][j]).abs();
}
}
norm /= active_points as f64;
let reference_norm : f64 = 2.*(iterations as f64 + 1.);
if (norm-reference_norm).abs() > epsilon {
println!("ERROR: L1 norm = {} Reference L1 norm = {}", norm, reference_norm);
return;
} else {
println!("Solution validates");
if cfg!(VERBOSE) {
println!("L1 norm = {} Reference L1 norm = {}", norm, reference_norm);
}
let flops : usize = (2*stencil_size+1) * active_points;
let avgtime : f64 = (stencil_time as f64) / (iterations as f64);
println!("Rate (MFlops/s): {:10.3} Avg time (s): {:10.3}", (1.0e-6_f64) * (flops as f64) / avgtime, avgtime);
}
}
