use std::env;
use std::time::Instant;
fn help() {
println!("Usage: <# iterations> <grid size>");
}
fn main()
{
println!("Parallel Research Kernels");
println!("Rust stencil execution on 2D grid");
let args : Vec<String> = env::args().collect();
let iterations : usize;
let n : usize;
if args.len() == 3 {
iterations = match args[1].parse() {
Ok(n) => { n },
Err(_) => { help(); return; },
};
n = match args[2].parse() {
Ok(n) => { n },
Err(_) => { help(); return; },
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
let r : usize =
if cfg!(radius = "1") { 1 } else
if cfg!(radius = "2") { 2 } else
if cfg!(radius = "3") { 3 } else
if cfg!(radius = "4") { 4 } else
if cfg!(radius = "5") { 5 } else
if cfg!(radius = "6") { 6 } else
{ 2 };
let grid : bool = if cfg!(grid) { true } else { false };
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
let nelems : usize = n*n;
let mut a : Vec<f64> = vec![0.0; nelems];
let mut b : Vec<f64> = vec![0.0; nelems];
let wdim : usize = 2 * r + 1;
let welems : usize = wdim*wdim;
let mut w : Vec<f64> = vec![0.0; welems];
for jj in 0..wdim {
for ii in 0..wdim {
let offset : usize = ii * wdim + jj;
w[offset] = 0.0;
}
}
let stencil_size : usize;
if grid {
stencil_size = (2*r+1)*(2*r+1);
for jj in 1..r+1 {
for ii in 1-jj..jj {
let denom : f64 = (4*jj*(2*jj-1)*r) as f64;
let offset : usize = ((r+ii) * wdim) + (r+jj);
w[offset] =  1./denom;
let offset : usize = ((r+ii) * wdim) + (r-jj);
w[offset] = -1./denom;
let offset : usize = ((r+jj) * wdim) + (r+ii);
w[offset] =  1./denom;
let offset : usize = ((r-jj) * wdim) + (r+ii);
w[offset] = -1./denom;
}
let denom : f64 = (4*jj*r) as f64;
let offset : usize = (r+jj) * wdim + (r+jj);
w[offset] = -1./denom;
let offset : usize = (r-jj) * wdim + (r-jj);
w[offset] = -1./denom;
}
}  else  {
stencil_size = 4*r+1;
for ii in 1..r+1 {
let denom : f64 = (2 * ii * r) as f64;
let offset : usize = ((r) * wdim) + (r+ii);
w[offset] =  1./denom;
let offset : usize = ((r) * wdim) + (r-ii);
w[offset] = -1./denom;
let offset : usize = ((r+ii) * wdim) + (r+ii);
w[offset] =  1./denom;
let offset : usize = ((r-ii) * wdim) + (r+ii);
w[offset] = -1./denom;
}
}
let active_points : usize = (n-2*r)*(n-2*r);
for j in 0..n {
for i in 0..n {
a[i*n+j] = (i+j) as f64;
b[i*n+j] = 0.0;
}
}
let mut t0 = Instant::now();
for k in 0..iterations+1 {
if k == 1 { t0 = Instant::now(); }
for i in r..n-r {
for j in r..n-r {
if grid {
for ii in 0-r..r+1 {
for jj in 0-r..r+1 {
let offset : usize = ((r+ii) * wdim) + (r+jj);
b[i*n+j] += w[offset]*a[(i+ii)*n+j+jj];
}
}
} else {
let offset : usize = ((r) * wdim) + (r);
b[i*n+j] += w[offset]*a[i*n+j];
for jj in r..0 {
let offset : usize = ((r) * wdim) + (r-jj);
b[i*n+j] += w[offset]*a[i*n+j-jj];
}
for jj in 1..r+1 {
let offset : usize = ((r) * wdim) + (r+jj);
b[i*n+j] += w[offset]*a[i*n+j+jj];
}
for ii in r..0 {
let offset : usize = ((r-ii) * wdim) + (r);
b[i*n+j] += w[offset]*a[(i-ii)*n+j];
}
for ii in 1..r+1 {
let offset : usize = ((r+ii) * wdim) + (r);
b[i*n+j] += w[offset]*a[(i+ii)*n+j];
}
}
}
}
for j in 0..n {
for i in 0..n {
a[i*n+j] += 1.0;
}
}
}
let t1 = Instant::now();
let stencil_time = t1 - t0;
let epsilon : f64 = 0.000000001;
let mut norm : f64 = 0.0;
for i in r..n-r+1 {
for j in r..n-r+1 {
norm += (b[i*n+j]).abs();
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
let avgtime : f64 = (stencil_time.as_secs() as f64) / (iterations as f64);
println!("Rate (MFlops/s): {:10.3} Avg time (s): {:10.3}", (0.000001 as f64) * (flops as f64) / avgtime, avgtime);
}
}
