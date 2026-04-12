//! elworthy CLI entry point.

use clap::{Parser, Subcommand};
use elworthy_expr::Expr;
use elworthy_rt::{euler_scalar_interp, euler_scalar_jit, euler_scalar_jit_delta_bel};

#[derive(Parser)]
#[command(name = "elworthy", about = "Bismut-Elworthy-Li JIT Monte Carlo")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scalar Euler-Maruyama check on geometric Brownian motion.
    Gbm {
        #[arg(long, default_value_t = 0.05)]
        r: f64,
        #[arg(long, default_value_t = 0.2)]
        sigma: f64,
        #[arg(long, default_value_t = 100.0)]
        x0: f64,
        #[arg(long, default_value_t = 1.0)]
        t: f64,
        #[arg(long, default_value_t = 256)]
        steps: usize,
        #[arg(long, default_value_t = 10_000)]
        paths: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Execution backend: "jit" or "interp".
        #[arg(long, default_value = "jit")]
        backend: String,
    },
    /// Price and Bismut-Elworthy-Li delta on geometric Brownian motion.
    GbmDelta {
        #[arg(long, default_value_t = 0.05)]
        r: f64,
        #[arg(long, default_value_t = 0.2)]
        sigma: f64,
        #[arg(long, default_value_t = 100.0)]
        x0: f64,
        #[arg(long, default_value_t = 1.0)]
        t: f64,
        #[arg(long, default_value_t = 256)]
        steps: usize,
        #[arg(long, default_value_t = 40_000)]
        paths: usize,
        #[arg(long, default_value_t = 1234)]
        seed: u64,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Cmd::Gbm {
            r,
            sigma,
            x0,
            t,
            steps,
            paths,
            seed,
            backend,
        } => {
            let mu = Expr::param(0) * Expr::state(0);
            let sig = Expr::param(1) * Expr::state(0);
            let payoff = Expr::state(0);
            let est = match backend.as_str() {
                "interp" => {
                    euler_scalar_interp(&mu, &sig, &payoff, &[r, sigma], x0, t, steps, paths, seed)
                }
                "jit" => {
                    euler_scalar_jit(&mu, &sig, &payoff, &[r, sigma], x0, t, steps, paths, seed)?
                }
                other => anyhow::bail!("unknown backend '{other}' (use 'jit' or 'interp')"),
            };
            println!(
                "[{backend}] E[X_T] ~ {:.4} (stderr {:.4}) | closed form {:.4}",
                est.mean,
                est.stderr,
                x0 * (r * t).exp(),
            );
        }
        Cmd::GbmDelta {
            r,
            sigma,
            x0,
            t,
            steps,
            paths,
            seed,
        } => {
            let mu = Expr::param(0) * Expr::state(0);
            let sig = Expr::param(1) * Expr::state(0);
            let payoff = Expr::state(0);
            // sigma(X_0) for GBM is sigma * x0.
            let sigma_at_x0 = sigma * x0;
            let out = euler_scalar_jit_delta_bel(
                &mu,
                &sig,
                &payoff,
                &[r, sigma],
                x0,
                t,
                sigma_at_x0,
                steps,
                paths,
                seed,
            )?;
            let analytic_price = x0 * (r * t).exp();
            let analytic_delta = (r * t).exp();
            println!(
                "price   ~ {:.4} (stderr {:.4}) | closed form {:.4}",
                out.price.mean, out.price.stderr, analytic_price
            );
            println!(
                "delta   ~ {:.4} (stderr {:.4}) | closed form {:.4}   [Bismut-Elworthy-Li]",
                out.delta.mean, out.delta.stderr, analytic_delta
            );
        }
    }
    Ok(())
}
