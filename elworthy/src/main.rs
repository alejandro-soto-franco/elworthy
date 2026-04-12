//! elworthy CLI entry point.

use clap::{Parser, Subcommand};
use elworthy_expr::Expr;
use elworthy_rt::euler_scalar;

#[derive(Parser)]
#[command(name = "elworthy", about = "Bismut-Elworthy-Li JIT Monte Carlo")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a scalar Euler-Maruyama check on geometric Brownian motion.
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
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Cmd::Gbm { r, sigma, x0, t, steps, paths, seed } => {
            let mu = Expr::param(0) * Expr::state(0);
            let sig = Expr::param(1) * Expr::state(0);
            let payoff = Expr::state(0);
            let est = euler_scalar(
                &mu, &sig, &payoff,
                &[r, sigma], x0, t, steps, paths, seed,
            );
            println!(
                "E[X_T] ~ {:.4} (stderr {:.4}) | closed form {:.4}",
                est.mean,
                est.stderr,
                x0 * (r * t).exp(),
            );
        }
    }
    Ok(())
}
