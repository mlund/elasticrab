//! The `elasticrab` command-line tool: normal-mode analysis of a protein,
//! animating modes into PDB or XTC trajectories. Built only with the `cli`
//! feature; the library crate carries the analysis.

// Mirror the library crate's deliberate clippy::nursery exceptions: readable sums
// over fused multiply-add, and a binary `Some`/`None` match over `map_or_else`.
#![allow(clippy::suboptimal_flops, clippy::option_if_let_else)]

mod cli;

fn main() -> std::process::ExitCode {
    cli::run()
}
