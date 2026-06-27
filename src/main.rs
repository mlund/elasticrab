//! The `elasticrab` command-line tool: normal-mode analysis of a protein,
//! animating modes into PDB or XTC trajectories. Built only with the `cli`
//! feature; the library crate carries the analysis.

// Readable accumulation sums are preferred over fused multiply-add, as in the
// library crate.
#![allow(clippy::suboptimal_flops)]

mod cli;

fn main() -> std::process::ExitCode {
    cli::run()
}
