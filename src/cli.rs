use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use rsomics_common::{CommonFlags, Result, RsomicsError, Tool, ToolMeta};
use rsomics_help::{Example, FlagSpec, HelpSpec, Origin, Section};

use rsomics_sc_pca::{Input, run};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Parser, Debug)]
#[command(name = "rsomics-sc-pca", version, about, long_about = None, disable_help_flag = true)]
pub struct Cli {
    /// Cell × feature matrix; reads stdin when "-" or omitted.
    #[arg(default_value = "-")]
    input: PathBuf,

    /// Input format. `mtx` = MatrixMarket coordinate (cells × features),
    /// otherwise dense tab- or comma-separated.
    #[arg(long, value_name = "tsv|csv|mtx", default_value = "tsv")]
    format: String,

    /// Number of principal components.
    #[arg(long, value_name = "N", default_value_t = 50)]
    n_comps: usize,

    /// Skip zero-centering (TruncatedSVD without mean subtraction).
    #[arg(long, default_value_t = false)]
    no_zero_center: bool,

    /// X_pca scores output path ("-" for stdout).
    #[arg(long, default_value = "-")]
    out_scores: String,

    /// Per-PC variance + variance_ratio output path.
    #[arg(long)]
    out_variance: Option<String>,

    /// PCs loadings output path (omit to skip writing loadings).
    #[arg(long)]
    out_loadings: Option<String>,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Tool for Cli {
    fn meta() -> ToolMeta {
        META
    }
    fn common(&self) -> &CommonFlags {
        &self.common
    }

    fn execute(self) -> Result<()> {
        self.common.install_rayon_pool()?;

        let input = match self.format.as_str() {
            "tsv" => Input::Tsv,
            "csv" => Input::Csv,
            "mtx" => Input::Mtx,
            other => {
                return Err(RsomicsError::InvalidInput(format!(
                    "unknown --format '{other}' (expected tsv, csv, or mtx)"
                )));
            }
        };

        let reader: Box<dyn std::io::BufRead> = if self.input.as_os_str() == "-" {
            Box::new(BufReader::new(std::io::stdin().lock()))
        } else {
            Box::new(BufReader::new(File::open(&self.input).map_err(|e| {
                RsomicsError::InvalidInput(format!("{}: {e}", self.input.display()))
            })?))
        };

        let scores_out = open_sink(&self.out_scores)?;
        let variance_out = match self.out_variance.as_deref() {
            Some(path) => open_sink(path)?,
            None => open_sink("-")?,
        };
        let loadings_out = match self.out_loadings.as_deref() {
            Some(path) => Some(open_sink(path)?),
            None => None,
        };

        run(
            reader,
            input,
            self.n_comps,
            !self.no_zero_center,
            scores_out,
            variance_out,
            loadings_out,
        )
    }
}

fn open_sink(path: &str) -> Result<Box<dyn Write>> {
    if path == "-" {
        Ok(Box::new(BufWriter::new(std::io::stdout().lock())))
    } else {
        Ok(Box::new(BufWriter::new(
            File::create(path).map_err(RsomicsError::Io)?,
        )))
    }
}

pub static HELP: HelpSpec = HelpSpec {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
    tagline: "Zero-centered truncated PCA of a single-cell matrix (scanpy pp.pca arpack).",
    origin: Some(Origin {
        upstream: "scanpy sc.pp.pca (sklearn PCA arpack)",
        upstream_license: "BSD-3-Clause",
        our_license: "MIT OR Apache-2.0",
        paper_doi: Some("10.1186/s13059-017-1382-0"),
    }),
    usage_lines: &[
        "[matrix] [--format tsv|csv|mtx] [--n-comps N] [--out-scores X_pca.tsv] [--out-variance var.tsv] [--out-loadings PCs.tsv]",
    ],
    sections: &[Section {
        title: "OPTIONS",
        flags: &[
            FlagSpec {
                short: None,
                long: "format",
                aliases: &[],
                value: Some("<tsv|csv|mtx>"),
                type_hint: None,
                required: false,
                default: Some("tsv"),
                description: "Input format: dense tsv/csv or MatrixMarket mtx (cells × features).",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "n-comps",
                aliases: &[],
                value: Some("<N>"),
                type_hint: Some("usize"),
                required: false,
                default: Some("50"),
                description: "Number of principal components.",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "no-zero-center",
                aliases: &[],
                value: None,
                type_hint: None,
                required: false,
                default: Some("false"),
                description: "Skip mean subtraction (TruncatedSVD).",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "out-scores",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: Some("-"),
                description: "X_pca scores output (- for stdout).",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "out-variance",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: None,
                description: "Per-PC variance and variance_ratio output.",
                why_default: None,
            },
            FlagSpec {
                short: None,
                long: "out-loadings",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: None,
                description: "PCs loadings output (features × PCs).",
                why_default: None,
            },
        ],
    }],
    examples: &[
        Example {
            description: "50-component PCA of a 10x matrix",
            command: "rsomics-sc-pca matrix.mtx --format mtx --out-scores X_pca.tsv --out-variance var.tsv",
        },
        Example {
            description: "Top 2 PCs of a dense TSV to stdout",
            command: "rsomics-sc-pca counts.tsv --n-comps 2",
        },
    ],
    json_result_schema_doc: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        Cli::command().debug_assert();
    }
}
