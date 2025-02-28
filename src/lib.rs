//! TODO: docs

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::inline_always)]
#![allow(clippy::doc_markdown)]
#![warn(missing_docs)]
#![warn(rust_2024_compatibility)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "thread_local")]
#[macro_use]
pub(crate) mod thread_local;

pub mod raw;
pub mod relax;

pub(crate) mod cfg;
pub(crate) mod fairness;
pub(crate) mod inner;
pub(crate) mod lock;
pub(crate) mod xoshiro;

#[cfg(test)]
pub(crate) mod test;

#[cfg(all(loom, test))]
#[cfg(not(tarpaulin))]
pub(crate) mod loom;
